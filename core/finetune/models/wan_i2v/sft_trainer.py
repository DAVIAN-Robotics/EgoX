from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanImageToVideoPipeline,
)
from core.finetune.models.wan_i2v.custom_transformer import WanTransformer3DModel_GGA as WanTransformer3DModel
from diffusers.utils import (
    replace_example_docstring,
    logging,
    is_torch_xla_available,
)
from diffusers.image_processor import PipelineImageInput
from diffusers.callbacks import PipelineCallback, MultiPipelineCallbacks
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
import PIL

# Constants and conditional imports
logger = logging.get_logger(__name__)

# XLA support (conditional)
try:
    XLA_AVAILABLE = is_torch_xla_available()
    if XLA_AVAILABLE:
        import torch_xla.core.xla_model as xm  # type: ignore
    else:
        xm = None  # type: ignore
except ImportError:
    XLA_AVAILABLE = False
    xm = None  # type: ignore

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import WanImageToVideoPipeline

        >>> pipe = WanImageToVideoPipeline.from_pretrained("path/to/model")
        >>> images = load_images("path/to/images/")  # Load all frames
        >>> video = pipe(images=images, prompt="A beautiful sunset").frames[0]
        ```
"""

from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
import numpy as np
from numpy import dtype
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel
from typing_extensions import override

from core.finetune.schemas import Wan_Components as Components
from core.finetune.trainer import Trainer
from core.finetune.utils import unwrap_model

from ..utils import register

from diffusers import (
    WanImageToVideoPipeline,
)
from diffusers.utils.torch_utils import randn_tensor

def generate_uniform_pointmap(height, width):
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xv, yv = np.meshgrid(x, y)
    
    # Create a spatially varying Z value: increases from bottom (row 0) to top (row height-1)
    zv = 1 - np.linspace(0, 1, height)[:, None]  # shape (H, 1)
    zv = np.repeat(zv, width, axis=1)        # shape (H, W)
    
    # Stack to get (H, W, 3) array
    pointmap = np.stack([xv, yv, zv], axis=-1)
    
    # Normalize XYZ to [0, 1] for image saving
    # X and Y are already in [-1, 1], so map to [0, 1]
    pointmap[..., 0] = (pointmap[..., 0] + 1) / 2
    pointmap[..., 1] = (pointmap[..., 1] + 1) / 2
    # Z is constant, but ensure it's in [0, 1]
    pointmap[..., 2] = (pointmap[..., 2] - np.min(pointmap[..., 2])) / (np.ptp(pointmap[..., 2]) + 1e-8)
    return pointmap

def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax": # 여기선 argmax로 돌아감
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

class WanHBHImageToVideoPipeline(WanImageToVideoPipeline):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        image_encoder: CLIPVisionModel,
        image_processor: CLIPImageProcessor,
        transformer: WanTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__(tokenizer, text_encoder, image_encoder, image_processor, transformer, vae, scheduler)
        self._shared_latent_exo_randn = None

    @override
    def prepare_latents(
        self,
        ego_prior_video,  # ego prior video for latent condition
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        exo_width: int = 784,
        ego_width: int = 448, 
        num_frames: int = 49,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        last_images: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Calculate latent dimensions for each view type
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        exo_latent_width = exo_width // self.vae_scale_factor_spatial  
        ego_latent_width = ego_width // self.vae_scale_factor_spatial
        total_latent_width = exo_latent_width + ego_latent_width

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, total_latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # Use shared IC-LoRA latent if available
        if self._shared_latent_exo_randn is not None:
            latents = self._shared_latent_exo_randn.to(device=device, dtype=dtype)
            print(f"Using shared IC-LoRA latent: latents_shape={latents.shape}")
        elif latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # Process ego prior video and replace ego region in shared latent
        # Add batch dimension if missing (video_processor outputs [C, F, H, W], VAE expects [B, C, F, H, W])
        if ego_prior_video.dim() == 4:
            ego_prior_video = ego_prior_video.unsqueeze(0)  # Add batch dimension
        
        ego_prior_latents = self.encode_video(ego_prior_video.to(device=device, dtype=self.vae.dtype))
        
        # Calculate ego region boundaries
        latent_total_width = self._shared_latent_exo_randn.shape[-1] 
        ego_latent_width = ego_prior_latents.shape[-1]
        exo_latent_width = latent_total_width - ego_latent_width
        
        # Create a copy to avoid modifying the shared variable
        video_condition = self._shared_latent_exo_randn.clone()
        
        # Replace ego region (right part) with ego_prior_latents
        video_condition[:, :, :, :, exo_latent_width:] = ego_prior_latents

        # Use video_condition (already encoded latents) directly as latent_condition
        latent_condition = video_condition

        latent_condition = latent_condition.to(dtype)
        # latent_condition = (latent_condition - latents_mean) * latents_std

        # Create mask for exo and ego views
        # Use num_frames (target frame count) like the parent class, not frame_num (input frame count)
        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, total_latent_width)
        
        # Set exo_view mask (left part) to 1.0
        mask_lat_size[:, :, :, :, :exo_latent_width] = 1.0
        # Set ego_view mask (right part) to 0.0  
        mask_lat_size[:, :, :, :, exo_latent_width:] = 0.0

        if last_images is None:
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
        else:
            mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal) # ch: 1 -> 4
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2) # ch: 48 -> 48 + 4
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, total_latent_width) # ch: 52 -> 13s
        mask_lat_size = mask_lat_size.transpose(1, 2) # vae에서 (B, F, C, H, W)으로 들어가기 때문에 맞춰주기 위함
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        return latents, torch.concat([mask_lat_size, latent_condition], dim=1), (exo_latent_width, ego_latent_width)
    
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video to latent space (copied from Trainer)"""
        # shape of input video: [B, C, F, H, W]
        vae = self.vae
        video = video.to(vae.device, dtype=vae.dtype)
        
        # Ensure video is in [B, C, F, H, W] format
        if video.dim() == 5 and video.shape[1] != 3:
            # If video is [B, F, C, H, W], transpose to [B, C, F, H, W]
            video = video.permute(0, 2, 1, 3, 4)
        
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample()
        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(latent.device, latent.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
            latent.device, latent.dtype
        )
        latent = (latent - latents_mean) * latents_std
        return latent
    
    def _setup_ic_lora_latent(self, exo_video, generator, device, height, width, 
                            num_frames, batch_size=1, num_videos_per_prompt=1):
        """Setup shared latent for IC-LoRA (exo + random ego)"""
        if exo_video is None:
            return
            
        # Calculate latent dimensions
        latent_height = height // self.vae_scale_factor_spatial
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        num_channels_latents = self.vae.config.z_dim
        
        # Actual ego = 448 pixels -> 56 latents  
        ego_pixel_width = 448
        ego_latent_width = ego_pixel_width // self.vae_scale_factor_spatial
        
        batch_size = batch_size * num_videos_per_prompt # prepare_latent에서 bsz를 이렇게 넘기고 있음
        
        # Encode exo_video using encode_video method
        # Add batch dimension if needed: [C, F, H, W] -> [1, C, F, H, W]
        if exo_video.dim() == 4:
            exo_video = exo_video.unsqueeze(0)
        exo_latents = self.encode_video(exo_video)  # [B, C, F, H, exo_latent_width]
        exo_latents = exo_latents.repeat(batch_size, 1, 1, 1, 1)
        
        # Generate random ego latent
        ego_shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, ego_latent_width)
        # Generate on CPU with generator, then move to device
        ego_latents = torch.randn(ego_shape, generator=generator, dtype=exo_latents.dtype).to(device)
        
        # Concatenate exo + ego latents
        self._shared_latent_exo_randn = torch.cat([exo_latents, ego_latents], dim=-1)
        
        print(f"Setup IC-LoRA latent: exo_shape={exo_latents.shape}, ego_shape={ego_latents.shape}, total_shape={self._shared_latent_exo_randn.shape}")

    @override
    def encode_image(
        self,
        exo_video: PipelineImageInput,
        device: Optional[torch.device] = None,
    ):
        """
        Override encode_image to use shared latent for image embedding
        Input: exo video only
        Return: CLIPEncode(decode(encode(exo video first frame)+randn_tensor))
        """
        device = device or self._execution_device
        
        # Use full latent for temporal consistency (not just first frame)
        full_latent = self._shared_latent_exo_randn  # [B, C, F, H, W]
        
        # Decode latent back to pixel space
        # Remove VAE normalization first
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(full_latent.device, full_latent.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            full_latent.device, full_latent.dtype
        )
        
        # Denormalize
        denorm_latent = full_latent / latents_std + latents_mean
        
        # Decode full video for temporal consistency, then extract first frame
        decoded_video = self.vae.decode(denorm_latent.to(self.vae.dtype), return_dict=False)[0]
        decoded_image = decoded_video[:, :, 0, :, :]  # Extract first frame [B, C, H, W]
        
        print(f"Decoded image shape: {decoded_image.shape} for CLIP encoding")
        
        # Convert to float32 for CLIP compatibility (BFloat16 not supported by transformers image processor)
        decoded_image = decoded_image.to(dtype=torch.float32)
        
        # Convert from [-1, 1] to [0, 1] range for CLIP processor
        decoded_image = (decoded_image + 1.0) / 2.0
        
        # Use decoded image for CLIP encoding - processor handles all normalization
        image = self.image_processor(images=decoded_image, return_tensors="pt", do_rescale=False).to(device)
        image_embeds = self.image_encoder(**image, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    @override
    def check_inputs(
        self,
        prompt,
        negative_prompt,
        exo_video,
        ego_prior_video,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if exo_video is None or ego_prior_video is None:
            raise ValueError(
                "Provide both `exo_video` or `ego_prior_video`. Cannot leave both `exo_video` and `ego_prior_video` undefined."
            )
        if exo_video is not None and not isinstance(exo_video, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(f"`exo_video` has to be of type `torch.Tensor`, `PIL.Image.Image`, or `list` but is {type(exo_video)}")
        if ego_prior_video is not None and not isinstance(ego_prior_video, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(f"`ego_prior_video` has to be of type `torch.Tensor`, `PIL.Image.Image`, or `list` but is {type(ego_prior_video)}")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

    @override
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        exo_video: PipelineImageInput,
        ego_prior_video: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 448,
        width: int = 784+448, #1232
        num_frames: int = 49,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        attention_GGA: Optional[torch.Tensor] = None,
        attention_mask_GGA: Optional[torch.Tensor] = None,
        cam_rays: Optional[torch.Tensor] = None,
        point_vecs_per_frame: Optional[torch.Tensor] = None,
        cos_sim_scaling_factor: float = 1.0,
        do_kv_cache: bool = True,
        do_custom_test: bool = False,
        save_attn_dir_name: Optional[str] = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PipelineImageInput`):
                The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, defaults to `480`):
                The height of the generated video.
            width (`int`, defaults to `832`):
                The width of the generated video.
            num_frames (`int`, defaults to `49`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `negative_prompt` input argument.
            image_embeds (`torch.Tensor`, *optional*):
                Pre-generated image embeddings. Can be used to easily tweak image inputs (weighting). If not provided,
                image embeddings are generated from the `image` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        device = self._execution_device
        
        # Preprocess exo_video for IC-LoRA setup
        if not isinstance(exo_video, torch.Tensor):
            # Truncate frames and convert PipelineImageInput to tensor and resize to exo dimensions
            exo_video_truncated = exo_video[:num_frames] if isinstance(exo_video, (list, tuple)) else exo_video
            exo_video_processed = self.video_processor.preprocess(exo_video_truncated, height=height, width=width-height).to(device, dtype=torch.float32)
        else:
            exo_video_processed = exo_video.to(device, dtype=torch.float32)
            
        # Setup IC-LoRA latent at the beginning
        self._setup_ic_lora_latent(
            exo_video=exo_video_processed,
            generator=generator,
            device=self._execution_device,
            height=height,
            width=width, # didn't used
            num_frames=num_frames,
            batch_size=1,
            num_videos_per_prompt=num_videos_per_prompt
        )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            exo_video,
            ego_prior_video,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # Encode image embedding
        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        if image_embeds is None:
            if last_image is None:
                image_embeds = self.encode_image(exo_video, device)
            else:
                image_embeds = self.encode_image([exo_video, last_image], device)
        image_embeds = image_embeds.repeat(batch_size, 1, 1)
        image_embeds = image_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.z_dim

        # Preprocess ego_prior_video with proper resizing
        if not isinstance(ego_prior_video, torch.Tensor):
            # Truncate frames and convert PipelineImageInput to tensor and resize to ego dimensions
            ego_prior_video_truncated = ego_prior_video[:num_frames] if isinstance(ego_prior_video, (list, tuple)) else ego_prior_video
            ego_prior_video = self.video_processor.preprocess(ego_prior_video_truncated, height=height, width=height).to(device, dtype=torch.float32)
        else:
            ego_prior_video = ego_prior_video.to(device, dtype=torch.float32)

        if last_image is not None:
            last_image = self.video_processor.preprocess(last_image, height=height, width=width).to(
                device, dtype=torch.float32
            )
        # Calculate individual widths from total width
        exo_width = width-height
        ego_width = height
        
        latents, condition, (exo_latent_width, ego_latent_width) = self.prepare_latents(
            ego_prior_video,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            exo_width,
            ego_width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
            last_image,
        )

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    attention_GGA=attention_GGA,
                    attention_mask_GGA=attention_mask_GGA,
                    point_vecs_per_frame = point_vecs_per_frame,
                    cam_rays = cam_rays,
                    cos_sim_scaling_factor= cos_sim_scaling_factor,
                    do_kv_cache=do_kv_cache,
                    do_custom_test=do_custom_test,
                    save_attn_dir_name=save_attn_dir_name,
                    save_attn_dir_step=f'step_{i:02d}',
                )[0]

                if self.do_classifier_free_guidance:
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                        attention_GGA=attention_GGA,
                        attention_mask_GGA=attention_mask_GGA,
                        point_vecs_per_frame=point_vecs_per_frame,
                        cam_rays=cam_rays,
                        cos_sim_scaling_factor=cos_sim_scaling_factor,
                        do_kv_cache=do_kv_cache,
                        do_custom_test=do_custom_test,
                        save_attn_dir_name=save_attn_dir_name,
                        save_attn_dir_step=f'step_{i:02d}',
                    )[0]
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                # Set exo region (left part) of noise_pred to 0 to prevent exo latents from being updated
                # ego_latent_width = ego_width // self.vae_scale_factor_spatial  # ego_width = height = 448
                noise_pred[:, :, :, :, :-ego_latent_width] = 0.0

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            # video = self.vae.decode(latents, return_dict=False)[0]
            latent_exo = latents[:, :, :, :, :exo_latent_width]  # [B, C, F, H, exo_latent_width]
            latent_ego = latents[:, :, :, :, exo_latent_width:]   # [B, C, F, H, ego_latent_width]
            
            # Decode each part separately
            video_exo = self.vae.decode(latent_exo, return_dict=False)[0]  # [B, 3, F, H_exo, W_exo]
            video_ego = self.vae.decode(latent_ego, return_dict=False)[0]  # [B, 3, F, H_ego, W_ego]
            
            # Concatenate along width dimension
            video = torch.cat([video_exo, video_ego], dim=-1)  # [B, 3, F, H, W_exo + W_ego]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)

class WanI2VSftTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder", "image_encoder", "image_processor"]

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = WanImageToVideoPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = UMT5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

        components.transformer = WanTransformer3DModel.from_pretrained(model_path, subfolder="transformer")

        components.vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae")

        components.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")

        components.image_encoder = CLIPVisionModel.from_pretrained(model_path, subfolder="image_encoder")

        components.image_processor = CLIPImageProcessor.from_pretrained(model_path, subfolder="image_processor")

        return components

    @override
    def prepare_models(self) -> None:
        self.state.transformer_config = self.components.transformer.config

    @override
    def initialize_pipeline(self) -> WanHBHImageToVideoPipeline:
        pipe = WanHBHImageToVideoPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
            image_encoder=self.components.image_encoder,
            image_processor=self.components.image_processor,
        )
        return pipe

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample()
        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(latent.device, latent.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
            latent.device, latent.dtype
        )
        latent = (latent - latents_mean) * latents_std
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(prompt_token_ids.to(self.accelerator.device))[0]
        return prompt_embedding

    @override
    def encode_first_frame(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image using CLIP image encoder for dataset preprocessing
        
        Args:
            image: [C, H, W] single image tensor
            
        Returns:
            image_embedding: [1, sequence_length, embedding_dimension]
        """
        # Ensure image is 3D [C, H, W]
        if image.dim() != 3:
            raise ValueError(f"Expected 3D image tensor [C, H, W], got {image.dim()}D tensor with shape {image.shape}")
        
        # Add batch dimension [1, C, H, W]
        image = image.unsqueeze(0)
        
        # Use CLIP image processor and encoder
        # Ensure image encoder is on the same device as inputs to avoid type/device mismatch
        self.components.image_encoder = self.components.image_encoder.to(self.accelerator.device)
        self.components.image_encoder.eval()

        processed_image = self.components.image_processor(images=image, return_tensors="pt")
        processed_image = {k: v.to(self.accelerator.device) for k, v in processed_image.items()}

        image_embeds = self.components.image_encoder(**processed_image, output_hidden_states=True)
        return image_embeds.hidden_states[-2]  # [1, sequence_length, embedding_dimension]

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {"encoded_exo_ego_gt_video": [], "prompt_embedding": [], "encoded_exo_ego_prior_video": [], "image_embedding": [], "attention_GGA": [], "attention_mask_GGA": [], "point_vecs_per_frame": [], "cam_rays": []}

        for sample in samples:
            encoded_exo_ego_gt_video = sample["encoded_exo_ego_gt_video"]  # exo+ego_gt concatenated and encoded
            prompt_embedding = sample["prompt_embedding"]
            encoded_exo_ego_prior_video = sample["encoded_exo_ego_prior_video"]  # exo+ego_prior concatenated and encoded
            image_embedding = sample["image_embedding"]  # from exo_video

            attention_GGA = sample["attention_GGA"]
            attention_mask_GGA = sample["attention_mask_GGA"]

            point_vecs_per_frame = sample["point_vecs_per_frame"]
            cam_rays = sample["cam_rays"]

            ret["encoded_exo_ego_gt_video"].append(encoded_exo_ego_gt_video)
            ret["prompt_embedding"].append(prompt_embedding)
            ret["encoded_exo_ego_prior_video"].append(encoded_exo_ego_prior_video)
            ret["image_embedding"].append(image_embedding)
            if attention_GGA is not None:
                ret["attention_GGA"].append(attention_GGA)
                ret["attention_mask_GGA"].append(attention_mask_GGA)
                ret["point_vecs_per_frame"].append(point_vecs_per_frame)
                ret["cam_rays"].append(cam_rays)

        ret["encoded_exo_ego_gt_video"] = torch.stack(ret["encoded_exo_ego_gt_video"])
        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        ret["encoded_exo_ego_prior_video"] = torch.stack(ret["encoded_exo_ego_prior_video"])  # [B, C, F, H, W_exo+W_ego]
        ret["image_embedding"] = torch.stack(ret["image_embedding"])

        if len(ret["attention_GGA"]) > 0:
            ret["attention_GGA"] = torch.stack(ret["attention_GGA"])
            ret["attention_mask_GGA"] = torch.stack(ret["attention_mask_GGA"])
            ret["point_vecs_per_frame"] = torch.stack(ret["point_vecs_per_frame"])
            ret["cam_rays"] = torch.stack(ret["cam_rays"])
        return ret

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.components.scheduler.sigmas.to(device=self.accelerator.device, dtype=dtype)
        schedule_timesteps = self.components.scheduler.timesteps.to(self.accelerator.device)
        timesteps = timesteps.to(self.accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        # Unwrap in case the transformer is wrapped by DDP/Accelerate
        transformer = unwrap_model(self.accelerator, self.components.transformer)
        transformer_dtype = transformer.dtype

        prompt_embedding = batch["prompt_embedding"].to(transformer_dtype)
        latent = batch["encoded_exo_ego_gt_video"].to(transformer_dtype)
        latent_condition = batch["encoded_exo_ego_prior_video"].to(transformer_dtype)
        image_embedding = batch["image_embedding"].to(transformer_dtype)

        #####
        attention_GGA = batch["attention_GGA"] if "attention_GGA" in batch else None
        attention_mask_GGA = batch["attention_mask_GGA"] if "attention_mask_GGA" in batch else None
        point_vecs_per_frame = batch["point_vecs_per_frame"] if "point_vecs_per_frame" in batch else None
        cam_rays = batch["cam_rays"] if "cam_rays" in batch else None
        #####


        batch_size, num_channels, num_frames, height, width = latent.shape
        vae_scale_factor_temporal = 2 ** sum(self.components.vae.config.temperal_downsample)
        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # latent_condition is already encoded in the dataset, no need to encode again

        # Get actual number of frames from video_condition
        actual_num_frames = (num_frames - 1) * vae_scale_factor_temporal + 1
        
        mask_lat_size = torch.ones(latent_condition.shape[0], 1, actual_num_frames, latent_condition.shape[3], latent_condition.shape[4])
        
        exo_width, ego_width = 784, 448 #! HARD CODING
        vae_scale_factor_spatial = 2 ** len(self.components.vae.config.temperal_downsample) # 8
        exo_latent_width = exo_width // vae_scale_factor_spatial  
        ego_latent_width = ego_width // vae_scale_factor_spatial
        # Set exo_view mask (left part) to 1.0
        mask_lat_size[:, :, :, :, :exo_latent_width] = 1.0
        # Set ego_view mask (right part) to 0.0  
        mask_lat_size[:, :, :, :, exo_latent_width:] = 0.0
        
        # Use the same masking logic as WanHBHImageToVideoPipeline.prepare_latents
        # Apply temporal masking (only first frame is visible, rest are masked)
        mask_lat_size[:, :, list(range(1, actual_num_frames - 1))] = 0
            
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, vae_scale_factor_temporal, latent_condition.shape[-2], latent_condition.shape[-1])
        mask_lat_size = mask_lat_size.transpose(1, 2)  # VAE expects (B, F, C, H, W)
        mask_lat_size = mask_lat_size.to(latent_condition)

        condition = torch.concat([mask_lat_size, latent_condition], dim=1) # [B, 20, 13, latent_H, latent_W]

        # Sample a random timestep for each sample
        timesteps_idx = torch.randint(0, self.components.scheduler.config.num_train_timesteps, (batch_size,))
        timesteps_idx = timesteps_idx.long()
        timesteps = self.components.scheduler.timesteps[timesteps_idx].to(device=latent.device)
        sigmas = self.get_sigmas(timesteps, n_dim=latent.ndim, dtype=latent.dtype)
        # Add noise to latent
        noise = torch.randn_like(latent)
        noisy_latents = (1.0 - sigmas) * latent + sigmas * noise
        
        # Apply IC-LoRA style masking: keep exo region (left) as original, only noise ego region (right)
        noisy_latents[:, :, :, :, :-ego_latent_width] = latent[:, :, :, :, :-ego_latent_width]
        
        target = noise - latent
        latent_model_input = torch.cat([noisy_latents, condition], dim=1) # [B, 36, 13, latent_H, latent_W]


        predicted_noise = self.components.transformer(  
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embedding,
            encoder_hidden_states_image=image_embedding,
            timestep=timesteps,
            return_dict=False,
            attention_GGA=attention_GGA,
            attention_mask_GGA=attention_mask_GGA,
            point_vecs_per_frame=point_vecs_per_frame,
            cam_rays=cam_rays,
            cos_sim_scaling_factor=self.args.cos_sim_scaling_factor,
            do_kv_cache=True,
        )[0]

        ego_predicted_noise = predicted_noise[:, :, :, :, -ego_latent_width:]
        ego_target = target[:, :, :, :, -ego_latent_width:]
        
        loss = torch.mean(((ego_predicted_noise.float() - ego_target.float()) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()

        return loss

    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: WanHBHImageToVideoPipeline
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        prompt, exo_video, ego_prior_video = eval_data["prompt"], eval_data["exo_video"], eval_data["ego_prior_video"]

        video_generate = pipe(
            num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            prompt=prompt,
            exo_video=exo_video,
            ego_prior_video=ego_prior_video,
            generator=self.state.generator,
            num_inference_steps=50,
            guidance_scale=5.0,
            num_videos_per_prompt=1,
        ).frames[0]
        return [("video", video_generate)]

register("wan-i2v", "sft", WanI2VSftTrainer)
