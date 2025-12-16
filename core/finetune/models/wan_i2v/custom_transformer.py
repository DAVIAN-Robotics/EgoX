# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, Optional, Tuple, Union
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class WanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        attention_GGA: Optional[torch.Tensor] = None,
        attention_mask_GGA: Optional[torch.Tensor] = None,
        cos_sim: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        do_kv_cache: bool = False,
    ) -> torch.Tensor:

        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
                x_rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        if cos_sim is not None and not do_kv_cache:
            cos_sim = cos_sim + 1.0
            attn_mask_from_cos_sim = torch.log(cos_sim + 1e-6)
            attention_mask_GGA = attention_mask_GGA[0,:,0] 
            #! Exo attention
            hidden_states_exo = F.scaled_dot_product_attention(
                query[:, :, attention_mask_GGA==1],
                key,
                value,
                dropout_p=0.0,
                is_causal=False
            )
            #! Ego attention
            attn_mask_ego = attn_mask_from_cos_sim[:, attention_mask_GGA==0, :]
            hidden_states_ego = F.scaled_dot_product_attention(
                query[:, :, attention_mask_GGA==0, :],
                key,
                value,
                attn_mask=attn_mask_ego.to(query.dtype),
                dropout_p=0.0,
                is_causal=False
            )   

            #! Combine exo and ego
            hidden_states = torch.zeros_like(query)
            hidden_states[:, :, attention_mask_GGA==1, :] = hidden_states_exo.to(query.dtype)
            hidden_states[:, :, attention_mask_GGA==0, :] = hidden_states_ego.to(query.dtype)
        else:
            # print("Normal attention")
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states, kv_cache


class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, pos_embed_seq_len=None):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_seq_len, in_features))
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.view(-1, 2 * seq_len, embed_dim)
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ):
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, theta: float = 10000.0
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        freqs = []
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=freqs_dtype
            )
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        freqs = self.freqs.to(hidden_states.device)
        freqs = freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        return freqs


class SelfAttention(Attention):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        qk_norm: Optional[str] = None,
        eps: float = 1e-6,
        bias: bool = True,
        cross_attention_dim: Optional[int] = None,
        out_bias: bool = True,
        processor: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(
            query_dim=query_dim,
            heads=heads,
            dim_head=dim_head,
            cross_attention_dim=cross_attention_dim,
            qk_norm=qk_norm,
            eps=eps,
            bias=bias,
            out_bias=out_bias,
            processor=processor,
            **kwargs,
        )

        self.kv_cache = None
        self.cond_size = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[torch.Tensor] = None,
        attention_GGA: Optional[torch.Tensor] = None,
        attention_mask_GGA: Optional[torch.Tensor] = None,
        cos_sim: Optional[torch.Tensor] = None,
        do_kv_cache: bool = False,
    ) -> torch.Tensor:

        hidden_states, self.kv_cache = super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            rotary_emb=rotary_emb,
            attention_GGA=attention_GGA,
            attention_mask_GGA=attention_mask_GGA,
            cos_sim=cos_sim,
            kv_cache=self.kv_cache,
            do_kv_cache=do_kv_cache,
        )
        return hidden_states


class CrossAttention(Attention):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        qk_norm: Optional[str] = None,
        eps: float = 1e-6,
        bias: bool = True,
        cross_attention_dim: Optional[int] = None,
        out_bias: bool = True,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: bool = True,
        processor: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(
            query_dim=query_dim,
            heads=heads,
            dim_head=dim_head,
            cross_attention_dim=cross_attention_dim,
            qk_norm=qk_norm,
            eps=eps,
            bias=bias,
            out_bias=out_bias,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=added_proj_bias,
            processor=processor,
            **kwargs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        hidden_states, _ = super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=None,
        )
        return hidden_states
    

class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = SelfAttention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=WanAttnProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = CrossAttention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=WanAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        attention_GGA: Optional[torch.Tensor] = None,
        attention_mask_GGA: Optional[torch.Tensor] = None,
        cos_sim: Optional[torch.Tensor] = None,
        do_kv_cache: bool = False,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)
        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, 
                                attention_GGA=attention_GGA, attention_mask_GGA=attention_mask_GGA, cos_sim=cos_sim, 
                                do_kv_cache=do_kv_cache)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states ) #, attention_GGA=attention_GGA, attention_mask_GGA=attention_mask_GGA)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class WanTransformer3DModel_GGA(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        attention_GGA: Optional[torch.Tensor] = None,
        attention_mask_GGA: Optional[torch.Tensor] = None,
        cam_rays: Optional[torch.Tensor] = None,
        point_vecs_per_frame: Optional[torch.Tensor] = None,
        cos_sim_scaling_factor: float = 1.0,
        do_kv_cache: bool = False,
        do_custom_test: bool = False,
        save_attn_map: bool = True,
        save_attn_dir_name: Optional[str] = None,
        save_attn_map_mask: Optional[torch.Tensor] = None,
        save_attn_map_avg: Optional[torch.Tensor] = None,
        save_attn_dir_step: str = "",
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w


        rotary_emb = self.rope(hidden_states)
        hidden_states = self.patch_embedding(hidden_states)

        if save_attn_map: # B, C, F, H, W
            save_attn_map_mask = torch.zeros_like(hidden_states[..., -(post_patch_height - post_patch_width):])
            save_attn_map_mask[...,:1,:,:] = 1
            save_attn_map_mask = save_attn_map_mask.flatten(2).transpose(1, 2)

        hidden_states = hidden_states.flatten(2).transpose(1, 2)


        if attention_GGA is not None: 

            if len(point_vecs_per_frame.shape) != 5:
                point_vecs_per_frame = point_vecs_per_frame.squeeze()
            if len(cam_rays.shape) != 4:
                cam_rays = cam_rays.squeeze()
            point_vecs_per_frame = torch.cat([point_vecs_per_frame, cam_rays.unsqueeze(0).repeat(point_vecs_per_frame.shape[0], 1, 1, 1, 1)], dim=-2) # F, F, H, W, C
            point_vecs_per_frame = point_vecs_per_frame.permute(0,4,1,2,3) # F, C, F, H, W
            point_vecs_per_frame = F.avg_pool3d(
                point_vecs_per_frame,
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2),
                padding=0,
                ceil_mode=False,
                count_include_pad=False,
            ).to(dtype=hidden_states.dtype)

            
            attention_GGA = attention_GGA.permute(0,4,1,2,3) # B, C, F, H, W
            attention_GGA = F.avg_pool3d(
                attention_GGA,
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2),
                padding=0,
                ceil_mode=False,
                count_include_pad=False,
            ).to(dtype=hidden_states.dtype)
            attention_mask_exo = torch.ones_like(attention_GGA[...,:-(post_patch_height - post_patch_width)])
            attention_mask_ego = torch.zeros_like(attention_GGA[...,-(post_patch_height - post_patch_width) :])
            attention_mask_GGA = torch.cat([attention_mask_exo, attention_mask_ego], dim=-1)

            attention_GGA = attention_GGA.flatten(2).transpose(1, 2)        # (f1,h1,w1), (f1,h1,w2), ... (f1,h_end,w_end), (f2,h1,w1), ...
            attention_mask_GGA = attention_mask_GGA.flatten(2).transpose(1, 2)

            FF, C, frame_num, H, W = point_vecs_per_frame.shape
            point_vecs_per_frame[...,:,W-H:] = torch.zeros_like(point_vecs_per_frame[...,:,W-H:])
            point_vecs_per_frame = point_vecs_per_frame.flatten(2).transpose(1,2)
            frame_cut = H * W

            cos_sim = None
            cos_sim = torch.zeros((1, attention_GGA.shape[1], attention_GGA.shape[1]), device=hidden_states.device) #! 1, 28028, 28028
            for i in range(FF):
                cos_sim[:, i*frame_cut:(i+1)*frame_cut,:] = torch.matmul(attention_GGA[:,i*frame_cut:(i+1)*frame_cut,:], point_vecs_per_frame[i:i+1].transpose(-1, -2)).to(hidden_states.device)

            cos_sim = torch.clamp(cos_sim, min=-1.0, max=1.0)
            
            #! Scaling with hyperparameter
            if cos_sim_scaling_factor > 1.0:
                mask = cos_sim > 0
                cos_sim[mask] = cos_sim[mask] * cos_sim_scaling_factor
            else:
                cos_sim = cos_sim * cos_sim_scaling_factor

            attention_mask_GGA_ = attention_mask_GGA[:,:,0] #! 1, 28028
            mask = (attention_mask_GGA_[:, :, None] == 1).to(hidden_states.device) 
            cos_sim.masked_fill_(mask, 0.0) 

        else:
            cos_sim = None
        
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        cur_block = 0
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, attention_GGA, attention_mask_GGA, cos_sim)
                cur_block += 1
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, attention_GGA, attention_mask_GGA, cos_sim=cos_sim, do_kv_cache=do_kv_cache)
                cur_block += 1
                
        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
