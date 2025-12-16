import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override
import PIL

from core.finetune.constants import LOG_LEVEL, LOG_NAME
import numpy as np

from .utils import (
    load_images,
    load_images_from_videos,
    load_prompts,
    load_videos,
    preprocess_image_with_resize,
    preprocess_video_with_buckets,
    preprocess_video_with_resize,
    generate_uniform_pointmap,
    load_from_json_file,
    iproj_disp
)


if TYPE_CHECKING:
    from core.finetune.trainer import Trainer

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(LOG_NAME, LOG_LEVEL)


class BaseWanDataset(Dataset):
    """
    Base dataset class for IC-LoRA Image-to-Video (I2V) training.

    This dataset loads prompts, exo videos, ego ground truth videos, and ego prior videos for IC-LoRA training.

    Args:
        data_root (str): Root directory containing the dataset files
        caption_column (str): Path to file containing text prompts/captions
        exo_video_column (str): Path to file containing exocentric video paths
        ego_gt_video_column (str): Path to file containing ego ground truth video paths
        ego_prior_video_column (str): Path to file containing ego prior video paths
        device (torch.device): Device to load the data on
        trainer (Trainer): Trainer instance containing encode functions
    """

    def __init__(
        self,
        data_root: str,
        caption_column: str,
        exo_video_column: str,
        ego_gt_video_column: str,
        ego_prior_video_column: str,
        prompt: str,
        device: torch.device,
        meta_data_file: str = None,
        trainer: "Trainer" = None,
        max_frames: int = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        data_root = Path(data_root)
        if meta_data_file is not None:
            import os
            meta_data = load_from_json_file(meta_data_file)
            with open(prompt, "r", encoding="utf-8") as file:
                self.prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
            self.exo_videos = []
            self.ego_gt_videos = []
            self.ego_prior_videos = []
            self.depth_map_paths = []
            self.depth_intrinsics = []
            self.camera_extrinsics = []
            self.camera_intrinsics = []
            self.ego_extrinsics = []
            self.ego_intrinsics = []

            for ego_filename, meta in meta_data.items():
                
                self.exo_videos.append(Path(os.path.join(meta['exo_video_path']))) #! dohyeon data
                self.ego_gt_videos.append(Path(os.path.join(meta['ego_video_path']))) #! Same but width wise concated
                self.ego_prior_videos.append(Path(os.path.join(meta['ego_prior_path']))) #! None
                
                self.depth_map_paths.append(Path(os.path.join(data_root, 'depth_maps', meta['take_name'])))
                self.depth_intrinsics.append(Path(os.path.join(meta['vipe_results_path'], 'intrinsics', meta['best_camera']+'.npz')))
                self.camera_extrinsics.append(meta['camera_extrinsics'])
                self.camera_intrinsics.append(meta['camera_intrinsics'])
                self.ego_extrinsics.append(meta['ego_extrinsics'])
                self.ego_intrinsics.append(meta['ego_intrinsics'])
        else:
            self.prompts = load_prompts(data_root / caption_column)
            self.exo_videos = load_videos(data_root / exo_video_column)
            self.ego_gt_videos = load_videos(data_root / ego_gt_video_column)
            self.ego_prior_videos = load_videos(data_root / ego_prior_video_column)
            self.depth_map_paths = None
            self.depth_intrinsics = None
            self.camera_extrinsics = None
            self.camera_intrinsics = None
            self.ego_extrinsics = None
            self.ego_intrinsics = None

        self.max_frames = max_frames
        
        self.trainer = trainer
        self.device = device
        self.encode_video = trainer.encode_video
        self.encode_text = trainer.encode_text

        # Check if number of prompts matches number of videos
        if not (len(self.exo_videos) == len(self.ego_gt_videos) == len(self.ego_prior_videos) == len(self.prompts)):
            raise ValueError(
                f"Expected length of prompts, exo_videos, ego_gt_videos and ego_prior_videos to be the same but found "
                f"{len(self.prompts)}, {len(self.exo_videos)}, {len(self.ego_gt_videos)}, {len(self.ego_prior_videos)}. "
                f"Please ensure that the number of caption prompts and videos match in your dataset."
            )

        # Check if all video files exist
        all_videos = self.exo_videos + self.ego_gt_videos + self.ego_prior_videos
        if any(not path.is_file() for path in all_videos):
            missing_file = next(path for path in all_videos if not path.is_file())
            raise ValueError(
                f"Some video files were not found. Please ensure that all video files exist in the dataset directory. Missing file: {missing_file}"
            )

    def __len__(self) -> int:
        return len(self.exo_videos)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        while True:
            try:
                ret = self.getitem(index)
                if ret['video_metadata']['num_frames'] != 13: # if 49 frames, 1+(49-1)/4=13
                    logger.warning(
                        f"Dataset __getitem__: latent num_frames mismatch at index={index} -> "
                        f"got={ret['video_metadata']['num_frames']} expected=13",
                        main_process_only=False,
                    )
                    raise ValueError("Not enough frames")
                break
            except Exception as e:
                logger.warning(f"Dataset __getitem__ retry due to: {e} (index={index})", main_process_only=False)
                index = (index + 1) % len(self.exo_videos)
        return ret

    def getitem(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            return index

        prompt = self.prompts[index]
        exo_video = self.exo_videos[index]
        ego_gt_video = self.ego_gt_videos[index]
        
        ego_prior_video = self.ego_prior_videos[index]

        if self.depth_map_paths is not None:
            depth_map_path = self.depth_map_paths[index]
            depth_intrinsics = self.depth_intrinsics[index]
            camera_extrinsics = self.camera_extrinsics[index]
            camera_intrinsics = self.camera_intrinsics[index]
            ego_extrinsics = self.ego_extrinsics[index]
            ego_intrinsics = self.ego_intrinsics[index]


        train_resolution = self.trainer.args.train_resolution
        if isinstance(train_resolution, (list, tuple)):
            train_resolution_str = f"{train_resolution[0]}x{train_resolution[1]}x{train_resolution[2]}"
        else:
            train_resolution_str = str(train_resolution)
        cache_dir = self.trainer.args.data_root / "cache"

        video_latent_dir = cache_dir / "video_latent" / "wan-i2v" / train_resolution_str
        prompt_embeddings_dir = cache_dir / "prompt_embeddings"
        image_embedding_dir = cache_dir / "image_embeddings" / "wan-i2v" / train_resolution_str
        
        video_latent_dir.mkdir(parents=True, exist_ok=True)
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)
        image_embedding_dir.mkdir(parents=True, exist_ok=True)

        prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
        prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
        
        if self.depth_map_paths is None:
            scene_name = str(exo_video).split('/')[-5] 
        else:
            scene_name = str(depth_intrinsics).split('/')[-4] 
        exo_name = f"{scene_name}_exo_{exo_video.stem}"
        combined_name = f"{exo_name}_ego_gt_prior"
        encoded_video_path = video_latent_dir / (combined_name + ".safetensors")
        image_embedding_combined_name = f"{exo_name}_ego_gt"
        image_embedding_path = image_embedding_dir / (image_embedding_combined_name + ".safetensors")
        # Load prompt embedding
        if prompt_embedding_path.exists():
            prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
            logger.debug(
                f"process {self.trainer.accelerator.process_index}: Loaded prompt embedding from {prompt_embedding_path}",
                main_process_only=False,
            )
        else:
            prompt_embedding = self.encode_text(prompt)
            prompt_embedding = prompt_embedding.to("cpu")
            # [1, seq_len, hidden_size] -> [seq_len, hidden_size]
            prompt_embedding = prompt_embedding[0]
            save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
            logger.info(f"Saved prompt embedding to {prompt_embedding_path}", main_process_only=False)
        # Load or process all videos first (avoid duplication)
        need_processing = not encoded_video_path.exists() or not image_embedding_path.exists()

        attn_maps_dir = cache_dir / "attn_maps" / "wan-i2v" / train_resolution_str
        attn_maps_dir.mkdir(parents=True, exist_ok=True)
        attn_maps_paths = attn_maps_dir / (combined_name + ".safetensors")

        if need_processing:
            logger.info(f"Processing videos for {encoded_video_path} and {image_embedding_path}", main_process_only=False)
            
            # Load and process all videos once
            if self.depth_map_paths is not None:
                exo_video_tensor, _ = self.preprocess(exo_video, None, "exo")
            else:
                exo_video_tensor, _ = self.preprocess(exo_video, None, "exo")

            try:
                logger.info(
                    f"Preprocess exo video: path={exo_video} shape={tuple(exo_video_tensor.shape)}",
                    main_process_only=False,
                )
            except Exception:
                pass
            exo_video_tensor = self.video_transform(exo_video_tensor)
            
            if self.depth_map_paths is not None:
                ego_gt_video_tensor, _ = self.preprocess(ego_gt_video, None, "ego_gt")
            else:
                ego_gt_video_tensor, _ = self.preprocess(ego_gt_video, None, "ego_gt")

            try:
                logger.info(
                    f"Preprocess ego_gt video: path={ego_gt_video} shape={tuple(ego_gt_video_tensor.shape)}",
                    main_process_only=False,
                )
            except Exception:
                pass
            ego_gt_video_tensor = self.video_transform(ego_gt_video_tensor)
            
            if self.depth_map_paths is not None:
                try:
                    ego_prior_video_tensor, _ = self.preprocess(ego_prior_video, None, "ego_prior")
                except:
                    ego_prior_video_tensor, _ = self.preprocess(ego_prior_video, None, "ego_prior_GGA")
            else:
                ego_prior_video_tensor, _ = self.preprocess(ego_prior_video, None, "ego_prior")
            try:
                logger.info(
                    f"Preprocess ego_prior video: path={ego_prior_video} shape={tuple(ego_prior_video_tensor.shape)}",
                    main_process_only=False,
                )
            except Exception:
                pass
            ego_prior_video_tensor = self.video_transform(ego_prior_video_tensor)
            
            # Encode each video separately, then concatenate in latent space
            with torch.no_grad():
                encoded_exo_video = self.encode_video(exo_video_tensor.permute(1, 0, 2, 3).to(self.device).unsqueeze(0))
                encoded_ego_gt_video = self.encode_video(ego_gt_video_tensor.permute(1, 0, 2, 3).to(self.device).unsqueeze(0))
                encoded_ego_prior_video = self.encode_video(ego_prior_video_tensor.permute(1, 0, 2, 3).to(self.device).unsqueeze(0))
                try:
                    logger.info(
                        f"Encoded shapes: exo={tuple(encoded_exo_video.shape)}, "
                        f"ego_gt={tuple(encoded_ego_gt_video.shape)}, ego_prior={tuple(encoded_ego_prior_video.shape)}",
                        main_process_only=False,
                    )
                except Exception:
                    pass
                
                # Concatenate in latent space (width-wise)
                encoded_exo_ego_gt_video = torch.cat([encoded_exo_video, encoded_ego_gt_video], dim=-1)
                encoded_exo_ego_prior_video = torch.cat([encoded_exo_video, encoded_ego_prior_video], dim=-1)
                try:
                    logger.info(
                        f"Concatenated latent shapes: exo+ego_gt={tuple(encoded_exo_ego_gt_video.shape)}, "
                        f"exo+ego_prior={tuple(encoded_exo_ego_prior_video.shape)}",
                        main_process_only=False,
                    )
                except Exception:
                    pass
                
                # Remove batch dim -> [C, F, H, W] to align with collate_fn stacking
                if encoded_exo_ego_gt_video.dim() == 5 and encoded_exo_ego_gt_video.shape[0] == 1:
                    encoded_exo_ego_gt_video = encoded_exo_ego_gt_video.squeeze(0)
                if encoded_exo_ego_prior_video.dim() == 5 and encoded_exo_ego_prior_video.shape[0] == 1:
                    encoded_exo_ego_prior_video = encoded_exo_ego_prior_video.squeeze(0)

                # Move to CPU for saving
                encoded_exo_ego_gt_video = encoded_exo_ego_gt_video.to("cpu")
                encoded_exo_ego_prior_video = encoded_exo_ego_prior_video.to("cpu")

            if self.depth_map_paths is not None:
                device = encoded_exo_ego_gt_video.device
                # 448, 1232

                C, F, H, W = encoded_exo_ego_gt_video.shape
                exo_H, exo_W = H, W - H
                W = H
                # in depth_map_paths folder, iterate all npy files
                depth_maps = []
                for depth_map_file in sorted(depth_map_path.glob("*.npy")):
                    depth_map = np.load(depth_map_file)
                    depth_maps.append(torch.from_numpy(depth_map).unsqueeze(0))
                depth_maps = torch.cat(depth_maps, dim=0) # [F, H, W]
                ego_intrinsic=torch.tensor(ego_intrinsics)          #! (3,3)
                ego_extrinsic=torch.tensor(ego_extrinsics)          #! (F, 3, 4)
                camera_extrinsics=torch.tensor(camera_extrinsics)   #! (3,4)
                camera_intrinsics=torch.tensor(camera_intrinsics)   #! (3,3)

                # make ego_extrinsic (f, 3, 4) to (f, 4, 4)
                if ego_extrinsic.shape[1] == 3 and ego_extrinsic.shape[2] == 4:
                    ego_extrinsic = torch.cat([ego_extrinsic, torch.tensor([[[0, 0, 0, 1]]], dtype=ego_extrinsic.dtype).expand(ego_extrinsic.shape[0], -1, -1)], dim=1)
                if camera_extrinsics.shape == (3, 4):
                    camera_extrinsics = torch.cat([torch.tensor(camera_extrinsics, dtype=ego_extrinsic.dtype), torch.tensor([[0, 0, 0, 1]], dtype=ego_extrinsic.dtype)], dim=0)
                scale = 1/8
                scaled_intrinsic = ego_intrinsic.clone()
                scaled_intrinsic[0, 0] *= scale  # fx' = fx * s
                scaled_intrinsic[1, 1] *= scale  # fy' = fy * s
                scaled_intrinsic[0, 2] *= scale  # cx' = cx * s
                scaled_intrinsic[1, 2] *= scale

                ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
                ones = torch.ones_like(xs)
                pixel_coords = torch.stack([xs, ys, ones], dim=-1).view(-1, 3).to(dtype=ego_intrinsic.dtype)  # (H*W, 3)
                inv_intrinsics = torch.linalg.inv(scaled_intrinsic)  # (3, 3)
                cam_rays = (inv_intrinsics @ pixel_coords.T).T  # (H*W, 3)
                cam_rays = cam_rays / torch.norm(cam_rays, dim=-1, keepdim=True)  # (H*W, 3)
                cam_rays = cam_rays.view(H, W, 3)  # (H, W, 3)
                cam_rays = cam_rays.unsqueeze(0).expand(F, -1, -1, -1)  # (F, H, W, 3)
                cam_rays = cam_rays.reshape(F, H * W, 3)  # (F, H*W, 3)
                cam_rays = cam_rays @ ego_extrinsic[::4, :3, :3].transpose(-1, -2) #! TODO: F : 49 -> 13

                cam_dirs = cam_rays  # (B, H*W, 3)
                cam_dirs = cam_dirs / torch.norm(cam_dirs, dim=-1, keepdim=True)  # (B, H*W, 3)
                cam_rays = cam_dirs
                cam_rays = cam_rays.view(F, H, W, 3)  # (B, H, W, 3)

                depth_intrinsics_ = np.load(depth_intrinsics)
                depth_intrinsics = depth_intrinsics_['data'][0:1,:] #! (3,3) 

                disp_v, disp_u = torch.meshgrid(
                    torch.arange(depth_maps.shape[1], device=device).float(),
                    torch.arange(depth_maps.shape[2],device=device).float(),
                    indexing="ij",
                )

                disp = torch.ones_like(disp_v)
                
                disp_cpu = disp.cpu()
                disp_u_cpu = disp_u.cpu()
                disp_v_cpu = disp_v.cpu()
                
                pts, _, _ = iproj_disp(torch.from_numpy(depth_intrinsics), disp_cpu, disp_u_cpu, disp_v_cpu)
                
                if isinstance(pts, torch.Tensor):
                    pts = pts.to(device)
                else:
                    pts = torch.from_numpy(pts).to(device).float()
                
                rays = pts[..., :3] 
                rays = rays / rays[..., 2:3]    
                rays = rays.unsqueeze(0).expand(depth_maps.size(0), -1, -1, -1)  # (F, H, W, 3)
                camera_extrinsics_c2w = torch.linalg.inv(camera_extrinsics)

                pcd_camera = rays * depth_maps.unsqueeze(-1)

                point_map = pcd_camera.to(dtype=camera_extrinsics_c2w.dtype)
                point_map = torch.tensor(point_map) #! [F, H, W, 3]

                p_f, p_h, p_w, p_p = point_map.shape
                point_map_world = point_map.reshape(-1, 3)
                #! Point map to world coords
                camera_extrinsics_c2w = torch.linalg.inv(camera_extrinsics)
                ones_point = torch.ones(point_map_world.shape[0], 1, device=point_map_world.device)
                point_map_world = torch.cat([point_map_world, ones_point], dim=-1) #! [F, H*W, 4]

                point_map_world = (camera_extrinsics_c2w @ point_map_world.T).T[...,:3]
                point_map = point_map_world.reshape(p_f, p_h, p_w, 3).permute(0, 3, 1, 2)

                point_map = torch.nn.functional.interpolate(point_map, size=(exo_H, exo_W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)#.squeeze(0).permute(1, 2, 0) #! [B, H, W, 3]
                
                ego_extrinsic_c2w = torch.linalg.inv(ego_extrinsic)

                cam_origins = ego_extrinsic_c2w[::4, :3, 3].unsqueeze(1).expand(-1, exo_H * exo_W, -1)  # (B, H*W, 3)
                cam_origins = cam_origins.view(F, exo_H, exo_W, 3)  # (B, H, W, 3)

                #! get point map to ego_cam vectors
                if point_map.size(0) != ego_extrinsic_c2w.size(0):
                    min_size = min(point_map.size(0), ego_extrinsic_c2w.size(0))
                    point_map = point_map[:min_size]
                
                point_vecs_per_frame = []
                for i in range(cam_origins.size(0)):
                    point_vec = point_map[::4] - cam_origins[i].unsqueeze(0)  #! [B, H, W, 3]
                    point_vec = point_vec / torch.norm(point_vec, dim=-1, keepdim=True)  #! [B,H, W, 3]
                    point_vecs_per_frame.append(point_vec)
                point_vecs_per_frame = torch.stack(point_vecs_per_frame, dim=0)  #! [B, B, H, W, 3]

                point_vecs = point_map[::4] - cam_origins #! [B, H, W, 3]
                point_vecs = point_vecs / torch.norm(point_vecs, dim=-1, keepdim=True) #! [B, H, W, 3]

                # Apply simple 90-degree rotation to fix image orientation
                cam_rays = cam_rays @ torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], device=cam_rays.device, dtype=cam_rays.dtype)  #! [B, H, W, 3]

                attn_maps = torch.cat((point_vecs, cam_rays), dim = 2) #! iclora
                attn_masks = torch.cat((torch.ones_like(point_vecs), torch.zeros_like(cam_rays)), dim = 2) #! iclora


        # Load image embedding
        if image_embedding_path.exists():
            image_embedding = load_file(image_embedding_path)["image_embedding"]
            # Backward compatibility: old caches may include a batch dim -> squeeze to 2D [L, D]
            if image_embedding.dim() == 3 and image_embedding.shape[0] == 1:
                image_embedding = image_embedding.squeeze(0)
            logger.debug(f"Loaded image embedding from {image_embedding_path}", main_process_only=False)
        else:
            # Generate image embedding from exo+ego_gt concat first frame
            # Concatenate first frames in pixel space for CLIP encoding
            exo_ego_gt_concat = torch.cat([exo_video_tensor, ego_gt_video_tensor], dim=-1)  # [F, C, H, W_total]
            exo_ego_gt_first_frame = exo_ego_gt_concat[0]  # [C, H, W_total] in [-1,1]
            # Convert [-1,1] -> [0,1] and pass tensor directly to image_processor path
            exo_ego_gt_first_frame = (exo_ego_gt_first_frame + 1.0) / 2.0  # [0,1]
            image_embedding = self.trainer.encode_first_frame(exo_ego_gt_first_frame)
            # encode_first_frame returns [1, L, D]; remove batch dim -> [L, D]
            if image_embedding.dim() == 3 and image_embedding.shape[0] == 1:
                image_embedding = image_embedding.squeeze(0)
            image_embedding = image_embedding.to("cpu")
            
            # Save image embedding to cache
            save_file({"image_embedding": image_embedding}, image_embedding_path)
            logger.info(f"Saved image embedding to {image_embedding_path}", main_process_only=False)

        # Load encoded videos
        if encoded_video_path.exists():
            loaded = load_file(encoded_video_path)
            encoded_exo_ego_gt_video = loaded["encoded_exo_ego_gt_video"]
            encoded_exo_ego_prior_video = loaded["encoded_exo_ego_prior_video"]
            # Backward compatibility: old caches may include a batch dim -> squeeze to 4D
            if encoded_exo_ego_gt_video.dim() == 5 and encoded_exo_ego_gt_video.shape[0] == 1:
                encoded_exo_ego_gt_video = encoded_exo_ego_gt_video.squeeze(0)
            if encoded_exo_ego_prior_video.dim() == 5 and encoded_exo_ego_prior_video.shape[0] == 1:
                encoded_exo_ego_prior_video = encoded_exo_ego_prior_video.squeeze(0)
            logger.debug(f"Loaded encoded videos from {encoded_video_path}", main_process_only=False)
        else:
            # encoded_exo_ego_gt_video and encoded_exo_ego_prior_video already computed above
            
            # Save video encodings to cache
            save_data = {
                "encoded_exo_ego_gt_video": encoded_exo_ego_gt_video,
                "encoded_exo_ego_prior_video": encoded_exo_ego_prior_video,
            }
            save_file(save_data, encoded_video_path)
            logger.info(f"Saved encoded videos to {encoded_video_path}", main_process_only=False)


        ##############
        if attn_maps_paths.exists() and not self.is_t2v:
            loaded = load_file(attn_maps_paths)
            attn_maps = loaded["attn_maps"]
            attn_masks = loaded["attn_masks"]
            cam_rays = loaded["cam_rays"]
            point_vecs_per_frame = loaded["point_vecs"]
            if attn_maps.dim() == 5 and attn_maps.shape[0] == 1:
                attn_maps = attn_maps.squeeze(0)
            if attn_masks.dim() == 5 and attn_masks.shape[0] == 1:
                attn_masks = attn_masks.squeeze(0)
            logger.debug(f"Loaded attention maps from {attn_maps_paths}", main_process_only=False)
        else:
            save_data = {
                "attn_maps": attn_maps,
                "attn_masks": attn_masks,
                "point_vecs": point_vecs_per_frame,
                "cam_rays": cam_rays,
            }
            save_file(save_data, attn_maps_paths)
            logger.info(f"Saved attention maps to {attn_maps_paths}", main_process_only=False)


        return {
            "prompt_embedding": prompt_embedding,
            "encoded_exo_ego_gt_video": encoded_exo_ego_gt_video,  # exo + ego_gt concatenated and encoded
            "encoded_exo_ego_prior_video": encoded_exo_ego_prior_video,  # exo + ego_prior concatenated and encoded
            "image_embedding": image_embedding,
            "attention_GGA": attn_maps,
            "attention_mask_GGA": attn_masks,
            "point_vecs_per_frame": point_vecs_per_frame,
            "cam_rays": cam_rays,
            "video_metadata": {
                "num_frames": encoded_exo_ego_gt_video.shape[1],
                "height": encoded_exo_ego_gt_video.shape[2],
                "width": encoded_exo_ego_gt_video.shape[3],
            },
        }

    def preprocess(self, video_path: Path | None, image_path: Path | None, video_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and preprocesses a video and an image.
        If either path is None, no preprocessing will be done for that input.

        Args:
            video_path: Path to the video file to load
            image_path: Path to the image file to load
            video_type: Type of video ("exo", "ego_gt", or "ego_prior")

        Returns:
            A tuple containing:
                - video(torch.Tensor) of shape [F, C, H, W] where F is number of frames,
                  C is number of channels, H is height and W is width
                - image(torch.Tensor) of shape [C, H, W]
        """
        raise NotImplementedError("Subclass must implement this method")

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to a video.

        Args:
            frames (torch.Tensor): A 4D tensor representing a video
                with shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed video tensor
        """
        raise NotImplementedError("Subclass must implement this method")

    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to an image.

        Args:
            image (torch.Tensor): A 3D tensor representing an image
                with shape [C, H, W] where:
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed image tensor
        """
        raise NotImplementedError("Subclass must implement this method")

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)

class WanI2VDatasetWithResize(BaseWanDataset):
    """
    A dataset class for image-to-video generation that resizes inputs to fixed dimensions.

    This class preprocesses videos and images by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width
    - Images are resized to height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos and images
        width (int): Target width for resizing videos and images
    """

    def __init__(self, max_num_frames: int, height: int, width: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        self.__frame_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(self, video_path: Path | None, image_path: Path | None, video_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if video_path is not None:
            video = preprocess_video_with_resize(video_path, self.max_num_frames, self.height, self.width, video_type)
        else:
            video = None
        if image_path is not None:
            image = preprocess_image_with_resize(image_path, self.height, self.width)
        else:
            image = None
        return video, image

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)