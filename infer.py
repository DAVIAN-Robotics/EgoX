import argparse
import os
import random
from typing import Optional
import torch
import numpy as np
from core.inference.wan import generate_video

from core.finetune.datasets.utils import load_from_json_file, iproj_disp
from pathlib import Path

def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    _set_seed(args.seed)

    prompt_list = []
    with open(args.prompt, 'r') as f:
        for line in f.readlines():
            prompt_list.append(line.strip())

    exo_video_list = []
    with open(args.exo_video_path, 'r') as f:
        for line in f.readlines():
            exo_video_list.append(line.strip())

    ego_prior_video_list = []
    with open(args.ego_prior_video_path, 'r') as f:
        for line in f.readlines():
            ego_prior_video_list.append(line.strip())
    
    assert len(prompt_list) == len(exo_video_list) == len(ego_prior_video_list)

    meta_data_file = args.meta_data_file
    depth_root = args.depth_root
    meta_data = load_from_json_file(meta_data_file)

    prompts = prompt_list
    exo_videos = exo_video_list
    ego_prior_videos = ego_prior_video_list
    depth_map_paths = []
    camera_intrinsics = []
    camera_extrinsics = []
    ego_extrinsics = []
    ego_intrinsics = []
    take_names = []

    meta_data = meta_data['test_datasets']

    i = 0
    for ego_filename, meta in enumerate(meta_data):
        if i >= len(exo_videos):
            break
        take_name = exo_videos[i].split('/')[-2]
        i += 1
        depth_map_paths.append(Path(os.path.join(depth_root, take_name)))
        camera_extrinsics.append(meta['camera_extrinsics'])
        camera_intrinsics.append(meta['camera_intrinsics'])
        ego_extrinsics.append(meta['ego_extrinsics'])
        ego_intrinsics.append(meta['ego_intrinsics'])
        take_names.append(take_name)
    
    import shutil
    from transformers import CLIPVisionModel

    model_path=args.model_path
    transformer_path = os.path.join(model_path, 'transformer')
    lora_path=args.lora_path
    dtype=torch.bfloat16

    from core.finetune.models.wan_i2v.custom_transformer import WanTransformer3DModel_GGA as WanTransformer3DModel
    transformer = WanTransformer3DModel.from_pretrained(transformer_path, torch_dtype=dtype)

    image_encoder = CLIPVisionModel.from_pretrained(model_path, subfolder="image_encoder", torch_dtype=torch.float32)

    from core.finetune.models.wan_i2v.sft_trainer import WanWidthConcatImageToVideoPipeline
    pipe = WanWidthConcatImageToVideoPipeline.from_pretrained(model_path, image_encoder=image_encoder, transformer=transformer, torch_dtype=dtype)
    
    if lora_path:
        print('loading lora')
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora(components=["transformer"], lora_scale = 1.0 )

    pipe.to("cuda")

    os.makedirs(args.out, exist_ok=True)
    for i in range(len(prompts)):
        if i!=args.idx and args.idx!=-1:
            continue
        if args.start_idx != -1 and args.end_idx != 1:
            if i < args.start_idx or i >= args.end_idx :
                continue
        prompt, exo_video_path, ego_prior_video_path = prompts[i], exo_videos[i], ego_prior_videos[i]

        take_name=take_names[i]
        exo_video_path = str(exo_video_path)
        ego_prior_video_path = str(ego_prior_video_path)
        print(exo_video_path)
        print(ego_prior_video_path)

        if args.use_GGA:
            depth_map_path = depth_map_paths[i]
            camera_intrinsic = camera_intrinsics[i]
            camera_extrinsic = camera_extrinsics[i]
            ego_extrinsic = ego_extrinsics[i]
            ego_intrinsic = ego_intrinsics[i]

            device = 'cpu'

            C, F, H, W = 16, 13, 56, 154 #! Hard coding
            exo_H, exo_W = H, W - H
            W = H

            depth_maps = []
            for depth_map_file in sorted(depth_map_path.glob("*.npy")):
                depth_map = np.load(depth_map_file)
                depth_maps.append(torch.from_numpy(depth_map).unsqueeze(0))
            depth_maps = torch.cat(depth_maps, dim=0) # [F, H, W]

            ego_intrinsic=torch.tensor(ego_intrinsic)           #! (3,3)
            ego_extrinsic=torch.tensor(ego_extrinsic)           #! (F, 3, 4)
            camera_extrinsic=torch.tensor(camera_extrinsic)     #! (3,4)
            camera_intrinsic=torch.tensor(camera_intrinsic)     #! (3,3)

            if ego_extrinsic.shape[1] == 3 and ego_extrinsic.shape[2] == 4:
                ego_extrinsic = torch.cat([ego_extrinsic, torch.tensor([[[0, 0, 0, 1]]], dtype=ego_extrinsic.dtype).expand(ego_extrinsic.shape[0], -1, -1)], dim=1)
            if camera_extrinsic.shape == (3, 4):
                camera_extrinsic = torch.cat([torch.tensor(camera_extrinsic, dtype=ego_extrinsic.dtype), torch.tensor([[0, 0, 0, 1]], dtype=ego_extrinsic.dtype)], dim=0)

            scale = 1/8
            scaled_intrinsic = ego_intrinsic.clone()
            scaled_intrinsic[0, 0] *= scale  # fx' = fx * s
            scaled_intrinsic[1, 1] *= scale  # fy' = fy * s
            scaled_intrinsic[0, 2] *= scale  # cx' = cx * s
            scaled_intrinsic[1, 2] *= scale

            ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
            ones = torch.ones_like(xs)
            pixel_coords = torch.stack([xs, ys, ones], dim=-1).view(-1, 3).to(dtype=ego_intrinsic.dtype)  # (H*W, 3)

            pixel_coords_cv = pixel_coords[..., :2].cpu().numpy().reshape(-1, 1, 2).astype(np.float32)
            K = scaled_intrinsic.cpu().numpy().astype(np.float32)

            import cv2
            #! Ego cam distorion coeffs (Project Aria) - Hard coded
            distortion_coeffs = np.array([[-0.02340373583137989,0.09388021379709244,-0.06088035926222801,0.0053304750472307205,0.003342868760228157,-0.0006356257363222539,0.0005087381578050554,-0.0004747129278257489,-0.0011330085108056664,-0.00025734835071489215,0.00009328465239377692,0.00009424977179151028]])
            D = distortion_coeffs.astype(np.float32) # ex: [k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4]
            normalized_points = cv2.undistortPoints(pixel_coords_cv, K, D, R=np.eye(3), P=np.eye(3))


            normalized_points = torch.from_numpy(normalized_points).squeeze(1).to(device) # (N, 2)

            ones = torch.ones_like(normalized_points[..., :1])
            cam_rays_fish = torch.cat([normalized_points, ones], dim=-1) # (N, 3)

            cam_rays = cam_rays_fish / torch.norm(cam_rays_fish, dim=-1, keepdim=True)

            cam_rays = cam_rays @ ego_extrinsic[::4, :3, :3]

            cam_rays = cam_rays.view(F, H, W, 3)  # (B, H, W, 3)

            height, width = depth_maps.shape[1], depth_maps.shape[2]
            cx = width / 2.0
            cy = height / 2.0
            camera_intrinsic_scale_y = cy / camera_intrinsic[1,2]
            camera_intrinsic_scale_x = cx / camera_intrinsic[0,2]
            camera_intrinsic[0, 0] = camera_intrinsic[0, 0] * camera_intrinsic_scale_x
            camera_intrinsic[1, 1] = camera_intrinsic[1, 1] * camera_intrinsic_scale_y
            camera_intrinsic[0, 2] = cx
            camera_intrinsic[1, 2] = cy

            camera_intrinsic = np.array([camera_intrinsic[0, 0] , camera_intrinsic[1, 1], cx, cy])
            
            disp_v, disp_u = torch.meshgrid(
                torch.arange(depth_maps.shape[1], device=device).float(),
                torch.arange(depth_maps.shape[2],device=device).float(),
                indexing="ij",
            )

            disp = torch.ones_like(disp_v)

            disp_cpu = disp.cpu()
            disp_u_cpu = disp_u.cpu()
            disp_v_cpu = disp_v.cpu()
            
            pts, _, _ = iproj_disp(torch.from_numpy(camera_intrinsic), disp_cpu, disp_u_cpu, disp_v_cpu)

            if isinstance(pts, torch.Tensor):
                pts = pts.to(device)
            else:
                pts = torch.from_numpy(pts).to(device).float()
            
            rays = pts[..., :3] 

            rays = rays / rays[..., 2:3]    
            rays = rays.unsqueeze(0).expand(depth_maps.size(0), -1, -1, -1)  # (F, H, W, 3)
            camera_extrinsics_c2w = torch.linalg.inv(camera_extrinsic)

            pcd_camera = rays * depth_maps.unsqueeze(-1)
            point_map = pcd_camera.to(dtype=camera_extrinsics_c2w.dtype)
            point_map = torch.tensor(point_map) #! [F, H, W, 3]

            p_f, p_h, p_w, p_p = point_map.shape
            point_map_world = point_map.reshape(-1, 3)

            camera_extrinsics_c2w = torch.linalg.inv(camera_extrinsic)
            ones_point = torch.ones(point_map_world.shape[0], 1, device=point_map_world.device)
            point_map_world = torch.cat([point_map_world, ones_point], dim=-1) #! [F, H*W, 4]
            point_map_world = (camera_extrinsics_c2w @ point_map_world.T).T[...,:3]
            point_map = point_map_world.reshape(p_f, p_h, p_w, 3).permute(0, 3, 1, 2)

            point_map = point_map[:, :, (point_map.shape[2] - 448)//2:(point_map.shape[2] + 448)//2, (point_map.shape[3] - 784)//2:(point_map.shape[3] + 784)//2] #! [F, 3, 448, 784]
            point_map = torch.nn.functional.interpolate(point_map, size=(exo_H, exo_W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1) #! [B, H, W, 3]

            ego_extrinsic_c2w = torch.linalg.inv(ego_extrinsic)

            cam_origins = ego_extrinsic_c2w[::4, :3, 3].unsqueeze(1).expand(-1, exo_H * exo_W, -1)  # (B, H*W, 3)
            cam_origins = cam_origins.view(F, exo_H, exo_W, 3)  # (B, H, W, 3)

            if point_map.size(0) != ego_extrinsic_c2w.size(0):
                min_size = min(point_map.size(0), ego_extrinsic_c2w.size(0))
                point_map = point_map[:min_size]
            

            point_vecs_per_frame = []
            for j in range(cam_origins.size(0)):
                point_vec = point_map[::4] - cam_origins[j].unsqueeze(0)  #! [B, H, W, 3]
                point_vec = point_vec / torch.norm(point_vec, dim=-1, keepdim=True)  #! [B,H, W, 3]
                point_vecs_per_frame.append(point_vec)
            point_vecs_per_frame = torch.stack(point_vecs_per_frame, dim=0)  #! [B, B, H, W, 3]

            point_vecs = point_map[::4] - cam_origins #! [B, H, W, 3]
            point_vecs = point_vecs / torch.norm(point_vecs, dim=-1, keepdim=True) #! [B, H, W, 3]

            cam_rays = torch.rot90(cam_rays, k=-1, dims=[1, 2])

            attn_maps = torch.cat((point_vecs, cam_rays), dim = 2)
            attn_masks = torch.cat((torch.ones_like(point_vecs), torch.zeros_like(cam_rays)), dim = 2)
        else:
            point_vecs_per_frame = None
            attn_maps = None
            attn_masks = None
            cam_rays = None

        output_filename = f'{take_name}.mp4'  
        output_path = os.path.join(args.out, output_filename)
        
        if args.idx==-1 or i==args.idx:
            print('\n', i)
            video = generate_video(
                prompt=prompt,
                exo_video_path=exo_video_path,
                ego_prior_video_path=ego_prior_video_path,
                output_path=output_path,
                num_frames=49,
                width=784+448,
                height=448,
                num_inference_steps=50,
                guidance_scale=5.0,
                fps=30,
                num_videos_per_prompt=1,
                seed=args.seed,
                attention_GGA=attn_maps.unsqueeze(0) if attn_maps is not None else None,
                attention_mask_GGA=attn_masks.unsqueeze(0) if attn_masks is not None else None,
                point_vecs_per_frame = point_vecs_per_frame,
                cam_rays = cam_rays,
                do_kv_cache = args.do_kv_cache,
                cos_sim_scaling_factor=args.cos_sim_scaling_factor,
                pipe = pipe,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using Wan")
    parser.add_argument("--prompt", type=str, required=True, help="prompt path list.txt")
    parser.add_argument("--exo_video_path", type=str, required=True, help="exo video path list.txt")
    parser.add_argument("--ego_prior_video_path", type=str, required=True, help="ego prior video path list.txt")
    parser.add_argument("--idx", type=int, default=-1)
    parser.add_argument("--model_path", type=str, default=None, help="The path of the SFT weights to be used")
    parser.add_argument("--out", type=str, default="/outputs", help="The path save generated video")
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=64, help="The rank of the LoRA weights to be used")
    parser.add_argument("--meta_data_file", type=str, default=None, help="meta data for GGA")
    parser.add_argument("--use_GGA", action='store_true', help="whether to use GGA or not")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible inference")
    parser.add_argument("--do_kv_cache", action='store_true', help="whether to use key-value caching")
    parser.add_argument("--cos_sim_scaling_factor", type=float, default=1.0, help="scaling factor for cos similarity attention map")
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--depth_root", type=str, default="./example/in_the_wild/depth_maps/", help="root path for depth maps")
    parser.add_argument("--in_the_wild", action='store_true', help="whether to use in-the-wild inference")
    args = parser.parse_args()
    main(args)
