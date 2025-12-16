#!/bin/bash
GPU_IDS=6
SEED=846514

export CUDA_VISIBLE_DEVICES=$GPU_IDS

echo "Using GPUs: $GPU_IDS"

#! In-the-wild inference
python3 infer.py \
    --prompt ./example/in_the_wild/caption.txt \
    --exo_video_path ./example/in_the_wild/exo_gt_path.txt \
    --ego_prior_video_path ./example/in_the_wild/ego_prior_path.txt \
    --meta_data_file ./example/in_the_wild/camera_params.json \
    --depth_root ./example/in_the_wild/depth_maps/ \
    --sft_path ./Wan2.1-I2V-14B-480P-Diffusers/transformer \
    --lora_path ./results/pytorch_lora_weights.safetensors \
    --lora_rank 256 \
    --out ./results \
    --seed $SEED \
    --use_GGA \
    --cos_sim_scaling_factor 3.0 \

#! Ego4D inference
python3 infer.py \
    --prompt ./example/ego4D/caption.txt \
    --exo_video_path ./example/ego4D/exo_gt_path.txt \
    --ego_prior_video_path ./example/ego4D/ego_prior_path.txt \
    --meta_data_file ./example/ego4D/camera_params.json \
    --depth_root ./example/ego4D/depth_maps/ \
    --sft_path ./Wan2.1-I2V-14B-480P-Diffusers/transformer \
    --lora_path ./results/pytorch_lora_weights.safetensors \
    --lora_rank 256 \
    --out ./results \
    --seed $SEED \
    --use_GGA \
    --cos_sim_scaling_factor 3.0 \
