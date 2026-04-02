# [CVPR 2026] EgoX: Egocentric Video Generation from a Single Exocentric Video

[![Hugging Face Paper](https://img.shields.io/badge/HuggingFace-Paper%20of%20the%20Day%20%231-orange)](https://huggingface.co/papers/2512.08269)
[![arXiv](https://img.shields.io/badge/arXiv-2510.26236-b31b1b.svg)](https://arxiv.org/abs/2512.08269)
[![Project Page](https://img.shields.io/badge/Project_Page-Visit-blue.svg)](https://keh0t0.github.io/EgoX/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/DAVIAN-Robotics/EgoX)

> [Taewoong Kang\*](https://keh0t0.github.io/), [Kinam Kim\*](https://kinam0252.github.io/), [Dohyeon Kim\*](https://linkedin.com/in/dohyeon-kim-a79231347), [Minho Park](https://pmh9960.github.io/), [Junha Hyung](https://junhahyung.github.io/), and [Jaegul Choo](https://sites.google.com/site/jaegulchoo/)
> 
> **DAVIAN Robotics, KAIST AI, SNU**  
> arXiv 2025. (\* indicates equal contribution)

## 🎬 Teaser Video


https://github.com/user-attachments/assets/5f599ad0-0922-414b-a8ab-e789da068efa

## About EgoX

**EgoX** is a novel egocentric video generation framework that produces first-person (ego-view) videos from a single third-person (exo-view) video input. By leveraging both exocentric observations and egocentric priors, EgoX enables realistic viewpoint transformation while preserving temporal consistency and scene structure. The method introduces a unified conditioning strategy that integrates spatial and channel-wise information within clean latent representations, requiring only lightweight LoRA-based adaptation. EgoX is built upon large-scale video diffusion models and is trained on the Ego-Exo4D dataset, making it a powerful tool for egocentric video synthesis and related research applications.


## 🛠️ Environment Setup

### System Requirements

- **GPU**: ≥ 80GB VRAM (for inference), ≥ 140GB VRAM (for training)
- **CUDA**: 12.1 or higher
- **Python**: 3.10
- **PyTorch**: Version compatible with CUDA 12.1

### Installation

Create a conda environment and install dependencies:

```bash
# Create conda environment
conda create -n egox python=3.10 -y
conda activate egox

# Install PyTorch with CUDA 12.1
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

## 📥 Model Weights Download

### 💾 Wan2.1-I2V-14B Pretrained Model

Download the [Wan2.1-I2V-14B](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers) model and save it to the `checkpoints/pretrained_model/` folder.

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers')"
```

### 💾 EgoX Model Weights Download

Download the trained EgoX LoRA weights using one of the following methods:

**Option 1: Hugging Face**
```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='DAVIAN-Robotics/EgoX', local_dir='./checkpoints/EgoX', allow_patterns='*.safetensors')"
```

**Option 2: Google Drive**
- Download from [Google Drive](https://drive.google.com/file/d/1Q7j7LVI4YiSkwzNMBBiyLS1rT3HMcNVB/view?usp=drive_link) and save to the `checkpoints/EgoX/` folder.


## 🚀 Inference

### Quick Start with Example Data

For quick testing, the codebase includes example data in the `example/` directory. You can run inference immediately:

```bash
# For in-the-wild example
bash scripts/infer_itw.sh

# For Ego4D example
bash scripts/infer_ego4d.sh
```

Edit the GPU ID and seed in the script if needed. Results will be saved to `./results/`.

### Custom Data Inference

To run inference with your own data, prepare the following file structure:

```
your_dataset/              # Your custom dataset folder
├── meta.json              # Meta information for each video
├── videos/                # Videos directory
│   └── take_name/
│       ├── ego_Prior.mp4
│       ├── exo.mp4
│       └── ...
└── depth_maps/            # Depth maps directory
    └── take_name/
        ├── frame_000.npy
        └── ...
```


<details>
<summary><b>meta.json</b> - Meta information for each video</summary>

JSON file containing exocentric video path, egocentric prior video path, prompt, camera intrinsic and extrinsic parameters for each video. The structure includes `test_datasets` array with entries for each videos.

**Example:**
```json
{
    "test_datasets": [
        {
            "exo_path": "./example/in_the_wild/videos/joker/exo.mp4",
            "ego_prior_path": "./example/in_the_wild/videos/joker/ego_Prior.mp4",
            "prompt": "[Exo view]\n**Scene Overview:**\nThe scene is set on a str...\n\n[Ego view]\n**Scene Overview:**\nFrom the inferred first-person perspective, the environment appears chaotic and filled with sm...",
            "camera_intrinsics": [
                [634.47327, 0.0, 392.0],
                [0.0, 634.4733, 224.0],
                [0.0, 0.0, 1.0]
            ],
            "camera_extrinsics": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0]
            ],
            "ego_intrinsics": [
                [150.0, 0.0, 255.5],
                [0.0, 150.0, 255.5],
                [0.0, 0.0, 1.0]
            ],
            "ego_extrinsics": [
                [[0.6263, 0.7788, -0.0336, 0.3432],
                 [-0.0557, 0.0018, -0.9984, 2.3936],
                 [-0.7776, 0.6272, 0.0445, 0.1299]],
                ...
            ]
        },
        ...
    ]
}
```

</details>

To prepare your own dataset, follow the instruction from [here](https://github.com/kdh8156/EgoX-EgoPriorRenderer/tree/main).

### Constraints
Since EgoX is trained on the Ego-Exo4D dataset where exocentric view camera poses are fixed, you must provide exocentric videos with fixed camera poses as input during inference.
Also, the model is trained on 448x448(ego), 448x784(exo) resolutions and 49 frames. Please preprocess your videos to these resolutions.

<details>
<summary><b>Custom dataset init structure</b></summary>

Before running the script, you need to create a custom dataset folder with the following structure:

```
your_dataset/              # Your custom dataset folder
├── videos/                # Videos directory
    └── take_name/
        └──  exo.mp4
```

Then, by using `meta_init.py`, you can create a meta.json file with the following command:

```
python meta_init.py --folder_path ./your_dataset --output_json ./your_dataset/meta.json --overwrite
```

```
your_dataset/              # Your custom dataset folder
├── meta.json              # Meta information for each video
├── videos/                # Videos directory
    └── take_name/
        └──  exo.mp4
```

Then, you can use `caption.py` to generate caption for each video with this command:

```
python caption.py --json_file ./your_dataset/meta.json --output_json ./your_dataset/meta.json --overwrite
```

Make sure that your api key is properly set in `caption.py`.

Finally, follow the instruction from [here](https://github.com/kdh8156/EgoX-EgoPriorRenderer/tree/main).
Then you can get depth maps, camera intrinsic, ego camera extrinsics for each video.

```
your_dataset/              # Your custom dataset folder
├── meta.json              # Meta information for each video
├── videos/                # Videos directory
    └── take_name/
        ├── ego_Prior.mp4
        ├── exo.mp4
        └── ...
└── depth_maps/            # Depth maps directory
    └── take_name/
        ├── frame_000.npy
        └── ...
```

</details>

Then, modify `scripts/infer_itw.sh` (or create a new script) to point to your data paths:

```bash
python3 infer.py \
    --meta_data_file ./example/your_dataset/meta.json \
    --model_path ./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers \
    --lora_path ./checkpoints/EgoX/pytorch_lora_weights.safetensors \
    --lora_rank 256 \
    --out ./results \
    --seed 42 \
    --use_GGA \
    --cos_sim_scaling_factor 3.0 \
    --in_the_wild
```

## 🏋️ Training

### Data Preparation

Prepare your training dataset with the following structure:

```
dataset/train/
├── meta_with_uid.json         # Metadata for training videos
├── videos/                    # Videos directory
│   └── take_name/
│       ├── exo.mp4            # Exocentric video
│       ├── ego.mp4            # Egocentric ground truth video
│       └── ego_Prior.mp4      # Egocentric prior video
└── depth_maps/                # Depth maps directory
    └── take_name/
        ├── frame_000.npy
        └── ...
```

The `meta_with_uid.json` follows the same format as `meta.json` used in inference, with additional `ego_path` field pointing to the egocentric ground truth video.

To generate metadata and captions for your dataset:
```bash
# Initialize metadata from folder structure
python meta_init.py --folder_path ./dataset/train --output_json ./dataset/train/meta_with_uid.json --overwrite

# Generate captions using GPT-4o (requires OpenAI API key configured in caption.py)
python caption.py --json_file ./dataset/train/meta_with_uid.json --output_json ./dataset/train/meta_with_uid.json --overwrite
```

For depth maps and camera parameters, follow the instructions from [EgoX-EgoPriorRenderer](https://github.com/kdh8156/EgoX-EgoPriorRenderer/tree/main).

### Quick Start

Run training with the default configuration (4 GPUs):

```bash
bash scripts/finetune.sh
```

This uses Hugging Face `accelerate` for distributed training. The script trains a LoRA adapter on top of the Wan2.1-I2V-14B pretrained model.

### Custom Training Configuration

You can modify `scripts/finetune.sh` or create a new script. Below is an example command with all key arguments.

<details>
<summary><b>Custom Training Configuration</b></summary>

```bash
export TOKENIZERS_PARALLELISM=false

accelerate launch \
    --config_file configs_acc/4gpu.yaml \
    --main_process_ip localhost \
    --main_process_port 29501 \
    --machine_rank 0 \
    --num_processes 4 \
    --num_machines 1 \
    finetune.py \
    --model_path ./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers \
    --model_name wan-i2v \
    --model_type wan-i2v \
    --training_type lora \
    --rank 256 \
    --lora_alpha 256 \
    --output_dir ./results/EgoX \
    --report_to tensorboard \
    --data_root ./dataset/train \
    --meta_data_file ./dataset/train/meta_with_uid.json \
    --train_resolution 49x448x1232 \
    --train_epochs 150 \
    --seed 42 \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --mixed_precision bf16 \
    --num_workers 16 \
    --pin_memory True \
    --nccl_timeout 1800 \
    --checkpointing_steps 250 \
    --checkpointing_limit 54 \
    --gen_fps 30 \
    --cos_sim_scaling_factor 1.0
```

To change the number of GPUs, use the corresponding accelerate config in `configs_acc/` (e.g., `1gpu.yaml`, `2gpu.yaml`, `4gpu.yaml`, `8gpu.yaml`) and update `--num_processes` accordingly.

</details>

### Resume Training

To resume training from a checkpoint, uncomment and modify the `--resume_from_checkpoint` argument in the script:

```bash
    --resume_from_checkpoint ./results/EgoX/checkpoint-10000
```

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=DAVIAN-Robotics/EgoX&type=date&legend=top-left)](https://www.star-history.com/#DAVIAN-Robotics/EgoX&type=date&legend=top-left)

## 🙏 Acknowledgements

This project is built upon the following works:

- [4DNeX](https://github.com/3DTopia/4DNeX)
- [Ego-Exo4D](https://github.com/facebookresearch/Ego-Exo)

## 📝 Citation

If you use this dataset or code in your research, please cite our paper:

```bibtex
@misc{kang2025egoxegocentricvideogeneration,
      title={EgoX: Egocentric Video Generation from a Single Exocentric Video}, 
      author={Taewoong Kang and Kinam Kim and Dohyeon Kim and Minho Park and Junha Hyung and Jaegul Choo},
      year={2025},
      eprint={2512.08269},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.08269}, 
}
```
