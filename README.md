# EgoX: Egocentric Video Generation from a Single Exocentric Video

[![Hugging Face Paper](https://img.shields.io/badge/HuggingFace-Paper%20of%20the%20Day%20%231-orange)](https://huggingface.co/papers/2512.08269)
[![arXiv](https://img.shields.io/badge/arXiv-2510.26236-b31b1b.svg)](https://arxiv.org/abs/2512.08269)
[![Project Page](https://img.shields.io/badge/Project_Page-Visit-blue.svg)](https://keh0t0.github.io/EgoX/)

> [Taewoong Kang\*](https://keh0t0.github.io/), [Kinam Kim\*](https://kinam0252.github.io/), [Dohyeon Kim\*](https://linkedin.com/in/dohyeon-kim-a79231347), [Minho Park](https://pmh9960.github.io/), [Junha Hyung](https://junhahyung.github.io/), and [Jaegul Choo](https://sites.google.com/site/jaegulchoo/)
> 
> **DAVIAN Robotics, KAIST AI**  
> arXiv 2025. (\* indicates equal contribution)

## 🎬 Teaser Video


https://github.com/user-attachments/assets/5f599ad0-0922-414b-a8ab-e789da068efa


## 📋 TODO

### 🔹 This Week
- [ ] Release **inference code**
- [ ] Release **model weights**

---

### 🔹 By End of December
- [ ] Release **training code**
- [ ] Release **data preprocessing code**
- [ ] Release **user-friendly interface**

## 🛠️ Environment Setup

### System Requirements

- **GPU**: TBD
- **CUDA**: 12.1 or higher
- **Python**: 3.10
- **PyTorch**: Compatible with CUDA 12.1

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

### Wan2.1-I2V-14B Pretrained Model

Download the [Wan2.1-I2V-14B](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers) model and save it to the `checkpoints/pretrained_model/` folder.

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers')"
```

### 📥 EgoX Model Weights Download

Download the trained EgoX LoRA weights from [Google Drive](https://drive.google.com/file/d/1Q7j7LVI4YiSkwzNMBBiyLS1rT3HMcNVB/view?usp=drive_link) and save them to the `checkpoints/EgoX/` folder.

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
├── caption.txt            # Text prompts (one per line)
├── exo_gt_path.txt        # Exocentric video paths (one per line)
├── ego_prior_path.txt     # Egocentric prior video paths (one per line)
├── camera_params.json     # Camera parameters
└── depth_maps/            # Depth maps directory
    └── take_name/
        ├── frame_000.npy
        └── ...
```

Then, modify `scripts/infer.sh` (or create a new script) to point to your data paths:

```bash
python3 infer.py \
    --prompt ./example/your_dataset/caption.txt \
    --exo_video_path ./example/your_dataset/exo_gt_path.txt \
    --ego_prior_video_path ./example/your_dataset/ego_prior_path.txt \
    --meta_data_file ./example/your_dataset/camera_params.json \
    --depth_root ./example/your_dataset/depth_maps/ \
    --sft_path ./Wan2.1-I2V-14B-480P-Diffusers/transformer \
    --lora_path ./results/pytorch_lora_weights.safetensors \
    --lora_rank 256 \
    --out ./results \
    --seed 846514 \
    --use_GGA \
    --cos_sim_scaling_factor 3.0
```

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
