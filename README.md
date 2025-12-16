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

```bash
conda create -n egox python=3.10 -y
conda activate egox
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 📥 Pretrained Model Download

Download the [Wan2.1-I2V-14B](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers) model and save it to the `pretrained_model/` folder.

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='./pretrained_model/Wan2.1-I2V-14B-480P-Diffusers')"
```

```

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
