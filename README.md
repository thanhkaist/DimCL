# DimCL: Dimensional Contrastive Learning for Improving Self-Supervised Learning

Official PyTorch implementation of  
**[DimCL: Dimensional Contrastive Learning for Improving Self-Supervised Learning](https://arxiv.org/abs/2309.11782)**  
IEEE Access, 2023

> 📚 Built upon [solo-learn](https://github.com/vturrisi/solo-learn): A library of self-supervised learning methods powered by PyTorch Lightning.

---

## 🧠 Overview

**DimCL** introduces a novel regularization strategy that applies contrastive learning **across embedding dimensions**, rather than across instances. This dimensional contrastive loss improves feature decorrelation and representation diversity, enhancing performance across self-supervised learning (SSL) methods.

This repository provides:
- A modular and extensible PyTorch implementation.
- Integration with multiple popular SSL methods and backbone architectures.
- Evaluation tools and logging using PyTorch Lightning.

---


## ✅ Supported Methods

| Method | Paper Link |
|--------|------------|
| DimCL (Ours) | [arXiv](https://arxiv.org/abs/2309.11782) |
| Barlow Twins | [arXiv](https://arxiv.org/abs/2103.03230) |
| BYOL | [arXiv](https://arxiv.org/abs/2006.07733) |
| DeepCluster V2 | [arXiv](https://arxiv.org/abs/2006.09882) |
| DINO | [arXiv](https://arxiv.org/abs/2104.14294) |
| MoCo V2+ | [arXiv](https://arxiv.org/abs/2003.04297) |
| NNBYOL / NNCLR / NNSiam | [arXiv](https://arxiv.org/abs/2104.14548) |
| ReSSL | [arXiv](https://arxiv.org/abs/2107.09282) |
| SimCLR | [arXiv](https://arxiv.org/abs/2002.05709) |
| SimSiam | [arXiv](https://arxiv.org/abs/2011.10566) |
| SupCon | [arXiv](https://arxiv.org/abs/2004.11362) |
| SwAV | [arXiv](https://arxiv.org/abs/2006.09882) |
| VICReg / VIbCReg | [VICReg](https://arxiv.org/abs/2105.04906), [VIbCReg](https://arxiv.org/abs/2109.00783) |
| W-MSE | [arXiv](https://arxiv.org/abs/2007.06346) |

---

## 🧩 Supported Backbones

- [ResNet](https://arxiv.org/abs/1512.03385)
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [PoolFormer](https://arxiv.org/abs/2111.11418)

---

## 📦 Key Features
### 🔁 Data & Augmentation
- Fast data loading with [NVIDIA DALI](https://github.com/NVIDIA/DALI) (up to 2× faster).
- Configurable and flexible data augmentations.

### 🧪 Evaluation & Logging
- Online/offline linear and K-NN evaluation.
- Feature visualization with UMAP (online & offline).
- Built-in support for PyTorch Lightning features:
  - Mixed precision
  - Gradient accumulation & clipping
  - Automatic logging
- Lightweight modular code for easy prototyping.

### 🛠 Training Utilities
- Multi-crop support (e.g., SwAV-style, currently SimCLR only).
- LARS optimizer improvements (e.g., excluding BatchNorm/bias).
- Optional LR scheduling tweaks for SimSiam.

---

## 🧰 Requirements

```bash
pip install torch torchvision pytorch-lightning lightning-bolts wandb einops tqdm torchmetrics timm scipy
```
Optional:

```bash
pip install nvidia-dali matplotlib seaborn pandas umap-learn
```
## 🚀 Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/dimcl.git
cd dimcl

# Full installation with DALI and UMAP
pip install .[dali,umap]

# Or basic installation
pip install .
```
💡 Trouble with DALI? Try:

```bash
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
```
Replace cuda110 with your specific CUDA version.

## 🏋️‍♂️ Training
Pretrain the backbone using one of the scripts in:

```bash
bash_files/pretrain/
```
Then run offline linear evaluation:

```bash
bash_files/linear/
```
To enable DimCL, set the following flags in your bash config:

```bash
--our_loss True \        # Enable DimCL
--lam 0.1 \              # DimCL weight
--tau_decor 0.1 \        # Hardness-aware contrast ratio
```
> 📝 Most bash files follow the recommended hyperparameters from the original papers—check and tune as needed.

## Citation
If you use this, please cite [DimCL](https://arxiv.org/abs/2103.03230):
```
@article{nguyen2023dimcl,
  title={DimCL: Dimensional Contrastive Learning for Improving Self-Supervised Learning},
  author={Nguyen, Thanh and Pham, Trung Xuan and Zhang, Chaoning and Luu, Tung M and Vu, Thang and Yoo, Chang D},
  journal={IEEE Access},
  volume={11},
  pages={21534--21545},
  year={2023},
  publisher={IEEE}
}
```

And solo-learn  [preprint](https://arxiv.org/abs/2108.01775v1):
```
@misc{turrisi2021sololearn,
      title={Solo-learn: A Library of Self-supervised Methods for Visual Representation Learning}, 
      author={Victor G. Turrisi da Costa and Enrico Fini and Moin Nabi and Nicu Sebe and Elisa Ricci},
      year={2021},
      eprint={2108.01775},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={\url{https://github.com/vturrisi/solo-learn}},
}
```
