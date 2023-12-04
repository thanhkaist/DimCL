# DimCL 

This source code is for the paper: DimCL: Dimensional Contrastive Learning For Improving Self-Supervised Learning

Link: [DimCL](https://arxiv.org/abs/2103.03230)

This source code is built upon [solo-learn](https://github.com/vturrisi/solo-learn) library which is a library of self-supervised methods for unsupervised visual representation learning powered by PyTorch Lightning.



## Methods available:
| [DimCL](https://arxiv.org/abs/2103.03230) | [Barlow Twins](https://arxiv.org/abs/2103.03230) |[BYOL](https://arxiv.org/abs/2006.07733) |  [DeepCluster V2](https://arxiv.org/abs/2006.09882) |  [DINO](https://arxiv.org/abs/2104.14294) |
|--------|-------|---------------|---------------|--------------|
| [MoCo V2+](https://arxiv.org/abs/2003.04297) | [NNBYOL](https://arxiv.org/abs/2104.14548) | [NNCLR](https://arxiv.org/abs/2104.14548) | [NNSiam](https://arxiv.org/abs/2104.14548) | [ReSSL](https://arxiv.org/abs/2107.09282) |
| [SimCLR](https://arxiv.org/abs/2002.05709) | [SimSiam](https://arxiv.org/abs/2011.10566) | [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) | [SwAV](https://arxiv.org/abs/2006.09882) | [VIbCReg](https://arxiv.org/abs/2109.00783) |
| [VICReg](https://arxiv.org/abs/2105.04906) | [W-MSE](https://arxiv.org/abs/2007.06346) | | | |

---

## Extra flavor

# Multiple backbones
* [ResNet](https://arxiv.org/abs/1512.03385)
* [ViT](https://arxiv.org/abs/2010.11929)
* [Swin](https://arxiv.org/abs/2103.14030)
* [PoolFormer](https://arxiv.org/abs/2111.11418)

### Data
* Increased data processing speed by up to 100% using [Nvidia Dali](https://github.com/NVIDIA/DALI).
* Flexible augmentations.

### Evaluation and logging
* Online linear evaluation via stop-gradient for easier debugging and prototyping (optionally available for the momentum backbone as well).
* Online and offlfine K-NN evaluation.
* Normal offline linear evaluation.
* All the perks of PyTorch Lightning (mixed precision, gradient accumulation, clipping, automatic logging and much more).
* Easy-to-extend modular code structure.
* Custom model logging with a simpler file organization.
* Automatic feature space visualization with UMAP.
* Offline UMAP.
* Common metrics and more to come...

### Training tricks
* Multi-cropping dataloading following [SwAV](https://arxiv.org/abs/2006.09882):
    * **Note**: currently, only SimCLR supports this.
* Exclude batchnorm and biases from LARS.
* No LR scheduler for the projection head in SimSiam.
---
## Requirements

* torch
* torchvision
* tqdm
* einops
* wandb
* pytorch-lightning
* lightning-bolts
* torchmetrics
* scipy
* timm

**Optional**:
* nvidia-dali
* matplotlib
* seaborn
* pandas
* umap-learn

---

## Installation

First clone the repo.

Then, to install DCL with Dali and/or UMAP support, use:
```
pip3 install .[dali,umap]
```

If no Dali/UMAP support is needed, the repository can be installed as:
```
pip3 install .
```

**NOTE:** if you are having trouble with dali, install it with `pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110` or with your specific cuda version.

**NOTE 2:** If you want to modify the library, install it in dev mode with `-e`.

**NOTE 3:** Soon to be on pip.

---

## Training

For pretraining the backbone, follow one of the many bash files in `bash_files/pretrain/`.

After that, for offline linear evaluation, follow the examples on `bash_files/linear`.

**NOTE:** Files try to be up-to-date and follow as closely as possible the recommended parameters of each paper, but check them before running.

DimCL is a regularizer that can be enabled to provide free performance gain.  In each bash file, adjust DimCL hyper-parameters to incorporate/fine-grained tune.

```
    --lam 0.1 \ # temperature weight
    --tau_decor 0.1 \ # hardness aware ratio
    --our_loss False \ # enable/disable DimCl
```



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
