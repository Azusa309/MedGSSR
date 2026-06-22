# MedGSSR: Generalizable Medical Image Super-Resolution 3D Reconstruction via Hierarchical Feed-forward Gaussian Splatting

**ECCV 2026**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Abstract

High-resolution volumetric medical imaging is critical for clinical diagnosis, yet acquisition is often limited by scanner hardware, scan time, and for CT, radiation dose. Medical 3D Super-Resolution (Med3DSR) offers a computational alternative, but existing methods commonly rely on per-subject optimization, pretrained priors, or coordinate-based implicit representations, which compromise anatomical fidelity and limit efficiency. To address these limitations, we present **MedGSSR**, a fully end-to-end feed-forward framework that represents volumes as an explicit 3D Gaussian field for Med3DSR. Unlike coordinate-based implicit functions, our explicit 3D Gaussian representation naturally enhances signal continuity and local high-frequency fidelity. Specifically, MedGSSR explicitly decouples the reconstruction process into coarse-grained structural preservation and fine-grained textural refinement through the proposed Pyramid Anatomical Encoder and a Hierarchical Gaussian Projector. To support arbitrary-scale super-resolution, we introduce sub-voxel Gaussian decomposition and a Differentiable Gaussian Voxelizer that directly queries the continuous 3D intensity field, reducing discretization artifacts. Extensive experiments on MRI and CT benchmarks demonstrate that MedGSSR significantly outperforms state-of-the-art methods. Notably, our framework exhibits robust generalizability across unseen datasets without requiring per-subject optimization, enabling fast inference and high-fidelity volumetric super-resolution in practical clinical settings.

## 📋 TODO

- [ ] Release pre-trained model weights
- [ ] Release training script


## 📦 Installation

```bash
git clone https://github.com/Azusa309/MedGSSR.git
cd MedGSSR
pip install -r requirements.txt
```



## 🔧 Inference

```bash
python inference.py
```


## 📝 Citation

```bibtex

```

## 📄 License

MIT License. 
