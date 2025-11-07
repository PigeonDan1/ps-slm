# TASU: Text-only Alignment for Speech Understanding

TASU (Text-only Alignment for Speech Understanding) is a newly proposed training paradigm for **Speech Large Language Models (Speech LLMs)**, with a primary focus on **semantic speech understanding**.  
This repository contains the **core implementation** of the TASU algorithm.

---

## 🔍 Overview

- **Goal**: Enhance speech semantic *understanding*.
- **Paradigm**: Text-only alignment–oriented training strategy for Speech LLMs.
- **This repo**: Implements the key components required to train and evaluate TASU-based Speech LLMs.

---

## ⚙️ Environment Setup

TASU is mainly developed and tested on **Huawei Ascend (910B) NPU clusters**, but can also be adapted to **NVIDIA GPU** environments.

### 1. Running on Huawei Ascend 910B (NPU)

We primarily run TASU on Huawei Ascend 910B clusters.  You can use the dockerfile like:
```bash
# Build image from Dockerfile in the current directory
docker build -t tasu:latest .

To make it easy to reproduce our experiments, we will also provide a **pre-built container image tarball** (TBD).

1. Obtain the provided image tarball (the `.tar` file linked in your environment).
2. Load it into your container runtime, for example:
   ```bash
   docker load -i <tasu_ascend_image.tar>
   
## 2. Running on NVIDIA GPUs

For NVIDIA GPU users, you can directly follow the environment setup of the **SLAM-LLM** project:

- SLAM-LLM repo:  
  https://github.com/X-LANCE/SLAM-LLM

In most cases, you can reuse the same environment and:

- Install packages as described in the SLAM-LLM README.
- When adapting scripts or configs from the NPU version, replace a small number of `.npu` usages with `.gpu` (e.g., device tags or backend names).

> Full GPU-specific support is under active development, and we plan to provide more detailed GPU setup and scripts soon.

---

## 🚀 Getting Started

Once your environment is ready (either Ascend NPU or NVIDIA GPU):

1. Clone this repository:
   ```bash
   git clone https://github.com/PigeonDan1/ps-slm.git
   cd ps-slm
   cd Multitask

2. Follow the Readme in the dir.

---

## 📖 Citation

If you find **TASU** or this codebase useful in your research, please consider citing:

```bibtex
@article{peng2025tasu,
  title   = {TASU: Text-Only Alignment for Speech Understanding},
  author  = {Peng, Jing and Yang, Yi and Li, Xu and Xi, Yu and Tang, Quanwei and Fang, Yangui and Li, Junjie and Yu, Kai},
  journal = {arXiv preprint arXiv:2511.03310},
  year    = {2025},
}

