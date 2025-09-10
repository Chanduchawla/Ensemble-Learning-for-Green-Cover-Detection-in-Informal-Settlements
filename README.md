# Ensemble Learning for Enhanced Green Cover and Open Space Classification in Informal Settlements

## 1. Description
This repository contains the implementation of the research work entitled:  
**“Ensemble Learning for Enhanced Green Cover and Open Space Classification in Informal Settlements.”**

- The base implementation uses a diverse set of seven **U-Net based architectures** for semantic segmentation.
- To improve accuracy and robustness, a **stacking ensemble framework with a neural network meta-learner** is used to intelligently fuse the predictions from the base models.

This ensemble framework has been tested on:
- A custom high-resolution satellite imagery dataset of **informal settlements**.
- The task is to classify pixels into four categories: (1) green cover, (2) open spaces, (3) built structures, and (4) other surfaces.

---

## 2. System Requirements
- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS 12+
- **Python:** 3.9 – 3.11
- **GPU (Recommended):** NVIDIA GPU with CUDA 11.6+ for training.
- **RAM:** Minimum 8 GB (16 GB recommended for training).
- **Storage:** At least 15 GB free (for datasets and model checkpoints).

---

## 3. Required Libraries
Install dependencies using:

```bash
pip install -r requirements.txt
