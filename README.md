# Ensemble Learning for Enhanced Green Cover and Open Space Classification in Informal Settlements

## 1. Description
This repository contains the implementation of the research work entitled:  
**“Ensemble Learning for Enhanced Green Cover and Open Space Classification in Informal Settlements.”**

- The base implementation uses a diverse set of seven **U-Net based architectures** for semantic segmentation.  
- To improve accuracy and robustness, a **stacking ensemble framework with a neural network meta-learner** is used to intelligently fuse the predictions from the base models.  

This ensemble framework has been tested on:  
- A custom high-resolution satellite imagery dataset of **informal settlements**.  
- The task is to classify pixels into four categories:  
  1. **Green Cover**  
  2. **Open Spaces**  
  3. **Built Structures**  
  4. **Other Surfaces**  

---

## 2. System Requirements
- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS 12+  
- **Python:** 3.9 – 3.11  
- **GPU (Recommended):** NVIDIA GPU with CUDA 11.6+ for training  
- **RAM:** Minimum 8 GB (16 GB recommended)  
- **Storage:** At least 15 GB free (for datasets and model checkpoints)  

---

## 3. Required Libraries
Install dependencies using:

```bash
pip install -r requirements.txt
````

Contents of `requirements.txt`:

```
tensorflow>=2.10.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
opencv-python>=4.6.0
Pillow>=9.3.0
albumentations>=1.3.0

```


---

## 4. Usage Instructions

1. **Clone the Repository.**

   ```bash
   git clone https://github.com/Chanduchawla/Ensemble-Learning-for-Green-Cover-Detection-in-Informal-Settlements
   cd Ensemble-Learning-for-Green-Cover-Detection-in-Informal-Settlements
   ```

2. **Set up the dataset.**
   Organize your dataset into the following structure:

   ```
   dataset/
     ├── train_images/
     │   └── train/
     ├── train_masks/
     │   └── train/
     ├── val_images/
     │   └── val/
     ├── val_masks/
     │   └── val/
     ├── test_images/
     │   └── test/
     └── test_masks/
         └── test/
   ```
Dataset:
```bash
https://www.kaggle.com/datasets/ayushdabra/sdsa-dse-406-606-demo-data
```
3. **Train Base Models & Evaluate Ensembles.**
   The entire workflow is contained in the Jupyter notebook:

   ```bash
   jupyter notebook chandu-conference-paper.ipynb
   ```

   This notebook trains the following models:

   * Standard U-Net
   * Attention U-Net
   * U-Net++
   * ResNet-UNet
   * VGG16-UNet
   * VGG19-UNet
   * Modified U-Net

---

## 5. Ensemble Inference

After training, the notebook applies **four ensemble methods**:

1. **Hard Voting (Majority Voting)**
2. **Soft Voting (Probability Averaging)**
3. **Weighted Averaging**
4. **Stacking with Meta-Learner (Proposed Method)**

---

## 6. Output

The framework produces:

* ✅ Saved model weights (`.keras` files) for each base model
* 📈 Performance metrics (Accuracy, F1-Score, IoU, Precision, Recall)
* 📉 Training history plots for each base model
* 🖼 Visualizations of predicted segmentation masks on test images

---

## 7. Reference

If you use this code or research in your work, please cite:

**Chevala Chandu**
*Ensemble Learning for Enhanced Green Cover and Open Space Classification in Informal Settlements.*

---

## 8. Contact

For queries, collaborations, or clarifications, please reach out:

👤 **Chevala Chandu**
📧 Email: [chanduchawla3820@gmail.com](mailto:chanduchawla3820@gmail.com)
📧 Via [GitHub Issues](https://github.com/Chanduchawla) or Profile

---



