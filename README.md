Ensemble Learning for Enhanced Green Cover and Open Space Classification in Informal Settlements
1. Description
This repository contains the implementation of the research work entitled:

“Ensemble Learning for Enhanced Green Cover and Open Space Classification in Informal Settlements.”

The base implementation uses a diverse set of seven U-Net based architectures for semantic segmentation.

To improve accuracy and robustness, a stacking ensemble framework with a neural network meta-learner is used to intelligently fuse the predictions from the base models.

This ensemble framework has been tested on:

A custom high-resolution satellite imagery dataset of informal settlements from South America, Africa, and Asia.

The task is to classify pixels into four categories: (1) green cover, (2) open spaces, (3) built structures, and (4) other surfaces.

2. System Requirements
Operating System: Windows 10/11, Ubuntu 20.04+, or macOS 12+

Python: 3.9 – 3.11

GPU (Recommended): NVIDIA GPU with CUDA 11.6+ for training.

RAM: Minimum 8 GB (16 GB recommended for training).

Storage: At least 15 GB free (for datasets and model checkpoints).

3. Required Libraries
Install dependencies using:

pip install -r requirements.txt

Contents of requirements.txt:

tensorflow>=2.10.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
opencv-python>=4.6.0
Pillow>=9.3.0
albumentations>=1.3.0

4. Usage Instructions
Clone or download the project.

git clone [https://github.com/Chanduchawla/Ensemble-Learning-for-Green-Cover-Detection-in-Informal-Settlements](https://github.com/Chanduchawla/Ensemble-Learning-for-Green-Cover-Detection-in-Informal-Settlements)

Set up the dataset.
Organize your dataset into the following structure:

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

Train Base Models and Evaluate Ensembles.
The entire workflow is contained within the main Jupyter notebook. This includes training all seven base models, evaluating them, and then applying the four different ensemble strategies.

# Launch Jupyter and run the cells in the following notebook:
ensamble-learning-98295d.ipynb

The notebook will train:

Standard U-Net, Attention U-Net, U-Net++, ResNet-UNet, VGG16-UNet, VGG19-UNet, and a Modified U-Net.

Ensemble Inference.
The notebook will automatically perform inference using four different ensemble methods after the base models are trained:

Hard Voting (Majority Voting)

Soft Voting (Probability Averaging)

Weighted Averaging

Stacking with a Meta-Learner (Our Proposed Method)

Output.
The program outputs:

Saved model weights (.keras files) for each base model.

Performance metrics for all models and ensembles (Accuracy, F1-Score, IoU, Precision, Recall).

Training history plots for each base model.

Visualizations of the final predicted segmentation masks on test images.

5. Reference
If you use this code or research in your work, please cite:

Chanduchawla
Ensemble Learning for Enhanced Green Cover and Open Space Classification in Informal Settlements.

6. Contact
For queries, collaborations, or clarifications, please contact:
Chanduchawla
📧 Via GitHub Issues or Profile
