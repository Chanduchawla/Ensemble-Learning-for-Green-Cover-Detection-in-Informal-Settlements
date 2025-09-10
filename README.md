Ensemble Deep Learning for Green Cover Detection in Informal Settlements
<div align="center">

</div>

This repository contains the official implementation for the paper: "Ensemble Deep Learning for Enhanced Green Cover and Open Space Classification in Informal Settlements." We propose a robust ensemble framework that intelligently combines multiple U-Net based architectures to achieve state-of-the-art semantic segmentation of satellite imagery.

🌟 Overview
Accurately mapping green cover and open spaces in informal settlements is crucial for sustainable urban planning, environmental monitoring, and improving quality of life. However, the heterogeneous and complex nature of these areas poses significant challenges for standard computer vision models.

This project tackles this challenge by leveraging the power of ensemble learning. Instead of relying on a single model, we train a diverse "team" of seven specialized U-Net variants and fuse their predictions using a sophisticated stacking meta-learner. This approach mitigates the weaknesses of individual models and produces a final segmentation map that is significantly more accurate and robust.

🏗️ Proposed Framework
Our methodology is built on a two-level learning architecture. Level-0 consists of seven parallel base models for initial prediction. Level-1 employs an ensemble learning framework, featuring a trainable meta-learner, to produce the final, high-accuracy classification.

<p align="center">
<img src="assets/framework_diagram.jpg" alt="Ensemble Framework Diagram" width="800"/>
</p>

🚀 Key Features
Diverse Model Portfolio: Implements 7 different U-Net based architectures, including standard U-Net, Attention U-Net, U-Net++, and variants with VGG/ResNet backbones.

Advanced Ensemble Strategy: Employs a trainable neural network (meta-learner) to intelligently combine predictions, outperforming simpler methods like voting or averaging.

State-of-the-Art Performance: Achieves 94.2% accuracy on the task, a 3-7% improvement over individual models.

High-Impact Application: Provides a powerful tool for urban planners and environmental scientists to accurately map green infrastructure in challenging environments.

Comprehensive Evaluation: Systematically compares four different ensemble strategies: Hard Voting, Soft Voting, Weighted Averaging, and Stacking.

📊 Results Showcase
Our ensemble approach consistently outperforms all individual models and simpler fusion techniques across all standard evaluation metrics.

<p align="center">
<img src="assets/results_f1_iou.png" alt="F1-Score and IoU Results" width="700"/>
<em><br><b>Figure 1:</b> Comparison of core segmentation metrics. Our ensemble (highlighted) shows superior performance in F1-Score and IoU.</em>
</p>
<br>
<p align="center">
<img src="assets/results_acc_prec_rec.png" alt="Accuracy, Precision, and Recall Results" width="700"/>
<em><br><b>Figure 2:</b> Comparison of classification metrics. The ensemble achieves the highest scores in Accuracy, Precision, and Recall.</em>
</p>

🛠️ Tech Stack & Models
This project is built using Python and the TensorFlow/Keras deep learning framework.

Frameworks: TensorFlow, Keras

Libraries: NumPy, Pandas, Scikit-learn, Matplotlib, OpenCV

Base Models:

Standard U-Net

Attention U-Net

U-Net++

ResNet-UNet

VGG16-UNet

VGG19-UNet

Modified U-Net (with SE Blocks & ASPP)

⚙️ Setup and Usage
To replicate the experiments, follow these steps:

1. Clone the Repository

git clone [https://github.com/Chanduchawla/Ensemble-Learning-for-Green-Cover-Detection-in-Informal-Settlements.git](https://github.com/Chanduchawla/Ensemble-Learning-for-Green-Cover-Detection-in-Informal-Settlements.git)
cd Ensemble-Learning-for-Green-Cover-Detection-in-Informal-Settlements

2. Create a Virtual Environment (Recommended)

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install Dependencies
Make sure you have a requirements.txt file in your repository.

pip install -r requirements.txt

4. Data Setup

Download the dataset and organize it into the following directory structure:

data/
├── train_images/
├── train_masks/
├── val_images/
├── val_masks/
├── test_images/
└── test_masks/

5. Run the Notebook

Launch Jupyter Notebook or JupyterLab:

jupyter notebook

Open the ensamble-learning-98295d.ipynb notebook and execute the cells.

📄 Citing This Work
If you find this work useful in your research, please consider citing the original paper:

@inproceedings{your_conference_shortname_2025,
  author    = {Author A and Author B and Author C},
  title     = {Ensemble Deep Learning for Enhanced Green Cover and Open Space Classification in Informal Settlements},
  booktitle = {Conference Name},
  year      = {2025},
}
