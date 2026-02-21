# VinBigData CXR Object Detection

![VinBigData Header](https://raw.githubusercontent.com/furious0420/VinBigData-CXR-Detection/main/predictions/pred_0046f681f078851293c4e710c4466058.jpg)

This repository contains a full object detection pipeline for detecting thoracic abnormalities in Chest X-ray images using the **VinBigData Chest X-ray Abnormalities Detection** dataset.

## 🚀 Overview

The model is built using **PyTorch** and **Faster R-CNN (ResNet-50-FPN)**. It is trained to identify and localize 14 different types of abnormalities, including:
- Aortic enlargement
- Pleural thickening
- Lung Opacity
- Infiltration
- Cardiomegaly
- ... and more.

## 📊 Results

- **Training Epochs**: 10
- **Final Loss**: ~0.19
- **Architecture**: Faster R-CNN with pre-trained ResNet-50 backbone.
- **Evaluation**: COCO-style mAP calculation.

The model shows strong localization capabilities, particularly for clear structural abnormalities like Aortic Enlargement and Cardiomegaly.

## 📂 Project Structure

- `src/dataset.py`: Custom PyTorch Dataset for loading COCO formatted images.
- `src/model.py`: Faster R-CNN model definition.
- `src/train.py`: The main training loop with per-epoch checkpointing.
- `src/evaluate.py`: COCO evaluation module (mAP metrics).
- `src/inference.py`: Script to run predictions on new X-ray images.
- `requirements.txt`: Python dependencies.

## 🛠️ Getting Started

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/furious0420/VinBigData-CXR-Detection.git
   cd VinBigData-CXR-Detection
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Inference**:
   Ensure you have a trained model in the `checkpoints/` folder.
   ```bash
   python src/inference.py
   ```

## 📜 Dataset

The dataset used is the [VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection) dataset, converted to COCO format.

## 🤝 Acknowledgments

Special thanks to VinBigData and the Kaggle community for Providing the dataset and competition framework.
