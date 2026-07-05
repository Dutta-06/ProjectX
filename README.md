# ProjectX: Crowd Counting with YOLOv8

This repository provides a complete pipeline for training and evaluating a crowd counting model using YOLOv8. It includes tools to convert `.mat` head-point annotations into YOLO bounding box formats.

## 🚀 Installation

```bash
git clone https://github.com/Dutta-06/ProjectX.git
cd ProjectX
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 🛠️ Usage

### 1. Data Preparation
Convert your `.mat` annotated data to YOLO format:
```bash
python prepare_data.py --train_dir data/train_data --test_dir data/test_data --output_dir yolo_dataset --box_size 20
```

### 2. Training
Train the YOLO model on the generated dataset:
```bash
python train_yolo.py --dataset_yaml yolo_dataset/dataset.yaml --epochs 50 --batch_size 16 --model yolov8n.pt
```

### 3. Evaluation
Evaluate the trained model against the test set:
```bash
python evaluate_crowd_counter.py --model_path yolo_training_runs/crowd_counting_experiment/weights/best.pt --test_img_dir data/test_data/images --test_gt_dir data/test_data/ground_truth
```
