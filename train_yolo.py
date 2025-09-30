from ultralytics import YOLO
import os

def train_model(dataset_yaml_path, epochs=100, batch_size=8, model_variant='yolov8n.pt'):
    """
    Trains a YOLOv8 model on the custom crowd counting dataset.

    Args:
        dataset_yaml_path (str): Path to the dataset.yaml file.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        model_variant (str): The base YOLOv8 model to start from (e.g., 'yolov8n.pt').
    """
    if not os.path.exists(dataset_yaml_path):
        print(f"Error: dataset.yaml not found at '{dataset_yaml_path}'")
        print("Please run the prepare_data.py script first.")
        return

    try:
        # Load a pretrained YOLOv8 model
        model = YOLO(model_variant)
        print(f"Loaded pretrained model: {model_variant}")

        print("Starting model training...")
        # Train the model
        model.train(
            data=dataset_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=640, # Image size for training
            project='yolo_training_runs',
            name='crowd_counting_experiment',
            max_det=1000,
        )
        
        print("\nTraining complete.")
        print("The trained model and results are saved in the 'yolo_training_runs' directory.")
        print("The best model weights are typically found at: 'yolo_training_runs/crowd_counting_experiment/weights/best.pt'")

    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == '__main__':
    # --- Configuration ---
    # Path to the yaml file created by prepare_data.py
    DATASET_YAML = 'yolo_dataset\\dataset.yaml'
    
    # Training parameters
    NUM_EPOCHS = 50  # Adjust as needed
    BATCH_SIZE = 16  # Adjust based on your GPU memory
    MODEL_VARIANT = 'yolov8n.pt' # 'yolov8s.pt' or 'yolov8m.pt' for better accuracy
    # -------------------
    
    train_model(DATASET_YAML, NUM_EPOCHS, BATCH_SIZE, MODEL_VARIANT)
