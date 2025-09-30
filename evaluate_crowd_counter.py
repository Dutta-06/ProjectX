import os
import glob
import cv2
import scipy.io
import numpy as np
from ultralytics import YOLO

def evaluate_model(model_path, test_image_dir, test_gt_dir):
    """
    Evaluates the trained crowd counting model on the test set and calculates metrics.

    Args:
        model_path (str): Path to the trained YOLOv8 model (.pt file).
        test_image_dir (str): Directory containing the test images.
        test_gt_dir (str): Directory containing the ground truth .mat files for the test set.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        return

    # Load the trained YOLOv8 model
    model = YOLO(model_path)
    print(f"Loaded model: {model_path}")

    image_files = sorted(glob.glob(os.path.join(test_image_dir, '*.jpg')))
    if not image_files:
        print(f"Error: No .jpg images found in '{test_image_dir}'")
        return

    total_predicted_count = 0
    total_actual_count = 0
    absolute_errors = []

    print("\nStarting evaluation on the test set...")
    for img_path in image_files:
        try:
            # --- Get Predicted Count ---
            results = model(img_path, classes=[0], max_det=1000, verbose=False) # Detect only 'person' class
            predicted_count = len(results[0].boxes)

            # --- Get Actual Count ---
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mat_name = f"GT_{base_name}.mat"
            mat_path = os.path.join(test_gt_dir, mat_name)

            if not os.path.exists(mat_path):
                print(f"Warning: Ground truth for {os.path.basename(img_path)} not found. Skipping.")
                continue
            
            mat_data = scipy.io.loadmat(mat_path)
            
            # Find the key containing ground truth points
            locations = None
            if 'image_info' in mat_data:
                locations = mat_data['image_info'][0, 0][0][0][0]
            else:
                for key in mat_data:
                    if isinstance(mat_data[key], np.ndarray) and mat_data[key].shape[1] == 2:
                        locations = mat_data[key]
                        break

            if locations is None:
                print(f"Could not find coordinate data in {mat_name}. Skipping.")
                continue

            actual_count = len(locations)
            
            # --- Accumulate Results ---
            total_predicted_count += predicted_count
            total_actual_count += actual_count
            
            # Calculate absolute error for this image
            error = abs(predicted_count - actual_count)
            absolute_errors.append(error)
            
            print(f"  - Image: {os.path.basename(img_path):<15} | Actual: {actual_count:<5} | Predicted: {predicted_count:<5} | Error: {error}")

        except Exception as e:
            print(f"An error occurred while processing {os.path.basename(img_path)}: {e}")

    # --- Calculate Performance Metrics ---
    if not absolute_errors:
        print("\nEvaluation could not be completed. No files were processed.")
        return

    mae = np.mean(absolute_errors)
    mse = np.mean(np.square(absolute_errors))
    rmse = np.sqrt(mse)

    print("\n--- Evaluation Metrics ---")
    print(f"Total Images Evaluated: {len(absolute_errors)}")
    print(f"Total Actual People:    {total_actual_count}")
    print(f"Total Predicted People: {total_predicted_count}")
    print("-" * 28)
    print(f"Mean Absolute Error (MAE):  {mae:.2f}")
    print(f"Mean Squared Error (MSE):   {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print("----------------------------")
    print("\nMAE is the average absolute difference between the predicted and actual counts.")
    print("Lower values for all metrics indicate better model performance.")

if __name__ == '__main__':
    # --- Configuration ---
    # Path to the BEST trained model from the training script
    TRAINED_MODEL_PATH = './yolo_training_runs/crowd_counting_experiment2/weights/best.pt'
    
    # Path to the original test dataset folders
    TEST_IMAGE_DIR = 'crowd_wala_dataset\\test_data\images'
    TEST_GT_DIR = 'crowd_wala_dataset\\test_data\ground_truth'
    # -------------------
    
    evaluate_model(TRAINED_MODEL_PATH, TEST_IMAGE_DIR, TEST_GT_DIR)
