import os
import scipy.io
import numpy as np
from PIL import Image
import yaml
import shutil
from pathlib import Path

def create_yolo_annotations(base_dir: Path, output_dir: Path, box_size=20):
    """
    Parses .mat ground truth files and converts them to YOLOv8 .txt format.

    Args:
        base_dir (Path): The root directory of the dataset (e.g., './train_data').
        output_dir (Path): The directory where the YOLO formatted data will be saved.
        box_size (int): The approximate pixel size for the bounding box around a head coordinate.
    """
    image_dir = base_dir / 'images'
    gt_dir = base_dir / 'ground_truth'

    # The final output folders will be named after the parent (e.g., 'train_data', 'test_data')
    output_img_dir = output_dir / 'images' / base_dir.name
    output_lbl_dir = output_dir / 'labels' / base_dir.name

    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing images from: {image_dir}")
    print(f"Processing ground truth from: {gt_dir}")

    image_files = sorted(list(image_dir.glob('*.jpg')))
    if not image_files:
        print(f"Warning: No .jpg images found in {image_dir}. Please check the path.")
        return

    for img_path in image_files:
        try:
            # Construct corresponding .mat file path
            mat_name = f"GT_{img_path.stem}.mat"
            mat_path = gt_dir / mat_name

            if not mat_path.exists():
                print(f"Warning: Annotation file not found for {img_path.name}. Skipping.")
                continue

            # Load image to get its dimensions
            with Image.open(img_path) as img:
                img_width, img_height = img.size

            # Load .mat file
            mat_data = scipy.io.loadmat(mat_path)
            
            # Extract head locations - trying to find the key automatically
            locations = None
            if 'image_info' in mat_data:
                locations = mat_data['image_info'][0, 0][0][0][0]
            else:
                for key in mat_data:
                    if isinstance(mat_data[key], np.ndarray) and mat_data[key].ndim == 2 and mat_data[key].shape[1] == 2:
                        locations = mat_data[key]
                        break
            
            if locations is None:
                print(f"Could not find coordinate data in {mat_name}. Skipping.")
                continue

            # Create YOLO annotation file
            yolo_txt_path = output_lbl_dir / f"{img_path.stem}.txt"
            with open(yolo_txt_path, 'w') as f:
                for point in locations:
                    x_center, y_center = point
                    
                    # --- VALIDATION STEP ---
                    # Check if the ground truth point is within the image boundaries before processing.
                    if not (0 <= x_center < img_width and 0 <= y_center < img_height):
                        print(f"  [!] Warning: Out-of-bounds annotation in {mat_name}. Point ({x_center:.1f}, {y_center:.1f}) is outside image dims ({img_width}, {img_height}). Skipping this point.")
                        continue # Skip this invalid point
                    
                    # Normalize coordinates
                    norm_x_center = x_center / img_width
                    norm_y_center = y_center / img_height
                    norm_width = box_size / img_width
                    norm_height = box_size / img_height
                    
                    # Write YOLO format: class_id x_center y_center width height
                    f.write(f"0 {norm_x_center:.6f} {norm_y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
            
            # Copy image to the new directory structure using a more robust method
            shutil.copy(img_path, output_img_dir / img_path.name)

        except Exception as e:
            print(f"Error processing file {img_path.name}: {e}")

    print(f"Finished processing directory: {base_dir}")

def create_dataset_yaml(output_dir: Path):
    """Creates the dataset.yaml file needed for YOLO training."""
    yaml_path = output_dir / 'dataset.yaml'
    
    # Get the absolute path to the image directories for robustness
    train_images_path = (output_dir / 'images' / 'train_data').resolve()
    val_images_path = (output_dir / 'images' / 'test_data').resolve()

    data = {
        # Using fully resolved, absolute paths to prevent any ambiguity.
        'train': str(train_images_path),
        'val': str(val_images_path),
        'nc': 1,
        'names': ['person']
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
        
    print(f"dataset.yaml created at {yaml_path}")
    print(f"YAML content:\n{yaml.dump(data)}") # Print content for verification
    return yaml_path

if __name__ == '__main__':
    # --- Configuration ---
    # Define the paths to your original train and test data
    TRAIN_DATA_DIR = Path('crowd_wala_dataset\\train_data')
    TEST_DATA_DIR = Path('crowd_wala_dataset\\test_data')
    
    # Define the output directory for the YOLO-formatted dataset
    YOLO_DATASET_DIR = Path('./yolo_dataset')
    # -------------------

    print("Starting dataset preparation process...")
    
    # Create output directory
    YOLO_DATASET_DIR.mkdir(exist_ok=True)
    
    # Process training data
    print("\n--- Processing Training Data ---")
    create_yolo_annotations(TRAIN_DATA_DIR, YOLO_DATASET_DIR)
    
    # Process test data (will be used as the validation set)
    print("\n--- Processing Test Data (for validation) ---")
    create_yolo_annotations(TEST_DATA_DIR, YOLO_DATASET_DIR)
    
    # Create the dataset.yaml file
    print("\n--- Creating dataset.yaml ---")
    create_dataset_yaml(YOLO_DATASET_DIR)
    
    print("\nData preparation complete.")
    print(f"Your YOLO-formatted dataset is ready in: {YOLO_DATASET_DIR}")

