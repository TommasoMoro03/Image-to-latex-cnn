import os
import pandas as pd
from PIL import Image, ImageOps
import re
import json
from collections import Counter
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = "./archive"
ORIGINAL_IMAGES_DIR = os.path.join(DATA_DIR, "formula_images_processed", "formula_images_processed")
TRAIN_CSV = os.path.join(DATA_DIR, 'im2latex_train.csv')
VALID_CSV = os.path.join(DATA_DIR, 'im2latex_validate.csv')
TEST_CSV = os.path.join(DATA_DIR, 'im2latex_test.csv')

# Output directories for processed images and updated CSVs
PROCESSED_DATA_DIR = "./processed_data"
PROCESSED_IMAGES_TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, 'train_images')
PROCESSED_IMAGES_VALID_DIR = os.path.join(PROCESSED_DATA_DIR, 'val_images')
PROCESSED_IMAGES_TEST_DIR = os.path.join(PROCESSED_DATA_DIR, 'test_images')

# Image Preprocessing Parameters
TARGET_IMG_HEIGHT = 160  # Based on my previous notebooks analysis
TARGET_IMG_WIDTH = 800


def load_dataset_split_csv(filepath):
    """Loads a CSV file containing image names and LaTeX formulas."""
    df = pd.read_csv(filepath)
    data = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {os.path.basename(filepath)}"):
        img_name = row['image']
        formula = str(row['formula'])
        img_path = os.path.join(ORIGINAL_IMAGES_DIR, img_name)
        data.append({'image_name': img_name, 'image_path': img_path, 'formula': formula})
    return data


def pad_image_to_size(img, target_size, background_color=(255, 255, 255)):
    """
    Pads an image to the target (width, height) with symmetric padding.
    The image is centered, and the background is white.
    """
    target_width, target_height = target_size
    img = img.convert("RGB")  # Ensure consistent mode

    width, height = img.size

    delta_width = target_width - width
    delta_height = target_height - height

    pad_left = delta_width // 2
    pad_right = delta_width - pad_left
    pad_top = delta_height // 2
    pad_bottom = delta_height - pad_top

    padding = (pad_left, pad_top, pad_right, pad_bottom)
    return ImageOps.expand(img, padding, fill=background_color)


def process_and_save_data(data_list, output_images_dir, output_csv_path, target_size, excluded_image_names=None):
    """
    Processes images (pads), saves them, and creates an updated CSV.
    Optionally excludes images by name.
    """
    os.makedirs(output_images_dir, exist_ok=True)

    processed_records = []
    excluded_image_names = excluded_image_names or set()

    for record in tqdm(data_list, desc=f"Processing and saving to {os.path.basename(output_images_dir)}"):
        original_img_path = record['image_path']
        original_img_name = record['image_name']
        formula = record['formula']

        if original_img_name in excluded_image_names:
            continue  # Skip excluded images

        try:
            with Image.open(original_img_path) as img:
                padded_img = pad_image_to_size(img, target_size)

                # New filename convention for processed images
                new_img_name = f"{os.path.splitext(original_img_name)[0]}_padded.png"
                save_path = os.path.join(output_images_dir, new_img_name)
                padded_img.save(save_path)

                processed_records.append({
                    'image_name': new_img_name,
                    'formula': formula
                })
        except Exception as e:
            print(f"Error processing {original_img_path}: {e}")
            continue

    # Save the updated CSV
    processed_df = pd.DataFrame(processed_records)
    processed_df.to_csv(output_csv_path, index=False)
    print(f"Saved {len(processed_records)} processed images and updated CSV to {output_csv_path}")


def execute():
    print("Starting data preprocessing...")

    # 1. Load original datasets
    train_data_raw = load_dataset_split_csv(TRAIN_CSV)
    valid_data_raw = load_dataset_split_csv(VALID_CSV)
    test_data_raw = load_dataset_split_csv(TEST_CSV)

    print(f"\nRaw data loaded: Train={len(train_data_raw)}, Valid={len(valid_data_raw)}, Test={len(test_data_raw)}")

    # 2. Identify tall images for exclusion from TEST set (as per your request)
    # Collect all formulas for vocabulary building later (from ALL splits)
    all_formulas_for_vocab = [item['formula'] for item in train_data_raw + valid_data_raw + test_data_raw]

    # Find tall images in the TEST set only
    tall_test_image_names = set()
    print(f"\nIdentifying images taller than {TARGET_IMG_HEIGHT}px in TEST set...")
    for record in tqdm(test_data_raw, desc="Scanning test images for height"):
        img_path = record['image_path']
        try:
            with Image.open(img_path) as img:
                _, height = img.size
                if height > TARGET_IMG_HEIGHT:
                    tall_test_image_names.add(record['image_name'])
        except Exception as e:
            print(f"Warning: Could not check height for {img_path}: {e}")
            # Consider if you want to exclude these as well or handle

    print(
        f"Found {len(tall_test_image_names)} images in TEST set taller than {TARGET_IMG_HEIGHT}px to exclude.")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # 3. Process and Save Datasets (with exclusions for test set)
    print("\nProcessing and saving training data...")
    process_and_save_data(train_data_raw, PROCESSED_IMAGES_TRAIN_DIR,
                          os.path.join(PROCESSED_DATA_DIR, 'train_processed.csv'),
                          (TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT))

    print("\nProcessing and saving validation data...")
    process_and_save_data(valid_data_raw, PROCESSED_IMAGES_VALID_DIR,
                          os.path.join(PROCESSED_DATA_DIR, 'val_processed.csv'), (TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT))

    print("\nProcessing and saving test data (excluding tall images)...")
    process_and_save_data(test_data_raw, PROCESSED_IMAGES_TEST_DIR,
                          os.path.join(PROCESSED_DATA_DIR, 'test_processed.csv'), (TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT),
                          excluded_image_names=tall_test_image_names)

    print("\nPreprocessing complete! Processed images and updated CSVs are in the 'processed_data' directory.")

if __name__ == "__main__":
    execute()