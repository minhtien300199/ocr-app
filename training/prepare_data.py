import os
from PIL import Image
from config.training_config import DATASET_CONFIG, TRAINING_CONFIG

def prepare_training_data(source_dir, dest_dir):
    """Prepare training data for Tesseract"""
    for lang in DATASET_CONFIG['languages']:
        # Create language directories
        train_lang_dir = os.path.join(dest_dir, 'train', lang)
        val_lang_dir = os.path.join(dest_dir, 'validation', lang)
        
        os.makedirs(train_lang_dir, exist_ok=True)
        os.makedirs(val_lang_dir, exist_ok=True)
        
        # Process source images
        source_lang_dir = os.path.join(source_dir, lang)
        if not os.path.exists(source_lang_dir):
            print(f"Warning: Source directory for language {lang} not found")
            continue
        
        # Get all image files
        images = [f for f in os.listdir(source_lang_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        # Split into train and validation
        split_idx = int(len(images) * (1 - TRAINING_CONFIG['validation_split']))
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Process training images
        for idx, img_file in enumerate(train_images):
            base_name = os.path.splitext(img_file)[0]
            new_name = f"train_{base_name}_{idx:04d}"
            process_training_image(
                os.path.join(source_lang_dir, img_file),
                os.path.join(train_lang_dir, new_name)
            )
        
        # Process validation images
        for idx, img_file in enumerate(val_images):
            base_name = os.path.splitext(img_file)[0]
            new_name = f"val_{base_name}_{idx:04d}"
            process_training_image(
                os.path.join(source_lang_dir, img_file),
                os.path.join(val_lang_dir, new_name)
            )

def process_training_image(src_path, dest_base):
    """Process a single image and create associated files for Tesseract training"""
    try:
        # Load and process image
        img = Image.open(src_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to binary
        img = img.convert('L')  # Convert to grayscale
        img = img.point(lambda x: 0 if x < 128 else 255, '1')  # Convert to binary
        
        # Save as TIFF
        tiff_path = f"{dest_base}.tif"
        img.save(tiff_path, format='TIFF', dpi=(300, 300), compression='group4')
        
        # Create ground truth file
        ground_truth = os.path.splitext(os.path.basename(src_path))[0].split('_')[0]
        gt_path = f"{dest_base}.gt.txt"
        with open(gt_path, 'w', encoding='utf-8') as f:
            f.write(ground_truth)
        
        # Create box file (optional, for fine-tuning)
        box_path = f"{dest_base}.box"
        create_box_file(ground_truth, box_path)
        
    except Exception as e:
        print(f"Error processing {src_path}: {str(e)}")

def create_box_file(text, box_path):
    """Create a box file for Tesseract training"""
    try:
        with open(box_path, 'w', encoding='utf-8') as f:
            # Simple box file format: char left bottom right top page
            left, bottom, right, top = 0, 0, 20, 20  # Default values
            for idx, char in enumerate(text):
                f.write(f"{char} {left} {bottom} {right} {top} 0\n")
                left += 20
                right += 20
    except Exception as e:
        print(f"Error creating box file: {str(e)}")

if __name__ == "__main__":
    # Example usage
    source_directory = "data/source"  # Your source images
    prepare_training_data(source_directory, DATASET_CONFIG['train_data_dir']) 