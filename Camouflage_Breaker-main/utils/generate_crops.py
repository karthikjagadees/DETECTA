import os
import sys
import cv2
import json
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_processing import crop_object


def generate_classification_dataset(root_dir="dataset", split="Train", 
                                    output_dir="dataset/crops"):
    """
    Generate cropped animal images from ground-truth masks.
    These will be used to train the ResNet50 classifier.
    """
    print("=" * 60)
    print(f"Generating Classification Dataset ({split})")
    print("=" * 60)
    
    image_dir = os.path.join(root_dir, split, "Image")
    mask_dir = os.path.join(root_dir, split, "GT_Object")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # Get all images
    image_files = [f for f in os.listdir(image_dir) 
                   if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"[INFO] Processing {len(image_files)} images...")
    
    generated = 0
    skipped = 0
    
    for img_file in tqdm(image_files):
        base_name = os.path.splitext(img_file)[0]
        
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, f"{base_name}.png")
        out_path = os.path.join(output_dir, split, f"{base_name}_crop.jpg")
        
        # Skip if no mask
        if not os.path.exists(mask_path):
            skipped += 1
            continue
        
        # Load
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if mask has any foreground
        if mask.max() == 0:
            skipped += 1
            continue
        
        # Crop
        crop, bbox = crop_object(image, mask / 255.0, padding=20, threshold=0.5)
        
        # Save
        cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        generated += 1
    
    print(f"\n[OK] Generated: {generated} crops")
    print(f"[OK] Skipped: {skipped} (no object or no mask)")
    print(f"[OK] Saved to: {output_dir}/{split}/")


if __name__ == "__main__":
    # Generate for both Train and Test
    generate_classification_dataset(split="Train")
    generate_classification_dataset(split="Test")
    
    print("\n[OK] All crops generated!")