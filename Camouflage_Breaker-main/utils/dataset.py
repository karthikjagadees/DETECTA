import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset


class COD10KDataset(Dataset):
    """
    PyTorch Dataset for COD10K-v3.
    - Segmentation masks from GT_Object/
    - Classification labels from CAM-NonCAM_Instance_{Split}.txt
    Handles filename extension mismatches between TXT and actual files.
    """
    
    def __init__(self, root_dir, split="Train", transform=None, 
                 image_size=352, return_label=False, cam_only=False):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.return_label = return_label
        self.transform = transform
        self.cam_only = cam_only  # If True, only return CAM images
        
        # Folders
        self.image_dir = os.path.join(root_dir, split, "Image")
        self.mask_dir = os.path.join(root_dir, split, "GT_Object")
        
        # Parse TXT file for real class labels (keyed by base name, no extension)
        self.label_cache = {}  # base_name -> label dict
        self._parse_txt_labels()
        
        # Build lookup of disk files by base name
        self.disk_files = {}  # base_name -> actual filename with extension
        for f in os.listdir(self.image_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                base = os.path.splitext(f)[0]
                self.disk_files[base] = f
        
        # Build final image list
        # Include all images that exist on disk and have a mask
        self.image_files = []
        self.base_names = []
        
        for base_name, actual_file in self.disk_files.items():
            mask_path = os.path.join(self.mask_dir, f"{base_name}.png")
            has_mask = os.path.exists(mask_path)
            
            # For segmentation: include all images (even NonCAM with empty masks)
            # For classification (return_label=True): include only labeled images
            if self.return_label:
                if base_name in self.label_cache:
                    if not self.cam_only or self.label_cache[base_name].get('cam_flag', 0) == 1:
                        self.image_files.append(actual_file)
                        self.base_names.append(base_name)
            else:
                # Segmentation mode: include all images that have masks
                if has_mask:
                    self.image_files.append(actual_file)
                    self.base_names.append(base_name)
        
        self.image_files = sorted(self.image_files)
        self.base_names = sorted(self.base_names)
        
        print(f"[INFO] Images ready: {len(self.image_files)}")
        print(f"[INFO] Labels available: {len(self.label_cache)}")
        print(f"[INFO] First image: {self.image_files[0] if self.image_files else 'NONE'}")
    
    def _parse_txt_labels(self):
        """
        Parse CAM-NonCAM_Instance_{Split}.txt
        Format per image (2 lines):
            [INFO] filename.png cam_flag
            CAM/Superclass/Subclass x1 y1 x2 y2 color
        Stores labels keyed by base filename (no extension).
        """
        txt_filename = f"CAM-NonCAM_Instance_{self.split}.txt"
        txt_path = os.path.join(self.root_dir, self.split, txt_filename)
        
        if not os.path.exists(txt_path):
            print(f"[WARN] TXT file not found: {txt_path}")
            return
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('[INFO]'):
                parts = line.split()
                if len(parts) >= 3:
                    filename = parts[1]
                    cam_flag = int(parts[2])
                    base_name = os.path.splitext(filename)[0]  # Remove extension
                    
                    # Parse next line (class info)
                    if i + 1 < len(lines):
                        class_line = lines[i + 1].strip()
                        class_parts = class_line.split()
                        
                        if len(class_parts) >= 1:
                            class_path = class_parts[0]
                            path_parts = class_path.split('/')
                            
                            if len(path_parts) >= 3:
                                cam_type = path_parts[0]
                                superclass = path_parts[1]
                                category = path_parts[2]
                            elif len(path_parts) == 2:
                                cam_type = path_parts[0]
                                superclass = path_parts[1]
                                category = path_parts[1]
                            else:
                                cam_type = 'unknown'
                                superclass = 'unknown'
                                category = 'unknown'
                            
                            self.label_cache[base_name] = {
                                'category': category,
                                'superclass': superclass,
                                'cam_type': cam_type,
                                'cam_flag': cam_flag
                            }
                            i += 2
                            continue
            i += 1
        
        print(f"[INFO] Parsed {len(self.label_cache)} labels from TXT.")
        if self.label_cache:
            sample_key = list(self.label_cache.keys())[0]
            print(f"[INFO] Sample: {sample_key} -> {self.label_cache[sample_key]}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        base_name = self.base_names[idx]
        
        img_path = os.path.join(self.image_dir, img_file)
        mask_path = os.path.join(self.mask_dir, f"{base_name}.png")
        
        # --- Load image ---
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # --- Load or create mask ---
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        else:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
        
        # --- Resize ---
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), 
                          interpolation=cv2.INTER_NEAREST)
        
        # --- Normalize image to float32 ---
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        # --- Process mask ---
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)
        
        # --- Convert to tensors ---
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        
        if self.return_label:
            label_info = self.label_cache.get(base_name, {
                'category': 'unknown',
                'superclass': 'unknown',
                'cam_type': 'unknown',
                'cam_flag': -1
            })
            return image, mask, label_info
        
        return image, mask


def get_dataloader(root_dir, split="Train", batch_size=8, 
                   shuffle=True, num_workers=0, **dataset_kwargs):
    dataset = COD10KDataset(root_dir=root_dir, split=split, **dataset_kwargs)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    return dataloader


# ==================== QUICK TEST ====================
if __name__ == "__main__":
    print("=" * 60)
    print("COD10K Dataset Loader Test (Fixed Extension Matching)")
    print("=" * 60)
    
    dataset_root = "dataset"
    
    # Test segmentation mode (all images with masks)
    print("\n--- Testing Segmentation Mode ---")
    seg_dataset = COD10KDataset(root_dir=dataset_root, split="Train", return_label=False)
    print(f"Dataset size: {len(seg_dataset)}")
    
    if len(seg_dataset) > 0:
        img, mask = seg_dataset[0]
        print(f"Image tensor shape: {img.shape}")
        print(f"Image dtype: {img.dtype}")
        print(f"Mask tensor shape: {mask.shape}")
        print(f"Mask unique values: {torch.unique(mask)}")
    
    # Test classification mode (only labeled images)
    print("\n--- Testing Classification Mode ---")
    cls_dataset = COD10KDataset(root_dir=dataset_root, split="Train", return_label=True)
    print(f"Dataset size: {len(cls_dataset)}")
    
    if len(cls_dataset) > 0:
        print("\n--- Sample Labels (first 10) ---")
        for i in range(min(10, len(cls_dataset))):
            _, _, lbl = cls_dataset[i]
            print(f"  {i}: {lbl}")
    
    # Test CAM-only classification
    print("\n--- Testing CAM-Only Classification ---")
    cam_dataset = COD10KDataset(root_dir=dataset_root, split="Train", 
                                 return_label=True, cam_only=True)
    print(f"CAM-only dataset size: {len(cam_dataset)}")
    
    # Test DataLoader
    print("\n--- Testing DataLoader ---")
    loader = get_dataloader(dataset_root, split="Train", batch_size=4, 
                            num_workers=0, return_label=True)
    batch_imgs, batch_masks, batch_labels = next(iter(loader))
    print(f"Batch image shape: {batch_imgs.shape}")
    print(f"Batch mask shape: {batch_masks.shape}")
    # batch_labels is now a dict of lists, not a list of dicts
    print(f"Sample categories: {batch_labels['category'][:4]}")
    print(f"Sample superclasses: {batch_labels['superclass'][:4]}")
    print(f"Sample cam_flags: {batch_labels['cam_flag'][:4]}")
    
    print("\n[OK] Dataset Loader test completed successfully!")