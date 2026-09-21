import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classifier import ResNet50Classifier


# ==================== CONFIG ====================
class Config:
    CROPS_ROOT = "dataset/crops"
    SPLIT = "Train"
    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    NUM_WORKERS = 0
    EPOCHS = 30
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    VAL_RATIO = 0.1
    SEED = 42
    
    CHECKPOINT_DIR = "saved_models"
    CHECKPOINT_NAME = "classifier_best.pth"
    PATIENCE = 5


def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================== DATASET ====================
class CropDataset(Dataset):
    """Dataset for cropped animal images with labels."""
    
    def __init__(self, crops_dir, split="Train", transform=None):
        self.crops_dir = os.path.join(crops_dir, split)
        self.transform = transform
        
        # Load labels from TXT
        self.samples = []  # [(image_path, class_id, class_name), ...]
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        self._load_labels()
    
    def _load_labels(self):
        """Parse CAM-NonCAM labels and map to crop files."""
        txt_path = f"dataset/{self.crops_dir.split('/')[-2] if '/' in self.crops_dir else 'dataset'}/{self.crops_dir.split('/')[-1] if '/' in self.crops_dir else 'Train'}/CAM-NonCAM_Instance_{self.crops_dir.split('/')[-1] if '/' in self.crops_dir else 'Train'}.txt"
        
        # Simpler: use the original TXT from dataset root
        split_name = os.path.basename(self.crops_dir)
        txt_path = f"dataset/{split_name}/CAM-NonCAM_Instance_{split_name}.txt"
        
        # Build label lookup
        label_map = {}
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('[INFO]'):
                parts = line.split()
                if len(parts) >= 3:
                    filename = parts[1]
                    base = os.path.splitext(filename)[0]
                    
                    if i + 1 < len(lines):
                        class_line = lines[i + 1].strip()
                        class_path = class_line.split()[0]
                        category = class_path.split('/')[-1] if '/' in class_path else 'unknown'
                        
                        label_map[base] = category
                        i += 2
                        continue
            i += 1
        
        # Build class index
        all_classes = sorted(set(label_map.values()))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Build samples list
        for f in os.listdir(self.crops_dir):
            if f.endswith('_crop.jpg'):
                base = f.replace('_crop.jpg', '')
                if base in label_map:
                    cls_name = label_map[base]
                    cls_id = self.class_to_idx[cls_name]
                    self.samples.append((
                        os.path.join(self.crops_dir, f),
                        cls_id,
                        cls_name
                    ))
        
        print(f"[INFO] Classes: {len(self.class_to_idx)}")
        print(f"[INFO] Samples: {len(self.samples)}")
        print(f"[INFO] Sample classes: {list(self.class_to_idx.keys())[:5]}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, cls_id, cls_name = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, cls_id


# ==================== TRAINING ====================
def get_dataloaders(config):
    """Create train/val dataloaders."""
    
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create full dataset
    full_dataset = CropDataset(config.CROPS_ROOT, config.SPLIT, transform=None)
    
    # Split
    total = len(full_dataset)
    val_size = int(total * config.VAL_RATIO)
    train_size = total - val_size
    
    train_set, val_set = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    
    # Apply transforms
    train_set.dataset.transform = transform
    val_set.dataset.transform = val_transform
    
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS)
    
    return train_loader, val_loader, full_dataset.class_to_idx, full_dataset.idx_to_class


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 
                         'acc': f"{100*correct/total:.2f}%"})
    
    return total_loss / len(loader), 100 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 
                         'acc': f"{100*correct/total:.2f}%"})
    
    return total_loss / len(loader), 100 * correct / total


def save_checkpoint(model, optimizer, epoch, path, class_to_idx, idx_to_class):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'num_classes': len(class_to_idx)
    }
    torch.save(checkpoint, path)
    print(f"[OK] Saved: {path}")


def main():
    print("=" * 60)
    print("Camouflage Breaker - Classifier Training")
    print("=" * 60)
    
    config = Config()
    set_seed(config.SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Data
    print("\n[INFO] Loading dataset...")
    train_loader, val_loader, class_to_idx, idx_to_class = get_dataloaders(config)
    num_classes = len(class_to_idx)
    print(f"[INFO] Number of classes: {num_classes}")
    
    # Model
    print("[INFO] Building ResNet50 classifier...")
    model = ResNet50Classifier(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Total parameters: {total:,}")
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.LR, 
                      weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training loop
    best_acc = 0
    patience_counter = 0
    
    print(f"\n[INFO] Training for {config.EPOCHS} epochs...")
    print("-" * 60)
    
    for epoch in range(config.EPOCHS):
        start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                            optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, 
                                      device, epoch)
        
        scheduler.step(val_acc)
        
        epoch_time = time.time() - start
        
        print(f"Epoch {epoch:02d}/{config.EPOCHS} | "
              f"Time: {epoch_time:.1f}s | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}\n"
              f"  Train -> Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%\n"
              f"  Val   -> Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            path = os.path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME)
            save_checkpoint(model, optimizer, epoch, path, class_to_idx, idx_to_class)
            print(f"  [★] New best! Val Acc: {best_acc:.2f}% -> Saved!")
        else:
            patience_counter += 1
            print(f"  [ ] No improvement ({patience_counter}/{config.PATIENCE})")
        
        if patience_counter >= config.PATIENCE:
            print(f"\n[STOP] Early stopping at epoch {epoch+1}")
            break
        
        print("-" * 60)
    
    print(f"\n{'=' * 60}")
    print(f"Training Complete!")
    print(f"Best Val Accuracy: {best_acc:.2f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()