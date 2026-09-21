import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy + Dice Loss for segmentation.
    """
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, targets):
        # BCE expects logits
        bce_loss = self.bce(logits, targets)
        
        # Dice expects probabilities
        probs = torch.sigmoid(logits)
        dice_loss = self.dice_coefficient(probs, targets)
        
        return self.bce_weight * bce_loss + self.dice_weight * (1 - dice_loss)
    
    def dice_coefficient(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return dice


def iou_score(preds, targets, threshold=0.5, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU) / Jaccard Index.
    """
    preds = (preds > threshold).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def dice_score(preds, targets, threshold=0.5, smooth=1e-6):
    """
    Calculate Dice Score (F1 for segmentation).
    """
    preds = (preds > threshold).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    intersection = (preds * targets).sum()
    dice = (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice.item()


def precision_score(preds, targets, threshold=0.5, smooth=1e-6):
    """Binary segmentation precision."""
    preds = (preds > threshold).float().view(-1)
    targets = targets.view(-1).float()
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    return ((tp + smooth) / (tp + fp + smooth)).item()


def recall_score(preds, targets, threshold=0.5, smooth=1e-6):
    """Binary segmentation recall."""
    preds = (preds > threshold).float().view(-1)
    targets = targets.view(-1).float()
    tp = (preds * targets).sum()
    fn = ((1 - preds) * targets).sum()
    return ((tp + smooth) / (tp + fn + smooth)).item()


# ==================== QUICK TEST ====================
if __name__ == "__main__":
    print("=" * 50)
    print("Metrics Test")
    print("=" * 50)
    
    # Create dummy data
    preds = torch.sigmoid(torch.randn(2, 1, 352, 352))
    targets = torch.randint(0, 2, (2, 1, 352, 352)).float()
    
    # Test BCE+Dice Loss
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    logits = torch.randn(2, 1, 352, 352)  # Raw logits
    loss = criterion(logits, targets)
    print(f"BCE+Dice Loss: {loss.item():.4f}")
    
    # Test IoU
    iou = iou_score(preds, targets)
    print(f"IoU Score: {iou:.4f}")
    
    # Test Dice
    dice = dice_score(preds, targets)
    print(f"Dice Score: {dice:.4f}")
    
    print("\n[OK] Metrics test passed!")