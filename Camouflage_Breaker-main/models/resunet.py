import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class ResUNet(nn.Module):
    """
    ResUNet for camouflaged object segmentation.
    Uses ResNet50 encoder + U-Net decoder.
    Input:  (B, 3, 352, 352)
    Output: (B, 1, 352, 352) - probability map
    """
    
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet"):
        super(ResUNet, self).__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None  # We'll apply sigmoid in loss or inference
        )
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        """Inference: returns sigmoid-activated probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


# ==================== QUICK TEST ====================
if __name__ == "__main__":
    print("=" * 50)
    print("ResUNet Model Test")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create model
    model = ResUNet(encoder_name="resnet50", encoder_weights="imagenet")
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 352, 352).to(device)
    output = model(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test predict
    prob = model.predict(dummy_input)
    print(f"Predict shape: {prob.shape}")
    print(f"Predict range: [{prob.min().item():.3f}, {prob.max().item():.3f}]")
    
    print("\n[OK] ResUNet model test passed!")