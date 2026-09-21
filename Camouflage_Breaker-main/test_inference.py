import os
import sys
import cv2
import torch
import numpy as np
from torchvision import transforms

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.resunet import ResUNet
from utils.image_processing import process_image


def load_model(checkpoint_path, device):
    """Load trained ResUNet from checkpoint."""
    model = ResUNet(encoder_name="resnet50", encoder_weights=None)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"[OK] Loaded model from epoch {checkpoint['epoch']}")
    return model


def preprocess_image(image_path, image_size=352):
    """
    Load and preprocess image for model inference.
    Returns: tensor (1, 3, 352, 352), original_image
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original = image.copy()
    
    # Resize
    image = cv2.resize(image, (image_size, image_size))
    
    # Normalize (same as training) - EXPLICITLY USE FLOAT32
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    
    # To tensor (float32)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.float()  # Force float32
    
    return image_tensor, original

def predict_mask(model, image_tensor, device):
    """Run model inference and return mask."""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
    
    # Convert to numpy
    mask = probs.squeeze().cpu().numpy()
    
    return mask


def save_results(results, output_dir="outputs"):
    """Save all processed images."""
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(f"{output_dir}/result_boundary.jpg", 
                cv2.cvtColor(results['boundary'], cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{output_dir}/result_overlay.jpg", 
                cv2.cvtColor(results['overlay'], cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{output_dir}/result_crop.jpg", 
                cv2.cvtColor(results['crop'], cv2.COLOR_RGB2BGR))
    
    print(f"[OK] Results saved to {output_dir}/")


def main():
    print("=" * 60)
    print("Camouflage Breaker - Inference Test")
    print("=" * 60)
    
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "saved_models/resunet_best.pth"
    
    # Find a test image
    test_dir = "dataset/Test/Image"
    if os.path.exists(test_dir):
        test_images = [f for f in os.listdir(test_dir) 
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
        if test_images:
            image_path = os.path.join(test_dir, test_images[0])
            print(f"[INFO] Using test image: {image_path}")
        else:
            print("[ERROR] No test images found")
            return
    else:
        print("[ERROR] Test directory not found")
        return
    
    # Load model
    print("\n[INFO] Loading model...")
    model = load_model(checkpoint_path, device)
    
    # Preprocess
    print("[INFO] Preprocessing image...")
    image_tensor, original = preprocess_image(image_path)
    print(f"Original shape: {original.shape}")
    print(f"Tensor shape: {image_tensor.shape}")
    
    # Predict
    print("[INFO] Running inference...")
    mask = predict_mask(model, image_tensor, device)
    print(f"Mask shape: {mask.shape}")
    print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
    
    # Resize mask back to original size
    mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]))
    
    # Process
    print("[INFO] Creating visualizations...")
    results = process_image(original, mask_resized, threshold=0.5)
    
    # Save
    save_results(results)
    
    print("\n" + "=" * 60)
    print("Results:")
    print(f"  Boundary: outputs/result_boundary.jpg")
    print(f"  Overlay:  outputs/result_overlay.jpg")
    print(f"  Crop:     outputs/result_crop.jpg")
    print("=" * 60)
    print("\n[OK] Inference test complete!")


if __name__ == "__main__":
    main()