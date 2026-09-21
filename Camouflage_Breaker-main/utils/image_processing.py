import cv2
import numpy as np


def get_contours_from_mask(mask, threshold=0.5):
    """
    Convert probability mask to contour points.
    mask: numpy array (H, W) with values 0-1
    Returns: list of contours
    """
    # Binarize mask
    binary_mask = (mask > threshold).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(
        binary_mask, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    return contours


def draw_boundary(image, mask, color=(0, 0, 255), thickness=2, threshold=0.5):
    """
    Draw red boundary around detected object.
    
    Args:
        image: numpy array (H, W, 3) in RGB, values 0-255
        mask: numpy array (H, W) probabilities 0-1
        color: BGR color tuple (default red)
        thickness: line thickness
    
    Returns:
        image with boundary drawn
    """
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    contours = get_contours_from_mask(mask, threshold)
    
    # Draw all contours
    cv2.drawContours(image_bgr, contours, -1, color, thickness)
    
    # Convert back to RGB
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def create_overlay(image, mask, color=(255, 0, 0), alpha=0.4, threshold=0.5):
    """
    Create semi-transparent colored overlay on detected object.
    
    Args:
        image: numpy array (H, W, 3) RGB, 0-255
        mask: numpy array (H, W) probabilities 0-1
        color: RGB color tuple (default blue)
        alpha: transparency (0= invisible, 1= solid)
    
    Returns:
        image with overlay
    """
    # Create colored mask
    colored_mask = np.zeros_like(image)
    binary_mask = (mask > threshold).astype(np.uint8)
    
    for c in range(3):
        colored_mask[:, :, c] = binary_mask * color[c]
    
    # Blend image and colored mask
    overlay = cv2.addWeighted(image, 1.0, colored_mask, alpha, 0)
    
    return overlay


def crop_object(image, mask, padding=20, threshold=0.5):
    """
    Crop the detected object with padding.
    
    Args:
        image: numpy array (H, W, 3) RGB, 0-255
        mask: numpy array (H, W) probabilities 0-1
        padding: pixels to add around the crop
    
    Returns:
        cropped_image, (x, y, w, h) of crop region
    """
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # Find bounding box of mask
    coords = cv2.findNonZero(binary_mask)
    
    if coords is None:
        # No object detected
        return image, (0, 0, image.shape[1], image.shape[0])
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)
    
    cropped = image[y1:y2, x1:x2]
    
    return cropped, (x1, y1, x2 - x1, y2 - y1)


def process_image(image, mask, boundary_color=(0, 0, 255), 
                  overlay_color=(255, 0, 0), overlay_alpha=0.4,
                  padding=20, threshold=0.5):
    """
    Complete processing pipeline: boundary + overlay + crop.
    
    Args:
        image: numpy array (H, W, 3) RGB, 0-255
        mask: numpy array (H, W) probabilities 0-1
    
    Returns:
        dict with 'boundary', 'overlay', 'crop', 'bbox'
    """
    # Ensure correct types
    image = image.astype(np.uint8)
    
    results = {
        'boundary': draw_boundary(image.copy(), mask, color=boundary_color, 
                                   thickness=2, threshold=threshold),
        'overlay': create_overlay(image.copy(), mask, color=overlay_color, 
                                   alpha=overlay_alpha, threshold=threshold),
        'crop': None,
        'bbox': None
    }
    
    crop_img, bbox = crop_object(image, mask, padding=padding, 
                                  threshold=threshold)
    results['crop'] = crop_img
    results['bbox'] = bbox
    
    return results


# ==================== QUICK TEST ====================
if __name__ == "__main__":
    print("=" * 50)
    print("Image Processing Test")
    print("=" * 50)
    
    # Create dummy image and mask
    image = np.random.randint(0, 255, (352, 352, 3), dtype=np.uint8)
    mask = np.zeros((352, 352), dtype=np.float32)
    mask[100:250, 100:250] = 0.9  # Fake object in center
    
    print(f"Input image shape: {image.shape}")
    print(f"Input mask shape: {mask.shape}")
    print(f"Mask max value: {mask.max():.2f}")
    
    # Process
    results = process_image(image, mask)
    
    print(f"\nBoundary image shape: {results['boundary'].shape}")
    print(f"Overlay image shape: {results['overlay'].shape}")
    print(f"Crop image shape: {results['crop'].shape}")
    print(f"Bounding box: {results['bbox']}")
    
    # Save test outputs
    import os
    os.makedirs("outputs", exist_ok=True)
    
    cv2.imwrite("outputs/test_boundary.jpg", 
                cv2.cvtColor(results['boundary'], cv2.COLOR_RGB2BGR))
    cv2.imwrite("outputs/test_overlay.jpg", 
                cv2.cvtColor(results['overlay'], cv2.COLOR_RGB2BGR))
    cv2.imwrite("outputs/test_crop.jpg", 
                cv2.cvtColor(results['crop'], cv2.COLOR_RGB2BGR))
    
    print("\n[OK] Test images saved to outputs/")
    print("[OK] Image processing test passed!")