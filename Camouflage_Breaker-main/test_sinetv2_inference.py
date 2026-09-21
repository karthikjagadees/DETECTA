"""
Standalone SINet-V2 inference smoke test.

Does NOT touch ResUNet. Proves:
- checkpoint loads
- multi-output (res5..res2) structure
- res2 → sigmoid → mask → resize
- visualization saved under outputs/sinetv2/
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models.sinetv2_wrapper import SINetV2Segmenter, resolve_checkpoint_path, select_torch_device


def main():
    print("=" * 70)
    print("SINet-V2 INFERENCE SMOKE TEST")
    print("=" * 70)

    ckpt = resolve_checkpoint_path()
    print("Checkpoint:", ckpt)

    device = select_torch_device()
    print("Device:", device)

    segmenter = SINetV2Segmenter(checkpoint_path=ckpt, device=device)
    print("Model loaded OK")

    test_dir = os.path.join(PROJECT_ROOT, "dataset", "Test", "Image")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(test_dir)

    images = sorted(
        f
        for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not images:
        raise RuntimeError("No test images found.")

    image_path = os.path.join(test_dir, images[0])
    print("Test image:", image_path)

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise RuntimeError(f"Failed to read {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # Build tensor the same way as wrapper training/inference
    resized = cv2.resize(rgb, (352, 352))
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = ((tensor - mean) / std).unsqueeze(0).to(segmenter.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        raw = segmenter.model(tensor)
    elapsed = time.perf_counter() - t0

    if not isinstance(raw, (tuple, list)) or len(raw) < 4:
        raise RuntimeError(f"Expected 4 SINet outputs, got: {type(raw)} / {getattr(raw, 'shape', None)}")

    print("Outputs:", len(raw), "shapes:", [tuple(t.shape) for t in raw])
    res2 = raw[3]
    print("Using res2:", tuple(res2.shape))

    mask = segmenter.predict(bgr, threshold=0.5, assume_bgr=True)
    print("Mask shape:", mask.shape, "dtype:", mask.dtype)
    print("Foreground pixels:", int(mask.sum()), f"/ {mask.size}")
    print(f"Forward time: {elapsed:.4f}s")

    if mask.shape != (h, w):
        raise RuntimeError(f"Mask size {mask.shape} != image {(h, w)}")

    overlay = bgr.copy()
    colored = np.zeros_like(bgr)
    colored[:, :, 1] = 255  # green
    mask3 = np.repeat(mask[:, :, None], 3, axis=2).astype(bool)
    overlay[mask3] = (0.45 * overlay[mask3] + 0.55 * colored[mask3]).astype(np.uint8)

    out_dir = os.path.join(PROJECT_ROOT, "outputs", "sinetv2")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(out_dir, f"{base}_mask.png")
    overlay_path = os.path.join(out_dir, f"{base}_overlay.jpg")
    cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
    cv2.imwrite(overlay_path, overlay)

    print("Saved:", mask_path)
    print("Saved:", overlay_path)

    if int(mask.sum()) == 0:
        print("WARNING: empty mask — check checkpoint / threshold.")
        return 2

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
