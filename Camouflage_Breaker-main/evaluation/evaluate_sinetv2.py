"""
Evaluate SINet-V2 on COD10K Test set with REAL metrics.

Writes:
  outputs/sinetv2_metrics.json

Does not modify ResUNet evaluation code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.sinetv2_wrapper import SINetV2Segmenter, resolve_checkpoint_path, select_torch_device


def pair_files(image_dir: str, mask_dir: str):
    images = sorted(
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    masks = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(mask_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    }
    pairs = []
    for img in images:
        base = os.path.splitext(img)[0]
        if base in masks:
            pairs.append((img, masks[base]))
    return pairs


def scores(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    tp = float(np.logical_and(pred == 1, gt == 1).sum())
    fp = float(np.logical_and(pred == 1, gt == 0).sum())
    fn = float(np.logical_and(pred == 0, gt == 1).sum())
    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return iou, dice, precision, recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=0, help="0 = all")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    image_dir = os.path.join(PROJECT_ROOT, "dataset", "Test", "Image")
    mask_dir = os.path.join(PROJECT_ROOT, "dataset", "Test", "GT_Object")
    if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
        raise FileNotFoundError("COD10K Test Image/GT_Object not found.")

    pairs = pair_files(image_dir, mask_dir)
    if args.max_images and args.max_images > 0:
        pairs = pairs[: args.max_images]
    if not pairs:
        raise RuntimeError("No image-mask pairs found.")

    ckpt = resolve_checkpoint_path(args.checkpoint)
    device = select_torch_device()
    print("=" * 70)
    print("SINet-V2 EVALUATION")
    print("=" * 70)
    print("Checkpoint:", ckpt)
    print("Device:", device)
    print("Images:", len(pairs))

    model = SINetV2Segmenter(checkpoint_path=ckpt, device=device)

    ious, dices, precs, recs, times = [], [], [], [], []
    for idx, (img_name, mask_name) in enumerate(pairs, 1):
        image = cv2.imread(os.path.join(image_dir, img_name))
        gt = cv2.imread(os.path.join(mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
        if image is None or gt is None:
            continue

        t0 = time.perf_counter()
        pred = model.predict(image, threshold=args.threshold, assume_bgr=True)
        dt = time.perf_counter() - t0

        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        iou, dice, precision, recall = scores(pred, gt)
        ious.append(iou)
        dices.append(dice)
        precs.append(precision)
        recs.append(recall)
        times.append(dt)

        if idx % 50 == 0 or idx == len(pairs):
            print(
                f"[{idx}/{len(pairs)}] "
                f"IoU={np.mean(ious):.4f} Dice={np.mean(dices):.4f} "
                f"time={np.mean(times):.3f}s"
            )

    result = {
        "model_name": "SINet-V2",
        "task": "camouflaged_object_segmentation",
        "checkpoint_path": ckpt,
        "dataset": "COD10K-v3 Test",
        "num_test_images": len(ious),
        "image_size": 352,
        "threshold": args.threshold,
        "device": str(device),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "IoU": float(np.mean(ious)) if ious else None,
            "Dice": float(np.mean(dices)) if dices else None,
            "Precision": float(np.mean(precs)) if precs else None,
            "Recall": float(np.mean(recs)) if recs else None,
            "average_inference_time_sec": float(np.mean(times)) if times else None,
        },
    }

    out_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sinetv2_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Saved:", out_path)
    print(json.dumps(result["metrics"], indent=2))
    return result


if __name__ == "__main__":
    main()
