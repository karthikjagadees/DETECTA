"""
Evaluate ResUNet on COD10K Test — additive script (does not rewrite evaluate_models.py).
Writes outputs/resunet_metrics.json with REAL metrics when checkpoint exists.
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

from models.resunet import ResUNet


def pair_files(image_dir, mask_dir):
    images = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    masks = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(mask_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    }
    return [(i, masks[os.path.splitext(i)[0]]) for i in images if os.path.splitext(i)[0] in masks]


def scores(pred, gt, eps=1e-7):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    tp = float(np.logical_and(pred == 1, gt == 1).sum())
    fp = float(np.logical_and(pred == 1, gt == 0).sum())
    fn = float(np.logical_and(pred == 0, gt == 1).sum())
    return (
        (tp + eps) / (tp + fp + fn + eps),
        (2 * tp + eps) / (2 * tp + fp + fn + eps),
        (tp + eps) / (tp + fp + eps),
        (tp + eps) / (tp + fn + eps),
    )


def preprocess(bgr, size=352, device="cpu"):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size))
    t = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(PROJECT_ROOT, "saved_models", "resunet_best.pth"),
    )
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "resunet_metrics.json")

    if not os.path.isfile(args.checkpoint):
        payload = {
            "model_name": "ResUNet",
            "task": "camouflaged_object_segmentation",
            "status": "Not evaluated",
            "checkpoint_path": args.checkpoint,
            "error": "Checkpoint file not found",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "IoU": None,
                "Dice": None,
                "Precision": None,
                "Recall": None,
                "average_inference_time_sec": None,
            },
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("ResUNet checkpoint missing — wrote Not evaluated:", out_path)
        return payload

    image_dir = os.path.join(PROJECT_ROOT, "dataset", "Test", "Image")
    mask_dir = os.path.join(PROJECT_ROOT, "dataset", "Test", "GT_Object")
    pairs = pair_files(image_dir, mask_dir)
    if args.max_images:
        pairs = pairs[: args.max_images]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required for ResUNet evaluation.")
    try:
        _ = torch.zeros(1, device=device) + 1
    except Exception as exc:
        raise RuntimeError(f"CUDA probe failed during evaluation: {exc}") from exc
    model = ResUNet(encoder_name="resnet50", encoder_weights=None)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()

    ious, dices, precs, recs, times = [], [], [], [], []
    for idx, (img_name, mask_name) in enumerate(pairs, 1):
        image = cv2.imread(os.path.join(image_dir, img_name))
        gt = cv2.imread(os.path.join(mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
        if image is None or gt is None:
            continue
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(preprocess(image, device=device))
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        dt = time.perf_counter() - t0
        pred = (cv2.resize(prob, (gt.shape[1], gt.shape[0])) > args.threshold).astype(np.uint8)
        iou, dice, precision, recall = scores(pred, gt)
        ious.append(iou); dices.append(dice); precs.append(precision); recs.append(recall); times.append(dt)
        if idx % 50 == 0 or idx == len(pairs):
            print(f"[{idx}/{len(pairs)}] IoU={np.mean(ious):.4f}")

    payload = {
        "model_name": "ResUNet",
        "task": "camouflaged_object_segmentation",
        "status": "ok",
        "checkpoint_path": args.checkpoint,
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
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("Saved", out_path)
    return payload


if __name__ == "__main__":
    main()
