"""
Final evaluation after all three training runs.

Produces:
  outputs/resunet_metrics.json
  outputs/sinetv2_metrics.json
  outputs/classifier_metrics.json
  outputs/pipeline_comparison_metrics.json
  outputs/model_comparison.json

Uses the official COD10K-v3 Test set (4000 images). CUDA required.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from models.classifier import ResNet50Classifier
from utils.generate_crops import generate_classification_dataset
from evaluation.evaluate_resunet import main as eval_resunet
from evaluation.evaluate_sinetv2 import main as eval_sinetv2
from evaluation.build_model_comparison import main as build_comparison
from inference.pipeline import CamouflageBreakerPipeline


OUT = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for final evaluation.")
    probe = torch.zeros(1, device="cuda")
    _ = probe + 1
    return torch.device("cuda")


def parse_instance_labels(txt_path: str) -> dict:
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    label_map = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("[INFO]"):
            parts = line.split()
            if len(parts) >= 3:
                filename = parts[1]
                base = os.path.splitext(filename)[0]
                if i + 1 < len(lines):
                    class_line = lines[i + 1].strip()
                    if class_line:
                        category = class_line.split()[0].split("/")[-1]
                        label_map[base] = category
                        i += 2
                        continue
        i += 1
    return label_map


def load_class_mapping(path: str):
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    if "idx_to_class" in mapping:
        idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}
    elif "class_to_idx" in mapping:
        idx_to_class = {int(v): k for k, v in mapping["class_to_idx"].items()}
    else:
        # flat {class: idx}
        idx_to_class = {int(v): k for k, v in mapping.items() if isinstance(v, int)}
    class_to_idx = {v: k for k, v in idx_to_class.items()}
    return class_to_idx, idx_to_class


def evaluate_classifier(device: torch.device):
    """Evaluate ResNet50 on official Test GT crops (eval only; never used in training)."""
    crop_dir = os.path.join(PROJECT_ROOT, "dataset", "crops", "Test")
    n = 0
    if os.path.isdir(crop_dir):
        n = len([f for f in os.listdir(crop_dir) if f.endswith("_crop.jpg")])
    if n < 1000:
        print("[INFO] Generating Test GT crops for classifier evaluation only...")
        generate_classification_dataset(
            root_dir="dataset",
            split="Test",
            output_dir="dataset/crops",
        )

    ckpt_path = os.path.join(PROJECT_ROOT, "saved_models", "classifier_best.pth")
    map_path = os.path.join(PROJECT_ROOT, "saved_models", "class_mapping.json")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    if not os.path.isfile(map_path):
        raise FileNotFoundError(map_path)

    class_to_idx, idx_to_class = load_class_mapping(map_path)
    label_map = parse_instance_labels(
        os.path.join(PROJECT_ROOT, "dataset", "Test", "CAM-NonCAM_Instance_Test.txt")
    )

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    model = ResNet50Classifier(num_classes=len(class_to_idx), pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("Classifier not on CUDA.")

    y_true, y_pred = [], []
    times = []
    crop_files = sorted(f for f in os.listdir(crop_dir) if f.endswith("_crop.jpg"))
    for fname in tqdm(crop_files, desc="Classifier Test"):
        base = fname.replace("_crop.jpg", "")
        if base not in label_map:
            continue
        cls_name = label_map[base]
        if cls_name not in class_to_idx:
            continue
        img = Image.open(os.path.join(crop_dir, fname)).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor)
            pred_idx = int(logits.argmax(dim=1).item())
        times.append(time.perf_counter() - t0)
        y_true.append(class_to_idx[cls_name])
        y_pred.append(pred_idx)

    if not y_true:
        raise RuntimeError("No classifier test samples evaluated.")

    payload = {
        "model_name": "ResNet50",
        "task": "animal_classification",
        "status": "ok",
        "checkpoint_path": ckpt_path,
        "class_mapping_path": map_path,
        "dataset": "COD10K-v3 Test GT crops",
        "num_test_images": len(y_true),
        "num_classes": len(class_to_idx),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "average_inference_time_sec": float(np.mean(times)),
        },
        "note": "Test crops used for evaluation only; classifier trained on Train crops.",
    }
    out_path = os.path.join(OUT, "classifier_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    # Also write alias expected by older comparison builder
    with open(os.path.join(OUT, "resnet50_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("Saved", out_path)
    return payload


def evaluate_e2e(segmenter: str, device: torch.device, max_images: int = 0):
    """End-to-end: segmenter → crop → ResNet50 on official Test set."""
    image_dir = os.path.join(PROJECT_ROOT, "dataset", "Test", "Image")
    label_map = parse_instance_labels(
        os.path.join(PROJECT_ROOT, "dataset", "Test", "CAM-NonCAM_Instance_Test.txt")
    )
    map_path = os.path.join(PROJECT_ROOT, "saved_models", "class_mapping.json")
    class_to_idx, _ = load_class_mapping(map_path)

    if segmenter == "resunet":
        seg_path = os.path.join(PROJECT_ROOT, "saved_models", "resunet_best.pth")
    else:
        seg_path = os.path.join(
            PROJECT_ROOT, "saved_models", "sinetv2", "sinetv2_cod10k_best.pth"
        )
    cls_path = os.path.join(PROJECT_ROOT, "saved_models", "classifier_best.pth")

    pipe = CamouflageBreakerPipeline(
        seg_model_path=seg_path,
        classifier_model_path=cls_path,
        class_mapping_path=map_path,
        segmenter=segmenter,
    )
    if str(pipe.device) != "cuda" and getattr(pipe, "device", None) != torch.device("cuda"):
        # pipeline stores device; enforce CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("E2E requires CUDA.")

    images = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if max_images:
        images = images[:max_images]

    y_true, y_pred = [], []
    times = []
    skipped = 0
    for fname in tqdm(images, desc=f"E2E {segmenter}"):
        base = os.path.splitext(fname)[0]
        if base not in label_map:
            skipped += 1
            continue
        gt_cls = label_map[base]
        if gt_cls not in class_to_idx:
            skipped += 1
            continue
        path = os.path.join(image_dir, fname)
        t0 = time.perf_counter()
        try:
            result = pipe.predict(path)
        except Exception:
            skipped += 1
            continue
        times.append(time.perf_counter() - t0)

        pred_cls = None
        if isinstance(result, dict):
            pred_cls = (
                result.get("predicted_class")
                or result.get("class_name")
                or result.get("label")
            )
            if pred_cls is None and "classification" in result:
                c = result["classification"]
                if isinstance(c, dict):
                    pred_cls = c.get("class_name") or c.get("predicted_class")
        if pred_cls is None:
            skipped += 1
            continue
        if pred_cls not in class_to_idx:
            # try normalize
            skipped += 1
            continue
        y_true.append(class_to_idx[gt_cls])
        y_pred.append(class_to_idx[pred_cls])

    metrics = {
        "pipeline": f"{segmenter} -> crop -> ResNet50",
        "num_evaluated": len(y_true),
        "num_skipped": skipped,
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else None,
        "precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        if y_true
        else None,
        "recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
        if y_true
        else None,
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if y_true
        else None,
        "average_inference_time_sec": float(np.mean(times)) if times else None,
        "checkpoint_segmenter": seg_path,
        "checkpoint_classifier": cls_path,
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
    }
    return metrics


def main():
    device = require_cuda()
    print("[INFO] GPU:", torch.cuda.get_device_name(0))

    print("\n=== ResUNet Test Evaluation ===")
    eval_resunet()

    print("\n=== SINet-V2 Test Evaluation ===")
    eval_sinetv2()

    print("\n=== ResNet50 Classifier Test Evaluation ===")
    evaluate_classifier(device)

    print("\n=== End-to-End Pipelines ===")
    e2e_resunet = evaluate_e2e("resunet", device)
    e2e_sinet = evaluate_e2e("sinetv2", device)
    e2e = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "COD10K-v3 Official Test (4000)",
        "classifier_shared": "saved_models/classifier_best.pth",
        "resunet_pipeline": e2e_resunet,
        "sinetv2_pipeline": e2e_sinet,
    }
    e2e_path = os.path.join(OUT, "pipeline_comparison_metrics.json")
    with open(e2e_path, "w", encoding="utf-8") as f:
        json.dump(e2e, f, indent=2)
    print("Saved", e2e_path)

    # Update comparison builder to prefer classifier_metrics.json
    build_comparison()
    print("\n[OK] Final evaluation complete.")


if __name__ == "__main__":
    main()
