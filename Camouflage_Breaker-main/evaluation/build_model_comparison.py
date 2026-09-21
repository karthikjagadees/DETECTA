"""
Build outputs/model_comparison.json from REAL evaluation artifacts.

Reads whatever metrics files exist; never invents values.
Missing evaluations are marked as null / "Not evaluated".
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

OUTPUTS = os.path.join(PROJECT_ROOT, "outputs")


def _load(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _seg_block(name: str, data):
    if not data:
        return {
            "model_name": name,
            "task": "camouflaged_object_segmentation",
            "status": "Not evaluated",
            "metrics": {
                "IoU": None,
                "Dice": None,
                "Precision": None,
                "Recall": None,
                "average_inference_time_sec": None,
            },
        }
    metrics = data.get("metrics", data)
    status = data.get("status")
    if status is None:
        status = "ok" if metrics.get("IoU") is not None else "Not evaluated"
    return {
        "model_name": name,
        "task": "camouflaged_object_segmentation",
        "status": status,
        "checkpoint_path": data.get("checkpoint_path"),
        "dataset": data.get("dataset"),
        "num_test_images": data.get("num_test_images"),
        "timestamp_utc": data.get("timestamp_utc"),
        "metrics": {
            "IoU": metrics.get("IoU"),
            "Dice": metrics.get("Dice"),
            "Precision": metrics.get("Precision"),
            "Recall": metrics.get("Recall"),
            "average_inference_time_sec": metrics.get(
                "average_inference_time_sec",
                metrics.get("inference_time_sec"),
            ),
        },
        "pipeline_with_resnet50": data.get("pipeline_with_resnet50"),
    }


def _cls_block(data):
    if not data:
        return {
            "model_name": "ResNet50",
            "task": "animal_classification",
            "status": "Not evaluated",
            "metrics": {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "average_inference_time_sec": None,
            },
        }
    metrics = data.get("metrics", data)
    return {
        "model_name": "ResNet50",
        "task": "animal_classification",
        "status": "ok",
        "checkpoint_path": data.get("checkpoint_path"),
        "dataset": data.get("dataset"),
        "num_test_images": data.get("num_test_images") or data.get("num_samples"),
        "timestamp_utc": data.get("timestamp_utc"),
        "metrics": {
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1") or metrics.get("f1_score"),
            "average_inference_time_sec": metrics.get(
                "average_inference_time_sec",
                metrics.get("inference_time_sec"),
            ),
        },
    }


def main():
    os.makedirs(OUTPUTS, exist_ok=True)

    sinet = _load(os.path.join(OUTPUTS, "sinetv2_metrics.json"))
    resunet = _load(os.path.join(OUTPUTS, "resunet_metrics.json"))
    resnet = _load(os.path.join(OUTPUTS, "classifier_metrics.json"))
    if resnet is None:
        resnet = _load(os.path.join(OUTPUTS, "resnet50_metrics.json"))
    # Optional end-to-end pipeline metrics
    e2e = _load(os.path.join(OUTPUTS, "pipeline_comparison_metrics.json"))

    comparison = {
        "project": "TUKU DEEP LEARNING",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Values are copied from evaluation JSON artifacts only. "
            "Null means evaluation has not been run yet."
        ),
        "models": {
            "ResNet50": _cls_block(resnet),
            "ResUNet": _seg_block("ResUNet", resunet),
            "SINet-V2": _seg_block("SINet-V2", sinet),
        },
        "end_to_end": e2e,
    }

    out_path = os.path.join(OUTPUTS, "model_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print("Wrote", out_path)
    return comparison


if __name__ == "__main__":
    main()
