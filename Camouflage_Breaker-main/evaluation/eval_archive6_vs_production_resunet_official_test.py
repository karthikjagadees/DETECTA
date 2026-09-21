"""
Controlled Official Test evaluation: production ResUNet vs Archive-6 ResUNet.

EVALUATION ONLY — does not train, tune, or overwrite production checkpoints.
Uses the same protocol as evaluation/evaluate_resunet.py:
  ImageNet normalize, 352x352, sigmoid 0.5, resize pred to GT, eps=1e-7 metrics.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.resunet import ResUNet

OUT_DIR = PROJECT_ROOT / "outputs" / "archive6_resunet_official_test_eval"
IMAGE_DIR = PROJECT_ROOT / "dataset" / "Test" / "Image"
MASK_DIR = PROJECT_ROOT / "dataset" / "Test" / "GT_Object"

PROD_CKPT = PROJECT_ROOT / "saved_models" / "resunet_best.pth"
NEW_CKPT = (
    PROJECT_ROOT
    / "outputs"
    / "archive6_resunet_manifest_s42"
    / "checkpoints"
    / "resunet_archive6_manifest_best.pth"
)
EXPECTED_NEW_SHA = "0a40d72a6caa92a2c02696eaf86c63056dc9f2c6b9a6f18388d776e2238b2ca5"

THRESHOLD = 0.5
IMAGE_SIZE = 352
EPS = 1e-7

PROTECTED = [
    PROJECT_ROOT / "saved_models" / "resunet_best.pth",
    PROJECT_ROOT / "saved_models" / "sinetv2" / "sinetv2_cod10k_best.pth",
    PROJECT_ROOT / "saved_models" / "classifier_best.pth",
    PROJECT_ROOT / "saved_models" / "escnet_finetune" / "escnet_cod10k_best.pth",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_meta(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    st = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "size": st.st_size,
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "mtime_ns": st.st_mtime_ns,
    }


def scores(pred: np.ndarray, gt: np.ndarray, eps: float = EPS):
    """Identical to evaluation/evaluate_resunet.py."""
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
        tp,
        fp,
        fn,
    )


def preprocess(bgr, size=IMAGE_SIZE, device="cpu"):
    """Identical to evaluation/evaluate_resunet.py (ImageNet mean/std)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size))
    t = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0).to(device)


def pair_files(image_dir: Path, mask_dir: Path):
    images = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    masks = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(mask_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    }
    pairs = [(i, masks[os.path.splitext(i)[0]]) for i in images if os.path.splitext(i)[0] in masks]
    missing_masks = [i for i in images if os.path.splitext(i)[0] not in masks]
    extra_masks = sorted(set(masks) - {os.path.splitext(i)[0] for i in images})
    return pairs, missing_masks, extra_masks


def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    model = ResUNet(encoder_name="resnet50", encoder_weights=None)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def evaluate_model(name: str, ckpt_path: Path, pairs, device: torch.device, csv_path: Path):
    model = load_model(ckpt_path, device)
    rows = []
    ious, dices, precs, recs, times = [], [], [], [], []
    gt_empty = pred_empty = both_empty = 0
    gt_ne_pred_e = gt_e_pred_ne = 0
    gt_nonempty = pred_nonempty = 0

    t_wall0 = time.perf_counter()
    for idx, (img_name, mask_name) in enumerate(pairs, 1):
        image = cv2.imread(str(IMAGE_DIR / img_name))
        gt = cv2.imread(str(MASK_DIR / mask_name), cv2.IMREAD_GRAYSCALE)
        if image is None or gt is None:
            raise RuntimeError(f"Failed to read {img_name} / {mask_name}")

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(preprocess(image, device=device))
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        dt = time.perf_counter() - t0

        pred = (cv2.resize(prob, (gt.shape[1], gt.shape[0])) > THRESHOLD).astype(np.uint8)
        gt_bin = (gt > 0).astype(np.uint8)
        iou, dice, precision, recall, tp, fp, fn = scores(pred, gt_bin)

        gt_fg = int(gt_bin.sum())
        pred_fg = int(pred.sum())
        ge = gt_fg == 0
        pe = pred_fg == 0
        if ge:
            gt_empty += 1
        else:
            gt_nonempty += 1
        if pe:
            pred_empty += 1
        else:
            pred_nonempty += 1
        if ge and pe:
            both_empty += 1
        if (not ge) and pe:
            gt_ne_pred_e += 1
        if ge and (not pe):
            gt_e_pred_ne += 1

        ious.append(iou)
        dices.append(dice)
        precs.append(precision)
        recs.append(recall)
        times.append(dt)

        rows.append(
            {
                "image_filename": img_name,
                "mask_filename": mask_name,
                "gt_foreground_pixels": gt_fg,
                "predicted_foreground_pixels": pred_fg,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "IoU": iou,
                "Dice": dice,
                "Precision": precision,
                "Recall": recall,
                "inference_time_sec": dt,
            }
        )
        if idx % 200 == 0 or idx == len(pairs):
            print(f"[{name}] {idx}/{len(pairs)} running_mean_IoU={np.mean(ious):.4f}")

    wall = time.perf_counter() - t_wall0

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # failure analysis lists (new model primarily; computed for both)
    by_iou = sorted(rows, key=lambda r: r["IoU"])
    lowest20 = by_iou[:20]
    highest20 = list(reversed(by_iou[-20:]))
    # largest FP: high predicted FG with low GT overlap — use fp count
    largest_fp = sorted(rows, key=lambda r: r["fp"], reverse=True)[:20]
    largest_fn = sorted(rows, key=lambda r: r["fn"], reverse=True)[:20]

    summary = {
        "model_name": name,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "num_evaluated": len(rows),
        "threshold": THRESHOLD,
        "image_size": IMAGE_SIZE,
        "device": str(device),
        "metrics": {
            "IoU": float(np.mean(ious)),
            "Dice": float(np.mean(dices)),
            "Precision": float(np.mean(precs)),
            "Recall": float(np.mean(recs)),
            "mean_inference_time_sec": float(np.mean(times)),
            "median_inference_time_sec": float(np.median(times)),
            "min_inference_time_sec": float(np.min(times)),
            "max_inference_time_sec": float(np.max(times)),
        },
        "counts": {
            "empty_GT_masks": gt_empty,
            "nonempty_GT_masks": gt_nonempty,
            "prediction_empty": pred_empty,
            "prediction_nonempty": pred_nonempty,
            "both_empty": both_empty,
            "GT_nonempty_prediction_empty": gt_ne_pred_e,
            "GT_empty_prediction_nonempty": gt_e_pred_ne,
        },
        "wall_time_sec": wall,
        "lowest_iou_20": [
            {"image_filename": r["image_filename"], "IoU": r["IoU"], "Dice": r["Dice"]}
            for r in lowest20
        ],
        "highest_iou_20": [
            {"image_filename": r["image_filename"], "IoU": r["IoU"], "Dice": r["Dice"]}
            for r in highest20
        ],
        "largest_false_positive_20": [
            {
                "image_filename": r["image_filename"],
                "fp": r["fp"],
                "gt_foreground_pixels": r["gt_foreground_pixels"],
                "predicted_foreground_pixels": r["predicted_foreground_pixels"],
                "IoU": r["IoU"],
            }
            for r in largest_fp
        ],
        "largest_false_negative_20": [
            {
                "image_filename": r["image_filename"],
                "fn": r["fn"],
                "gt_foreground_pixels": r["gt_foreground_pixels"],
                "predicted_foreground_pixels": r["predicted_foreground_pixels"],
                "IoU": r["IoU"],
            }
            for r in largest_fn
        ],
    }
    return summary, rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg: str):
        print(msg, flush=True)
        log_lines.append(msg)

    before_prot = {str(p): file_meta(p) for p in PROTECTED}
    (OUT_DIR / "protected_checkpoints_before.json").write_text(
        json.dumps(before_prot, indent=2), encoding="utf-8"
    )

    # Checkpoint integrity
    assert PROD_CKPT.is_file(), f"Missing production ckpt: {PROD_CKPT}"
    assert NEW_CKPT.is_file(), f"Missing archive-6 ckpt: {NEW_CKPT}"
    prod_sha = sha256_file(PROD_CKPT)
    new_sha = sha256_file(NEW_CKPT)
    log(f"PROD_SHA256={prod_sha}")
    log(f"NEW_SHA256={new_sha}")
    if new_sha != EXPECTED_NEW_SHA:
        raise RuntimeError(
            f"Archive-6 SHA256 mismatch.\n expected={EXPECTED_NEW_SHA}\n got={new_sha}"
        )
    log("Archive-6 SHA256 MATCH")

    # Verify load
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    _ = torch.zeros(1, device=device) + 1
    load_model(PROD_CKPT, device)
    load_model(NEW_CKPT, device)
    log("Both models load OK")

    pairs, missing_masks, extra_masks = pair_files(IMAGE_DIR, MASK_DIR)
    n_img = len([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    n_gt = len([f for f in os.listdir(MASK_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    dataset_audit = {
        "image_dir": str(IMAGE_DIR),
        "mask_dir": str(MASK_DIR),
        "total_Test_images": n_img,
        "total_GT_masks": n_gt,
        "paired": len(pairs),
        "missing_masks": len(missing_masks),
        "missing_images_for_masks": len(extra_masks),
        "missing_mask_samples": missing_masks[:10],
        "extra_mask_samples": extra_masks[:10],
    }
    log(f"DATASET {json.dumps(dataset_audit)}")
    if n_img != 4000 or n_gt != 4000 or len(pairs) != 4000 or missing_masks or extra_masks:
        raise RuntimeError(f"Official Test pairing failed: {dataset_audit}")

    eval_config = {
        "task": "official_test_comparison_resunet_production_vs_archive6",
        "protocol": "evaluation/evaluate_resunet.py identical",
        "threshold": THRESHOLD,
        "image_size": IMAGE_SIZE,
        "eps": EPS,
        "preprocessing": "BGR->RGB, resize 352, /255, ImageNet mean/std",
        "output_handling": "sigmoid -> resize to GT -> threshold 0.5",
        "empty_GT_included": True,
        "denom": 4000,
        "models": {
            "existing_production": str(PROD_CKPT),
            "archive6": str(NEW_CKPT),
        },
        "expected_archive6_sha256": EXPECTED_NEW_SHA,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "no_training": True,
        "no_threshold_tuning": True,
        "no_promotion": True,
    }
    (OUT_DIR / "evaluation_config.json").write_text(json.dumps(eval_config, indent=2), encoding="utf-8")
    (OUT_DIR / "model_hashes.json").write_text(
        json.dumps(
            {
                "production_resunet": {"path": str(PROD_CKPT), "sha256": prod_sha, **file_meta(PROD_CKPT)},
                "archive6_resunet": {
                    "path": str(NEW_CKPT),
                    "sha256": new_sha,
                    "expected_sha256": EXPECTED_NEW_SHA,
                    "match": new_sha == EXPECTED_NEW_SHA,
                    **file_meta(NEW_CKPT),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    prod_sum, _ = evaluate_model(
        "Existing_Production_ResUNet",
        PROD_CKPT,
        pairs,
        device,
        OUT_DIR / "per_image_metrics_production.csv",
    )
    new_sum, _ = evaluate_model(
        "Archive6_ResUNet",
        NEW_CKPT,
        pairs,
        device,
        OUT_DIR / "per_image_metrics_archive6.csv",
    )

    # Save worst/best lists for new model as dedicated files
    (OUT_DIR / "worst_iou_20_archive6.json").write_text(
        json.dumps(new_sum["lowest_iou_20"], indent=2), encoding="utf-8"
    )
    (OUT_DIR / "best_iou_20_archive6.json").write_text(
        json.dumps(new_sum["highest_iou_20"], indent=2), encoding="utf-8"
    )
    (OUT_DIR / "largest_fp_20_archive6.json").write_text(
        json.dumps(new_sum["largest_false_positive_20"], indent=2), encoding="utf-8"
    )
    (OUT_DIR / "largest_fn_20_archive6.json").write_text(
        json.dumps(new_sum["largest_false_negative_20"], indent=2), encoding="utf-8"
    )

    pm = prod_sum["metrics"]
    nm = new_sum["metrics"]

    def diff(a, b):
        d = b - a
        pct = (d / a * 100.0) if a != 0 else None
        return {"absolute": d, "percent_vs_existing": pct}

    comparison = {
        "IoU": {
            "existing": pm["IoU"],
            "archive6": nm["IoU"],
            **diff(pm["IoU"], nm["IoU"]),
        },
        "Dice": {
            "existing": pm["Dice"],
            "archive6": nm["Dice"],
            **diff(pm["Dice"], nm["Dice"]),
        },
        "Precision": {
            "existing": pm["Precision"],
            "archive6": nm["Precision"],
            **diff(pm["Precision"], nm["Precision"]),
        },
        "Recall": {
            "existing": pm["Recall"],
            "archive6": nm["Recall"],
            **diff(pm["Recall"], nm["Recall"]),
        },
        "mean_inference_time_sec": {
            "existing": pm["mean_inference_time_sec"],
            "archive6": nm["mean_inference_time_sec"],
            **diff(pm["mean_inference_time_sec"], nm["mean_inference_time_sec"]),
        },
        "median_inference_time_sec": {
            "existing": pm["median_inference_time_sec"],
            "archive6": nm["median_inference_time_sec"],
            **diff(pm["median_inference_time_sec"], nm["median_inference_time_sec"]),
        },
    }

    after_prot = {str(p): file_meta(p) for p in PROTECTED}
    integrity = {}
    for k in before_prot:
        b, a = before_prot[k], after_prot[k]
        if not b.get("exists") and not a.get("exists"):
            integrity[k] = {"unchanged": True, "note": "absent before and after"}
        elif b.get("exists") and a.get("exists"):
            integrity[k] = {
                "unchanged": b["mtime_ns"] == a["mtime_ns"] and b["size"] == a["size"],
                "before": b,
                "after": a,
            }
        else:
            integrity[k] = {"unchanged": False, "before": b, "after": a}

    (OUT_DIR / "protected_checkpoints_after.json").write_text(
        json.dumps(after_prot, indent=2), encoding="utf-8"
    )

    summary = {
        "status": "COMPLETED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_audit": dataset_audit,
        "production_model": prod_sum,
        "archive6_model": new_sum,
        "comparison": comparison,
        "reference_registry_resunet": {
            "IoU": 0.5710,
            "Dice": 0.6200,
            "Precision": 0.7616,
            "Recall": 0.7207,
            "inference": 0.0189,
            "note": "registry reference only; measured values above are authoritative for this run",
        },
        "production_checkpoint_integrity": integrity,
        "output_directory": str(OUT_DIR),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "dataset_counts.json").write_text(
        json.dumps(
            {
                "dataset": dataset_audit,
                "production_counts": prod_sum["counts"],
                "archive6_counts": new_sum["counts"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Human audit report
    report = f"""# Archive-6 ResUNet Official Test Evaluation Report

**Status:** EVALUATION ONLY — no training, no promotion, no production overwrite.

## Comparison table

| Metric | Existing ResUNet | Archive-6 ResUNet | Difference (Abs) | Diff % vs Existing |
|--------|-----------------:|------------------:|-----------------:|-------------------:|
| IoU | {pm['IoU']:.6f} | {nm['IoU']:.6f} | {comparison['IoU']['absolute']:+.6f} | {comparison['IoU']['percent_vs_existing']:+.3f}% |
| Dice | {pm['Dice']:.6f} | {nm['Dice']:.6f} | {comparison['Dice']['absolute']:+.6f} | {comparison['Dice']['percent_vs_existing']:+.3f}% |
| Precision | {pm['Precision']:.6f} | {nm['Precision']:.6f} | {comparison['Precision']['absolute']:+.6f} | {comparison['Precision']['percent_vs_existing']:+.3f}% |
| Recall | {pm['Recall']:.6f} | {nm['Recall']:.6f} | {comparison['Recall']['absolute']:+.6f} | {comparison['Recall']['percent_vs_existing']:+.3f}% |
| Mean infer (s) | {pm['mean_inference_time_sec']:.6f} | {nm['mean_inference_time_sec']:.6f} | {comparison['mean_inference_time_sec']['absolute']:+.6f} | {comparison['mean_inference_time_sec']['percent_vs_existing']:+.3f}% |
| Median infer (s) | {pm['median_inference_time_sec']:.6f} | {nm['median_inference_time_sec']:.6f} | {comparison['median_inference_time_sec']['absolute']:+.6f} | {comparison['median_inference_time_sec']['percent_vs_existing']:+.3f}% |

## Dataset audit
- Test images: {n_img}
- GT masks: {n_gt}
- Paired: {len(pairs)}
- Missing pairs: 0

## Checkpoint hashes
- Production: `{prod_sha}`
- Archive-6: `{new_sha}` (matches expected)

## Empty / nonempty (Archive-6)
{json.dumps(new_sum['counts'], indent=2)}

## Empty / nonempty (Production)
{json.dumps(prod_sum['counts'], indent=2)}

## Production integrity
{json.dumps(integrity, indent=2)}

## Output directory
`{OUT_DIR}`
"""
    (OUT_DIR / "audit_report.md").write_text(report, encoding="utf-8")
    (OUT_DIR / "console.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    log("SUMMARY IoU existing={:.6f} archive6={:.6f} diff={:+.6f}".format(
        pm["IoU"], nm["IoU"], comparison["IoU"]["absolute"]
    ))
    log(f"Wrote outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
