"""
READ-ONLY QA audit for EXTERNAL_TEST_24 ground-truth masks.

Does NOT modify masks, images, models, or datasets.
Does NOT run model evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

DEFAULT_ROOT = Path(
    r"C:\Users\batka\Downloads\TUKU DEEP LEARING\Camouflage_Breaker-main\outputs\external_test_24_annotation"
)
EXPECTED_N = 24
EXPECTED_W, EXPECTED_H = 1536, 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(root: Path) -> dict:
    images_dir = root / "images"
    masks_dir = root / "masks"
    manifest_path = root / "metadata" / "annotation_manifest.csv"

    images = sorted(images_dir.glob("*.png"))
    masks = sorted(masks_dir.glob("*.png")) if masks_dir.is_dir() else []

    image_by_name = {p.name: p for p in images}
    mask_by_name = {p.name: p for p in masks}

    image_hashes = {}
    mask_hashes = {}
    hash_to_images = defaultdict(list)
    hash_to_masks = defaultdict(list)

    per_image = []
    invalid = []
    empty_masks = []
    missing_masks = []
    orphan_masks = []
    fg_pcts = []

    for name, ipath in image_by_name.items():
        isha = sha256_file(ipath)
        image_hashes[name] = isha
        hash_to_images[isha].append(name)

        im = cv2.imread(str(ipath), cv2.IMREAD_COLOR)
        entry = {
            "image": name,
            "image_exists": True,
            "image_sha256": isha,
            "image_decode_ok": im is not None,
            "image_width": None if im is None else im.shape[1],
            "image_height": None if im is None else im.shape[0],
            "mask_exists": name in mask_by_name,
            "mask_status": "MISSING",
        }
        if im is not None and (im.shape[1], im.shape[0]) != (EXPECTED_W, EXPECTED_H):
            entry["image_dim_warning"] = f"{im.shape[1]}x{im.shape[0]}"

        mpath = mask_by_name.get(name)
        if mpath is None:
            missing_masks.append(name)
            per_image.append(entry)
            continue

        msha = sha256_file(mpath)
        mask_hashes[name] = msha
        hash_to_masks[msha].append(name)
        m = cv2.imread(str(mpath), cv2.IMREAD_UNCHANGED)
        entry["mask_sha256"] = msha
        entry["mask_decode_ok"] = m is not None
        if m is None:
            entry["mask_status"] = "CORRUPT"
            invalid.append({"image": name, "reason": "corrupt_or_unreadable"})
            per_image.append(entry)
            continue

        if m.ndim == 3:
            entry["mask_status"] = "INVALID_RGB_OR_MULTI"
            invalid.append({"image": name, "reason": f"ndim={m.ndim} shape={m.shape}"})
            per_image.append(entry)
            continue

        if m.dtype != np.uint8:
            entry["mask_status"] = "INVALID_DTYPE"
            invalid.append({"image": name, "reason": f"dtype={m.dtype}"})
            per_image.append(entry)
            continue

        h, w = m.shape[:2]
        entry["mask_width"] = w
        entry["mask_height"] = h
        if (w, h) != (EXPECTED_W, EXPECTED_H):
            entry["mask_status"] = "INVALID_DIMS"
            invalid.append({"image": name, "reason": f"dims={w}x{h}"})
            per_image.append(entry)
            continue

        if im is not None and (im.shape[1], im.shape[0]) != (w, h):
            entry["mask_status"] = "DIM_MISMATCH_WITH_IMAGE"
            invalid.append({"image": name, "reason": "image/mask dim mismatch"})
            per_image.append(entry)
            continue

        uniq = sorted(int(x) for x in np.unique(m))
        entry["mask_unique_values"] = uniq
        if uniq not in ([0], [255], [0, 255]):
            entry["mask_status"] = "INVALID_VALUES"
            invalid.append({"image": name, "reason": f"unique={uniq}"})
            per_image.append(entry)
            continue

        fg = int((m == 255).sum())
        pct = 100.0 * fg / (w * h)
        entry["foreground_pixels"] = fg
        entry["foreground_percent"] = pct
        ys, xs = np.where(m == 255)
        if len(xs):
            entry["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        else:
            entry["bbox"] = None
            empty_masks.append(name)
        num_labels, _ = cv2.connectedComponents((m == 255).astype(np.uint8))
        entry["connected_components"] = int(num_labels - 1)
        entry["tiny_object_warning"] = bool(0 < fg < 50)
        entry["mask_status"] = "OK"
        fg_pcts.append(pct)
        per_image.append(entry)

    for name in mask_by_name:
        if name not in image_by_name:
            orphan_masks.append(name)

    dup_images = {h: names for h, names in hash_to_images.items() if len(names) > 1}
    dup_masks = {h: names for h, names in hash_to_masks.items() if len(names) > 1}

    # manifest statuses
    status_counts = Counter()
    review_counts = Counter()
    if manifest_path.is_file():
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                status_counts[row.get("annotation_status") or "UNANNOTATED"] += 1
                review_counts[row.get("review_status") or "PENDING"] += 1

    def pct_stats(vals):
        if not vals:
            return {"mean": None, "median": None, "min": None, "max": None, "n": 0}
        a = np.array(vals, dtype=np.float64)
        return {
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "n": int(len(a)),
        }

    report = {
        "audit_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "read_only": True,
        "expected_images": EXPECTED_N,
        "found_images": len(images),
        "found_masks": len(masks),
        "image_count_ok": len(images) == EXPECTED_N,
        "paired_ok_masks": sum(1 for e in per_image if e.get("mask_status") == "OK"),
        "missing_masks": missing_masks,
        "orphan_masks": orphan_masks,
        "empty_masks": empty_masks,
        "invalid_masks": invalid,
        "duplicate_image_hashes": dup_images,
        "duplicate_mask_hashes": dup_masks,
        "foreground_percent_stats_completed_ok_masks_only": pct_stats(fg_pcts),
        "annotation_status_counts": dict(status_counts),
        "review_status_counts": dict(review_counts),
        "per_image": per_image,
        "notes": [
            "Tiny foreground is allowed for hard camouflage and is not auto-rejected.",
            "This script never modifies masks.",
            "Do not run model evaluation until explicitly authorized.",
        ],
    }
    return report


def main():
    ap = argparse.ArgumentParser(description="READ-ONLY QA for EXTERNAL_TEST_24 masks")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path (default: metadata/qa_audit.json under root)",
    )
    args = ap.parse_args()
    root = args.root
    report = audit(root)
    out = args.out or (root / "metadata" / "qa_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # also refresh annotation_status.md skeleton numbers from live data
    status_path = root / "reports" / "annotation_status.md"
    sc = report["annotation_status_counts"]
    rc = report["review_status_counts"]
    fg = report["foreground_percent_stats_completed_ok_masks_only"]
    md = f"""# EXTERNAL_TEST_24 Annotation Status

**Updated (UTC):** {report['audit_utc']}  
**Mode:** annotation / QA only — no training, no model evaluation

> Ground-truth masks must be independently annotated. Model predictions are reference only.

## Counts

| Metric | Value |
|--------|------:|
| Total images expected | 24 |
| Images present | {report['found_images']} |
| Masks present | {report['found_masks']} |
| Valid paired masks (OK) | {report['paired_ok_masks']} |
| Missing masks | {len(report['missing_masks'])} |
| Invalid masks | {len(report['invalid_masks'])} |
| Empty masks (valid binary, 0 FG) | {len(report['empty_masks'])} |
| Orphan masks | {len(report['orphan_masks'])} |

## Annotation status (from manifest)

| Status | Count |
|--------|------:|
| UNANNOTATED | {sc.get('UNANNOTATED', 0)} |
| IN_PROGRESS | {sc.get('IN_PROGRESS', 0)} |
| COMPLETED | {sc.get('COMPLETED', 0)} |
| AMBIGUOUS | {sc.get('AMBIGUOUS', 0)} |
| REQUIRES_REVIEW | {sc.get('REQUIRES_REVIEW', 0)} |

## Review status

| Status | Count |
|--------|------:|
| PENDING | {rc.get('PENDING', 0)} |
| PASSED | {rc.get('PASSED', 0)} |
| NEEDS_CORRECTION | {rc.get('NEEDS_CORRECTION', 0)} |

## Foreground % (valid OK masks only)

| Stat | Value |
|------|-------|
| n | {fg['n']} |
| min | {fg['min']} |
| max | {fg['max']} |
| mean | {fg['mean']} |
| median | {fg['median']} |

_If n=0, no completed valid masks exist yet — values are null by design._

## Duplicate hashes

- Duplicate image SHA groups: {len(report['duplicate_image_hashes'])}
- Duplicate mask SHA groups: {len(report['duplicate_mask_hashes'])}

## Held-out reminder

These images remain **EXTERNAL_TEST_24**. Do not copy into TUKU_DATASET_V2 or COD10K. Do not train/fine-tune/threshold-tune on them. Do not evaluate models until explicitly authorized.

## QA command

```bash
python tools/qa_external_test_masks.py
```
"""
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(md, encoding="utf-8")

    print(json.dumps({
        "images": report["found_images"],
        "masks": report["found_masks"],
        "ok_masks": report["paired_ok_masks"],
        "missing": len(report["missing_masks"]),
        "invalid": len(report["invalid_masks"]),
        "out": str(out),
        "status_md": str(status_path),
    }, indent=2))


if __name__ == "__main__":
    main()
