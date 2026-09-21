#!/usr/bin/env python3
"""
TUKU-v2 dataset ingestion QA utility (additive, audit-only by default).

Operates ONLY on TUKU_DATASET_V2 (or an explicit --root).
Never scans or writes the production COD10K dataset.

Usage:
  python tools/qa_dataset.py audit
  python tools/qa_dataset.py audit --root "C:\\path\\to\\TUKU_DATASET_V2"
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MASK_EXTS = {".png"}

MIN_WIDTH = 320
MIN_HEIGHT = 320
TINY_BBOX = 32
TINY_FG_RATIO = 0.0005  # 0.05%

# Hamming distance on 64-bit aHash; document threshold for near-dup flag
PHASH_NEAR_DUP_HAMMING = 5

REQUIRED_MANIFEST_FIELDS = [
    "sample_id",
    "image_filename",
    "mask_filename",
    "split",
    "source",
    "animal_class",
    "primary_camouflage",
    "environment",
    "annotation_status",
    "quality_flag",
]

SPLITS = ("train", "val", "hard_negative")


def default_dataset_root() -> Path:
    # tools/ -> TUKU_DATASET_V2/
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def average_hash64(rgb: np.ndarray) -> int:
    """
    Simple 8x8 average hash (aHash) as a 64-bit integer.
    Documented near-duplicate threshold: Hamming distance <= PHASH_NEAR_DUP_HAMMING.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    mean = float(small.mean())
    bits = (small >= mean).astype(np.uint8).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return int(value)


def hamming64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


# ---------------------------------------------------------------------------
# Pair discovery
# ---------------------------------------------------------------------------

@dataclass
class PairRef:
    split: str
    stem: str
    image_path: Optional[Path] = None
    mask_path: Optional[Path] = None


def list_by_stem(folder: Path, exts: Set[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not folder.is_dir():
        return out
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        stem = p.stem
        # Prefer first lexicographic path if collision
        if stem not in out:
            out[stem] = p
    return out


def discover_pairs(root: Path) -> List[PairRef]:
    pairs: List[PairRef] = []
    for split in SPLITS:
        img_dir = root / split / "images"
        mask_dir = root / split / "masks"
        imgs = list_by_stem(img_dir, IMAGE_EXTS)
        masks = list_by_stem(mask_dir, MASK_EXTS)
        stems = sorted(set(imgs) | set(masks))
        for stem in stems:
            pairs.append(
                PairRef(
                    split=split,
                    stem=stem,
                    image_path=imgs.get(stem),
                    mask_path=masks.get(stem),
                )
            )
    return pairs


# ---------------------------------------------------------------------------
# Per-pair audit
# ---------------------------------------------------------------------------

@dataclass
class PairResult:
    split: str
    stem: str
    image_path: Optional[str] = None
    mask_path: Optional[str] = None
    flags: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    width: Optional[int] = None
    height: Optional[int] = None
    mask_width: Optional[int] = None
    mask_height: Optional[int] = None
    mask_unique: Optional[List[int]] = None
    empty: Optional[bool] = None
    fg_pixels: Optional[int] = None
    fg_ratio: Optional[float] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    bbox_area_ratio: Optional[float] = None
    image_sha256: Optional[str] = None
    mask_sha256: Optional[str] = None
    phash: Optional[int] = None
    valid_pair: bool = False


def decode_image(path: Path) -> Optional[np.ndarray]:
    """Return RGB uint8 or None if corrupt."""
    try:
        # Prefer OpenCV for jpg/png; Pillow for webp fallback
        if path.suffix.lower() == ".webp":
            with Image.open(path) as im:
                return np.array(im.convert("RGB"))
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            with Image.open(path) as im:
                return np.array(im.convert("RGB"))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def decode_mask(path: Path) -> Optional[np.ndarray]:
    try:
        m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if m is None:
            with Image.open(path) as im:
                m = np.array(im)
        if m is None:
            return None
        if m.ndim == 3:
            m = m[:, :, 0]
        return m
    except Exception:
        return None


def compute_bbox(binary01: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(binary01 > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def audit_pair(ref: PairRef) -> PairResult:
    r = PairResult(split=ref.split, stem=ref.stem)
    if ref.image_path is None:
        r.errors.append("MISSING_IMAGE")
    else:
        r.image_path = str(ref.image_path)

    if ref.mask_path is None:
        r.errors.append("MISSING_MASK")
    else:
        r.mask_path = str(ref.mask_path)

    if ref.image_path is None or ref.mask_path is None:
        return r

    rgb = decode_image(ref.image_path)
    if rgb is None:
        r.errors.append("IMAGE_CORRUPT")
        return r
    r.height, r.width = int(rgb.shape[0]), int(rgb.shape[1])

    mask = decode_mask(ref.mask_path)
    if mask is None:
        r.errors.append("MASK_CORRUPT")
        return r
    r.mask_height, r.mask_width = int(mask.shape[0]), int(mask.shape[1])

    if (r.width, r.height) != (r.mask_width, r.mask_height):
        r.errors.append("DIMENSION_MISMATCH")

    if r.width is not None and r.height is not None:
        if r.width < MIN_WIDTH or r.height < MIN_HEIGHT:
            r.flags.append("LOW_RESOLUTION")

    uniq = sorted(int(v) for v in np.unique(mask).tolist())
    r.mask_unique = uniq
    uniq_set = set(uniq)

    if uniq_set <= {0, 255}:
        binary01 = (mask > 0).astype(np.uint8)
    elif uniq_set <= {0, 1}:
        r.flags.append("MASK_REQUIRES_NORMALIZATION")
        binary01 = (mask > 0).astype(np.uint8)
    else:
        r.errors.append("INVALID_MASK_VALUES")
        binary01 = (mask > 0).astype(np.uint8)

    r.fg_pixels = int(binary01.sum())
    area = int(binary01.size)
    r.fg_ratio = float(r.fg_pixels / area) if area else 0.0
    r.empty = r.fg_pixels == 0

    if ref.split == "hard_negative" and not r.empty:
        r.errors.append("HARD_NEGATIVE_NOT_EMPTY")

    bbox = compute_bbox(binary01)
    r.bbox = bbox
    if bbox is not None:
        _x, _y, bw, bh = bbox
        r.bbox_area_ratio = float((bw * bh) / area) if area else 0.0
        if bw < TINY_BBOX or bh < TINY_BBOX or r.fg_ratio < TINY_FG_RATIO:
            r.flags.append("TINY_OBJECT")

    try:
        r.image_sha256 = sha256_file(ref.image_path)
        r.mask_sha256 = sha256_file(ref.mask_path)
        r.phash = average_hash64(rgb)
    except Exception:
        r.errors.append("HASH_FAILED")

    r.valid_pair = len(r.errors) == 0
    return r


# ---------------------------------------------------------------------------
# Cross-pair / manifest checks
# ---------------------------------------------------------------------------

def find_exact_duplicates(results: Sequence[PairResult]) -> List[str]:
    by_hash: Dict[str, List[PairResult]] = defaultdict(list)
    for r in results:
        if r.image_sha256:
            by_hash[r.image_sha256].append(r)
    lines = []
    for h, group in by_hash.items():
        if len(group) < 2:
            continue
        labels = [f"{g.split}/{g.stem}" for g in group]
        lines.append(f"EXACT_DUPLICATE sha256={h[:12]}… files={labels}")
    return lines


def find_near_duplicates(results: Sequence[PairResult]) -> List[str]:
    items = [(r, r.phash) for r in results if r.phash is not None]
    lines = []
    for i in range(len(items)):
        ri, hi = items[i]
        for j in range(i + 1, len(items)):
            rj, hj = items[j]
            d = hamming64(hi, hj)
            if d <= PHASH_NEAR_DUP_HAMMING:
                # Skip if already exact same sha
                if ri.image_sha256 and ri.image_sha256 == rj.image_sha256:
                    continue
                lines.append(
                    f"POSSIBLE_NEAR_DUPLICATE hamming={d} "
                    f"{ri.split}/{ri.stem} <-> {rj.split}/{rj.stem}"
                )
    return lines


def check_manifest(root: Path) -> Tuple[List[str], List[str]]:
    """Return (errors, infos)."""
    path = root / "metadata" / "dataset_manifest.csv"
    errors: List[str] = []
    infos: List[str] = []
    if not path.is_file():
        errors.append("MANIFEST_MISSING: metadata/dataset_manifest.csv")
        return errors, infos

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            errors.append("MANIFEST_EMPTY_HEADER")
            return errors, infos
        missing_cols = [c for c in REQUIRED_MANIFEST_FIELDS if c not in reader.fieldnames]
        if missing_cols:
            errors.append(f"MANIFEST_MISSING_COLUMNS: {missing_cols}")

        rows = list(reader)
        infos.append(f"manifest_data_rows={len(rows)}")

        group_splits: Dict[str, Set[str]] = defaultdict(set)
        for i, row in enumerate(rows, start=2):
            status = (row.get("annotation_status") or "").strip().lower()
            # Only enforce required fields for accepted rows
            if status and status not in {"accepted", "accept"}:
                continue
            if not any((row.get(c) or "").strip() for c in REQUIRED_MANIFEST_FIELDS):
                # completely empty row
                continue
            for c in REQUIRED_MANIFEST_FIELDS:
                if not (row.get(c) or "").strip():
                    errors.append(f"MANIFEST_REQUIRED_EMPTY row={i} field={c}")
            gid = (row.get("group_id") or "").strip()
            split = (row.get("split") or "").strip()
            if gid and split:
                group_splits[gid].add(split)

        for gid, splits in group_splits.items():
            if "train" in splits and "val" in splits:
                errors.append(f"GROUP_LEAKAGE group_id={gid} splits={sorted(splits)}")

    return errors, infos


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(root: Path, results: Sequence[PairResult]) -> int:
    print("TUKU-V2 DATASET QA")
    print("=" * 50)
    print(f"Root: {root}")
    print(f"Near-dup threshold: aHash Hamming <= {PHASH_NEAR_DUP_HAMMING}")
    print(f"Min resolution: {MIN_WIDTH}x{MIN_HEIGHT}")
    print()

    def subset(split: str) -> List[PairResult]:
        return [r for r in results if r.split == split]

    for split in SPLITS:
        rs = subset(split)
        n_img = sum(1 for r in rs if r.image_path)
        n_mask = sum(1 for r in rs if r.mask_path)
        valid = sum(1 for r in rs if r.valid_pair)
        empty = sum(1 for r in rs if r.empty is True)
        nonempty = sum(1 for r in rs if r.empty is False)
        print(split.upper().replace("_", " "))
        print(f"  Images: {n_img}")
        print(f"  Masks: {n_mask}")
        print(f"  Stems seen: {len(rs)}")
        print(f"  Valid pairs: {valid}")
        print(f"  Empty: {empty}")
        print(f"  Non-empty: {nonempty}")
        print()

    missing_masks = sum(1 for r in results if "MISSING_MASK" in r.errors)
    missing_images = sum(1 for r in results if "MISSING_IMAGE" in r.errors)
    dim_mismatch = sum(1 for r in results if "DIMENSION_MISMATCH" in r.errors)
    invalid_masks = sum(1 for r in results if "INVALID_MASK_VALUES" in r.errors)
    hn_fg = sum(1 for r in results if "HARD_NEGATIVE_NOT_EMPTY" in r.errors)
    low_res = sum(1 for r in results if "LOW_RESOLUTION" in r.flags)
    tiny = sum(1 for r in results if "TINY_OBJECT" in r.flags)
    needs_norm = sum(1 for r in results if "MASK_REQUIRES_NORMALIZATION" in r.flags)
    corrupt_img = sum(1 for r in results if "IMAGE_CORRUPT" in r.errors)
    corrupt_mask = sum(1 for r in results if "MASK_CORRUPT" in r.errors)

    exact = find_exact_duplicates(results)
    near = find_near_duplicates(results)
    man_err, man_info = check_manifest(root)

    print("ERRORS / FLAGS")
    print("-" * 50)
    print(f"Missing masks: {missing_masks}")
    print(f"Missing images: {missing_images}")
    print(f"Dimension mismatch: {dim_mismatch}")
    print(f"Invalid masks: {invalid_masks}")
    print(f"Corrupt images: {corrupt_img}")
    print(f"Corrupt masks: {corrupt_mask}")
    print(f"Hard-negative foreground: {hn_fg}")
    print(f"Low-resolution images: {low_res}")
    print(f"Tiny objects: {tiny}")
    print(f"Masks needing {{0,1}}->{{0,255}} norm (flag only): {needs_norm}")
    print(f"Exact duplicates: {len(exact)}")
    for line in exact:
        print(f"  {line}")
    print(f"Near duplicates: {len(near)}")
    for line in near[:50]:
        print(f"  {line}")
    if len(near) > 50:
        print(f"  … {len(near) - 50} more")
    print(f"Manifest info: {'; '.join(man_info) if man_info else 'n/a'}")
    print(f"Manifest / group problems: {len(man_err)}")
    for line in man_err:
        print(f"  {line}")

    # Pair-level error dump (compact)
    bad = [r for r in results if r.errors]
    if bad:
        print()
        print("PAIR ISSUES")
        print("-" * 50)
        for r in bad[:100]:
            print(f"  {r.split}/{r.stem}: {', '.join(r.errors + r.flags)}")
        if len(bad) > 100:
            print(f"  … {len(bad) - 100} more")

    hard_errors = (
        missing_masks
        + missing_images
        + dim_mismatch
        + invalid_masks
        + hn_fg
        + corrupt_img
        + corrupt_mask
        + len(exact)
        + len(man_err)
    )
    # Near-duplicates are warnings (do not alone force FAIL on empty set)
    # Spec asks overall PASS/FAIL — treat near-dups as FAIL only if present with content
    if near:
        hard_errors += 0  # informational for now; still listed above
        # Spec: report near-duplicates; for empty dataset PASS. With data, near-dups = FAIL soft?
        # User said FINAL STATUS PASS or FAIL. Treat EXACT dup + structural as FAIL;
        # near-dups also FAIL to be safe when any exist.
        hard_errors += len(near)

    print()
    print("FINAL STATUS:")
    if hard_errors == 0:
        print("PASS")
        return 0
    print("FAIL")
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_audit(root: Path) -> int:
    root = root.resolve()
    if not root.is_dir():
        print(f"ERROR: dataset root not found: {root}", file=sys.stderr)
        return 2

    # Soft guard: refuse obvious production dataset path
    prod_markers = {"Train", "Test", "GT_Object"}
    names = {p.name for p in root.iterdir()} if root.is_dir() else set()
    if prod_markers.issubset(names) and "train" not in names:
        print(
            "ERROR: Refusing to audit a path that looks like the production "
            "COD10K dataset (Train/Test). Point --root at TUKU_DATASET_V2.",
            file=sys.stderr,
        )
        return 2

    pairs = discover_pairs(root)
    results = [audit_pair(p) for p in pairs]
    return print_report(root, results)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TUKU-v2 dataset QA (audit-only; no file modifications)."
    )
    sub = p.add_subparsers(dest="command", required=True)
    audit_p = sub.add_parser(
        "audit", help="Read-only QA audit of train/val/hard_negative"
    )
    audit_p.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Dataset root (default: parent of tools/, i.e. TUKU_DATASET_V2)",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.root if getattr(args, "root", None) is not None else default_dataset_root()
    if args.command == "audit":
        return cmd_audit(root)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
