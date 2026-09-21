"""
Isolated ResUNet training run prepared for COD10K-v3 archive (6) manifest.

CRITICAL SAFETY:
- Sole data source: training_source_manifest.csv (Train/Val only)
- Official COD10K Test is NEVER read
- Production checkpoints are NEVER overwritten
- Default mode is PREPARE-ONLY (no training)

Enable training later with:
  python training/train_resunet_archive6_manifest.py --train
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.resunet import ResUNet
from utils.metrics import (
    BCEDiceLoss,
    dice_score,
    iou_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Fixed experiment paths / hyperparams (do not point at production best ckpts)
# ---------------------------------------------------------------------------

MANIFEST_PATH = Path(r"C:\Users\batka\Downloads\TUKU_COD10K_ARCHIVE_SOURCE\training_source_manifest.csv")
ARCHIVE_TRAIN_IMAGE = Path(r"C:\Users\batka\Downloads\TUKU_COD10K_ARCHIVE_SOURCE\COD10K-v3\Train\Image")
ARCHIVE_TRAIN_MASK = Path(r"C:\Users\batka\Downloads\TUKU_COD10K_ARCHIVE_SOURCE\COD10K-v3\Train\GT_Object")
# Explicitly unused (frozen Official Test — must never be opened by this script)
FORBIDDEN_TEST_IMAGE = Path(r"C:\Users\batka\Downloads\TUKU_COD10K_ARCHIVE_SOURCE\COD10K-v3\Test\Image")
FORBIDDEN_TEST_MASK = Path(r"C:\Users\batka\Downloads\TUKU_COD10K_ARCHIVE_SOURCE\COD10K-v3\Test\GT_Object")
FORBIDDEN_PROD_TEST = Path(r"C:\Users\batka\Downloads\TUKU DEEP LEARING\Camouflage_Breaker-main\dataset\Test")

EXP_DIR = PROJECT_ROOT / "outputs" / "archive6_resunet_manifest_s42"
CKPT_DIR = EXP_DIR / "checkpoints"
LOG_DIR = EXP_DIR / "logs"
RUN_CONFIG_PATH = EXP_DIR / "run_config.json"
PREPARE_STATUS_PATH = EXP_DIR / "prepare_status.json"

# Production paths that must never be written by this experiment
PROTECTED_CHECKPOINTS = [
    PROJECT_ROOT / "saved_models" / "resunet_best.pth",
    PROJECT_ROOT / "saved_models" / "sinetv2" / "sinetv2_cod10k_best.pth",
    PROJECT_ROOT / "saved_models" / "classifier_best.pth",
    PROJECT_ROOT / "saved_models" / "escnet_finetune" / "escnet_cod10k_best.pth",
]

SEED = 42
IMAGE_SIZE = 352
BATCH_SIZE = 4
NUM_WORKERS = 0
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-5
BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3
EARLY_STOP_PATIENCE = 5
SAVE_INTERVAL = 5
ENCODER_NAME = "resnet50"
ENCODER_WEIGHTS = "imagenet"  # fresh experiment init; does NOT load production best
BEST_CKPT_NAME = "resunet_archive6_manifest_best.pth"
SOURCE_TAG = "COD10K-v3-archive-6"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ManifestCOD10KDataset(Dataset):
    """Image/mask pairs from training_source_manifest.csv only (TRAIN or VAL)."""

    def __init__(self, rows: list[dict], image_size: int = IMAGE_SIZE):
        self.rows = rows
        self.image_size = image_size
        for r in self.rows:
            ip = ARCHIVE_TRAIN_IMAGE / r["image_filename"]
            mp = ARCHIVE_TRAIN_MASK / r["mask_filename"]
            if not ip.is_file():
                raise FileNotFoundError(f"Missing train image: {ip}")
            if not mp.is_file():
                raise FileNotFoundError(f"Missing train mask: {mp}")
            # Refuse any path that resolves under Test
            for p in (ip, mp):
                resolved = str(p.resolve()).lower()
                if "\\test\\" in resolved or "/test/" in resolved:
                    raise RuntimeError(f"REFUSED Test path in dataset: {p}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        ip = ARCHIVE_TRAIN_IMAGE / r["image_filename"]
        mp = ARCHIVE_TRAIN_MASK / r["mask_filename"]
        image = cv2.imread(str(ip))
        if image is None:
            raise RuntimeError(f"Failed to read image: {ip}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mp}")
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)
        image_t = torch.from_numpy(image.transpose(2, 0, 1))
        mask_t = torch.from_numpy(mask[None, ...])
        return image_t, mask_t


def load_manifest(path: Path) -> tuple[list[dict], list[dict]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    train_rows, val_rows = [], []
    for r in rows:
        split = (r.get("split") or "").strip().upper()
        src = (r.get("source") or "").strip()
        if src != SOURCE_TAG:
            raise RuntimeError(f"Unexpected source={src!r} (expected {SOURCE_TAG})")
        name = r["image_filename"]
        # Filename / notes must not indicate Test usage
        blob = f"{r['image_filename']}|{r['mask_filename']}|{r.get('notes','')}".lower()
        if "\\test\\" in blob or "/test/" in blob or blob.startswith("test/"):
            raise RuntimeError(f"Test leakage in manifest row: {r}")
        if split == "TRAIN":
            train_rows.append(r)
        elif split == "VAL":
            val_rows.append(r)
        else:
            raise RuntimeError(f"Invalid split={split!r} for {name}")
    return train_rows, val_rows


def assert_no_test_stem_overlap(train_rows: list[dict], val_rows: list[dict]) -> None:
    stems = {Path(r["image_filename"]).stem for r in train_rows + val_rows}
    if FORBIDDEN_TEST_IMAGE.exists():
        test_stems = {p.stem for p in FORBIDDEN_TEST_IMAGE.glob("*") if p.is_file()}
        overlap = stems & test_stems
        if overlap:
            raise RuntimeError(f"Manifest stems overlap Official Test: {sorted(overlap)[:10]}")


def assert_protected_not_targets() -> None:
    best = CKPT_DIR / BEST_CKPT_NAME
    for p in PROTECTED_CHECKPOINTS:
        if best.resolve() == p.resolve():
            raise RuntimeError(f"Experiment checkpoint collides with protected path: {p}")


def build_run_config(train_n: int, val_n: int) -> dict:
    return {
        "experiment_id": "archive6_resunet_manifest_s42",
        "status": "PREPARED_NOT_STARTED",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": "ResUNet",
            "architecture": "smp.Unet",
            "encoder": ENCODER_NAME,
            "encoder_weights_init": ENCODER_WEIGHTS,
            "base_checkpoint": None,
            "base_checkpoint_note": (
                "Fresh ImageNet encoder init. Does NOT load or overwrite "
                "saved_models/resunet_best.pth"
            ),
        },
        "dataset": {
            "manifest": str(MANIFEST_PATH),
            "image_root": str(ARCHIVE_TRAIN_IMAGE),
            "mask_root": str(ARCHIVE_TRAIN_MASK),
            "source": SOURCE_TAG,
            "train_count": train_n,
            "val_count": val_n,
            "seed": SEED,
            "test_forbidden": True,
            "test_paths_never_used": [
                str(FORBIDDEN_TEST_IMAGE),
                str(FORBIDDEN_TEST_MASK),
                str(FORBIDDEN_PROD_TEST),
            ],
        },
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "image_resolution": IMAGE_SIZE,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "Adam",
            "epochs": EPOCHS,
            "scheduler": {
                "name": "ReduceLROnPlateau",
                "mode": "max",
                "factor": SCHEDULER_FACTOR,
                "patience": SCHEDULER_PATIENCE,
                "monitored_metric": "val_IoU",
            },
            "loss": {
                "name": "BCEDiceLoss",
                "bce_weight": BCE_WEIGHT,
                "dice_weight": DICE_WEIGHT,
            },
            "early_stopping": {
                "metric": "val_IoU",
                "mode": "max",
                "patience": EARLY_STOP_PATIENCE,
            },
            "num_workers": NUM_WORKERS,
            "save_interval_epochs": SAVE_INTERVAL,
        },
        "outputs": {
            "experiment_dir": str(EXP_DIR),
            "checkpoint_dir": str(CKPT_DIR),
            "best_checkpoint": str(CKPT_DIR / BEST_CKPT_NAME),
            "logs_dir": str(LOG_DIR),
            "protected_production_checkpoints_never_written": [str(p) for p in PROTECTED_CHECKPOINTS],
        },
        "reproducibility": {
            "seed": SEED,
            "torch_manual_seed": SEED,
            "numpy_seed": SEED,
            "python_random_seed": SEED,
            "dataloader_generator_seed": SEED,
        },
        "training_enabled_by_default": False,
        "launch_when_approved": (
            f'python training/train_resunet_archive6_manifest.py --train'
        ),
    }


def prepare() -> dict:
    set_seed(SEED)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    assert_protected_not_targets()

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(MANIFEST_PATH)

    train_rows, val_rows = load_manifest(MANIFEST_PATH)
    if len(train_rows) != 5400 or len(val_rows) != 600:
        raise RuntimeError(
            f"Expected 5400 TRAIN / 600 VAL, got {len(train_rows)} / {len(val_rows)}"
        )
    assert_no_test_stem_overlap(train_rows, val_rows)

    # Write frozen split lists for audit
    (EXP_DIR / "train_files.txt").write_text(
        "\n".join(r["image_filename"] for r in train_rows) + "\n", encoding="utf-8"
    )
    (EXP_DIR / "val_files.txt").write_text(
        "\n".join(r["image_filename"] for r in val_rows) + "\n", encoding="utf-8"
    )

    # Build datasets to confirm paths resolve (read a few samples, not full epoch)
    train_ds = ManifestCOD10KDataset(train_rows, IMAGE_SIZE)
    val_ds = ManifestCOD10KDataset(val_rows, IMAGE_SIZE)
    assert len(train_ds) == 5400
    assert len(val_ds) == 600
    _ = train_ds[0]
    _ = val_ds[0]

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    cfg = build_run_config(len(train_ds), len(val_ds))
    cfg["dataloader_check"] = {
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "sample_train_image": train_rows[0]["image_filename"],
        "sample_val_image": val_rows[0]["image_filename"],
    }
    RUN_CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    status = {
        "ok": True,
        "mode": "PREPARE_ONLY",
        "train_count": 5400,
        "val_count": 600,
        "test_in_manifest": False,
        "manifest": str(MANIFEST_PATH),
        "run_config": str(RUN_CONFIG_PATH),
        "model": "ResUNet",
        "encoder_weights_init": ENCODER_WEIGHTS,
        "output_dir": str(EXP_DIR),
        "training_started": False,
    }
    PREPARE_STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return cfg


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    totals = {"loss": 0.0, "iou": 0.0, "dice": 0.0, "prec": 0.0, "rec": 0.0}
    n = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        probs = torch.sigmoid(logits)
        bi = iou_score(probs, masks)
        bd = dice_score(probs, masks)
        bp = precision_score(probs, masks)
        br = recall_score(probs, masks)
        totals["loss"] += float(loss.item())
        totals["iou"] += float(bi)
        totals["dice"] += float(bd)
        totals["prec"] += float(bp)
        totals["rec"] += float(br)
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{bi:.4f}")
    n = max(n, 1)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    totals = {"loss": 0.0, "iou": 0.0, "dice": 0.0, "prec": 0.0, "rec": 0.0}
    n = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        probs = torch.sigmoid(logits)
        bi = iou_score(probs, masks)
        bd = dice_score(probs, masks)
        bp = precision_score(probs, masks)
        br = recall_score(probs, masks)
        totals["loss"] += float(loss.item())
        totals["iou"] += float(bi)
        totals["dice"] += float(bd)
        totals["prec"] += float(bp)
        totals["rec"] += float(br)
        n += 1
    n = max(n, 1)
    return {k: v / n for k, v in totals.items()}


def run_train() -> None:
    """Only invoked with --train. Writes solely under EXP_DIR/checkpoints."""
    cfg = prepare()
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this ResUNet training run.")
    device = torch.device("cuda")

    train_rows, val_rows = load_manifest(MANIFEST_PATH)
    train_loader = DataLoader(
        ManifestCOD10KDataset(train_rows),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=torch.Generator().manual_seed(SEED),
        pin_memory=True,
    )
    val_loader = DataLoader(
        ManifestCOD10KDataset(val_rows),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = ResUNet(encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS).to(device)
    criterion = BCEDiceLoss(bce_weight=BCE_WEIGHT, dice_weight=DICE_WEIGHT)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )

    best_path = CKPT_DIR / BEST_CKPT_NAME
    # Hard refuse writing protected production files
    for p in PROTECTED_CHECKPOINTS:
        if best_path.resolve() == p.resolve():
            raise RuntimeError("Refusing to overwrite production checkpoint")

    best_iou = -1.0
    no_improve = 0
    history = []
    cfg["status"] = "TRAINING"
    cfg["training_started_at_utc"] = datetime.now(timezone.utc).isoformat()
    RUN_CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    for epoch in range(EPOCHS):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        va = validate(model, val_loader, criterion, device, epoch)
        scheduler.step(va["iou"])
        row = {
            "epoch": epoch,
            "train": tr,
            "val": va,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_iou={tr['iou']:.4f} val_iou={va['iou']:.4f} lr={row['lr']:.6f}"
        )
        if va["iou"] > best_iou:
            best_iou = va["iou"]
            no_improve = 0
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "metrics": {"iou": best_iou, "val_iou": best_iou, **{f"val_{k}": v for k, v in va.items()}},
                "experiment_id": "archive6_resunet_manifest_s42",
                "manifest": str(MANIFEST_PATH),
            }
            tmp = str(best_path) + ".tmp"
            torch.save(ckpt, tmp)
            os.replace(tmp, best_path)
            print(f"  [OK] Saved best -> {best_path}")
        else:
            no_improve += 1
            print(f"  [ ] No improvement ({no_improve}/{EARLY_STOP_PATIENCE})")
        if (epoch + 1) % SAVE_INTERVAL == 0:
            periodic = CKPT_DIR / f"resunet_archive6_epoch_{epoch+1}.pth"
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, periodic)
        (LOG_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if no_improve >= EARLY_STOP_PATIENCE:
            print("[STOP] Early stopping on val IoU.")
            break

    cfg["status"] = "COMPLETED"
    cfg["best_val_iou"] = best_iou
    cfg["best_checkpoint"] = str(best_path)
    RUN_CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        action="store_true",
        help="Actually start training. Without this flag, PREPARE ONLY.",
    )
    args = parser.parse_args()
    if args.train:
        run_train()
    else:
        prepare()
        print("\n[STOP] Preparation complete. Training NOT started.")
        print("Re-run with --train only after explicit approval.")


if __name__ == "__main__":
    main()
