"""
Run after ResUNet training completes:
  1) SINet-V2 full GPU training
  2) ResNet50 classifier (Train crops only)
  3) Official Test evaluation + E2E + metadata

CUDA is required; CPU fallback is refused.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def require_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required but unavailable.")
    t = torch.zeros(1, device="cuda")
    _ = t + 1
    name = torch.cuda.get_device_name(0)
    print(f"[OK] CUDA: {name} | torch={torch.__version__} | cuda={torch.version.cuda}")
    return {
        "pytorch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda": torch.version.cuda,
        "gpu": name,
        "capability": list(torch.cuda.get_device_capability(0)),
    }


def run_step(name: str, cmd: list[str], env: dict | None = None):
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    print(f"\n{'=' * 70}\nSTART: {name}\nCMD: {' '.join(cmd)}\nLOG: {log_path}\n{'=' * 70}")
    merged = os.environ.copy()
    if env:
        merged.update(env)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=merged,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Step {name} failed with code {proc.returncode}. See {log_path}")
    print(f"[OK] {name} finished in {elapsed / 60:.1f} min")
    return elapsed


def ensure_train_crops():
    crop_dir = os.path.join(PROJECT_ROOT, "dataset", "crops", "Train")
    n = 0
    if os.path.isdir(crop_dir):
        n = len([f for f in os.listdir(crop_dir) if f.endswith("_crop.jpg")])
    if n >= 5000:
        print(f"[OK] Train crops already present: {n}")
        return
    print(f"[INFO] Train crops count={n}; generating...")
    from utils.generate_crops import generate_classification_dataset

    generate_classification_dataset(
        root_dir="dataset",
        split="Train",
        output_dir="dataset/crops",
    )


def main():
    started = datetime.now(timezone.utc).isoformat()
    cuda_info = require_cuda()

    resunet_ckpt = os.path.join(PROJECT_ROOT, "saved_models", "resunet_best.pth")
    if not os.path.isfile(resunet_ckpt):
        raise FileNotFoundError(f"ResUNet best checkpoint missing: {resunet_ckpt}")

    times = {}

    # SINet-V2 full training (clear any leftover env smoke limits)
    env = {
        "SINET_EPOCHS": "20",
        "SINET_BATCH_SIZE": "8",
        "SINET_MAX_SAMPLES": "0",
        "PYTHONUNBUFFERED": "1",
    }
    times["sinetv2_sec"] = run_step(
        "sinetv2_train",
        [sys.executable, "-u", "training/train_sinetv2.py"],
        env=env,
    )

    ensure_train_crops()
    times["classifier_sec"] = run_step(
        "classifier_train",
        [sys.executable, "-u", "training/train_classifier_final.py"],
        env={"PYTHONUNBUFFERED": "1"},
    )

    times["eval_sec"] = run_step(
        "final_eval",
        [sys.executable, "-u", "evaluation/run_final_evaluation.py"],
        env={"PYTHONUNBUFFERED": "1"},
    )

    meta = {
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        **cuda_info,
        "random_seed": 42,
        "dataset": {
            "name": "COD10K-v3",
            "train_images": 6000,
            "val_ratio": 0.1,
            "approx_train_subset": 5400,
            "approx_val": 600,
            "official_test": 4000,
            "test_held_out": True,
        },
        "step_times_sec": times,
        "checkpoints": {
            "resunet": "saved_models/resunet_best.pth",
            "sinetv2": "saved_models/sinetv2/sinetv2_cod10k_best.pth",
            "classifier": "saved_models/classifier_best.pth",
            "class_mapping": "saved_models/class_mapping.json",
        },
    }
    with open(os.path.join(OUT_DIR, "training_run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("\n[OK] Remaining pipeline complete.")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
