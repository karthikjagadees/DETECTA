"""
Watch for ResUNet training completion, then run SINet → classifier → eval.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

RESUNET_LOG = os.path.join(PROJECT_ROOT, "logs", "resunet_train.log")
RESUNET_CKPT = os.path.join(PROJECT_ROOT, "saved_models", "resunet_best.pth")
MARKER = os.path.join(PROJECT_ROOT, "logs", "remaining_pipeline_started.flag")


def resunet_done() -> bool:
    if not os.path.isfile(RESUNET_LOG):
        return False
    try:
        with open(RESUNET_LOG, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return False
    if "Training Complete!" not in text and "Best checkpoint:" not in text:
        # also detect crash
        if "Traceback (most recent call last)" in text and "Epoch" in text:
            # might still be running after a warning traceback from PowerShell
            pass
        return False
    return os.path.isfile(RESUNET_CKPT)


def main():
    if os.path.isfile(MARKER):
        print("Remaining pipeline already started previously.")
        return

    print("Waiting for ResUNet training to finish...")
    while not resunet_done():
        # detect hard failure: process gone and no complete marker
        time.sleep(30)
        print(f"[watch] still waiting... ckpt={os.path.isfile(RESUNET_CKPT)}")

    print("ResUNet done. Starting remaining pipeline...")
    with open(MARKER, "w", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))

    log_path = os.path.join(PROJECT_ROOT, "logs", "remaining_pipeline.log")
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            [sys.executable, "-u", "training/run_remaining_after_resunet.py"],
            cwd=PROJECT_ROOT,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    if proc.returncode != 0:
        raise SystemExit(f"Remaining pipeline failed: see {log_path}")
    print("All remaining steps finished.")


if __name__ == "__main__":
    main()
