"""
End-to-end smoke: SINet-V2 → crop → ResNet50 (if classifier checkpoint exists).

ResUNet path is also probed but never modified.
"""

from __future__ import annotations

import os
import sys

import cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from inference.pipeline import CamouflageBreakerPipeline


def probe(segmenter: str, image_path: str):
    print("-" * 70)
    print("Segmenter:", segmenter)
    try:
        pipe = CamouflageBreakerPipeline(segmenter=segmenter)
    except Exception as exc:
        print("LOAD FAILED:", exc)
        return False

    result = pipe.predict(image_path)
    print("detected:", result["object_detected"])
    print("class:", result["class_name"])
    print("confidence:", result["confidence"])
    print("mask sum:", int(result["mask"].sum()) if result["mask"] is not None else None)
    print("crop:", None if result["crop"] is None else result["crop"].shape)
    return True


def main():
    test_dir = os.path.join(PROJECT_ROOT, "dataset", "Test", "Image")
    images = sorted(
        f for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    image_path = os.path.join(test_dir, images[0])
    print("Image:", image_path)

    ok_sinet = probe("sinetv2", image_path)
    ok_resunet = probe("resunet", image_path)
    print("=" * 70)
    print("SINet-V2 path:", "OK" if ok_sinet else "FAILED")
    print("ResUNet path:", "OK" if ok_resunet else "FAILED (checkpoint may be missing)")
    return 0 if ok_sinet else 1


if __name__ == "__main__":
    raise SystemExit(main())
