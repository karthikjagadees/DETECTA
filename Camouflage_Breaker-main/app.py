"""
DETECTA (premium Streamlit dashboard)

Upload → ResUNet + SINet-V2 + ESCNet → comparison → selected detection
      → crop → ResNet50 + OpenCLIP refinement (69 COD classes) → animal info

Official Test metrics: static registry (N=4000). No Test-set I/O.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from inference.pipeline import CamouflageBreakerPipeline

PROJECT_ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = PROJECT_ROOT / "config" / "model_registry.json"
ANIMAL_INFO_PATH = PROJECT_ROOT / "saved_models" / "animal_info.json"
THRESHOLD = 0.5

SEGMENTER_ORDER = ("ResUNet", "SINet-V2", "ESCNet")
NAME_TO_KEY = {
    "ResUNet": "resunet",
    "SINet-V2": "sinetv2",
    "ESCNet": "escnet",
}
MODEL_COLOR = {
    "ResUNet": "#2196FF",
    "SINet-V2": "#9B5CFF",
    "ESCNet": "#19E68C",
}

st.set_page_config(
    page_title="DETECTA",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --bg0: #050B14;
  --bg1: #07111F;
  --panel: rgba(12, 20, 34, 0.78);
  --text: #E8EEF6;
  --muted: #8FA3BB;
  --blue: #2196FF;
  --cyan: #00D9FF;
  --purple: #9B5CFF;
  --green: #19E68C;
  --warn: #E0B35A;
  --line: rgba(120, 160, 210, 0.16);
  --glow: rgba(33, 150, 255, 0.18);
}

html, body, .stApp {
  background:
    radial-gradient(980px 460px at 6% -12%, rgba(33,150,255,0.14), transparent 58%),
    radial-gradient(820px 420px at 96% 0%, rgba(155,92,255,0.10), transparent 55%),
    radial-gradient(700px 380px at 70% 18%, rgba(0,217,255,0.05), transparent 50%),
    linear-gradient(180deg, var(--bg0), var(--bg1) 42%, #060e1a);
  color: var(--text);
  font-family: 'IBM Plex Sans', sans-serif;
}

.block-container {
  padding-top: 0.65rem !important;
  padding-bottom: 0.85rem !important;
  max-width: 1480px;
  padding-left: 1.4rem !important;
  padding-right: 1.4rem !important;
}

section[data-testid="stSidebar"] {
  background:
    linear-gradient(180deg, #060d18 0%, #081321 100%);
  border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--panel);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  box-shadow: 0 0 0 1px rgba(255,255,255,0.02), 0 10px 28px rgba(0,0,0,0.28);
  padding: 0.05rem !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div { gap: 0.4rem !important; }
div[data-testid="column"] > div[data-testid="stVerticalBlockBorderWrapper"] { height: 100%; }

h1,h2,h3,.brand,.hero-h,.sum-val,.pred-name,.agree-big {
  font-family: 'Space Grotesk', sans-serif;
}

/* Sidebar */
.sb-brand { font-size: 1.45rem; font-weight: 700; color: #f4f7fb; letter-spacing: 0.02em; }
.sb-tag {
  font-size: 0.66rem; letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--cyan); margin: 0.2rem 0 1rem 0; font-weight: 600;
}
.sb-item {
  display: flex; align-items: center; gap: 0.55rem;
  padding: 0.42rem 0.6rem; margin: 0.18rem 0; border-radius: 8px;
  color: var(--muted); font-size: 0.86rem; border: 1px solid transparent;
}
.sb-item.active {
  color: var(--text);
  background: rgba(33,150,255,0.12);
  border-color: rgba(33,150,255,0.28);
  box-shadow: 0 0 16px rgba(33,150,255,0.12);
}
.sb-ico {
  width: 1.35rem; height: 1.35rem; border-radius: 50%;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 0.7rem; border: 1px solid var(--line); background: rgba(255,255,255,0.03);
}
.sb-foot {
  margin-top: 1.4rem; padding-top: 0.85rem; border-top: 1px solid var(--line);
  color: var(--muted); font-size: 0.74rem; line-height: 1.45;
}
.sb-foot b { color: var(--text); font-size: 0.84rem; }
.sb-note { margin-top: 0.7rem; font-size: 0.7rem; color: #6f849c; line-height: 1.35; }

/* Hero */
.hero {
  position: relative; overflow: hidden;
  display: flex; justify-content: space-between; gap: 1.2rem; flex-wrap: wrap;
  align-items: flex-end;
  margin: 0 0 0.75rem 0; padding: 1.05rem 1.2rem 1.1rem;
  border: 1px solid var(--line); border-radius: 14px;
  background:
    linear-gradient(115deg, rgba(8,18,34,0.92) 0%, rgba(10,22,40,0.72) 48%, rgba(8,16,30,0.55) 100%),
    repeating-linear-gradient(118deg, rgba(33,150,255,0.045) 0 10px, transparent 10px 22px),
    repeating-linear-gradient(62deg, rgba(155,92,255,0.035) 0 8px, transparent 8px 20px),
    radial-gradient(420px 220px at 88% 40%, rgba(25,230,140,0.08), transparent 70%);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 12px 36px rgba(0,0,0,0.35);
}
.hero::after {
  content: ""; position: absolute; inset: auto -10% -40% 55%; height: 140%;
  background:
    radial-gradient(ellipse at center, rgba(0,217,255,0.07), transparent 62%),
    radial-gradient(ellipse at 70% 40%, rgba(33,150,255,0.10), transparent 55%);
  pointer-events: none;
}
.brand { font-size: 2.55rem; font-weight: 700; margin: 0; line-height: 0.92; color: #f5f8fc; position: relative; z-index: 1; }
.deep {
  font-size: 0.72rem; letter-spacing: 0.22em; text-transform: uppercase;
  color: var(--cyan); font-weight: 600; margin: 0.35rem 0 0.55rem; position: relative; z-index: 1;
}
.hero-h { font-size: 1.55rem; font-weight: 600; margin: 0; color: #f0f4fa; position: relative; z-index: 1; }
.hero-sub { color: var(--muted); margin: 0.3rem 0 0; font-size: 0.92rem; max-width: 34rem; position: relative; z-index: 1; }
.hero-quote {
  margin-top: 0.55rem; font-size: 0.78rem; color: #a9bdd4; letter-spacing: 0.04em;
  border-left: 2px solid rgba(0,217,255,0.45); padding-left: 0.55rem; position: relative; z-index: 1;
}
.status-grid {
  display: grid; grid-template-columns: repeat(3, minmax(6.8rem, 1fr)); gap: 0.45rem;
  position: relative; z-index: 1;
}
.st-card {
  min-width: 6.8rem; padding: 0.55rem 0.65rem; border-radius: 10px;
  border: 1px solid var(--line); background: rgba(5,12,24,0.72);
  box-shadow: 0 0 18px rgba(33,150,255,0.08);
}
.st-card .k { font-size: 0.62rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }
.st-card .v { font-family: 'Space Grotesk', sans-serif; font-size: 0.92rem; font-weight: 700; margin-top: 0.2rem; color: #f2f6fb; }

/* Summary */
.sum-wrap { min-height: 5.6rem; }
.sum-ring {
  width: 1.7rem; height: 1.7rem; border-radius: 50%;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 0.72rem; font-weight: 700; margin-bottom: 0.28rem;
  border: 1.5px solid; background: rgba(255,255,255,0.03);
}
.sum-label { font-size: 0.64rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
.sum-val { font-size: 1.55rem; font-weight: 700; margin: 0.08rem 0; line-height: 1.1; min-height: 1.7rem; display: flex; align-items: center; }
.sum-detail { color: var(--muted); font-size: 0.76rem; line-height: 1.25; }

.band {
  font-size: 0.66rem; letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--muted); margin: 0.7rem 0 0.35rem; font-family: 'Space Grotesk', sans-serif;
}
.sec-h { font-family: 'Space Grotesk', sans-serif; font-size: 0.98rem; font-weight: 600; margin: 0 0 0.12rem; }
.sec-s { color: var(--muted); font-size: 0.76rem; margin: 0 0 0.45rem; }

.note {
  color: var(--muted); font-size: 0.73rem; line-height: 1.35;
  border-left: 2px solid var(--cyan); padding-left: 0.5rem; margin: 0.1rem 0 0.4rem;
}
.note b { color: #c9d7e8; font-weight: 600; }

.badge {
  display: inline-block; font-size: 0.64rem; letter-spacing: 0.08em; text-transform: uppercase;
  border: 1px solid rgba(224,179,90,0.45); color: var(--warn);
  border-radius: 5px; padding: 0.12rem 0.4rem; margin-bottom: 0.3rem;
}

.upload-zone {
  border: 1.5px dashed rgba(0,217,255,0.35); border-radius: 11px;
  padding: 0.85rem 0.7rem; text-align: center;
  background: linear-gradient(180deg, rgba(33,150,255,0.08), rgba(0,217,255,0.03));
  margin-bottom: 0.35rem; box-shadow: inset 0 0 24px rgba(33,150,255,0.05);
}
.upload-zone .t { font-weight: 600; font-size: 0.88rem; }
.upload-zone .s { color: var(--muted); font-size: 0.74rem; margin-top: 0.2rem; }

.mhead {
  display: flex; align-items: center; gap: 0.4rem;
  padding: 0.32rem 0.5rem; border-radius: 8px; margin-bottom: 0.35rem;
  border: 1px solid; font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 0.9rem;
}
.dot { width: 0.5rem; height: 0.5rem; border-radius: 50%; display: inline-block; margin-right: 0.25rem; }

.ok { color: var(--green); font-weight: 600; font-size: 0.86rem; }
.warn { color: var(--warn); font-weight: 600; font-size: 0.86rem; }

.kv {
  display: flex; justify-content: space-between; gap: 0.6rem;
  padding: 0.2rem 0; border-bottom: 1px solid rgba(255,255,255,0.045); font-size: 0.8rem;
}
.kv span:last-child { color: var(--cyan); font-weight: 600; }

.pred-box {
  margin: 0.15rem 0 0.45rem; padding: 0.55rem 0.65rem; border-radius: 10px;
  border: 1px solid rgba(25,230,140,0.28);
  background: linear-gradient(135deg, rgba(25,230,140,0.10), rgba(33,150,255,0.04));
  box-shadow: 0 0 22px rgba(25,230,140,0.08);
}
.pred-name { font-size: 2.15rem; font-weight: 700; margin: 0; line-height: 1.02; letter-spacing: 0.01em; }
.pred-pct { font-family: 'Space Grotesk', sans-serif; font-size: 1.9rem; font-weight: 700; color: var(--green); line-height: 1; margin-top: 0.2rem; }
.pred-lab { font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-top: 0.12rem; }

.card-accent { border-radius: 12px; }

.bar-wrap { margin: 0.25rem 0 0.4rem; }
.bar-row {
  display: grid; grid-template-columns: 6.2rem 1fr 3.5rem; gap: 0.45rem;
  align-items: center; margin: 0.38rem 0;
}
.bar-name { font-size: 0.86rem; font-weight: 700; }
.bar-track {
  height: 0.78rem; border-radius: 999px; background: rgba(255,255,255,0.06);
  overflow: hidden; border: 1px solid rgba(255,255,255,0.05);
}
.bar-fill { height: 100%; border-radius: 999px; box-shadow: 0 0 10px rgba(255,255,255,0.08); }
.bar-val { font-size: 0.84rem; font-weight: 700; text-align: right; font-variant-numeric: tabular-nums; }

.agree-big { font-size: 2.2rem; font-weight: 700; color: var(--green); line-height: 1; }
.agree-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 0.38rem 0; border-bottom: 1px solid rgba(255,255,255,0.045); font-size: 0.84rem;
}

.insight-row {
  display: grid; grid-template-columns: 2.15rem 1fr; gap: 0.5rem;
  padding: 0.38rem 0; border-bottom: 1px solid rgba(255,255,255,0.045);
}
.insight-num {
  width: 1.7rem; height: 1.7rem; border-radius: 50%;
  display: inline-flex; align-items: center; justify-content: center;
  font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 0.72rem;
  color: var(--cyan); border: 1px solid rgba(0,217,255,0.35);
  background: rgba(0,217,255,0.08);
}
.insight-text { font-size: 0.86rem; line-height: 1.35; padding-top: 0.15rem; }

footer.foot {
  color: var(--muted); text-align: center; margin: 1.1rem 0 0.3rem;
  font-size: 0.78rem; border-top: 1px solid var(--line); padding-top: 0.8rem; line-height: 1.45;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# ML helpers (behaviour unchanged)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_registry() -> dict:
    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="Loading segmentation + classifier models…")
def load_pipeline(segmenter_key: str):
    return CamouflageBreakerPipeline(segmenter=segmenter_key)


@st.cache_data(show_spinner=False)
def load_animal_info() -> dict:
    if not ANIMAL_INFO_PATH.is_file():
        return {}
    with ANIMAL_INFO_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def empty_result(segmenter_key: str, error: str | None = None) -> dict:
    return {
        "segmenter_key": segmenter_key,
        "original_bgr": None,
        "mask": None,
        "outline": None,
        "overlay": None,
        "crop": None,
        "object_detected": False,
        "class_name": "No object detected",
        "confidence": 0.0,
        "predicted_index": None,
        "fg_pixels": 0,
        "mask_coverage_pct": 0.0,
        "bbox": None,
        "crop_w": None,
        "crop_h": None,
        "total_s": 0.0,
        "device": "—",
        "device_name": "—",
        "threshold": THRESHOLD,
        "top_k": [],
        "resnet_class_name": None,
        "resnet_confidence": 0.0,
        "clip_class_name": None,
        "clip_confidence": 0.0,
        "classifier_source": None,
        "error": error,
    }


def run_one(segmenter_key: str, image_bgr: np.ndarray) -> dict:
    """Single-model E2E. Failures are isolated so other models still run."""
    t0 = time.perf_counter()
    try:
        pipe = load_pipeline(segmenter_key)
        result = pipe.predict(image_bgr, threshold=THRESHOLD)
    except Exception as exc:
        out = empty_result(segmenter_key, error=str(exc))
        out["total_s"] = time.perf_counter() - t0
        return out

    total_s = time.perf_counter() - t0
    mask = result["mask"]
    if mask is not None and mask.max() <= 1:
        mask_u8 = (mask * 255).astype(np.uint8)
    else:
        mask_u8 = mask.astype(np.uint8) if mask is not None else None

    has_fg = mask_u8 is not None and bool((mask_u8 > 0).any())
    outline = make_outline(image_bgr, mask_u8) if has_fg else image_bgr.copy()
    overlay = make_overlay(image_bgr, mask_u8) if has_fg else image_bgr.copy()
    bbox = bbox_from_mask(mask_u8) if mask_u8 is not None else None
    fg = int((mask_u8 > 0).sum()) if mask_u8 is not None else 0
    crop = result.get("crop")

    return {
        "segmenter_key": segmenter_key,
        "original_bgr": image_bgr,
        "mask": mask_u8,
        "outline": outline,
        "overlay": overlay,
        "crop": crop,
        "object_detected": bool(result.get("object_detected")),
        "class_name": result.get("class_name"),
        "confidence": float(result.get("confidence") or 0.0),
        "predicted_index": result.get("predicted_index"),
        "fg_pixels": fg,
        "mask_coverage_pct": mask_coverage_pct(mask_u8),
        "bbox": bbox,
        "crop_w": int(crop.shape[1]) if crop is not None else None,
        "crop_h": int(crop.shape[0]) if crop is not None else None,
        "total_s": total_s,
        "device": str(pipe.device),
        "device_name": device_label(),
        "threshold": THRESHOLD,
        "top_k": result.get("top_k") or [],
        "resnet_class_name": result.get("resnet_class_name"),
        "resnet_confidence": float(result.get("resnet_confidence") or 0.0),
        "clip_class_name": result.get("clip_class_name"),
        "clip_confidence": float(result.get("clip_confidence") or 0.0),
        "classifier_source": result.get("classifier_source"),
        "error": None,
    }


def device_label() -> str:
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "CUDA"
    return "CPU"


def gpu_short() -> str:
    name = device_label()
    if "5050" in name:
        return "RTX 5050"
    if name.startswith("NVIDIA "):
        return name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")[:18]
    return name[:18]


def cuda_status() -> str:
    if torch.cuda.is_available():
        return f"CUDA {torch.version.cuda or '?'}"
    return "CPU only"


def to_rgb(img):
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr


def make_outline(image_bgr: np.ndarray, mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    out = image_bgr.copy()
    m = mask
    if m.max() <= 1:
        m = (m * 255).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 220, 255), thickness)
    return out


def make_overlay(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = image_bgr.copy()
    m = mask
    if m.max() <= 1:
        m = (m * 255).astype(np.uint8)
    color = np.zeros_like(out)
    color[:, :] = (40, 180, 255)
    sel = m > 0
    out[sel] = cv2.addWeighted(out, 1 - alpha, color, alpha, 0)[sel]
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 255), 2)
    return out


def bbox_from_mask(mask: np.ndarray, padding: int = 20):
    m = mask
    if m.max() <= 1:
        m = (m * 255).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    h, w = m.shape[:2]
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(w, int(xs.max()) + padding + 1)
    y1 = min(h, int(ys.max()) + padding + 1)
    return (x0, y0, x1, y1)


def mask_coverage_pct(mask: np.ndarray | None) -> float:
    if mask is None or mask.size == 0:
        return 0.0
    m = mask
    if m.max() <= 1:
        m = (m * 255).astype(np.uint8)
    return 100.0 * float((m > 0).sum()) / float(m.size)


def file_fingerprint(uploaded) -> str:
    return hashlib.sha256(uploaded.getvalue()).hexdigest()


def pick_best_detection(results: dict[str, dict], registry: dict) -> tuple[str, str]:
    """
    Best detection from segmentation + recognition evidence.
    1) Prefer object_detected
    2) Prefer higher animal-ID confidence
    3) Maximize foreground pixels
    4) Tie-break by Official Test IoU (static registry)
    """
    def iou_of(name: str) -> float:
        return float(
            registry["segmenters"]
            .get(name, {})
            .get("official_test", {})
            .get("IoU", 0.0)
        )

    ranked = sorted(
        results.keys(),
        key=lambda n: (
            1 if results[n]["object_detected"] else 0,
            0 if results[n].get("error") else 1,
            float(results[n].get("confidence") or 0.0),
            results[n]["fg_pixels"],
            iou_of(n),
        ),
        reverse=True,
    )
    best = ranked[0]
    r = results[best]
    if r.get("error"):
        reason = f"{best} selected but reported an error: {r['error']}"
    elif not r["object_detected"]:
        reason = (
            f"No model found foreground; selected {best} "
            f"(highest Official Test IoU among empty results)."
        )
    else:
        reason = (
            f"{best} selected by detection confidence "
            f"({r['confidence']:.1f}% · {r.get('class_name')}), "
            f"foreground {r['fg_pixels']:,} px."
        )
    return best, reason


def build_insights(trio: dict[str, dict], selected: str, registry: dict) -> list[str]:
    insights: list[str] = []
    n_det = sum(1 for n in SEGMENTER_ORDER if trio[n]["object_detected"])
    insights.append(f"{n_det}/3 segmentation models detected an object on this image.")

    best_bench = max(
        SEGMENTER_ORDER,
        key=lambda n: float(registry["segmenters"][n]["official_test"]["IoU"]),
    )
    iou = registry["segmenters"][best_bench]["official_test"]["IoU"]
    insights.append(
        f"{best_bench} has the highest COD10K-v3 Official Test IoU ({iou:.4f})."
    )

    sel = trio[selected]
    if sel["object_detected"]:
        insights.append(
            f"ResNet50 classified the selected crop ({selected}) as "
            f"{sel['class_name']} ({sel['confidence']:.2f}% confidence)."
        )
    else:
        insights.append(
            f"Selected view ({selected}): no object detected — ResNet50 was not applied."
        )

    fastest = min(SEGMENTER_ORDER, key=lambda n: trio[n]["total_s"])
    insights.append(
        f"{fastest} had the shortest measured live inference time "
        f"({trio[fastest]['total_s']:.3f} s) on this image."
    )

    densest = max(SEGMENTER_ORDER, key=lambda n: trio[n]["fg_pixels"])
    if trio[densest]["fg_pixels"] > 0:
        insights.append(
            f"{densest} produced the largest foreground mask area on this image "
            f"({trio[densest]['mask_coverage_pct']:.2f}% coverage)."
        )
    return insights


def render_benchmark_bars(metric_choice: str, registry: dict) -> str:
    vals = []
    for name in SEGMENTER_ORDER:
        ot = registry["segmenters"][name]["official_test"]
        if metric_choice == "Inference Time":
            vals.append((name, float(ot["mean_inference_s"])))
        else:
            vals.append((name, float(ot[metric_choice])))
    max_v = max(v for _, v in vals) or 1.0
    rows = []
    for name, v in vals:
        pct = 100.0 * (v / max_v)
        color = MODEL_COLOR[name]
        label = f"{v:.4f}s" if metric_choice == "Inference Time" else f"{v:.4f}"
        rows.append(
            f'<div class="bar-row">'
            f'<div class="bar-name" style="color:{color};">{name}</div>'
            f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{pct:.1f}%;background:{color};"></div></div>'
            f'<div class="bar-val">{label}</div></div>'
        )
    return '<div class="bar-wrap">' + "".join(rows) + "</div>"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
if "trio" not in st.session_state:
    st.session_state.trio = None
if "trio_fp" not in st.session_state:
    st.session_state.trio_fp = None
if "best_name" not in st.session_state:
    st.session_state.best_name = None
if "best_reason" not in st.session_state:
    st.session_state.best_reason = None
if "original_bgr" not in st.session_state:
    st.session_state.original_bgr = None
if "upload_meta" not in st.session_state:
    st.session_state.upload_meta = None

registry = load_registry()
trio = st.session_state.trio
analyzed = trio is not None

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
<div class="sb-brand">DETECTA</div>
<div class="sb-tag">Camouflaged Animal Detection &amp; Segmentation</div>
<div class="sb-item active"><span class="sb-ico">⌂</span>Home</div>
<div class="sb-item"><span class="sb-ico">◎</span>Analyze</div>
<div class="sb-item"><span class="sb-ico">▣</span>Model Comparison</div>
<div class="sb-item"><span class="sb-ico">▤</span>Benchmark Results</div>
<div class="sb-item"><span class="sb-ico">ℹ</span>About DETECTA</div>
<div class="sb-note">Single-page dashboard — navigation is visual only.</div>
<div class="sb-foot">
  <b>DETECTA</b><br/>
  Camouflaged Animal Detection &amp; Segmentation<br/>
  v1.0.0
</div>
""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.markdown(
    f"""
<div class="hero">
  <div>
    <div class="brand">DETECTA</div>
    <div class="deep">Camouflaged Animal Detection &amp; Segmentation</div>
    <div class="hero-h">Detect. Compare. Reveal.</div>
    <p class="hero-sub">AI-powered camouflage detection and animal recognition.</p>
    <div class="hero-quote">“Hidden in Plain Sight” · Revealed by AI.</div>
  </div>
  <div class="status-grid">
    <div class="st-card"><div class="k">GPU</div><div class="v">{gpu_short()}</div></div>
    <div class="st-card"><div class="k">Models</div><div class="v">3 / 3 Online</div></div>
    <div class="st-card"><div class="k">Device</div><div class="v">{cuda_status()}</div></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
n_agree = (
    sum(1 for n in SEGMENTER_ORDER if trio[n]["object_detected"]) if analyzed else None
)
top_pred = "—"
avg_time = "—"
if analyzed:
    bn = st.session_state.best_name or SEGMENTER_ORDER[0]
    br = trio[bn]
    top_pred = br["class_name"] if br["object_detected"] else "No object"
    avg_time = f"{(sum(trio[n]['total_s'] for n in SEGMENTER_ORDER) / 3.0):.2f}s"

cards = [
    ("3", "Segmentation Models", "ResUNet · SINet-V2 · ESCNet", "#2196FF", "◈"),
    ("1" if analyzed else "0", "Image Analyzed", "Current session", "#00D9FF", "▣"),
    (
        f"{n_agree} / 3" if n_agree is not None else "—",
        "Models Detected Object",
        "Live agreement",
        "#9B5CFF",
        "◉",
    ),
    (top_pred, "Top Prediction", "ResNet50", "#19E68C", "◆"),
    (avg_time, "Average Inference Time", "Segmentation", "#2196FF", "⏱"),
]
cols = st.columns(5, gap="small")
for col, (val, label, detail, accent, ico) in zip(cols, cards):
    with col:
        with st.container(border=True):
            fs = "1.05rem" if len(str(val)) > 10 else "1.55rem"
            st.markdown(
                f"""
<div class="sum-wrap">
  <div class="sum-ring" style="color:{accent};border-color:{accent}88;">{ico}</div>
  <div class="sum-label">{label}</div>
  <div class="sum-val" style="font-size:{fs};">{val}</div>
  <div class="sum-detail">{detail}</div>
</div>
""",
                unsafe_allow_html=True,
            )

st.markdown('<div class="band">Main Analysis</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Upload | Input
# ---------------------------------------------------------------------------
c_up, c_in = st.columns([0.92, 1.28], gap="medium")

with c_up:
    with st.container(border=True):
        st.markdown('<p class="sec-h">1. Upload Image</p>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="upload-zone">
  <div class="t">Drag & drop an image here</div>
  <div class="s">or click to browse · JPG · JPEG · PNG · BMP · WEBP</div>
</div>
""",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Choose Image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="User uploads only. COD10K Test files are never loaded.",
            label_visibility="collapsed",
        )
        preview_bgr = None
        upload_name = None
        upload_wh = None
        if uploaded is not None:
            try:
                pil = Image.open(uploaded).convert("RGB")
                preview_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                upload_name = uploaded.name
                upload_wh = (pil.width, pil.height)
                st.caption(f"**{upload_name}** · {upload_wh[0]} × {upload_wh[1]}")
            except Exception as exc:
                st.error(f"Could not read image: {exc}")
                uploaded = None
        st.caption(f"Threshold {THRESHOLD} (+ adaptive fallback) · crop padding 20 · ResNet50 + OpenCLIP")
        run_btn = st.button(
            "Analyze · Run 3-model comparison",
            type="primary",
            use_container_width=True,
            disabled=uploaded is None,
        )

with c_in:
    with st.container(border=True):
        st.markdown('<p class="sec-h">2. Input Image</p>', unsafe_allow_html=True)
        meta = st.session_state.upload_meta
        orig = st.session_state.original_bgr
        show_img = None
        show_cap = None
        if orig is not None and meta is not None:
            show_img = to_rgb(orig)
            show_cap = f"{meta['name']} · {meta['width']} × {meta['height']}"
        elif uploaded is not None and preview_bgr is not None:
            show_img = to_rgb(preview_bgr)
            show_cap = f"{upload_name} · {upload_wh[0]} × {upload_wh[1]}"
        if show_img is not None:
            st.caption(show_cap)
            st.image(show_img, use_column_width=True)
            with st.expander("Zoom / inspect full resolution", expanded=False):
                st.image(show_img, use_column_width=True)
        else:
            st.markdown(
                '<p class="sec-s">Upload an image to begin analysis.</p>',
                unsafe_allow_html=True,
            )
            st.info("Waiting for input image.")

if uploaded is not None and run_btn and preview_bgr is not None:
    fp = file_fingerprint(uploaded)
    try:
        progress = st.progress(0.0, text="Running ResUNet…")
        new_trio: dict[str, dict] = {}
        for i, name in enumerate(SEGMENTER_ORDER):
            progress.progress(
                i / len(SEGMENTER_ORDER),
                text=f"Running {name} → mask → crop → ResNet50 + OpenCLIP…",
            )
            new_trio[name] = run_one(NAME_TO_KEY[name], preview_bgr)
            new_trio[name]["display_name"] = name
            if new_trio[name].get("error"):
                st.warning(f"{name}: {new_trio[name]['error']}")
        progress.progress(1.0, text="Comparison complete")
        best, reason = pick_best_detection(new_trio, registry)
        st.session_state.trio = new_trio
        st.session_state.trio_fp = fp
        st.session_state.best_name = best
        st.session_state.best_reason = reason
        st.session_state.original_bgr = preview_bgr
        st.session_state.upload_meta = {
            "name": upload_name,
            "width": upload_wh[0],
            "height": upload_wh[1],
        }
        st.session_state.viz_view = "Compare"
        progress.empty()
        st.rerun()
    except FileNotFoundError as exc:
        st.error(f"Checkpoint or config missing: {exc}")
    except Exception as exc:
        st.error(f"Inference failed: {exc}")

trio = st.session_state.trio
analyzed = trio is not None

# ---------------------------------------------------------------------------
# Selected Detection | Prediction
# ---------------------------------------------------------------------------
c_best, c_pred = st.columns([1.2, 1.0], gap="medium")
selected_name = st.session_state.best_name
selected_reason = st.session_state.best_reason or ""

with c_best:
    with st.container(border=True):
        st.markdown(
            '<p class="sec-h">3. Selected Detection</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="note">'
            "<b>Live selection</b> uses detection status and foreground extent on this upload. "
            "<b>Official Test IoU</b> is COD10K-v3 benchmark metadata used only as a tie-break — "
            "not a live quality score for this image."
            "</div>",
            unsafe_allow_html=True,
        )
        if not analyzed:
            st.caption("Run analysis to populate this panel.")
        else:
            auto_best = st.session_state.best_name
            selected_name = st.radio(
                "Use detection from",
                list(SEGMENTER_ORDER),
                index=SEGMENTER_ORDER.index(auto_best)
                if auto_best in SEGMENTER_ORDER
                else 0,
                horizontal=True,
                key="override_model",
                help="Automatic default uses detection status + foreground extent; "
                "Official Test IoU only as tie-break.",
            )
            if selected_name != auto_best:
                selected_reason = (
                    f"Manual override — viewing {selected_name} "
                    f"(automatic selection was {auto_best})."
                )
            else:
                selected_reason = st.session_state.best_reason or ""

            r = trio[selected_name]
            color = MODEL_COLOR[selected_name]
            st.markdown(
                f'<div class="mhead" style="border-color:{color}77;background:{color}18;color:{color};">'
                f"{selected_name} · Selected Detection View</div>",
                unsafe_allow_html=True,
            )
            st.caption(selected_reason)
            st.image(to_rgb(r["overlay"]), use_column_width=True)
            if r["object_detected"]:
                st.markdown(
                    '<div class="ok">✓ Camouflage Object Detected</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="warn">— No Object Detected</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<div class="kv"><span>Mask Coverage</span>'
                f'<span>{r["mask_coverage_pct"]:.2f}%</span></div>'
                f'<div class="kv"><span>Bounding Box</span>'
                f'<span>{r["bbox"] if r["bbox"] else "—"}</span></div>'
                f'<div class="kv"><span>Inference Time</span>'
                f'<span>{r["total_s"]:.3f} s</span></div>',
                unsafe_allow_html=True,
            )

with c_pred:
    with st.container(border=True):
        st.markdown('<p class="sec-h">4. Prediction Result</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sec-s">ResNet50 + OpenCLIP refinement · 69 COD classes</p>',
            unsafe_allow_html=True,
        )
        if not analyzed or selected_name is None:
            st.caption("Awaiting analysis.")
        else:
            r = trio[selected_name]
            if r.get("error"):
                st.error(f"{selected_name} failed: {r['error']}")
            elif not r["object_detected"]:
                st.markdown(
                    '<div class="warn">NO OBJECT DETECTED</div>',
                    unsafe_allow_html=True,
                )
                st.caption("Empty segmentation — classifier was not run for this view.")
            else:
                st.markdown(
                    f'<div class="pred-box">'
                    f'<div class="pred-name">{str(r["class_name"]).upper()}</div>'
                    f'<div class="pred-pct">{r["confidence"]:.1f}%</div>'
                    f'<div class="pred-lab">Confidence</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                src = r.get("classifier_source") or "resnet50"
                st.caption(
                    f"Primary ID source: {'OpenCLIP ViT-L/14' if src == 'openclip' else 'ResNet50'}"
                )
                st.caption("Detected object crop")
                if r["crop"] is not None:
                    st.image(to_rgb(r["crop"]), use_column_width=True)

                info = load_animal_info().get(str(r["class_name"]), {})
                if info:
                    st.markdown("**Animal information**")
                    st.markdown(
                        f'<div class="kv"><span>Name</span>'
                        f'<span>{info.get("display_name", r["class_name"])}</span></div>'
                        f'<div class="kv"><span>Category</span>'
                        f'<span>{info.get("category", "—")}</span></div>'
                        f'<div class="kv"><span>Habitat</span>'
                        f'<span>{info.get("habitat", "—")}</span></div>',
                        unsafe_allow_html=True,
                    )
                    st.write(info.get("summary", ""))
                    if info.get("recognition_tip"):
                        st.caption(info["recognition_tip"])

                top_k = r.get("top_k") or []
                if top_k:
                    st.markdown("**Top alternatives**")
                    for item in top_k[:5]:
                        st.markdown(
                            f'<div class="kv"><span>{item.get("class_name")}</span>'
                            f'<span>{float(item.get("confidence", 0)):.1f}%</span></div>',
                            unsafe_allow_html=True,
                        )

                st.markdown(
                    f'<div class="kv"><span>ResNet50</span>'
                    f'<span>{r.get("resnet_class_name") or "—"} '
                    f'({float(r.get("resnet_confidence") or 0):.1f}%)</span></div>'
                    f'<div class="kv"><span>OpenCLIP</span>'
                    f'<span>{r.get("clip_class_name") or "—"} '
                    f'({float(r.get("clip_confidence") or 0):.1f}%)</span></div>'
                    f'<div class="kv"><span>Segmenter</span><span>{selected_name}</span></div>'
                    f'<div class="kv"><span>Class index</span>'
                    f'<span>{r["predicted_index"]}</span></div>',
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------
st.markdown('<div class="band">Model Comparison</div>', unsafe_allow_html=True)
with st.container(border=True):
    st.markdown('<p class="sec-h">5. Model Comparison</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sec-s">Results on this image · live upload only '
        "(not Official Test metrics)</p>",
        unsafe_allow_html=True,
    )
    if not analyzed:
        st.info("Run analysis to compare ResUNet, SINet-V2 and ESCNet on your image.")
    else:
        mcols = st.columns(3, gap="medium")
        for col, name in zip(mcols, SEGMENTER_ORDER):
            r = trio[name]
            color = MODEL_COLOR[name]
            with col:
                with st.container(border=True):
                    st.markdown(
                        f'<div class="mhead" style="border-color:{color};'
                        f'background:{color}1a;color:{color};'
                        f'box-shadow:0 0 18px {color}22;">{name}</div>',
                        unsafe_allow_html=True,
                    )
                    if r.get("error"):
                        st.markdown(
                            '<div class="warn">— Model Error</div>',
                            unsafe_allow_html=True,
                        )
                        st.caption(str(r["error"])[:180])
                    elif r["object_detected"]:
                        st.markdown(
                            '<div class="ok">✓ Detected</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="warn">— Not Detected</div>',
                            unsafe_allow_html=True,
                        )
                    if r.get("mask") is not None:
                        t_mask, t_over, t_crop = st.tabs(["Mask", "Overlay", "Crop"])
                        with t_mask:
                            st.image(r["mask"], use_column_width=True, clamp=True)
                        with t_over:
                            st.image(to_rgb(r["overlay"]), use_column_width=True)
                        with t_crop:
                            if r["crop"] is None:
                                st.warning("No crop")
                            else:
                                st.image(to_rgb(r["crop"]), use_column_width=True)
                    if r.get("object_detected"):
                        st.markdown(
                            f'<div class="kv"><span>Class</span>'
                            f'<span>{r.get("class_name")} ({float(r.get("confidence") or 0):.1f}%)</span></div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        f'<div class="kv"><span>Mask Coverage</span>'
                        f'<span>{r["mask_coverage_pct"]:.2f}%</span></div>'
                        f'<div class="kv"><span>Inference Time</span>'
                        f'<span>{r["total_s"]:.3f} s</span></div>',
                        unsafe_allow_html=True,
                    )

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
st.markdown('<div class="band">Performance</div>', unsafe_allow_html=True)
with st.container(border=True):
    st.markdown(
        '<p class="sec-h">6. Segmentation Performance</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="badge">Official Test Benchmark</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="sec-s">COD10K-v3 Official Test Results · N = '
        f'{registry.get("test_n", 4000)} · <b>static research metadata</b> — '
        "not computed from the uploaded image</p>",
        unsafe_allow_html=True,
    )
    metric_choice = st.selectbox(
        "Benchmark metric",
        ["IoU", "Dice", "Precision", "Recall", "Inference Time"],
        index=0,
        key="bench_metric",
    )
    chart_col, table_col = st.columns([1.15, 1.0], gap="large")
    table_rows = []
    for name in SEGMENTER_ORDER:
        ot = registry["segmenters"][name]["official_test"]
        table_rows.append(
            {
                "Model": name,
                "IoU": ot["IoU"],
                "Dice": ot["Dice"],
                "Precision": ot["Precision"],
                "Recall": ot["Recall"],
                "Inference Time (s)": ot["mean_inference_s"],
            }
        )
    with chart_col:
        st.caption(f"Official Test · {metric_choice}")
        st.markdown(
            render_benchmark_bars(metric_choice, registry),
            unsafe_allow_html=True,
        )
    with table_col:
        st.caption("Official Test table")
        st.dataframe(
            pd.DataFrame(table_rows),
            use_container_width=True,
            hide_index=True,
            height=160,
        )
    st.markdown(
        f'<div class="note">{registry.get("e2e_limitation_note", "")}</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Visualization | Agreement
# ---------------------------------------------------------------------------
st.markdown('<div class="band">Lower Analysis</div>', unsafe_allow_html=True)
v1, v2 = st.columns([1.3, 1.0], gap="medium")

with v1:
    with st.container(border=True):
        st.markdown(
            '<p class="sec-h">7. Detection Visualization</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="sec-s">Cached results · changing views does not rerun inference</p>',
            unsafe_allow_html=True,
        )
        if not analyzed:
            st.caption("No results yet.")
        else:
            view = st.radio(
                "View",
                [
                    "Original",
                    "ResUNet Mask",
                    "SINet-V2 Mask",
                    "ESCNet Mask",
                    "Final Crop",
                    "Compare",
                ],
                horizontal=True,
                key="viz_view",
                label_visibility="collapsed",
            )
            if view == "Original":
                st.image(to_rgb(st.session_state.original_bgr), use_column_width=True)
            elif view == "ResUNet Mask":
                st.image(trio["ResUNet"]["mask"], use_column_width=True, clamp=True)
            elif view == "SINet-V2 Mask":
                st.image(trio["SINet-V2"]["mask"], use_column_width=True, clamp=True)
            elif view == "ESCNet Mask":
                st.image(trio["ESCNet"]["mask"], use_column_width=True, clamp=True)
            elif view == "Final Crop":
                crop = trio[selected_name or st.session_state.best_name]["crop"]
                if crop is None:
                    st.warning("No crop for selected detection.")
                else:
                    st.image(to_rgb(crop), use_column_width=True)
            else:
                mc = st.columns(3)
                for col, name in zip(mc, SEGMENTER_ORDER):
                    with col:
                        color = MODEL_COLOR[name]
                        st.markdown(
                            f'<span style="color:{color};font-weight:600;">{name}</span>',
                            unsafe_allow_html=True,
                        )
                        st.image(
                            to_rgb(trio[name]["overlay"]),
                            use_column_width=True,
                        )

with v2:
    with st.container(border=True):
        st.markdown('<p class="sec-h">8. Model Agreement</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sec-s">Object-detection agreement on this image '
            "(not geometric identity)</p>",
            unsafe_allow_html=True,
        )
        if not analyzed:
            st.caption("—")
        else:
            n_det = sum(1 for n in SEGMENTER_ORDER if trio[n]["object_detected"])
            st.markdown(
                f'<div class="agree-big">{n_det} / 3</div>',
                unsafe_allow_html=True,
            )
            st.caption("Models detected an object")
            for name in SEGMENTER_ORDER:
                r = trio[name]
                color = MODEL_COLOR[name]
                mark = "✓" if r["object_detected"] else "—"
                label = "Detected" if r["object_detected"] else "No Object"
                st.markdown(
                    f'<div class="agree-row">'
                    f'<span><span class="dot" style="background:{color};"></span>'
                    f"<b style='color:{color};'>{name}</b></span>"
                    f"<span>{mark} {label} · {r['mask_coverage_pct']:.2f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

with st.container(border=True):
    st.markdown('<p class="sec-h">9. Key Insights</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sec-s">Facts from this run and static Official Test metadata</p>',
        unsafe_allow_html=True,
    )
    if not analyzed:
        st.caption("Insights appear after analysis.")
    else:
        sel = selected_name or st.session_state.best_name
        for i, line in enumerate(build_insights(trio, sel, registry), start=1):
            st.markdown(
                f'<div class="insight-row">'
                f'<div class="insight-num">{i:02d}</div>'
                f'<div class="insight-text">{line}</div></div>',
                unsafe_allow_html=True,
            )

st.markdown(
    """
<footer class="foot">
<strong>DETECTA</strong><br/>
Detect. Understand. Protect.<br/>
Camouflaged Animal Detection &amp; Segmentation
</footer>
""",
    unsafe_allow_html=True,
)
