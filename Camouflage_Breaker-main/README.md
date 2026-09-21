# DETECTA

Multi-model camouflaged-object detection and animal recognition.

> Previously developed under the internal names **TUKU Deep Learning** / **Camouflage Breaker**. Technical models, checkpoints, and datasets are unchanged.

Models:

- **ResUNet** — existing segmentation (default)
- **SINet-V2** — additive segmentation option
- **ESCNet** — fine-tuned segmentation option
- **ResNet50** — existing 69-class animal classifier (shared)

## Dataset used

**COD10K-v3** from `archive (5).zip` (same content as archive 6).

Why this one:

| Archive | Content | Match to project |
|---------|---------|------------------|
| archive (4) | COD10K-**v2** (`Train/Images/Image`, `GT_Objects/GT_Object`) | Extra nesting; not the layout `utils/dataset.py` expects |
| archive (5) | COD10K-**v3** (`Train/Image`, `Train/GT_Object`) | Exact match to project loaders / SINet training scripts |
| archive (6) | Duplicate of archive (5) | Same as (5) — not needed |

Laid out as:

```
dataset/
  Train/Image + Train/GT_Object   (6000 paired)
  Test/Image  + Test/GT_Object    (4000 paired)
  Info/CAM_train.txt, CAM_test.txt
```

## SINet-V2 setup

```
models/sinetv2/source/          # official GewelsJI/SINet-V2
models/sinetv2/res2net50_...pth
models/sinetv2/snapshot/SINet_V2/Net_epoch_best.pth
saved_models/sinetv2/sinetv2_cod10k_best.pth
models/sinetv2_wrapper.py
```

### Train

```bash
# Full intended run (needs compatible CUDA or patience on CPU)
python training/train_sinetv2.py

# Smoke / low-VRAM
set SINET_EPOCHS=1
set SINET_BATCH_SIZE=2
set SINET_MAX_SAMPLES=400
python training/train_sinetv2.py
```

### Evaluate

```bash
python evaluation/evaluate_sinetv2.py
python evaluation/evaluate_resunet.py
python evaluation/build_model_comparison.py
```

Metrics are written under `outputs/` and read by Streamlit. Never hardcoded.

### App

```bash
streamlit run app.py
```

- Default segmenter: **ResUNet**
- Optional: **SINet-V2**
- Mode: **Compare Models** runs both segmenters on the same image, then ResNet50 on each crop

## GPU note (RTX 50-series)

PyTorch 2.6+cu124 does not include kernels for `sm_120` (RTX 5050). The project auto-falls back to CPU. For GPU training, install a build that supports Blackwell (e.g. recent PyTorch nightly cu128) when available.

## Missing legacy checkpoints

If `saved_models/resunet_best.pth` or `saved_models/classifier_best.pth` are absent, ResUNet / classification metrics show **Not evaluated**. Place the existing trained files into `saved_models/` without renaming — do not overwrite with SINet weights.
