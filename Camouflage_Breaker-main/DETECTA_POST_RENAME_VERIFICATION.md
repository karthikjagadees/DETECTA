# DETECTA — Post–Folder-Rename Final Verification

**Mode:** READ-ONLY (no training; no checkpoint/dataset/model/metric writes)  
**Verified (local):** 2026-09-20  

---

## Final project path

| Item | Value |
|------|--------|
| Expected root | `C:\Users\batka\Downloads\TUKU DEEP LEARING\DETECTA` |
| Expected root exists | **NO** |
| Actual root on disk | `C:\Users\batka\Downloads\TUKU DEEP LEARING\Camouflage_Breaker-main` |
| Cursor workspace parent | `C:\Users\batka\Downloads\TUKU DEEP LEARING` |
| Sibling `TUKU_DATASET_V2` | present |
| `RENAME_TO_DETECTA.bat` (parent) | present (rename not applied) |

---

## Folder rename status

**NOT VERIFIED.** Physical folder is still named `Camouflage_Breaker-main`.  
`DETECTA` does not exist under `TUKU DEEP LEARING`.

All checks below were executed against the **actual** root:

`C:\Users\batka\Downloads\TUKU DEEP LEARING\Camouflage_Breaker-main`

---

## Production checkpoint presence

| Path | Status |
|------|--------|
| `app.py` | present |
| `saved_models/resunet_best.pth` | present |
| `saved_models/sinetv2/sinetv2_cod10k_best.pth` | present (**actual** SINet filename) |
| `saved_models/sinetv2/sinetv2_cod10d_best.pth` | absent (typo alias) |
| `saved_models/classifier_best.pth` | present |
| ESCNet WSL `/home/batka/ESCNet/checkpoints/escnet_finetune/escnet_cod10k_best.pth` | present |
| `escnet_wrapper.resolve_escnet_paths()` | resolves to WSL best checkpoint |

---

## Production checkpoint hash comparison

Live SHA256 vs `DETECTA_RENAME_PROTECTED_HASHES_BEFORE.json` and `DETECTA_RENAME_PROTECTED_HASHES_AFTER.json`.

| Asset | Live SHA256 | = BEFORE | = AFTER |
|-------|-------------|----------|---------|
| resunet_best.pth | `807098f60128c5c78c9d995aff711f123174962ec793b37c5b8288491a668fbd` | YES | YES |
| sinetv2_cod10k_best.pth | `b8618b72260648eea2b1e1ec129813a0e19607cf2dbc5eb2be4355093ad7d949` | YES | YES |
| classifier_best.pth | `f9bc1e904110b38f347a348f3e86020a7e415db91427886f8d56ceabf0852c11` | YES | YES |
| class_mapping.json | `b3b4527f4488b4b5de055e4a883511f67402a9694f8d6fbf73afc23bb17d0bfd` | YES | YES |
| ESCNet best (WSL) | `d6ffec2d4d281846bd55016dbed6c8a5a938ba0a126158bef5f9949c92f55e82` | YES | N/A* |

\* `DETECTA_RENAME_PROTECTED_HASHES_AFTER.json` omits the `escnet` key; ESCNet matches BEFORE and live WSL hash.

**All compared production hashes identical — checkpoints unchanged.**

---

## Active old-path references (`Camouflage_Breaker-main`)

| Area | Result |
|------|--------|
| `app.py` | none |
| `inference/` | none |
| `models/` | none |
| Runtime path resolution | `Path(__file__).resolve().parent` (folder-name independent) |

**Remaining hardcoded hits (non-dashboard helpers / provenance — not modified):**

- One-off cleanup scripts: `_run_final_cleanup.py`, `_write_cleanup_report.py`
- Tooling: `tools/qa_external_test_masks.py`
- Training helpers: `training/train_resunet_archive6_manifest.py`, `training/*.ps1`
- Output one-offs under `outputs/external_*` and Archive-6 status JSON
- Rename/cleanup hash/report JSON and historical Markdown (left intact per instructions)

No active Streamlit/dashboard dependency on the old folder name.

---

## Dashboard branding verification

| Check | Result |
|-------|--------|
| Product brand `DETECTA` (`page_title`, sidebar, hero) | YES |
| Presented as current app name: `TUKU DEEP LEARNING` | **NO** |
| Presented as current app name: `Camouflage Breaker` | **NO** |
| README title `# DETECTA` | YES (historical alias note only) |

---

## Model import verification

Python: `Python 3.10` (`torch 2.11.0+cu128`)

| Module | Result |
|--------|--------|
| `app.py` | AST parse OK; branding verified (Streamlit side effects avoided) |
| `inference.pipeline` (`CamouflageBreakerPipeline`) | OK |
| `models.resunet` | OK |
| `models.classifier` | OK |
| `models.sinetv2_wrapper` | OK |
| `models.escnet_wrapper` | OK (WSL path resolve OK) |

---

## Inference smoke-test result

- Checkpoint: `saved_models/resunet_best.pth`
- Sample: Official Test `COD10K-CAM-1-Aquatic-1-BatFish-2.jpg`
- Forward only; nothing saved
- **SMOKE_OK** — output shape `(1, 1, 352, 352)`, mean sigmoid ≈ `0.138`

---

## Dataset verification

| Path | Status |
|------|--------|
| `models/` | present (~40 files) |
| `saved_models/` | present |
| `inference/` | present |
| `dataset/` | present |
| `dataset/Train/Image` | **6000** |
| `dataset/Test/Image` | **4000** (COD10K Official Test intact) |
| `dataset/Train/GT_Object` | 6000 |
| `dataset/Test/GT_Object` | 4000 |
| `TUKU_DATASET_V2/` (sibling) | present |
| `EXTERNAL_TEST_24/` as top-level name | **not present under that exact name** |
| External-24 workspace | `outputs/external_test_24_annotation/` (24 images) + `outputs/external_24_qualitative_test/` (24 originals) |

---

## Warnings

1. **Physical folder rename incomplete** — still `Camouflage_Breaker-main`; expected `DETECTA` missing. Close locks / Cursor handles on the folder, then run `RENAME_TO_DETECTA.bat`, and re-verify against `...\DETECTA`.
2. SINet production file is `sinetv2_cod10k_best.pth` (not `…cod10d…`).
3. `DETECTA_RENAME_PROTECTED_HASHES_AFTER.json` does not record ESCNet; ESCNet integrity confirmed via BEFORE + live WSL SHA256.
4. Top-level folder named exactly `EXTERNAL_TEST_24/` is absent; the 24-image external set lives under `outputs/external_test_24_*`.

---

## Verdict

| Required statement | Status |
|--------------------|--------|
| DETECTA FOLDER RENAME VERIFIED | **FAIL** (folder still `Camouflage_Breaker-main`) |
| PRODUCTION CHECKPOINTS UNCHANGED | **PASS** |
| PRODUCTION MODELS UNCHANGED | **PASS** |
| DATASETS UNCHANGED | **PASS** |
| DASHBOARD BRANDING VERIFIED | **PASS** |

### Confirmed (partial)

```
PRODUCTION CHECKPOINTS UNCHANGED
PRODUCTION MODELS UNCHANGED
DATASETS UNCHANGED
DASHBOARD BRANDING VERIFIED
```

### Not confirmed

```
DETECTA FOLDER RENAME VERIFIED
```

**Overall: PARTIAL PASS — branding, checkpoints, models, and datasets OK; on-disk folder rename not completed.**
