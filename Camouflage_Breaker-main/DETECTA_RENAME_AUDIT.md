# DETECTA Rename Audit

**Date (UTC):** 2026-09-20  
**Mode:** Branding + project-path migration only  
**ML / checkpoints / datasets:** NOT modified by this plan

## Roots

| | Path |
|--|------|
| Current project root | `C:\Users\batka\Downloads\TUKU DEEP LEARING\Camouflage_Breaker-main` |
| Proposed project root | `C:\Users\batka\Downloads\TUKU DEEP LEARING\DETECTA` |
| Parent folder (unchanged) | `TUKU DEEP LEARING` (filesystem parent; not the app brand) |
| Dataset sibling (unchanged name) | `TUKU_DATASET_V2/` |

---

## Classification of occurrences

### BRANDING — UPDATE (user-facing current identity)

| File | Location | Current | Action |
|------|----------|---------|--------|
| `app.py` | module docstring L2 | `TUKU — Camouflage Intelligence System` | → DETECTA branding note |
| `app.py` | `page_title` L43 | `TUKU — Deep Learning` | → `DETECTA` |
| `app.py` | sidebar L573–584 | `TUKU` / Deep Learning / About TUKU | → DETECTA |
| `app.py` | hero L597–598 | `TUKU` / Deep Learning | → DETECTA (+ existing subtitle pattern) |
| `app.py` | footer L1098 | `TUKU · Camouflage Intelligence System` | → DETECTA |
| `README.md` | H1 L1 | `TUKU Deep Learning — Camouflage Breaker` | → `# DETECTA` + historical note |

### ACTIVE_PATH — UPDATE via folder rename

Active Python uses `Path(__file__).resolve().parent` / `PROJECT_ROOT` — **no hard-coded Camouflage_Breaker-main in runtime app/pipeline/models**.  
Folder rename alone updates on-disk root.

### INTERNAL_IDENTIFIER — KEEP

| Symbol | File | Reason |
|--------|------|--------|
| `CamouflageBreakerPipeline` | `inference/pipeline.py`, `app.py` import | Internal class; not user brand |
| Print banners in `inference/pipeline.py` | console only | **inference/ is protected — do not edit** |

### DATASET — KEEP name

- `TUKU_DATASET_V2`
- `EXTERNAL_TEST_24` / `external_test_24_*`
- `COD10K-v3` / `dataset/`

### MODEL — KEEP

- ResUNet, SINet-V2, ESCNet, ResNet50

### HISTORICAL — KEEP (do not mass-rewrite)

- `PROJECT_CLEANUP_*.md/json/csv`
- Archive-6 / external-test reports with old absolute paths
- `logs/_step*.py`, `_escnet_step*.py`
- One-off scripts `_run_*.py`, `_write_cleanup_report.py` with old absolute roots (historical)

### OTHER / UI copy — KEEP

| Text | Reason |
|------|--------|
| `Camouflage Object Detected` in `app.py` | Describes detection result, not product name |
| Hero quote / “camouflage detection” subtitle | Domain description, not old brand mark |

### EXPERIMENT — KEEP

- Archive-6 naming, continue_archive6, etc.

---

## Planned file content edits

1. `app.py` — branding strings only  
2. `README.md` — current project title + one-line prior-name note  

## Planned filesystem rename

`Camouflage_Breaker-main` → `DETECTA`

## Explicitly NOT planned

- No edits under `models/`, `saved_models/` contents, `inference/`, `dataset/`, `config/` metric values  
- No checkpoint re-save  
- No rename of `TUKU_DATASET_V2`  
- No rewrite of historical cleanup/eval reports  
- No rename of `CamouflageBreakerPipeline`
