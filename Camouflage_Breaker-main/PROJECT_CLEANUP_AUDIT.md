# TUKU Project Cleanup Audit

**Date (UTC context):** 2026-09-20  
**Mode:** AUDIT ONLY — **nothing deleted, moved, renamed, or modified**  
**Primary tree:** `C:\Users\batka\Downloads\TUKU DEEP LEARING\Camouflage_Breaker-main`

---

## Safety confirmation

| Check | Result |
|-------|--------|
| Production models / architectures touched | **No** |
| Production checkpoints touched | **No** |
| COD10K / Official Test touched | **No** |
| TUKU_DATASET_V2 touched | **No** |
| Streamlit / inference / configs touched | **No** |
| ESCNet WSL approved checkpoint touched | **No** |
| EXTERNAL_TEST_24 outputs touched | **No** |
| Any deletion performed | **No** |

---

## Protected (DO NOT DELETE)

### Windows production (REQUIRED)

- `models/` (including SINet-V2 PyTorch `source/lib`, ResUNet, classifier, ESCNet wrapper)
- `saved_models/resunet_best.pth` (current SHA256 `807098f6…1a668fbd`)
- `saved_models/sinetv2/sinetv2_cod10k_best.pth` (current SHA256 `b8618b72…3ad7d949`)
- `saved_models/classifier_best.pth`
- `saved_models/class_mapping.json`
- `config/`, `inference/`, `utils/`, `dataset/`, `app.py`
- `models/sinetv2/res2net50_v1b_26w_4s-3cf99910.pth` (SINet backbone init)
- `models/sinetv2/source/lib/*.py` (SINet architecture import path)
- `models/sinetv2/snapshot/SINet_V2/Net_epoch_best.pth` (**fallback** if finetuned missing — keep)

### WSL production ESCNet (REQUIRED)

- `/home/batka/ESCNet/` project (inference source)
- `/home/batka/ESCNet/checkpoints/escnet_finetune/escnet_cod10k_best.pth`  
  SHA256 `d6ffec2d4d281846bd55016dbed6c8a5a938ba0a126158bef5f9949c92f55e82`
- Conda env `/home/batka/tuku_escnet` (ESCNet runtime)

### Held-out / in-progress (KEEP until workflow finished)

- `outputs/external_24_qualitative_test/`
- `outputs/external_test_24_annotation/`
- `TUKU_DATASET_V2/` (metadata + tools even with 0 images)

### Historical evidence to KEEP

- Archive-6 **best** checkpoint + final Official Test metrics
- Rejected-model evaluation **reports** (if any)
- Important training summaries / integrity JSON
- Files containing **historical checkpoint hashes** (do not delete for “staleness”)

---

## Inspection scope

| Scope | Approx. files | Approx. size |
|-------|--------------:|-------------:|
| `Camouflage_Breaker-main` | ~25,655 | ~7.1 GB |
| `TUKU_DATASET_V2` | 19 | ~0.15 MB |
| Related Downloads COD10K copies | — | ~7+ GB |
| WSL rejected / ESCNet extras | — | ~20+ GB |

---

## A. DEFINITELY SAFE TO DELETE

High confidence; not imported by ResUNet / SINet-V2 / ESCNet / ResNet50 production paths.

| Path | Size | Reason |
|------|-----:|--------|
| `Camouflage_Breaker-main/__pycache__/` | ~0.05 MB | Bytecode cache |
| `models/sinetv2/source/lib/__pycache__/` | ~0.04 MB | Bytecode cache |
| `TUKU DEEP LEARING/;/` (empty dir) | 0 | Accidental empty folder |
| `TUKU DEEP LEARING/cp/` (empty dir) | 0 | Accidental empty folder |
| `Downloads/archive/`, `archive (1)/`, `archive (2)/` | ~0.8 MB | Near-empty extract stubs |

**Estimated safe-now:** ≪ 1 MB (inside project) + tiny Downloads stubs.

---

## B. SAFE TO DELETE AFTER ARCHIVING

Recommend zip/offload first; not needed for live production inference once archived.

| Path | Size | Reason |
|------|-----:|--------|
| `saved_models/resunet_epoch_{5,10,15}.pth` | ~1.12 GB | Intermediate ResUNet epochs; best exists |
| `saved_models/sinetv2/sinetv2_cod10k_best_smoke_backup.pth` | ~103 MB | Smoke backup; SHA ≠ production |
| `outputs/archive6_…/checkpoints/resunet_archive6_epoch_{5,10,15}.pth` | ~373 MB | Intermediate Archive-6; **keep best** |
| `outputs/continue_archive6/*.log` | ~8 MB | Orchestration console logs |
| Large `logs/*.log` train consoles (`resunet_train.log`, `sinetv2_train.log`, …) | ~tens of MB | Raw training stdout; summaries already in JSON/MD |
| `models/sinetv2/source/jittor_lib/` | small | Jittor port; production uses PyTorch `source/lib` only |
| `models/sinetv2/source/imgs/`, `AWESOME_COD_LIST.md` | ~1.7 MB | Docs/images from upstream clone; not imported |
| One of `Downloads/archive (5).zip` **or** `archive (6).zip` | ~2.3 GB | Duplicate COD10K-v3 archives (keep one) |
| `Downloads/TUKU_COD10K_ARCHIVE_SOURCE/` | ~2.4 GB | Extract used for Archive-6; production uses `dataset/` |
| WSL `/home/batka/ESCNet/checkpoints/escnet_smoke/` | ~263 MB | Smoke weights; not approved |
| WSL `escnet_cod10k_final.pth` | ~263 MB | Final-of-run twin; **approved is `…_best.pth`** |
| WSL `/home/batka/C3Net/` + `/home/batka/tuku_c3net/` | ~4.7 + 7.2 GB | Rejected C3Net experiment + env |
| WSL `/home/batka/tmp/` | ~189 MB | Temporary |

**Keep when archiving Archive-6:**  
`resunet_archive6_manifest_best.pth` + `outputs/archive6_resunet_official_test_eval/` + `post_train_audit.json` / `run_config.json`.

---

## C. HISTORICAL — KEEP

| Path | Why |
|------|-----|
| `saved_models/backup_pre_continue_archive6_20260920/` | Provenance of **older** checkpoint SHAs (matches step-14 historical hashes) |
| `outputs/archive6_resunet_manifest_s42/checkpoints/*_best.pth` | Experimental best |
| `outputs/archive6_resunet_official_test_eval/` | Official Test comparison evidence |
| `logs/_escnet_step*.py`, `_step14_write_report.py`, etc. | Experiment / report generators |
| `logs/_step14_write_report.py` hash table | Historical SHA registry (conflicts with current — do not erase) |
| Integrity / audit JSON under `outputs/` | Reproducibility |

---

## D. REQUIRED — KEEP

Everything listed under **Protected** above, plus:

- `training/`, `evaluation/` scripts that load production models  
- `requirements.txt`, `README.md`  
- `models/escnet_wrapper.py`, `config/escnet.json`  
- SINet `source/lib/Network_*.py`, `Res2Net_v1b.py`

---

## E. UNCERTAIN — DO NOT TOUCH

| Item | Why uncertain |
|------|----------------|
| `saved_models/backup_pre_continue_…` vs deleting after “confirm continue-train” | Historical SHA provenance; user may want forever |
| Deleting **both** archive zips | Need at least one offline COD10K recovery copy |
| Deleting WSL `tuku_escnet` | Required for ESCNet |
| Deleting WSL official `epoch_120.pth` | Init reference / identity check vs finetuned |
| `outputs/external_*` | Annotation incomplete — protected by policy |
| Root `TUKU DEEP LEARING/debug.log` | Tiny; purpose unclear |
| Smoke-test scripts (`test_*.py`) | Not production runtime, but useful regression |

---

## F. DUPLICATE DATASET COPIES

| Location | Size | Role | Recommendation |
|----------|-----:|------|----------------|
| `Camouflage_Breaker-main/dataset/` | ~2.86 GB | **Production COD10K** | **REQUIRED — KEEP** |
| `Downloads/TUKU_COD10K_ARCHIVE_SOURCE/` | ~2.44 GB | Archive-6 training extract | SAFE AFTER ARCHIVING |
| `Downloads/archive (5).zip` | ~2.31 GB | COD10K-v3 archive | KEEP one of (5)/(6) |
| `Downloads/archive (6).zip` | ~2.31 GB | Duplicate of (5) | SAFE AFTER ARCHIVING (keep other) |
| Empty `Downloads/archive*` extract dirs | ~0 | Stubs | SAFE_TO_DELETE |

**Note:** `DATASET FOR DEEP CLEANING` was **not found** at the prior Downloads path during this audit; working copies exist under `outputs/external_test_24_annotation/images/` (**KEEP**).

---

## G. REJECTED MODEL ARTIFACTS

| Artifact | Location | Size | In Windows TUKU imports? | Status |
|----------|----------|-----:|--------------------------|--------|
| C3Net source + ckpts | WSL `/home/batka/C3Net` | ~4.7 GB | No | SAFE AFTER ARCHIVING |
| `tuku_c3net` conda env | WSL | ~7.2 GB | No | SAFE AFTER ARCHIVING |
| ZoomNeXt / CFF-KDNet / SAE-Net / DGNet / SAM3 installs | Searched WSL+Downloads | **Not found** as project trees | — | N/A |
| DGNet mention | `models/sinetv2/source/AWESOME_COD_LIST.md` only | docs | Not a dependency | SAFE with that file |

---

## H. TEMPORARY / CACHE FILES

| Item | Status |
|------|--------|
| `__pycache__` / `.pyc` | SAFE_TO_DELETE |
| Train/eval `.log` consoles | SAFE AFTER ARCHIVING |
| WSL `/home/batka/tmp` | SAFE AFTER ARCHIVING |
| ESCNet smoke ckpt | SAFE AFTER ARCHIVING |

---

## I. DOCUMENTATION CLEANUP

| Item | Action |
|------|--------|
| Upstream SINet README/LICENSE | KEEP (license) |
| `AWESOME_COD_LIST.md`, award images | SAFE AFTER ARCHIVING |
| Hash records in reports | **KEEP** (see hash section) |
| Do **not** delete for “obsolete hash” | Historical evidence |

---

## J. EXPERIMENTAL OUTPUTS

| Path | Action |
|------|--------|
| Archive-6 **best** + Official Test eval folder | HISTORICAL — KEEP |
| Archive-6 **epoch_5/10/15** | SAFE AFTER ARCHIVING |
| `continue_archive6/` logs | SAFE AFTER ARCHIVING |
| `outputs/sinetv2/` smoke images (~0.25 MB) | SAFE_TO_DELETE or KEEP trivial |
| `external_24_*` / annotation | **KEEP** (protected) |
| `external_test_dataset_audit/` | KEEP (small audit) |

---

## Special check — checkpoint hashes

| Model | CURRENT VERIFIED | HISTORICAL | Where historical appears | Context |
|-------|------------------|------------|--------------------------|---------|
| ResUNet | `807098f60128c5c78c9d995aff711f123174962ec793b37c5b8288491a668fbd` | `2f0ceb1fb0b7c1818aa036c5d87a9bcb1da565e482fb66dd0570f01de74cbd9d` | `logs/_step14_write_report.py`; **matches** `backup_pre_continue_…/resunet_best.pth` | Pre-continue backup / older report |
| SINet-V2 | `b8618b72260648eea2b1e1ec129813a0e19607cf2dbc5eb2be4355093ad7d949` | `8826b06c19935d327ec3d0a538124347eafe0236db66ceea017cd5e8ca4b502a` | step14; **matches** backup sinet | Same |
| SINet smoke backup | `be3fb7bb498d9add58ac1381dd09e536b8f572e68f1d2acdd3c703f677ecaf1c` | — | smoke file only | Not production |
| ESCNet finetuned | `d6ffec2d4d281846bd55016dbed6c8a5a938ba0a126158bef5f9949c92f55e82` | same in step14 | consistent | Approved |
| ESCNet official | `4d2d57a61f51d70aad197270dfb68e58…` (epoch_120) | documented in step14 | WSL official | Not finetuned |
| Archive-6 ResUNet | `0a40d72a6caa92a2c02696eaf86c63056dc9f2c6b9a6f18388d776e2238b2ca5` | — | eval artifacts | Experimental best |

**Do not delete** files solely because they contain historical hashes.

---

## Size rollup (approximate)

| Bucket | Est. size |
|--------|----------:|
| SAFE_TO_DELETE (now, in-tree) | ≪ 10 MB |
| SAFE AFTER ARCHIVING (Windows + Downloads zips/extract) | ~**6–7 GB** |
| SAFE AFTER ARCHIVING (WSL rejected C3Net + env + smoke/tmp) | ~**12–13 GB** |
| HISTORICAL KEEP (Archive-6 best + eval + backup folder) | ~**1.5–2 GB** |
| REQUIRED production (models+saved_models prod+dataset+code) | ~**5+ GB** in CB tree |
| UNCERTAIN / protected external-test | ~**0.7 GB** |
| Duplicate COD10K (extra extract + one zip) reclaimable | ~**4.7 GB** |

---

## Totals for this audit

| Metric | Value |
|--------|------:|
| Files inspected (CB tree) | ~25,655 |
| Cleanup candidate rows (CSV) | see `PROJECT_CLEANUP_CANDIDATES.csv` |
| Deletion executed | **0** |

---

## Recommended next step (for you)

1. Review CSV statuses.  
2. Authorize a **Phase 1** delete: caches + empty dirs only.  
3. Authorize a **Phase 2** archive-then-delete: intermediate epochs, smoke backups, one COD10K zip, WSL C3Net.  
4. Never authorize deletion of protected production / EXTERNAL_TEST_24 / TUKU_DATASET_V2 without a separate written plan.

**STOP — audit complete. No deletions performed.**
