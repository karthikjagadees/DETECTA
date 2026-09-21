# TUKU Final Project Cleanup Report

**Completed (UTC):** 2026-09-20T19:10:03.218430+00:00  
**Mode:** Cleanup only — no training, no architecture/inference/config changes

---

## Success checks

| Check | Result |
|-------|--------|
| Production checkpoints byte-identical | **YES** |
| COD10K untouched (Train 6000 / Test 4000) | **YES** |
| TUKU_DATASET_V2 retained | **YES** |
| EXTERNAL_TEST_24 annotation retained | **YES** |
| Archive-6 best + Official Test eval retained | **YES** |
| backup_pre_continue retained (historical SHA evidence) | **YES** |
| ESCNet approved best retained + loads | **YES** |
| Uncertain items deleted | **NO (none deleted)** |
| Smoke: ResUNet / SINet / Classifier / Pipeline / ESCNet load | **PASS** |

---

## A. Totals

| Metric | Value |
|--------|------:|
| Deletion entries recorded | 41 |
| Apparent reclaim (sum of recorded sizes) | **17,279,628,246 bytes (~17.28 GB)** |
| Spurious pyc missing-after-parent-delete notes | 21 (ignored) |
| Real deletion errors | 0 |

---

## B–G. Deleted by category

- **DUPLICATE_DATASET**: 4 items, 2,555,238,060 bytes (~2.56 GB)
- **INTERMEDIATE_CHECKPOINT**: 6 items, 1,563,693,632 bytes (~1.56 GB)
- **REJECTED_MODEL_INFRASTRUCTURE**: 2 items, 12,545,074,516 bytes (~12.55 GB)
- **SAFE_TO_DELETE**: 7 items, 223,620 bytes (~0.00 GB)
- **TEMPORARY_ARTIFACT**: 22 items, 615,398,418 bytes (~0.62 GB)

### Deleted checkpoints (intermediate / smoke)
- `saved_models/resunet_epoch_{5,10,15}.pth`
- `saved_models/sinetv2/sinetv2_cod10k_best_smoke_backup.pth`
- `outputs/.../resunet_archive6_epoch_{5,10,15}.pth`
- WSL `ESCNet/checkpoints/escnet_smoke/`

### Deleted duplicate datasets
- `Downloads/TUKU_COD10K_ARCHIVE_SOURCE/` (~2.4 GB extract)
- Empty `Downloads/archive`, `archive (1)`, `archive (2)` stubs

**NOT deleted:** `archive (5).zip` and `archive (6).zip` (same size, **different SHA256**)

### Deleted rejected-model infrastructure
- WSL `/home/batka/C3Net` (~4.7 GB)
- WSL `/home/batka/tuku_c3net` (~7.2 GB)

### Deleted caches / temp
- All `__pycache__` under Camouflage_Breaker-main
- Console train/eval logs listed in predelete manifest
- `models/sinetv2/source/jittor_lib/`, `imgs/`, `AWESOME_COD_LIST.md`
- `outputs/sinetv2/` smoke outputs
- Empty `TUKU DEEP LEARING/;` and `cp/`
- WSL `/home/batka/tmp`

---

## H–I. Intentionally retained

### Production (REQUIRED)
- `saved_models/resunet_best.pth`
- `saved_models/sinetv2/sinetv2_cod10k_best.pth`
- `saved_models/classifier_best.pth` + `class_mapping.json`
- `models/` production sources + SINet PyTorch lib + Res2Net weights + Net_epoch_best fallback
- `app.py`, `inference/`, `config/`, `utils/`, `dataset/`

### Historical / research
- Archive-6 **best** checkpoint + Official Test eval folder
- `saved_models/backup_pre_continue_archive6_20260920/` (explains historical vs current SHA)
- `logs/_escnet_step*.py`, `_step14_write_report.py`, `e2e_skip_audit.log`
- ESCNet `escnet_cod10k_final.pth` (**different SHA from best** — unique weights)
- ESCNet official `epoch_120.pth`
- Both COD10K zip archives (5) and (6)

### Active research
- `TUKU_DATASET_V2/`
- `outputs/external_test_24_annotation/`
- `outputs/external_24_qualitative_test/`

---

## J–K. Protected checkpoint SHA256 BEFORE vs AFTER

| Asset | SHA256 | Match |
|-------|--------|-------|
| resunet_best.pth | `807098f60128c5c78c9d995aff711f123174962ec793b37c5b8288491a668fbd` | YES |
| sinetv2_cod10k_best.pth | `b8618b72260648eea2b1e1ec129813a0e19607cf2dbc5eb2be4355093ad7d949` | YES |
| classifier_best.pth | `f9bc1e904110b38f347a348f3e86020a7e415db91427886f8d56ceabf0852c11` | YES |
| class_mapping.json | `b3b4527f4488b4b5de055e4a883511f67402a9694f8d6fbf73afc23bb17d0bfd` | YES |
| ESCNet escnet_cod10k_best.pth | `d6ffec2d4d281846bd55016dbed6c8a5a938ba0a126158bef5f9949c92f55e82` | YES |

---

## L–N. Dataset verification

| Dataset | Status |
|---------|--------|
| COD10K Train images | 6000 |
| COD10K Official Test images | 4000 |
| TUKU_DATASET_V2 README | present |
| EXTERNAL_TEST_24 annotation manifest | present |

---

## O. Uncertain — NOT deleted

1. `Downloads/archive (5).zip` — SHA `8d79f8c3…` (≠ zip6)
2. `Downloads/archive (6).zip` — SHA `3a7674e0…` (≠ zip5)
3. `/home/batka/ESCNet/checkpoints/escnet_finetune/escnet_cod10k_final.pth` — SHA `7ed2e38d…` ≠ best
4. `/home/batka/ESCNet/checkpoints/escnet/epoch_120.pth` — official baseline
5. `saved_models/backup_pre_continue_archive6_20260920/` — historical SHA evidence
6. Smoke-test Python scripts (`test_*.py`, `extract_classes.py`) — kept
7. `training/train_sinetv2_40.py` — kept

---

## Artifacts

- `PROJECT_CLEANUP_PROTECTED_HASHES_BEFORE.json`
- `PROJECT_CLEANUP_PROTECTED_HASHES_AFTER.json`
- `PROJECT_CLEANUP_PREDELETE_MANIFEST.json`
- `PROJECT_CLEANUP_DELETE_RESULT.json`
- `PROJECT_CLEANUP_DELETED_FILES.csv`
- `PROJECT_CLEANUP_FINAL_REPORT.md` (this file)

**No production model code, checkpoints, COD10K, TUKU_DATASET_V2, or EXTERNAL_TEST_24 were modified.**
