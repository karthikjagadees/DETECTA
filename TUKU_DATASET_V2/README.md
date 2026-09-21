# TUKU Dataset v2 (Hard Camouflage)

## Purpose

**TUKU-v2** is a research-grade, hard-camouflage dataset designed to improve camouflaged-animal **segmentation** beyond what COD10K Train alone provides. Focus is difficulty and diversity—not raw image count.

It is **separate** from the production COD10K-v3 tree used by the locked TUKU models.

## Relationship to COD10K

| Asset | Role |
|-------|------|
| COD10K Train (`Camouflage_Breaker-main/dataset/Train`) | Historical / frozen production training source |
| **COD10K Official Test** (`Camouflage_Breaker-main/dataset/Test`) | **Frozen benchmark — never used for TUKU-v2 train/val** |
| **TUKU_DATASET_V2** (this folder) | New hard-camouflage + hard-negative collection |

Never copy, symlink, or train on Official Test images or masks.

## Current production freeze

The following remain **locked** until an explicit future Stage-2 plan:

- ResUNet, SINet-V2, ESCNet, ResNet50 checkpoints  
- `app.py`, inference pipeline, model registry, `escnet.json`  
- Official Test evaluation protocol  

Any future ESCNet fine-tune must start from a **copy** of the frozen approved checkpoint  
(`/home/batka/ESCNet/checkpoints/escnet_finetune/escnet_cod10k_best.pth`), never overwrite production in-place without approval.

## Layout

```
TUKU_DATASET_V2/
├── train/{images,masks}/
├── val/{images,masks}/
├── hard_negative/{images,masks}/
├── metadata/
│   ├── annotation_guidelines.md
│   ├── collection_manifest.csv      ← pre-collection (header only until acquisition)
│   ├── collection_checklist.md
│   ├── dataset_manifest.csv
│   ├── dataset_balance.md           ← report TEMPLATE
│   ├── split_manifest.csv
│   └── annotation_guidelines.md
└── tools/
    └── qa_dataset.py                ← read-only audit
```

## Focus areas

1. **Hard camouflage** — foliage, bark, rock, underwater, low-contrast, occlusion, tiny objects, etc.  
2. **Hard negatives** — deceptive empty backgrounds with **all-zero** masks  
3. **Provenance & licensing** — every external image must record source, URL, license, creator, original ID  
4. **Annotation quality** — binary masks `{0,255}`, matching dimensions, QA via `tools/qa_dataset.py`  
5. **Leakage prevention** — SHA256, perceptual near-dup, group_id series; split seed **42**

## Collection workflow (summary)

```
DISCOVERY → PROVENANCE CHECK → LICENSE CHECK → CANDIDATE RECORD
  → ACQUISITION → IMAGE/MASK PAIRING → ANNOTATION → QA
  → DUPLICATE / NEAR-DUPLICATE CHECK → GROUP CHECK
  → SPLIT ASSIGNMENT → READY FOR TRAINING
```

Details: `metadata/collection_checklist.md`, `metadata/collection_workflow.md`.

## QA

```bash
cd TUKU_DATASET_V2
python tools/qa_dataset.py audit
```

Audit never modifies files. Empty dataset must PASS.

## Status

- Structure: created  
- Collection: **not started** (0 images)  
- Training: **not started**
