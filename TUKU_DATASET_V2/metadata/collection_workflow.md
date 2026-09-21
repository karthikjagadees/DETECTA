# TUKU-v2 Collection Workflow & Controlled Vocabularies

This document defines the **pre-collection** workflow and allowed field values for `collection_manifest.csv`.  
No images are downloaded in the foundation phase.

**Split seed (future assignments only):** `42`  
**Do not assign TRAIN/VAL until after QA.** Default `split = UNASSIGNED`.

---

## 1. Collection workflow

```
DISCOVERY
   ↓
PROVENANCE CHECK
   ↓
LICENSE CHECK
   ↓
CANDIDATE RECORD          → collection_manifest.csv row
   ↓
ACQUISITION               → obtain file only if license OK
   ↓
IMAGE/MASK PAIRING        → same stem under images/ + masks/
   ↓
ANNOTATION                → binary mask; taxonomy tags
   ↓
QA                        → tools/qa_dataset.py audit
   ↓
DUPLICATE / NEAR-DUPLICATE CHECK
   ↓
GROUP CHECK               → group_id / series
   ↓
SPLIT ASSIGNMENT          → TRAIN | VAL (seed 42); record split_manifest.csv
   ↓
READY FOR TRAINING
```

No image enters the final training set without passing the appropriate checks  
(see `collection_checklist.md`).

---

## 2. Status vocabularies

### `collection_status`

| Value | Meaning |
|-------|---------|
| `CANDIDATE` | Identified; not yet acquired |
| `ACQUIRED` | File obtained; may await annotation |
| `REJECTED` | Rejected for quality/domain/other |
| `DUPLICATE` | Exact or policy duplicate |
| `LICENSE_UNCLEAR` | Must not silently assume permission |
| `READY_FOR_ANNOTATION` | Provenance OK; ready to mask |
| `ANNOTATED` | Mask exists; pending QA |
| `QA_PASS` | Passed dataset QA |
| `QA_FAIL` | Failed QA; fix or reject |

### `annotation_status`

| Value |
|-------|
| `NOT_STARTED` |
| `IN_PROGRESS` |
| `COMPLETED` |
| `REVIEW_REQUIRED` |
| `REJECTED` |

### `qa_status`

| Value |
|-------|
| `NOT_CHECKED` |
| `PASS` |
| `FAIL` |
| `REVIEW_REQUIRED` |

### `split`

| Value | When |
|-------|------|
| `UNASSIGNED` | Default until after QA |
| `TRAIN` | After leakage-safe assignment |
| `VAL` | After leakage-safe assignment |

Do **not** assign TRAIN/VAL at the candidate/collection-planning stage.

---

## 3. Licensing / provenance policy

Every future **external** image must record:

- `source_name`
- `source_url`
- `license`
- `license_url`
- `creator`
- `original_id`

**Do not** accept an image into the actual training dataset if licensing/provenance is unknown or incompatible with intended research/project use.

If unclear: `collection_status = LICENSE_UNCLEAR`.  
Do not assume permission. Do not download under unclear license in automated pipelines without human review.

Suggested `source_type` examples (free text, keep consistent):  
`museum`, `research_repo`, `photographer`, `own_capture`, `partner`, `other`.

---

## 4. Hard-camouflage taxonomy (primary)

Use existing approved categories only (document any new primary before use):

`foliage` · `branch` · `bark` · `rock` · `sand/mud` · `underwater` ·  
`shadow/low-light` · `texture` · `partial occlusion` · `tiny/distant` ·  
`broken silhouette` · `low-contrast` · `cluttered background` · `multiple distractors`

`secondary_camouflage`: semicolon-separated list of additional categories.

---

## 5. Environment taxonomy

Controlled values for `environment`:

`forest` · `jungle` · `grassland` · `wetland` · `river` · `lake` · `pond` ·  
`ocean` · `coral reef` · `seabed` · `mangrove` · `rocky terrain` · `desert` ·  
`sand` · `mudflat` · `woodland` · `mountain` · `cave` · `snow/ice` ·  
`urban-edge/natural interface` · `other`

Use `environment_detail` for free-text specificity.  
`water_or_land`: `water` | `land` | `interface` | `unknown`.

---

## 6. Animal diversity

Track `animal_class`, `animal_common_name`, `animal_scientific_name`, `environment`.

Goals (not hard quotas):

- Diversity across species, body shape, environment, camouflage mechanism, scale, viewing angle  
- Avoid excessive repetition of the same animal/environment combination  
- No fixed mandatory species list  

Use `dataset_balance.md` reviews to detect over-representation.

---

## 7. Hardness level

| Level | Guidance |
|-------|----------|
| `EASY` | Animal clearly distinguishable despite camouflage |
| `MODERATE` | Partial blend; silhouette mostly recoverable |
| `HARD` | Low contrast, disrupted boundaries, occlusion, or strong texture similarity |
| `EXTREME` | Very low contrast, severe occlusion, tiny object, broken silhouette, or highly deceptive background |

Hardness is metadata; it does **not** replace objective QA (`qa_dataset.py`).

---

## 8. Object scale (descriptive)

| Label | Role |
|-------|------|
| `TINY` | Descriptive; often aligns with tiny QA flag |
| `SMALL` | Descriptive |
| `MEDIUM` | Descriptive |
| `LARGE` | Descriptive |

Quantitative mask metrics (`foreground_pixels`, `foreground_ratio`, bbox, `bbox_area_ratio`) are computed **after** annotation—never fabricate them in the pre-collection manifest.

Related: `contrast_level` (e.g. `low` / `medium` / `high`), `silhouette_quality` (e.g. `clear` / `broken` / `ambiguous`), `occlusion` (e.g. `none` / `partial` / `heavy`), `multiple_animals` (`yes` / `no`).

---

## 9. Initial planning targets (NOT mandatory quotas)

Approximate starting guide (~320 total). Adjust for quality and diversity.

| Theme | Suggested count |
|-------|----------------:|
| foliage/vegetation | 40 |
| branch/bark | 25 |
| rock/sand/mud | 25 |
| underwater | 30 |
| shadow/low-light | 20 |
| low-contrast/texture | 30 |
| partial occlusion | 30 |
| tiny/distant | 20 |
| broken/cluttered | 25 |
| hard negatives | 75 |
| **Planning total** | **~320** |

Quality beats hitting the number.

---

## 10. Hard negatives

Categories:

`vegetation-only` · `rocks-only` · `branches-only` · `shadows-only` ·  
`water-only` · `sand/mud-only` · `empty jungle` · `empty underwater` ·  
`animal-like textures without animals` · `other deceptive backgrounds`

Rules:

- Masks **must be all-zero**  
- Must be visually difficult for a camouflage detector (not random easy landscapes)  
- Store under `hard_negative/` when acquired  

---

## 11. Duplicate / leakage policy

| Check | Action |
|-------|--------|
| Exact image SHA256 | Flag `DUPLICATE` / EXACT_DUPLICATE; cannot cross splits |
| Exact mask SHA256 | Track; investigate shared masks |
| Perceptual near-dup (aHash Hamming ≤ 5 in `qa_dataset.py`) | Keep same split; investigate |
| Source-series / burst / sequence / same-scene | Shared `group_id`; same split |
| COD10K Official Test | **Forbidden** as source for v2 |

Train/val assignment happens **only after** QA, using seed **42**, and is recorded in `split_manifest.csv`.  
Do not assign splits in this pre-collection phase.
