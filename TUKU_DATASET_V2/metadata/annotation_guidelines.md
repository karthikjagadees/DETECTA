# TUKU-v2 Annotation Guidelines

**Purpose:** Build a targeted hard-camouflage dataset to improve camouflaged-animal segmentation. Quality and difficulty matter more than volume.

**Location:** `TUKU_DATASET_V2/` (sibling of production `Camouflage_Breaker-main/dataset/`).

**Non-negotiable:** Never use, copy, or symlink anything from COD10K Official Test (`Camouflage_Breaker-main/dataset/Test/`).

**Split seed:** `42` (deterministic). Record every assignment in `metadata/split_manifest.csv`.

---

## 1. Directory roles

| Path | Role |
|------|------|
| `train/images` + `train/masks` | Primary hard-camouflage training pairs |
| `val/images` + `val/masks` | Held-out validation (no near-dup leakage with train) |
| `hard_negative/images` + `hard_negative/masks` | Background-only scenes; masks **must be all-zero** |
| `metadata/` | Manifests + this guide |

Every accepted image must have **exactly one** corresponding mask with the same basename (e.g. `sample001.jpg` ↔ `sample001.png`).

---

## 2. Hard-camouflage primary categories

Use one **primary_camouflage** value. Optional **secondary_camouflage** may list multiple (semicolon-separated).

1. `foliage`
2. `branch`
3. `bark`
4. `rock`
5. `sand/mud`
6. `underwater`
7. `shadow/low-light`
8. `texture`
9. `partial occlusion`
10. `tiny/distant`
11. `broken silhouette`
12. `low-contrast`
13. `cluttered background`
14. `multiple distractors`

### Collection priority (examples)

- Animal matching foliage / branches / bark / rocks
- Underwater animals blending with substrate
- Animals in shadows; low-contrast animals
- Partially occluded animals; tiny/distant animals
- Broken silhouettes; cluttered natural scenes
- Strong camouflage-like textures

Do **not** prioritize easy, high-contrast animal photos.

No fixed animal species list is required at this stage. Record `animal_class` when known.

---

## 3. Hard-negative categories

Hard negatives deliberately confuse a camouflage detector but contain **no target animal**.

1. `vegetation-only`
2. `rocks-only`
3. `branches-only`
4. `shadows-only`
5. `water-only`
6. `sand/mud-only`
7. `empty jungle`
8. `empty underwater`
9. `animal-like textures without animals`
10. `other deceptive backgrounds`

**Mask rule:** hard-negative masks MUST be completely empty (all zeros; canonical `{0}` or all-zero `{0,255}` image with no 255).

Place hard negatives under `hard_negative/` (or tag `split=hard_negative` in manifests when consolidated later).

---

## 4. Annotation / mask rules

| Rule | Requirement |
|------|-------------|
| Dimensions | Image and mask **H×W must match exactly** |
| Encoding | Binary masks; canonical values **`{0,255}`** |
| Conversion | Incoming `{0,1}` may be converted to `{0,255}` at ingest |
| Resolution | Preferred minimum **≥320×320**; reject lower unless explicitly justified |
| Pairing | One image ↔ one mask; same stem |
| Tiny objects | May keep if useful; tag `tiny`. Reference: bbox **≥32×32** OR area **≥0.05%** of image |
| Occlusion | Annotate visible animal boundary; tag `occluded` |
| Multiple animals | Annotate all target FG; tag `multi` |
| Ambiguous boundary | Mark `uncertain`; exclude from primary high-quality train until reviewed |
| Reject | Blurred, corrupted, irrelevant, wrong domain, unusable |

Suggested `annotation_status` values: `accepted`, `rejected`, `uncertain`, `needs_review`.  
Suggested `quality_flag` values: `high`, `medium`, `low`, `reject`.

---

## 5. Leakage prevention

1. **Never** use COD10K Official Test data.
2. **Never** copy or symlink from `dataset/Test`.
3. Exact duplicate **image SHA256** cannot cross splits.
4. Track exact duplicate **mask SHA256**.
5. Perceptual near-duplicates (**phash**) must stay in the **same** split.
6. Burst / sequence / source-series share a **group_id** and stay in the same split.
7. Positive scenes and corresponding hard negatives must not leak visually across train and val.
8. Deterministic seed **`42`**.
9. Record every assignment in `split_manifest.csv` (`assigned_by`, `seed`, `assignment_date`).

---

## 6. Manifest columns (reference)

### `dataset_manifest.csv`

`sample_id, image_filename, mask_filename, split, source, animal_class, primary_camouflage, secondary_camouflage, environment, occlusion, object_scale, contrast_level, multiple_animals, annotation_status, quality_flag, width, height, mask_foreground_pixels, mask_foreground_ratio, image_sha256, mask_sha256, phash, group_id, notes`

### `split_manifest.csv`

`sample_id, split, source, group_id, image_sha256, phash, assigned_by, seed, assignment_date, notes`

Header-only until first samples are ingested. Seed for future assignments: **42**.

---

## 7. Future ingestion QA (not implemented yet)

When QA scripts are added (additive only), they must check:

1. Image exists  
2. Mask exists  
3. Image/mask dimensions match  
4. Mask is binary  
5. Mask is not corrupt  
6. Empty / non-empty status correct for split type  
7. Duplicate image SHA256  
8. Perceptual duplicate (phash)  
9. Resolution (≥320×320 preferred)  
10. Split leakage  
11. Group leakage  
12. Required metadata fields present  
13. Annotation status valid  
14. Hard-negative mask is all-zero  

---

## 8. Quality principle

TUKU-v2 exists for **hard-camouflage improvement**, not to inflate image count. Prefer difficult, well-annotated cases and deceptive empty backgrounds over easy animals.
