# TUKU-v2 Dataset Balance Report (TEMPLATE)

Fill this after collection milestones. **Do not invent statistics.**

**Date:** _YYYY-MM-DD_  
**Reviewer:** _name_  
**Seed (for splits):** `42`  
**Dataset root:** `TUKU_DATASET_V2/`

---

## Totals

| Metric | Count |
|--------|------:|
| Total images | |
| Total masks | |
| Valid pairs | |
| Unique animal classes | |
| Unique species (scientific names) | |
| Hard-negative images | |
| Sources (unique `source_name`) | |

---

## By environment

| environment | count |
|-------------|------:|
| forest | |
| jungle | |
| grassland | |
| wetland | |
| river | |
| lake | |
| pond | |
| ocean | |
| coral reef | |
| seabed | |
| mangrove | |
| rocky terrain | |
| desert | |
| sand | |
| mudflat | |
| woodland | |
| mountain | |
| cave | |
| snow/ice | |
| urban-edge/natural interface | |
| other | |

---

## By primary camouflage

| primary_camouflage | count |
|--------------------|------:|
| foliage | |
| branch | |
| bark | |
| rock | |
| sand/mud | |
| underwater | |
| shadow/low-light | |
| texture | |
| partial occlusion | |
| tiny/distant | |
| broken silhouette | |
| low-contrast | |
| cluttered background | |
| multiple distractors | |

---

## By hardness level

| hardness_level | count |
|----------------|------:|
| EASY | |
| MODERATE | |
| HARD | |
| EXTREME | |

---

## Difficulty / structure flags

| Flag / metric | count |
|---------------|------:|
| Tiny-object (label or QA) | |
| Occluded | |
| Multi-animal | |
| Hard negatives | |
| Uncertain / review-required | |

---

## Provenance

| Dimension | Notes / counts |
|-----------|----------------|
| Source distribution | |
| License distribution | |
| LICENSE_UNCLEAR count | |
| Creator concentration (top sources) | |

---

## Splits & leakage

| Metric | count |
|--------|------:|
| TRAIN | |
| VAL | |
| UNASSIGNED | |
| Exact duplicates | |
| Near-duplicates | |
| Group leakage (train∩val) | |

---

## Notes / actions

- Imbalances to address:
- Categories under-collected:
- Categories over-represented:
- Next collection focus:

---

*Planning targets in `README.md` / collection workflow are guides only; adjust based on quality and diversity.*
