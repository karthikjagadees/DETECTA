# TUKU-v2 Collection Checklist

Use this checklist for **every** candidate before it enters the final training set.

## Discovery & provenance

- [ ] Source identified (`source_name`)
- [ ] Source URL recorded (`source_url`)
- [ ] License recorded (`license`)
- [ ] License URL recorded (`license_url`)
- [ ] Creator recorded (`creator`)
- [ ] Original ID recorded (`original_id`)
- [ ] Provenance compatible with intended research/project use
- [ ] If license unclear → `collection_status = LICENSE_UNCLEAR` (do not acquire silently)

## Acquisition

- [ ] Image acquired
- [ ] Image readable / decodable
- [ ] Correct animal / background for hard-camouflage goal
- [ ] Recorded in `collection_manifest.csv` as `CANDIDATE` then `ACQUIRED`
- [ ] File placed under the correct staging / split folder when ready

## Taxonomy labels

- [ ] Camouflage category assigned (`primary_camouflage`, optional `secondary_camouflage`)
- [ ] Environment assigned (`environment`, optional `environment_detail`)
- [ ] Hardness assigned (`hardness_level`: EASY / MODERATE / HARD / EXTREME)
- [ ] Object scale assigned (`object_scale`: TINY / SMALL / MEDIUM / LARGE)
- [ ] Animal diversity checked (species / class / environment not over-repeated without reason)
- [ ] Occlusion / multi-animal / contrast tags set when applicable

## Annotation

- [ ] Annotation completed (`annotation_status = COMPLETED` or `REVIEW_REQUIRED`)
- [ ] Image ↔ mask stem pairing correct
- [ ] Mask binary `{0,255}` (or documented `{0,1}` pending ingest normalization)
- [ ] Image/mask dimensions match
- [ ] Hard-negative mask is **all-zero** (if hard negative)
- [ ] Tiny / occluded / uncertain cases tagged per `annotation_guidelines.md`

## QA & leakage

- [ ] Duplicate check passed (exact SHA256)
- [ ] Near-duplicate check passed (aHash / documented threshold)
- [ ] Group checked (`group_id` for series / burst / same-scene)
- [ ] QA passed (`qa_status = PASS`) via `tools/qa_dataset.py audit` when applicable
- [ ] No COD10K Official Test content used

## Split & finalize

- [ ] Split assigned (`TRAIN` / `VAL`) — only after QA; seed **42**; record in `split_manifest.csv`
- [ ] Final `dataset_manifest.csv` updated
- [ ] `collection_status = QA_PASS` (or terminal reject status)

## Hard stop reminders

- Do **not** use `Camouflage_Breaker-main/dataset/Test/`
- Do **not** train until QA + leakage policy are satisfied
- Production models/checkpoints remain locked
