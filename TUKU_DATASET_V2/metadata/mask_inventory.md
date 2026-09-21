# TUKU-v2 Existing Image / Mask Inventory

**Audit date:** 2026-09-20  
**Mode:** Locate & audit only — **no copy, move, rename, download, annotation, or training**

---

## Summary

| Metric | Count |
|--------|------:|
| **Total likely TUKU-v2 images** | **0** |
| Images with masks | 0 |
| Images without masks | 0 |
| Unclear mask relationships | 0 |
| Hard negatives (metadata-confirmed) | 0 |
| Candidate IDs mapped (CAND-001..044) | 0 |
| Unmapped images | 0 |
| Exact SHA256 duplicates (among TUKU-v2 finds) | 0 |
| Likely near-duplicates (aHash Hamming ≤ 5) | 0 |

Inventory file: `metadata/existing_image_inventory.csv` (**header only** — no data rows).

---

## Search scope

Searched (read-only):

- `C:\Users\batka\Downloads`
- `C:\Users\batka\Pictures`
- `C:\Users\batka\Documents`
- Project-adjacent: `C:\Users\batka\Downloads\TUKU DEEP LEARING\` (excluding Official Test contents as TUKU-v2 candidates)
- OneDrive top-level presence check

**Not searched (per policy):** Windows system dirs, Program Files, AppData, WSL, node_modules, conda/venv trees, unrestricted full-drive scan.

**Desktop:** path not present / unavailable on this machine.

---

## Matching method

Looked for:

- Filenames containing `CAND-` / `CAND_`
- Filenames matching `candidate_sources.csv` Commons `File:` / URL stems (e.g. Peacock_Flounder, Uroplatus, Thick_Forest, …)
- Directories named with tuku / camouflage / candidate / hard-neg hints
- Images modified after candidate research (today ≥ 17:30) outside unrelated project trees

**Result:** no local files matched TUKU-v2 candidate IDs or candidate Commons filenames.

`TUKU_DATASET_V2` train/val/hard_negative image and mask folders remain **empty**.

---

## Non-TUKU finds (explicitly excluded from inventory)

### `archive (6).zip`

| Field | Value |
|-------|--------|
| Path | `C:\Users\batka\Downloads\archive (6).zip` |
| Size | ~2.42 GB |
| Content | **COD10K-v3** (`COD10K-v3/Train`, `COD10K-v3/Test`, Info) |
| Zip image-ish entries | Train-like ~24000 · Test-like ~16000 (includes edges/instances etc.) |
| CAND- / TUKU hits inside zip | **0** |
| Role for TUKU-v2 candidates | **Not a TUKU-v2 candidate download** |
| Production mirror | Already present under `Camouflage_Breaker-main\dataset\` (Train + Official Test) |

**Do not treat COD10K Official Test (inside zip or on disk) as TUKU-v2 candidates.**  
**This audit step does not extract, copy, or train on archive (6).**

### Other recent image batches (not TUKU-v2)

- `Downloads\VINARA 941\` — numbered PNG batch; **no** camouflage/candidate metadata; **not** inventoried as TUKU-v2
- Production `Camouflage_Breaker-main\dataset\Train` — COD10K Train (already in production); **not** mapped to CAND-001..044
- Production `dataset\Test` — Official COD10K Test; **isolated; not inventoried as TUKU-v2**

---

## Mask status

| Status | Count |
|--------|------:|
| MASK_FOUND | 0 |
| MASK_NOT_FOUND | 0 |
| MASK_UNCLEAR | 0 |

No TUKU-v2 image set was located, so no mask pairing was possible.

---

## Candidate / source metadata availability

| Source | Status |
|--------|--------|
| `metadata/candidate_sources.csv` | Present (44 CANDIDATE rows — research metadata only) |
| `metadata/candidate_research.md` | Present |
| Local image files for CAND-001..044 | **Not found** |
| Local license/creator sidecars next to images | **Not found** |

---

## Production protection check

| Check | Result |
|-------|--------|
| Official Test images (`dataset\Test\Image`) | **4000** |
| Official Test masks (`dataset\Test\GT_Object`) | **4000** |
| TUKU-v2 image/mask dirs | still **0** files |
| Existing downloaded files modified | **No** |
| Models / checkpoints / configs changed this step | **No** |
| QA audit (`tools/qa_dataset.py`) | **PASS** (empty dataset) |

---

## Conclusion

**No likely TUKU-v2 downloaded candidate images were found** in the allowed search locations.

If the user downloaded candidates under another path, provide that folder path for a follow-up audit.

**Training note:** `archive (6).zip` is COD10K-v3, not the CAND-001..044 set. This step **did not train**. Any train/fine-tune request is a separate, explicit step after ingestion policy is approved.

---

## Exact next step

1. User points to the folder that actually contains the downloaded TUKU-v2 / CAND images **or** confirms downloads still need to be performed.  
2. Re-run locate/audit on that path.  
3. Only after a non-empty inventory + license review: plan safe ingestion (still no silent copy into train/val).

**STOP — DO NOT COPY · DO NOT MOVE · DO NOT DOWNLOAD · DO NOT ANNOTATE · DO NOT TRAIN.**
