# TUKU-v2 Image Ingestion Report

**Date:** 2026-09-20 (local)  
**Mode:** Discovery + verification + staging policy only  
**Training / masks / train-val split:** **NOT performed**

---

## Executive summary

| Result | Count |
|--------|------:|
| CAND-001..044 records searched | **44** |
| CAND images matched on disk | **0** |
| Verified & staged to `staging/images/` | **0** |
| CAND still missing | **44** |
| Other images located (ChatGPT-named Downloads) | **24** |
| Of those: exact internal SHA duplicates | **1 pair (2 files)** |
| Of those: COD10K Train/Test SHA leaks | **0** |
| Staged as verified candidates | **0** (not in CAND list; license/provenance unclear; ChatGPT-named) |
| archive (6) / COD10K imported | **0** (explicitly excluded) |

**Conclusion:** No file was eligible for **verified** staging under CAND-001..044 matching rules.  
`staging/images/` remains **empty**. Original downloads were **not** moved or modified.

---

## 1. Candidate records searched

**44** rows from `metadata/candidate_sources.csv` (CAND-001 … CAND-044), cross-checked with `candidate_research.md`.

---

## 2. Downloaded images found

### A. Matched to CAND-001..044

**0**

Search used:

- `CAND-` / `CAND_` filenames  
- Commons `File:` / URL stems from the candidate CSV  
- Distinctive stems (peacock_flounder, uroplatus, stonefish, thick_forest, …)

Roots searched: Downloads, Pictures, Documents, OneDrive; Desktop unavailable.  
Excluded: `archive (6)`, `TUKU_COD10K_ARCHIVE_SOURCE`, production `dataset\`, COD10K trees, unrelated project folders.

### B. Located but **not** CAND-matched

**24** files in `C:\Users\batka\Downloads\` named:

`ChatGPT Image Sep 20, 2026, 09_15_*.png` / `09_16_*.png`

| Property | Value |
|----------|--------|
| Decodable | 24/24 OK |
| Size | all **1536×1024**, RGB, PNG |
| Unique SHA256 | 23 (1 exact duplicate pair) |
| COD10K SHA overlap | **0** |

These were **catalogued** as `UNMAPPED-CGPT-*` with `LICENSE_UNCLEAR` / `DUPLICATE`, and **not** copied into `staging/images/` because:

1. They do **not** match any CAND-001..044 ID/filename (IDs not invented)  
2. Filenames indicate **ChatGPT export** → provenance/license unclear for research ingestion  
3. Verified staging requires clear candidate match + clear license  

### C. False positives (rejected)

Animal-name substrings (e.g. “flounder”, “crocodile”) hit **COD10K classifier crops** under `Camouflage_Breaker-main\dataset\crops\Test\`.  

**Not imported** (Official Test / COD10K isolation). See `staging/rejected/FALSE_POSITIVE_COD10K_CROPS.txt`.

---

## 3. Successfully verified (for staging)

**0**

---

## 4. Missing (CAND-001..044)

**44 / 44** — all still `MISSING_LOCAL_FILE` in `collection_manifest.csv` (`collection_status=CANDIDATE`).

---

## 5. Duplicates

| Type | Count |
|------|------:|
| Exact SHA among ChatGPT Sep20 set | **1 group** (2 files share hash `c2e14ac9…73551`) |
| Exact SHA vs COD10K Train/Test | **0** (for the 24 ChatGPT files) |
| CAND↔CAND local dups | N/A (no CAND files) |

---

## 6. Unclear provenance / license

**24** ChatGPT-named files → `LICENSE_UNCLEAR` (plus one marked `DUPLICATE`).  
**14** CAND research rows previously flagged for manual license review remain relevant **once those files are acquired**.

---

## 7. Requiring annotation

If later approved for staging: all animal samples would be `NEEDS_ANNOTATION`.  
**Currently staged needing annotation:** **0**

---

## 8. Hard negatives

CAND-036..044 remain metadata-only hard-negative *candidates* — **0** local image files.

---

## 9. Rejected / not staged

| Item | Action |
|------|--------|
| archive (6) / COD10K | Excluded by policy |
| COD10K crop false positives | Rejected |
| 24 ChatGPT-named Downloads | Catalogued, **not** verified-staged |

---

## 10. Exact locations

| Path | Role |
|------|------|
| `C:\Users\batka\Downloads\ChatGPT Image Sep 20, 2026, *.png` | Unmapped discoveries (24) |
| `TUKU_DATASET_V2\staging\images\` | Empty (verified staging) |
| `TUKU_DATASET_V2\staging\rejected\` | Policy rejection notes |
| `Camouflage_Breaker-main\dataset\` | Untouched; used only for SHA anti-leak checks |

Full paths listed in `metadata/existing_image_inventory.csv`.

---

## 11. SHA256

Per-file SHA256 for all 24 ChatGPT discoveries: see `existing_image_inventory.csv` and `discovered_unmapped_chatgpt_sep20.json`.

COD10K index size used for leak check: **9987** unique Train+Test image hashes.

---

## 12. COD10K duplicate check

- **archive (6):** not imported (COD10K-v3).  
- **ChatGPT set vs COD10K:** **0** exact SHA matches.  
- **False name hits on COD10K crops:** rejected without copy.

---

## 13–15. Breakdowns (verified staged only)

All **0** (nothing verified-staged).

Research-plan breakdowns remain only in `candidate_sources.csv`.

---

## CAND-001..044 match table

| Status | IDs |
|--------|-----|
| Matched & staged | **none** |
| Missing | **CAND-001 … CAND-044 (all)** |

---

## Metadata written

| File | Content |
|------|---------|
| `metadata/collection_manifest.csv` | 44 missing CAND rows + 24 UNMAPPED-CGPT rows |
| `metadata/existing_image_inventory.csv` | 24 ChatGPT discoveries |
| `metadata/discovered_unmapped_chatgpt_sep20.json` | Probe details |
| `metadata/ingestion_report.md` | This report |

---

## Safety confirmation

| Check | Result |
|-------|--------|
| Production COD10K Train/Test modified | **No** |
| `saved_models/*` modified | **No** |
| Models / inference modified | **No** |
| train/val/hard_negative populated | **No** (still 0) |
| Masks created | **No** |
| Training | **No** |
| Original Downloads moved/deleted | **No** |

---

## Exact next step

Provide the folder that contains the **actual CAND-001..044** (or other licensed hard-camouflage) downloads — **or** explicitly authorize how to treat the 24 ChatGPT-named files (they are currently **not** verified TUKU-v2 candidates).

**STOP.** Awaiting next instruction.
