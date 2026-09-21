# TUKU-v2 tools

## qa_dataset.py

Read-only dataset QA for `TUKU_DATASET_V2`.

```bash
python tools/qa_dataset.py audit
python tools/qa_dataset.py audit --root "C:\Users\batka\Downloads\TUKU DEEP LEARING\TUKU_DATASET_V2"
```

Does **not** modify images, masks, or metadata. Does **not** touch production `Camouflage_Breaker-main\dataset`.

Near-duplicate threshold: 8×8 average-hash Hamming distance ≤ 5.
