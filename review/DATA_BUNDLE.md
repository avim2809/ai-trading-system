# Data bundle (not in git)

- File: `/local/store/review_bundles/review_bundle_20261001T004018Z.tar.zst` (3.35 GB, zstd)
- SHA-256: `064cd0968f2f76605e8d1a8237cffa0cb67c2fd1e717807412971881462170c2`
- Built: 20261001T004018Z from git HEAD `5c5575792c`
- Contents: `inputs/` (every input data file, byte for byte), `large_artifacts/` (run outputs > 10 MB, e.g. edge-search step-1 logs, S4 panel, S5 per-coin cache), `MANIFEST.csv.gz`.
- Licensing: EODHD, Tiingo and cached vendor prices are licensed. Sharing this bundle is the owner's decision. A reviewer with their own subscription can re-download the vendor data with the committed scripts and compare it against `review/data/MANIFEST.csv.gz`. Vendors revise history, so some hashes may differ.
- Extract: `zstd -d review_bundle_*.tar.zst -c | tar -x`, then place `inputs/data/...` under the repo root and `inputs/scratchpad/...` in a scratch directory passed to the scripts.
