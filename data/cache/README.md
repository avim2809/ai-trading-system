# Data cache layout

Parquet files are keyed by SHA-256 hashes under this directory (`firm.data.cache.ParquetCache`).

## Combined panels (backtest `data_source: cache`)

| Cache key | Purpose |
|-----------|---------|
| `combined/prices` | OHLCV panel (symbol × date) |
| `combined/fundamentals` | Point-in-time fundamentals |
| `combined/universe_membership` | Index membership windows (`index`, `symbol`, `added_date`, `removed_date`) |

Import membership from vendor CSV/Parquet:

```bash
python scripts/import_universe_membership.py path/to/membership.csv --index SP500
```

Sharadar bulk export (SEP/SF1 + membership):

```bash
python scripts/etl_sharadar_to_cache.py --prices SEP.csv --fundamentals SF1.csv \\
  --membership sp500_membership.csv --index SP500 --merge
```

Tiingo free-tier backfill (live 25-symbol universe → 2010+):

```bash
python scripts/backfill_tiingo_prices.py
```

See `docs/longer_dataset_options.md` for vendor options and schema notes.
