"""Download + parse SEC's quarterly Insider Transactions Data Sets (Form 3/4/5).

Builds the point-in-time open-market-purchase table and the owner trading-
activity calendar (for CMP routine/opportunistic classification) that
``src/firm/strategies/insider_cluster.py`` consumes.

Everything is written under ``--cache-dir`` (never inside the repo — this is
bulk external research data, see CLAUDE.md and
``src/firm/data/insider_transactions.py``'s module docstring):

    <cache-dir>/raw/<quarter>_form345.zip          one per quarter, as downloaded
    <cache-dir>/company_tickers.json               CIK -> current ticker
    <cache-dir>/parsed/purchases_<quarter>.parquet  per-quarter purchase rows
    <cache-dir>/parsed/activity_<quarter>.parquet   per-quarter owner-month activity
    <cache-dir>/point_in_time_purchases.parquet     combined, deduped, ticker-resolved
    <cache-dir>/owner_activity_calendar.parquet     combined owner-month activity

Safe to re-run: quarters already downloaded/parsed are skipped. SEC fair
access: <= 10 requests/second (RateLimiter), descriptive User-Agent required.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running via the repo's venv without an editable install.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from firm.data import insider_transactions as it

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True, help="Output/cache directory (NOT inside the repo)")
    ap.add_argument("--quarters", nargs="*", default=None,
                     help="Specific quarters e.g. 2006q1 2012q4 (default: all discovered)")
    ap.add_argument("--user-agent", default=it.DEFAULT_USER_AGENT)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cache_dir = Path(args.cache_dir).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    if cache_dir.is_relative_to(repo_root):
        raise SystemExit(f"refusing to cache inside the repo checkout ({repo_root}): {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir = cache_dir / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    session = it.new_session(args.user_agent)
    limiter = it.RateLimiter()

    available = it.discover_quarters(user_agent=args.user_agent, session=session, rate_limiter=limiter)
    quarters = args.quarters or sorted(available)
    unknown = [q for q in quarters if q not in available]
    if unknown:
        log.warning("requested quarters not found on SEC's index page: %s", unknown)
    quarters = [q for q in quarters if q in available]
    log.info("processing %d quarters: %s .. %s", len(quarters), quarters[0] if quarters else "-", quarters[-1] if quarters else "-")

    company_tickers = it.fetch_company_tickers(cache_dir, user_agent=args.user_agent, session=session, rate_limiter=limiter)

    n_ok, n_failed = 0, 0
    for label in quarters:
        purchases_out = parsed_dir / f"purchases_{label}.parquet"
        activity_out = parsed_dir / f"activity_{label}.parquet"
        if purchases_out.exists() and activity_out.exists():
            log.info("quarter %s already parsed, skipping", label)
            n_ok += 1
            continue
        try:
            zip_path = it.download_quarter(
                label, available[label], cache_dir, user_agent=args.user_agent,
                session=session, rate_limiter=limiter,
            )
            purchases = it.parse_quarter_purchases(zip_path)
            purchases.to_parquet(purchases_out, index=False)
            activity = it.parse_quarter_owner_activity(zip_path)
            activity.to_parquet(activity_out, index=False)
            log.info("quarter %s: %d purchase rows, %d owner-month activity rows",
                      label, len(purchases), len(activity))
            n_ok += 1
        except Exception:
            log.exception("failed to process quarter %s — leaving it out of the combined table", label)
            n_failed += 1

    log.info("per-quarter processing done: %d ok, %d failed", n_ok, n_failed)

    # Combine, dedupe amendments, resolve tickers.
    all_purchases = it.load_cached_purchases(cache_dir, quarters)
    log.info("combined raw purchase rows across %d quarters: %d", len(quarters), len(all_purchases))
    deduped = it.dedupe_amendments(all_purchases)
    resolved = it.resolve_tickers(deduped, company_tickers)
    resolved.to_parquet(cache_dir / "point_in_time_purchases.parquet", index=False)
    log.info("wrote %d point-in-time purchase rows -> %s", len(resolved), cache_dir / "point_in_time_purchases.parquet")

    activity_frames = []
    for label in quarters:
        path = parsed_dir / f"activity_{label}.parquet"
        if path.exists():
            activity_frames.append(pd.read_parquet(path))
    if activity_frames:
        combined_activity = pd.concat(activity_frames, ignore_index=True).drop_duplicates()
        combined_activity.to_parquet(cache_dir / "owner_activity_calendar.parquet", index=False)
        log.info("wrote %d owner-month activity rows -> %s", len(combined_activity), cache_dir / "owner_activity_calendar.parquet")


if __name__ == "__main__":
    main()
