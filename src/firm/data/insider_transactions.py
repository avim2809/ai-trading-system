"""SEC EDGAR Form 3/4/5 "Insider Transactions Data Sets" — download, cache, parse.

Source: SEC's Division of Economic and Risk Analysis (DERA) publishes the
XBRL/XML-derived content of every Form 3, 4 and 5 filing as flat, tab-separated
quarterly zip files, back to 2006 Q1:

    https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets

Each quarterly zip contains up to 8 tables (schema per the SEC's own
``insider_transactions_readme.pdf``, verified 2026-09-30 against the live
2006q1 and 2025q3 files): SUBMISSION, REPORTINGOWNER, NONDERIV_TRANS,
NONDERIV_HOLDING, DERIV_TRANS, DERIV_HOLDING, FOOTNOTES, OWNER_SIGNATURE. This
module only ever reads SUBMISSION, REPORTINGOWNER and NONDERIV_TRANS — the
other five tables (derivative positions, footnote text, signature blocks) are
never extracted, since they are irrelevant to open-market common-stock
purchases and roughly double the CPU/IO cost of a full backfill.

Point-in-time discipline
-------------------------
The dataset's only knowable-date field is ``SUBMISSION.FILING_DATE`` (a
calendar DATE, no intraday timestamp — the DERA extract does not carry
EDGAR's acceptance datetime). This module therefore treats ``FILING_DATE`` as
the day a filing's content is *known*, and NEVER exposes ``TRANS_DATE``
(the transaction's execution date, which can lag its filing by up to the
statutory 2-business-day Section 16 deadline, sometimes longer for
late/amended filings) as a usable "as of" date. Callers building a tradable
signal must additionally lag to the next trading session after
``known_date`` (filings are commonly submitted after the 5:30pm ET EDGAR
cutoff), matching this system's ``pit_store`` / execution-lag convention
elsewhere (see CLAUDE.md).

SEC fair-access rules (verified against the EDGAR developer docs, 2026-09-30):
max 10 requests/second, a descriptive User-Agent naming the application and a
contact. This module defaults to the contact used for this research pass;
override via ``user_agent=`` or the ``FIRM_SEC_EDGAR_USER_AGENT`` env var
before any production/live use.

Known gaps (quantify, don't hide):
    - Ticker resolution: SUBMISSION.ISSUERTRADINGSYMBOL is the ticker *as
      filed at the time*, which is the best available point-in-time ticker
      and is used first. Where it's blank, this module falls back to SEC's
      ``company_tickers.json`` (CIK -> ticker), but that file only lists
      *currently* SEC-reporting companies — it cannot recover the historical
      ticker of an issuer that has since been delisted, acquired, or
      renamed. ``resolve_tickers()`` reports the exact unresolved count so
      this gap is measured, not assumed.
    - Amendment de-duplication: the DERA tables do not carry a link from a
      Form 4/A back to the accession number it amends (only
      ``DATE_OF_ORIG_SUB``, a date). ``dedupe_amendments()`` uses a
      heuristic identity key (issuer, owner, transaction date, shares,
      price) and keeps the most recently filed row — documented as an
      approximation, not a guaranteed 1:1 amendment linkage.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

SEC_INDEX_URL = "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
DEFAULT_USER_AGENT = os.environ.get(
    "FIRM_SEC_EDGAR_USER_AGENT", "ai-trading-system research avim2809@gmail.com"
)
MAX_REQUESTS_PER_SECOND = 10.0

# Open-market/private purchase, per the readme's Trans Code List (Appendix 6.2).
# Note this code also covers non-derivative *private* purchases (not only
# open-market), a known imprecision inherited from the source data and from
# the CMP (2012) methodology itself, which uses the identical proxy.
PURCHASE_TRANS_CODE = "P"
ACQUIRED_CODE = "A"

_SUBMISSION_COLS = [
    "ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT", "DATE_OF_ORIG_SUB",
    "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERNAME", "ISSUERTRADINGSYMBOL",
]
_REPORTINGOWNER_COLS = [
    "ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE",
]
_NONDERIV_TRANS_COLS = [
    "ACCESSION_NUMBER", "NONDERIV_TRANS_SK", "TRANS_DATE", "TRANS_CODE",
    "TRANS_SHARES", "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD",
    "SHRS_OWND_FOLWNG_TRANS", "DIRECT_INDIRECT_OWNERSHIP",
]


class RateLimiter:
    """Simple token-paced limiter honoring SEC EDGAR's 10 req/sec fair-access cap."""

    def __init__(self, max_per_second: float = MAX_REQUESTS_PER_SECOND):
        self._min_interval = 1.0 / max_per_second
        self._last_call = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()


@dataclass
class QuarterSpec:
    label: str  # e.g. "2006q1"
    url: str


def new_session(user_agent: str) -> requests.Session:
    """Build a requests.Session carrying the SEC-required descriptive User-Agent."""
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent})
    return s


# Internal alias kept for brevity within this module.
_session = new_session


def discover_quarters(
    user_agent: str = DEFAULT_USER_AGENT,
    session: requests.Session | None = None,
    rate_limiter: RateLimiter | None = None,
) -> dict[str, str]:
    """Scrape the SEC index page for every available quarterly zip URL.

    Returns ``{"2006q1": "https://www.sec.gov/files/.../2006q1_form345.zip", ...}``.
    Scraping (rather than hardcoding a single URL-path template) is
    deliberate: SEC has changed the path prefix at least once mid-history
    (``structureddata`` through 2026q1, ``datastandardsinnovation`` from
    2026q2 — verified 2026-09-30), so a fixed template silently drops the
    newest quarter(s).
    """
    sess = session or _session(user_agent)
    limiter = rate_limiter or RateLimiter()
    limiter.wait()
    resp = sess.get(SEC_INDEX_URL, timeout=60)
    resp.raise_for_status()
    hrefs = re.findall(r'href="([^"]*form345[^"]*\.zip)"', resp.text)
    out: dict[str, str] = {}
    for href in hrefs:
        m = re.search(r"(\d{4}q[1-4])_form345\.zip", href)
        if not m:
            continue
        label = m.group(1)
        url = href if href.startswith("http") else f"https://www.sec.gov{href}"
        out[label] = url
    log.info("discovered %d insider-transactions quarters (%s .. %s)",
              len(out), min(out) if out else "-", max(out) if out else "-")
    return out


def download_quarter(
    label: str,
    url: str,
    cache_dir: Path,
    user_agent: str = DEFAULT_USER_AGENT,
    session: requests.Session | None = None,
    rate_limiter: RateLimiter | None = None,
) -> Path:
    """Download one quarter's zip to ``cache_dir/raw/`` if not already cached.

    ``cache_dir`` must be supplied by the caller — this module never
    hardcodes a default inside the repo checkout (see CLAUDE.md: this
    dataset is bulk external research data, not a repo asset).
    """
    raw_dir = Path(cache_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{label}_form345.zip"
    if dest.exists() and dest.stat().st_size > 0:
        log.debug("insider quarter %s already cached at %s", label, dest)
        return dest

    sess = session or _session(user_agent)
    limiter = rate_limiter or RateLimiter()
    limiter.wait()
    log.info("downloading insider-transactions quarter %s from %s", label, url)
    resp = sess.get(url, timeout=120)
    resp.raise_for_status()
    tmp = dest.with_suffix(".zip.part")
    tmp.write_bytes(resp.content)
    tmp.rename(dest)
    log.info("cached insider quarter %s: %d bytes -> %s", label, len(resp.content), dest)
    return dest


def _read_member_tsv(zf: zipfile.ZipFile, member: str, usecols: list[str]) -> pd.DataFrame:
    with zf.open(member) as fh:
        df = pd.read_csv(
            fh, sep="\t", encoding="utf-8", dtype=str, usecols=lambda c: c in usecols,
            low_memory=False, na_values=[""], keep_default_na=True,
        )
    return df


def parse_quarter_purchases(zip_path: Path) -> pd.DataFrame:
    """Parse one quarterly zip into open-market/private *purchase* rows.

    Filters ``NONDERIV_TRANS`` to ``TRANS_CODE == "P"`` and
    ``TRANS_ACQUIRED_DISP_CD == "A"`` (an acquisition, not a disposition —
    code P is almost always "A" but a handful of malformed filings set "D";
    those are dropped rather than silently counted as buys) *before*
    joining to SUBMISSION/REPORTINGOWNER, which keeps peak memory to a small
    fraction of the full quarterly table (non-derivative purchases are a
    small minority of all Table I rows — sales, grants and option exercises
    dominate).

    Non-derivative transactions only: derivative (options/RSUs) purchases
    are out of scope for this candidate (see docs/research brief — the
    signal is common-stock open-market buying).
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        missing = {"SUBMISSION.tsv", "REPORTINGOWNER.tsv", "NONDERIV_TRANS.tsv"} - names
        if missing:
            raise FileNotFoundError(f"{zip_path}: missing expected member(s) {missing}")

        trans = _read_member_tsv(zf, "NONDERIV_TRANS.tsv", _NONDERIV_TRANS_COLS)
        n_total = len(trans)
        trans = trans[
            (trans["TRANS_CODE"] == PURCHASE_TRANS_CODE)
            & (trans["TRANS_ACQUIRED_DISP_CD"] == ACQUIRED_CODE)
        ].copy()
        log.info("%s: %d/%d NONDERIV_TRANS rows are open-market/private purchases (code P, acquired)",
                  zip_path.name, len(trans), n_total)
        if trans.empty:
            return _empty_purchases_frame()

        sub = _read_member_tsv(zf, "SUBMISSION.tsv", _SUBMISSION_COLS)
        owners = _read_member_tsv(zf, "REPORTINGOWNER.tsv", _REPORTINGOWNER_COLS)

    merged = trans.merge(sub, on="ACCESSION_NUMBER", how="left")
    merged = merged.merge(owners, on="ACCESSION_NUMBER", how="left")

    for col in ("TRANS_SHARES", "TRANS_PRICEPERSHARE", "SHRS_OWND_FOLWNG_TRANS"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["FILING_DATE"] = pd.to_datetime(merged["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
    merged["TRANS_DATE"] = pd.to_datetime(merged["TRANS_DATE"], format="%d-%b-%Y", errors="coerce")

    out = pd.DataFrame({
        "accession_number": merged["ACCESSION_NUMBER"],
        "nonderiv_trans_sk": merged["NONDERIV_TRANS_SK"],
        "known_date": merged["FILING_DATE"],       # point-in-time knowable date
        "trans_date": merged["TRANS_DATE"],         # NOT usable as an "as of" date
        "document_type": merged["DOCUMENT_TYPE"],
        "issuer_cik": merged["ISSUERCIK"],
        "issuer_name": merged["ISSUERNAME"],
        "issuer_ticker_filed": merged["ISSUERTRADINGSYMBOL"].str.strip().replace("", pd.NA),
        "owner_cik": merged["RPTOWNERCIK"],
        "owner_name": merged["RPTOWNERNAME"],
        "owner_relationship": merged["RPTOWNER_RELATIONSHIP"],
        "trans_shares": merged["TRANS_SHARES"],
        "trans_price_per_share": merged["TRANS_PRICEPERSHARE"],
        "shares_owned_following": merged["SHRS_OWND_FOLWNG_TRANS"],
    })
    n_bad_dates = out["known_date"].isna().sum() + out["trans_date"].isna().sum()
    if n_bad_dates:
        log.warning("%s: %d rows with unparseable FILING_DATE/TRANS_DATE", zip_path.name, n_bad_dates)
    return out.dropna(subset=["known_date", "trans_date", "issuer_cik", "owner_cik"])


def _empty_purchases_frame() -> pd.DataFrame:
    cols = [
        "accession_number", "nonderiv_trans_sk", "known_date", "trans_date", "document_type",
        "issuer_cik", "issuer_name", "issuer_ticker_filed", "owner_cik", "owner_name",
        "owner_relationship", "trans_shares", "trans_price_per_share", "shares_owned_following",
    ]
    return pd.DataFrame(columns=cols)


def parse_quarter_owner_activity(zip_path: Path) -> pd.DataFrame:
    """Distinct (owner_cik, calendar year-month) pairs from ALL non-derivative
    transactions in this quarter (every TRANS_CODE, not just purchases).

    This is the minimal input Cohen/Malloy/Pomorski's "routine vs
    opportunistic" classifier needs: it asks whether an insider traded in
    the *same calendar month* in each of the prior three years, using ANY
    transaction (sale, purchase, grant, exercise ...), not only purchases.
    Keeping only the distinct (owner, month) pair — not the full transaction
    — keeps this table two to three orders of magnitude smaller than the
    full NONDERIV_TRANS table.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        missing = {"REPORTINGOWNER.tsv", "NONDERIV_TRANS.tsv"} - names
        if missing:
            raise FileNotFoundError(f"{zip_path}: missing expected member(s) {missing}")
        trans = _read_member_tsv(zf, "NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "TRANS_DATE"])
        owners = _read_member_tsv(zf, "REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK"])

    merged = trans.merge(owners, on="ACCESSION_NUMBER", how="inner")
    merged["TRANS_DATE"] = pd.to_datetime(merged["TRANS_DATE"], format="%d-%b-%Y", errors="coerce")
    merged = merged.dropna(subset=["TRANS_DATE", "RPTOWNERCIK"])
    merged["year_month"] = merged["TRANS_DATE"].dt.to_period("M").astype(str)
    out = merged[["RPTOWNERCIK", "year_month"]].drop_duplicates()
    out.columns = ["owner_cik", "year_month"]
    return out.reset_index(drop=True)


def dedupe_amendments(purchases: pd.DataFrame) -> pd.DataFrame:
    """Collapse Form 4 / Form 4-A duplicate rows describing the same trade.

    Heuristic identity key: (issuer_cik, owner_cik, trans_date, trans_shares,
    trans_price_per_share). The DERA tables carry no explicit
    amendment -> original accession-number link (only ``DATE_OF_ORIG_SUB``,
    itself often blank on the original), so an exact 1:1 linkage isn't
    reconstructable from this dataset alone. Keeps the row with the latest
    ``known_date`` (falling back to the lexicographically-last accession
    number, which is monotone in filing sequence within a day) per key.
    """
    if purchases.empty:
        return purchases
    before = len(purchases)
    key = ["issuer_cik", "owner_cik", "trans_date", "trans_shares", "trans_price_per_share"]
    deduped = (
        purchases.sort_values(["known_date", "accession_number"])
        .drop_duplicates(subset=key, keep="last")
        .reset_index(drop=True)
    )
    log.info("dedupe_amendments: %d -> %d rows (%d likely amendment/duplicate rows collapsed)",
              before, len(deduped), before - len(deduped))
    return deduped


def fetch_company_tickers(
    cache_dir: Path,
    user_agent: str = DEFAULT_USER_AGENT,
    session: requests.Session | None = None,
    rate_limiter: RateLimiter | None = None,
    max_age_days: int = 7,
) -> pd.DataFrame:
    """Download/cache SEC's CIK -> current-ticker map.

    KNOWN GAP: this file lists only *currently* SEC-reporting companies. An
    issuer delisted, acquired, or renamed before today will not appear here
    (or will appear under a ticker that doesn't match what was filed
    historically). It is used only as a fallback when a filing's own
    ``ISSUERTRADINGSYMBOL`` is blank — see ``resolve_tickers()``.
    """
    cache_path = Path(cache_dir) / "company_tickers.json"
    stale = True
    if cache_path.exists():
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        stale = age_days > max_age_days
    if stale:
        sess = session or _session(user_agent)
        limiter = rate_limiter or RateLimiter()
        limiter.wait()
        resp = sess.get(COMPANY_TICKERS_URL, timeout=60)
        resp.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(resp.text)
        log.info("refreshed company_tickers.json cache -> %s", cache_path)
    payload = json.loads(cache_path.read_text())
    df = pd.DataFrame(payload.values())
    df["cik_str"] = df["cik_str"].astype(str).str.zfill(10)
    return df.rename(columns={"cik_str": "cik", "title": "company_name"})


def resolve_tickers(purchases: pd.DataFrame, company_tickers: pd.DataFrame) -> pd.DataFrame:
    """Attach a resolved ``ticker``/``ticker_source`` column and log the gap.

    Preference order: (1) the ticker as filed at the time
    (``issuer_ticker_filed``) — the only truly point-in-time-correct source;
    (2) SEC's current CIK->ticker map, which recovers the issuer's *current*
    ticker only (wrong for anything renamed since, and absent entirely for
    anything delisted/acquired/gone private since); (3) unresolved.
    """
    out = purchases.copy()
    out["issuer_cik_norm"] = out["issuer_cik"].astype(str).str.zfill(10)
    ct = company_tickers[["cik", "ticker"]].rename(columns={"ticker": "_ct_ticker"}).copy()
    # Normalize independently of whether the caller already zero-padded CIKs
    # (fetch_company_tickers() does; a caller-supplied frame in tests/ad-hoc
    # use might not) — merge on a consistently-padded key either way.
    ct["cik"] = ct["cik"].astype(str).str.zfill(10)
    # SEC's company_tickers.json lists more than one row per CIK for issuers
    # with multiple share classes/tickers (e.g. Alphabet's CIK carries both
    # GOOG and GOOGL) — a left merge without deduping fans a single purchase
    # row out into one row per alias ticker, silently inflating row counts.
    # Keep the first (lowest-index / "primary") alias per CIK.
    ct = ct.drop_duplicates(subset="cik", keep="first")
    out = out.merge(ct, left_on="issuer_cik_norm", right_on="cik", how="left")

    has_filed = out["issuer_ticker_filed"].notna()
    out["ticker"] = out["issuer_ticker_filed"]
    out.loc[~has_filed, "ticker"] = out.loc[~has_filed, "_ct_ticker"]
    out["ticker_source"] = "unresolved"
    out.loc[has_filed, "ticker_source"] = "filed"
    out.loc[(~has_filed) & out["_ct_ticker"].notna(), "ticker_source"] = "company_tickers_current"

    n = len(out)
    n_filed = int((out["ticker_source"] == "filed").sum())
    n_current = int((out["ticker_source"] == "company_tickers_current").sum())
    n_unresolved = int((out["ticker_source"] == "unresolved").sum())
    log.info(
        "resolve_tickers: %d/%d (%.1f%%) from as-filed ISSUERTRADINGSYMBOL, "
        "%d/%d (%.1f%%) recovered from current company_tickers.json (delisted/renamed issuers "
        "cannot be recovered this way), %d/%d (%.1f%%) unresolved",
        n_filed, n, 100 * n_filed / n if n else 0.0,
        n_current, n, 100 * n_current / n if n else 0.0,
        n_unresolved, n, 100 * n_unresolved / n if n else 0.0,
    )
    return out.drop(columns=["issuer_cik_norm", "cik", "_ct_ticker"])


def load_cached_purchases(cache_dir: Path, quarters: list[str]) -> pd.DataFrame:
    """Load already-parsed per-quarter purchase parquet files from ``cache_dir/parsed``."""
    parsed_dir = Path(cache_dir) / "parsed"
    frames = []
    for q in quarters:
        path = parsed_dir / f"purchases_{q}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
        else:
            log.warning("no parsed purchases cached for quarter %s at %s", q, path)
    if not frames:
        return _empty_purchases_frame()
    return pd.concat(frames, ignore_index=True)
