"""Tests for src/firm/data/insider_transactions.py.

All fixture data is synthesized in-memory (small hand-built TSVs matching
the exact SEC DERA schema, zipped on the fly) — no network access, matching
this dataset's schema as verified against the live 2006q1/2025q3 files and
the SEC's own insider_transactions_readme.pdf (2026-09-30).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from firm.data import insider_transactions as it

# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------

_SUBMISSION_HEADER = (
    "ACCESSION_NUMBER\tFILING_DATE\tPERIOD_OF_REPORT\tDATE_OF_ORIG_SUB\t"
    "NO_SECURITIES_OWNED\tNOT_SUBJECT_SEC16\tFORM3_HOLDINGS_REPORTED\t"
    "FORM4_TRANS_REPORTED\tDOCUMENT_TYPE\tISSUERCIK\tISSUERNAME\t"
    "ISSUERTRADINGSYMBOL\tREMARKS"
)
_OWNER_HEADER = (
    "ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNERNAME\tRPTOWNER_RELATIONSHIP\t"
    "RPTOWNER_TITLE\tRPTOWNER_TXT\tRPTOWNER_STREET1\tRPTOWNER_STREET2\t"
    "RPTOWNER_CITY\tRPTOWNER_STATE\tRPTOWNER_ZIPCODE\tRPTOWNER_STATE_DESC\tFILE_NUMBER"
)
_TRANS_HEADER = (
    "ACCESSION_NUMBER\tNONDERIV_TRANS_SK\tSECURITY_TITLE\tSECURITY_TITLE_FN\t"
    "TRANS_DATE\tTRANS_DATE_FN\tDEEMED_EXECUTION_DATE\tDEEMED_EXECUTION_DATE_FN\t"
    "TRANS_FORM_TYPE\tTRANS_CODE\tEQUITY_SWAP_INVOLVED\tEQUITY_SWAP_TRANS_CD_FN\t"
    "TRANS_TIMELINESS\tTRANS_TIMELINESS_FN\tTRANS_SHARES\tTRANS_SHARES_FN\t"
    "TRANS_PRICEPERSHARE\tTRANS_PRICEPERSHARE_FN\tTRANS_ACQUIRED_DISP_CD\t"
    "TRANS_ACQUIRED_DISP_CD_FN\tSHRS_OWND_FOLWNG_TRANS\tSHRS_OWND_FOLWNG_TRANS_FN\t"
    "VALU_OWND_FOLWNG_TRANS\tVALU_OWND_FOLWNG_TRANS_FN\tDIRECT_INDIRECT_OWNERSHIP\t"
    "DIRECT_INDIRECT_OWNERSHIP_FN\tNATURE_OF_OWNERSHIP\tNATURE_OF_OWNERSHIP_FN"
)


def _submission_row(acc, filing_date, doc_type, issuer_cik, issuer_name, ticker, period="15-JAN-2020"):
    return "\t".join([
        acc, filing_date, period, "", "0", "", "", "", doc_type,
        issuer_cik, issuer_name, ticker, "",
    ])


def _owner_row(acc, owner_cik, owner_name, relationship="Officer"):
    return "\t".join([
        acc, owner_cik, owner_name, relationship, "VP", "", "1 MAIN ST", "",
        "NYC", "NY", "10001", "", "",
    ])


def _trans_row(acc, sk, trans_date, trans_code, shares, price, acq_disp="A", shrs_following="1000"):
    return "\t".join([
        acc, str(sk), "Common Stock", "", trans_date, "", "", "", "4", trans_code,
        "", "", "", "", str(shares), "", str(price), "", acq_disp, "",
        shrs_following, "", "", "", "D", "", "", "",
    ])


def _make_quarter_zip(tmp_path: Path, name: str, submission_rows, owner_rows, trans_rows) -> Path:
    zpath = tmp_path / name
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("SUBMISSION.tsv", "\n".join([_SUBMISSION_HEADER, *submission_rows]) + "\n")
        zf.writestr("REPORTINGOWNER.tsv", "\n".join([_OWNER_HEADER, *owner_rows]) + "\n")
        zf.writestr("NONDERIV_TRANS.tsv", "\n".join([_TRANS_HEADER, *trans_rows]) + "\n")
    return zpath


@pytest.fixture
def sample_zip(tmp_path) -> Path:
    """One synthetic quarter:

    A1: real open-market purchase, owner O1 buys issuer CIK 1000001 (ticker ABCD).
    A2: a later Form 4/A re-filing the *same* trade as A1 (should collapse in dedupe).
    A3: a SALE (code S) by O1 in the same issuer — must never appear in the purchases
        table but must count toward O1's activity calendar.
    A4: a purchase in issuer CIK 1000002 with a BLANK filed ticker (tests the
        company_tickers.json fallback path).
    A5: a code "P" row incorrectly marked Disposed (D) rather than Acquired (A) —
        must be dropped, not counted as a buy.
    """
    submissions = [
        _submission_row("0001-20-000001", "07-JAN-2020", "4", "1000001", "ABC CORP", "ABCD"),
        _submission_row("0001-20-000002", "10-JAN-2020", "4/A", "1000001", "ABC CORP", "ABCD"),
        _submission_row("0001-20-000003", "03-FEB-2020", "4", "1000001", "ABC CORP", "ABCD"),
        _submission_row("0001-20-000004", "15-MAR-2020", "4", "1000002", "XYZ CORP", ""),
        _submission_row("0001-20-000005", "20-MAR-2020", "4", "1000003", "UNKNOWN CORP", ""),
    ]
    owners = [
        _owner_row("0001-20-000001", "2000001", "INSIDER ONE"),
        _owner_row("0001-20-000002", "2000001", "INSIDER ONE"),
        _owner_row("0001-20-000003", "2000001", "INSIDER ONE"),
        _owner_row("0001-20-000004", "2000002", "INSIDER TWO"),
        _owner_row("0001-20-000005", "2000003", "INSIDER THREE"),
    ]
    trans = [
        _trans_row("0001-20-000001", 1, "05-JAN-2020", "P", 100, "10.00"),
        _trans_row("0001-20-000002", 1, "05-JAN-2020", "P", 100, "10.00"),  # same trade, amended
        _trans_row("0001-20-000003", 2, "01-FEB-2020", "S", 50, "12.00", acq_disp="D"),
        _trans_row("0001-20-000004", 3, "14-MAR-2020", "P", 200, "5.00"),
        _trans_row("0001-20-000005", 4, "19-MAR-2020", "P", 300, "7.00", acq_disp="D"),  # malformed
    ]
    return _make_quarter_zip(tmp_path, "2020q1_form345.zip", submissions, owners, trans)


# ---------------------------------------------------------------------------
# parse_quarter_purchases
# ---------------------------------------------------------------------------

class TestParseQuarterPurchases:
    def test_filters_to_open_market_purchases_only(self, sample_zip):
        df = it.parse_quarter_purchases(sample_zip)
        # A3 (sale) and A5 (malformed P/D) must be excluded; A1, A2, A4 remain.
        assert set(df["accession_number"]) == {
            "0001-20-000001", "0001-20-000002", "0001-20-000004",
        }

    def test_known_date_is_filing_date_not_trans_date(self, sample_zip):
        df = it.parse_quarter_purchases(sample_zip)
        row = df[df["accession_number"] == "0001-20-000001"].iloc[0]
        assert row["known_date"] == pd.Timestamp("2020-01-07")
        assert row["trans_date"] == pd.Timestamp("2020-01-05")
        assert row["known_date"] > row["trans_date"]

    def test_blank_ticker_preserved_as_na_not_dropped(self, sample_zip):
        df = it.parse_quarter_purchases(sample_zip)
        row = df[df["accession_number"] == "0001-20-000004"].iloc[0]
        assert pd.isna(row["issuer_ticker_filed"])

    def test_missing_member_raises(self, tmp_path):
        bad_zip = tmp_path / "broken.zip"
        with zipfile.ZipFile(bad_zip, "w") as zf:
            zf.writestr("SUBMISSION.tsv", _SUBMISSION_HEADER + "\n")
        with pytest.raises(FileNotFoundError):
            it.parse_quarter_purchases(bad_zip)


class TestParseQuarterOwnerActivity:
    def test_includes_every_trans_code_not_only_purchases(self, sample_zip):
        activity = it.parse_quarter_owner_activity(sample_zip)
        # Owner 2000001 traded in Jan (purchase) and Feb (sale) -> two distinct months.
        owner1 = activity[activity["owner_cik"] == "2000001"]
        assert set(owner1["year_month"]) == {"2020-01", "2020-02"}

    def test_distinct_owner_month_pairs_only(self, sample_zip):
        activity = it.parse_quarter_owner_activity(sample_zip)
        # A1 and A2 are both owner 2000001 in January -> collapsed to one row.
        owner1_jan = activity[(activity["owner_cik"] == "2000001") & (activity["year_month"] == "2020-01")]
        assert len(owner1_jan) == 1


class TestDedupeAmendments:
    def test_collapses_amendment_to_latest_filing(self, sample_zip):
        raw = it.parse_quarter_purchases(sample_zip)
        deduped = it.dedupe_amendments(raw)
        # A1 and A2 describe the identical trade -> exactly one survives, and it's
        # the one with the LATER known_date (the amendment, A2).
        matches = deduped[
            (deduped["issuer_cik"] == "1000001") & (deduped["trans_date"] == pd.Timestamp("2020-01-05"))
        ]
        assert len(matches) == 1
        assert matches.iloc[0]["accession_number"] == "0001-20-000002"

    def test_distinct_trades_are_not_collapsed(self, sample_zip):
        raw = it.parse_quarter_purchases(sample_zip)
        deduped = it.dedupe_amendments(raw)
        assert "0001-20-000004" in set(deduped["accession_number"])

    def test_empty_input(self):
        empty = it._empty_purchases_frame()
        assert it.dedupe_amendments(empty).empty


class TestResolveTickers:
    def test_prefers_as_filed_ticker(self, sample_zip):
        raw = it.parse_quarter_purchases(sample_zip)
        company_tickers = pd.DataFrame({"cik": ["1000001"], "ticker": ["WRONG"], "company_name": ["x"]})
        resolved = it.resolve_tickers(raw, company_tickers)
        row = resolved[resolved["accession_number"] == "0001-20-000001"].iloc[0]
        assert row["ticker"] == "ABCD"
        assert row["ticker_source"] == "filed"

    def test_falls_back_to_company_tickers_when_blank(self, sample_zip):
        raw = it.parse_quarter_purchases(sample_zip)
        company_tickers = pd.DataFrame({"cik": ["1000002"], "ticker": ["WXYZ"], "company_name": ["XYZ CORP"]})
        resolved = it.resolve_tickers(raw, company_tickers)
        row = resolved[resolved["accession_number"] == "0001-20-000004"].iloc[0]
        assert row["ticker"] == "WXYZ"
        assert row["ticker_source"] == "company_tickers_current"

    def test_multi_class_issuer_in_company_tickers_does_not_fan_out_rows(self, sample_zip):
        # Regression: SEC's company_tickers.json lists more than one row for a
        # single CIK when an issuer has multiple listed share classes (e.g. a
        # GOOG/GOOGL-style dual ticker). A naive left-merge on cik silently
        # duplicates every purchase row for that issuer once per alias.
        raw = it.parse_quarter_purchases(sample_zip)
        n_before = len(raw)
        company_tickers = pd.DataFrame({
            "cik": ["1000001", "1000001", "1000002"],
            "ticker": ["ABCD", "ABCD.B", "WXYZ"],
            "company_name": ["ABC CORP", "ABC CORP", "XYZ CORP"],
        })
        resolved = it.resolve_tickers(raw, company_tickers)
        assert len(resolved) == n_before

    def test_unresolved_when_no_fallback_available(self, sample_zip):
        # Add a fabricated purchase for issuer 1000003, which has no filed ticker
        # and no entry in company_tickers.json either (a genuinely delisted issuer).
        raw = it.parse_quarter_purchases(sample_zip)
        extra = raw.iloc[[0]].copy()
        extra["issuer_cik"] = "1000003"
        extra["issuer_ticker_filed"] = pd.NA
        extra["accession_number"] = "0001-20-999999"
        raw2 = pd.concat([raw, extra], ignore_index=True)
        company_tickers = pd.DataFrame({"cik": ["1000001"], "ticker": ["ABCD"], "company_name": ["x"]})
        resolved = it.resolve_tickers(raw2, company_tickers)
        row = resolved[resolved["accession_number"] == "0001-20-999999"].iloc[0]
        assert row["ticker_source"] == "unresolved"
        assert pd.isna(row["ticker"])


class TestFetchCompanyTickers:
    def test_reads_and_normalizes_cached_json_without_network(self, tmp_path):
        payload = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
        (tmp_path / "company_tickers.json").write_text(json.dumps(payload))
        df = it.fetch_company_tickers(tmp_path, max_age_days=99999)
        assert df.iloc[0]["ticker"] == "AAPL"
        assert df.iloc[0]["cik"] == "0000320193"


class TestRateLimiter:
    def test_enforces_minimum_interval(self):
        import time
        limiter = it.RateLimiter(max_per_second=20.0)  # 50ms min interval
        t0 = time.monotonic()
        limiter.wait()
        limiter.wait()
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.045  # allow small scheduling slack


class TestDiscoverQuarters:
    def test_parses_quarter_labels_from_hrefs(self, monkeypatch):
        class FakeResp:
            text = (
                '<a href="/files/structureddata/data/insider-transactions-data-sets/2006q1_form345.zip">x</a>'
                '<a href="/files/datastandardsinnovation/data/insider-transactions-data-sets/2026q2_form345.zip">y</a>'
            )
            def raise_for_status(self):
                pass

        class FakeSession:
            def get(self, url, timeout=None):
                return FakeResp()

        out = it.discover_quarters(session=FakeSession(), rate_limiter=it.RateLimiter(max_per_second=1000))
        assert out["2006q1"].endswith("2006q1_form345.zip")
        assert out["2026q2"].startswith("https://www.sec.gov")
