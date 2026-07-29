# LLM RAG look-ahead audit

Prompted by the P/L-improvement research pass (2026-07-29): the literature's
single biggest caveat on LLM-sentiment/fundamental alpha is **look-ahead
bias** — a model seeing (or retrieving) information that wasn't actually
available at decision time. This is the gate before trusting Arm B
(`llm_enhanced`) results in `docs/llm_ab_experiment_log.md`: does the as-of
retrieval guard actually hold, end to end?

## What was checked

**The retrieval layer itself (`src/firm/rag/retriever.py`, `store.py`)** —
both the dense (Chroma) and lexical (BM25) channels take an explicit `asof`
and are documented to only return documents whose `metadata["date"] <= asof`.
This part was already correctly designed. Two gaps were found and fixed (see
below): a crash bug in the dense channel's date check, and duplicated
fail-closed logic between the two channels.

**Every ingestor's date source (`src/firm/rag/ingestors/`)** — the retrieval
guard is only as good as the dates it's filtering on. Checked what raw value
each ingestor feeds into `firm.rag.dates.normalize_date`:

| Collection | Ingestor | Date source | Verdict |
|---|---|---|---|
| `sec_filings` | `sec_ingestor.py` | `file_date` from SEC EDGAR's own metadata (real filing date) | ✅ Correct — genuine availability date |
| `news` | `news_ingestor.py` | Provider-native publish timestamp (RSS `pubDate`, `publishedDate`, `time_published`, `published_utc`) | ✅ Correct — genuine publish date |
| `research` | `research_ingestor.py` | arXiv `published` date | ✅ Correct — genuine publish date |
| `system_docs` | `system_ingestor.py` | `ALWAYS_AVAILABLE_DATE` sentinel (1900-01-01) | ✅ Correct by design — timeless reference material (strategy docs, config), not time-sensitive external info |
| `earnings` | `earnings_ingestor.py` | Fiscal-quarter label + 45-day reporting lag heuristic (`normalize_date(f"{year}-Q{quarter}")`) | ⚠️ **Conservative, not risky** — same category of imprecision the fundamentals-PIT work fixed for price/fundamentals data (real filing dates now preferred there), but never carried over to this RAG ingestor. The heuristic *overestimates* the lag (safe direction — it can only make the LLM unable to see something that's actually already public, never leak something not yet public). Worth tightening to a real transcript/report date if a provider exposes one, matching the fundamentals-PIT fix, but not a look-ahead bug. |
| `run_notes` | `run_ingestor.py` | Backtest run's `period_end`/`period_start`, **no reporting lag at all** | ⚠️ **Latent risk, currently inert** — a run isn't actually ingested/available until well after its covered period ends (the backtest has to finish executing), so `period_end` alone understates true availability lag. **However**: grepping every LLM-enhanced agent's `collections=[...]` argument (`bull_researcher_llm.py`, `bear_researcher_llm.py`, `debate_llm.py`, `trader_llm.py`, `risk_llm.py`, `technical_analyst_llm.py`, `fundamental_analyst_llm.py`, `sentiment_analyst_llm.py`) shows **none of them query `run_notes`** — only `news`, `sec_filings`, `earnings`, `research`, `system_docs`. So this has no effect on any current trading decision. Flagged so it isn't missed if `run_notes` is ever wired into a live-decision path with an `asof` filter (e.g. a future "what have similar past runs shown" agent) — add a lag before that happens.

## Bugs found and fixed

1. **Dense-channel crash on an explicit `None` date** (`src/firm/rag/store.py`,
   `VectorStore.query`). The original filter was
   `metadata.get("date", UNKNOWN_DATE) > asof_str` — `.get(key, default)`
   only substitutes the sentinel when the key is *missing*. A document whose
   metadata explicitly carries `"date": None` (a malformed/legacy record
   that bypassed `normalize_date`) still slipped through as `None`, and
   `None > "2023-06-01"` raises `TypeError`, crashing the whole query
   instead of just excluding that one doc. Fixed with a new
   `_doc_available_by(metadata, asof_str)` helper that treats a missing,
   `None`, or empty date the same as `UNKNOWN_DATE` (always excluded), and
   wraps the comparison in a try/except so any other malformed value also
   fails closed rather than raising.
2. **Duplicated fail-closed logic** — the BM25 channel
   (`RAGRetriever._passes_filters`) had its own, slightly different inline
   date check. Both channels now share the one `_doc_available_by` helper,
   so there's a single place that defines "is this doc available by this
   date" instead of two implementations that could silently drift apart.

Both channels are covered by regression tests: `tests/test_rag.py::TestStoreDateFilter::test_asof_excludes_explicit_none_date_without_crashing`
(the crash fix) and `tests/test_rag_hybrid_contextual.py::TestHybridRetrieval::test_passes_filters_excludes_missing_or_malformed_dates`
(missing/`None`/empty dates never survive the BM25 channel's filter either).

## What this means for the LLM A/B (`docs/llm_ab_experiment_log.md`)

The retrieval-layer as-of guard is sound for every collection Arm B's
`fundamental_analyst`/`sentiment_analyst` actually query
(`sec_filings`, `earnings`, `news`) — dates are genuine availability dates
(or a safely-conservative heuristic for `earnings`), and the guard itself no
longer has a crash edge case. **No look-ahead leak was found** in the path
that matters for the current A/B. The `run_notes` gap is real but inert —
worth a one-line fix (add a reporting lag) before any future agent starts
querying that collection with an `asof` filter, not before trusting Arm B.

## Follow-ups (not blocking, not done here)

- Tighten `earnings_ingestor.py` to a real transcript/report date if/when a
  provider exposes one, mirroring the fundamentals-PIT fix already applied
  to price/fundamentals data (`resolve_filing_date` in
  `firm.data.providers.base`).
- Add a reporting lag to `run_ingestor.py`'s `period_end`-as-availability-date
  before any live-decision-path agent starts consuming `run_notes` with an
  `asof` filter.
