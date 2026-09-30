# RAG + AI Implementation Plan

**Status:** Implemented (branch `feat/rag-trading-artifacts`, 460 tests green) ·
**Date:** 2026-06-30 · **Scope:** `src/firm` (the `firm` package)

> All phases below are implemented. Files: `backtest/analyzers.py`
> (`TradeLogAnalyzer`), `eval/reports.py` (`save_trades`/`TRADE_COLUMNS`),
> `rag/structured.py` (`RunStore`), `rag/ingestors/run_ingestor.py`,
> `rag/assistant.py` (`TradingAssistant`), `eval/rag_eval.py`
> (`RagTriadEvaluator`), `rag/chunker.py` (contextual flag), `rag/retriever.py`
> + `rag/store.py` (hybrid BM25/RRF), `llm/config.py` (shared loader). Tests:
> `tests/test_trade_log.py`, `test_structured.py`, `test_run_ingestor.py`,
> `test_rag_assistant.py`, `test_rag_eval.py`, `test_rag_hybrid_contextual.py`.
> Self-improving loop remains out of scope by design.

This plan extends the trading system's existing RAG/LLM subsystem to cover the
system's *own* trading artifacts (backtest runs, trade logs, strategy notes),
grounded in a deep-research review of 2025–2026 financial-RAG literature. It is a
**gap analysis with file-level actions**, not a greenfield design — most of the
retrieval core already exists and is research-aligned.

---

## 1. Current state (what already exists)

The `firm` package already implements most of the research's *retrieval*
recommendations. We build on these; we do **not** rewrite them.

| Capability | Location | Notes |
|---|---|---|
| Vector store (Chroma, persistent) | `rag/store.py` | Per-collection; metadata filtering |
| **Point-in-time safety** | `rag/store.py` `query(asof=...)` | `date <= asof` filter — prevents look-ahead leakage in retrieval |
| Cross-encoder reranking | `rag/retriever.py` | `ms-marco-MiniLM-L-6-v2`; research calls reranking "essential for precision" |
| Embedding model registry | `rag/embeddings.py` | MiniLM (default), MPNet, BGE, nomic, GTE-Qwen2, E5 |
| Document chunking | `rag/chunker.py` | Size/overlap + section-aware; metadata attached for filtering |
| External-text ingestors | `rag/ingestors/` | news, earnings, SEC, research, system_docs |
| LLM service (multi-provider) | `llm/provider.py` | LiteLLM; free default (`groq/llama-3.3-70b-versatile`), Claude-switchable |
| Response caching | `llm/cache.py` | SQLite, TTL (`data/llm_cache.db`) |
| Context compression | `llm/compression.py` | |
| Agent routing (quant vs LLM) | `config/llm.yaml` `agent_modes` | trader/risk/technical/fundamental → `quant` (deterministic) |

**Key alignment already achieved:** the `agent_modes` config already keeps the
numeric agents (trader, risk, technical, fundamental) deterministic and reserves
the LLM for narrative agents (sentiment, bull, bear, debate). This matches the
research finding that LLMs must **not** perform financial arithmetic.

---

## 2. Research basis (verified findings)

From a 6-angle deep-research pass (26 sources → 123 claims → 25 adversarially
verified, 23 confirmed). Confidence levels and citations carried through.

1. **Retrieval recipe (high):** contextual (metadata-enriched) embeddings +
   hybrid BM25/dense + reranker gives best precision (92–97% Pass@k vs 81–90%
   baseline). Reranker essential. — [arXiv 2510.24402](https://arxiv.org/abs/2510.24402),
   [Anthropic contextual retrieval](https://anthropic.com/engineering/contextual-retrieval)
   *Caveat:* benchmarks were on code/financial-text, not mostly-numeric trade
   logs — pilot on local data. The "metadata embedding is the single largest
   gain" sub-claim was **refuted (0-3)**; reranking (already present) matters more.
2. **Tables as structured data (high):** flat-chunking numeric tables causes
   "structural information loss" + "lack of global view," breaking multi-hop /
   aggregation queries. Preserve structure; query via SQL. —
   [TableRAG 2506.10380](https://arxiv.org/html/2506.10380v1),
   [GTR 2504.01346](https://arxiv.org/abs/2504.01346)
3. **LLMs hallucinate on financial math (high):** top models show 10–20% error
   on multivariate calculation; some open models ~0%. Compute numbers
   deterministically. — [FAITH 2508.05201](https://arxiv.org/pdf/2508.05201)
4. **Grounding necessary but insufficient (high):** LLMs add unsupported content
   even with relevant context; best auto-detection <78%. Must evaluate. —
   [arXiv 2505.04847](https://arxiv.org/html/2505.04847v2)
5. **Evaluation (high):** RAG Triad — context relevance, groundedness, answer
   relevance (TruLens). LLM-as-judge (FaithJudge ~84%, CCRS) viable. —
   [TruLens](https://www.trulens.org/getting_started/core_concepts/rag_triad/),
   [CCRS 2506.20128](https://arxiv.org/pdf/2506.20128)
6. **Claude cloud tier (high, official docs):** Haiku 4.5 `$1/$5`, Sonnet 4.6
   `$3/$15`, Opus 4.8 `$5/$25` per MTok. **Prompt caching → cached input 0.1×
   (90% cheaper)**; Citations API returns verifiable `cited_text` spans. —
   [pricing](https://platform.claude.com/docs/en/about-claude/pricing),
   [caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching),
   [citations](https://platform.claude.com/docs/en/build-with-claude/citations)
   *Caveat:* model IDs/prices are time-sensitive (verified mid-2026) — re-check
   before relying on exact figures.
7. **Self-improving loop (weak evidence):** almost no verified support; documented
   risks (overfitting, look-ahead/data-snooping, non-stationarity, reflexivity)
   not substantiated here. **Out of scope** for this plan.

---

## 3. Gap analysis

| # | Gap | Research basis | Priority |
|---|---|---|---|
| G1 | RAG ingests external text + system docs, but **not** `runs/` or trade logs. Per-trade data in `backtest/analyzers.py` / `portfolio/attribution.py` is **not persisted** (run dirs hold only `config.json`, `equity.json`, `report.json` aggregates). | #2 (the canonical use case) | **P1** |
| G2 | Numeric metrics/trades would be flattened into Chroma text chunks instead of SQL-queried. | #2, #3 | **P1** |
| G3 | No RAG-quality evaluation. `eval/` measures backtest performance (Sharpe), not faithfulness. | #4, #5 | **P2** |
| G4 | Generation returns plain text — no citations/grounding; Anthropic prompt-caching unused. | #6 | **P2** |
| G5 | Chunker attaches metadata for *filtering* but doesn't prepend it into *embedded text* (contextual embeddings). | #1 | **P3** |
| G6 | Retriever is pure dense + rerank; no BM25/lexical hybrid. | #1 | **P3** |

**Out of scope:** self-improving learning loop (finding #7). The existing `asof`
discipline (`rag/store.py`) is the correct foundation, but no learning loop is
built on it until honest, look-ahead-safe evaluation exists.

---

## 4. Implementation phases

Each phase is independently shippable and testable against the existing suite
(407 tests). Build order optimizes for unblocking downstream work.

### Phase 1 — Persist trade logs + structured query layer (P1, G1+G2)

**1a. Persist per-trade logs.** In `backtest/engine.py` (run-artifact writer),
emit `trades.parquet` per run dir alongside the existing JSON. Columns (minimum):
`entry_dt, exit_dt, symbol, strategy, side, qty, entry_px, exit_px, pnl,
return_pct, bars_held, mae, mfe`. Source data already exists in
`backtest/analyzers.py` and `portfolio/attribution.py`.
- *New/changed:* `backtest/engine.py` (write step), reuse `pyarrow` (already a dep).
- *Test:* a backtest run produces a non-empty, schema-valid `trades.parquet`.

**1b. Structured-query layer (DuckDB).** New `src/firm/rag/structured.py`:
```python
class RunStore:
    """Read-only SQL view over runs/*/{report.json, trades.parquet, equity.json}."""
    def __init__(self, runs_dir: str = "runs") -> None: ...
    def query(self, sql: str) -> pd.DataFrame: ...        # parameterized, read-only
    def schema(self) -> str: ...                          # table/column description for the LLM
```
DuckDB reads parquet/JSON directly (zero-server). Numeric questions ("avg Sharpe
by strategy", "worst 5 trades by P&L", "drawdown contributors in run X") go to
SQL; the LLM only narrates results — never computes them.
- *New dep:* add `duckdb` to the `llm` extra in `pyproject.toml`.
- *Test:* canned questions → expected aggregates over a fixture run.

**1c. Run-artifact ingestor (narrative only).** New
`rag/ingestors/run_ingestor.py` (subclass `BaseIngestor`) indexing **textual**
run context — strategy descriptions, config rationale, human run notes — into a
`run_notes` collection with `date` metadata for `asof` safety. Numeric tables stay
in DuckDB, not Chroma.
- *Test:* ingest a fixture run dir → docs land in `run_notes`, retrievable by symbol/date.

### Phase 2 — Grounded assistant + evaluation (P2, G3+G4)

**2a. Assistant orchestrator.** New `src/firm/rag/assistant.py`:
```python
class TradingAssistant:
    """Answers questions over runs: routes numeric→SQL, narrative→retrieval,
    then synthesizes a grounded, cited answer."""
    def ask(self, question: str, asof=None) -> AssistantAnswer: ...
```
Flow: classify question → `RunStore.query()` for numbers + `RAGRetriever.retrieve()`
for context → synthesize with `LLMService`. Returns answer + source citations
(SQL rows used + retrieved chunk IDs).
- *Anthropic cost lever:* when the configured provider is `anthropic`, pass
  `cache_control` on the large static context (schema + retrieved chunks) so
  repeated questions over the same run hit cached input at 0.1× (LiteLLM supports
  Anthropic `cache_control`). Optionally use the Citations API for `cited_text`.
- *Test:* numeric question returns SQL-derived figure (not LLM arithmetic);
  narrative question cites retrieved chunks.

**2b. RAG evaluation harness.** New `src/firm/eval/rag_eval.py` implementing the
RAG Triad: `context_relevance`, `groundedness`, `answer_relevance` over a small
fixture Q/A set in `tests/fixtures/rag_eval/`. LLM-as-judge optional (gated behind
provider config so CI can run a deterministic stub).
- *Test:* harness runs on fixtures and emits the three scores; a known-bad
  (hallucinated) answer scores low on groundedness.

### Phase 3 — Retrieval-precision enhancements (P3, G5+G6)

**3a. Contextual embeddings.** In `rag/chunker.py`, optionally prepend a short
metadata header (`strategy / symbol / date / section`) into the embedded text
(behind a `contextual: true` config flag; default off until piloted, since the
"largest gain" claim was refuted and this needs validation on local data).
- *Test:* contextual chunks embed the header; flag off reproduces current behavior.

**3b. Hybrid BM25 + dense.** Add a lexical (BM25) channel in `rag/retriever.py`,
fuse with dense via reciprocal-rank fusion before reranking. Behind
`hybrid: true` config flag.
- *Test:* an exact-keyword query (e.g. a ticker or run-id) that dense retrieval
  misses is recovered by the lexical channel.

---

## 5. Config changes (`config/llm.yaml`)

Add under existing keys (all default to current behavior):
```yaml
rag:
  # existing: persist_dir, embedding_model, chunk_size, chunk_overlap, reranking, default_n_results
  contextual: false      # Phase 3a — prepend metadata into embedded text
  hybrid: false          # Phase 3b — BM25 + dense fusion
  runs_dir: "runs"       # Phase 1b — DuckDB source

assistant:               # Phase 2a
  enabled: false
  prompt_caching: true   # use Anthropic cache_control when provider is anthropic
  citations: false       # use Claude Citations API
```

---

## 6. Dependencies

- Add `duckdb` to the `[project.optional-dependencies].llm` group in `pyproject.toml`.
- Everything else (chromadb, sentence-transformers, litellm, pyarrow) already present.

---

## 7. Risks, caveats & non-goals

- **Benchmark transfer is unproven.** The strong retrieval numbers were measured
  on code/financial-text, not mostly-numeric trade logs. Treat Phase 3 as
  *piloted*, not assumed — gate with the Phase 2b eval harness.
- **Don't let the LLM do math.** Phase 1b/2a strictly route numeric computation to
  DuckDB. Any regression where the LLM derives figures is a correctness bug.
- **Time-sensitive Claude facts.** Re-verify model IDs/prices against
  platform.claude.com before relying on the table in §2.
- **Look-ahead discipline.** Every new collection/query path must thread `asof`
  (see `rag/store.py`). The run ingestor and assistant must respect it.
- **Non-goal: self-improving loop.** Explicitly deferred; insufficient verified
  evidence and high failure-mode risk (overfitting, data-snooping, reflexivity).

---

## 8. Suggested sequencing

1. **Phase 1a → 1b → 1c** (highest leverage; 1a unblocks everything).
2. **Phase 2a → 2b** (usable assistant, then quality gate).
3. **Phase 3a / 3b** (precision tuning, validated by 2b).

Full research report, verified findings, and source list:
[rag-ai-research-report.md](rag-ai-research-report.md). Existing subsystem map:
`src/firm/rag/`, `src/firm/llm/`, `config/llm.yaml`.
