# RAG + AI for the Trading System — Deep-Research Report

**Date:** 2026-06-30 · **Method:** fan-out web search → fetch → adversarial
verification → synthesis.

This is the durable, in-repo record of the deep-research pass that grounds
[rag-ai-implementation-plan.md](rag-ai-implementation-plan.md). Pipeline:
**6 search angles → 26 sources fetched → 123 claims extracted → 25
adversarially verified (2-of-3 refute kills a claim) → 23 confirmed, 2 killed →
9 synthesized findings.**

## Question

How should a mature Python multi-agent algorithmic trading system (analysts →
bull/bear → debate → trader → risk → execution; Backtrader backtester;
Gaussian-HMM regime strategy; React frontend; producing backtest runs, trade
logs, strategy docs) integrate RAG and AI in 2026, using a **local-first**
architecture where cloud LLM APIs (incl. Anthropic Claude) are permitted where
they materially help? Three areas: (1) RAG architecture over trading artifacts;
(2) model & infra selection; (3) self-improving learning loop (skeptically).

## Executive summary

Build RAG with **metadata-enriched ("contextual") embeddings + a reranker** as
the retrieval core, but treat **numeric tables as first-class structured data**
(SQL via relational storage, as in TableRAG) rather than flat text chunks.
**Cloud LLMs are justified for the generation/synthesis step** because even
frontier models hallucinate 10–20% on multi-step numerical reasoning over
financial tables (smaller open models collapse to ~0% on multivariate calc) —
so financial math should be executed deterministically (SQL/code) and the LLM
used for grounded language synthesis with verifiable citations. Anthropic Claude
is a concrete, well-documented generation tier, with prompt caching (0.1×
cached input) making repeated grounded QA over the same context economical and a
Citations API for verifiable source spans. Measure quality with the **RAG Triad**
(TruLens) or LLM-as-judge. Dominant caveats: grounding is necessary but not
sufficient; the strongest retrieval benchmarks were on code/financial-text, not
mostly-numeric trade logs; and the self-improving-loop area surfaced little
verified evidence — proceed skeptically given overfitting, look-ahead bias,
non-stationarity, and reflexivity risks.

## Verified findings

### 1. Retrieval recipe — contextual embeddings + hybrid + reranker (confidence: high)
Optimal financial-QA architecture combines LLM-driven pre-retrieval
optimizations with contextual embeddings, with a reranker "essential for
precision." Anthropic's Contextual Retrieval cookbook measures
contextual-embeddings + hybrid BM25 + reranking at **92.15 / 95.26 / 97.45%
Pass@5/10/20 vs 80.92 / 87.15 / 90.06% baseline** (248 queries, 9 codebases;
Voyage-2 + Cohere rerank-v3.0); contextual embeddings alone cut top-20 retrieval
failure rate by 35%.
Sources: [arXiv 2510.24402](https://arxiv.org/abs/2510.24402) ·
[Anthropic cookbook](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide) ·
[Anthropic engineering](https://anthropic.com/engineering/contextual-retrieval)
(corroborated by FinSage 2504.14493, FinGEAR 2509.12042, Multi-Reranker 2411.16732).

### 2. Preserve tables as structured data; query with SQL (confidence: high)
Naive chunking of tables causes "Structural Information Loss" and "Lack of
Global View," degrading multi-hop queries. TableRAG stores tables relationally
(SQL execution) + text retrieval to preserve integrity and enable heterogeneous
reasoning; GTR/Graph-Table-RAG independently confirms (linearization disrupts
structure). SQL is one valid remedy among several.
Sources: [TableRAG arXiv 2506.10380](https://arxiv.org/html/2506.10380v1) ·
[GTR arXiv 2504.01346](https://arxiv.org/abs/2504.01346)

### 3. LLMs hallucinate on multi-step financial math (confidence: high)
FAITH benchmark (S&P 500 annual-report tables, ICAIF 2025): Claude-Sonnet-4
(95.6% overall) and Gemini-2.5-Pro (91.9%) lead but show **10–20% error on
multivariate calculation**; several larger open models (Llama-3.3-70B 0.0%,
Mistral-small-24B) score ~0% on multivariate calc — "fundamental breakdown…
leading to fabrication." → Route numeric computation to SQL/code; reserve the
LLM for grounded synthesis.
Source: [FAITH arXiv 2508.05201](https://arxiv.org/pdf/2508.05201)

### 4. Retrieval grounding is necessary but NOT sufficient (confidence: high)
LLMs "still frequently introduce unsupported information or contradictions even
when provided with relevant context" (FaithBench/HHEM across 160+ LLMs; best
automatic detection <78% balanced accuracy). The mitigation literature
presupposes retrieval is necessary but insufficient → must evaluate + cite.
Source: [arXiv 2505.04847](https://arxiv.org/html/2505.04847v2)

### 5. Evaluate with the RAG Triad (confidence: high)
TruLens RAG Triad = context relevance ("any irrelevant info could be woven into
a hallucination"), groundedness (each claim supported by context), answer
relevance. Corroborated by DeepEval, RAGAS, Snowflake, Comet.
Source: [TruLens RAG Triad](https://www.trulens.org/getting_started/core_concepts/rag_triad/)

### 6. LLM-as-judge frameworks are viable (confidence: high)
FaithJudge (conditions on human-annotated peer responses) ~84% balanced
accuracy / 82.1% F1-macro on FaithBench (~15pt over prior ~68.8%). CCRS
(zero-shot single-LLM, 5 metrics) reaches comparable/superior discrimination to
RAGChecker at ~5× speedup. MDPI 2025 e-governance study demonstrates a modular
statement-level faithfulness framework using GPT-4.1, Claude Sonnet-4, Gemini
2.5 Pro as judges.
Sources: [arXiv 2505.04847](https://arxiv.org/html/2505.04847v2) ·
[CCRS arXiv 2506.20128](https://arxiv.org/pdf/2506.20128) ·
[MDPI 9/12/309](https://www.mdpi.com/2504-2289/9/12/309)

### 7. Anthropic Claude cost-tiered generation option (confidence: high)
As of early-to-mid 2026 (official pricing; TIME-SENSITIVE — re-verify):
Haiku 4.5 (`claude-haiku-4-5`) **$1/$5** per MTok in/out, 200K context;
Sonnet 4.6 (`claude-sonnet-4-6`) **$3/$15**, 1M context;
Opus 4.8 (`claude-opus-4-8`) **$5/$25**, 1M context.
Source: [platform.claude.com/docs pricing](https://platform.claude.com/docs/en/about-claude/pricing)

### 8. Prompt caching makes repeated grounded QA economical (confidence: high)
Cache reads = **0.1× base input (~90% cheaper)** — the core cost lever for
repetitive tasks with consistent elements (system prompts, large documents).
Cache writes 1.25× (5-min TTL) / 2× (1-hour TTL); break-even after one read
(5m) or two (1h). For Opus 4.8, cached input ≈ $0.50/MTok. Directly applicable
to repeatedly querying the same cached trade-log/strategy context.
Sources: [pricing](https://platform.claude.com/docs/en/about-claude/pricing) ·
[prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)

### 9. Claude Citations API grounds responses in source spans (confidence: high)
Returns exact `cited_text` with guaranteed-valid pointers; in Anthropic's own
evals "significantly more likely to cite the most relevant quotes" than
prompt-based citation. Caveats: vendor self-eval; the guarantee is mechanical
(span extracted from source), not a guarantee of correctness/relevance.
Source: [citations docs](https://platform.claude.com/docs/en/build-with-claude/citations)

## Killed claims (did not survive verification)

- **"Embedding chunk metadata is the single LARGEST gain vs reranking/filtering"
  — refuted 0-3.** Contextual embeddings help, but reranking matters more. (This
  is why the chunker `contextual` flag defaults **off** and needs piloting.)
  Source: arXiv 2510.24402.
- **A specific cache-write cost framing — refuted 1-2.** Source: prompt-caching docs.

## Caveats & limitations

1. **Transfer risk:** the strongest retrieval numbers were measured on CODE
   (9 codebases) and financial TEXT/tables (FinanceBench, S&P 500 filings),
   **not** mostly-numeric trade-by-trade logs or time-series summaries — do not
   assume transfer; pilot on local data.
2. **Source strength:** several key results are vendor self-published or
   single-preprint without independent replication (Anthropic contextual-retrieval
   & citations evals; arXiv 2510.24402; CCRS's RAGChecker comparison).
3. **Time-sensitive:** Claude pricing/model-IDs verified early-to-mid 2026 — drift
   expected; re-verify against platform.claude.com.
4. **Naming:** third-party papers use "Claude Sonnet-4.0" etc. — formatting, not
   factual, discrepancies.
5. The 10–20% hallucination figure is the **hardest multivariate-calculation
   subset**, not all multi-step tasks.
6. **Section 2 (local models/infra):** no verified specifics survived on free/local
   embedding models (nomic, MiniLM, BGE, GTE, Stella, mxbai) with MTEB
   standings/footprints, nor local vector-store tradeoffs (Chroma/FAISS/LanceDB/
   Qdrant), nor local generation LLMs (Llama/Qwen/Mistral/Phi) — unverified here.
7. **Section 3 (self-improving loop):** weakest-covered; only LLM-as-judge eval
   claims survived. No verified claims on RL feedback, retrieval-augmented trade
   memory, or agentic backtesting/reflection; documented risks (overfitting,
   look-ahead/data-snooping, non-stationarity, reflexivity) were not
   independently substantiated → treat as **unestablished**; left unbuilt.

## Open questions

- Which free/local embedding model + vector store best fit thousands-to-millions
  of mostly-numeric trade-log vectors on one workstation (by MTEB, latency, RAM)?
- Crossover point where a local LLM (Llama/Qwen/Mistral/Phi via Ollama) suffices
  vs Claude materially winning for grounded financial QA — given numeric work is
  offloaded to SQL/code?
- Do contextual-embeddings + reranking gains (measured on code/financial-text)
  transfer to numeric, semi-structured trade logs; best chunk size/overlap/
  metadata tagging for them?
- For the self-improving loop, what is the real evidence base + failure modes for
  retrieval-augmented trade memory, RL feedback, agentic backtesting/reflection?

## Source list

Primary: arXiv 2510.24402, 2508.05201, 2506.10380, 2504.01346, 2505.04847,
2506.20128; TruLens docs; MDPI 9/12/309; Anthropic platform docs (pricing,
prompt-caching, contextual-embeddings cookbook, citations); self-improving-agent
preprints (2508.17565, 2311.13743, 2509.11420, 2408.06361) and skeptical surveys
(2605.19337, 2505.07078, 2606.00061). Secondary/blog: CodeSOTA MTEB, marktechpost
vector-DB comparison, TowardsDataScience hybrid-search, Daloopa, FutureAGI,
Infrabase.

Stats: angles 6 · sources fetched 26 · claims extracted 123 · verified 25 ·
confirmed 23 · killed 2 · synthesized 9 · agent calls 109.
