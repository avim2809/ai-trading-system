"""Shared loader for ``config/llm.yaml``.

A single place to read the LLM/RAG/assistant configuration so the RAG
modules (structured store, assistant) and agents don't each re-implement
YAML loading + defaults. All accessors are total: a missing file or
missing section yields documented defaults rather than raising.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

_CONFIG_PATH = Path("config/llm.yaml")

_RAG_DEFAULTS: dict[str, Any] = {
    "persist_dir": "data/vectordb",
    # External (hosted) RAG models by default — no local torch needed.
    "embedding_provider": "voyage",        # "voyage" | "local"
    "embedding_model": "voyage-finance-2",  # local example: all-MiniLM-L6-v2
    "reranker_provider": "voyage",         # "voyage" | "local" | "none"
    "reranker_model": "rerank-2.5",        # local: cross-encoder/ms-marco-MiniLM-L-6-v2
    "chunk_size": 500,
    "chunk_overlap": 50,
    "reranking": True,
    "default_n_results": 5,
    "contextual": False,   # Phase 3a — prepend metadata into embedded text
    "hybrid": False,       # Phase 3b — fuse BM25 with dense retrieval
    "runs_dir": "runs",    # Phase 1b — DuckDB structured-query source
}

_PROVIDER_DEFAULTS: dict[str, Any] = {
    # groq/llama-3.3-70b-versatile deprecated by Groq 2026-06-17, confirmed
    # dead live 2026-08-29 — see src/firm/llm/providers.py's groq entry.
    "default_model": "groq/openai/gpt-oss-120b",
    "fallback_models": [],
    "load_balance": False,
    "temperature": 0.3,
    "max_tokens": 2000,
    "request_timeout": 90,
}

_ASSISTANT_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "prompt_caching": True,   # use Anthropic cache_control when provider is anthropic
    "citations": False,       # use the Claude Citations API
    "n_results": 5,
}

_ENHANCEMENT_DEFAULTS: dict[str, Any] = {
    # live_calls — call the provider (subject to caps below).
    # cache_only — use LLM only when the response is already in llm_cache.db.
    "policy": "live_calls",
    # Per-signal agents: skip weak quant scores; keep top-N by |score| per cycle.
    "min_abs_score": 0.2,
    "max_signals_per_agent": 8,
    # Thesis / debate agents: skip low-conviction names; cap debate breadth.
    "min_conviction": 0.25,
    "max_theses_per_agent": 5,
    "max_debate_symbols": 5,
    # RAG chunks per retrieval (Voyage query-embed cost is 1/call regardless).
    "rag_n_results": 2,
    # Self-consistency sampling (see firm.live.planning_cycle and
    # LLMAgentMixin._call_llm): >1 calls the LLM N times against the
    # identical input and aggregates before returning, instead of the
    # usual single call. 1 (default) is a no-op -- every _call_llm site
    # behaves exactly as before this existed. Only ever set above 1 by
    # Orchestrator._apply_cycle_llm_mode for a "planning" cycle specifically
    # (config/live*.yaml's planning_cycle.self_consistency_samples); never
    # configured directly here.
    "self_consistency_samples": 1,
    # Per-call sampling temperature override for enhancement/scoring calls
    # (None = use LLMService's global provider.temperature default,
    # unchanged prior behavior). Enhancement calls feed straight into the
    # z-scored analyst signal / bull-bear conviction / debate net_conviction
    # the rest of the pipeline treats as a quant-grade number — a genuinely
    # different (but similarly-worded) day's prompt sampled at a high
    # temperature can swing an otherwise-stable signal on pure LLM sampling
    # noise, on top of whatever real information changed. See
    # LLMAgentMixin._call_llm.
    "temperature": None,
    # Portfolio-level agents (off by default — 1 call each when enabled).
    "enhance_portfolio_review": False,
    "enhance_risk_review": False,
}

_OPTIMIZATION_DEFAULTS: dict[str, Any] = {
    "compression_enabled": True,
    "compression_ratio": 0.5,
    # Off by default — real LLMLingua-2 loads a ~700MB BERT model (CPU
    # inference, 1-2s/call) that competes with live trading for CPU/RAM on
    # small hosts. Compression falls back to cheap sentence-sampling unless
    # this is explicitly set true (e.g. for an offline/backtest run).
    "use_llmlingua": False,
    "cache_enabled": True,
    "cache_db": "data/llm_cache.db",
    "cache_ttl_hours": 168,
}


def _resolve_config_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.getenv("FIRM_LLM_CONFIG", "").strip()
    if override:
        return Path(override)
    return _CONFIG_PATH


def load_llm_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load LLM YAML (``config/llm.yaml`` or ``FIRM_LLM_CONFIG``)."""
    cfg_path = _resolve_config_path(path)
    try:
        if cfg_path.exists():
            return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if cfg_path != _CONFIG_PATH:
            log.warning("FIRM_LLM_CONFIG path does not exist: %s", cfg_path)
    except Exception:
        log.warning("Failed to load LLM config from %s", cfg_path, exc_info=True)
    return {}


def _section(cfg: dict[str, Any] | None, key: str, defaults: dict[str, Any]) -> dict[str, Any]:
    cfg = cfg if cfg is not None else load_llm_config()
    section = cfg.get(key) or {}
    return {**defaults, **section}


def rag_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the ``rag`` section merged over defaults."""
    return _section(cfg, "rag", _RAG_DEFAULTS)


def provider_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the ``provider`` section merged over defaults."""
    return _section(cfg, "provider", _PROVIDER_DEFAULTS)


def assistant_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the ``assistant`` section merged over defaults."""
    return _section(cfg, "assistant", _ASSISTANT_DEFAULTS)


def optimization_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the ``optimization`` section merged over defaults."""
    return _section(cfg, "optimization", _OPTIMIZATION_DEFAULTS)


def llm_service_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the config dict :class:`firm.llm.provider.LLMService` expects.

    ``LLMService.__init__`` reads model/provider knobs (``default_model``,
    ``fallback_models``, ``load_balance``, ``temperature``, ``max_tokens``,
    ``request_timeout`` — the ``provider`` section) *and* two cache knobs
    (``cache_enabled``, ``cache_db`` — from the ``optimization`` section)
    out of the single flat dict it's constructed with. ``provider_config()``
    alone never includes the ``optimization`` section, so a construction
    site that passes just that dict silently loses any non-default
    ``cache_enabled``/``cache_db`` instead of merely defaulting it.
    """
    cfg = cfg if cfg is not None else load_llm_config()
    merged = provider_config(cfg)
    opt = optimization_config(cfg)
    for key in ("cache_enabled", "cache_db"):
        if key in opt:
            merged[key] = opt[key]
    return merged


def enhancement_config(
    cfg: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the ``enhancement`` section merged over defaults and *overrides*."""
    merged = _section(cfg, "enhancement", _ENHANCEMENT_DEFAULTS)
    if overrides:
        merged = {**merged, **overrides}
    return merged


def is_anthropic(model: str) -> bool:
    """True when *model* routes to Anthropic Claude (for prompt-caching/citations)."""
    m = (model or "").lower()
    return "claude" in m or m.startswith("anthropic/")
