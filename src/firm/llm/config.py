"""Shared loader for ``config/llm.yaml``.

A single place to read the LLM/RAG/assistant configuration so the RAG
modules (structured store, assistant) and agents don't each re-implement
YAML loading + defaults. All accessors are total: a missing file or
missing section yields documented defaults rather than raising.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    "default_model": "groq/llama-3.3-70b-versatile",
    "temperature": 0.3,
    "max_tokens": 2000,
}

_ASSISTANT_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "prompt_caching": True,   # use Anthropic cache_control when provider is anthropic
    "citations": False,       # use the Claude Citations API
    "n_results": 5,
}


def load_llm_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config/llm.yaml`` (or *path*), returning ``{}`` on any failure."""
    cfg_path = Path(path) if path else _CONFIG_PATH
    try:
        import yaml
        if cfg_path.exists():
            return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        pass
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


def is_anthropic(model: str) -> bool:
    """True when *model* routes to Anthropic Claude (for prompt-caching/citations)."""
    m = (model or "").lower()
    return "claude" in m or m.startswith("anthropic/")
