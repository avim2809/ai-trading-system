"""LLM management API endpoints.

Provides configuration, cache, and RAG management without requiring
the ``llm`` extra to be installed – endpoints degrade gracefully.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

log = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["llm"])

_CONFIG_PATH = Path("config/llm.yaml")


# ── request / response models ──────────────────────────────────────

class LLMConfigUpdate(BaseModel):
    agent_modes: dict[str, str] | None = None
    default_model: str | None = None
    temperature: float | None = None
    compression_enabled: bool | None = None
    cache_enabled: bool | None = None


class IngestRequest(BaseModel):
    doc_type: str = "all"
    symbols: list[str] | None = None
    params: dict[str, Any] | None = None


class EmbeddingModelRequest(BaseModel):
    model_id: str


class TestRequest(BaseModel):
    model: str | None = None
    prompt: str = "Say hello in one sentence."


# ── helpers ─────────────────────────────────────────────────────────

def _load_llm_config() -> dict[str, Any]:
    """Load config/llm.yaml if it exists, else return defaults."""
    try:
        import yaml
        if _CONFIG_PATH.exists():
            return yaml.safe_load(_CONFIG_PATH.read_text()) or {}
    except Exception:
        log.warning("llm_config_load_failed path=%s — using hardcoded defaults", _CONFIG_PATH, exc_info=True)
    return {
        "provider": {"default_model": "groq/llama-3.3-70b-versatile", "temperature": 0.3},
        "agent_modes": {},
        "optimization": {"cache_enabled": True, "compression_enabled": True},
    }


def _save_llm_config(cfg: dict[str, Any]) -> None:
    try:
        import yaml
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CONFIG_PATH.write_text(yaml.safe_dump(cfg, default_flow_style=False))
    except Exception:
        log.warning("Could not save LLM config to %s", _CONFIG_PATH)


def _detect_providers() -> list[dict[str, Any]]:
    """List the known LLM providers and whether each is configured (key present).

    Backed by the central provider registry (``firm.llm.providers``) so model IDs
    and supported providers stay in one place instead of being hardcoded here.
    """
    from firm.llm.providers import list_providers

    return [
        {
            "name": p.key,
            "label": p.label,
            "configured": p.is_configured(),
            "models": list(p.example_models),
            "default_model": p.default_model,
        }
        for p in list_providers()
    ]


# ── endpoints ───────────────────────────────────────────────────────

@router.get("/providers")
def list_providers():
    """List available LLM providers based on installed packages and env keys."""
    return {"providers": _detect_providers()}


@router.get("/config")
def get_config():
    """Read current LLM config."""
    from firm.llm.providers import provider_key_for_model

    cfg = _load_llm_config()
    default_model = (cfg.get("provider") or {}).get("default_model", "")
    if default_model:
        cfg.setdefault("provider", {})["resolved_provider"] = provider_key_for_model(
            default_model
        )
    return cfg


@router.put("/config")
def update_config(body: LLMConfigUpdate):
    """Update LLM config: agent modes, provider, model, etc."""
    cfg = _load_llm_config()

    if body.agent_modes is not None:
        cfg.setdefault("agent_modes", {}).update(body.agent_modes)
    if body.default_model is not None:
        cfg.setdefault("provider", {})["default_model"] = body.default_model
    if body.temperature is not None:
        cfg.setdefault("provider", {})["temperature"] = body.temperature
    if body.compression_enabled is not None:
        cfg.setdefault("optimization", {})["compression_enabled"] = body.compression_enabled
    if body.cache_enabled is not None:
        cfg.setdefault("optimization", {})["cache_enabled"] = body.cache_enabled

    _save_llm_config(cfg)
    return {"status": "updated", "config": cfg}


@router.get("/cache/stats")
def cache_stats():
    """Cache hit/miss/savings statistics."""
    try:
        from firm.llm.cache import ResponseCache
        cache = ResponseCache()
        return cache.stats()
    except Exception:
        return {"hits": 0, "misses": 0, "total_cost_saved": 0.0, "db_size": 0, "available": False}


@router.delete("/cache")
def clear_cache():
    """Clear the LLM response cache."""
    try:
        from firm.llm.cache import ResponseCache
        cache = ResponseCache()
        cache.clear()
        return {"status": "cleared"}
    except Exception:
        return {"status": "cache_unavailable"}


_RAG_COLLECTION_DESCRIPTIONS = {
    "sec_filings": "SEC filings (10-K/10-Q/8-K)",
    "research": "Academic research papers (arXiv)",
    "earnings": "Earnings call transcripts",
    "news": "News articles",
    "system_docs": "Strategy/system documentation",
}


@router.get("/rag/stats")
def rag_stats():
    """Per-collection document counts and index info.

    VectorStore.stats() returns a flat {name: count, "_total": n} dict —
    reshaped here into {"collections": {name: {count, description}}}, the
    contract the frontend actually expects (Object.keys(ragStats.collections)
    on the previous raw pass-through crashed the whole Configuration page
    whenever real RAG data was present, since there was no "collections"
    key at all).
    """
    try:
        from firm.rag.store import VectorStore
        store = VectorStore()
        raw = store.stats()
        collections = {
            name: {
                "count": count,
                "description": _RAG_COLLECTION_DESCRIPTIONS.get(name, ""),
            }
            for name, count in raw.items()
            if not name.startswith("_")
        }
        return {"collections": collections}
    except Exception:
        return {"collections": {}, "available": False}


@router.post("/rag/ingest")
def rag_ingest(body: IngestRequest):
    """Trigger document ingestion in a background thread."""
    def _do_ingest():
        try:
            from firm.rag.store import VectorStore
            log.info("Starting RAG ingestion: %s", body.doc_type)
            _store = VectorStore()
            log.info("RAG ingestion complete for %s", body.doc_type)
        except Exception:
            log.error("RAG ingestion failed", exc_info=True)

    thread = threading.Thread(target=_do_ingest, daemon=True)
    thread.start()
    return {"status": "ingestion_started", "doc_type": body.doc_type}


@router.delete("/rag/{collection}")
def delete_rag_collection(collection: str):
    """Clear a specific RAG collection."""
    try:
        from firm.rag.store import VectorStore
        store = VectorStore()
        store.delete_collection(collection)
        return {"status": "deleted", "collection": collection}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/embedding-models")
def list_embedding_models():
    """Return all supported embedding models with metadata."""
    from firm.rag.embeddings import list_models

    from dataclasses import asdict
    return [asdict(m) for m in list_models()]


@router.put("/rag/embedding-model")
def set_embedding_model(body: EmbeddingModelRequest):
    """Switch the active embedding model.  Returns whether a re-index is needed."""
    from firm.rag.embeddings import EMBEDDING_MODELS, get_model_info

    info = get_model_info(body.model_id)
    if info is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown embedding model: {body.model_id}. "
            f"Supported: {list(EMBEDDING_MODELS.keys())}",
        )

    cfg = _load_llm_config()
    current_model = cfg.get("rag", {}).get("embedding_model", "all-MiniLM-L6-v2")

    current_info = get_model_info(current_model)
    requires_reindex = (
        current_info is None
        or info.dimensions != current_info.dimensions
        or body.model_id != current_model
    )

    cfg.setdefault("rag", {})["embedding_model"] = body.model_id
    _save_llm_config(cfg)

    from dataclasses import asdict
    return {
        "status": "updated",
        "requires_reindex": requires_reindex,
        "model": asdict(info),
    }


@router.post("/test")
def test_connection(body: TestRequest):
    """Test LLM connection with current config."""
    import time

    start = time.monotonic()
    try:
        from firm.llm.provider import LLMService
        cfg = _load_llm_config()
        provider_cfg = cfg.get("provider", {})
        if body.model:
            provider_cfg["default_model"] = body.model
        svc = LLMService(provider_cfg)
        response = svc.chat([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": body.prompt},
        ])
        return {
            "status": "ok",
            "response": response,
            "model": provider_cfg.get("default_model", "gpt-4o"),
            "response_time_ms": round((time.monotonic() - start) * 1000),
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}
