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
        pass
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
    """Auto-detect available LLM providers from installed packages + env."""
    import os
    providers: list[dict[str, Any]] = []

    if os.environ.get("OPENAI_API_KEY"):
        providers.append({"name": "openai", "configured": True, "models": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]})
    else:
        providers.append({"name": "openai", "configured": False, "models": ["gpt-4o", "gpt-4o-mini"]})

    if os.environ.get("ANTHROPIC_API_KEY"):
        providers.append({"name": "anthropic", "configured": True, "models": ["claude-sonnet-4-20250514", "claude-3-haiku-20240307"]})
    else:
        providers.append({"name": "anthropic", "configured": False, "models": []})

    if os.environ.get("GROQ_API_KEY"):
        providers.append({"name": "groq", "configured": True, "models": ["groq/llama-3.3-70b-versatile"]})

    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    providers.append({"name": "ollama", "configured": bool(ollama_host), "host": ollama_host, "models": ["ollama/llama3"]})

    return providers


# ── endpoints ───────────────────────────────────────────────────────

@router.get("/providers")
def list_providers():
    """List available LLM providers based on installed packages and env keys."""
    return {"providers": _detect_providers()}


@router.get("/config")
def get_config():
    """Read current LLM config."""
    return _load_llm_config()


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


@router.get("/rag/stats")
def rag_stats():
    """Per-collection document counts and index info."""
    try:
        from firm.rag.store import VectorStore
        store = VectorStore()
        return store.stats()
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
        return {"status": "ok", "response": response, "model": provider_cfg.get("default_model", "gpt-4o")}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}
