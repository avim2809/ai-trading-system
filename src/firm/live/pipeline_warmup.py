"""Pre-load heavy pipeline dependencies before the first live cycle.

The first catch-up cycle otherwise pays one-time import and model-init costs
(HMM/sklearn, voyageai → transformers, chromadb) inside the analyst pool,
which can stall the cycle worker for tens of minutes and block shutdown.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_LLM_MODES = frozenset({"llm_enhanced", "llm_only"})
_DEFAULT_WARMUP_WAIT_SECONDS = 600.0


class PipelineWarmupGate:
    """Coordinates background pipeline warmup before the first live cycle."""

    def __init__(self) -> None:
        self._ready = threading.Event()
        self._started = False
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    def wait_ready(self, timeout: float | None = None) -> bool:
        return self._ready.wait(timeout=timeout)

    def start_background(self, config: dict[str, Any]) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            if not config.get("pipeline_warmup", True):
                log.info("Pipeline warmup disabled (pipeline_warmup=false)")
                self._ready.set()
                return
            threading.Thread(
                target=self._run,
                args=(config,),
                name="pipeline-warmup",
                daemon=True,
            ).start()

    def _run(self, config: dict[str, Any]) -> None:
        try:
            warm_pipeline_dependencies(config)
        except Exception:
            log.error("Background pipeline warmup failed", exc_info=True)
        finally:
            self._ready.set()


def warmup_wait_seconds() -> float:
    raw = os.getenv("FIRM_PIPELINE_WARMUP_WAIT_SEC", str(_DEFAULT_WARMUP_WAIT_SECONDS))
    try:
        return max(1.0, float(raw))
    except ValueError:
        return _DEFAULT_WARMUP_WAIT_SECONDS


def _strategy_names(config: dict[str, Any]) -> set[str]:
    names = config.get("strategies")
    if names:
        return set(names)
    try:
        from firm.strategies import list_strategies

        return set(list_strategies())
    except Exception:
        return set()


def _uses_llm_agents(config: dict[str, Any]) -> bool:
    modes = config.get("agent_modes") or {}
    return any(m in _LLM_MODES for m in modes.values())


def _warm_hmm() -> None:
    from firm.regime.model import GaussianRegimeModel, RegimeUnavailable

    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 4))
    try:
        GaussianRegimeModel(n_states=3, n_iter=5).fit(X)
        log.info("Pipeline warmup: HMM fit probe succeeded")
    except RegimeUnavailable as exc:
        log.debug("Pipeline warmup: HMM skipped (%s)", exc)
    except Exception:
        log.warning("Pipeline warmup: HMM fit probe failed", exc_info=True)


def _warm_rag_imports(config: dict[str, Any]) -> None:
    """Import the RAG / embedding stack without network I/O."""
    try:
        from firm.llm.config import load_llm_config, rag_config

        cfg = load_llm_config()
        llm_overrides = (config.get("llm_config") or {}).get("rag")
        if isinstance(llm_overrides, dict):
            cfg = {**cfg, "rag": {**(cfg.get("rag") or {}), **llm_overrides}}
        rag = rag_config(cfg)
    except Exception:
        log.warning("Pipeline warmup: could not load RAG config", exc_info=True)
        return

    provider = str(rag.get("embedding_provider", "voyage")).lower()
    if provider == "voyage":
        try:
            import voyageai  # noqa: F401 — pulls langchain_text_splitters / transformers
        except ImportError:
            log.debug("Pipeline warmup: voyageai not installed")
        except Exception:
            log.warning("Pipeline warmup: voyageai import failed", exc_info=True)
    else:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            log.debug("Pipeline warmup: sentence_transformers not installed")
        except Exception:
            log.warning(
                "Pipeline warmup: sentence_transformers import failed", exc_info=True,
            )

    try:
        import chromadb  # noqa: F401
    except ImportError:
        log.debug("Pipeline warmup: chromadb not installed")
    except Exception:
        log.warning("Pipeline warmup: chromadb import failed", exc_info=True)

    try:
        from firm.rag.store import VectorStore

        VectorStore(
            persist_dir=rag.get("persist_dir"),
            embedding_model=rag.get("embedding_model"),
            embedding_provider=rag.get("embedding_provider"),
        )
        log.info("Pipeline warmup: VectorStore initialised")
    except Exception:
        log.warning("Pipeline warmup: VectorStore init failed", exc_info=True)


def warm_pipeline_dependencies(config: dict[str, Any]) -> None:
    """Import and lightly exercise heavy deps before the first live cycle."""
    if not config.get("pipeline_warmup", True):
        log.info("Pipeline warmup disabled (pipeline_warmup=false)")
        return

    log.info("Pipeline warmup starting")
    strategies = _strategy_names(config)
    if "regime_hmm" in strategies:
        _warm_hmm()
    else:
        log.debug("Pipeline warmup: regime_hmm not enabled — skipping HMM")

    if _uses_llm_agents(config):
        _warm_rag_imports(config)
    else:
        log.debug("Pipeline warmup: no LLM agents — skipping RAG imports")

    log.info("Pipeline warmup finished")
