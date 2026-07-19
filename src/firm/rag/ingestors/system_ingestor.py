"""System documentation ingestor.

Indexes the project's own strategy docstrings, config files, and README
for internal knowledge retrieval.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

from firm.rag.chunker import DocumentChunker
from firm.rag.dates import ALWAYS_AVAILABLE_DATE
from firm.rag.ingestors.base_ingestor import BaseIngestor
from firm.rag.models import Document
from firm.rag.store import VectorStore

log = logging.getLogger("firm.rag.ingestors.system")

COLLECTION = "system_docs"

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


class SystemIngestor(BaseIngestor):
    """Indexes system documentation: strategy code, configs, and README."""

    def __init__(self, store: VectorStore, chunker: DocumentChunker) -> None:
        super().__init__(store, chunker)

    def ingest(self, **kwargs: Any) -> int:
        docs: list[Document] = []
        docs.extend(self._ingest_strategies())
        docs.extend(self._ingest_configs())
        docs.extend(self._ingest_readme())

        if docs:
            return self.store.add_documents(COLLECTION, docs)
        return 0

    def _ingest_strategies(self) -> list[Document]:
        """Extract docstrings from strategy modules."""
        docs: list[Document] = []
        strategies_dir = _PROJECT_ROOT / "src" / "firm" / "strategies"
        if not strategies_dir.exists():
            return docs

        for py_file in strategies_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            module_name = f"firm.strategies.{py_file.stem}"
            try:
                mod = importlib.import_module(module_name)
                docstring = mod.__doc__ or ""
                if not docstring.strip():
                    continue

                metadata = {
                    "source": "system",
                    "doc_type": "strategy_doc",
                    "module": module_name,
                    "file": str(py_file.relative_to(_PROJECT_ROOT)),
                    "date": ALWAYS_AVAILABLE_DATE,
                }
                chunks = self.chunker.chunk(docstring.strip(), metadata)
                docs.extend(chunks)
            except Exception:
                log.debug(
                    "strategy_module_import_failed module=%s — falling back to "
                    "reading the file directly", module_name, exc_info=True,
                )
                # Fall back to reading the file directly for docstring
                try:
                    content = py_file.read_text(encoding="utf-8")
                    # Extract module docstring
                    if content.startswith('"""') or content.startswith("'''"):
                        quote = content[:3]
                        end = content.find(quote, 3)
                        if end > 0:
                            docstring = content[3:end].strip()
                            if docstring:
                                metadata = {
                                    "source": "system",
                                    "doc_type": "strategy_doc",
                                    "file": str(py_file.relative_to(_PROJECT_ROOT)),
                                    "date": ALWAYS_AVAILABLE_DATE,
                                }
                                chunks = self.chunker.chunk(docstring, metadata)
                                docs.extend(chunks)
                except Exception:
                    log.warning(
                        "strategy_file_read_failed file=%s — module docstring "
                        "not indexed", py_file, exc_info=True,
                    )

        return docs

    def _ingest_configs(self) -> list[Document]:
        """Index YAML config files."""
        docs: list[Document] = []
        config_dir = _PROJECT_ROOT / "config"
        if not config_dir.exists():
            return docs

        for yaml_file in config_dir.glob("*.yaml"):
            try:
                content = yaml_file.read_text(encoding="utf-8")
                if not content.strip():
                    continue

                metadata = {
                    "source": "system",
                    "doc_type": "config",
                    "file": str(yaml_file.relative_to(_PROJECT_ROOT)),
                    "date": ALWAYS_AVAILABLE_DATE,
                }
                chunks = self.chunker.chunk(
                    f"Configuration file: {yaml_file.name}\n\n{content}", metadata
                )
                docs.extend(chunks)
            except Exception:
                log.warning("config_index_failed file=%s", yaml_file, exc_info=True)

        return docs

    def _ingest_readme(self) -> list[Document]:
        """Index the project README."""
        docs: list[Document] = []
        readme = _PROJECT_ROOT / "README.md"
        if not readme.exists():
            return docs

        try:
            content = readme.read_text(encoding="utf-8")
            if not content.strip():
                return docs

            metadata = {
                "source": "system",
                "doc_type": "readme",
                "file": "README.md",
                "date": ALWAYS_AVAILABLE_DATE,
            }
            chunks = self.chunker.chunk(content, metadata)
            docs.extend(chunks)
        except Exception:
            log.warning("readme_index_failed", exc_info=True)

        return docs
