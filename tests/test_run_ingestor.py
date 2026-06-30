"""Phase 1c: run-artifact ingestor builds narrative docs from run reports."""

from __future__ import annotations

import json

from firm.rag.chunker import DocumentChunker
from firm.rag.dates import normalize_date
from firm.rag.ingestors.run_ingestor import COLLECTION, RunArtifactIngestor
from firm.rag.models import Document


class _FakeStore:
    """Records documents added, standing in for the Chroma-backed VectorStore."""

    def __init__(self):
        self.added: dict[str, list[Document]] = {}

    def add_documents(self, collection_name, docs):
        self.added.setdefault(collection_name, []).extend(docs)
        return len(docs)


def _make_run(base, run_id, *, sharpe, end="2021-12-31", notes=""):
    d = base / run_id
    d.mkdir(parents=True)
    (d / "report.json").write_text(json.dumps({
        "portfolio": {"sharpe_ratio": sharpe, "max_drawdown": -0.12,
                      "total_return": 0.2, "cagr": 0.1, "hit_rate": 0.5},
        "period": {"start": "2021-01-01", "end": end},
    }), encoding="utf-8")
    # registry supplies notes/strategies.
    reg = base / "registry.json"
    existing = json.loads(reg.read_text()) if reg.exists() else []
    existing.append({"run_id": run_id, "status": "completed", "notes": notes,
                     "seed": 42, "config": {"strategies": ["momentum"]}})
    reg.write_text(json.dumps(existing), encoding="utf-8")


class TestRunArtifactIngestor:
    def test_ingests_narrative_per_run(self, tmp_path):
        base = tmp_path / "runs"
        base.mkdir()
        _make_run(base, "run_a", sharpe=1.5, notes="baseline momentum sweep")
        _make_run(base, "run_b", sharpe=0.8)

        store = _FakeStore()
        ingestor = RunArtifactIngestor(store, DocumentChunker(), runs_dir=str(base))
        n = ingestor.ingest()

        assert n >= 2
        docs = store.added[COLLECTION]
        text = " ".join(d.text for d in docs)
        assert "run_a" in text and "run_b" in text
        assert "Sharpe ratio 1.50" in text
        assert "baseline momentum sweep" in text  # notes flowed through
        assert "momentum" in text                  # strategies flowed through

    def test_metadata_is_pit_safe(self, tmp_path):
        base = tmp_path / "runs"
        base.mkdir()
        _make_run(base, "run_a", sharpe=1.0, end="2021-06-30")
        store = _FakeStore()
        RunArtifactIngestor(store, DocumentChunker(), runs_dir=str(base)).ingest()
        doc = store.added[COLLECTION][0]
        assert doc.metadata["doc_type"] == "run_note"
        assert doc.metadata["run_id"] == "run_a"
        # availability date == run period-end, normalised
        assert doc.metadata["date"] == normalize_date("2021-06-30")

    def test_empty_runs_dir_ingests_nothing(self, tmp_path):
        store = _FakeStore()
        n = RunArtifactIngestor(store, DocumentChunker(),
                                runs_dir=str(tmp_path / "none")).ingest()
        assert n == 0
        assert store.added == {}
