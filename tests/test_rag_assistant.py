"""Phase 2a: grounded TradingAssistant (numeric→SQL, narrative→retrieval).

Uses an injected fake LLM so routing and the SQL invariant are tested with no
network access. The central guarantee under test: numeric answers come from
deterministic SQL, not from the LLM.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from firm.rag.assistant import TradingAssistant
from firm.rag.structured import RunStore


def _runs_dir(tmp_path):
    base = tmp_path / "runs"
    for run_id, sharpe in [("run_a", 1.5), ("run_b", 0.8)]:
        d = base / run_id
        d.mkdir(parents=True)
        (d / "report.json").write_text(json.dumps({
            "portfolio": {"sharpe_ratio": sharpe, "max_drawdown": -0.1},
            "period": {"start": "2021-01-01", "end": "2021-12-31"},
        }), encoding="utf-8")
        pd.DataFrame([], columns=["symbol"]).to_parquet(d / "trades.parquet")
    return str(base)


class FakeLLM:
    """Returns canned SQL for the SQL-gen step, a canned answer otherwise."""

    def __init__(self, sql="SELECT run_id, sharpe_ratio FROM runs "
                            "ORDER BY sharpe_ratio DESC LIMIT 1",
                 answer="The best run is run_a. [run_a]"):
        self.sql = sql
        self.answer = answer
        self.calls: list = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        if "Return ONLY the SQL" in str(messages):
            return self.sql
        return self.answer


@pytest.fixture
def assistant(tmp_path):
    rs = RunStore(_runs_dir(tmp_path))
    llm = FakeLLM()
    a = TradingAssistant(run_store=rs, retriever=None, llm=llm, config={})
    return a, llm


class TestRouting:
    def test_numeric_question_uses_sql(self, assistant):
        a, llm = assistant
        ans = a.ask("Which run had the best Sharpe ratio?")
        assert ans.used_sql is True
        assert ans.sql and ans.sql.lower().startswith("select")
        # Number comes from SQL execution, not the model.
        assert ans.rows[0]["run_id"] == "run_a"
        assert ans.rows[0]["sharpe_ratio"] == 1.5
        assert ans.error is None

    def test_narrative_question_skips_sql(self, assistant):
        a, llm = assistant
        ans = a.ask("Describe what happened in these experiments overall.")
        assert ans.used_sql is False
        assert ans.sql is None
        assert ans.answer == llm.answer

    def test_non_select_sql_is_rejected(self, tmp_path):
        rs = RunStore(_runs_dir(tmp_path))
        llm = FakeLLM(sql="DROP TABLE runs")
        a = TradingAssistant(run_store=rs, retriever=None, llm=llm, config={})
        ans = a.ask("How many trades were there?")
        assert ans.used_sql is False
        assert ans.error and "read-only" in ans.error.lower()
        # Still produces a (synthesised) answer rather than crashing.
        assert ans.answer

    def test_llm_unavailable_returns_data(self, tmp_path):
        class DeadLLM:
            def chat(self, *a, **k):
                raise RuntimeError("no network")
        rs = RunStore(_runs_dir(tmp_path))
        a = TradingAssistant(run_store=rs, retriever=None, llm=DeadLLM(), config={})
        ans = a.ask("Describe the runs.")
        assert ans.error  # synthesis failed, but call didn't raise
        assert "LLM unavailable" in ans.answer


class TestPromptCaching:
    def test_anthropic_marks_static_context_cacheable(self, tmp_path):
        rs = RunStore(_runs_dir(tmp_path))
        cfg = {"provider": {"default_model": "claude-opus-4-8"},
               "assistant": {"prompt_caching": True}}
        a = TradingAssistant(run_store=rs, retriever=None, llm=FakeLLM(), config=cfg)
        msgs = a._messages("BIG STATIC CONTEXT", "instructions", "question")
        system = msgs[0]["content"]
        assert isinstance(system, list)
        assert system[0]["cache_control"] == {"type": "ephemeral"}
        assert system[0]["text"] == "BIG STATIC CONTEXT"

    def test_non_anthropic_uses_plain_string(self, tmp_path):
        rs = RunStore(_runs_dir(tmp_path))
        cfg = {"provider": {"default_model": "groq/llama-3.3-70b-versatile"}}
        a = TradingAssistant(run_store=rs, retriever=None, llm=FakeLLM(), config=cfg)
        msgs = a._messages("BIG STATIC CONTEXT", "instructions", "question")
        assert isinstance(msgs[0]["content"], str)
