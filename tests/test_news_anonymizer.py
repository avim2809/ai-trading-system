"""Tests for firm.agents.llm.news_anonymizer and its wiring into
LLMAgentMixin._retrieve_context.

Per Glasserman & Lin (arXiv 2309.17322), raw news text naming a company
contaminates an LLM's read of it (look-ahead + distraction effects); the
mitigation is to anonymize company name/ticker before the text reaches a
prompt. This must apply only to the "news" collection's documents (via
doc_type metadata), never to "research"/"sec_filings"/"system_docs".
"""

from __future__ import annotations

from unittest.mock import MagicMock

from firm.agents.llm.base_llm_agent import LLMAgentMixin
from firm.agents.llm.news_anonymizer import (
    COMPANY_PLACEHOLDER,
    TICKER_PLACEHOLDER,
    anonymize_news_text,
)
from firm.rag.models import RetrievedDoc


class TestAnonymizeNewsText:
    def test_strips_company_name_and_ticker(self):
        text = "Apple Inc. reported record iPhone sales; AAPL shares rose 3%."
        result = anonymize_news_text(text, "AAPL")
        assert "Apple" not in result
        assert "AAPL" not in result
        assert COMPANY_PLACEHOLDER in result
        assert TICKER_PLACEHOLDER in result

    def test_case_insensitive_company_name_match(self):
        result = anonymize_news_text("apple's new product launch", "AAPL")
        assert "apple" not in result.lower()
        assert COMPANY_PLACEHOLDER in result

    def test_cashtag_ticker_form_is_stripped(self):
        result = anonymize_news_text("Traders are watching $AAPL closely.", "AAPL")
        assert "AAPL" not in result
        assert TICKER_PLACEHOLDER in result

    def test_unmapped_symbol_still_strips_raw_ticker(self):
        result = anonymize_news_text("XYZQ announced a new product today.", "XYZQ")
        assert "XYZQ" not in result
        assert TICKER_PLACEHOLDER in result

    def test_empty_text_is_noop(self):
        assert anonymize_news_text("", "AAPL") == ""

    def test_leaves_unrelated_text_untouched(self):
        text = "The broader market rallied on lower rate expectations."
        assert anonymize_news_text(text, "AAPL") == text


class _DummyAgent(LLMAgentMixin):
    """Minimal concrete agent for exercising _retrieve_context directly."""


class TestRetrieveContextAnonymization:
    def _agent_with_docs(self, docs: list[RetrievedDoc]) -> _DummyAgent:
        agent = _DummyAgent(llm_config={})
        retriever = MagicMock()
        retriever.retrieve_for_symbol.return_value = docs
        agent._retriever = retriever
        # rag_n_results default lookup goes through _enhancement_cfg(); stub
        # it so this test doesn't depend on config/llm.yaml's real contents.
        agent._enhancement_cfg = lambda: {"rag_n_results": 2}
        return agent

    def test_news_doc_text_is_anonymized(self):
        docs = [
            RetrievedDoc(
                doc_id="1",
                text="Apple Inc. (AAPL) beat earnings estimates this quarter.",
                metadata={"source": "massive", "doc_type": "news"},
            ),
        ]
        agent = self._agent_with_docs(docs)
        result = agent._retrieve_context("AAPL", "earnings", collections=["news"])
        assert "Apple" not in result
        assert "AAPL" not in result
        assert COMPANY_PLACEHOLDER in result
        assert TICKER_PLACEHOLDER in result

    def test_non_news_doc_is_left_unchanged(self):
        docs = [
            RetrievedDoc(
                doc_id="2",
                text="Apple Inc. (AAPL) 10-K risk factors discuss supply chain exposure.",
                metadata={"source": "sec_edgar", "doc_type": "10-K"},
            ),
        ]
        agent = self._agent_with_docs(docs)
        result = agent._retrieve_context("AAPL", "risk factors", collections=["sec_filings"])
        assert "Apple Inc." in result
        assert "AAPL" in result

    def test_mixed_collections_only_anonymize_the_news_doc(self):
        docs = [
            RetrievedDoc(
                doc_id="1",
                text="Apple Inc. (AAPL) beat earnings estimates.",
                metadata={"source": "massive", "doc_type": "news"},
            ),
            RetrievedDoc(
                doc_id="2",
                text="Apple Inc. filed an 8-K regarding a new product.",
                metadata={"source": "sec_edgar", "doc_type": "8-K"},
            ),
        ]
        agent = self._agent_with_docs(docs)
        result = agent._retrieve_context(
            "AAPL", "news and filings", collections=["news", "sec_filings"]
        )
        lines = result.split("\n\n")
        assert COMPANY_PLACEHOLDER in lines[0]
        assert "AAPL" not in lines[0]
        assert "Apple Inc." in lines[1]
        assert "8-K" in lines[1]
