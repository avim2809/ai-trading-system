"""Tests for firm.runtime.load_ai_scores."""

from __future__ import annotations

import pandas as pd

from firm.config import Settings
from firm.data.cache import ParquetCache
from firm.runtime import load_ai_scores


def _settings(cache_dir) -> Settings:
    s = Settings()
    s.data.cache_dir = str(cache_dir)
    return s


class TestLoadAiScores:
    def test_reads_combined_ai_scores_key(self, tmp_path):
        cache = ParquetCache(tmp_path)
        df = pd.DataFrame({
            "date": ["2026-07-30"],
            "symbol": ["AAPL"],
            "ai_score": [7],
            "fundamental_score": [7],
            "technical_score": [5],
            "sentiment_score": [5],
            "low_risk_score": [6],
        })
        cache.put("combined/ai_scores", df)

        result = load_ai_scores(_settings(tmp_path))
        assert result is not None
        assert list(result["symbol"]) == ["AAPL"]

    def test_returns_none_when_absent(self, tmp_path):
        assert load_ai_scores(_settings(tmp_path)) is None
