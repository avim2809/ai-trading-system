"""Tests for in-app fundamentals cache refresh."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from firm.live.fundamentals_refresh import (
    cache_refresh_due,
    maybe_refresh_fundamentals_cache_on_start,
    refresh_fundamentals_cache,
)


@patch("firm.live.fundamentals_refresh.save_refresh_meta")
@patch("firm.live.fundamentals_refresh.ParquetCache")
@patch("firm.live.fundamentals_refresh.FallbackProvider")
def test_refresh_fundamentals_cache_writes_parquet(mock_fb, mock_cache_cls, mock_meta, tmp_path):
    mock_fb.return_value.get_fundamentals.return_value = pd.DataFrame({
        "date": ["2024-01-01"],
        "symbol": ["AAPL"],
        "pe_ratio": [20.0],
    })
    mock_cache_cls.return_value.merge_combined.return_value = mock_fb.return_value.get_fundamentals.return_value

    settings = MagicMock()
    settings.data.cache_dir = str(tmp_path)

    assert refresh_fundamentals_cache(["AAPL"], settings=settings) is True
    mock_cache_cls.return_value.put.assert_called_once()


@patch("firm.live.fundamentals_refresh._refresh_in_background")
@patch("firm.live.fundamentals_refresh.cache_refresh_due", return_value=True)
def test_maybe_refresh_on_start_when_stale(mock_due, mock_bg):
    maybe_refresh_fundamentals_cache_on_start(["AAPL", "MSFT"])
    mock_bg.assert_called_once()
    assert mock_bg.call_args.kwargs["reason"] == "boot"


@patch("firm.live.fundamentals_refresh._refresh_in_background")
@patch("firm.live.fundamentals_refresh.cache_refresh_due", return_value=False)
def test_maybe_refresh_skips_when_fresh(mock_due, mock_bg):
    maybe_refresh_fundamentals_cache_on_start(["AAPL"])
    mock_bg.assert_not_called()


def test_cache_refresh_due_when_no_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "firm.live.fundamentals_refresh.get_settings",
        lambda: MagicMock(data=MagicMock(cache_dir=str(tmp_path))),
    )
    assert cache_refresh_due(str(tmp_path)) is True
