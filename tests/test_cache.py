"""Tests for ParquetCache round-trip integrity."""

from pathlib import Path

import pandas as pd
import pytest

from firm.data.cache import ParquetCache


@pytest.fixture()
def cache(tmp_path: Path) -> ParquetCache:
    return ParquetCache(str(tmp_path / "cache"))


class TestParquetCache:
    def test_round_trip(self, cache: ParquetCache) -> None:
        """put then get should return an identical DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})
        cache.put("test/key", df)
        result = cache.get("test/key")
        assert result is not None
        pd.testing.assert_frame_equal(result, df)

    def test_has(self, cache: ParquetCache) -> None:
        assert not cache.has("missing")
        cache.put("present", pd.DataFrame({"x": [1]}))
        assert cache.has("present")

    def test_get_missing_returns_none(self, cache: ParquetCache) -> None:
        assert cache.get("nonexistent") is None

    def test_invalidate(self, cache: ParquetCache) -> None:
        cache.put("ephemeral", pd.DataFrame({"v": [42]}))
        assert cache.has("ephemeral")
        cache.invalidate("ephemeral")
        assert not cache.has("ephemeral")
        assert cache.get("ephemeral") is None
