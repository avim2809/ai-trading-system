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


class TestParquetCacheApi:
    def test_make_key_and_write_read_round_trip(self, cache: ParquetCache) -> None:
        """Regression: make_key/write/read must exist (scripts_entry uses them)."""
        key = ParquetCache.make_key(
            "prices", provider="polygon", symbols=["MSFT", "AAPL"],
            start="2020-01-01", end="2020-12-31",
        )
        assert "polygon" in key and "AAPL,MSFT" in key  # symbols sorted
        df = pd.DataFrame({"x": [1, 2]})
        cache.write(key, df)
        pd.testing.assert_frame_equal(cache.read(key), df)

    def test_distinct_keys_do_not_collide(self, cache: ParquetCache) -> None:
        cache.put("a/one", pd.DataFrame({"v": [1]}))
        cache.put("a/two", pd.DataFrame({"v": [2]}))
        assert cache.get("a/one")["v"].iloc[0] == 1
        assert cache.get("a/two")["v"].iloc[0] == 2


class TestTokenCompressor:
    def test_truncation_samples_across_document(self) -> None:
        """Regression: fallback truncation must not keep only the head."""
        from firm.llm.compression import TokenCompressor

        sentences = [f"S{i}." for i in range(10)]
        text = " ".join(sentences)
        out = TokenCompressor()._simple_truncate(text, 0.5)
        # A tail sentence must survive (head-only truncation would drop them).
        assert "S9." in out or "S8." in out
        # And the head is still represented.
        assert "S0." in out


class TestResponseCacheKey:
    """Regression: the LLM cache key must include every output-affecting param."""

    def test_temperature_changes_key(self) -> None:
        from firm.llm.cache import ResponseCache

        msgs = [{"role": "user", "content": "hi"}]
        k0 = ResponseCache._hash("gpt-4o", msgs, temperature=0.0)
        k9 = ResponseCache._hash("gpt-4o", msgs, temperature=0.9)
        assert k0 != k9

    def test_json_mode_changes_key(self) -> None:
        from firm.llm.cache import ResponseCache

        msgs = [{"role": "user", "content": "hi"}]
        plain = ResponseCache._hash("gpt-4o", msgs, temperature=0.3, json_mode=False)
        js = ResponseCache._hash("gpt-4o", msgs, temperature=0.3, json_mode=True)
        assert plain != js

    def test_max_tokens_changes_key(self) -> None:
        from firm.llm.cache import ResponseCache

        msgs = [{"role": "user", "content": "hi"}]
        a = ResponseCache._hash("gpt-4o", msgs, temperature=0.3, max_tokens=100)
        b = ResponseCache._hash("gpt-4o", msgs, temperature=0.3, max_tokens=2000)
        assert a != b

    def test_identical_params_same_key(self) -> None:
        from firm.llm.cache import ResponseCache

        msgs = [{"role": "user", "content": "hi"}]
        a = ResponseCache._hash("gpt-4o", msgs, temperature=0.3, max_tokens=100, json_mode=True)
        b = ResponseCache._hash("gpt-4o", msgs, temperature=0.3, max_tokens=100, json_mode=True)
        assert a == b
