"""Token compression utilities for reducing LLM input costs.

Attempts LLMLingua-based compression when available, falling back to a simple
sentence-level truncation strategy.
"""

from __future__ import annotations


def _try_tiktoken_count(text: str) -> int | None:
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        return len(enc.encode(text))
    except Exception:
        return None


class TokenCompressor:
    """Compress text to reduce token usage while preserving semantic content."""

    def __init__(self, target_ratio: float = 0.5) -> None:
        self.target_ratio = target_ratio
        self._lingua = None
        self._lingua_checked = False

    def _get_lingua(self):
        if not self._lingua_checked:
            self._lingua_checked = True
            try:
                from llmlingua import PromptCompressor
                self._lingua = PromptCompressor(
                    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                    use_llmlingua2=True,
                )
            except (ImportError, Exception):
                self._lingua = None
        return self._lingua

    def compress(self, text: str, target_ratio: float | None = None) -> str:
        """Compress text to approximately target_ratio of its original length.

        Uses LLMLingua if available, otherwise falls back to sentence truncation.
        """
        if not text:
            return text

        ratio = target_ratio or self.target_ratio

        lingua = self._get_lingua()
        if lingua is not None:
            try:
                result = lingua.compress_prompt(
                    [text],
                    rate=ratio,
                    force_tokens=["\n", ".", "?", "!"],
                )
                return result.get("compressed_prompt", text)
            except Exception:
                pass

        return self._simple_truncate(text, ratio)

    def _simple_truncate(self, text: str, ratio: float) -> str:
        """Sentence-level truncation keeping the first `ratio` fraction."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences:
            return text
        target_count = max(1, int(len(sentences) * ratio))
        return " ".join(sentences[:target_count])

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken if available, else heuristic."""
        count = _try_tiktoken_count(text)
        if count is not None:
            return count
        return len(text) // 4
