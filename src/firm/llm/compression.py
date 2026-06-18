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
        """Sentence-level downsampling spread across the whole document.

        Selects ``ratio`` of the sentences sampled at a uniform stride (in
        original order) rather than only the leading fraction, so material in
        the body/tail of a document (e.g. a filing's risk factors) is not
        silently discarded.
        """
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        n = len(sentences)
        if n <= 1:
            return text
        target_count = max(1, int(n * ratio))
        if target_count >= n:
            return text
        # Evenly-spaced indices across [0, n) preserving original order.
        step = n / target_count
        idx = sorted({min(n - 1, int(i * step)) for i in range(target_count)})
        return " ".join(sentences[i] for i in idx)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken if available, else heuristic."""
        count = _try_tiktoken_count(text)
        if count is not None:
            return count
        return len(text) // 4
