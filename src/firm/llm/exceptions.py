"""LLM-layer exceptions."""


class LLMEnhancementSkipped(Exception):
    """Raised when cost policy skips an LLM call (e.g. cache_only miss)."""
