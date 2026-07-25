"""Lightweight headline sentiment scoring for providers without native scores."""

from __future__ import annotations

_POSITIVE_WORDS = frozenset({
    "beat", "beats", "surge", "surges", "soar", "soars", "jump", "jumps", "rally",
    "rallies", "gain", "gains", "rise", "rises", "upgrade", "upgrades", "raised",
    "raises", "outperform", "strong", "record", "profit", "growth", "bullish",
    "boost", "boosts", "win", "wins", "approval", "approved", "beat-estimates",
    "top", "tops", "positive", "expands", "expansion", "dividend", "buyback",
})
_NEGATIVE_WORDS = frozenset({
    "miss", "misses", "missed", "plunge", "plunges", "drop", "drops", "fall",
    "falls", "slump", "slumps", "decline", "declines", "downgrade", "downgrades",
    "cut", "cuts", "loss", "losses", "weak", "warning", "warns", "bearish",
    "lawsuit", "probe", "investigation", "recall", "bankruptcy", "fraud",
    "slashes", "slash", "negative", "halts", "halt", "layoffs", "default",
})


def score_headline(text: str) -> float:
    """Return a sentiment score in [-1, 1] from headline word polarity."""
    if not text:
        return 0.0
    tokens = [t.strip(".,!?:;'\"()[]").lower() for t in text.split()]
    pos = sum(1 for t in tokens if t in _POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in _NEGATIVE_WORDS)
    if pos == 0 and neg == 0:
        return 0.0
    return float((pos - neg) / (pos + neg))
