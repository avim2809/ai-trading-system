"""Anonymizes company identity in retrieved news text before it reaches an
LLM prompt.

Glasserman & Lin (arXiv 2309.17322) find LLM-generated trading signals
derived from news text are contaminated by two effects whenever the
company's identity is visible in the prompt:

- **Look-ahead**: the model's own training-data knowledge of what actually
  happened to that specific company leaks into its "prediction" of the
  article's sentiment/implication.
- **Distraction**: the mere presence of a recognizable company name/ticker
  biases the sentiment read independent of the article's actual content.

Their mitigation — shown to generalize out-of-sample — is to strip the
company name and ticker from the text before it is ever shown to the LLM.
This module applies that specifically to the RAG "news" collection (not
"research"/"sec_filings"/"system_docs", which don't carry the same
look-ahead risk profile: they're not raw news-text-about-a-specific-event).
It is wired into :meth:`firm.agents.llm.base_llm_agent.LLMAgentMixin.
_retrieve_context`, the single choke point every LLM-enhanced agent's news
retrieval passes through, so no individual calling agent
(sentiment/fundamental analyst, bull/bear researcher, risk) has to
remember to anonymize itself.

The company-name map below is a first-pass, regex-based approach covering
the live trading universe (config/live.yaml / config/live_alpaca.yaml) —
not a full NER model. It intentionally trades some false positives (e.g.
"MA"/"V" as ordinary English tokens) for simplicity; an unmapped ticker
still gets its raw ticker symbol stripped even without a name entry.
"""

from __future__ import annotations

import re

COMPANY_PLACEHOLDER = "[COMPANY]"
TICKER_PLACEHOLDER = "[TICKER]"

# Ticker -> known name variants. Longest-first substitution is applied at
# anonymize time (not here), so entries need not be pre-sorted by length.
SYMBOL_COMPANY_NAMES: dict[str, tuple[str, ...]] = {
    "AAPL": ("Apple Inc.", "Apple Inc", "Apple"),
    "MSFT": ("Microsoft Corporation", "Microsoft Corp.", "Microsoft Corp", "Microsoft"),
    "NVDA": ("NVIDIA Corporation", "Nvidia Corporation", "Nvidia Corp", "NVIDIA", "Nvidia"),
    "GOOG": ("Alphabet Inc.", "Alphabet Inc", "Alphabet", "Google"),
    "AMZN": ("Amazon.com, Inc.", "Amazon.com Inc", "Amazon.com", "Amazon"),
    "META": (
        "Meta Platforms, Inc.", "Meta Platforms Inc", "Meta Platforms",
        "Meta", "Facebook",
    ),
    "TSLA": ("Tesla, Inc.", "Tesla Inc", "Tesla"),
    "AVGO": ("Broadcom Inc.", "Broadcom Inc", "Broadcom"),
    "AMD": ("Advanced Micro Devices, Inc.", "Advanced Micro Devices"),
    "CRM": ("Salesforce, Inc.", "Salesforce.com", "Salesforce"),
    "NFLX": ("Netflix, Inc.", "Netflix Inc", "Netflix"),
    "ADBE": ("Adobe Inc.", "Adobe Systems", "Adobe"),
    "JPM": ("JPMorgan Chase & Co.", "JPMorgan Chase", "JP Morgan", "JPMorgan"),
    "GS": ("Goldman Sachs Group, Inc.", "Goldman Sachs Group", "Goldman Sachs"),
    "BAC": ("Bank of America Corporation", "Bank of America Corp", "Bank of America"),
    "V": ("Visa Inc.", "Visa Inc"),
    "MA": ("Mastercard Incorporated", "Mastercard Inc", "Mastercard"),
    "JNJ": ("Johnson & Johnson",),
    "UNH": ("UnitedHealth Group Incorporated", "UnitedHealth Group", "UnitedHealth"),
    "LLY": ("Eli Lilly and Company", "Eli Lilly"),
    "XOM": ("Exxon Mobil Corporation", "ExxonMobil", "Exxon Mobil", "Exxon"),
    "CVX": ("Chevron Corporation", "Chevron Corp", "Chevron"),
    "SPY": ("SPDR S&P 500 ETF Trust", "SPDR S&P 500"),
    "QQQ": ("Invesco QQQ Trust", "Invesco QQQ"),
    "IWM": ("iShares Russell 2000 ETF",),
    # Dynamic-universe additions (sp500_dynamic_universe, Alpaca) -- not
    # part of the original static list above, but need the same coverage.
    "CAT": ("Caterpillar Inc.", "Caterpillar Inc", "Caterpillar"),
    "CEG": ("Constellation Energy Corporation", "Constellation Energy"),
    "EQIX": ("Equinix, Inc.", "Equinix Inc", "Equinix"),
    "FCX": ("Freeport-McMoRan Inc.", "Freeport-McMoRan"),
    "GEV": ("GE Vernova Inc.", "GE Vernova"),
    "KO": ("The Coca-Cola Company", "Coca-Cola Company", "Coca-Cola"),
    "LIN": ("Linde plc", "Linde"),
    "NEE": ("NextEra Energy, Inc.", "NextEra Energy"),
    "WELL": ("Welltower Inc.", "Welltower"),
    "WMT": ("Walmart Inc.", "Walmart"),
}

_TICKER_RE_CACHE: dict[str, re.Pattern[str]] = {}
_NAME_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _ticker_pattern(symbol: str) -> re.Pattern[str]:
    pat = _TICKER_RE_CACHE.get(symbol)
    if pat is None:
        # Optional leading "$" cashtag form (e.g. "$AAPL"); \b already
        # anchors correctly whether or not the "$" is present since "$" is
        # itself a non-word character.
        pat = re.compile(r"\$?\b" + re.escape(symbol) + r"\b")
        _TICKER_RE_CACHE[symbol] = pat
    return pat


def _name_pattern(name: str) -> re.Pattern[str]:
    pat = _NAME_RE_CACHE.get(name)
    if pat is None:
        pat = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
        _NAME_RE_CACHE[name] = pat
    return pat


def mentions_symbol(text: str, symbol: str) -> bool:
    """True if *text* actually contains *symbol*'s ticker or a known
    company-name alias.

    A ticker-search API can return an article that doesn't actually
    discuss the queried company (broad keyword/related-ticker matching on
    the provider's side) — this lets an ingestor drop those before they
    reach the RAG collection tagged with the wrong symbol. An unmapped
    symbol only checks the raw ticker (same partial-coverage tradeoff as
    :func:`anonymize_news_text`).
    """
    if not text or not symbol:
        return False
    symbol = symbol.strip().upper()
    if _ticker_pattern(symbol).search(text):
        return True
    return any(
        _name_pattern(name).search(text)
        for name in SYMBOL_COMPANY_NAMES.get(symbol, ())
    )


def anonymize_news_text(text: str, symbol: str) -> str:
    """Strip *symbol*'s ticker and known company-name aliases from *text*.

    No-ops on empty/falsy *text* or *symbol*. An unmapped symbol still has
    its raw ticker stripped via the ticker regex even though it has no
    :data:`SYMBOL_COMPANY_NAMES` entry — a partial anonymization (ticker
    only) is strictly better than none for a universe symbol added after
    this map was last updated.
    """
    if not text:
        return text
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return text

    result = text
    names = sorted(SYMBOL_COMPANY_NAMES.get(symbol, ()), key=len, reverse=True)
    for name in names:
        result = _name_pattern(name).sub(COMPANY_PLACEHOLDER, result)
    result = _ticker_pattern(symbol).sub(TICKER_PLACEHOLDER, result)
    return result
