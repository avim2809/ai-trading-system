"""Registry + factory for the common LLM providers.

The system speaks to every provider through LiteLLM (see ``provider.py``), so a
"provider" here is just metadata: which env var holds its key, a sensible default
model, the LiteLLM route prefix used to recognise its models, and whether it
honours Anthropic-style ``cache_control`` prompt-cache breakpoints.

``build_llm_service()`` is the factory: give it a provider name and/or a model
string and it returns a configured ``LLMService``, validating that the key is
present so failures are clear and early rather than a 401 deep in a backtest.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMProvider:
    key: str                      # short id, e.g. "anthropic"
    label: str                    # human label
    env_var: str | None           # API-key env var (None for local Ollama)
    default_model: str            # LiteLLM model string
    prefix: str                   # LiteLLM route prefix used to detect the provider
    supports_cache_control: bool  # honours Anthropic-style cache_control breakpoints
    example_models: tuple[str, ...] = field(default_factory=tuple)

    def is_configured(self) -> bool:
        """True when the provider needs no key, or its key is set in the env."""
        return self.env_var is None or bool(os.environ.get(self.env_var))


# Model IDs verified against current provider docs (Claude: Opus 4.8 / Sonnet 4.6
# / Haiku 4.5). Only Anthropic honours explicit cache_control breakpoints; OpenAI
# / Gemini / DeepSeek cache automatically server-side (no client action needed).
PROVIDERS: dict[str, LLMProvider] = {
    "groq": LLMProvider(
        "groq", "Groq (free tier)", "GROQ_API_KEY",
        "groq/llama-3.3-70b-versatile", "groq/", False,
        ("groq/llama-3.3-70b-versatile", "groq/llama-3.1-8b-instant"),
    ),
    "anthropic": LLMProvider(
        "anthropic", "Anthropic Claude", "ANTHROPIC_API_KEY",
        "anthropic/claude-opus-4-8", "anthropic/", True,
        ("anthropic/claude-opus-4-8", "anthropic/claude-sonnet-4-6",
         "anthropic/claude-haiku-4-5"),
    ),
    "openai": LLMProvider(
        "openai", "OpenAI", "OPENAI_API_KEY",
        "gpt-4o", "gpt-", True,
        ("gpt-4o", "gpt-4o-mini"),
    ),
    "google": LLMProvider(
        "google", "Google Gemini", "GEMINI_API_KEY",
        "gemini/gemini-1.5-pro", "gemini/", True,
        ("gemini/gemini-1.5-pro", "gemini/gemini-1.5-flash"),
    ),
    "mistral": LLMProvider(
        "mistral", "Mistral", "MISTRAL_API_KEY",
        "mistral/mistral-large-latest", "mistral/", False,
        ("mistral/mistral-large-latest", "mistral/mistral-small-latest"),
    ),
    "deepseek": LLMProvider(
        "deepseek", "DeepSeek", "DEEPSEEK_API_KEY",
        "deepseek/deepseek-chat", "deepseek/", True,
        ("deepseek/deepseek-chat", "deepseek/deepseek-reasoner"),
    ),
    "ollama": LLMProvider(
        "ollama", "Ollama (local)", None,
        "ollama/llama3", "ollama/", False,
        ("ollama/llama3", "ollama/qwen2.5"),
    ),
}


def list_providers() -> list[LLMProvider]:
    """Return all known providers."""
    return list(PROVIDERS.values())


def get_provider(key: str) -> LLMProvider:
    """Look up a provider by key, raising a clear error on an unknown name."""
    try:
        return PROVIDERS[key]
    except KeyError:
        raise ValueError(
            f"Unknown LLM provider {key!r}. Known: {', '.join(PROVIDERS)}"
        )


def resolve_provider(model: str) -> LLMProvider | None:
    """Infer the provider from a LiteLLM model string, or None if unrecognised."""
    if not model:
        return None
    for prov in PROVIDERS.values():
        if model.startswith(prov.prefix):
            return prov
    # Bare Anthropic IDs (e.g. "claude-opus-4-8") carry no "anthropic/" prefix.
    if "claude" in model:
        return PROVIDERS["anthropic"]
    return None


def available_providers() -> list[LLMProvider]:
    """Return providers whose API key is present (or that need none)."""
    return [p for p in PROVIDERS.values() if p.is_configured()]


def build_llm_service(
    provider: str | None = None,
    model: str | None = None,
    config: dict | None = None,
    require_key: bool = True,
):
    """Build an ``LLMService`` for *provider*/*model*, validating the key is set.

    Pass a provider name (uses its default model), a model string (provider is
    inferred), or both. Other config keys (temperature, cache settings) are
    passed through unchanged.
    """
    from firm.llm.provider import LLMService

    cfg = dict(config or {})

    prov = get_provider(provider) if provider else resolve_provider(model or "")
    if model is None:
        model = prov.default_model if prov else cfg.get("default_model")
    if prov is None:
        prov = resolve_provider(model or "")

    if require_key and prov is not None and prov.env_var and not os.environ.get(prov.env_var):
        raise RuntimeError(
            f"{prov.label} requires {prov.env_var} to be set. "
            f"Export it, or pick a configured provider (e.g. Groq's free tier)."
        )

    if model:
        cfg["default_model"] = model
    return LLMService(cfg)
