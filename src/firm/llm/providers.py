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
        if self.env_var is None:
            return True
        if self.key == "gemini":
            # LiteLLM reads GEMINI_API_KEY; Google AI Studio also documents
            # GOOGLE_API_KEY for the same credential.
            return bool(
                os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            )
        return bool(os.environ.get(self.env_var))


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
    "gemini": LLMProvider(
        "gemini", "Google Gemini", "GEMINI_API_KEY",
        "gemini/gemini-2.0-flash", "gemini/", True,
        (
            # gemini-2.5-flash was decommissioned for new API keys (404 from
            # Google, confirmed live 2026-08-05) — dropped from the chain.
            # gemini-2.5-pro hit the identical decommissioning (confirmed
            # live 2026-08-11 through 2026-08-14: every reflection call's
            # first fallback attempt 404'd against it before falling
            # through) — dropped for the same reason. gemini-2.0-flash is
            # the only model in this chain still actually live; re-verify
            # against current Google docs before adding anything above it.
            "gemini/gemini-2.0-flash",
        ),
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
    "openrouter": LLMProvider(
        "openrouter", "OpenRouter (free tier)", "OPENROUTER_API_KEY",
        "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free", "openrouter/", False,
        # Pinned free models give reproducible behaviour but do get rotated
        # out of OpenRouter's free lineup over time — recheck against
        # openrouter.ai/models?max_price=0 if one starts erroring.
        # "openrouter/openrouter/free" auto-routes to whichever :free model
        # fits the request instead, trading reproducibility for durability.
        (
            "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
            "openrouter/openai/gpt-oss-20b:free",
            "openrouter/openrouter/free",
        ),
    ),
}


def list_providers() -> list[LLMProvider]:
    """Return all known providers."""
    return list(PROVIDERS.values())


_PROVIDER_ALIASES: dict[str, str] = {
    # Backward compat + UI shorthand ("google" / bare "gemini" route prefix).
    "google": "gemini",
}


def get_provider(key: str) -> LLMProvider:
    """Look up a provider by key, raising a clear error on an unknown name."""
    resolved = _PROVIDER_ALIASES.get(key, key)
    try:
        return PROVIDERS[resolved]
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
    # Bare Gemini IDs (e.g. "gemini-2.5-flash") carry no "gemini/" prefix.
    if model.startswith("gemini-"):
        return PROVIDERS["gemini"]
    return None


def provider_key_for_model(model: str) -> str | None:
    """Return the registry key for *model*, or None if unrecognised."""
    prov = resolve_provider(model)
    return prov.key if prov else None


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

    if require_key and prov is not None and not prov.is_configured():
        env_hint = prov.env_var or "API key"
        raise RuntimeError(
            f"{prov.label} requires {env_hint} to be set. "
            f"Export it, or pick a configured provider (e.g. Groq's free tier)."
        )

    # LiteLLM only reads GEMINI_API_KEY; mirror GOOGLE_API_KEY when set.
    if prov is not None and prov.key == "gemini":
        if not os.environ.get("GEMINI_API_KEY") and os.environ.get("GOOGLE_API_KEY"):
            os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

    if model:
        cfg["default_model"] = model
    return LLMService(cfg)
