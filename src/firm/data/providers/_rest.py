"""Small shared HTTP helper used by the concrete REST adapters.

Centralizes session reuse, timeouts, simple retry/backoff and JSON decoding so
each vendor adapter stays focused on response-shape normalization. All transport
failures are converted into :class:`firm.data.providers.base.ProviderError`.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from firm.config import Settings, get_settings
from firm.data.providers.base import ProviderError
from firm.logging_setup import get_logger

log = get_logger(__name__)


class RestClient:
    """Thin ``requests.Session`` wrapper with retry/backoff and JSON parsing."""

    def __init__(self, base_url: str, settings: Settings | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.settings = settings or get_settings()
        self._session = requests.Session()

    def get_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """GET ``base_url + path`` and return decoded JSON.

        Args:
            path: Path appended to ``base_url`` (leading slash optional), or a
                full ``http(s)://`` URL (used as-is).
            params: Query-string parameters.
            headers: Extra request headers.

        Raises:
            ProviderError: On network failure, non-2xx status, or bad JSON after
                exhausting retries.
        """
        url = path if path.startswith("http") else f"{self.base_url}/{path.lstrip('/')}"
        timeout = self.settings.request_timeout_seconds
        attempts = max(1, self.settings.max_retries)

        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                resp = self._session.get(
                    url, params=params, headers=headers, timeout=timeout
                )
            except requests.RequestException as exc:  # network-level failure
                last_exc = exc
                log.warning(
                    "http_request_failed",
                    extra={"context": {"url": url, "attempt": attempt, "error": str(exc)}},
                )
            else:
                # Retry transient server / rate-limit responses.
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_exc = ProviderError(
                        f"{url} returned HTTP {resp.status_code}"
                    )
                    log.warning(
                        "http_retryable_status",
                        extra={"context": {"url": url, "status": resp.status_code}},
                    )
                elif not resp.ok:
                    raise ProviderError(
                        f"{url} returned HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                else:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        raise ProviderError(f"Invalid JSON from {url}: {exc}") from exc

            if attempt < attempts:
                time.sleep(min(2.0 ** attempt, 10.0))  # exponential backoff

        raise ProviderError(f"Request to {url} failed after {attempts} attempts") from last_exc
