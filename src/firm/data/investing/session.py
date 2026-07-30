"""Authenticated Investing.com Pro session — browser-driven fetch, throttle, retry.

Investing.com has no official API, and (confirmed empirically — a plain
`curl` with a realistic browser User-Agent against a public, unauthenticated
page returned HTTP 403) blocks non-browser HTTP clients at the edge. So
**every** request in this module — not just login — is routed through a
real headless-browser engine (Playwright/Chromium), not `requests`. Design
goals, in priority order:

  1. **Fail closed by default.** Every public method raises
     :class:`InvestingDisabledError` unless
     ``settings.investing_scraper_enabled`` is truthy — constructing/importing
     this class never touches the network on its own.
  2. **One browser, reused.** A single Chromium instance/context is launched
     lazily on first use and kept alive across `get()` calls within a run
     (each call opens/closes its own page) — a fresh browser per fetch would
     multiply an already-nontrivial resource cost on a 2-core/3.3GB VPS.
     Call :meth:`close` (or use as a context manager) when done for the run;
     the once-daily live scheduler cadence means this is a bounded,
     once-a-day cost, not a per-cycle-loop one.
  3. **Log in once, reuse the session.** The login step is the most
     detectable/fragile part of scraping a login-gated account, so the
     browser's full storage state (cookies + localStorage) is persisted to
     disk (``data/cache/investing_storage_state.json``, mode 600, gitignored
     via ``data/cache/``) and reloaded on construction — `login()` is a
     no-op once a prior session is loaded.
  4. **Polite pacing.** A minimum interval + jitter between every request
     (default 3s) — this project's calendar-cadence data source, run at most
     once/day by the live scheduler, never needs to hammer the site.
  5. **Two interchangeable mechanisms** (``investing_auth_method``):
     - ``"playwright"`` (default): the browser-driven path described above —
       no need to know/guess any internal endpoint. Requires the optional
       ``investing`` extra (lazily imported) + a one-time
       ``playwright install chromium``.
     - ``"endpoint"``: a plain ``requests.Session`` POSTing directly to a
       reverse-engineered login endpoint and reusing it for subsequent GETs.
       Lighter-weight *if* it works, but the 403 finding above means it's
       unlikely to get past the edge bot-detection for GETs either — kept
       as an option for a differently-configured account/IP or a future
       finding, not the recommended path. The URL/field names are
       deliberately NOT hardcoded anywhere in this file — this codebase does
       not fabricate third-party API/endpoint details; supply them via
       ``investing_login_page_url`` / ``investing_login_post_url`` /
       ``investing_login_field_map`` in ``.env`` after capturing the real
       request from a browser's devtools Network tab. See
       ``docs/investing_pro_integration.md`` (Phase 0 spike).

Status (2026-07-31): every step of the login flow has been verified against
the live site via a real browser's devtools (this environment cannot reach
the live, authenticated site itself) — the cookie-consent banner, the
header "Sign In" trigger, the "Sign in with Email" toggle (the modal offers
"Continue with Google" first), and the email/password/submit selectors (see
``_COOKIE_BANNER_SELECTOR`` / ``_LOGIN_TRIGGER_SELECTORS`` /
``_EMAIL_LOGIN_TOGGLE_TEXTS`` / ``_EMAIL_SELECTORS`` / ``_PASSWORD_SELECTORS``
/ ``_SUBMIT_SELECTORS`` below). Not yet verified: an actual end-to-end login
with real credentials (only the DOM structure was inspected manually) — that
first real run is what ``scripts/investing_smoke.py`` is for, and remains
the go/no-go signal if anything past this point (2FA, a CAPTCHA/DataDome
challenge, a post-login redirect this code doesn't expect) surfaces.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from firm.config import Settings, get_settings
from firm.data.providers.base import ProviderError

log = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://www.investing.com"
_DEFAULT_COOKIE_PATH = Path("data/cache/investing_cookies.json")
_DEFAULT_STORAGE_STATE_PATH = Path("data/cache/investing_storage_state.json")
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_DEFAULT_MIN_INTERVAL_SEC = 3.0
_MAX_RETRIES = 3

# Verified 2026-07-31 against the live site (via a real browser's devtools,
# not this environment): the header "Sign In" trigger is
# button[data-test='login-btn'] (text-based selectors like get_by_text
# unreliably match/click it — a OneTrust cookie-consent banner's dark-filter
# overlay also intercepts clicks until #onetrust-accept-btn-handler is
# dismissed first, see _pw_dismiss_cookie_banner below). Email/password
# fields confirmed as input[name='email'] (type="text", NOT type="email")
# and input[name='password'] — kept as the first candidates below. The
# submit-button selector and the step(s) needed to reveal the email/password
# fields from the initial modal (which shows a Google sign-in option first)
# are still unverified — see the "Known gap" note in the module docstring.
_COOKIE_BANNER_SELECTOR = "#onetrust-accept-btn-handler"
_LOGIN_TRIGGER_SELECTORS = ("button[data-test='login-btn']",)
_LOGIN_TRIGGER_TEXTS = ("Sign In", "Log In", "Login")
# The initial modal offers "Continue with Google" first; this text-based
# toggle (no stable data-test/id — just a CSS-module-hashed class) reveals
# the actual email/password fields. Best-effort/non-fatal: some flows may
# show the fields directly without this extra step.
_EMAIL_LOGIN_TOGGLE_TEXTS = ("Sign in with Email",)
_EMAIL_SELECTORS = ("input[name='email']", "input[type='email']", "#loginFormUser_email")
_PASSWORD_SELECTORS = ("input[name='password']", "input[type='password']")
_SUBMIT_SELECTORS = ("button[type='submit']", "input[type='submit']")


def _pw_click_first_match(page: Any, candidates: tuple[str, ...], *, by_text: bool = False) -> bool:
    """Try each selector/text in turn; return True on the first click that works."""
    for candidate in candidates:
        try:
            locator = page.get_by_text(candidate, exact=False) if by_text else page.locator(candidate)
            locator.first.click(timeout=3000)
            return True
        except Exception:
            continue
    return False


def _pw_fill_first_match(page: Any, selectors: tuple[str, ...], value: str) -> bool:
    """Try each selector in turn; return True on the first fill that works."""
    for selector in selectors:
        try:
            page.fill(selector, value, timeout=3000)
            return True
        except Exception:
            continue
    return False


def _pw_dismiss_cookie_banner(page: Any) -> None:
    """Best-effort: dismiss the OneTrust cookie-consent banner if present.

    Verified 2026-07-31: the banner's dark-filter overlay intercepts clicks
    on the login trigger until dismissed, and (a real race observed live)
    the banner can still be mounting when a fixed short sleep would have
    already given up — so this explicitly waits for the accept button to
    become visible rather than guessing a delay. Silently no-ops if the
    banner never appears (e.g. a region without a consent requirement, or
    a previously-dismissed choice already recorded in the browser profile).
    """
    try:
        page.wait_for_selector(_COOKIE_BANNER_SELECTOR, state="visible", timeout=8000)
        page.locator(_COOKIE_BANNER_SELECTOR).click(timeout=3000)
    except Exception:
        pass


@dataclass
class PageResponse:
    """Minimal response wrapper shared by both fetch mechanisms.

    Mirrors just the bits of ``requests.Response`` callers actually use
    (status_code / text / content) so fetchers built on top of
    :class:`InvestingSession` don't need to know which transport served them.
    """

    status_code: int
    text: str

    @property
    def content(self) -> bytes:
        return self.text.encode("utf-8", errors="replace")

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 400


class InvestingDisabledError(ProviderError):
    """Raised when the Investing.com scraper is used while disabled/unconfigured."""


class InvestingSession:
    """Authenticated, session-persisting client for Investing.com Pro."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        cookie_path: Path | str = _DEFAULT_COOKIE_PATH,
        storage_state_path: Path | str = _DEFAULT_STORAGE_STATE_PATH,
        min_interval_sec: float | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.base_url = base_url.rstrip("/")
        self.cookie_path = Path(cookie_path)  # used only by the "endpoint" method
        self.storage_state_path = Path(storage_state_path)  # used only by "playwright"
        self._min_interval = (
            min_interval_sec if min_interval_sec is not None else _DEFAULT_MIN_INTERVAL_SEC
        )
        self._last_request_at = 0.0

        # "endpoint" method state
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": _DEFAULT_USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
        )
        self._authenticated = False
        self._load_cookies()

        # "playwright" method state — lazily populated by _ensure_browser().
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        if self.storage_state_path.exists():
            self._authenticated = True

    def __enter__(self) -> "InvestingSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the browser (if one was launched). Safe to call anytime."""
        if self._context is not None:
            self._save_storage_state()
        for obj, closer in (
            (self._browser, "close"),
            (self._playwright, "stop"),
        ):
            if obj is not None:
                try:
                    getattr(obj, closer)()
                except Exception as exc:
                    log.warning("investing_browser_close_failed: %s", exc, exc_info=True)
        self._context = None
        self._browser = None
        self._playwright = None

    # ------------------------------------------------------------------
    # Master switch
    # ------------------------------------------------------------------

    def _require_enabled(self) -> None:
        if not getattr(self.settings, "investing_scraper_enabled", False):
            raise InvestingDisabledError(
                "Investing.com scraper is disabled (INVESTING_SCRAPER_ENABLED "
                "unset/false) — this is the default. Set it truthy in .env "
                "only once you've reviewed the ToS/account-suspension risk in "
                "docs/investing_pro_integration.md."
            )

    def _auth_method(self) -> str:
        return getattr(self.settings, "investing_auth_method", "playwright")

    # ------------------------------------------------------------------
    # "endpoint" method: cookie-jar persistence (requests.Session)
    # ------------------------------------------------------------------

    def _load_cookies(self) -> None:
        if not self.cookie_path.exists():
            return
        try:
            import json

            raw = json.loads(self.cookie_path.read_text(encoding="utf-8"))
            jar = requests.utils.cookiejar_from_dict(raw)
            self._session.cookies.update(jar)
            self._authenticated = bool(raw)
        except Exception as exc:
            log.warning(
                "investing_cookie_load_failed path=%s: %s",
                self.cookie_path, exc, exc_info=True,
            )

    def _save_cookies(self) -> None:
        try:
            import json

            self.cookie_path.parent.mkdir(parents=True, exist_ok=True)
            raw = requests.utils.dict_from_cookiejar(self._session.cookies)
            self.cookie_path.write_text(json.dumps(raw), encoding="utf-8")
            self.cookie_path.chmod(0o600)
        except Exception as exc:
            log.warning(
                "investing_cookie_save_failed path=%s: %s",
                self.cookie_path, exc, exc_info=True,
            )

    # ------------------------------------------------------------------
    # "playwright" method: persistent browser + storage-state persistence
    # ------------------------------------------------------------------

    def _ensure_browser(self) -> None:
        if self._context is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise ProviderError(
                "Playwright is not installed. Install the 'investing' extra "
                "(pip install -e '.[investing]') then run "
                "`playwright install chromium` once, or switch "
                "investing_auth_method to 'endpoint'."
            ) from exc

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        state = str(self.storage_state_path) if self.storage_state_path.exists() else None
        self._context = self._browser.new_context(
            user_agent=_DEFAULT_USER_AGENT, storage_state=state,
        )

    def _save_storage_state(self) -> None:
        try:
            self.storage_state_path.parent.mkdir(parents=True, exist_ok=True)
            self._context.storage_state(path=str(self.storage_state_path))
            self.storage_state_path.chmod(0o600)
        except Exception as exc:
            log.warning(
                "investing_storage_state_save_failed path=%s: %s",
                self.storage_state_path, exc, exc_info=True,
            )

    # ------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------

    def _field_map(self) -> dict[str, str]:
        raw = getattr(self.settings, "investing_login_field_map", "") or ""
        mapping: dict[str, str] = {}
        for part in raw.split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            role, field_name = part.split(":", 1)
            mapping[role.strip()] = field_name.strip()
        return mapping

    def login(self, *, force: bool = False) -> None:
        """Authenticate and persist the resulting session state.

        No-op if a prior session was already loaded/authenticated, unless
        *force* is True (used internally on a 401/403 to re-authenticate
        once). Dispatches to the Playwright or endpoint-POST mechanism per
        ``settings.investing_auth_method`` — see the module docstring.
        """
        self._require_enabled()
        if self._authenticated and not force:
            return

        method = self._auth_method()
        if method == "playwright":
            self._playwright_login()
        elif method == "endpoint":
            self._endpoint_login()
        else:
            raise ProviderError(
                f"Unknown investing_auth_method={method!r} "
                "(expected 'playwright' or 'endpoint')."
            )

    def _playwright_login(self) -> None:
        """Drive the real login form with a headless browser, no endpoint needed."""
        try:
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        except ImportError as exc:
            raise ProviderError(
                "Playwright is not installed. Install the 'investing' extra "
                "(pip install -e '.[investing]') then run "
                "`playwright install chromium` once, or switch "
                "investing_auth_method to 'endpoint'."
            ) from exc

        email = getattr(self.settings, "investing_email", "")
        password = getattr(self.settings, "investing_password", "")
        if not email or not password:
            raise ProviderError(
                "InvestingSession Playwright login requires investing_email "
                "and investing_password to be set in .env."
            )
        login_url = getattr(self.settings, "investing_login_page_url", "") or self.base_url

        self._ensure_browser()
        page = self._context.new_page()
        try:
            try:
                page.goto(login_url, wait_until="domcontentloaded", timeout=30_000)
                # Verified 2026-07-31: a OneTrust cookie-consent banner's
                # dark-filter overlay intercepts clicks on the login trigger
                # until dismissed.
                _pw_dismiss_cookie_banner(page)
                # The header "Sign In" trigger — data-test selector verified
                # against the live site; text-based matching is kept as a
                # fallback (unreliable on its own — see the module docstring).
                if not _pw_click_first_match(page, _LOGIN_TRIGGER_SELECTORS):
                    _pw_click_first_match(page, _LOGIN_TRIGGER_TEXTS, by_text=True)
                page.wait_for_timeout(500)  # let the modal finish animating in
                # The modal offers "Continue with Google" first — switch to
                # the email/password form. Best-effort: no-op if the fields
                # are already visible without this step.
                _pw_click_first_match(page, _EMAIL_LOGIN_TOGGLE_TEXTS, by_text=True)
                page.wait_for_timeout(300)
                if not _pw_fill_first_match(page, _EMAIL_SELECTORS, email):
                    raise ProviderError(
                        "Could not find an email field on the Investing.com "
                        "login form — _EMAIL_SELECTORS in "
                        "firm.data.investing.session needs updating for the "
                        "current site layout."
                    )
                if not _pw_fill_first_match(page, _PASSWORD_SELECTORS, password):
                    raise ProviderError(
                        "Could not find a password field on the Investing.com "
                        "login form — _PASSWORD_SELECTORS needs updating."
                    )
                if not _pw_click_first_match(page, _SUBMIT_SELECTORS):
                    raise ProviderError(
                        "Could not find a submit button on the Investing.com "
                        "login form — _SUBMIT_SELECTORS needs updating."
                    )
                page.wait_for_timeout(2000)  # let post-login redirect settle
            except PlaywrightTimeoutError as exc:
                raise ProviderError(
                    f"Playwright login timed out — the page may be showing a "
                    f"CAPTCHA/DataDome/Cloudflare challenge, or a selector "
                    f"needs updating for the current site layout: {exc}"
                ) from exc
        finally:
            page.close()

        self._authenticated = True
        self._save_storage_state()
        log.info("investing_login_succeeded method=playwright")

    def _endpoint_login(self) -> None:
        """POST directly to a reverse-engineered login endpoint (no browser).

        Raises ``ProviderError`` with a clear, actionable message if the
        login endpoint/field names haven't been configured — see the module
        docstring; this project does not guess them. Note: the 403 finding
        documented in the module docstring means this path is unlikely to
        get past Investing.com's edge bot-detection even if the endpoint
        details are correct — kept as an option, not the recommended path.
        """
        page_url = getattr(self.settings, "investing_login_page_url", "")
        post_url = getattr(self.settings, "investing_login_post_url", "")
        field_map = self._field_map()
        email = getattr(self.settings, "investing_email", "")
        password = getattr(self.settings, "investing_password", "")

        missing = [
            name
            for name, value in (
                ("investing_login_page_url", page_url),
                ("investing_login_post_url", post_url),
                ("investing_email", email),
                ("investing_password", password),
            )
            if not value
        ]
        if not {"email", "password"} <= field_map.keys():
            missing.append("investing_login_field_map")
        if missing:
            raise ProviderError(
                f"InvestingSession.login() is not configured: missing {missing}. "
                "These are deliberately unset by default (this codebase does not "
                "guess third-party login internals) — capture the real login "
                "request from your browser's devtools Network tab and set them "
                "in .env. See docs/investing_pro_integration.md."
            )

        self._throttle()
        try:
            page_resp = self._session.get(
                page_url, timeout=self.settings.request_timeout_seconds
            )
            page_resp.raise_for_status()
        except requests.RequestException as exc:
            raise ProviderError(f"Failed to load Investing.com login page: {exc}") from exc

        payload = {field_map["email"]: email, field_map["password"]: password}
        self._throttle()
        try:
            resp = self._session.post(
                post_url, data=payload, timeout=self.settings.request_timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ProviderError(f"Investing.com login request failed: {exc}") from exc
        if resp.status_code in (401, 403):
            raise ProviderError(
                f"Investing.com login rejected (HTTP {resp.status_code}) — check "
                "credentials, or the site may be challenging the request as a "
                "bot (CAPTCHA/DataDome/Cloudflare)."
            )
        if not resp.ok:
            raise ProviderError(f"Investing.com login failed: HTTP {resp.status_code}")

        self._authenticated = True
        self._save_cookies()
        log.info("investing_login_succeeded method=endpoint")

    # ------------------------------------------------------------------
    # Throttled, authenticated fetch (dispatches per investing_auth_method)
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait + random.uniform(0.1, 0.6))  # jitter
        self._last_request_at = time.monotonic()

    def get(self, path: str, *, params: dict[str, Any] | None = None) -> PageResponse:
        """Throttled, retrying, authenticated GET. Raises ProviderError on failure."""
        self._require_enabled()
        if self._auth_method() == "playwright":
            # Must run before login(): a prior on-disk storage_state makes
            # login() a no-op, which would otherwise leave self._context
            # unset (browser never launched) the first time get() is called
            # in a new process.
            self._ensure_browser()
        self.login()  # no-op once authenticated
        url = path if path.startswith("http") else f"{self.base_url}/{path.lstrip('/')}"
        if params:
            url = requests.models.PreparedRequest().prepare_url(url, params) or url

        if self._auth_method() == "playwright":
            return self._playwright_get(url)
        return self._endpoint_get(url)

    def _playwright_get(self, url: str) -> PageResponse:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            self._throttle()
            page = self._context.new_page()
            try:
                try:
                    response = page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                except PlaywrightTimeoutError as exc:
                    last_exc = ProviderError(f"Timed out loading {url}: {exc}")
                    log.warning("investing_request_timeout url=%s attempt=%d", url, attempt)
                    if attempt < _MAX_RETRIES:
                        time.sleep(min(2.0 ** attempt, 10.0))
                    continue

                status = response.status if response is not None else 200
                if status in (401, 403):
                    # Auth refresh, not a transient error — retry immediately,
                    # no backoff (matches _endpoint_get's 401 handling).
                    self._authenticated = False
                    self.login(force=True)
                    continue
                if status in (429, 500, 502, 503, 504):
                    last_exc = ProviderError(f"{url} returned HTTP {status}")
                    log.warning("investing_retryable_status url=%s status=%d", url, status)
                    if attempt < _MAX_RETRIES:
                        time.sleep(min(2.0 ** attempt, 10.0))
                    continue
                if status >= 400:
                    raise ProviderError(f"{url} returned HTTP {status}: {page.content()[:200]}")

                return PageResponse(status_code=status, text=page.content())
            finally:
                page.close()
        raise ProviderError(
            f"Request to {url} failed after {_MAX_RETRIES} attempts"
        ) from last_exc

    def _endpoint_get(self, url: str) -> PageResponse:
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            self._throttle()
            try:
                resp = self._session.get(url, timeout=self.settings.request_timeout_seconds)
            except requests.RequestException as exc:
                last_exc = exc
                log.warning(
                    "investing_request_failed url=%s attempt=%d: %s",
                    url, attempt, exc, exc_info=True,
                )
            else:
                if resp.status_code == 401:
                    self._authenticated = False
                    self.login(force=True)
                    continue
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_exc = ProviderError(f"{url} returned HTTP {resp.status_code}")
                    log.warning(
                        "investing_retryable_status url=%s status=%d",
                        url, resp.status_code,
                    )
                elif not resp.ok:
                    raise ProviderError(
                        f"{url} returned HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                else:
                    return PageResponse(status_code=resp.status_code, text=resp.text)
            if attempt < _MAX_RETRIES:
                time.sleep(min(2.0 ** attempt, 10.0))
        raise ProviderError(
            f"Request to {url} failed after {_MAX_RETRIES} attempts"
        ) from last_exc
