"""Phase 0 go/no-go spike: does authenticated Investing.com Pro access work?

Run from the repo root:
    python scripts/investing_smoke.py

Requires INVESTING_SCRAPER_ENABLED=1 and INVESTING_EMAIL/PASSWORD in .env.
Default auth method is "playwright" (drives the real login form with a
headless browser — needs the `investing` extra installed:
`pip install -e '.[investing]' && playwright install chromium`). Set
INVESTING_AUTH_METHOD=endpoint instead to POST directly to a
reverse-engineered login endpoint (INVESTING_LOGIN_PAGE_URL /
INVESTING_LOGIN_POST_URL / INVESTING_LOGIN_FIELD_MAP — none of these are
defaulted, by design; see firm.data.investing.session).

This script is the explicit go/no-go gate for the whole Investing.com Pro
integration plan (docs/investing_pro_integration.md): if login is blocked
here (CAPTCHA/DataDome challenge, a selector that needs updating for the
current site layout, wrong endpoint field names, etc.), stop and reconsider
the approach before building anything on top.

This makes real authenticated requests against the live site — do not run
it in CI, and expect it to log in at most once per invocation (the session
persists its state to data/cache/investing_storage_state.json — or
investing_cookies.json for the endpoint method — and reuses it next time).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firm.config import get_settings
from firm.data.investing.session import InvestingDisabledError, InvestingSession
from firm.data.providers.base import ProviderError

OK = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"


def main() -> int:
    cfg = get_settings()

    if not getattr(cfg, "investing_scraper_enabled", False):
        print(
            f"{FAIL} INVESTING_SCRAPER_ENABLED is not set — nothing to test.\n"
            "  Set it (and INVESTING_EMAIL/PASSWORD) in .env once you've "
            "reviewed the risk in docs/investing_pro_integration.md."
        )
        return 1

    method = getattr(cfg, "investing_auth_method", "playwright")
    print(f"Auth method: {method}")

    with InvestingSession(settings=cfg) as session:
        print("Step 1/2: authenticate (or reuse a cached session)...")
        try:
            session.login()
        except InvestingDisabledError as exc:
            print(f"{FAIL} {exc}")
            return 1
        except ProviderError as exc:
            print(f"{FAIL} Login failed: {exc}")
            print(
                "  This is the go/no-go signal: if this is a CAPTCHA/DataDome "
                "challenge rather than a config mistake, stop here and revisit "
                "the approach (see docs/investing_pro_integration.md Phase 0)."
            )
            return 1
        state_file = (
            "data/cache/investing_storage_state.json" if method == "playwright"
            else "data/cache/investing_cookies.json"
        )
        print(f"{OK} Authenticated (session state saved to {state_file})")

        print("Step 2/2: fetch one authenticated page as a smoke check...")
        target = sys.argv[1] if len(sys.argv) > 1 else "https://www.investing.com/"
        try:
            resp = session.get(target)
        except ProviderError as exc:
            print(f"{FAIL} Authenticated fetch failed: {exc}")
            return 1
        print(f"{OK} Fetched {target} — HTTP {resp.status_code}, {len(resp.content)} bytes")
    print(f"\n{OK} Phase 0 spike passed — safe to proceed to Phase 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
