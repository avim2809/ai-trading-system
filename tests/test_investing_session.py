"""Tests for firm.data.investing.session.InvestingSession.

All HTTP/browser interaction is mocked (unittest.mock, matching the repo's
provider-test convention — no real network, no real Playwright/Chromium
install required to run these).
"""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from firm.config import Settings
from firm.data.investing.session import InvestingDisabledError, InvestingSession, PageResponse
from firm.data.providers.base import ProviderError


def _settings(**overrides):
    base = {
        "investing_scraper_enabled": True,
        # MagicMock auto-creates unset attributes rather than raising
        # AttributeError, so getattr(settings, key, default) never falls
        # back to `default` — every field login()/get() reads must be set
        # explicitly here, including the auth-method switch.
        "investing_auth_method": "endpoint",
        "investing_email": "me@example.com",
        "investing_password": "hunter2",
        "investing_login_page_url": "https://www.investing.com/login-page",
        "investing_login_post_url": "https://www.investing.com/login-post",
        "investing_login_field_map": "email:email_field,password:password_field",
        "request_timeout_seconds": 10,
        "max_retries": 3,
    }
    base.update(overrides)
    settings = MagicMock()
    for key, value in base.items():
        setattr(settings, key, value)
    return settings


def _session(tmp_path, **overrides):
    return InvestingSession(
        settings=_settings(**overrides),
        cookie_path=tmp_path / "cookies.json",
        storage_state_path=tmp_path / "storage_state.json",
        min_interval_sec=0.0,
    )


class TestDisabledGate:
    def test_login_raises_when_disabled(self, tmp_path):
        sess = _session(tmp_path, investing_scraper_enabled=False)
        with pytest.raises(InvestingDisabledError):
            sess.login()

    def test_get_raises_when_disabled(self, tmp_path):
        sess = _session(tmp_path, investing_scraper_enabled=False)
        with pytest.raises(InvestingDisabledError):
            sess.get("https://www.investing.com/")

    def test_disabled_never_touches_network(self, tmp_path):
        sess = _session(tmp_path, investing_scraper_enabled=False)
        with patch.object(sess._session, "get") as mock_get, \
             patch.object(sess._session, "post") as mock_post:
            with pytest.raises(InvestingDisabledError):
                sess.login()
            with pytest.raises(InvestingDisabledError):
                sess.get("https://www.investing.com/")
        mock_get.assert_not_called()
        mock_post.assert_not_called()


class TestAuthMethodDispatch:
    def test_default_settings_default_to_playwright(self):
        # Real (non-mock) Settings — confirms the actual production default,
        # not just what the test fixture happens to configure.
        assert Settings().investing_auth_method == "playwright"

    def test_unknown_auth_method_raises(self, tmp_path):
        sess = _session(tmp_path, investing_auth_method="carrier-pigeon")
        with pytest.raises(ProviderError, match="Unknown investing_auth_method"):
            sess.login()


class TestEndpointLogin:
    """The "endpoint" mechanism: plain requests.Session, no browser."""

    def test_login_raises_clear_error_when_endpoint_unconfigured(self, tmp_path):
        sess = _session(
            tmp_path,
            investing_login_page_url="",
            investing_login_post_url="",
            investing_login_field_map="",
        )
        with pytest.raises(ProviderError, match="not configured"):
            sess.login()

    def test_login_success_persists_cookies_and_authenticates(self, tmp_path):
        sess = _session(tmp_path)
        with patch.object(sess._session, "get") as mock_get, \
             patch.object(sess._session, "post") as mock_post:
            mock_get.return_value = MagicMock(ok=True, status_code=200)
            mock_post.return_value = MagicMock(ok=True, status_code=200)
            sess._session.cookies.set("sid", "abc123", domain="www.investing.com")
            sess.login()

        assert sess._authenticated is True
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["data"] == {"email_field": "me@example.com", "password_field": "hunter2"}
        saved = json.loads(sess.cookie_path.read_text())
        assert saved.get("sid") == "abc123"
        assert oct(sess.cookie_path.stat().st_mode)[-3:] == "600"

    def test_login_is_noop_when_already_authenticated(self, tmp_path):
        sess = _session(tmp_path)
        sess._authenticated = True
        with patch.object(sess._session, "post") as mock_post:
            sess.login()
        mock_post.assert_not_called()

    def test_login_rejected_raises_provider_error(self, tmp_path):
        sess = _session(tmp_path)
        with patch.object(sess._session, "get") as mock_get, \
             patch.object(sess._session, "post") as mock_post:
            mock_get.return_value = MagicMock(ok=True, status_code=200)
            mock_post.return_value = MagicMock(ok=False, status_code=403)
            with pytest.raises(ProviderError, match="rejected"):
                sess.login()
        assert sess._authenticated is False

    def test_cookies_reloaded_on_next_construction(self, tmp_path):
        sess = _session(tmp_path)
        with patch.object(sess._session, "get") as mock_get, \
             patch.object(sess._session, "post") as mock_post:
            mock_get.return_value = MagicMock(ok=True, status_code=200)
            mock_post.return_value = MagicMock(ok=True, status_code=200)
            sess._session.cookies.set("sid", "abc123", domain="www.investing.com")
            sess.login()

        sess2 = InvestingSession(
            settings=_settings(), cookie_path=sess.cookie_path,
            storage_state_path=tmp_path / "storage_state.json", min_interval_sec=0.0,
        )
        assert sess2._authenticated is True
        with patch.object(sess2._session, "post") as mock_post2:
            sess2.login()
        mock_post2.assert_not_called()


class TestEndpointGet:
    def test_get_reauthenticates_once_on_401(self, tmp_path):
        sess = _session(tmp_path)
        sess._authenticated = True  # skip the initial login() no-op path

        ok_resp = MagicMock(ok=True, status_code=200, text="<html>ok</html>")
        unauth_resp = MagicMock(ok=False, status_code=401)
        with patch.object(sess._session, "get", side_effect=[unauth_resp, ok_resp]) as mock_get, \
             patch.object(sess, "login") as mock_login:
            result = sess.get("https://www.investing.com/some-page")
        assert isinstance(result, PageResponse)
        assert result.status_code == 200
        assert mock_get.call_count == 2
        assert mock_login.call_count == 2

    @patch("firm.data.investing.session.time.sleep")
    def test_get_retries_transient_5xx_then_succeeds(self, mock_sleep, tmp_path):
        sess = _session(tmp_path)
        sess._authenticated = True

        bad = MagicMock(ok=False, status_code=503)
        good = MagicMock(ok=True, status_code=200, text="<html>ok</html>")
        with patch.object(sess._session, "get", side_effect=[bad, good]):
            result = sess.get("https://www.investing.com/some-page")
        assert result.status_code == 200

    def test_get_raises_provider_error_on_hard_failure(self, tmp_path):
        sess = _session(tmp_path)
        sess._authenticated = True
        bad = MagicMock(ok=False, status_code=404, text="not found")
        with patch.object(sess._session, "get", return_value=bad):
            with pytest.raises(ProviderError, match="404"):
                sess.get("https://www.investing.com/missing")

    @patch("firm.data.investing.session.time.sleep")
    def test_get_exhausts_retries_and_raises(self, mock_sleep, tmp_path):
        sess = _session(tmp_path)
        sess._authenticated = True
        bad = MagicMock(ok=False, status_code=503)
        with patch.object(sess._session, "get", return_value=bad):
            with pytest.raises(ProviderError, match="failed after"):
                sess.get("https://www.investing.com/some-page")


def _fake_playwright_module(context_factory):
    """Register a fake `playwright.sync_api` module in sys.modules so lazy
    `from playwright.sync_api import ...` calls resolve to test doubles
    instead of requiring the real (heavy, browser-bundled) package."""
    browser = MagicMock()
    browser.new_context.side_effect = context_factory

    chromium = MagicMock()
    chromium.launch.return_value = browser

    pw_obj = MagicMock()
    pw_obj.chromium = chromium
    pw_obj.stop = MagicMock()

    fake_module = types.ModuleType("playwright.sync_api")
    fake_module.sync_playwright = MagicMock(return_value=MagicMock(start=MagicMock(return_value=pw_obj)))

    class _FakeTimeoutError(Exception):
        pass

    fake_module.TimeoutError = _FakeTimeoutError
    return fake_module, browser, pw_obj


def _install_fake_playwright(monkeypatch, context):
    fake_module, browser, pw_obj = _fake_playwright_module(lambda **_: context)
    monkeypatch.setitem(sys.modules, "playwright", types.ModuleType("playwright"))
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_module)
    return browser, pw_obj, fake_module.TimeoutError


class TestPlaywrightLogin:
    def _page(self, *, email_ok=True, password_ok=True, submit_ok=True):
        page = MagicMock()
        page.get_by_text.side_effect = Exception("no login-trigger link on this page")

        def fill(selector, value, timeout=None):
            if "email" in selector and not email_ok:
                raise Exception("no such element")
            if "password" in selector and not password_ok:
                raise Exception("no such element")

        page.fill.side_effect = fill

        def locator(selector):
            mock_locator = MagicMock()
            if "submit" in selector and not submit_ok:
                mock_locator.first.click.side_effect = Exception("no such element")
            return mock_locator

        page.locator.side_effect = locator
        return page

    def _context(self, page):
        context = MagicMock()
        context.new_page.return_value = page
        return context

    def test_successful_login_persists_storage_state(self, tmp_path, monkeypatch):
        sess = _session(tmp_path, investing_auth_method="playwright")
        page = self._page()
        context = self._context(page)
        _install_fake_playwright(monkeypatch, context)

        sess.login()

        assert sess._authenticated is True
        context.storage_state.assert_called_once_with(path=str(sess.storage_state_path))

    def test_missing_email_field_raises_actionable_error(self, tmp_path, monkeypatch):
        sess = _session(tmp_path, investing_auth_method="playwright")
        context = self._context(self._page(email_ok=False))
        _install_fake_playwright(monkeypatch, context)

        with pytest.raises(ProviderError, match="email field"):
            sess.login()

    def test_missing_password_field_raises_actionable_error(self, tmp_path, monkeypatch):
        sess = _session(tmp_path, investing_auth_method="playwright")
        context = self._context(self._page(password_ok=False))
        _install_fake_playwright(monkeypatch, context)

        with pytest.raises(ProviderError, match="password field"):
            sess.login()

    def test_missing_submit_button_raises_actionable_error(self, tmp_path, monkeypatch):
        sess = _session(tmp_path, investing_auth_method="playwright")
        context = self._context(self._page(submit_ok=False))
        _install_fake_playwright(monkeypatch, context)

        with pytest.raises(ProviderError, match="submit button"):
            sess.login()

    def test_missing_credentials_raises_before_launching_browser(self, tmp_path, monkeypatch):
        sess = _session(tmp_path, investing_auth_method="playwright", investing_email="")
        context = self._context(self._page())
        browser, _pw_obj, _timeout_cls = _install_fake_playwright(monkeypatch, context)

        with pytest.raises(ProviderError, match="investing_email"):
            sess.login()
        browser.new_context.assert_not_called()

    def test_playwright_not_installed_raises_actionable_error(self, tmp_path, monkeypatch):
        sess = _session(tmp_path, investing_auth_method="playwright")
        # Simulate the optional dependency being absent: a sys.modules entry
        # of None makes the `from playwright.sync_api import ...` statement
        # raise ImportError, same as if the package were never installed.
        monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
        monkeypatch.setitem(sys.modules, "playwright", None)

        with pytest.raises(ProviderError, match="not installed"):
            sess.login()

    def test_timeout_raises_actionable_error(self, tmp_path, monkeypatch):
        sess = _session(tmp_path, investing_auth_method="playwright")
        page = self._page()
        context = self._context(page)
        browser, _pw_obj, timeout_cls = _install_fake_playwright(monkeypatch, context)
        page.goto.side_effect = timeout_cls("stuck on a challenge page")

        with pytest.raises(ProviderError, match="timed out"):
            sess.login()
        page.close.assert_called_once()


class TestPlaywrightGet:
    def _context_with_pages(self, pages):
        context = MagicMock()
        context.new_page.side_effect = pages
        return context

    def _page(self, status, content="<html>ok</html>"):
        page = MagicMock()
        page.goto.return_value = MagicMock(status=status)
        page.content.return_value = content
        return page

    def test_get_returns_page_response_on_success(self, tmp_path, monkeypatch):
        sess = _session(tmp_path, investing_auth_method="playwright")
        sess._authenticated = True  # skip login()
        page = self._page(200, content="<html>hello</html>")
        context = self._context_with_pages([page])
        _install_fake_playwright(monkeypatch, context)

        result = sess.get("https://www.investing.com/economic-calendar/")

        assert isinstance(result, PageResponse)
        assert result.status_code == 200
        assert result.text == "<html>hello</html>"
        assert result.ok is True
        page.close.assert_called_once()

    def test_get_reauthenticates_once_on_403(self, tmp_path, monkeypatch):
        sess = _session(tmp_path, investing_auth_method="playwright")
        sess._authenticated = True
        blocked_page = self._page(403)
        ok_page = self._page(200)
        context = self._context_with_pages([blocked_page, ok_page])
        _install_fake_playwright(monkeypatch, context)

        with patch.object(sess, "login") as mock_login:
            result = sess.get("https://www.investing.com/some-page")

        assert result.status_code == 200
        # One no-op call at the top of get() (already authenticated) plus
        # one forced re-login inside _playwright_get after the 403.
        assert mock_login.call_count == 2

    @patch("firm.data.investing.session.time.sleep")
    def test_get_retries_transient_5xx_then_succeeds(self, mock_sleep, tmp_path, monkeypatch):
        sess = _session(tmp_path, investing_auth_method="playwright")
        sess._authenticated = True
        bad_page = self._page(503)
        good_page = self._page(200)
        context = self._context_with_pages([bad_page, good_page])
        _install_fake_playwright(monkeypatch, context)

        result = sess.get("https://www.investing.com/some-page")
        assert result.status_code == 200
        bad_page.close.assert_called_once()

    def test_get_raises_provider_error_on_hard_failure(self, tmp_path, monkeypatch):
        sess = _session(tmp_path, investing_auth_method="playwright")
        sess._authenticated = True
        page = self._page(404, content="not found")
        context = self._context_with_pages([page])
        _install_fake_playwright(monkeypatch, context)

        with pytest.raises(ProviderError, match="404"):
            sess.get("https://www.investing.com/missing")

    @patch("firm.data.investing.session.time.sleep")
    def test_get_exhausts_retries_and_raises(self, mock_sleep, tmp_path, monkeypatch):
        sess = _session(tmp_path, investing_auth_method="playwright")
        sess._authenticated = True
        pages = [self._page(503) for _ in range(3)]
        context = self._context_with_pages(pages)
        _install_fake_playwright(monkeypatch, context)

        with pytest.raises(ProviderError, match="failed after"):
            sess.get("https://www.investing.com/some-page")


class TestClose:
    def test_close_saves_storage_state_and_stops_browser(self, tmp_path, monkeypatch):
        sess = _session(tmp_path, investing_auth_method="playwright")
        sess._authenticated = True
        context = self._context_helper()
        browser, pw_obj, _ = _install_fake_playwright(monkeypatch, context)
        sess._ensure_browser()

        sess.close()

        context.storage_state.assert_called_once()
        browser.close.assert_called_once()
        pw_obj.stop.assert_called_once()
        assert sess._context is None

    def _context_helper(self):
        return MagicMock()

    def test_context_manager_closes_on_exit(self, tmp_path, monkeypatch):
        context = MagicMock()
        with _session(tmp_path, investing_auth_method="playwright") as sess:
            browser, pw_obj, _ = _install_fake_playwright(monkeypatch, context)
            sess._ensure_browser()
        browser.close.assert_called_once()

    def test_close_is_safe_when_browser_never_launched(self, tmp_path):
        sess = _session(tmp_path, investing_auth_method="endpoint")
        sess.close()  # must not raise
