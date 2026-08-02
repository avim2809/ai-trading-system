"""Danelfin AI stock-scoring adapter — genuine REST API, not a scraper.

Verified live 2026-07-31 against the real (trial) API key. Official docs
(danelfin.com/docs/api, provided by the user — the docs *website* sits
behind Cloudflare but the API subdomain does not):

  Base URL:  https://apirest.danelfin.com
  Auth:      header ``x-api-key: <key>`` on every request.
  Plans:     Free (500 calls/mo, 10/min), Basic (2,500/mo, 60/min),
             Expert (10,000/mo, 120/min), Max (50,000/mo, 180/min).

  ``GET /ranking?ticker=<SYMBOL>`` — the only endpoint with genuine
    historical depth (real per-date scores back to ~2016-12, matching the
    advertised "since 2017"). Documented query params: ``ticker``, ``date``,
    ``aiscore``/``*_min`` filters, ``sector``, ``industry``, ``asset``,
    ``fields``, ``market``. Response: JSON object keyed by date string,
    each value ``{"aiscore": int, "technical": int, "fundamental":
    int|None, "sentiment": int, "low_risk": int}`` (1-10 scale;
    ``fundamental`` can be null on some dates — Danelfin's own data gap).
    **Not officially documented**: passing ``page=<N>`` (verified live,
    N=1..25+) walks further back in time, ~100 trading days per page — this
    is how :meth:`get_ai_scores` reaches multi-year history since the
    documented params don't include a date-range/pagination parameter.

  ``/v3/*`` endpoints — always the **latest snapshot only, no historical
    dates** (per the docs) — not usable for backtesting, only as live
    signals: ``/v3/beststocks`` (top-25 AI-curated picks — the closest
    thing to a "ProPicks" equivalent), ``/v3/trading-parameters`` (entry/
    stop-loss/take-profit levels + buy/hold/sell signal per ticker),
    ``/v3/price-forecast`` (probabilistic return distribution by horizon),
    ``/v3/performance`` (historical win-rate/alpha track record by signal),
    ``/v3/trade-ideas`` (filterable screener). Exposed here as read-only
    fetchers for live use, feeding :class:`firm.strategies.danelfin_live_signals
    .DanelfinLiveSignalsStrategy` — **deliberately not wired into any
    risk/execution logic** (e.g. ``trading-parameters``' stop-loss/
    take-profit levels could inform ``RiskAgent``/``ExecutionAgent``, but
    that's a live-risk-relevant behavioral change that needs its own
    explicit review, not something to fold in silently alongside a new
    alpha signal).

    Field names verified live 2026-07-31 against real AAPL responses (not
    just guessed from the docs page, which is Cloudflare-blocked):
    ``trading-parameters`` → ``{entry_price, stop_loss, stop_loss_pct,
    take_profit, take_profit_pct, horizon, currency, signal}`` where
    ``stop_loss_pct``/``take_profit_pct`` are **percentage points** (e.g.
    ``-5.29`` == -5.29%, not a 0-1 decimal); ``price-forecast`` →
    ``{signal, median_3m, q05_3m, q16_3m, q84_3m, q95_3m, take_profit_3m,
    stop_loss_3m}`` where these ARE 0-1 decimals (e.g. ``0.064`` == +6.4%)
    — a real unit mismatch between the two endpoints, not a typo;
    ``performance`` → ``{signal, win_rate_1m/3m/6m/1y, alpha_win_rate_*,
    avg_perf_*, avg_alpha_*}``.

Account is on the Expert plan (10,000 calls/mo, 120/min, confirmed by the
user) — pacing below (1s/request) stays well under that with a safety
margin; an early rate-limit hit during initial testing was very likely
before the plan/key had fully propagated, not a real 10/min Free-tier cap.
"""

from __future__ import annotations

import time
from typing import Sequence

import pandas as pd

from firm.config import Settings, get_settings
from firm.data.providers._rest import RestClient
from firm.data.providers.base import DataProvider, ProviderError
from firm.data.schemas import AI_SCORE_COLS, LIVE_SIGNAL_COLS
from firm.logging_setup import get_logger

log = get_logger(__name__)

_BASE_URL = "https://apirest.danelfin.com"
_MAX_PAGES = 30  # ~30 * 100 trading days ≈ 12 years — comfortably covers "since 2017"
# Expert plan = 120 calls/min (0.5s/call minimum); 1s/request keeps a 2x
# safety margin without making multi-symbol historical fetches too slow.
_REQUEST_PAUSE_SECONDS = 1.0


class DanelfinProvider(DataProvider):
    """Adapter for Danelfin's AI Score API (AI/Fundamental/Technical/Sentiment/Low-Risk scores)."""

    name = "danelfin"

    def __init__(self, api_key: str = "", settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._api_key = api_key or self.settings.require("danelfin_api_key")
        self._client = RestClient(_BASE_URL, self.settings)

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key}

    def _get(self, path: str, params: dict) -> dict:
        data = self._client.get_json(path, params=params, headers=self._headers())
        if not isinstance(data, dict):
            raise ProviderError(f"Danelfin: unexpected response type for {path} {params}")
        return data

    def _get_with_retry(self, path: str, params: dict) -> dict:
        """A response of exactly ``{"message": "..."}`` (seen live — an
        ordinary HTTP 200, not a documented rate-limit/error status) is
        retried once after a full pause, then returned as-is (caller decides
        whether an empty/message-only result is fatal for its use case)."""
        data = self._get(path, params)
        if len(data) == 1 and "message" in data:
            time.sleep(_REQUEST_PAUSE_SECONDS)
            data = self._get(path, params)
        return data

    # ------------------------------------------------------------------
    # /ranking — the only endpoint with genuine historical depth
    # ------------------------------------------------------------------

    def _fetch_symbol_history(self, ticker: str, start_ts: pd.Timestamp) -> list[dict]:
        """Paginate back through /ranking until reaching start_ts or _MAX_PAGES."""
        rows: list[dict] = []
        earliest_seen: pd.Timestamp | None = None
        for page in range(1, _MAX_PAGES + 1):
            if page > 1:
                time.sleep(_REQUEST_PAUSE_SECONDS)
            data = self._get_with_retry("/ranking", {"ticker": ticker, "page": page})
            if not data or (len(data) == 1 and "message" in data):
                log.warning(
                    "danelfin_page_skipped ticker=%s page=%d reason=%s",
                    ticker, page, data.get("message", "empty") if data else "empty",
                )
                continue
            page_min = min(pd.Timestamp(d) for d in data)
            if earliest_seen is not None and page_min >= earliest_seen:
                # Pagination stopped making progress (e.g. hit the provider's
                # own history boundary and it's repeating) — stop *before*
                # appending this page's (already-seen) rows, so a stalled
                # page never adds duplicates on top of what an earlier page
                # already contributed, rather than looping to _MAX_PAGES.
                break
            earliest_seen = page_min
            for date_str, scores in data.items():
                rows.append(
                    {
                        "date": date_str,
                        "symbol": ticker,
                        "ai_score": scores.get("aiscore"),
                        "fundamental_score": scores.get("fundamental"),
                        "technical_score": scores.get("technical"),
                        "sentiment_score": scores.get("sentiment"),
                        "low_risk_score": scores.get("low_risk"),
                    }
                )
            if page_min <= start_ts:
                break
        return rows

    def get_ai_scores(
        self, symbols: Sequence[str], start: str, end: str
    ) -> pd.DataFrame:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            try:
                rows = self._fetch_symbol_history(symbol, start_ts)
            except ProviderError as exc:
                if "402" in str(exc) or "403" in str(exc):
                    log.warning("danelfin_ai_scores_unavailable symbol=%s (%s)", symbol, exc)
                else:
                    log.warning("danelfin_ai_scores_failed symbol=%s (%s)", symbol, exc)
                continue
            except Exception:
                log.exception("danelfin_ai_scores_failed symbol=%s", symbol)
                continue
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=AI_SCORE_COLS)
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
            if not df.empty:
                frames.append(df)
        if not frames:
            return self.empty_ai_scores()
        merged = pd.concat(frames, ignore_index=True)
        # Defensive: a pagination boundary can legitimately repeat a date
        # across two consecutive pages (inclusive ranges) — dedup rather
        # than let a boundary overlap silently double-count a row.
        return (
            merged.sort_values(["symbol", "date"])
            .drop_duplicates(subset=["symbol", "date"], keep="last")
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # /v3/* — latest-snapshot-only live signals (no historical dates).
    # Read-only fetchers; NOT wired into any strategy/risk/execution logic
    # here — see the module docstring.
    # ------------------------------------------------------------------

    def get_best_stocks(self) -> pd.DataFrame:
        """Latest Top-25 AI-curated picks (``/v3/beststocks``) — Danelfin's
        closest equivalent to Investing.com Pro's "ProPicks". No historical
        dates; a fresh call always reflects "today"."""
        data = self._get_with_retry("/v3/beststocks", {})
        if not data:
            return pd.DataFrame(columns=[
                "date", "symbol", "rank", "ai_score", "ai_score_change",
                "fundamental_score", "technical_score", "sentiment_score",
                "low_risk_score", "perf_ytd", "sector", "country",
            ])
        rows = []
        for date_str, tickers in data.items():
            for symbol, info in tickers.items():
                rows.append({
                    "date": date_str,
                    "symbol": symbol,
                    "rank": info.get("rank"),
                    "ai_score": info.get("aiscore"),
                    "ai_score_change": info.get("aiscore_change"),
                    "fundamental_score": info.get("fundamental"),
                    "technical_score": info.get("technical"),
                    "sentiment_score": info.get("sentiment"),
                    "low_risk_score": info.get("low_risk"),
                    "perf_ytd": info.get("perf_ytd"),
                    "sector": info.get("sector"),
                    "country": info.get("country"),
                })
        return pd.DataFrame(rows)

    def get_trading_parameters(self, ticker: str) -> dict | None:
        """Latest suggested entry/stop-loss/take-profit levels + buy/hold/
        sell signal (``/v3/trading-parameters``) for a single ticker."""
        data = self._get_with_retry("/v3/trading-parameters", {"ticker": ticker})
        if not data or "message" in data:
            return None
        return next(iter(data.values()), None)

    def get_price_forecast(self, ticker: str, horizon: str = "3m") -> dict | None:
        """Latest probabilistic return-distribution forecast
        (``/v3/price-forecast``) for a single ticker/horizon
        (``1m``/``3m``/``6m``/``1y``). Values are decimal returns
        (e.g. 0.132 == +13.2%)."""
        data = self._get_with_retry(
            "/v3/price-forecast", {"ticker": ticker, "horizon": horizon},
        )
        if not data or "message" in data:
            return None
        return next(iter(data.values()), None)

    def get_performance(self, ticker: str, signal: str = "buy") -> dict | None:
        """Historical win-rate/alpha track record (``/v3/performance``) for
        a ticker's buy or sell signal — a meta confidence-weighting input,
        not a per-symbol timing signal."""
        data = self._get_with_retry("/v3/performance", {"ticker": ticker, "signal": signal})
        if not data or "message" in data:
            return None
        return next(iter(data.values()), None)

    def get_trade_ideas(self, **filters) -> pd.DataFrame:
        """Filterable screener (``/v3/trade-ideas``) — one row per symbol,
        columns include ``sector``/``industry``, ``aiscore``/``low_risk``/
        ``fundamental``/``technical``/``sentiment``, ``average_volume_3m``,
        and ``win_rate_*``/``alpha_win_rate_*``/``avg_perf_*``/``avg_alpha_*``
        (1m/3m/6m/1y each).

        Verified live 2026-07-31 (this method's earlier implementation
        assumed a ``{"items": [...]}`` response shape, which was wrong and
        silently always returned empty — the real shape is
        ``{date_str: {symbol: {...}}}``, identical to ``/ranking`` and
        ``get_best_stocks``). Confirmed real, working filter params:
        ``aiscore=N`` and ``low_risk=N`` are **minimum-threshold** filters
        (not exact-match, despite the singular name), ``average_volume_3m=N``
        is likewise a minimum, ``sector=<kebab-case>`` (e.g.
        ``"information-technology"``, ``"energy"``, ``"health-care"``,
        confirmed live), and ``limit`` is capped at 100 (HTTP 400 above
        that) with **no pagination param** (``page`` is rejected as unknown,
        unlike ``/ranking``) — so a single call can return at most 100
        symbols; a broader sweep needs one call per ``sector`` filter.
        There is no ``signal``/buy-sell filter param (confirmed rejected as
        unknown) — every sampled trade-idea symbol's own
        ``/v3/trading-parameters`` call returned ``signal: "buy"`` in
        spot-checks, consistent with this endpoint's purpose (a "trade
        idea" is inherently a buy call), but that hasn't been exhaustively
        verified for every result.

        The top-level response also carries sibling ``total``/``limit``/
        ``offset`` int keys alongside the date key (confirmed live) —
        unlike ``/ranking``/``get_best_stocks``, which only ever have date
        keys. These must be skipped, not iterated as if they were dates.
        """
        data = self._get_with_retry("/v3/trade-ideas", filters)
        if not data or (len(data) == 1 and "message" in data):
            return pd.DataFrame()
        rows = []
        for date_str, symbols in data.items():
            if not isinstance(symbols, dict):
                continue  # skip sibling total/limit/offset metadata keys
            for symbol, info in symbols.items():
                rows.append({"date": date_str, "symbol": symbol, **info})
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def get_historical_sector_scores(
        self,
        sector: str,
        date: str,
        low_risk_values: Sequence[int] = (5, 6, 7, 8, 9, 10),
    ) -> pd.DataFrame:
        """Bulk **historical** ``/ranking`` scan: every symbol in *sector*
        on *date* (a specific historical date, not "latest") — genuinely
        historical, unlike ``/v3/trade-ideas``. Discovered/verified live
        2026-07-31 while investigating whether the Best-Stocks methodology
        could be backtested at all (it initially looked snapshot-only,
        because ``/v3/trade-ideas`` is — this bulk mode of ``/ranking``
        itself is not).

        Three real, non-obvious things confirmed live before writing this:

        1. Passing ``date`` + ``sector`` (+ optionally ``low_risk``) with
           **no** ``ticker`` returns every matching symbol for that date,
           not one ticker's timeline — e.g.
           ``/ranking?date=2024-06-03&sector=information-technology&low_risk=5``
           returned 100 real symbols (AMD, APPN, ARM, ...).
        2. Unlike ``/v3/trade-ideas``'s minimum-threshold filters, this
           bulk mode's ``low_risk``/``aiscore`` filters are **exact
           match** (confirmed: a ``low_risk=5`` query returns ONLY
           low_risk-exactly-5 rows, not >=5) — so "low_risk >= 5" requires
           one call per exact value 5..10 and a local union, which is what
           *low_risk_values* is for.
        3. A single call still caps at 100 rows (confirmed: exactly 100
           returned with no low_risk filter at all) — but (unlike
           ``/v3/trade-ideas``) ``page=N`` **does** paginate past that cap
           here (confirmed: page=2 returned 86 more, non-overlapping,
           symbols) — the opposite pagination-support split from
           ``/v3/trade-ideas``, so don't assume the two endpoints behave
           the same way.

        A 404 here means **zero rows match this exact (sector, date,
        low_risk) combination** — confirmed live: querying
        information-technology/2024-06-03 across low_risk 5..10
        individually returned real non-empty results for 5, 6, 7, 9 and a
        404 for 8 and 10 (simply no stocks had exactly that score that
        day), not "invalid date". An earlier version of this docstring
        wrongly assumed 404 meant "no data for this date at all" (based on
        a single observation at a market holiday, 2018-01-15, which
        happened to coincide) and built a whole date-revalidation step in
        the backtest script around that wrong assumption — it silently
        rejected perfectly valid trading dates whenever a narrow probe
        query (e.g. a single high low_risk value) legitimately had zero
        matches. Fixed: this method now treats a 404 exactly like an empty
        result for that one low_risk value (no rows added, move to the
        next value) rather than propagating it as an error.

        There is NO equivalent of ``/v3/trading-parameters``' buy/hold/sell
        "signal" anywhere in ``/ranking`` — Danelfin's "Proven Buy Signal"
        criterion has no historical depth at all and cannot be replicated
        here; only the low_risk/aiscore/sector data can be reconstructed
        historically. Document this gap wherever this method feeds a
        historical Best-Stocks reconstruction — it is a real methodology
        difference from the live arm, not an oversight.
        """
        rows: list[dict] = []
        first_call = True
        for low_risk in low_risk_values:
            page = 1
            while True:
                if not first_call:
                    time.sleep(_REQUEST_PAUSE_SECONDS)
                first_call = False
                params: dict = {"date": date, "sector": sector, "low_risk": low_risk}
                if page > 1:
                    params["page"] = page
                try:
                    data = self._get_with_retry("/ranking", params)
                except ProviderError as exc:
                    if "404" in str(exc):
                        log.debug(
                            "danelfin_historical_sector_scores_empty sector=%s date=%s low_risk=%d page=%d "
                            "(zero rows match this exact combination — not an error)",
                            sector, date, low_risk, page,
                        )
                    else:
                        log.warning(
                            "danelfin_historical_sector_scan_failed sector=%s date=%s low_risk=%d page=%d (%s)",
                            sector, date, low_risk, page, exc,
                        )
                    break
                inner = data.get(date, {}) if isinstance(data, dict) else {}
                if not inner:
                    break
                for symbol, scores in inner.items():
                    rows.append({"symbol": symbol, "sector": sector, "date": date, **scores})
                if len(inner) < 100:
                    break
                page += 1
        if not rows:
            return pd.DataFrame()
        return (
            pd.DataFrame(rows)
            .drop_duplicates(subset=["symbol"])  # a symbol's low_risk is fixed on a given historical date
            .reset_index(drop=True)
        )

    def get_live_signals(self, symbols: Sequence[str]) -> pd.DataFrame:
        """Combine trading-parameters + price-forecast + performance into
        one row per symbol (LIVE_SIGNAL_COLS) — the actual "feed this into
        the analysts" capability requested by the user, distinct from
        get_ai_scores (which has genuine history and can be backtested).
        This one cannot: per Danelfin's own docs, /v3/* always reflects
        "right now" with no historical dates, so this is live-only by
        construction — a strategy reading firm.strategies.base.PitView
        .live_signals() will always see an empty frame in backtests (no
        cache-backed history exists to populate one, ever) and only ever
        sees real data in live cycles.

        3 calls per symbol (trading-parameters, price-forecast,
        performance) at the same ~1s/request pacing as get_ai_scores —
        for a 25-symbol universe that's ~75 calls, well inside the Expert
        plan's 10,000/month, 120/min limits, run once per live cycle
        (once/day).
        """
        today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
        rows: list[dict] = []
        for symbol in symbols:
            try:
                tp = self.get_trading_parameters(symbol)
            except ProviderError:
                tp = None
            except Exception:
                log.exception("danelfin_trading_parameters_failed symbol=%s", symbol)
                tp = None
            try:
                pf = self.get_price_forecast(symbol, horizon="3m")
            except ProviderError:
                pf = None
            except Exception:
                log.exception("danelfin_price_forecast_failed symbol=%s", symbol)
                pf = None
            # Query the track record for whichever signal trading-parameters
            # actually recommends (verified live: "buy"/"sell"), not always
            # "buy" — the /v3/performance win-rate is meaningless as a
            # confidence measure for a "sell" call if it's the buy signal's
            # own historical track record.
            perf_signal = (tp or {}).get("signal")
            if perf_signal not in ("buy", "sell"):
                # get_performance only documents buy/sell tracks; a "hold"
                # (or missing) trading-parameters signal falls back to buy's
                # track record rather than guessing at an unsupported value.
                perf_signal = "buy"
            try:
                perf = self.get_performance(symbol, signal=perf_signal)
            except ProviderError:
                perf = None
            except Exception:
                log.exception("danelfin_performance_failed symbol=%s", symbol)
                perf = None

            if tp is None and pf is None and perf is None:
                continue
            tp = tp or {}
            pf = pf or {}
            perf = perf or {}
            rows.append({
                "date": today,
                "symbol": symbol,
                "tp_signal": tp.get("signal"),
                "tp_entry_price": tp.get("entry_price"),
                "tp_stop_loss_pct": tp.get("stop_loss_pct"),
                "tp_take_profit_pct": tp.get("take_profit_pct"),
                "pf_median_return_3m": pf.get("median_3m"),
                "pf_q05_return_3m": pf.get("q05_3m"),
                "pf_q95_return_3m": pf.get("q95_3m"),
                "perf_win_rate_3m": perf.get("win_rate_3m"),
                "perf_alpha_win_rate_3m": perf.get("alpha_win_rate_3m"),
                "perf_win_rate_1m": perf.get("win_rate_1m"),
                "perf_win_rate_6m": perf.get("win_rate_6m"),
                "perf_win_rate_1y": perf.get("win_rate_1y"),
                "perf_avg_alpha_3m": perf.get("avg_alpha_3m"),
            })
        if not rows:
            return self.empty_live_signals()
        return pd.DataFrame(rows, columns=LIVE_SIGNAL_COLS)

    # ------------------------------------------------------------------
    # Unsupported DataProvider capabilities — Danelfin is a single-purpose
    # AI-score provider, not a general market-data source.
    # ------------------------------------------------------------------

    def get_prices(self, symbols: Sequence[str], start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError("DanelfinProvider does not provide prices.")

    def get_fundamentals(self, symbols: Sequence[str], start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError("DanelfinProvider does not provide fundamentals; use FMP.")

    def get_news_sentiment(self, symbols: Sequence[str], start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError("DanelfinProvider does not provide news sentiment; use Massive.")

    def get_corporate_actions(self, symbols: Sequence[str], start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError("DanelfinProvider does not provide corporate actions; use Massive.")

    def get_universe_constituents(self, index: str, date: str = "") -> list[str]:
        raise NotImplementedError("DanelfinProvider does not supply index constituents; use FMP.")

    def get_analyst_ratings(self, symbols: Sequence[str], start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError("DanelfinProvider does not provide analyst ratings; use FMP.")
