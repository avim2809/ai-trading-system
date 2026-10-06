#!/usr/bin/env python
"""Measure what a ``pattern_recognition`` confirmation is *actually* worth to
trade, at an executable fill price, on the real cached panel.

WHY THIS SCRIPT EXISTS
======================
``firm.patterns.confirmation.find_confirmation`` documents itself as finding
the bar where price *crossed* a level, but implements ``close[i] > level`` --
an "is beyond" test with no edge condition -- searching newest-first. It
therefore returns *today* whenever price is still past the level, and every
rule module then sets ``entry = level_at(confirm_index)``. So ``match.entry``
is a price the market has already traded through by the time the signal is
emitted: it is not a fill anyone can get.

Every existing measurement of this strategy's per-trade edge
(``firm.patterns.ml.labeling.label_triple_barrier_with_exit`` and everything
built on it) prices the trade *at* ``match.entry`` and denominates risk as
``|entry - stop|``. If ``entry`` is unreachable, that number is a fiction,
and so is the ``min_risk_reward: 1.5`` gate that reads ``match.risk_reward``.

This script prices the *same* confirmations five ways so the size of that gap
is measured rather than asserted, and reports symbol-clustered confidence
intervals so the reader can tell signal from noise.

THE FIVE PRICINGS
=================
1. ``nominal``   -- fill at ``match.entry``, risk ``|entry - stop|``. This is
   exactly what ``label_triple_barrier_with_exit`` measures. Reported *only*
   as the baseline to compare against; it is not achievable.
2. ``executable`` -- fill at the **next bar's open** after the bar the signal
   would be emitted on, risk ``|fill - stop|``. This is the headline number.
3. ``placebo``   -- identical to (2) but with the direction flipped and the
   barriers mirrored about the fill, so risk is identical and only the sign
   of the thesis changes. A control: if the inverse of the strategy beats the
   strategy, no amount of execution engineering rescues it.
4. ``resting_limit`` -- a limit order resting *at* the level, good for
   ``--limit-good-bars`` bars. The "buy the retest" variant: it gets the
   nominal entry price, but only on the subset of events where price comes
   back to it, which is an adversely-selected subset.
5. ``edge_triggered`` -- the fix Workstream C proposes, measured in advance:
   locate the *true first crossing* of the level after the pattern's last
   structural pivot, and fill at the next open after that. This is what
   ``entry`` would mean if ``find_confirmation`` matched its own docstring.

DESIGN CHOICES, AND WHY
=======================
**Next-open fill.** ``firm.backtest.engine`` submits orders in ``next()`` and
fills them at the following bar's open (``slip_open=True``,
``engine.py:75``). A daily scan that sees a confirmation on bar ``c`` can only
act on bar ``c+1``'s open. We deliberately use ``confirm_index + 1`` rather
than ``cutoff`` (the rolling-scan cutoff, which can be up to ``step_bars``
later): a *live* scanner runs every day and would have caught it on bar ``c``.
That is the assumption most favourable to the strategy, chosen on purpose --
a negative result under a favourable assumption is a strong result.

**Risk denominated as ``|fill - stop|``.** The stop is a structural price
level; it does not move because you got a worse entry. Denominating in
``|entry - stop|`` while filling somewhere else silently shrinks every loss
and inflates every win. Events where the fill is already at or through the
stop have no defined R at all; they are counted and excluded, and a
sensitivity line reports what happens if they are instead booked at -1R.

**Label against the FULL series, features/signals from the as-of window.**
``build_dataset`` in ``scripts/train_pattern_ml.py:233-237`` does this and it
is the single easiest thing to get wrong here: if barriers are walked only
against the truncated as-of ``window``, ~92% of events "time out" purely
because the data ends (documented at ``docs/pattern_recognition_plan.md``
§6.8). Signals and features come from ``window``; every barrier walk here
indexes the full per-symbol array.

**Symbol-block bootstrap, not i.i.d. resampling.** These events are heavily
overlapping in time (a rolling cutoff re-finds the same breakout) and
cross-sectionally clustered (25 mega-caps in one market move together). An
i.i.d. trade-level bootstrap -- which is what
``firm.eval.robustness.MonteCarloAnalyzer`` does (``eval/robustness.py:51-72``)
-- treats each of ~1,800 trades as independent evidence and returns falsely
tight intervals. We resample **whole symbols** with replacement instead, which
is the same clustering unit ``firm.patterns.ml.sample_weights`` and
``firm.patterns.ml.purged_cv`` already use for training. With 25 symbols the
resulting intervals are wide; that width is the honest answer, not a defect.
A naive i.i.d. t-statistic is printed alongside purely so the reader can see
how much of the apparent significance came from the wrong clustering
assumption.

**Costs at live-accurate rates.** ``config/live.yaml:471-473``:
``commission_pct 0.0005``, ``slippage_pct 0.0005``, ``spread_pct 0.0002`` ->
12 bps per side, 24 bps round trip. ``config/settings.yaml`` carries a stale
10 bps commission (34 bps round trip) and is deliberately not read here.
Gross and net-of-cost figures are both reported for every pricing.

**Dedup convention.** One event per ``(symbol, confirm_index)``, first
discovery wins -- identical to ``build_dataset``'s ``seen_confirm_indices``,
so the event count and triple-barrier label balance are directly comparable
to the real training run (the ``--expect-*`` sanity check below).

WHAT IS APPROXIMATED
====================
``level_at(i)`` (the sloped neckline/trendline function) lives inside each
rule module and is not carried on ``PatternMatch``. The ``edge_triggered`` and
``resting_limit`` pricings therefore use the *constant* level
``match.entry`` (= ``level_at(confirm_index)``) as a stand-in. This is exact
for horizontal levels (necklines, rectangles, double tops/bottoms) and
approximate for sloped ones (triangles, wedges, flags); the report breaks the
result out by pattern family so the reader can see whether the conclusion
depends on the approximated subset.

USAGE
=====
    python scripts/measure_pattern_executable_expectancy.py \
        --start 2010-01-01 --end 2026-09-21 \
        --json-out /tmp/pattern_exec_full.json

    # Comparison slice matching the ad-hoc script this is reproducing
    python scripts/measure_pattern_executable_expectancy.py \
        --start 2023-01-01 --json-out /tmp/pattern_exec_2023.json

Reads only local disk (``data/cache`` via ``firm.runtime.load_prices``) and
``data/models/*.onnx``. Writes nothing except its optional ``--json-out``.
Touches no config and no production code.

RUN IT FROM A PINNED WORKTREE
=============================
The measured numbers are a property of the *detector code*, not just the data.
This was learned the hard way on 2026-09-27: a first full-panel run of this
script silently measured an in-flight, uncommitted edge-triggered rewrite of
``firm.patterns.confirmation.find_confirmation`` that a parallel workstream
had just written into the shared checkout, and the event count came back at
970 instead of the expected 1,849. The result was correct *for that code* and
completely wrong as a measurement of production.

So: ``git worktree add /tmp/pin <commit>``, symlink ``data/`` into it, and run
there. This script prints the resolved git commit and flags any dirty file
under ``src/firm/patterns`` at startup, but a printed warning is not a
substitute for pinning.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from firm.eval.metrics import compute_trade_metrics
from firm.patterns.match import PatternMatch
from firm.patterns.ml.feature_engineering import build_features

# `first_barrier_hit` (public) returns only the label; every executable
# pricing here also needs the bar the barrier was touched on, to mark the
# trade out at a price. `_first_barrier_hit_with_offset` is that same walk
# with the offset retained -- imported directly, rather than re-implemented,
# so there stays exactly ONE copy of the same-bar-collision ("stop wins") and
# timeout semantics in the codebase.
from firm.patterns.ml.labeling import (
    _first_barrier_hit_with_offset,
    label_triple_barrier_with_exit,
)
from firm.patterns.ml.xgb_inference import score_pattern_meta_confirmation
from firm.patterns.scanner import scan_symbol
from firm.strategies.pattern_recognition import _adjusted_ohlc

log = logging.getLogger("measure_pattern_executable_expectancy")

# config/live.yaml:471-473. NOT config/settings.yaml (stale 10bps commission).
COMMISSION_PCT = 0.0005
SLIPPAGE_PCT = 0.0005
SPREAD_PCT = 0.0002
COST_PER_SIDE = COMMISSION_PCT + SLIPPAGE_PCT + SPREAD_PCT  # 12 bps
COST_ROUND_TRIP = 2 * COST_PER_SIDE  # 24 bps

# config/live.yaml `universe.symbols` (the real live universe), read at
# runtime by default so this script cannot drift from the config.
_FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "AVGO", "AMD",
    "CRM", "NFLX", "ADBE", "JPM", "GS", "BAC", "V", "MA", "JNJ", "UNH",
    "LLY", "XOM", "CVX", "SPY", "QQQ", "IWM",
]

# firm.strategies.pattern_recognition.PatternRecognitionStrategy.default_params
SCAN_DEFAULTS = {
    "zigzag_pct": 0.03,
    "min_score": 60.0,
    "min_risk_reward": 1.5,
    "confirm_lookback_bars": 3,
    "stop_atr_floor": 1.5,
}


# ---------------------------------------------------------------------------
# Panel loading
# ---------------------------------------------------------------------------

def code_provenance() -> dict[str, Any]:
    """Which *detector code* produced these numbers.

    Every figure this script prints depends on ``firm.patterns``' behaviour,
    not only on the cached prices, so the commit and the cleanliness of that
    package are part of the result and are recorded with it. A dirty file
    under ``src/firm/patterns`` means the run is measuring something that
    does not exist in any commit -- see the module docstring's account of the
    2026-09-27 incident where that silently happened.
    """
    import subprocess

    root = Path(__file__).resolve().parents[1]

    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=30,
        ).stdout.strip()

    try:
        head = _git("rev-parse", "--short", "HEAD")
        dirty = [ln[3:] for ln in _git("status", "--porcelain", "--", "src/firm/patterns").splitlines()]
    except Exception:
        log.exception("could not resolve git provenance")
        return {"head": None, "dirty_pattern_files": []}
    return {"head": head, "dirty_pattern_files": dirty}


def live_universe() -> list[str]:
    """``universe.symbols`` from ``config/live.yaml``; falls back to a frozen
    copy (logged loudly) only if the file can't be read, so a config typo
    surfaces rather than silently changing the measured population.
    """
    path = Path(__file__).resolve().parents[1] / "config" / "live.yaml"
    try:
        import yaml

        with open(path) as fh:
            cfg = yaml.safe_load(fh)
        symbols = cfg["universe"]["symbols"]
        if not symbols:
            raise ValueError("config/live.yaml universe.symbols is empty")
        return [str(s).strip().upper() for s in symbols]
    except Exception:
        log.exception("could not read universe.symbols from %s -- using frozen fallback", path)
        return list(_FALLBACK_UNIVERSE)


def load_panel(symbols: list[str], start: str | None, end: str | None) -> pd.DataFrame:
    """Tidy ``(date, symbol, open, high, low, close, adj_close, volume)`` panel
    from ``data/cache`` -- the same local-disk-only path
    ``scripts/train_pattern_ml.py --data-source cache`` uses. No network.
    """
    from firm.config import get_settings
    from firm.runtime import load_prices

    settings = get_settings()
    panel = load_prices(settings)
    panel = panel[panel["symbol"].isin(symbols)].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    if start:
        panel = panel[panel["date"] >= pd.Timestamp(start)]
    if end:
        panel = panel[panel["date"] <= pd.Timestamp(end)]
    if "adj_close" not in panel.columns:
        log.warning("cached panel has no adj_close -- treating close as already adjusted")
        panel["adj_close"] = panel["close"]
    missing = sorted(set(symbols) - set(panel["symbol"].unique()))
    if missing:
        log.warning("universe symbols absent from the cache: %s", missing)
    return panel


def adjusted_open(sym_df: pd.DataFrame) -> np.ndarray:
    """The split/dividend-adjusted OPEN, on the same factor
    ``firm.strategies.pattern_recognition._adjusted_ohlc`` applies to high/low.

    That helper deliberately returns only high/low/close/volume (the scanner
    never needs an open). Every executable fill in this script is an open, so
    the same factor is recomputed here rather than mixing a raw open with an
    adjusted close -- which across a split boundary would invent a fill price
    off by the split ratio.
    """
    close = sym_df["close"].to_numpy(dtype=float)
    adj_close = sym_df["adj_close"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(close != 0, adj_close / close, 1.0)
    factor = np.nan_to_num(factor, nan=1.0, posinf=1.0, neginf=1.0)
    return sym_df["open"].to_numpy(dtype=float) * factor


# ---------------------------------------------------------------------------
# Point-in-time rolling scan
# ---------------------------------------------------------------------------

class SymbolSeries:
    """Full (untruncated) per-symbol arrays, used for every barrier walk."""

    __slots__ = ("close", "dates", "high", "low", "n", "ohlcv", "open", "symbol")

    def __init__(self, symbol: str, sym_df: pd.DataFrame):
        self.symbol = symbol
        self.ohlcv = _adjusted_ohlc(sym_df)
        self.open = adjusted_open(sym_df)
        self.high = self.ohlcv["high"].to_numpy(dtype=float)
        self.low = self.ohlcv["low"].to_numpy(dtype=float)
        self.close = self.ohlcv["close"].to_numpy(dtype=float)
        self.dates = pd.to_datetime(sym_df["date"]).to_numpy()
        self.n = len(self.ohlcv)


def scan_events(
    panel: pd.DataFrame,
    *,
    zigzag_pct: float,
    min_score: float,
    confirm_lookback_bars: int,
    stop_atr_floor: float,
    min_window_bars: int,
    step_bars: int,
    score_meta: bool,
) -> tuple[list[dict[str, Any]], dict[str, SymbolSeries]]:
    """Rolling point-in-time scan, one event per ``(symbol, confirm_index)``.

    Mirrors ``scripts/train_pattern_ml.py``'s ``build_dataset`` exactly --
    same cutoff walk, same ``seen_confirm_indices`` dedup, same market-proxy
    construction -- so the event count and label balance are comparable to the
    real training run. The only additions are the bookkeeping this
    measurement needs (``rank`` within the cutoff's ranked match list,
    ``cutoff``, and the raw ``PatternMatch`` fields) plus an optional meta-model
    ``p_act``.

    Returns ``(events, series_by_symbol)``. The events carry *no* forward
    information; all pricing happens afterwards against ``series_by_symbol``.
    """
    events: list[dict[str, Any]] = []
    series_by_symbol: dict[str, SymbolSeries] = {}

    try:
        market_proxy_by_date = (
            panel.pivot_table(index="date", columns="symbol", values="adj_close").mean(axis=1).sort_index()
        )
    except Exception:
        log.exception("market-proxy construction failed -- market-context features unavailable")
        market_proxy_by_date = None

    for symbol, sym_df in panel.groupby("symbol"):
        sym_df = sym_df.sort_values("date").reset_index(drop=True)
        try:
            series = SymbolSeries(str(symbol), sym_df)
        except Exception:
            log.exception("failed to build adjusted OHLC for %s", symbol)
            continue
        series_by_symbol[str(symbol)] = series

        market_ohlcv_full = None
        if market_proxy_by_date is not None:
            try:
                aligned = market_proxy_by_date.reindex(sym_df["date"].to_numpy())
                if not aligned.isna().any():
                    market_ohlcv_full = pd.DataFrame({"close": aligned.to_numpy()})
            except Exception:
                log.debug("market-proxy alignment failed for %s", symbol, exc_info=True)

        n = series.n
        if n < min_window_bars:
            continue
        seen_confirm_indices: set[int] = set()
        for cutoff in range(min_window_bars, n + 1, step_bars):
            window = series.ohlcv.iloc[:cutoff]
            market_window = market_ohlcv_full.iloc[:cutoff] if market_ohlcv_full is not None else None
            try:
                matches = scan_symbol(
                    window,
                    zigzag_pct=zigzag_pct,
                    min_score=min_score,
                    confirm_lookback_bars=confirm_lookback_bars,
                    stop_atr_floor=stop_atr_floor,
                )
            except Exception:
                log.exception("scan_symbol failed for %s at cutoff=%d", symbol, cutoff)
                continue

            for rank, match in enumerate(matches):
                if not match.confirmed or match.confirm_index in seen_confirm_indices:
                    continue
                seen_confirm_indices.add(match.confirm_index)

                p_act = None
                if score_meta:
                    try:
                        feats = build_features(match, window, market_ohlcv=market_window)
                        p_act = score_pattern_meta_confirmation(
                            np.asarray(list(feats.values()), dtype=float)
                        )
                    except Exception:
                        log.debug("meta scoring failed for %s/%s", symbol, match.pattern, exc_info=True)

                events.append({
                    "symbol": str(symbol),
                    "pattern": match.pattern,
                    "direction": match.direction,
                    "confirm_index": int(match.confirm_index),
                    "cutoff": int(cutoff),
                    # `rank == 0` is the only match the live strategy would
                    # ever emit for this symbol/bar (it takes matches[0]) --
                    # kept so the "what the strategy actually emits" subset is
                    # separable from the full labelled population the ML layer
                    # trains on.
                    "rank": int(rank),
                    "entry": float(match.entry),
                    "stop": float(match.stop),
                    "target": float(match.target),
                    "risk_reward": float(match.risk_reward),
                    "quality_score": float(match.quality_score),
                    "confirm_date": pd.Timestamp(series.dates[match.confirm_index]),
                    "last_pivot_index": int(max(p.index for p in match.pivots)),
                    "p_act": p_act,
                    "_match": match,
                })
    return events, series_by_symbol


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def _sign(direction: str) -> int:
    return 1 if direction == "long" else -1


def _walk_barriers(
    series: SymbolSeries, start: int, timeout_bars: int, direction: str, stop: float, target: float,
) -> tuple[int, int] | None:
    """First barrier touched over ``[start, start + timeout_bars)`` of the
    FULL series (never the as-of window -- see module docstring, §6.8 trap).
    Returns ``(label, absolute_exit_index)`` or ``None`` if ``start`` is past
    the end of the data.
    """
    if start >= series.n:
        return None
    end = min(series.n, start + timeout_bars)
    label, offset = _first_barrier_hit_with_offset(
        series.high[start:end], series.low[start:end], direction=direction, stop=stop, target=target,
    )
    return label, start + offset


def _book_trade(
    series: SymbolSeries,
    *,
    fill: float,
    stop: float,
    target: float,
    direction: str,
    start: int,
    timeout_bars: int,
) -> dict[str, Any] | None:
    """Price one round trip: enter at ``fill``, exit at whichever barrier is
    touched first (or mark to market at the vertical barrier), and express the
    result in R against ``|fill - stop|``.

    A barrier exit is booked *at the barrier price*, which flatters the
    strategy slightly (real stops slip through); costs are then charged on
    both legs at the live rate. Returns ``None`` when there is no data left to
    trade into, and flags ``degenerate_risk`` when the fill is already at or
    beyond the stop (R is undefined there -- the position is stopped the
    instant it exists).
    """
    walked = _walk_barriers(series, start, timeout_bars, direction, stop, target)
    if walked is None:
        return None
    label, exit_index = walked
    sgn = _sign(direction)
    exit_price = stop if label == -1 else target if label == 1 else float(series.close[exit_index])

    risk = sgn * (fill - stop)  # positive when the stop is on the losing side
    gross = sgn * (exit_price - fill)
    cost = COST_PER_SIDE * (abs(fill) + abs(exit_price))
    degenerate = risk <= 1e-9
    return {
        "label": int(label),
        "entry_index": int(start),
        "exit_index": int(exit_index),
        "bars_held": int(exit_index - start + 1),
        "fill": float(fill),
        "exit_price": float(exit_price),
        "risk": float(risk),
        "degenerate_risk": bool(degenerate),
        "r_gross": float(gross / risk) if not degenerate else float("nan"),
        "r_net": float((gross - cost) / risk) if not degenerate else float("nan"),
        "ret_gross": float(gross / fill) if fill > 0 else float("nan"),
        "ret_net": float((gross - cost) / fill) if fill > 0 else float("nan"),
    }


def price_nominal(ev: dict, series: SymbolSeries, timeout_bars: int) -> dict[str, Any] | None:
    """Pricing 1: fill at ``match.entry``, risk ``|entry - stop|``.

    Cross-checked against ``label_triple_barrier_with_exit`` (the function
    every existing number in this layer is built on) -- the label and exit bar
    must agree exactly, and a mismatch is surfaced rather than swallowed.
    """
    match: PatternMatch = ev["_match"]
    booked = _book_trade(
        series, fill=ev["entry"], stop=ev["stop"], target=ev["target"],
        direction=ev["direction"], start=ev["confirm_index"] + 1, timeout_bars=timeout_bars,
    )
    if booked is None:
        return None
    ref_label, ref_exit = label_triple_barrier_with_exit(match, series.ohlcv, timeout_bars=timeout_bars)
    if (ref_label, ref_exit) != (booked["label"], booked["exit_index"]):
        log.warning(
            "nominal pricing disagrees with label_triple_barrier_with_exit for %s/%s @%d: "
            "(%d,%d) vs (%d,%d)",
            ev["symbol"], ev["pattern"], ev["confirm_index"],
            booked["label"], booked["exit_index"], ref_label, ref_exit,
        )
    return booked


def price_executable(ev: dict, series: SymbolSeries, timeout_bars: int) -> dict[str, Any] | None:
    """Pricing 2: fill at the next bar's open after the confirmation bar."""
    start = ev["confirm_index"] + 1
    if start >= series.n:
        return None
    return _book_trade(
        series, fill=float(series.open[start]), stop=ev["stop"], target=ev["target"],
        direction=ev["direction"], start=start, timeout_bars=timeout_bars,
    )


def price_placebo(ev: dict, series: SymbolSeries, timeout_bars: int) -> dict[str, Any] | None:
    """Pricing 3: the direction-flipped control.

    Barriers are *mirrored about the fill* (``stop' = 2*fill - stop``,
    ``target' = 2*fill - target``) and the direction is flipped, so the
    placebo trade has byte-identical risk ``|fill - stop|`` and an identical
    reward:risk ratio to the real trade. The ONLY thing that changes is which
    way the thesis points -- so any difference between (2) and (3) is
    directional edge, not a barrier-geometry artifact.
    """
    start = ev["confirm_index"] + 1
    if start >= series.n:
        return None
    fill = float(series.open[start])
    flipped = "short" if ev["direction"] == "long" else "long"
    return _book_trade(
        series, fill=fill, stop=2 * fill - ev["stop"], target=2 * fill - ev["target"],
        direction=flipped, start=start, timeout_bars=timeout_bars,
    )


def price_resting_limit(
    ev: dict, series: SymbolSeries, timeout_bars: int, good_bars: int,
) -> dict[str, Any] | None:
    """Pricing 4: a limit order resting AT the level, good for ``good_bars``.

    Gets the nominal entry price, but only if price returns to it. A gap
    through the limit fills at that bar's open (better than the limit), which
    is the standard optimistic-but-realistic convention. Unfilled orders are
    reported as a fill rate, not silently dropped -- the fill rate is half the
    finding, because an order that only fills when the breakout is failing is
    adversely selected by construction.

    The timeout is held at the same absolute horizon as the other pricings
    (``confirm_index + 1 + timeout_bars``) so a late fill gets *less* time,
    exactly as a real good-till-date bracket would.
    """
    sgn = _sign(ev["direction"])
    level = ev["entry"]
    first = ev["confirm_index"] + 1
    last = min(series.n, first + good_bars)
    fill_index = None
    fill = None
    for i in range(first, last):
        if sgn > 0 and series.low[i] <= level:
            fill_index, fill = i, min(level, float(series.open[i]))
            break
        if sgn < 0 and series.high[i] >= level:
            fill_index, fill = i, max(level, float(series.open[i]))
            break
    if fill_index is None:
        return {"filled": False}
    horizon = (first + timeout_bars) - fill_index
    if horizon <= 0:
        return {"filled": False}
    booked = _book_trade(
        series, fill=fill, stop=ev["stop"], target=ev["target"],
        direction=ev["direction"], start=fill_index, timeout_bars=horizon,
    )
    if booked is None:
        return {"filled": False}
    booked["filled"] = True
    return booked


def _first_crossing(
    close: np.ndarray, level: float, direction: str, lo: int, hi: int,
) -> int | None:
    """First bar in ``(lo, hi]`` where ``close`` crosses ``level`` from the
    inside to the outside -- i.e. ``close[j]`` beyond and ``close[j-1]`` not
    beyond. This is the edge-triggered predicate
    ``firm.patterns.confirmation.find_confirmation`` documents but does not
    implement (Workstream C's proposed fix), evaluated here against the
    constant level ``match.entry`` (see module docstring's approximation note).
    """
    sgn = _sign(direction)
    lo = max(lo, 1)
    for j in range(lo, hi + 1):
        if sgn * (close[j] - level) > 0 and sgn * (close[j - 1] - level) <= 0:
            return j
    return None


def price_edge_triggered(ev: dict, series: SymbolSeries, timeout_bars: int) -> dict[str, Any] | None:
    """Pricing 5: fill at the next open after the TRUE first crossing.

    Searches forward from the pattern's own last structural pivot (the same
    ``min_index`` guard ``find_confirmation`` applies -- a "breakout" before
    the pattern finished forming isn't one) up to the reported
    ``confirm_index``. Events with no genuine crossing in that span are
    reported, not dropped silently: they are precisely the emissions where
    price was *already* beyond the level for the whole window, which is the
    mechanism under test.

    **This pricing is LOOK-AHEAD CONTAMINATED and is an upper bound, not an
    estimate.** It buys at bar ``j + 1``, but the pattern is only *detected*
    (and only clears ``min_score``, and only has a stop/target) at the later
    ``confirm_index`` -- a median of 5 bars later on the real panel. So the
    trade is entered using knowledge that did not exist at entry time: that a
    scoring-60+ pattern would later be found here at all. Read it as "what an
    oracle that knew which crossings would become patterns would earn."

    The honest, point-in-time version of this question is NOT this function:
    it is to actually fix ``find_confirmation`` to be edge-triggered and
    re-run the whole rolling scan, so the signal is only emitted on the bar
    the crossing is detectable. That is Workstream C; run this script from a
    worktree carrying that fix to measure it properly. The two answers differ
    a lot, and only the second one is tradeable.
    """
    j = _first_crossing(
        series.close, ev["entry"], ev["direction"], ev["last_pivot_index"] + 1, ev["confirm_index"],
    )
    if j is None:
        return {"crossed": False}
    start = j + 1
    if start >= series.n:
        return {"crossed": False}
    booked = _book_trade(
        series, fill=float(series.open[start]), stop=ev["stop"], target=ev["target"],
        direction=ev["direction"], start=start, timeout_bars=timeout_bars,
    )
    if booked is None:
        return {"crossed": False}
    booked["crossed"] = True
    booked["cross_index"] = int(j)
    booked["staleness_bars"] = int(ev["confirm_index"] - j)
    return booked


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def symbol_block_bootstrap(
    values: np.ndarray, symbols: np.ndarray, *, n_boot: int, seed: int, alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile CI for the mean, resampling WHOLE SYMBOLS with replacement.

    Deliberately not ``firm.eval.robustness.MonteCarloAnalyzer``: that
    resamples individual observations i.i.d. (``eval/robustness.py:51-72``),
    which is invalid here. These events overlap in time within a symbol (the
    rolling cutoff re-finds the same breakout; barrier windows overlap) and
    are cross-sectionally correlated across symbols (one market). Resampling
    the symbol -- carrying all of its events together -- keeps both
    dependence structures intact and is the same clustering unit
    ``firm.patterns.ml.sample_weights`` / ``purged_cv`` use for training.

    Also returns a cluster-robust t-statistic (symbol as the cluster) and the
    naive i.i.d. t-statistic, so the reader can see how much apparent
    significance is an artifact of pretending trades are independent.
    """
    mask = np.isfinite(values)
    values, symbols = values[mask], symbols[mask]
    out: dict[str, float] = {"n": len(values)}
    if len(values) == 0:
        return {**out, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "t_cluster": float("nan"), "t_iid": float("nan"), "n_symbols": 0}

    mean = float(values.mean())
    uniq = np.unique(symbols)
    by_symbol = [values[symbols == s] for s in uniq]

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    idx = rng.integers(0, len(by_symbol), size=(n_boot, len(by_symbol)))
    for b in range(n_boot):
        boot[b] = np.concatenate([by_symbol[k] for k in idx[b]]).mean()
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # Cluster-robust SE of the mean: treat each symbol's total deviation as
    # one independent draw (standard CR1-style sandwich for a sample mean).
    n = len(values)
    cluster_sums = np.array([v.sum() - len(v) * mean for v in by_symbol])
    g = len(uniq)
    se_cluster = float(np.sqrt((cluster_sums ** 2).sum()) / n) if g > 1 else float("nan")
    se_iid = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")

    return {
        **out,
        "n_symbols": int(g),
        "mean": mean,
        "median": float(np.median(values)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "se_cluster": se_cluster,
        "se_iid": se_iid,
        "t_cluster": float(mean / se_cluster) if se_cluster and np.isfinite(se_cluster) and se_cluster > 0 else float("nan"),
        "t_iid": float(mean / se_iid) if se_iid and np.isfinite(se_iid) and se_iid > 0 else float("nan"),
    }


def pct(series: np.ndarray, q: float) -> float:
    s = series[np.isfinite(series)]
    return float(np.percentile(s, q)) if len(s) else float("nan")


def describe(values: np.ndarray) -> dict[str, float]:
    v = values[np.isfinite(values)]
    if not len(v):
        return {"n": 0}
    return {
        "n": len(v), "mean": float(v.mean()), "p10": pct(v, 10), "median": pct(v, 50),
        "p90": pct(v, 90),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _trade_dicts(r_values: np.ndarray) -> list[dict]:
    """Adapt R multiples to ``firm.eval.metrics.compute_trade_metrics``'s
    per-trade log shape, so profit factor / win rate / expectancy come from
    the same implementation the backtest reports use rather than a second
    hand-rolled copy. R is the P&L unit here (every trade risks 1R by
    construction), which is exactly what that function's ``expectancy`` means.
    """
    return [{"pnl": float(r)} for r in r_values if np.isfinite(r)]


def summarize_pricing(
    name: str, rows: list[dict], events: list[dict], *, n_boot: int, seed: int,
) -> dict[str, Any]:
    """Bootstrap + trade metrics for one pricing, on gross and net R."""
    syms = np.array([e["symbol"] for e, r in zip(events, rows) if r is not None])
    r_gross = np.array([r["r_gross"] for r in rows if r is not None], dtype=float)
    r_net = np.array([r["r_net"] for r in rows if r is not None], dtype=float)
    labels = np.array([r["label"] for r in rows if r is not None], dtype=float)
    degen = int(sum(1 for r in rows if r is not None and r["degenerate_risk"]))
    held = np.array([r["bars_held"] for r in rows if r is not None], dtype=float)

    # Sensitivity: degenerate-risk fills (already at/through the stop) booked
    # at -1R instead of excluded, so the headline isn't quietly survivorship-
    # filtered by dropping the worst entries.
    r_net_with_degen = np.where(np.isfinite(r_net), r_net, -1.0) if len(r_net) else r_net
    syms_all = syms

    # Fat-tail robustness. R is a RATIO with |fill - stop| in the denominator,
    # so an event whose next-open fill lands a cent away from its stop produces
    # an R in the hundreds and can dominate a mean on its own -- and the cost
    # term (a fixed fraction of NOTIONAL, divided by that same small risk)
    # blows up with it. Winsorizing at 1%/99% bounds that without discarding
    # the observation. If the winsorized and raw means disagree materially,
    # the raw mean is being driven by a handful of near-degenerate risk
    # denominators and should not be the headline.
    if len(r_net):
        finite = r_net[np.isfinite(r_net)]
        lo_c, hi_c = (np.percentile(finite, [1, 99]) if len(finite) else (np.nan, np.nan))
        r_net_w = np.clip(r_net, lo_c, hi_c)
    else:
        r_net_w = r_net

    return {
        "pricing": name,
        "n_priced": int(sum(1 for r in rows if r is not None)),
        "n_degenerate_risk": degen,
        "target_rate": float((labels == 1).mean()) if len(labels) else float("nan"),
        "stop_rate": float((labels == -1).mean()) if len(labels) else float("nan"),
        "timeout_rate": float((labels == 0).mean()) if len(labels) else float("nan"),
        "median_bars_held": float(np.median(held)) if len(held) else float("nan"),
        "gross": symbol_block_bootstrap(r_gross, syms, n_boot=n_boot, seed=seed),
        "net": symbol_block_bootstrap(r_net, syms, n_boot=n_boot, seed=seed),
        "net_degenerate_as_loss": symbol_block_bootstrap(
            r_net_with_degen, syms_all, n_boot=n_boot, seed=seed,
        ) if len(r_net) else {},
        "net_winsorized_1pct": symbol_block_bootstrap(
            r_net_w, syms, n_boot=n_boot, seed=seed,
        ) if len(r_net) else {},
        "trade_metrics_net": compute_trade_metrics(_trade_dicts(r_net)),
    }


def _fmt_ci(d: dict[str, Any]) -> str:
    if not d or not np.isfinite(d.get("mean", float("nan"))):
        return "n/a"
    return (f"{d['mean']:+.3f}  CI[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]  "
            f"t_clu={d['t_cluster']:+.2f} t_iid={d['t_iid']:+.2f}  n={d['n']}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default=None, help="Comma-separated override; default = config/live.yaml universe.symbols")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--zigzag-pct", type=float, default=SCAN_DEFAULTS["zigzag_pct"])
    ap.add_argument("--min-score", type=float, default=SCAN_DEFAULTS["min_score"])
    ap.add_argument("--min-risk-reward", type=float, default=SCAN_DEFAULTS["min_risk_reward"])
    ap.add_argument("--confirm-lookback-bars", type=int, default=SCAN_DEFAULTS["confirm_lookback_bars"])
    ap.add_argument("--stop-atr-floor", type=float, default=SCAN_DEFAULTS["stop_atr_floor"])
    ap.add_argument("--timeout-bars", type=int, default=20)
    ap.add_argument("--min-window-bars", type=int, default=60)
    ap.add_argument("--step-bars", type=int, default=10, help="Rolling cutoff step; 10 matches the real training run")
    ap.add_argument("--limit-good-bars", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-meta", action="store_true", help="Skip meta-model p_act scoring (faster)")
    ap.add_argument("--expect-events", type=int, default=1849, help="Sanity check: known training-run match count")
    ap.add_argument("--expect-labels", default="1120,394,335", help="Sanity check: target,stop,timeout from the known run")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    prov = code_provenance()
    print(f"Detector code: git {prov['head']}")
    if prov["dirty_pattern_files"]:
        print("*** WARNING: src/firm/patterns has uncommitted changes -- these numbers describe")
        print("*** code that exists in no commit. Re-run from a pinned worktree. Dirty files:")
        for f in prov["dirty_pattern_files"]:
            print(f"***   {f}")

    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else live_universe()
    panel = load_panel(symbols, args.start, args.end)
    if panel.empty:
        print("No cached rows for this universe/date range.", file=sys.stderr)
        return 1
    print(f"Panel: {len(panel)} rows, {panel['symbol'].nunique()} symbols, "
          f"{panel['date'].min().date()} .. {panel['date'].max().date()}")

    events, series_by_symbol = scan_events(
        panel,
        zigzag_pct=args.zigzag_pct, min_score=args.min_score,
        confirm_lookback_bars=args.confirm_lookback_bars, stop_atr_floor=args.stop_atr_floor,
        min_window_bars=args.min_window_bars, step_bars=args.step_bars,
        score_meta=not args.no_meta,
    )
    print(f"Point-in-time scan: {len(events)} confirmed events "
          f"(dedup = one per (symbol, confirm_index), as in build_dataset)")
    if not events:
        return 1

    # ---- price every event five ways -------------------------------------
    nominal, executable, placebo, resting, edge = [], [], [], [], []
    for ev in events:
        series = series_by_symbol[ev["symbol"]]
        nominal.append(price_nominal(ev, series, args.timeout_bars))
        executable.append(price_executable(ev, series, args.timeout_bars))
        placebo.append(price_placebo(ev, series, args.timeout_bars))
        resting.append(price_resting_limit(ev, series, args.timeout_bars, args.limit_good_bars))
        edge.append(price_edge_triggered(ev, series, args.timeout_bars))

    # ---- sanity check against the known real training run ----------------
    nom_labels = np.array([r["label"] for r in nominal if r is not None])
    got = (int((nom_labels == 1).sum()), int((nom_labels == -1).sum()), int((nom_labels == 0).sum()))
    want = tuple(int(x) for x in args.expect_labels.split(","))
    print("\n=== SANITY CHECK vs the known training run ===")
    print(f"  events:  got {len(events)}  expected {args.expect_events}")
    print(f"  labels (target/stop/timeout): got {got}  expected {want}")
    ok = abs(len(events) - args.expect_events) <= max(5, 0.02 * args.expect_events) and all(
        abs(g - w) <= max(5, 0.05 * w) for g, w in zip(got, want)
    )
    if ok:
        print("  -> MATCHES. The scan reproduces the real training run's label balance.")
    else:
        print("  -> *** MISMATCH *** The scan does NOT reproduce the known label balance.")
        print("     Either the date range/universe differs from that run, or the scan is wrong.")
        print("     Do not trust the numbers below until this is explained.")
    if len(nom_labels):
        timeout_frac = float((nom_labels == 0).mean())
        if timeout_frac > 0.6:
            print(f"  *** {timeout_frac:.1%} timeouts -- the §6.8 truncated-window trap signature. ***")

    # ---- headline table --------------------------------------------------
    results: dict[str, Any] = {"provenance": prov, "n_events": len(events), "sanity_ok": bool(ok),
                               "labels_got": got, "labels_expected": list(want),
                               "costs": {"commission_pct": COMMISSION_PCT, "slippage_pct": SLIPPAGE_PCT,
                                         "spread_pct": SPREAD_PCT, "round_trip": COST_ROUND_TRIP}}
    print("\n=== R PER TRADE, FIVE PRICINGS (symbol-block bootstrap, 95% CI) ===")
    for name, rows in (("nominal (phantom entry)", nominal), ("executable next-open", executable),
                       ("placebo (direction-flipped)", placebo)):
        s = summarize_pricing(name, rows, events, n_boot=args.n_boot, seed=args.seed)
        results.setdefault("pricings", {})[name] = s
        print(f"\n{name}:  n={s['n_priced']}  degenerate_risk={s['n_degenerate_risk']}  "
              f"target/stop/timeout = {s['target_rate']:.1%}/{s['stop_rate']:.1%}/{s['timeout_rate']:.1%}  "
              f"median_hold={s['median_bars_held']:.0f}b")
        print(f"   gross R: {_fmt_ci(s['gross'])}")
        print(f"   net   R: {_fmt_ci(s['net'])}")
        if s.get("net_winsorized_1pct"):
            print(f"   net R (winsorized 1/99): {_fmt_ci(s['net_winsorized_1pct'])}")
        if s.get("net_degenerate_as_loss"):
            print(f"   net R (degenerate fills booked -1R): {_fmt_ci(s['net_degenerate_as_loss'])}")
        print(f"   profit_factor={s['trade_metrics_net']['profit_factor']:.3f} "
              f"win_rate={s['trade_metrics_net']['trade_win_rate']:.1%}")

    # resting limit
    filled_idx = [i for i, r in enumerate(resting) if r and r.get("filled")]
    fill_rate = len(filled_idx) / len(events)
    f_rows = [resting[i] for i in filled_idx]
    f_evs = [events[i] for i in filled_idx]
    s_lim = summarize_pricing("resting limit @ level", f_rows, f_evs, n_boot=args.n_boot, seed=args.seed)
    results.setdefault("pricings", {})["resting limit @ level"] = {**s_lim, "fill_rate": fill_rate}
    print(f"\nresting limit @ level (good {args.limit_good_bars} bars):  "
          f"fill_rate={fill_rate:.1%} ({len(filled_idx)}/{len(events)})  "
          f"stop-first={s_lim['stop_rate']:.1%}  target-first={s_lim['target_rate']:.1%}")
    print(f"   conditional gross R: {_fmt_ci(s_lim['gross'])}")
    print(f"   conditional net   R: {_fmt_ci(s_lim['net'])}")

    # edge triggered
    cross_idx = [i for i, r in enumerate(edge) if r and r.get("crossed")]
    cross_rate = len(cross_idx) / len(events)
    e_rows = [edge[i] for i in cross_idx]
    e_evs = [events[i] for i in cross_idx]
    s_edge = summarize_pricing("edge-triggered next-open", e_rows, e_evs, n_boot=args.n_boot, seed=args.seed)
    stale = np.array([edge[i]["staleness_bars"] for i in cross_idx], dtype=float)
    results.setdefault("pricings", {})["edge-triggered next-open"] = {
        **s_edge, "crossing_found_rate": cross_rate, "staleness": describe(stale),
    }
    print(f"\nedge-triggered (true first crossing) next-open:  "
          f"crossing located for {cross_rate:.1%} ({len(cross_idx)}/{len(events)})  "
          f"median staleness (confirm_index - cross_index) = {pct(stale, 50):.0f} bars")
    print(f"   gross R: {_fmt_ci(s_edge['gross'])}")
    print(f"   net   R: {_fmt_ci(s_edge['net'])}")

    # ---- distributions ---------------------------------------------------
    print("\n=== ENTRY-PRICE DIAGNOSTICS ===")
    sgn = np.array([_sign(e["direction"]) for e in events], dtype=float)
    entry = np.array([e["entry"] for e in events], dtype=float)
    stop = np.array([e["stop"] for e in events], dtype=float)
    target = np.array([e["target"] for e in events], dtype=float)
    qual = np.array([e["quality_score"] for e in events], dtype=float)
    nominal_rr = np.array([e["risk_reward"] for e in events], dtype=float)
    close_at_confirm = np.array(
        [series_by_symbol[e["symbol"]].close[e["confirm_index"]] for e in events], dtype=float)
    nom_risk = np.abs(entry - stop)
    overshoot = np.where(nom_risk > 1e-9, sgn * (close_at_confirm - entry) / nom_risk, np.nan)

    fills = np.array([r["fill"] if r is not None else np.nan for r in executable], dtype=float)
    real_risk = sgn * (fills - stop)
    real_reward = sgn * (target - fills)
    real_rr = np.where(real_risk > 1e-9, real_reward / real_risk, np.nan)

    print(f"  overshoot (close_at_confirm - entry)/|entry-stop| : {describe(overshoot)}")
    print(f"  nominal risk_reward (what min_risk_reward gates on): {describe(nominal_rr)}")
    print(f"  REAL risk_reward at the executable fill            : {describe(real_rr)}")
    gate = args.min_risk_reward
    passes_nominal = nominal_rr >= gate
    both = np.isfinite(real_rr) & passes_nominal
    below = float((real_rr[both] < gate).mean()) if both.any() else float("nan")
    print(f"  emitted signals (nominal RR >= {gate}) whose REAL RR is below {gate}: {below:.1%} "
          f"(n={int(both.sum())})")
    print(f"  fills already at/through the stop (real RR undefined): "
          f"{int((real_risk <= 1e-9).sum())}/{len(events)}")

    fin = np.isfinite(overshoot) & np.isfinite(qual)
    corr_p = float(np.corrcoef(qual[fin], overshoot[fin])[0, 1]) if fin.sum() > 2 else float("nan")
    corr_s = float(pd.Series(qual[fin]).corr(pd.Series(overshoot[fin]), method="spearman")) if fin.sum() > 2 else float("nan")
    print(f"  corr(quality_score, overshoot): pearson={corr_p:+.3f} spearman={corr_s:+.3f} (n={int(fin.sum())})")
    results["diagnostics"] = {
        "overshoot": describe(overshoot), "nominal_rr": describe(nominal_rr),
        "real_rr": describe(real_rr), "frac_emitted_real_rr_below_gate": below,
        "corr_quality_overshoot_pearson": corr_p, "corr_quality_overshoot_spearman": corr_s,
        "n_fill_through_stop": int((real_risk <= 1e-9).sum()),
    }

    # ---- expectancy by meta-model p_act ---------------------------------
    p_act = np.array([e["p_act"] if e["p_act"] is not None else np.nan for e in events], dtype=float)
    exec_net = np.array([r["r_net"] if r is not None else np.nan for r in executable], dtype=float)
    exec_syms = np.array([e["symbol"] for e in events])
    if np.isfinite(p_act).sum() > 20:
        print("\n=== EXECUTABLE NET R CONDITIONED ON META-MODEL p_act ===")
        print("  (in-sample w.r.t. the shipped artifact -- it was trained on this same panel;")
        print("   read as an upper bound on what the gate could deliver, not an OOS estimate)")
        buckets = results["p_act_buckets"] = {}
        edges = [0.0, 0.3, 0.5, 0.7, 0.85, 1.01]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = np.isfinite(p_act) & (p_act >= lo) & (p_act < hi) & np.isfinite(exec_net)
            if m.sum() < 10:
                print(f"  p_act [{lo:.2f},{hi:.2f}): n={int(m.sum())} (too few)")
                continue
            b = symbol_block_bootstrap(exec_net[m], exec_syms[m], n_boot=args.n_boot, seed=args.seed)
            buckets[f"[{lo:.2f},{hi:.2f})"] = b
            print(f"  p_act [{lo:.2f},{hi:.2f}): {_fmt_ci(b)}")
        for thr in (0.5, 0.6, 0.7, 0.8):
            m = np.isfinite(p_act) & (p_act >= thr) & np.isfinite(exec_net)
            if m.sum() < 10:
                continue
            b = symbol_block_bootstrap(exec_net[m], exec_syms[m], n_boot=args.n_boot, seed=args.seed)
            buckets[f">={thr:.2f}"] = b
            print(f"  p_act >= {thr:.2f}   : {_fmt_ci(b)}")
    else:
        print("\n(meta-model p_act unavailable -- skipped conditional expectancy)")

    # ---- strategy-emitted subset ----------------------------------------
    emit_mask = np.array([e["rank"] == 0 and e["risk_reward"] >= gate for e in events])
    if emit_mask.sum() > 20:
        b = symbol_block_bootstrap(exec_net[emit_mask], exec_syms[emit_mask], n_boot=args.n_boot, seed=args.seed)
        results["emitted_subset_executable_net"] = b
        print("\n=== SUBSET THE LIVE STRATEGY WOULD ACTUALLY EMIT "
              f"(best match per scan AND nominal RR >= {gate}) ===")
        print(f"  executable net R: {_fmt_ci(b)}")

    # ---- by pattern family (does the sloped-level approximation matter?) -
    fam_h = {"head_shoulders_top", "inverse_head_shoulders", "double_top", "double_bottom",
             "triple_top", "triple_bottom", "rectangle", "cup_handle", "rounding_bottom"}
    is_h = np.array([e["pattern"] in fam_h for e in events])
    for label, m in (("horizontal-level patterns (exact)", is_h), ("sloped-level patterns (approx)", ~is_h)):
        m2 = m & np.isfinite(exec_net)
        if m2.sum() > 20:
            b = symbol_block_bootstrap(exec_net[m2], exec_syms[m2], n_boot=args.n_boot, seed=args.seed)
            results.setdefault("by_level_geometry", {})[label] = b
            print(f"\n{label}: executable net R {_fmt_ci(b)}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=1, default=str)
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
