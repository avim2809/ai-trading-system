"""Allocation forward-test I1-I5 monitor (P5-06).

Computes the frozen implementation bars of ``scripts/allocation_forward_test_preregistered.py``
(live NAV vs the offline replay of the real allocator). It only reads (GET-only
allowlisted loopback client) and only alerts: it never executes
``IMPLEMENTATION_ACTION.any_fail`` and never touches broker, engine or config.

Research-session rule: code and SYNTHETIC tests only; real outputs are owner-read.
No ``.env`` access anywhere: secrets come from ``os.environ`` (injected by the unit).
"""

from __future__ import annotations

import contextlib
import importlib
import json
import logging
import math
import os
import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _ROOT / "scripts"

# --- encoded thresholds (checked against the frozen bar text by check_thresholds_against_frozen)
I1_WINDOW_DAYS = 63
I1_MEAN_MAX = 5e-4          # 5 bps/day
I1_SD_MAX = 15e-4           # 15 bps/day
I2_MAX_REL = 0.02
I3_MEDIAN_BPS = {"etf": 15.0, "crypto": 60.0}
I3_SINGLE_BPS = {"etf": 100.0, "crypto": 300.0}
I4_COUNT_TOL = 1
SLEEVE_WEIGHT = 0.08
SLEEVE_BAND = 0.10
SLEEVE_ZERO_TOL = 1e-9
LAG_DAYS_IMPLEMENTED = 1    # BtcTrendSleeve has no lag parameter: the 1-day lag is structural (fills at t, plan on <t data)

LIVE_NAV_RULE_ID = "last_snapshot_at_or_after_16:00_ET_per_us_trading_day"
LIVE_NAV_CUTOFF_HOUR_ET = 16
ENTRY_COST_NOTE = ("initial_nav is POST-fill equity (no pre_fill_nav in trial history): I1/I2 for the first window "
                   "carry an entry-cost offset")
DATA_END_FAR = "2200-01-01"
ALLOWLIST = ("/api/live/portfolio-history", "/api/live/orders", "/api/live/cycles", "/api/live/status")
_LOOPBACK = ("127.0.0.1", "localhost", "::1")
INPUT_FILES = (
    [f"tiingo_{t}.parquet" for t in ("SPY", "IEF", "EFA", "EEM", "TLT", "GLD", "DBC", "VNQ", "SVXY")]
    + ["tiingo_BTCUSD.parquet", "cboe_VIX.parquet", "cboe_VIX3M.parquet", "cboe_PUT.parquet",
       "fred_DTB3.parquet", "fomc_scheduled.parquet"]
)
BAR_IDS = ("I1_tracking_error", "I2_cumulative_drift", "I3_fills", "I4_turnover",
           "I5_no_unmanaged_drift", "BAR_SLEEVE_DRIFT")


@dataclass(frozen=True)
class BarResult:
    bar_id: str
    passed: bool | None          # None = not yet evaluable
    value: dict[str, float]
    note: str


# ---------------------------------------------------------------------------
# Frozen modules (imported from scripts/, never edited)
# ---------------------------------------------------------------------------

def _scripts_on_path() -> None:
    for p in (str(_ROOT / "src"), str(_SCRIPTS)):
        if p not in sys.path:
            sys.path.insert(0, p)


def load_frozen():
    _scripts_on_path()
    return importlib.import_module("allocation_forward_test_preregistered")


def check_thresholds_against_frozen(frozen=None) -> None:
    """Raise ValueError if any encoded threshold differs from the number in the frozen bar text."""
    frozen = frozen or load_frozen()
    rules = {b["id"]: b["rule"] for b in frozen.BARS_IMPLEMENTATION}
    missing = [b for b in BAR_IDS if b not in rules]
    if missing:
        raise ValueError(f"frozen bars missing {missing}")

    def need(cond: bool, what: str) -> None:
        if not cond:
            raise ValueError(f"encoded threshold mismatch vs frozen bar text: {what}")

    r = rules["I1_tracking_error"]
    need(f"rolling {I1_WINDOW_DAYS}-trading-day" in r, "I1 window")
    need(f"|mean| <= {I1_MEAN_MAX * 1e4:g} bps/day" in r, "I1 mean")
    need(f"stdev <= {I1_SD_MAX * 1e4:g} bps/day" in r, "I1 sd")
    need(f"<= {I2_MAX_REL * 100:g}% at every month-end" in rules["I2_cumulative_drift"], "I2")
    r = rules["I3_fills"]
    need(f"<= {I3_MEDIAN_BPS['etf']:g} bps for SPY/IEF and <= {I3_MEDIAN_BPS['crypto']:g} bps for BTC/USD" in r, "I3 median")
    need(f"no single fill > {I3_SINGLE_BPS['etf']:g} bps / {I3_SINGLE_BPS['crypto']:g} bps" in r, "I3 single")
    need(f"+/-{I4_COUNT_TOL} of the simulation" in rules["I4_turnover"], "I4")
    r = rules["BAR_SLEEVE_DRIFT"]
    need(f"{SLEEVE_WEIGHT:.2f} x [target_within +/- {SLEEVE_BAND:.2f}]" in r, "sleeve drift")
    need("or be 0 when the trend is off" in r, "sleeve zero")
    need("bit-for-bit" in rules["I5_no_unmanaged_drift"], "I5")


# ---------------------------------------------------------------------------
# Simulation wrapper with in-process DATA_END override
# ---------------------------------------------------------------------------

_OVERRIDE_LOCK = threading.Lock()


@contextlib.contextmanager
def _data_end_override(as_of: str) -> Iterator[None]:
    """Set ``run_alt_premia_evaluation.prereg.DATA_END`` in-process only; always restored."""
    _scripts_on_path()
    ev = importlib.import_module("run_alt_premia_evaluation")
    with _OVERRIDE_LOCK:
        original = ev.prereg.DATA_END
        ev.prereg.DATA_END = str(as_of)
        try:
            yield
        finally:
            ev.prereg.DATA_END = original


def simulated_nav(allocation_cfg: dict, data_dir: Path, start: str,
                  end: str | None, initial_nav: float) -> pd.Series:
    """NAV series from ``scripts/allocation_replay.replay`` under the DATA_END override (never a bare call)."""
    _scripts_on_path()
    replay_mod = importlib.import_module("allocation_replay")
    as_of = end or DATA_END_FAR
    with _data_end_override(as_of):
        res = replay_mod.replay(allocation_cfg, Path(data_dir), start, as_of, initial_nav)
    nav = res["_nav"]
    nav.index = pd.DatetimeIndex(nav.index)
    return nav


# ---------------------------------------------------------------------------
# GET-only allowlist client
# ---------------------------------------------------------------------------

class ForbiddenRequest(Exception):
    pass


class GetOnlyClient:
    """Loopback GET-only client; refuses anything else BEFORE any request is made (OD-19 a)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8001", *, session=None, timeout: float = 15.0,
                 allowlist: tuple[str, ...] = ALLOWLIST) -> None:
        parts = urlsplit(base_url)
        if parts.scheme != "http" or parts.hostname not in _LOOPBACK or parts.username or parts.password:
            raise ForbiddenRequest(f"base_url must be plain http to loopback without credentials: {base_url!r}")
        self.base_url = f"http://{parts.netloc}"
        self.timeout = timeout
        self.allowlist = tuple(allowlist)
        if session is None:
            import requests
            session = requests.Session()
            session.trust_env = False   # no proxies / netrc from the environment
        self._session = session

    def _check(self, method: str, path: str) -> None:
        if str(method).upper() != "GET":
            raise ForbiddenRequest(f"method {method!r} not allowed (GET only)")
        if path not in self.allowlist:  # exact match: no query, scheme, host, traversal
            raise ForbiddenRequest(f"path {path!r} not in allowlist")

    def request(self, method: str, path: str, **kwargs: Any):
        self._check(method, path)
        if set(kwargs) - {"params"}:
            raise ForbiddenRequest(f"unsupported request options {sorted(set(kwargs) - {'params'})}")
        return self.get(path, **kwargs)

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        self._check("GET", path)
        resp = self._session.request("GET", self.base_url + path, params=params, timeout=self.timeout,
                                     allow_redirects=False)
        resp.raise_for_status()
        return resp.json()


def fetch_live_snapshots(client: GetOnlyClient) -> pd.Series:
    """Portfolio snapshots (timestamp -> NAV) from GET /api/live/portfolio-history."""
    body = client.get("/api/live/portfolio-history")
    idx = pd.to_datetime(pd.Series(body.get("dates", [])), utc=False)
    return pd.Series([float(v) for v in body.get("values", [])], index=pd.DatetimeIndex(idx), dtype=float)


# ---------------------------------------------------------------------------
# live_nav selection and tracking error
# ---------------------------------------------------------------------------

def select_live_nav(snapshots: pd.Series, trading_days: pd.DatetimeIndex | None = None
                    ) -> tuple[pd.Series, dict[str, str]]:
    """Last snapshot at or after 16:00 ET per US trading day; days without one are flagged, never filled.

    Naive timestamps are UTC (engine convention). ``trading_days`` (market dates) restricts/flags the
    expected days; default = US trading days spanned by the snapshots.
    """
    from firm.allocation.calendar import is_us_trading_day

    if len(snapshots):
        idx = pd.DatetimeIndex(snapshots.index)
        idx = idx.tz_localize("UTC") if idx.tz is None else idx
        et = idx.tz_convert(ZoneInfo("US/Eastern"))
        df = pd.DataFrame({"nav": snapshots.to_numpy(float), "et": et, "day": et.normalize().tz_localize(None)})
        late = df[df["et"].dt.hour >= LIVE_NAV_CUTOFF_HOUR_ET].sort_values("et")
        picked = late.groupby("day")["nav"].last()
        days_seen = pd.DatetimeIndex(sorted(set(df["day"])))
    else:
        picked = pd.Series(dtype=float)
        days_seen = pd.DatetimeIndex([])
    if trading_days is None:
        trading_days = pd.DatetimeIndex([d for d in days_seen if is_us_trading_day(d.date())])
    flagged = {str(d.date()): "no snapshot at or after 16:00 ET" for d in trading_days if d not in picked.index}
    out = picked[picked.index.isin(trading_days)].sort_index()
    out.index = pd.DatetimeIndex(out.index)
    return out, flagged


def tracking_error(live_nav: pd.Series, sim_nav: pd.Series,
                   excluded_days: Mapping[str, str] | None = None) -> pd.Series:
    """Daily (live return - sim return) on common days; excluded days removed (listed by the caller)."""
    excluded = {str(pd.Timestamp(k).date()) for k in (excluded_days or {})}
    common = live_nav.index.intersection(sim_nav.index).sort_values()
    r_live = live_nav.reindex(common).pct_change()
    r_sim = sim_nav.reindex(common).pct_change()
    te = (r_live - r_sim).dropna()
    return te[[str(d.date()) not in excluded for d in te.index]]


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------

def bar_i1(live_nav: pd.Series, sim_nav: pd.Series, excluded_days: Mapping[str, str] | None = None) -> BarResult:
    excluded = dict(excluded_days or {})
    te = tracking_error(live_nav, sim_nav, excluded)
    ex_note = f" excluded_days={sorted(excluded)}" if excluded else ""
    if len(te) < I1_WINDOW_DAYS:
        return BarResult("I1_tracking_error", None,
                         {"n": float(len(te)), "mean": float(te.mean()) if len(te) else float("nan"),
                          "sd": float(te.std(ddof=1)) if len(te) > 1 else float("nan")},
                         f"not evaluable: {len(te)} < {I1_WINDOW_DAYS} days.{ex_note}")
    mean = te.rolling(I1_WINDOW_DAYS).mean().dropna()
    sd = te.rolling(I1_WINDOW_DAYS).std(ddof=1).dropna()
    worst_mean, worst_sd = float(mean.abs().max()), float(sd.max())
    ok = worst_mean <= I1_MEAN_MAX and worst_sd <= I1_SD_MAX
    return BarResult("I1_tracking_error", bool(ok),
                     {"n": float(len(te)), "worst_abs_mean": worst_mean, "worst_sd": worst_sd,
                      "latest_mean": float(mean.iloc[-1]), "latest_sd": float(sd.iloc[-1])},
                     f"all rolling {I1_WINDOW_DAYS}d windows checked (sample sd, ddof=1).{ex_note}")


def _month_end_days(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    out = []
    for _, grp in pd.Series(index, index=index).groupby(index.to_period("M")):
        last = grp.iloc[-1]
        if (last + pd.offsets.BDay(1)).month != last.month or last != index[-1]:
            out.append(last)
    return out


def bar_i2(live_nav: pd.Series, sim_nav: pd.Series) -> BarResult:
    common = live_nav.index.intersection(sim_nav.index).sort_values()
    if len(common) == 0:
        return BarResult("I2_cumulative_drift", None, {}, "no common days")
    ends = _month_end_days(common)
    if not ends:
        return BarResult("I2_cumulative_drift", None, {}, "no completed month-end yet")
    dev = {str(d.date()): float(live_nav[d] / sim_nav[d] - 1) for d in ends}
    worst = max(abs(v) for v in dev.values())
    return BarResult("I2_cumulative_drift", bool(worst <= I2_MAX_REL),
                     {"worst_abs_dev": worst, "n_month_ends": float(len(dev))},
                     "month-end deviations: " + json.dumps({k: round(v, 6) for k, v in dev.items()}))


def _kind(symbol: str) -> str:
    return "crypto" if "/" in str(symbol) else "etf"


def bar_i3(fills: pd.DataFrame | None) -> BarResult:
    if fills is None or len(fills) == 0:
        return BarResult("I3_fills", None, {}, "no fills supplied")
    f = fills.copy()
    f["kind"] = f["symbol"].map(_kind)
    f["bps"] = (f["fill_price"] / f["close"] - 1).abs() * 1e4
    reason = f["reason"] if "reason" in f else pd.Series([None] * len(f), index=f.index)
    f["has_reason"] = reason.map(lambda r: isinstance(r, str) and r.strip() != "")
    values: dict[str, float] = {}
    problems = []
    for kind in ("etf", "crypto"):
        g = f[f["kind"] == kind]
        if g.empty:
            continue
        med = float(g["bps"].median())
        values[f"median_bps_{kind}"] = med
        if med > I3_MEDIAN_BPS[kind]:
            problems.append(f"{kind} median {med:.1f} bps > {I3_MEDIAN_BPS[kind]:g}")
        bad = g[(g["bps"] > I3_SINGLE_BPS[kind]) & ~g["has_reason"]]
        if len(bad):
            problems.append(f"{len(bad)} {kind} fill(s) > {I3_SINGLE_BPS[kind]:g} bps without logged reason")
    return BarResult("I3_fills", not problems, values, "; ".join(problems) or "ok")


def bar_i4(fills: pd.DataFrame | None, decisions: pd.DataFrame | None,
           sim_core_trades_by_month: Mapping[str, int] | None) -> BarResult:
    if fills is None or decisions is None:
        return BarResult("I4_turnover", None, {}, "orders/decisions not supplied")
    problems = []
    dec = decisions.set_index("decision_id") if "decision_id" in decisions else pd.DataFrame()
    untraceable = 0
    for _, o in fills.iterrows():
        did = o.get("decision_id")
        if did is None or (isinstance(did, float) and math.isnan(did)) or did not in dec.index:
            continue  # unmapped orders are an I5 failure; I4 only judges triggers of mapped ones
        d = dec.loc[did]
        trig = d.get("trigger")
        if not isinstance(trig, str) or not trig.strip() or trig == "drift_band" and not np.isfinite(d.get("breach_weight", np.nan)):
            untraceable += 1
    if untraceable:
        problems.append(f"{untraceable} trade(s) without traceable trigger/breach weight")
    values = {"untraceable": float(untraceable)}
    if sim_core_trades_by_month is None:
        return BarResult("I4_turnover", False if problems else None, values,
                         "; ".join(problems + ["sim monthly counts unavailable (replay() exposes only a total)"]))
    core = fills[fills["sleeve"] == "core"].copy()
    core["month"] = pd.to_datetime(core["date"]).dt.strftime("%Y-%m")
    live = core.groupby("month").size().to_dict()
    for m in sorted(set(live) | set(sim_core_trades_by_month)):
        diff = abs(int(live.get(m, 0)) - int(sim_core_trades_by_month.get(m, 0)))
        values[f"diff_{m}"] = float(diff)
        if diff > I4_COUNT_TOL:
            problems.append(f"{m}: live {live.get(m, 0)} vs sim {sim_core_trades_by_month.get(m, 0)}")
    return BarResult("I4_turnover", not problems, values, "; ".join(problems) or "ok")


def _live_params(live_cfg: dict, cost_bps_crypto: float | None) -> dict[str, Any]:
    from firm.allocation.allocator import build_allocator

    alloc = build_allocator(live_cfg)
    core = next(s for s in alloc.sleeves if s.name == "core")
    sat = next(s for s in alloc.sleeves if s.name == "btc_trend")
    return {
        "core.weight_of_nav": core.weight,
        "core.targets": dict(live_cfg["sleeves"][0]["weights"]),
        "core.drift_band_abs": alloc.band_abs,
        "satellite.weight_of_nav": sat.weight,
        "satellite.band_abs": sat.band_abs,
        "satellite.target_vol": sat.target_vol,
        "satellite.lag_days": LAG_DAYS_IMPLEMENTED,
        "satellite.cost_bps_per_side": cost_bps_crypto,
        "cash_buffer": alloc.cash_buffer,
    }


def _frozen_params(frozen_portfolio: dict) -> dict[str, Any]:
    p = frozen_portfolio
    return {
        "core.weight_of_nav": p["core"]["weight_of_nav"],
        "core.targets": dict(p["core"]["targets"]),
        "core.drift_band_abs": p["core"]["drift_band_abs"],
        "satellite.weight_of_nav": p["satellite"]["weight_of_nav"],
        "satellite.band_abs": p["satellite"]["band_abs"],
        "satellite.target_vol": 0.40,   # frozen text: min(1, 0.40 / ...)
        "satellite.lag_days": p["satellite"]["lag_days"],
        "satellite.cost_bps_per_side": p["satellite"]["cost_bps_per_side"],
        "cash_buffer": p["cash_buffer"],
    }


def verify_frozen_params(live_cfg: dict, frozen: dict, cost_bps_crypto: float | None = None) -> BarResult:
    """I5 parameter half: live allocation config vs frozen PORTFOLIO, exact equality (no tolerance)."""
    if cost_bps_crypto is None:
        _scripts_on_path()
        cost_bps_crypto = float(importlib.import_module("allocation_replay").COST_BPS["crypto"])
    live, fz = _live_params(live_cfg, cost_bps_crypto), _frozen_params(frozen)
    diffs = {k: (live[k], fz[k]) for k in fz if live[k] != fz[k]}
    return BarResult("I5_no_unmanaged_drift", not diffs, {"n_mismatch": float(len(diffs))},
                     "ok" if not diffs else "mismatch: " + json.dumps(diffs, default=str))


def _bar_i5(params: BarResult, fills: pd.DataFrame | None, decisions: pd.DataFrame | None) -> BarResult:
    if fills is None or decisions is None:
        return BarResult("I5_no_unmanaged_drift", False if params.passed is False else None, params.value,
                         f"params: {params.note}; orders/decisions not supplied")
    ids = set(decisions["decision_id"]) if "decision_id" in decisions else set()
    unmapped = [r for r in fills.get("decision_id", pd.Series([None] * len(fills))) if r not in ids]
    ok = params.passed is True and not unmapped
    return BarResult("I5_no_unmanaged_drift", bool(ok), {**params.value, "unmapped_orders": float(len(unmapped))},
                     f"params: {params.note}; unmapped orders: {len(unmapped)}")


def bar_sleeve_drift(reviews: pd.DataFrame | None) -> BarResult:
    """reviews: one row per weekly review AFTER its fills: date, share_of_nav, target_within, trend_on."""
    if reviews is None or len(reviews) == 0:
        return BarResult("BAR_SLEEVE_DRIFT", None, {}, "no weekly reviews supplied")
    breaches = []
    for _, r in reviews.iterrows():
        share = float(r["share_of_nav"])
        if not bool(r["trend_on"]):
            if abs(share) > SLEEVE_ZERO_TOL:
                breaches.append(f"{r['date']}: trend off, share {share:.6f}")
            continue
        tw = float(r["target_within"])
        lo, hi = SLEEVE_WEIGHT * (tw - SLEEVE_BAND), SLEEVE_WEIGHT * (tw + SLEEVE_BAND)
        if not (lo - 1e-12 <= share <= hi + 1e-12):
            breaches.append(f"{r['date']}: share {share:.6f} outside [{lo:.6f}, {hi:.6f}]")
    return BarResult("BAR_SLEEVE_DRIFT", not breaches, {"n_reviews": float(len(reviews)), "n_breach": float(len(breaches))},
                     "; ".join(breaches) or "ok")


def evaluate_bars(live_nav: pd.Series, sim_nav: pd.Series, fills: pd.DataFrame | None,
                  decisions: pd.DataFrame | None, frozen_portfolio: dict, *, live_cfg: dict | None = None,
                  excluded_days: Mapping[str, str] | None = None,
                  sim_core_trades_by_month: Mapping[str, int] | None = None,
                  sleeve_reviews: pd.DataFrame | None = None,
                  cost_bps_crypto: float | None = None) -> list[BarResult]:
    if live_cfg is not None:
        params = verify_frozen_params(live_cfg, frozen_portfolio, cost_bps_crypto)
    else:
        params = BarResult("I5_no_unmanaged_drift", None, {}, "live config not supplied")
    return [
        bar_i1(live_nav, sim_nav, excluded_days),
        bar_i2(live_nav, sim_nav),
        bar_i3(fills),
        bar_i4(fills, decisions, sim_core_trades_by_month),
        _bar_i5(params, fills, decisions),
        bar_sleeve_drift(sleeve_reviews),
    ]


# ---------------------------------------------------------------------------
# Deployment record, alerts, state
# ---------------------------------------------------------------------------

@dataclass
class Deployment:
    start_date: str
    expected_fingerprint: str
    initial_nav: float
    initial_nav_note: str
    rule_entry_ok: bool


def read_deployment(trial_history_path: Path) -> Deployment:
    hist = json.loads(Path(trial_history_path).read_text())
    entry = next(e for e in reversed(hist["entries"]) if "allocation_forward_test" in str(e.get("prereg", "")))
    pre = entry.get("pre_fill_nav")
    nav = float(pre if pre is not None else entry["starting_state_after_fills"]["equity"])
    rule_ok = any(e.get("live_nav_selection_rule") == LIVE_NAV_RULE_ID for e in hist["entries"])
    return Deployment(str(entry["start_date"]), str(entry["fingerprint"]), nav,
                      "pre-fill NAV" if pre is not None else ENTRY_COST_NOTE, rule_ok)


def load_replay_config(config_path: Path) -> dict:
    return yaml.safe_load(Path(config_path).read_text())["allocation"]


def make_notify(timeout: float = 5.0) -> Callable[..., None]:
    """Webhook notifier; URL from os.environ only (never .env)."""
    def _notify(*, severity: str, kind: str, message: str, **extra: Any) -> None:
        url = os.environ.get("ALERT_WEBHOOK_URL", "").strip()
        alert = {"severity": severity, "kind": kind, "message": message,
                 "timestamp": datetime.now(UTC).isoformat(), **extra}
        if not url:
            log.warning("ALERT (no webhook configured) %s %s: %s", severity, kind, message)
            return
        from firm.live.notifications import _post_webhook
        try:
            _post_webhook(url, alert, timeout)
        except Exception:
            log.warning("alert delivery failed for %s", kind, exc_info=True)
    return _notify


def _guard_state_dir(state_dir: Path) -> Path:
    sd = Path(state_dir).resolve()
    allowed = (_ROOT / "data" / "forward_monitors").resolve()
    try:
        sd.relative_to(_ROOT)
    except ValueError:
        return sd           # outside the repo (e.g. tmp dir): cannot dirty a tracked path
    try:
        sd.relative_to(allowed)
    except ValueError:
        raise ValueError(f"state_dir {sd} is inside the repo but not under data/forward_monitors/") from None
    return sd


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str, sort_keys=True))
    os.replace(tmp, path)


def run_daily(state_dir: Path, *, live_nav_fetcher: Callable[[], pd.Series],
              notify: Callable[..., None], trial_history_path: Path | None = None,
              config_path: Path | None = None, frozen=None,
              fills_fetcher: Callable[[], pd.DataFrame | None] | None = None,
              decisions_fetcher: Callable[[], pd.DataFrame | None] | None = None,
              sleeve_reviews_fetcher: Callable[[], pd.DataFrame | None] | None = None,
              excluded_days: Mapping[str, str] | None = None,
              sim_core_trades_by_month: Mapping[str, int] | None = None) -> dict:
    """One daily evaluation. Writes only under ``state_dir``; alerts only; never halts anything."""
    sd = _guard_state_dir(state_dir)
    frozen = frozen or load_frozen()
    trial_history_path = Path(trial_history_path or _ROOT / "docs" / "allocation_forward_test_trial_history.json")
    config_path = Path(config_path or _ROOT / "config" / "live_alpaca.yaml")
    inputs_dir = sd / "inputs"
    result: dict[str, Any] = {"status": "ok", "run_at": datetime.now(UTC).isoformat()}

    def finish(status: str) -> dict:
        result["status"] = status
        _write_json(sd / "state.json", {k: v for k, v in result.items() if k != "bars"} | {"n_bars": len(result.get("bars", []))})
        _write_json(sd / "daily" / f"{result['run_at'][:10]}.json", result)
        return result

    dep = read_deployment(trial_history_path)
    result["start_date"] = dep.start_date
    fp = frozen.bars_fingerprint()
    if fp != dep.expected_fingerprint:
        notify(severity="critical", kind="allocation_forward_integrity_failure",
               message=f"frozen bars fingerprint {fp[:12]} != recorded {dep.expected_fingerprint[:12]}")
        return finish("integrity_failure")
    try:
        check_thresholds_against_frozen(frozen)
    except ValueError as exc:
        notify(severity="critical", kind="allocation_forward_integrity_failure", message=str(exc))
        return finish("integrity_failure")
    if not dep.rule_entry_ok:
        notify(severity="warning", kind="allocation_forward_blocked",
               message=f"trial history lacks live_nav_selection_rule={LIVE_NAV_RULE_ID!r}; owner must append it")
        return finish("blocked_missing_live_nav_rule")

    cfg = load_replay_config(config_path)
    verify_frozen_params(cfg, frozen.PORTFOLIO)
    spy = pd.read_parquet(inputs_dir / "tiingo_SPY.parquet")["date"]
    as_of = str(pd.Timestamp(spy.max()).date())
    sim = simulated_nav(cfg, inputs_dir, dep.start_date, as_of, dep.initial_nav)
    result["data_end_override"] = as_of
    result["as_of"] = str(sim.index[-1].date())
    result["initial_nav_note"] = dep.initial_nav_note

    live_nav, flagged = select_live_nav(live_nav_fetcher(), pd.DatetimeIndex(sim.index))
    result["flagged_days"] = flagged
    excl = {**dict(excluded_days or {})}
    bars = evaluate_bars(
        live_nav, sim,
        fills_fetcher() if fills_fetcher else None,
        decisions_fetcher() if decisions_fetcher else None,
        frozen.PORTFOLIO, live_cfg=cfg, excluded_days=excl,
        sim_core_trades_by_month=sim_core_trades_by_month,
        sleeve_reviews=sleeve_reviews_fetcher() if sleeve_reviews_fetcher else None)
    result["bars"] = [asdict(b) for b in bars]
    result["excluded_days"] = excl
    failed = [b.bar_id for b in bars if b.passed is False]
    result["failed"] = failed
    if failed:
        notify(severity="critical", kind="allocation_forward_bar_fail",
               message="implementation bars failed: " + ", ".join(failed) + " (alert only; halt is an owner/incident action)",
               failed=failed)
    return finish("fail" if failed else "ok")


# ---------------------------------------------------------------------------
# Input refresh (network, owner-approved; fetchers injected)
# ---------------------------------------------------------------------------

def refresh_inputs(inputs_dir: Path, fetchers: Mapping[str, Callable[[], pd.DataFrame]]) -> list[str]:
    """Fetch every input file into a temp dir, then move into place only if all 15 succeeded."""
    missing = [f for f in INPUT_FILES if f not in fetchers]
    if missing:
        raise ValueError(f"no fetcher for {missing}")
    inputs_dir = Path(inputs_dir)
    tmp = inputs_dir.with_name(inputs_dir.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    for name in INPUT_FILES:
        fetchers[name]().to_parquet(tmp / name)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    for name in INPUT_FILES:
        os.replace(tmp / name, inputs_dir / name)
    tmp.rmdir()
    return list(INPUT_FILES)
