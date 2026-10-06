"""Shadow forward test of the S2 breadth overlay (frozen: scripts/s2_forward_preregistered.py).

Places NO orders and touches no live config. Keeps a private ledger under
``data/research/s2_forward/``:

    decisions.jsonl   one line per monthly check (signal, state, weights, status)
    nav.csv           shadow NAV of overlay_v1, plain_60_40, spy, overlay_v3 (descriptive)
    status.json       last run summary

    python scripts/s2_forward_shadow.py run [--dry-run]       # daily (systemd timer)
    python scripts/s2_forward_shadow.py validate               # launch validation vs the frozen builder
    python scripts/s2_forward_shadow.py report                 # decision-rule evaluation
    python scripts/s2_forward_shadow.py export OUT.json        # snapshot for the review package

Monthly check = fresh per-ticker EOD pull of every active US common stock, run through
the frozen cleaner and the frozen per-ticker eligibility function
(``eodhd_breadth._ticker_signal``). The NAV ledger is recomputed from scratch from the
decision log on every run.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import eodhd_breadth as eb
import s2_forward_preregistered as pre

from firm.allocation.calendar import first_trading_day_of_month

log = logging.getLogger(__name__)

STATE_DIR = _ROOT / "data" / "research" / "s2_forward"
LISTED_EXCHANGES = {"NASDAQ", "NYSE", "AMEX", "NYSE MKT", "NYSE ARCA", "BATS", "NYSEARCA"}
HISTORY_DAYS = 760
PORTFOLIOS = ("overlay_v1", "plain_60_40", "spy", "overlay_v3")
THRESHOLD = {"overlay_v1": 0.50, "overlay_v3": 0.40}
CUT_W = pre.RULE["cut_weights"]
PLAIN_W = pre.RULE["plain_weights"]
COST = pre.COSTS["bps_per_side"] / 1e4


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------

def check_days(start: date, upto: date) -> list[date]:
    """First NYSE trading day of every month from ``start``'s month through ``upto`` (inclusive)."""
    out, y, m = [], start.year, start.month
    while True:
        d = first_trading_day_of_month(y, m)
        if d > upto:
            return out
        if d >= start:
            out.append(d)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def prior_session(d: date, sessions: list[date]) -> date | None:
    prev = [s for s in sessions if s < d]
    return prev[-1] if prev else None


def decide(breadth: float, threshold: float) -> bool:
    """True = cut state (the frozen rule: breadth strictly below the threshold)."""
    return breadth < threshold


def weights_for(cut: bool) -> dict[str, float]:
    return dict(CUT_W if cut else PLAIN_W)


# ---------------------------------------------------------------------------
# Breadth from a fresh universe pull
# ---------------------------------------------------------------------------

def _one_ticker(client, code: str, start: str, calendar: pd.DatetimeIndex, pos: dict, asofs: list[pd.Timestamp]):
    """(had_bar_by_asof, eligible_by_asof, above_by_asof) boolean arrays, or None."""
    status, data = client.get(f"eod/{code}.US", **{"from": start})
    if status != 200 or not isinstance(data, list) or not data or "date" not in data[0]:
        return None
    d = pd.DataFrame(data)
    d["date"] = pd.to_datetime(d["date"])
    d = d[eb.REQUIRED_COLS]
    for c in ("open", "close", "adjusted_close"):
        d[c] = pd.to_numeric(d[c], errors="coerce").astype("float32")
    d["volume"] = pd.to_numeric(d["volume"], errors="coerce").astype("float64")
    out = eb._ticker_signal(d, calendar, pos)
    if out is None:
        return None
    idx, elig, above, _ = out
    where = {int(i): k for k, i in enumerate(idx)}
    had, el, ab = [], [], []
    for a in asofs:
        k = where.get(int(calendar.get_loc(a)))
        had.append(k is not None)
        el.append(bool(k is not None and elig[k]))
        ab.append(bool(k is not None and above[k]))
    return np.array(had), np.array(el), np.array(ab)


def active_common_stocks(client) -> list[str]:
    status, data = client.get("exchange-symbol-list/US")
    if status != 200 or not isinstance(data, list):
        raise RuntimeError(f"symbol list failed: HTTP {status}")
    codes = sorted({x["Code"] for x in data if x.get("Type") == "Common Stock" and x.get("Exchange") in LISTED_EXCHANGES})
    return [c.replace(".", "-").replace("/", "-") for c in codes]


def pull_breadth(client, codes: list[str], calendar: pd.DatetimeIndex, asofs: list[pd.Timestamp],
                 start: str, workers: int = 10) -> dict[pd.Timestamp, dict]:
    pos = {d: i for i, d in enumerate(calendar.to_numpy())}
    n = len(asofs)
    had, elig, above = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one_ticker, client, c, start, calendar, pos, asofs) for c in codes]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as exc:  # one bad ticker must not sink the whole check
                log.warning("ticker failed: %s", exc)
                r = None
            if r is None:
                failed += 1
            else:
                had += r[0]
                elig += r[1]
                above += r[2]
            if i % 1000 == 0:
                log.info("pulled %d/%d tickers", i, len(codes))
    log.info("breadth pull: %d tickers, %d without usable data", len(codes), failed)
    return {a: {"eligible": int(elig[k]), "above": int(above[k]), "with_bar": int(had[k]),
                "breadth": float(above[k] / elig[k]) if elig[k] else float("nan")} for k, a in enumerate(asofs)}


def integrity_ok(res: dict, prev_with_bar: int) -> tuple[bool, str]:
    lo, hi = pre.INTEGRITY["eligible_count_band"]
    if not (lo <= res["eligible"] <= hi):
        return False, f"eligible_count {res['eligible']} outside [{lo}, {hi}]"
    if prev_with_bar and res["with_bar"] < pre.INTEGRITY["fresh_data_ratio_min"] * prev_with_bar:
        return False, f"only {res['with_bar']} stocks have a bar on the signal date vs {prev_with_bar} the session before"
    if not math.isfinite(res["breadth"]):
        return False, "breadth is NaN"
    return True, "ok"


# ---------------------------------------------------------------------------
# Shadow NAV (recomputed from scratch from the decision log)
# ---------------------------------------------------------------------------

def adj_frame(rows: list[dict]) -> pd.DataFrame:
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").drop_duplicates("date").set_index("date")
    ratio = d["adjusted_close"] / d["close"]
    return pd.DataFrame({"adj_open": d["open"] * ratio, "adj_close": d["adjusted_close"]})


def shadow_nav(decisions: list[dict], spy: pd.DataFrame, ief: pd.DataFrame, start_nav: float = 100.0) -> pd.DataFrame:
    """Daily NAV of the four shadow portfolios. Rebalances happen at the check day's adjusted open;
    between checks holdings (in adjusted shares) are constant. ``decisions`` carry per-portfolio
    ``cut`` flags for the overlays; plain_60_40 rebalances on the same days to 60/40; spy is held."""
    by_day = {pd.Timestamp(d["check_day"]): d for d in decisions}
    first = min(by_day) if by_day else None
    if first is None:
        return pd.DataFrame(columns=["date", *PORTFOLIOS])
    days = spy.index[spy.index >= first].intersection(ief.index)
    hold = {p: {"SPY": 0.0, "IEF": 0.0} for p in PORTFOLIOS}
    cash = {p: start_nav for p in PORTFOLIOS}     # uninvested until the first rebalance
    last_cut = {"overlay_v1": False, "overlay_v3": False}
    rows = []
    for t in days:
        px_o = {"SPY": float(spy.at[t, "adj_open"]), "IEF": float(ief.at[t, "adj_open"])}
        px_c = {"SPY": float(spy.at[t, "adj_close"]), "IEF": float(ief.at[t, "adj_close"])}
        if t in by_day:
            dec = by_day[t]
            for p in PORTFOLIOS:
                if p == "spy":
                    if t == first:
                        tgt = {"SPY": 1.0, "IEF": 0.0}
                    else:
                        continue
                elif p == "plain_60_40":
                    tgt = dict(PLAIN_W)
                else:
                    last_cut[p] = bool(dec["cut"][p])
                    tgt = weights_for(last_cut[p])
                v_open = cash[p] + sum(hold[p][k] * px_o[k] for k in hold[p])
                traded = sum(abs(tgt[k] * v_open - hold[p][k] * px_o[k]) for k in tgt)
                v_after = v_open - COST * traded
                for k in tgt:
                    hold[p][k] = tgt[k] * v_after / px_o[k]
                cash[p] = 0.0
        rows.append({"date": t, **{p: cash[p] + sum(hold[p][k] * px_c[k] for k in hold[p]) for p in PORTFOLIOS}})
    return pd.DataFrame(rows, columns=["date", *PORTFOLIOS])


def perf(nav: pd.Series, cash_ret: pd.Series) -> dict:
    r = nav.pct_change().dropna()
    ex = r - cash_ret.reindex(r.index).fillna(0.0)
    yrs = max(len(r) / 252.0, 1e-9)
    dd = float((1 - nav / nav.cummax()).max())
    cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1) if len(nav) > 1 else float("nan")
    sd = ex.std(ddof=1)
    return {"sharpe": float(ex.mean() / sd * math.sqrt(252)) if sd and sd > 0 else float("nan"),
            "cagr": cagr, "max_dd": dd, "calmar": float(cagr / dd) if dd > 0 else float("nan"), "days": len(r)}


def completed_episodes(cuts: list[bool]) -> int:
    n, run = 0, False
    for c in cuts:
        if run and not c:
            n += 1
        run = c
    return n


def evaluate_rules(months: int, bad_checks: int, episodes: int, p_ov: dict, p_pl: dict) -> dict:
    gap = p_ov["sharpe"] - p_pl["sharpe"]
    promote_parts = {
        "months>=24": months >= pre.MIN_MONTHS,
        "bad_checks<=1": bad_checks <= 1,
        "episodes>=2": episodes >= 2,
        "maxDD<=plain": p_ov["max_dd"] <= p_pl["max_dd"],
        "calmar>plain": p_ov["calmar"] > p_pl["calmar"],
        "gap>=-0.05": gap >= -0.05,
    }
    reject = months >= pre.MIN_MONTHS and episodes >= 2 and (p_ov["max_dd"] > p_pl["max_dd"] or gap < -0.05)
    if all(promote_parts.values()):
        outcome = pre.PROMOTE["outcome"]
    elif reject:
        outcome = pre.REJECT["outcome"]
    elif months >= pre.MAX_MONTHS:
        outcome = "INCONCLUSIVE_CLOSE"
    else:
        outcome = pre.OTHERWISE["outcome"]
    return {"outcome": outcome, "sharpe_gap": gap, "promote_conditions": promote_parts}


# ---------------------------------------------------------------------------
# Ledger I/O, notification
# ---------------------------------------------------------------------------

def load_log(state: Path) -> list[dict]:
    """Every record ever written (append-only audit log)."""
    p = state / "decisions.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def load_decisions(state: Path) -> list[dict]:
    """The governing record per check day: the latest one, so a retried anomaly supersedes itself."""
    latest: dict[str, dict] = {}
    for rec in load_log(state):
        latest[rec["check_day"]] = rec
    return [latest[k] for k in sorted(latest)]


def append_decision(state: Path, rec: dict) -> None:
    state.mkdir(parents=True, exist_ok=True)
    with open(state / "decisions.jsonl", "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def notify(severity: str, message: str, **extra) -> None:
    try:
        from dotenv import dotenv_values

        from firm.live.notifications import _post_webhook
        url = os.environ.get("ALERT_WEBHOOK_URL") or dotenv_values(_ROOT / ".env").get("ALERT_WEBHOOK_URL")
        if url:
            _post_webhook(url, {"kind": "s2_forward_shadow", "severity": severity, "message": message,
                                "timestamp": datetime.now(UTC).isoformat(), **extra}, timeout=10)
        else:
            log.warning("no ALERT_WEBHOOK_URL; not notifying: %s", message)
    except Exception as exc:  # a failed alert must never fail the ledger
        log.warning("notification failed: %s", exc)


def client_from_env(max_rps: float = 12.0):
    from fetch_eodhd_prices import Client, _key
    return Client(_key(), max_rps=max_rps)


def fetch_adj(client, code: str, start: str) -> pd.DataFrame:
    status, data = client.get(f"eod/{code}", **{"from": start})
    if status != 200 or not isinstance(data, list) or not data:
        raise RuntimeError(f"EODHD {code}: HTTP {status}")
    return adj_frame(data)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def rebuild_ledger(state: Path, client) -> dict:
    decisions = load_decisions(state)
    spy, ief, bil = (fetch_adj(client, f"{s}.US", "2026-09-01") for s in ("SPY", "IEF", "BIL"))
    nav = shadow_nav(decisions, spy, ief, pre.START["nav_start"])
    state.mkdir(parents=True, exist_ok=True)
    nav.to_csv(state / "nav.csv", index=False)
    return {"nav": nav, "bil": bil, "last_session": str(spy.index.max().date())}


def cmd_run(args) -> int:
    state = Path(args.state_dir)
    client = client_from_env()
    today = datetime.now(UTC).date()
    market_today = today
    start = date.fromisoformat(pre.START["first_check_day"])
    spy_hist = fetch_adj(client, "SPY.US", (start - timedelta(days=HISTORY_DAYS)).isoformat())
    sessions = [d.date() for d in spy_hist.index]
    decisions = load_decisions(state)
    done = {d["check_day"] for d in decisions if d["status"] in ("ok", "late_backfill")}
    due = [d for d in check_days(start, market_today) if d.isoformat() not in done]
    for f in due:
        asof = prior_session(f, sessions)
        if asof is None or (f > market_today):
            continue
        late = f < market_today
        log.info("check %s (signal as of %s)%s", f, asof, " [late backfill]" if late else "")
        codes = active_common_stocks(client)
        calendar = pd.DatetimeIndex(spy_hist.index)
        prev = prior_session(asof, sessions)
        asofs = [pd.Timestamp(asof)] + ([pd.Timestamp(prev)] if prev else [])
        from_date = (asof - timedelta(days=HISTORY_DAYS)).isoformat()
        res = pull_breadth(client, codes, calendar, asofs, from_date, workers=10)
        cur, prv = res[asofs[0]], (res[asofs[1]] if prev else {"with_bar": 0})
        ok, why = integrity_ok(cur, prv["with_bar"])
        last = next((d for d in reversed(decisions) if d["status"] in ("ok", "late_backfill")
                     and d["check_day"] < f.isoformat()), None)
        if ok:
            cuts = {"overlay_v1": decide(cur["breadth"], 0.50), "overlay_v3": decide(cur["breadth"], 0.40)}
            status = "late_backfill" if late else "ok"
        else:
            cuts = last["cut"] if last else {"overlay_v1": False, "overlay_v3": False}
            status = "anomaly"
        rec = {"check_day": f.isoformat(), "signal_asof": asof.isoformat(), "status": status, "reason": why,
               "breadth": cur["breadth"], "eligible": cur["eligible"], "with_bar": cur["with_bar"],
               "cut": cuts, "weights_v1": weights_for(cuts["overlay_v1"]),
               "forward_fingerprint": pre.bars_fingerprint(), "computed_at": datetime.now(UTC).isoformat()}
        if args.dry_run:
            print(json.dumps(rec, indent=1, default=str))
            continue
        append_decision(state, rec)
        decisions = [d for d in decisions if d["check_day"] != rec["check_day"]] + [rec]
        decisions.sort(key=lambda d: d["check_day"])
        sev = "info" if ok else "error"
        notify(sev, f"S2 shadow check {f}: breadth {cur['breadth']:.1%} ({cur['eligible']} eligible) -> "
                    f"{'CUT (SPY 30/IEF 70)' if cuts['overlay_v1'] else 'plain 60/40'} [{status}] {'' if ok else why}")
    if args.dry_run:
        return 0
    led = rebuild_ledger(state, client)
    nav = led["nav"]
    summary = {"run_at": datetime.now(UTC).isoformat(), "last_session": led["last_session"],
               "checks": len(load_decisions(state)),
               "nav_last": None if nav.empty else {p: float(nav[p].iloc[-1]) for p in PORTFOLIOS}}
    (state / "status.json").write_text(json.dumps(summary, indent=1))
    log.info("ledger rebuilt: %s", summary)
    return 0


def cmd_validate(args) -> int:
    """Launch validation: the live pipeline vs the frozen builder's breadth on past session dates."""
    client = client_from_env()
    ref = pd.read_parquet(args.reference).set_index("date")
    spy = fetch_adj(client, "SPY.US", (date(2026, 9, 29) - timedelta(days=HISTORY_DAYS)).isoformat())
    cal = pd.DatetimeIndex(spy.index)
    month_ends = [pd.Timestamp(d) for d in ("2026-04-30", "2026-05-29", "2026-06-30", "2026-07-31", "2026-08-31")]
    recent = [d for d in cal if d <= pd.Timestamp("2026-09-29")][-6:]
    asofs = sorted({d for d in month_ends + list(recent) if d in cal})
    codes = active_common_stocks(client)
    res = pull_breadth(client, codes, cal, asofs, (asofs[0] - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d"))
    rows, ok_all = [], True
    for a in asofs:
        r, f = res[a], ref.loc[a]
        db = abs(r["breadth"] - float(f["pct_above_200sma"])) * 100
        de = abs(r["eligible"] - float(f["eligible_count"])) / float(f["eligible_count"]) * 100
        ok = db <= 0.5 and de <= 3.0
        ok_all &= ok
        rows.append({"asof": str(a.date()), "live_breadth": r["breadth"], "frozen_breadth": float(f["pct_above_200sma"]),
                     "diff_pp": db, "live_eligible": r["eligible"], "frozen_eligible": int(f["eligible_count"]),
                     "eligible_diff_pct": de, "pass": ok})
    out = {"validated_at": datetime.now(UTC).isoformat(), "n_dates": len(rows), "all_pass": ok_all,
           "rule": pre.INTEGRITY["launch_validation"]["rule"], "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"all_pass": ok_all, "n_dates": len(rows),
                      "max_diff_pp": max(r["diff_pp"] for r in rows),
                      "max_eligible_diff_pct": max(r["eligible_diff_pct"] for r in rows)}))
    return 0 if ok_all else 2


def cmd_report(args) -> int:
    state = Path(args.state_dir)
    decisions = load_decisions(state)
    nav = pd.read_csv(state / "nav.csv", parse_dates=["date"]).set_index("date")
    client = client_from_env()
    bil = fetch_adj(client, "BIL.US", "2026-09-01")
    cash_ret = bil["adj_close"].pct_change()
    pf = {p: perf(nav[p], cash_ret) for p in PORTFOLIOS}
    good = [d for d in decisions if d["status"] in ("ok", "late_backfill")]
    bad = [d for d in decisions if d["status"] not in ("ok", "late_backfill")]
    months = len(decisions)
    eps = completed_episodes([bool(d["cut"]["overlay_v1"]) for d in good])
    rules = evaluate_rules(months, len(bad), eps, pf["overlay_v1"], pf["plain_60_40"])
    out = {"as_of": str(nav.index.max().date()), "months_checked": months, "bad_checks": len(bad),
           "completed_episodes": eps, "performance": pf, "rules": rules, "forward_fingerprint": pre.bars_fingerprint(),
           "descriptive_note": "no decision before 24 months; overlay_v3 is descriptive only"}
    print(json.dumps(out, indent=1, default=float))
    return 0


def cmd_export(args) -> int:
    state = Path(args.state_dir)
    decisions = load_decisions(state)
    nav = pd.read_csv(state / "nav.csv") if (state / "nav.csv").exists() else pd.DataFrame()
    Path(args.out).write_text(json.dumps({"exported_at": datetime.now(UTC).isoformat(),
                                          "forward_fingerprint": pre.bars_fingerprint(),
                                          "decisions": decisions, "nav": nav.to_dict("records")}, indent=1, default=str))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default=str(STATE_DIR))
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dry-run", action="store_true")
    v = sub.add_parser("validate")
    v.add_argument("--reference", required=True, help="frozen builder output (breadth_full.parquet)")
    v.add_argument("--out", required=True)
    sub.add_parser("report")
    e = sub.add_parser("export")
    e.add_argument("out")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if pre.DRAFT:
        raise SystemExit("forward-test pre-registration is DRAFT")
    try:
        return {"run": cmd_run, "validate": cmd_validate, "report": cmd_report, "export": cmd_export}[args.cmd](args)
    except Exception as exc:
        log.exception("s2 forward shadow %s failed", args.cmd)
        if args.cmd == "run":
            notify("error", f"S2 shadow forward run FAILED: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
