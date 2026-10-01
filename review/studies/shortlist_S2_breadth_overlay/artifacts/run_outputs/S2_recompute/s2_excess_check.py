"""Post-hoc check: recompute A2 (DSR) and A3 (placebo) on the EXCESS-vs-BM2 basis,
matching the primary harness's (self-documented) convention, to see whether that
convention alone explains the A2/A3 mismatch against my raw-Sharpe convention.
Does not change s2_recompute.py's own (already-final) numbers; this is a
supplementary diagnostic only, run after my own numbers were final.
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-local-store-git-ai-trading-system/c4c796e1-061f-42c7-9fa9-255931a43502/scratchpad/runs/S2_recompute")
import s2_recompute as s2
import numpy as np
import pandas as pd

calendar_full = s2.equity_calendar()
window_mask_full = (calendar_full >= pd.Timestamp(s2.WINDOW_START)) & (calendar_full <= pd.Timestamp(s2.DATA_END))
calendar = calendar_full[window_mask_full]
n = len(calendar)

spy_df = s2.load_etf("SPY", calendar_full)
ief_df = s2.load_etf("IEF", calendar_full)
bil_df = s2.load_etf("BIL", calendar_full)
vfitx_df = s2.load_etf("VFITX", calendar_full, asset="nav")
spy_f = s2.build_calendar_frame("spy", spy_df, calendar_full)
ief_f = s2.build_calendar_frame("ief", ief_df, calendar_full)
bil_f = s2.build_calendar_frame("bil", bil_df, calendar_full)
vfitx_f = s2.build_calendar_frame("vfitx", vfitx_df, calendar_full)
spy_bps_full = s2.adv20_bps(spy_df, calendar_full)
ief_bps_full = s2.adv20_bps(ief_df, calendar_full)
bil_bps_full = s2.adv20_bps(bil_df, calendar_full)
vfitx_ret_full = vfitx_f["vfitx_adjclose"].pct_change().to_numpy()
dtb3_ret_full = s2.dtb3_accrual_returns(calendar_full)

spy_leg_full = s2.build_spy_leg(spy_f["spy_adjclose"], spy_f["spy_adjopen"], spy_bps_full, calendar_full)
ief_leg_full = s2.build_leg(ief_f["ief_adjclose"], ief_f["ief_adjopen"], ief_bps_full, vfitx_ret_full, s2.IEF_HANDOFF, calendar_full, None)
bil_leg_full = s2.build_leg(bil_f["bil_adjclose"], bil_f["bil_adjopen"], bil_bps_full, dtb3_ret_full, s2.BIL_HANDOFF, calendar_full, None)
wm = np.asarray(window_mask_full)
spy_leg = {k: v[wm] for k, v in spy_leg_full.items()}
ief_leg = {k: v[wm] for k, v in ief_leg_full.items()}
bil_leg = {k: v[wm] for k, v in bil_leg_full.items()}

rebal_mask = s2.rebal_mask_from_calendar(calendar)
bm2 = s2.simulate_two_asset(rebal_mask, np.full(rebal_mask.sum(), s2.CORE_SPY), spy_leg, ief_leg)
breadth = pd.read_parquet(s2.OUT / "breadth_mine.parquet")
dest_leg_for = {"IEF": ief_leg, "BIL": bil_leg}

trial_excess_sharpes = []
per_variant = {}
for vname, spec in s2.VARIANTS.items():
    on_state = s2.build_overlay_decision(breadth, calendar, rebal_mask, spec["measure"], spec["threshold"])
    w_seq = np.where(on_state, s2.SPY_WEIGHT_CUT, s2.CORE_SPY)
    dest_leg = dest_leg_for[spec["destination"]]
    v_ret = s2.simulate_two_asset(rebal_mask, w_seq, spy_leg, dest_leg, cost_mult=1.0)
    excess = v_ret[1:] - bm2[1:]
    real_excess_sharpe = s2.ann_sharpe(excess)
    trial_excess_sharpes.append(s2.period_sharpe(excess))
    placebo_excess = s2.placebo_sharpes.__wrapped__ if False else None
    # recompute placebo but track EXCESS-vs-BM2 sharpe per draw (not raw variant sharpe)
    rng = np.random.default_rng(s2.SEED + 1 + hash(vname) % 1000)
    daily_real = s2.daily_state_from_monthly(rebal_mask, on_state)
    rebal_positions = np.flatnonzero(rebal_mask)
    n_draws = 500
    placebo_excess_sh = np.empty(n_draws)
    for d in range(n_draws):
        daily_perm = s2.permute_blocks(daily_real, 63, rng)
        monthly_perm_state = daily_perm[rebal_positions]
        w_seq_p = np.where(monthly_perm_state, s2.SPY_WEIGHT_CUT, s2.CORE_SPY)
        r = s2.simulate_two_asset(rebal_mask, w_seq_p, spy_leg, dest_leg, cost_mult=1.0)
        placebo_excess_sh[d] = s2.ann_sharpe(r[1:] - bm2[1:])
    p95 = float(np.percentile(placebo_excess_sh, 95.0))
    a3_excess = real_excess_sharpe > p95
    per_variant[vname] = {
        "real_excess_sharpe": real_excess_sharpe, "placebo_p95_excess": p95, "A3_excess_convention": a3_excess,
    }
    print(vname, "real_excess_sharpe=%.4f placebo_p95_excess=%.4f A3(excess-convention)=%s" %
          (real_excess_sharpe, p95, a3_excess))

# DSR on excess-vs-BM2, prior_trials=206, matching primary's convention
from firm.eval.overfitting import deflated_sharpe
trial_excess_sharpes = np.array(trial_excess_sharpes)
print("trial (period) excess sharpes:", trial_excess_sharpes)
for vname, spec in s2.VARIANTS.items():
    on_state = s2.build_overlay_decision(breadth, calendar, rebal_mask, spec["measure"], spec["threshold"])
    w_seq = np.where(on_state, s2.SPY_WEIGHT_CUT, s2.CORE_SPY)
    dest_leg = dest_leg_for[spec["destination"]]
    v_ret = s2.simulate_two_asset(rebal_mask, w_seq, spy_leg, dest_leg, cost_mult=1.0)
    excess = v_ret[1:] - bm2[1:]
    dsr_excess = float(deflated_sharpe(excess, trial_excess_sharpes, prior_trials=206))
    print(vname, "DSR(excess-convention)=%.5f" % dsr_excess)
    per_variant[vname]["dsr_excess"] = dsr_excess
