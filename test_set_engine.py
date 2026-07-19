"""ทดสอบ set_engine แบบ offline: python3 test_set_engine.py"""
import math
import numpy as np
import pandas as pd

import set_engine as S
import engine as E

ok = fail = 0
def check(name, cond, detail=""):
    global ok, fail
    if cond: ok += 1; print(f"PASS  {name} {detail}")
    else: fail += 1; print(f"FAIL  {name} {detail}")

# universe
u = S.to_yahoo(S.SET100_H2_2026)
check("universe_100", len(S.SET100_H2_2026) == 100, f"n={len(S.SET100_H2_2026)}")
check("universe_unique", len(set(S.SET100_H2_2026)) == 100)
check("h2_2026_ins_present", all(x in S.SET100_H2_2026 for x in
      ["MRDIYT","THAI","THCOM","WHAUP","BCP","TFG","GFPT","PTG","STECON"]))
check("h2_2026_outs_absent", all(x not in S.SET100_H2_2026 for x in
      ["JAS","JMART","SISB","SJWD"]))
check("to_yahoo_bk", u[0].endswith(".BK") and S.to_yahoo(["THB=X"]) == ["THB=X"])

# synthetic data
idx = pd.bdate_range("2022-01-01", periods=760)
def make(seed, drift):
    r = np.random.default_rng(seed)
    close = 10*np.exp(np.cumsum(r.normal(drift, 0.015, len(idx))))
    return pd.DataFrame({"Open": close*(1+r.normal(0,0.004,len(idx))),
        "High": close*(1+np.abs(r.normal(0,0.006,len(idx)))),
        "Low": close*(1-np.abs(r.normal(0,0.006,len(idx)))),
        "Close": close,
        "Volume": r.integers(1_000_000,9_000_000,len(idx)).astype(float)}, index=idx)
bench = make(1, 0.0003)["Close"]
prices = {"AAA.BK": make(2, 0.0006), "BBB.BK": make(3, -0.0002),
          "CCC.BK": make(4, 0.0004), "DDD.BK": make(5, 0.0001)}

# indicators
check("trend_finite", math.isfinite(S.trend_score(bench)))
check("mom61_finite", math.isfinite(S.momentum_6_1(bench)))
check("resmom_finite", math.isfinite(S.residual_momentum(prices["AAA.BK"]["Close"], bench)))
check("annvol_pos", S.ann_vol(bench) > 0)
check("turnover_pos", S.median_turnover_thb(prices["AAA.BK"]) > 0)
check("rsi_range", 0 <= float(S.rsi_series(bench).iloc[-1]) <= 100)

# scoreboard
board = S.build_scoreboard(prices, bench, 5e6)
check("board_rows", len(board) == 4)
check("board_composite_any", board["composite"].notna().any())
check("board_gate_bool", board["liq_pass"].dtype == bool)
check("board_minfactors", (board.loc[board["n_factors"] < 3, "composite"].isna()).all())

# cost
cm = S.ThaiCost()
check("cost_side", 0.1 < cm.per_side_pct() < 0.6, f"={cm.per_side_pct():.3f}")
check("cost_rt_2x", abs(cm.round_trip_pct() - 2*cm.per_side_pct()) < 1e-12)

# validation: DSR <= PSR, verdicts
rng = np.random.default_rng(7)
rets = rng.normal(0.0005, 0.012, 300)
psr = E.probabilistic_sharpe_ratio(pd.Series(rets))
dsr = S.deflated_sharpe(rets, 50)
check("psr_range", 0 <= psr <= 1)
check("dsr_range", 0 <= dsr <= 1)
check("dsr_le_psr", dsr <= psr + 1e-9, f"dsr={dsr:.3f} psr={psr:.3f}")
check("dsr_trials1_eq_psr", abs(S.deflated_sharpe(rets, 1) - psr) < 1e-9)
check("verdict_fail", S.sample_verdict(9)[0] == "fail")
check("verdict_warn", S.sample_verdict(60)[0] == "warn")
check("verdict_ok", S.sample_verdict(150)[0] == "ok")

# backtest
res = S.sma_cross_backtest(prices["AAA.BK"]["Close"], 20, 50,
                           cm.per_side_pct(), S.regime_series(bench))
check("bt_runs", res is not None)
check("bt_equity_finite", math.isfinite(float(res["equity"].iloc[-1])))
st = S.trade_stats(res["trades"])
check("bt_stats_n", "n" in st and st["n"] >= 0)
# look-ahead guard: สัญญาณวันแรกต้องเป็น 0 (shift แล้ว)
res2 = S.sma_cross_backtest(prices["AAA.BK"]["Close"], 5, 30, 0.0)
check("bt_first_flat", abs(float(res2["daily"].iloc[0])) < 1e-12)

# rrg + tom
rr = S.compute_rrg(prices, bench)
check("rrg_dict", isinstance(rr, dict) and all(set(d.columns)=={"x","y"} for d in rr.values()))
ts = S.tom_stats(bench)
check("tom_ok", ts.get("ok") is True and ts["tom_n"] > 0)
check("quadrant", S.quadrant_name(101, 101).startswith("Leading"))

print(f"\n== {ok} passed, {fail} failed ==")
raise SystemExit(1 if fail else 0)
