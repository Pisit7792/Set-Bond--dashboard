"""ทดสอบ gold.py (พอร์ต XAU-RTP v6.4): python3 test_gold.py"""
import math

import numpy as np
import pandas as pd

import gold as G

ok = fail = 0
def check(name, cond, detail=""):
    global ok, fail
    if cond: ok += 1; print(f"PASS  {name} {detail}")
    else: fail += 1; print(f"FAIL  {name} {detail}")

# ---------- indicators ----------
up = pd.Series(np.arange(100, 200, dtype=float))
check("rsi_uptrend_high", float(G.rsi_wilder(up).iloc[-1]) > 95)
check("er_straightline~1", float(G.efficiency_ratio(up, 20).iloc[-1]) > 0.99)
check("pctrank_top", abs(float(G.pct_rank(up, 60).iloc[-1]) - 100) < 1e-9)
ohlc0 = pd.DataFrame({"Open": up, "High": up + 2, "Low": up - 2, "Close": up})
check("atr_positive", float(G.atr_wilder(ohlc0).iloc[-1]) > 0)

# ---------- synthetic XAU: อัปเทรนด์ + ย่อเป็นจังหวะ (บังคับให้เกิด pullback) ----------
rng = np.random.default_rng(5)
n = 900
idx = pd.bdate_range("2022-01-03", periods=n)
base = 1800 + np.arange(n) * 0.9
wave = 16 * np.sin(np.arange(n) / 5.5)
close = base + wave + rng.normal(0, 1.2, n)
open_ = np.r_[close[0], close[:-1]]              # เปิด = ปิดเมื่อวาน (gap=0)
high = np.maximum(open_, close) + 3.0
low = np.minimum(open_, close) - 3.0
xau = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close},
                   index=idx)

p = G.GoldParams()
fr = G.compute_frame(xau, dxy_close=None, p=p)
check("frame_cols", all(c in fr.columns for c in
      ["regime_up", "trig_l", "long_cond", "stop_dist", "score_l"]))
tail = fr.iloc[300:]
check("regime_up_in_trend", tail["regime_up"].mean() > 0.7,
      f"={tail['regime_up'].mean():.2f}")
check("pullback_triggers_exist", int(tail["trig_l"].sum()) >= 5,
      f"n={int(tail['trig_l'].sum())}")
nL = int(fr["long_cond"].sum())
check("long_cond_fires", nL >= 3, f"n={nL}")
check("no_dxy_gate_passes", fr["dxy_ok_l"].all())
check("gap_zero_ok", fr["gap_ok_l"].iloc[300:].all())

# DXY veto: DXY ขาขึ้นยืนยัน → long_cond ต้องหายทั้งช่วง
dxy_up = pd.Series(np.linspace(95, 120, n), index=idx)
fr_v = G.compute_frame(xau, dxy_close=dxy_up, p=p)
lateL = int(fr_v["long_cond"].iloc[120:].sum())
check("dxy_veto_blocks_longs", lateL == 0 and not fr_v["dxy_ok_l"].iloc[200:].any(),
      f"longs_after_confirm={lateL}")

# ---------- state_today ----------
stt = G.state_today(fr, p)
check("state_keys", all(k in stt for k in
      ("regime", "status", "checklist", "score_l", "plan")))
check("state_checklist_tiers", {c["Tier"] for c in stt["checklist"]} <= {"A", "B"})
check("state_regime_up", stt["regime"] == "UP")

# ---------- backtest ----------
bt = G.backtest(fr, p, equity0=10000.0)
check("bt_trades>=2", bt["n"] >= 2, f"n={bt['n']}")
td = bt["trades"]
# entry ต้องเป็น open ของแท่งถัดจากสัญญาณ (กัน look-ahead)
sig_days = list(fr.index[fr["long_cond"] | fr["short_cond"]])
first_entry = pd.Timestamp(td.iloc[0]["เข้า"])
prev_ok = any(fr.index.get_loc(first_entry) - fr.index.get_loc(s) == 1
              for s in sig_days if s < first_entry)
check("entry_next_bar_open", prev_ok)
epx = float(td.iloc[0]["ราคาเข้า"])
i_e = fr.index.get_loc(first_entry)
check("entry_price_is_open", abs(epx - float(fr["Open"].iloc[i_e])) < 0.01,
      f"{epx} vs {fr['Open'].iloc[i_e]:.2f}")
check("R_not_catastrophic", td["R"].min() > -3.0, f"minR={td['R'].min()}")
longs = td[td["ทิศ"] == "LONG"]
check("swap_long_negative", (longs["swap$ (ประมาณ)"] <= 0).all() if len(longs)
      else True)
check("net_after_swap_math",
      float(abs(td["สุทธิหลัง swap$"]
                - (td["กำไร$ (หัก spread)"] + td["swap$ (ประมาณ)"])).max()) < 0.02)
check("equity_finite", np.isfinite(float(bt["equity"].iloc[-1])))
check("stats_present", all(k in bt for k in
      ("win_rate", "ci", "pf_after_swap", "max_dd", "psr")))
lo, hi = bt["ci"]
check("wilson_used", 0 <= lo <= hi <= 1)

lvl, msg = G.validation_verdict(bt)
if bt["n"] < 30:
    check("verdict_small_n_fail", lvl == "fail" and "30" in msg, msg[:60])
else:
    check("verdict_has_target", "1.5" in msg)
check("verdict_zero", G.validation_verdict({"n": 0})[0] == "fail")

print(f"\n== {ok} passed, {fail} failed ==")
raise SystemExit(1 if fail else 0)
