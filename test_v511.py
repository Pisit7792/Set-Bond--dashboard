"""ทดสอบเฟส v5.11: set_context + set_swing + crypto — python3 test_v511.py"""
import math
import numpy as np
import pandas as pd

import crypto as CR
import set_context as CX
import set_swing as SW

ok = fail = 0
def check(name, cond, detail=""):
    global ok, fail
    if cond: ok += 1; print(f"PASS  {name} {detail}")
    else: fail += 1; print(f"FAIL  {name} {detail}")

idx = pd.bdate_range("2022-06-01", periods=760)
rng = np.random.default_rng(9)

def ohlc(close, volmul=1.0):
    close = pd.Series(close, index=idx)
    o = close.shift(1).fillna(close.iloc[0])
    h = pd.concat([o, close], axis=1).max(axis=1) + 0.6
    l = pd.concat([o, close], axis=1).min(axis=1) - 0.6
    v = pd.Series(rng.integers(2_000_000, 6_000_000, len(idx)).astype(float)
                  * volmul, index=idx)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": close,
                         "Volume": v})

# ---------- Market Context v1.0 ----------
set_c = pd.Series(1300 + np.arange(760) * 0.55 + rng.normal(0, 3, 760), index=idx)
vix = pd.Series(np.r_[np.linspace(30, 14, 700), np.full(60, 13.0)], index=idx)
thb = pd.Series(np.full(760, 34.0), index=idx)
spx = pd.Series(4000 + np.arange(760) * 2.0, index=idx)
eem = pd.Series(40 + np.arange(760) * 0.005, index=idx)
mc = CX.market_context(set_c, vix, thb, spx, eem)
check("mc_ok", mc["ok"])
check("mc_parts", mc["parts"]["regime(±35)"] == 35 and
      mc["parts"]["momentum6M(±25)"] == 25, str(mc["parts"]))
check("mc_flow_na_zero", mc["flow_na"] and mc["parts"]["flowCMF(±20)"] == 0)
check("mc_macro_pos", mc["parts"]["macro(±20)"] >= 10,
      f"press={mc['press']} votes={mc['press_votes']}")
check("mc_zone_buy", mc["score"] >= 60 and "BUY" in mc["zone"],
      f"score={mc['score']}")
check("mc_align_text", len(mc["align"]) > 10)
mc_dn = CX.market_context(pd.Series(1700 - np.arange(760) * 0.6, index=idx),
                          pd.Series(np.linspace(15, 45, 760), index=idx),
                          pd.Series(np.linspace(33, 36, 760), index=idx),
                          pd.Series(5000 - np.arange(760) * 2.0, index=idx), eem)
check("mc_zone_sell", mc_dn["score"] <= -60, f"score={mc_dn['score']}")
cal_idx = pd.bdate_range("2024-03-01", periods=520)
mc_cal = CX.market_context(pd.Series(1500 + np.arange(520) * 0.3, index=cal_idx))
check("mc_cal_msci_feb", any("MSCI" in x for x in mc_cal["calendar"]),
      str(mc_cal["calendar"]))

# ---------- Stock Context v1.1 ----------
trend = 20 + np.arange(760) * 0.05 + 2.0 * np.sin(np.arange(760) / 7)
stock = ohlc(trend)
stock.iloc[-1, stock.columns.get_loc("Close")] = float(stock["High"].iloc[-22:-1].max()) + 1.5
stock.iloc[-1, stock.columns.get_loc("High")] = stock["Close"].iloc[-1] + 0.6
stock.iloc[-1, stock.columns.get_loc("Volume")] = float(stock["Volume"].iloc[-21:-1].mean()) * 3
bench = set_c
sc = CX.stock_context(stock, bench, "PTT", CX.StockCtxParams())
check("sc_ok", sc["ok"])
check("sc_profile_matched", sc["prof"]["matched"] and sc["prof"]["sector"] == "Energy")
check("sc_regime_up", sc["regime"] == "UP")
check("sc_conf_high", sc["conf_l"] >= 55, f"confL={sc['conf_l']}")
check("sc_status_met_or_wait", "context met" in sc["status"]
      or "รอเบรก" in sc["status"], sc["status"])
check("sc_rows_21", len(sc["rows"]) == 21, f"n={len(sc['rows'])}")
sc_surv = CX.stock_context(stock, bench, "PTT", surv_flag=True)
check("sc_waterfall_surv", sc_surv["status"].startswith("Surveillance"))
p_liq = CX.StockCtxParams(liq_min_m=1e6, use_profile=False)
sc_liq = CX.stock_context(stock, bench, "XXXX", p_liq)
check("sc_waterfall_liq", "สภาพคล่อง" in sc_liq["status"], sc_liq["status"])
check("sc_unmatched_neutral", CX.profile_of("XXXX")["matched"] is False)

# ---------- Swing v5.11 ----------
p = SW.SwingParams()
check("sw_defaults", p.risk_pc == 0.5 and p.conf_min == 55 and not p.use_tp1
      and p.use_tick and p.max_trades_m == 6)
check("sw_tick_band", SW.set_tick(1.5) == 0.01 and SW.set_tick(7) == 0.05
      and SW.set_tick(150) == 0.5 and SW.set_tick(450) == 2.0)
check("sw_rnd", abs(SW.rnd_set(101.13) - 101.0) < 1e-9
      and abs(SW.rnd_set(3.117) - 3.12) < 1e-9)
eff = SW.effective("DELTA", p)
check("sw_eff_profile", eff["matched"] and eff["vt_tgt"] == 2.8
      and eff["liq_max"] == 3.0, str(eff))
fr = SW.compute_frame(stock, bench, "PTT", p)
check("sw_frame_cols", all(c_ in fr.columns for c_ in
      ["regime_up", "bos", "conf_l", "long_cond", "sl_dist", "lot"]))
nL = int(fr["long_cond"].sum())
check("sw_longcond_fires", nL >= 1, f"n={nL}")
stt = SW.state_today(fr, p)
check("sw_state", stt["regime"] == "UP" and len(stt["checklist"]) == 9
      and stt["lot"] in (50, 100))
# ชุด backtest: ฝังสัญญาณกลางซีรีส์ (ไม่ใช่แท่งสุดท้าย) สองจุด
stock_bt = ohlc(trend)
for pos in (-200, -60):
    j = len(stock_bt) + pos
    brk = float(stock_bt["High"].iloc[j-21:j-1].max()) + 1.5
    stock_bt.iloc[j, stock_bt.columns.get_loc("Close")] = brk
    stock_bt.iloc[j, stock_bt.columns.get_loc("High")] = brk + 0.6
    stock_bt.iloc[j, stock_bt.columns.get_loc("Volume")] = \
        float(stock_bt["Volume"].iloc[j-21:j-1].mean()) * 3
fr_bt = SW.compute_frame(stock_bt, bench, "PTT", p)
check("sw_bt_signals_mid", int(fr_bt["long_cond"].iloc[:-5].sum()) >= 1,
      f"n={int(fr_bt['long_cond'].sum())}")
bt = SW.backtest(fr_bt, p, 1_000_000)
check("sw_bt_trades", bt["n"] >= 1, f"n={bt['n']}")
if bt["n"]:
    td = bt["trades"]
    first_entry = pd.Timestamp(td.iloc[0]["เข้า"])
    sig_days = fr_bt.index[fr_bt["long_cond"]]
    check("sw_nextbar_entry", any(fr_bt.index.get_loc(first_entry)
          - fr_bt.index.get_loc(s) == 1 for s in sig_days if s < first_entry))
    i_e = fr_bt.index.get_loc(first_entry)
    check("sw_entry_open", abs(float(td.iloc[0]["ราคาเข้า"])
          - float(fr_bt["Open"].iloc[i_e])) < 0.01)
    check("sw_qty_lot", all(int(q) % 50 == 0 for q in td["หุ้น"]))
    check("sw_R_sane", td["R"].min() > -3.0, f"minR={td['R'].min()}")
    lo, hi = bt["ci"]
    check("sw_stats", 0 <= lo <= hi <= 1 and np.isfinite(bt["net_thb"]))
# regime exit off + surv blocks
p2 = SW.SwingParams(surv_flag=True)
fr2 = SW.compute_frame(stock, bench, "PTT", p2)
check("sw_surv_blocks", int(fr2["long_cond"].sum()) == 0)

# ---------- Crypto Toolkit v6 ----------
days = pd.date_range("2020-01-01", periods=2300, freq="D")
r2 = np.random.default_rng(3)
btc = pd.Series(20000 * np.exp(np.cumsum(r2.normal(0.0012, 0.03, 2300))), index=days)
dfb = pd.DataFrame({"Open": btc.shift(1).fillna(btc.iloc[0]),
                    "High": btc * 1.01, "Low": btc * 0.99, "Close": btc})
frb = CR.compute(dfb)
stb = CR.state(frb, is_btc=True)
check("cr_keys", all(k in stb for k in ("regime", "mayer", "rsi", "ann_vol",
      "dd_ath", "halv_days", "pi_gap")))
check("cr_mayer_calc", abs(stb["mayer"] - float(btc.iloc[-1]
      / btc.rolling(200).mean().iloc[-1])) < 0.01)
check("cr_dd_le0", stb["dd_ath"] <= 0 + 1e-9)
check("cr_vol_pos", stb["ann_vol"] and stb["ann_vol"] > 0)
check("cr_halv_days", stb["halv_days"] and stb["halv_days"] > 700)
alt = CR.state(frb, is_btc=False)
check("cr_alt_no_pi", alt["pi_gap"] is None and alt["halv_days"] is None)
# pi cross detection on crafted series
x = pd.Series(np.r_[np.full(10, 100.0), np.full(10, 500.0)],
              index=pd.date_range("2020-01-01", periods=20, freq="D"))
dfx = pd.DataFrame({"Open": x, "High": x, "Low": x, "Close": x})
frx = CR.compute(dfx, CR.CryptoParams(pi_fast=3, pi_slow=10, reg_len=5,
                                      vol_len=5))
check("cr_pi_cross_fires", bool(frx["pi_cross"].any()))
check("cr_tables", len(CR.EVIDENCE_TWO_SIDES) >= 5 and len(CR.BENCH_BEAR) >= 3
      and len(CR.RISKS) >= 4 and len(CR.ALT_CAVEAT) > 40)

print(f"\n== {ok} passed, {fail} failed ==")
raise SystemExit(1 if fail else 0)
