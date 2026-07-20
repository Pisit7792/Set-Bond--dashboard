# -*- coding: utf-8 -*-
"""
set_swing.py — พอร์ต SET SWING TREND-REGIME BREAKOUT v5.11 (Daily)

ซื่อตรงต่อต้นฉบับ:
- ค่าตั้งต้นทุกตัว = Pine v5.11 (Long only, risk 0.5%, SMA200+ดัชนี gate,
  BOS 20, score≥55, vol-shock 90, ceiling 30/5, gap 1.5 ATR, stop 2 ATR,
  chandelier 22/3, TP1 OFF, tick-round ON, VT>80th, DD cut 10%→×0.5,
  แพ้ติด≥2→×0.6, beta>1.3 trim, kill 6%/แพ้ติด 5, เพดาน 6 เทรด/เดือน,
  โปรไฟล์ SET100 ON, board lot อัตโนมัติ 50/100)
- ตัวเลือก v5.2-5.9 ที่ต้นฉบับปิดไว้ (flow/FX/event/skew/HTF/squeeze/ER/
  breadth/RS/illiq/FRM/rnd-stop/outside-mode) = ปิดที่นี่เช่นกัน และ *ยังไม่ port*
  ตัวคูณเหล่านั้น — เปิดไม่ได้จนกว่าจะ validate (บอกบนจอ)
- จุดต่าง (บอกตรงๆ): งบ/XD ใช้วันที่กรอกเอง (fail-open), backtest หักต้นทุน
  ไทยจริงทุกข้าง (คอม×1.07+ค่าธรรมเนียม+สเปรด) ซึ่ง *เข้มกว่า* TradingView
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import engine as E
import gold as G
from profile_data import PROFILE
from set_context import zsc


def set_tick(p: float) -> float:
    return (0.01 if p < 2 else 0.02 if p < 5 else 0.05 if p < 10 else
            0.10 if p < 25 else 0.25 if p < 100 else 0.50 if p < 200 else
            1.0 if p < 400 else 2.0)


def rnd_set(p: float) -> float:
    if p != p:
        return p
    t = set_tick(p)
    return round(round(p / t) * t, 2)


@dataclass
class SwingParams:
    trade_dir: str = "Long only"        # Long only / Short only / Both
    sma_len: int = 200
    use_idx: bool = True
    bos_len: int = 20
    use_score: bool = True
    mom_len: int = 126
    conf_min: int = 55
    vol_min_r: float = 1.5
    er_len: int = 20
    er_min: float = 0.30
    use_vol: bool = True
    vol_pc: float = 90.0
    use_ceil: bool = True
    lim_pct: float = 30.0
    ceil_bf: float = 5.0
    max_gap_atr: float = 1.5
    use_blk: bool = True
    use_kill: bool = True
    kill_pct: float = 6.0
    max_ls: int = 5
    use_liq: bool = True
    liq_min_m: float = 20.0
    liq_max_p: float = 3.0
    min_px: float = 2.0
    risk_pc: float = 0.5
    lot_sz: int = 100
    atr_len: int = 14
    sl_mlt: float = 2.0
    tr_mlt: float = 3.0
    tr_len: int = 22
    use_tp1: bool = False
    tp1_r: float = 1.5
    tp1_qty: float = 50.0
    use_fin: bool = False
    fin_r: float = 3.0
    use_tick: bool = True
    use_vt: bool = True
    vt_thr: float = 80.0
    vt_target: float = 2.0
    use_dd: bool = True
    dd_trig: float = 10.0
    use_beta: bool = True
    beta_hi: float = 1.3
    comm_side: float = 0.15
    spread_e: float = 0.15
    use_turncap: bool = True
    max_trades_m: int = 6
    use_profile: bool = True
    auto_lot: bool = True
    use_reg_exit: bool = False
    surv_flag: bool = False


def effective(ticker: str, p: SwingParams) -> dict:
    t = (ticker or "").replace(".BK", "").upper()
    if p.use_profile and t in PROFILE:
        vol_t, liq_t, fx_t, sect = PROFILE[t]
        return {"matched": True, "sector": sect, "vol_tier": vol_t,
                "liq_tier": liq_t, "fx_tier": fx_t,
                "liq_min": {1: 30.0, 2: 20.0, 3: 15.0}[liq_t],
                "liq_max": {1: 3.0, 2: 2.5, 3: 1.5}[liq_t],
                "vt_tgt": {1: 1.5, 2: 2.0, 3: 2.8}[vol_t]}
    return {"matched": False, "sector": "Other", "vol_tier": 2, "liq_tier": 2,
            "fx_tier": 0, "liq_min": p.liq_min_m, "liq_max": p.liq_max_p,
            "vt_tgt": p.vt_target}


def compute_frame(df: pd.DataFrame, bench_close=None, ticker: str = "",
                  p: SwingParams = None, blackout_dates=None) -> pd.DataFrame:
    p = p or SwingParams()
    eff = effective(ticker, p)
    fr = df.copy()
    c, o = fr["Close"], fr["Open"]
    h, l = fr["High"], fr["Low"]
    v = fr.get("Volume", pd.Series(0.0, index=fr.index))

    atrv = G.atr_wilder(fr, p.atr_len)
    fr["atr"] = atrv
    fr["sma_r"] = c.rolling(p.sma_len).mean()
    stk_up = c > fr["sma_r"]
    stk_dn = c < fr["sma_r"]
    if bench_close is not None and len(pd.Series(bench_close).dropna()) > p.sma_len:
        b = pd.Series(bench_close).reindex(fr.index).ffill()
        bs = b.rolling(p.sma_len).mean()
        idx_up = (~pd.Series(p.use_idx, index=fr.index)) | (b > bs) \
            if not p.use_idx else (b > bs)
        idx_dn = (b < bs) if p.use_idx else pd.Series(True, index=fr.index)
        idx_up = (b > bs) if p.use_idx else pd.Series(True, index=fr.index)
    else:
        idx_up = idx_dn = pd.Series(True, index=fr.index)
    fr["regime_up"] = (stk_up & idx_up).fillna(False)
    fr["regime_dn"] = (stk_dn & idx_dn).fillna(False)

    swing_hi = h.shift(1).rolling(p.bos_len).max()
    swing_lo = l.shift(1).rolling(p.bos_len).min()
    fr["swing_hi"], fr["swing_lo"] = swing_hi, swing_lo
    fr["bos"] = (c > swing_hi) & (c.shift(1) <= swing_hi.shift(1))
    fr["bos_dn"] = (c < swing_lo) & (c.shift(1) >= swing_lo.shift(1))

    er_num = (c - c.shift(p.er_len)).abs()
    er_den = c.diff().abs().rolling(p.er_len).sum()
    eff_r = (er_num / er_den.replace(0, np.nan)).fillna(0.0)
    fr["eff_r"] = eff_r
    mom_norm = (c - c.shift(p.mom_len)) / (atrv + 1e-10)
    avg_vol = v.rolling(20).mean()
    vol_ratio = (v / avg_vol.replace(0, np.nan)).fillna(0.0)
    hh252 = h.rolling(252).max()
    lo252 = l.rolling(252).min()
    ppl = ((c / hh252 - 0.80) / 0.20 * 20.0).clip(0, 20).round()
    pps = ((lo252 / c - 0.80) / 0.20 * 20.0).clip(0, 20).round()
    er_pts = (eff_r >= p.er_min) * 25
    vol_pts = (vol_ratio >= p.vol_min_r) * 20
    fr["conf_l"] = (mom_norm > 0) * 35 + er_pts + vol_pts + ppl
    fr["conf_s"] = (mom_norm < 0) * 35 + er_pts + vol_pts + pps
    fr["score_up"] = (~pd.Series(p.use_score, index=fr.index)) | (fr["conf_l"] >= p.conf_min) \
        if not p.use_score else (fr["conf_l"] >= p.conf_min)
    fr["score_dn"] = (fr["conf_s"] >= p.conf_min) if p.use_score \
        else pd.Series(True, index=fr.index)
    fr["trend_z"] = zsc(mom_norm, 100)

    atr_pct = (atrv / c * 100.0)
    fr["atr_pct"] = atr_pct
    fr["vol_rank"] = G.pct_rank(atr_pct, 100)
    fr["vol_shock"] = p.use_vol & (fr["vol_rank"] > p.vol_pc)

    prev_c = c.shift(1)
    ceil_d = (prev_c * (1 + p.lim_pct / 100) - c) / c * 100
    floor_d = (c - prev_c * (1 - p.lim_pct / 100)) / c * 100
    fr["ceil_ok"] = (~pd.Series(p.use_ceil, index=fr.index)) | (ceil_d >= p.ceil_bf) \
        if not p.use_ceil else (ceil_d >= p.ceil_bf)
    fr["floor_ok"] = (floor_d >= p.ceil_bf) if p.use_ceil \
        else pd.Series(True, index=fr.index)
    gap_atr = ((o - prev_c) / atrv.replace(0, np.nan)).fillna(0.0)
    fr["gap_ok_l"] = (p.max_gap_atr <= 0) | (gap_atr <= p.max_gap_atr)
    fr["gap_ok_s"] = (p.max_gap_atr <= 0) | (-gap_atr <= p.max_gap_atr)

    blk = pd.Series(False, index=fr.index)
    if p.use_blk and blackout_dates:
        for d in blackout_dates:
            try:
                dd = pd.Timestamp(d).normalize()
            except Exception:
                continue
            m = (fr.index.normalize() >= dd - pd.Timedelta(days=1)) & \
                (fr.index.normalize() <= dd + pd.Timedelta(days=3))
            blk |= pd.Series(m, index=fr.index)
    fr["in_blk"] = blk

    liq_val = (c * v).rolling(20).mean().shift(1)
    fr["liq_val"] = liq_val
    fr["liq_ok"] = (~pd.Series(p.use_liq, index=fr.index)) if not p.use_liq else \
        (liq_val.isna() | (liq_val >= eff["liq_min"] * 1e6))
    fr["price_ok"] = (p.min_px <= 0) | (c >= p.min_px)

    if bench_close is not None and len(pd.Series(bench_close).dropna()) > 130:
        b2 = pd.Series(bench_close).reindex(fr.index).ffill()
        ds, di = c.pct_change(), b2.pct_change()
        cov = (ds * di).rolling(120).mean() - ds.rolling(120).mean() * di.rolling(120).mean()
        var = di.rolling(120).std() ** 2
        fr["beta"] = cov / var.clip(lower=1e-12)
    else:
        fr["beta"] = np.nan

    fr["sl_dist"] = atrv * p.sl_mlt
    lot_hi = c.rolling(120).min() >= 500.0
    fr["lot"] = np.where(p.auto_lot, np.where(lot_hi, 50, 100), p.lot_sz)

    gates_l = ((~fr["vol_shock"]) & fr["ceil_ok"] & fr["gap_ok_l"]
               & (~fr["in_blk"]) & fr["liq_ok"] & fr["price_ok"]
               & (not p.surv_flag))
    gates_s = ((~fr["vol_shock"]) & fr["floor_ok"] & fr["gap_ok_s"]
               & (~fr["in_blk"]) & fr["liq_ok"] & fr["price_ok"]
               & (not p.surv_flag))
    allow_l = p.trade_dir != "Short only"
    allow_s = p.trade_dir != "Long only"
    fr["long_cond"] = (fr["regime_up"] & fr["bos"] & fr["score_up"]
                       & gates_l & allow_l).fillna(False)
    fr["short_cond"] = (fr["regime_dn"] & fr["bos_dn"] & fr["score_dn"]
                        & gates_s & allow_s).fillna(False)
    fr.attrs["eff"] = eff
    return fr


def size_mult_at(fr: pd.DataFrame, i: int, p: SwingParams, dd_pct: float,
                 loss_streak: int) -> float:
    vt = 1.0
    vr, ap = fr["vol_rank"].iloc[i], fr["atr_pct"].iloc[i]
    if p.use_vt and vr == vr and vr > p.vt_thr and ap > 0:
        vt = max(0.4, min(1.0, fr.attrs["eff"]["vt_tgt"] / ap))
    dd = 0.5 if (p.use_dd and dd_pct >= p.dd_trig) else 1.0
    ls = 0.6 if loss_streak >= 2 else 1.0
    bv = fr["beta"].iloc[i]
    bt = 1.0 if (not p.use_beta or bv != bv or bv <= p.beta_hi) \
        else max(0.5, p.beta_hi / bv)
    return max(0.2, vt * dd * ls * bt)


def state_today(fr: pd.DataFrame, p: SwingParams, equity: float = 1_000_000.0) -> dict:
    r = fr.iloc[-1]
    eff = fr.attrs["eff"]
    reg = "UP" if r["regime_up"] else ("DOWN" if r["regime_dn"] else "FLAT")
    sm = size_mult_at(fr, len(fr) - 1, p, 0.0, 0)
    sl = float(r["sl_dist"])
    lot = int(r["lot"])
    risk_cash = equity * p.risk_pc / 100 * sm
    qty = risk_cash / sl if sl > 0 else float("nan")
    if r["liq_val"] == r["liq_val"] and r["liq_val"] > 0:
        qty = min(qty, r["liq_val"] * eff["liq_max"] / 100 / r["Close"])
    qty = min(qty, equity / r["Close"])
    board = int(max(0, math.floor(qty / lot) * lot))
    ck = [
        ("Regime (หุ้น+ดัชนี > SMA200)", reg != "FLAT", f"{reg}"),
        (f"Breakout เหนือ high {p.bos_len} แท่ง",
         bool(r["bos"] or r["bos_dn"]),
         "เกิดแท่งนี้" if (r["bos"] or r["bos_dn"]) else
         f"trigger L={r['swing_hi']:.2f}"),
        (f"Confluence ≥ {p.conf_min}",
         bool(r["score_up"] if reg == "UP" else r["score_dn"] if reg == "DOWN"
              else False),
         f"L={int(r['conf_l'])} S={int(r['conf_s'])}"),
        ("Vol-shock gate", not bool(r["vol_shock"]),
         f"rank {r['vol_rank']:.0f}"),
        ("Ceiling/Floor ห่างพอ", bool(r["ceil_ok"] and r["floor_ok"]), ""),
        ("Gap ≤ 1.5 ATR", bool(r["gap_ok_l"]), ""),
        ("งบ/XD (กรอกเอง)", not bool(r["in_blk"]),
         "อยู่ในหน้าต่าง" if r["in_blk"] else "clear (fail-open)"),
        ("สภาพคล่อง + ราคา ≥ 2 บาท",
         bool(r["liq_ok"] and r["price_ok"]),
         f"พื้น {eff['liq_min']:.0f} ลบ. (โปรไฟล์)"),
        ("Surveillance (ธงมือ)", not p.surv_flag, ""),
    ]
    trig = bool(r["long_cond"] or r["short_cond"])
    return {"regime": reg, "triggered": trig, "checklist": ck,
            "sl_dist": round(sl, 2), "lot": lot, "size_mult": round(sm, 2),
            "board_qty": board,
            "entry_note": ("เงื่อนไขครบเมื่อปิดแท่งล่าสุด — ตามกติกา เข้า "
                           "open แท่งถัดไป" if trig else
                           "ยังไม่ครบเงื่อนไข — ห้ามไล่ราคา"),
            "eff": eff}


def backtest(fr: pd.DataFrame, p: SwingParams = None,
             equity0: float = 1_000_000.0) -> dict:
    p = p or SwingParams()
    eff = fr.attrs["eff"]
    o = fr["Open"].to_numpy(float)
    h = fr["High"].to_numpy(float)
    l = fr["Low"].to_numpy(float)
    c = fr["Close"].to_numpy(float)
    atr = fr["atr"].to_numpy(float)
    sma_r = fr["sma_r"].to_numpy(float)
    longc = fr["long_cond"].to_numpy(bool)
    shortc = fr["short_cond"].to_numpy(bool)
    sld = fr["sl_dist"].to_numpy(float)
    volrk = fr["vol_rank"].to_numpy(float)
    liqv = fr["liq_val"].to_numpy(float)
    lots = fr["lot"].to_numpy(int)
    idx = fr.index
    month = idx.to_period("M")
    cost_side = (p.comm_side * 1.07 + 0.007 + p.spread_e) / 100.0

    eq = equity0
    peak_m = equity0
    loss_streak = 0
    m_trades = 0
    cur_m = month[0] if len(month) else None
    pos = 0
    qty = 0.0
    entry_px = riskR = trail = np.nan
    entry_i = -1
    eq_at_entry = equity0
    tp1_done = False
    trades = []
    eq_curve = np.full(len(fr), np.nan)

    def mtm(i):
        return eq + ((c[i] - entry_px) * qty * pos if pos != 0 else 0.0)

    for i in range(len(fr)):
        if cur_m is not None and month[i] != cur_m:
            cur_m = month[i]
            peak_m = mtm(i - 1) if i else eq
            m_trades = 0
        if pos != 0:
            if pos > 0:
                chand = np.nanmax(h[max(0, i - p.tr_len + 1):i + 1]) - atr[i] * p.tr_mlt
                base = entry_px - riskR
                t_new = max(base, chand)
                trail = t_new if trail != trail else max(trail, t_new)
                stop = rnd_set(trail) if p.use_tick else trail
                if p.use_tp1 and not tp1_done:
                    tp1 = entry_px + riskR * p.tp1_r
                    tp1 = rnd_set(tp1) if p.use_tick else tp1
                    if h[i] >= tp1:
                        part = qty * p.tp1_qty / 100.0
                        pnl = (tp1 - entry_px) * part - cost_side * tp1 * part
                        eq += pnl
                        qty -= part
                        tp1_done = True
                hit = l[i] <= stop
                px = o[i] if o[i] < stop else stop
                reg_exit = p.use_reg_exit and c[i] < sma_r[i]
            else:
                chand = np.nanmin(l[max(0, i - p.tr_len + 1):i + 1]) + atr[i] * p.tr_mlt
                base = entry_px + riskR
                t_new = min(base, chand)
                trail = t_new if trail != trail else min(trail, t_new)
                stop = rnd_set(trail) if p.use_tick else trail
                hit = h[i] >= stop
                px = o[i] if o[i] > stop else stop
                reg_exit = p.use_reg_exit and c[i] > sma_r[i]
            if hit or reg_exit:
                exit_px = c[i] if (reg_exit and not hit) else px
                gross = (exit_px - entry_px) * qty * pos
                cost = cost_side * (entry_px + exit_px) * qty
                pnl = gross - cost
                eq += pnl
                closed_eq = eq
                trades.append({
                    "เข้า": idx[entry_i].date(), "ออก": idx[i].date(),
                    "ทิศ": "LONG" if pos > 0 else "SHORT",
                    "ราคาเข้า": round(entry_px, 2),
                    "ราคาออก": round(exit_px, 2),
                    "หุ้น": int(round(qty / max(lots[entry_i], 1))
                                * max(lots[entry_i], 1)) if False else int(qty),
                    "R": round((exit_px - entry_px) * pos / riskR, 2),
                    "กำไรสุทธิ (บาท)": round(pnl, 0),
                    "วันถือ": int(i - entry_i),
                })
                loss_streak = loss_streak + 1 if closed_eq < eq_at_entry else 0
                pos = 0
                qty = 0.0
                trail = np.nan
                tp1_done = False
        if pos == 0 and i > 0 and (longc[i - 1] or shortc[i - 1]):
            j = i - 1
            dd_pct = (peak_m - eq) / peak_m * 100 if peak_m > 0 else 0.0
            halted = p.use_kill and (dd_pct >= p.kill_pct
                                     or loss_streak >= p.max_ls)
            turn_ok = (not p.use_turncap) or (m_trades < p.max_trades_m)
            r_ = sld[j]
            if (not halted) and turn_ok and r_ == r_ and r_ > 0:
                sm = size_mult_at(fr, j, p, dd_pct, loss_streak)
                q = eq * p.risk_pc / 100 * sm / r_
                if p.use_liq and liqv[j] == liqv[j] and liqv[j] > 0:
                    q = min(q, liqv[j] * eff["liq_max"] / 100 / c[j])
                q = min(q, eq / c[j])
                lot = max(int(lots[j]), 1)
                q = math.floor(q / lot) * lot
                if q > 0:
                    pos = 1 if longc[j] else -1
                    qty = float(q)
                    entry_px = o[i]
                    riskR = r_
                    entry_i = i
                    eq_at_entry = eq
                    trail = np.nan
                    tp1_done = False
                    m_trades += 1
        m = mtm(i)
        eq_curve[i] = m
        peak_m = max(peak_m, m)

    eqs = pd.Series(eq_curve, index=idx).dropna()
    tdf = pd.DataFrame(trades)
    out = {"trades": tdf, "equity": eqs, "n": len(tdf)}
    if len(tdf):
        pnl = tdf["กำไรสุทธิ (บาท)"]
        wins = int((pnl > 0).sum())
        gp, gl = float(pnl[pnl > 0].sum()), float(-pnl[pnl < 0].sum())
        lo, hi = E.wilson_ci(wins, len(tdf))
        ret = eqs.pct_change().dropna()
        out.update({
            "wins": wins, "win_rate": wins / len(tdf), "ci": (lo, hi),
            "pf": gp / gl if gl > 0 else float("inf"),
            "expectancy_r": float(tdf["R"].mean()),
            "avg_hold": float(tdf["วันถือ"].mean()),
            "net_thb": float(pnl.sum()),
            "max_dd": float((eqs / eqs.cummax() - 1).min()),
            "psr": E.probabilistic_sharpe_ratio(ret) if len(ret) > 30
            else float("nan"),
        })
    return out
