# -*- coding: utf-8 -*-
"""
set_context.py — พอร์ต SET Market Context v1.0 + SET Stock Context v1.1
(คณิตเดียวกับ Pine ของผู้ใช้ "same math, same defaults" เพื่อให้จออ่านตรงกัน)

ซื่อตรงต่อข้อจำกัด:
- sFlow (CMF บนดัชนี): yfinance ไม่มี volume ของ ^SET.BK → n/a = 0 คะแนน (บอกบนจอ)
- Earnings/XD: yfinance ไม่มีฟีดไทยที่เชื่อถือได้ → ใช้วันที่กรอกเอง (fail-open)
- ROE: ไม่มีฟีด → fail-open (n/a) ตามดีไซน์ต้นฉบับ
- Self-test ledger ของ Market v1.0 ยังไม่ port (อยู่ในสคริปต์ Pine ต่อไป)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import gold as G  # rma/atr_wilder/pct_rank (Pine-faithful)
from profile_data import PROFILE


def _s(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def zsc(s: pd.Series, n: int) -> pd.Series:
    return (s - s.rolling(n).mean()) / (s.rolling(n).std() + 1e-10)


# ===========================================================================
# MARKET CONTEXT v1.0
# ===========================================================================

def market_context(set_close: pd.Series, vix_close=None, usdthb_close=None,
                   spx_close=None, eem_close=None, set_h=None, set_l=None,
                   set_v=None, sma_len=200, mom_len=126, len_flow=20,
                   len_mac=20, len_rel=63, pct_w=252, vix_hi=70.0, vix_lo=35.0,
                   thb_thr=3.0, zone_lvl=60) -> dict:
    c = pd.Series(set_close).dropna()
    if len(c) < sma_len + 5:
        return {"ok": False}
    idx = c.index
    smaL = c.rolling(sma_len).mean()
    reg_up = c.iloc[-1] > smaL.iloc[-1]
    reg_dn = c.iloc[-1] < smaL.iloc[-1]
    s_reg = 35 if reg_up else (-35 if reg_dn else 0)
    reg_pct = (c.iloc[-1] / smaL.iloc[-1] - 1) * 100

    mom_ret = (c.iloc[-1] / c.iloc[-1 - mom_len] - 1) * 100 if len(c) > mom_len else None
    s_mom = 0 if mom_ret is None else (25 if mom_ret > 0 else (-25 if mom_ret < 0 else 0))

    # [3] CMF บนดัชนี — ต้องมี H/L/V; yfinance ไม่ให้ volume ดัชนี → n/a
    f_cmf, s_flow, flow_na = None, 0, True
    if set_v is not None and set_h is not None and set_l is not None:
        H = pd.Series(set_h).reindex(idx)
        L = pd.Series(set_l).reindex(idx)
        V = pd.Series(set_v).reindex(idx).fillna(0)
        rng = (H - L)
        ad = np.where(rng > 0, (2 * c - H - L) / rng * V, 0.0)
        vs = V.rolling(len_flow).sum()
        if vs.iloc[-1] and vs.iloc[-1] > 0:
            f_cmf = float(pd.Series(ad, index=idx).rolling(len_flow).sum().iloc[-1]
                          / vs.iloc[-1])
            s_flow = 20 if f_cmf > 0.05 else (-20 if f_cmf < -0.05 else 0)
            flow_na = False

    # [4] macro push
    v_vix = v_thb = v_us = v_rel = 0
    vix_rank = thb_ret = rel_ret = None
    if vix_close is not None:
        vx = pd.Series(vix_close).dropna()
        if len(vx) > 60:
            vix_rank = float(G.pct_rank(vx, pct_w).iloc[-1])
            if vix_rank == vix_rank:
                v_vix = -1 if vix_rank >= vix_hi else (1 if vix_rank <= vix_lo else 0)
    if usdthb_close is not None:
        tb = pd.Series(usdthb_close).dropna()
        if len(tb) > len_mac:
            thb_ret = float((tb.iloc[-1] / tb.iloc[-1 - len_mac] - 1) * 100)
            v_thb = 1 if thb_ret <= -thb_thr else (-1 if thb_ret >= thb_thr else 0)
    if spx_close is not None:
        us = pd.Series(spx_close).dropna()
        if len(us) > sma_len:
            v_us = 1 if us.iloc[-1] > us.rolling(sma_len).mean().iloc[-1] else -1
    if eem_close is not None and len(c) > len_rel:
        em = pd.Series(eem_close).dropna()
        if len(em) > len_rel:
            set_r = (c.iloc[-1] / c.iloc[-1 - len_rel] - 1) * 100
            em_r = (em.iloc[-1] / em.iloc[-1 - len_rel] - 1) * 100
            rel_ret = float(set_r - em_r)
            v_rel = 1 if rel_ret > 0 else -1
    press = v_vix + v_thb + v_us + v_rel
    s_mac = (20 if press >= 2 else 10 if press == 1 else
             -10 if press == -1 else -20 if press <= -2 else 0)

    score = s_reg + s_mom + s_flow + s_mac
    zone = ("BUY zone" if score >= zone_lvl else
            "SELL zone" if score <= -zone_lvl else "กลาง")

    sma_s = c.rolling(20).mean()
    dom_tone = ((1 if c.iloc[-1] > smaL.iloc[-1] else -1)
                + (1 if c.iloc[-1] > sma_s.iloc[-1] else -1))
    f_sign = 1 if press > 0 else (-1 if press < 0 else 0)
    d_sign = 1 if dom_tone > 0 else (-1 if dom_tone < 0 else 0)
    if f_sign == 0:
        align = "ต่างชาติ (proxy) เป็นกลาง"
    elif f_sign == d_sign:
        align = ("สอดคล้อง: แรงผลัก 'ต่างชาติ' และโทนในประเทศไปทาง"
                 + ("บวก" if f_sign > 0 else "ลบ") + "เดียวกัน")
    else:
        align = "แยกทาง: แรงผลักต่างชาติสวนโทนในประเทศ (proxy — จับตา)"

    ret = c.pct_change()
    rv20 = ret.rolling(20).std()
    rv_rank = float(G.pct_rank(rv20.dropna(), pct_w).iloc[-1]) \
        if rv20.notna().sum() > 60 else None
    vol_risk = rv_rank is not None and rv_rank > 90

    d = idx[-1]
    mo, dd_ = d.month, d.day
    cal = []
    if mo == 1 and dd_ <= 15:
        cal.append("ม.ค. ต้นเดือน: หน้าต่างแรงขาย LTF เดิม + รีบาลานซ์ดัชนีมีผล")
    if mo == 12 and dd_ >= 15:
        cal.append("ธ.ค. ปลายเดือน: หน้าต่างซื้อกองทุนลดหย่อนภาษี")
    if mo in (2, 5, 8, 11) and dd_ >= 24:
        cal.append("สัปดาห์สุดท้าย ก.พ./พ.ค./ส.ค./พ.ย.: หน้าต่าง MSCI effective")
    if mo in (3, 6, 9) and dd_ >= 24:
        cal.append("สิ้นไตรมาส: หน้าต่าง window dressing")
    if mo in (1, 7) and dd_ <= 3:
        cal.append("วันแรกๆ ม.ค./ก.ค.: SET50/100 รอบใหม่มีผล")

    return {"ok": True, "score": int(score), "zone": zone,
            "parts": {"regime(±35)": s_reg, "momentum6M(±25)": s_mom,
                      "flowCMF(±20)": s_flow, "macro(±20)": s_mac},
            "reg_pct": round(float(reg_pct), 1),
            "mom_ret": None if mom_ret is None else round(float(mom_ret), 1),
            "cmf": None if f_cmf is None else round(f_cmf, 3),
            "flow_na": flow_na,
            "press": press,
            "press_votes": {"VIX": v_vix, "THB": v_thb, "US": v_us, "SETvsEM": v_rel},
            "vix_rank": None if vix_rank is None else round(vix_rank, 0),
            "thb_ret20": None if thb_ret is None else round(thb_ret, 1),
            "rel_ret": None if rel_ret is None else round(rel_ret, 1),
            "align": align, "vol_risk": bool(vol_risk),
            "rv_rank": None if rv_rank is None else round(rv_rank, 0),
            "calendar": cal,
            "note": ("คะแนน = บริบท ไม่ใช่สัญญาณ | flow/macro เป็น proxy เกรด C | "
                     "self-test ledger อยู่ในสคริปต์ Pine (ยังไม่ port)")}


# ===========================================================================
# STOCK CONTEXT v1.1
# ===========================================================================

@dataclass
class StockCtxParams:
    use_profile: bool = True
    sma_len: int = 200
    use_idx: bool = True
    rs_len: int = 63
    bos_len: int = 20
    mom_len: int = 126
    er_len: int = 20
    er_min: float = 0.30
    conf_min: int = 55
    vol_min_r: float = 1.5
    scr_len: int = 100
    atr_len: int = 14
    vol_pc: float = 90.0
    lim_pct: float = 30.0
    ceil_bf: float = 5.0
    use_blk: bool = True
    liq_min_m: float = 20.0
    min_px: float = 2.0
    use_illiq_g: bool = False
    illiq_pc_th: float = 90.0
    roe_min: float = 8.0
    beta_hi: float = 1.3
    skew_len: int = 120
    skew_thr: float = 1.0
    use_frm_veto: bool = False
    frm_vol_len: int = 60
    frm_vol_mult: float = 5.0
    frm_pump_v: float = 2.5
    frm_run_len: int = 20
    frm_run_pct: float = 40.0
    htf_ema_len: int = 40
    use_acc: bool = True
    acc_len: int = 20
    acc_flat: float = 2.0
    acc_ratio: float = 1.25
    comm_side: float = 0.15
    spread_e: float = 0.15


def profile_of(ticker: str):
    t = (ticker or "").replace(".BK", "").upper()
    if t in PROFILE:
        v = PROFILE[t]
        return {"matched": True, "vol_tier": v[0], "liq_tier": v[1],
                "fx_tier": v[2], "sector": v[3]}
    return {"matched": False, "vol_tier": 2, "liq_tier": 2, "fx_tier": 0,
            "sector": "Other"}


def _blackout(last_date, dates, use_blk: bool) -> bool:
    if not use_blk or not dates:
        return False
    ld = pd.Timestamp(last_date).normalize()
    for d in dates:
        try:
            dd = pd.Timestamp(d).normalize()
        except Exception:
            continue
        delta = (ld - dd).days
        if -1 <= delta <= 3:   # ประมาณหน้าต่าง 2 แท่งหลังเหตุการณ์ + วันก่อนหน้า
            return True
    return False


def stock_context(df: pd.DataFrame, bench_close=None, ticker: str = "",
                  p: StockCtxParams = None, surv_flag: bool = False,
                  blackout_dates=None, roe=None) -> dict:
    p = p or StockCtxParams()
    c, o = df["Close"].dropna(), df["Open"]
    h, l, v = df["High"], df["Low"], df.get("Volume", pd.Series(index=df.index))
    if len(c) < 260:
        return {"ok": False, "status": "ข้อมูลไม่พอ (ต้องการ ≥ 260 แท่ง)"}
    prof = profile_of(ticker)
    prof_on = p.use_profile and prof["matched"]
    liq_min_eff = ({1: 30.0, 2: 20.0, 3: 15.0}[prof["liq_tier"]]
                   if prof_on else p.liq_min_m)

    atrv = G.atr_wilder(df, p.atr_len)
    sma_r = c.rolling(p.sma_len).mean()
    avg_vol = v.rolling(20).mean()
    atr_pct = (atrv / c * 100.0)
    vol_rank = G.pct_rank(atr_pct.dropna(), 100)
    vol_shock = bool(vol_rank.iloc[-1] == vol_rank.iloc[-1]
                     and vol_rank.iloc[-1] > p.vol_pc)
    cost_rt = (p.comm_side * 1.07 + 0.007 + p.spread_e) * 2.0
    cva = float(cost_rt / atr_pct.iloc[-1] * 100.0) if atr_pct.iloc[-1] > 0 else None

    er_num = (c - c.shift(p.er_len)).abs()
    er_den = c.diff().abs().rolling(p.er_len).sum()
    eff_r = float((er_num / er_den.replace(0, np.nan)).fillna(0).iloc[-1])

    bench_ok = bench_close is not None and len(pd.Series(bench_close).dropna()) > p.sma_len
    if bench_ok:
        b = pd.Series(bench_close).reindex(c.index).ffill()
        b_sma = b.rolling(p.sma_len).mean()
        idx_up = bool(b.iloc[-1] > b_sma.iloc[-1]) if not p.use_idx else \
            bool(b.iloc[-1] > b_sma.iloc[-1])
        idx_up = (not p.use_idx) or bool(b.iloc[-1] > b_sma.iloc[-1])
        idx_dn = (not p.use_idx) or bool(b.iloc[-1] < b_sma.iloc[-1])
    else:
        b = None
        idx_up = idx_dn = True
    stk_up = bool(c.iloc[-1] > sma_r.iloc[-1])
    stk_dn = bool(c.iloc[-1] < sma_r.iloc[-1])
    regime_up = stk_up and idx_up
    regime_dn = stk_dn and idx_dn
    reg_dist = float((c.iloc[-1] / sma_r.iloc[-1] - 1) * 100)
    rs_diff = None
    if bench_ok and len(c) > p.rs_len:
        rs_diff = float(((c.iloc[-1] / c.iloc[-1 - p.rs_len] - 1)
                         - (b.iloc[-1] / b.iloc[-1 - p.rs_len] - 1)) * 100)

    mom_norm = (c - c.shift(p.mom_len)) / (atrv + 1e-10)
    trend_z = float(zsc(mom_norm, p.scr_len).iloc[-1])
    vol_ratio = float(v.iloc[-1] / avg_vol.iloc[-1]) if avg_vol.iloc[-1] else 0.0
    hh252 = h.rolling(252).max()
    lo252 = l.rolling(252).min()
    prox_hi = float(c.iloc[-1] / hh252.iloc[-1]) if hh252.iloc[-1] > 0 else 0.0
    prox_lo = float(lo252.iloc[-1] / c.iloc[-1]) if c.iloc[-1] > 0 else 0.0
    ppl = int(round(max(0.0, min(20.0, (prox_hi - 0.80) / 0.20 * 20.0))))
    pps = int(round(max(0.0, min(20.0, (prox_lo - 0.80) / 0.20 * 20.0))))
    er_pts = 25 if eff_r >= p.er_min else 0
    vol_pts = 20 if vol_ratio >= p.vol_min_r else 0
    conf_l = (35 if mom_norm.iloc[-1] > 0 else 0) + er_pts + vol_pts + ppl
    conf_s = (35 if mom_norm.iloc[-1] < 0 else 0) + er_pts + vol_pts + pps

    swing_hi = h.shift(1).rolling(p.bos_len).max()
    swing_lo = l.shift(1).rolling(p.bos_len).min()
    above = bool(c.iloc[-1] > swing_hi.iloc[-1])
    below = bool(c.iloc[-1] < swing_lo.iloc[-1])
    dist_l = float((swing_hi.iloc[-1] - c.iloc[-1]) / atrv.iloc[-1]) \
        if atrv.iloc[-1] > 0 else None
    dist_s = float((c.iloc[-1] - swing_lo.iloc[-1]) / atrv.iloc[-1]) \
        if atrv.iloc[-1] > 0 else None

    prev_c = float(c.iloc[-2])
    ceil_d = (prev_c * (1 + p.lim_pct / 100) - c.iloc[-1]) / c.iloc[-1] * 100
    floor_d = (c.iloc[-1] - prev_c * (1 - p.lim_pct / 100)) / c.iloc[-1] * 100
    ceil_near = ceil_d < p.ceil_bf
    floor_near = floor_d < p.ceil_bf

    in_blk = _blackout(c.index[-1], blackout_dates, p.use_blk)

    liq_val = (c * v).rolling(20).mean().shift(1).iloc[-1]
    liq_ok = (liq_val != liq_val) or liq_val >= liq_min_eff * 1e6
    price_ok = p.min_px <= 0 or c.iloc[-1] >= p.min_px
    ret1 = (c / c.shift(1) - 1).abs()
    amihud = (ret1 / (v * c).clip(lower=1.0)).rolling(60).mean()
    ill_rank = G.pct_rank(amihud.dropna(), 252)
    ill_val = float(ill_rank.iloc[-1]) if len(ill_rank) and \
        ill_rank.iloc[-1] == ill_rank.iloc[-1] else 50.0
    ill_high = ill_val >= p.illiq_pc_th
    ill_ok = (not p.use_illiq_g) or (not ill_high)

    frm_sma = v.rolling(p.frm_vol_len).mean()
    frm_ratio = float(v.iloc[-1] / frm_sma.iloc[-1]) if frm_sma.iloc[-1] else 1.0
    frm_r5 = float((v / frm_sma.replace(0, np.nan)).rolling(5).mean().iloc[-1])
    frm_run = float((c.iloc[-1] / c.iloc[-1 - p.frm_run_len] - 1) * 100) \
        if len(c) > p.frm_run_len else 0.0
    frm_spike = frm_ratio >= p.frm_vol_mult
    frm_pump = frm_run >= p.frm_run_pct and frm_r5 >= p.frm_pump_v
    frm_hit = frm_spike or frm_pump
    frm_ok = (not p.use_frm_veto) or (not frm_hit)

    beta_val = skew_v = None
    if bench_ok:
        ds = c.pct_change()
        di = b.pct_change()
        cov = (ds * di).rolling(120).mean() - ds.rolling(120).mean() * di.rolling(120).mean()
        var = di.rolling(120).std() ** 2
        bv = (cov / var.clip(lower=1e-12)).iloc[-1]
        beta_val = float(bv) if bv == bv else None
    ds_ = c.pct_change()
    m1 = ds_.rolling(p.skew_len).mean()
    m2 = (ds_ ** 2).rolling(p.skew_len).mean()
    m3 = (ds_ ** 3).rolling(p.skew_len).mean()
    var_ = (m2 - m1 ** 2).clip(lower=0)
    sk = ((m3 - 3 * m1 * m2 + 2 * m1 ** 3) / var_.pow(1.5).replace(0, np.nan)).iloc[-1]
    skew_v = float(sk) if sk == sk else None
    beta_flag = beta_val is not None and beta_val > p.beta_hi
    skew_flag = skew_v is not None and skew_v > p.skew_thr

    wc = c.resample("W-FRI").last().dropna()
    we = wc.ewm(span=p.htf_ema_len, adjust=False).mean()
    htf_up = bool(len(wc) > p.htf_ema_len and wc.iloc[-2] > we.iloc[-2]) \
        if len(wc) > p.htf_ema_len + 1 else None

    up_vol = (v.where(c > c.shift(1), 0.0)).rolling(p.acc_len).sum().iloc[-1]
    dn_vol = (v.where(c < c.shift(1), 0.0)).rolling(p.acc_len).sum().iloc[-1]
    rng_ = (h - l)
    clv = np.where(rng_ > 0, ((c - l) - (h - c)) / rng_, 0.0)
    clv_avg = float(pd.Series(clv, index=c.index).rolling(p.acc_len).mean().iloc[-1])
    vol_act = float(v.rolling(p.acc_len).mean().iloc[-1]
                    / max(float(v.rolling(100).mean().iloc[-1]), 1.0))
    rng_hi = h.rolling(p.acc_len).max().iloc[-1]
    rng_lo = l.rolling(p.acc_len).min().iloc[-1]
    pos_rng = (c.iloc[-1] - rng_lo) / (rng_hi - rng_lo) if rng_hi > rng_lo else 0.5
    votes = int((abs(c.iloc[-1] - c.iloc[-1 - p.acc_len]) <= p.acc_flat * atrv.iloc[-1])
                + (dn_vol > 0 and up_vol >= p.acc_ratio * dn_vol)
                + (clv_avg >= 0.10) + (vol_act >= 0.7))
    acc_watch = p.use_acc and (c.iloc[-1] < swing_hi.iloc[-1]) and pos_rng <= 0.65 \
        and votes >= 3

    struct_ok = (not surv_flag) and liq_ok and price_ok and ill_ok
    risk_clear = (not vol_shock) and (not in_blk) and frm_ok
    ready_l = regime_up and conf_l >= p.conf_min and above and struct_ok \
        and risk_clear and not ceil_near
    ready_s = regime_dn and conf_s >= p.conf_min and below and struct_ok \
        and risk_clear and not floor_near

    if surv_flag:
        status = "Surveillance flag (CB/TA)"
    elif not liq_ok:
        status = "ต่ำกว่าพื้นสภาพคล่อง"
    elif not price_ok:
        status = "ราคาต่ำกว่าขั้นต่ำ (tick band)"
    elif vol_shock:
        status = "Vol shock"
    elif in_blk:
        status = "หน้าต่างงบ/XD"
    elif not ill_ok:
        status = "Illiquid (Amihud veto ON)"
    elif not frm_ok:
        status = "Tape anomaly (FRM veto ON)"
    elif not (regime_up or regime_dn):
        status = "Regime flat / ขัดกับดัชนี"
    elif regime_up and ceil_near:
        status = "ใกล้เพดานวัน (ceiling)"
    elif regime_dn and floor_near:
        status = "ใกล้พื้นวัน (floor)"
    elif regime_up and conf_l < p.conf_min:
        status = "ขาขึ้นแต่ confluence อ่อน"
    elif regime_dn and conf_s < p.conf_min:
        status = "ขาลงแต่ confluence อ่อน"
    elif regime_up and not above:
        status = "ขาขึ้น — รอเบรกแนว 20 แท่ง"
    elif regime_dn and not below:
        status = "ขาลง — รอหลุดแนว 20 แท่ง"
    else:
        status = "LONG setup context met" if regime_up else "SHORT setup context met"

    lv = lambda ok, warn=False: "ok" if ok else ("warn" if warn else "bad")
    rows = [
        ("โปรไฟล์", f"{prof['sector']} V{prof['vol_tier']} L{prof['liq_tier']} "
         f"FX{prof['fx_tier']:+d}" + ("" if prof_on else " (นอกตาราง/ปิด)"),
         "ok" if prof_on else "na"),
        ("สภาพคล่อง 20 วัน", ("n/a (fail-open)" if liq_val != liq_val else
         f"{liq_val/1e6:,.0f} ลบ. (พื้น {liq_min_eff:.0f})"),
         "na" if liq_val != liq_val else lv(liq_ok)),
        ("Amihud illiquidity", f"pctile {ill_val:.0f}"
         + ("  สูงเทียบ 1 ปีตัวเอง" if ill_high else "")
         + ("  [veto ON]" if p.use_illiq_g else "  [veto off]"),
         "warn" if ill_high else "na"),
        ("ราคา/tick band", f"{c.iloc[-1]:.2f} บาท"
         + ("" if price_ok else f"  ต่ำกว่า {p.min_px:.1f}"), lv(price_ok)),
        ("Beta 120 วัน", ("n/a" if beta_val is None else f"{beta_val:.2f}")
         + ("  สูง → หลักฐานให้ลดไซส์" if beta_flag else ""),
         "warn" if beta_flag else "na"),
        (f"Skew {p.skew_len} วัน", ("n/a" if skew_v is None else f"{skew_v:.2f}")
         + ("  โปรไฟล์หวย → ลดไซส์" if skew_flag else ""),
         "warn" if skew_flag else "na"),
        (f"ROE (lag)", "n/a (fail-open — ไม่มีฟีดงบไทย)" if roe is None
         else f"{roe:.1f}%", "na" if roe is None else
         ("ok" if roe >= p.roe_min else "warn")),
        ("ต้นทุนไป-กลับ", f"{cost_rt:.2f}% = "
         + ("-" if cva is None else f"{cva:.0f}") + "% ของ 1 ATR",
         "warn" if (cva or 0) > 40 else "na"),
        ("Surveillance", "CB/TA BLOCK (ธงมือ)" if surv_flag
         else "clear ตามธงมือ — ตรวจ set.or.th", lv(not surv_flag)),
        (f"หุ้น vs SMA{p.sma_len}", ("UP " if stk_up else "DOWN " if stk_dn
         else "flat ") + f"{reg_dist:+.1f}%",
         "ok" if stk_up else ("bad" if stk_dn else "na")),
        ("SET index regime", ("gate off" if not p.use_idx else
         "n/a (fail-open)" if not bench_ok else
         ("เหนือ SMA — risk-on" if idx_up else "ใต้ SMA — risk-off")),
         "na" if (not p.use_idx or not bench_ok) else lv(idx_up)),
        (f"RS {p.rs_len} วัน vs SET", ("n/a" if rs_diff is None else
         f"{rs_diff:+.1f}%") + "  (info — หลักฐานไทยอ่อน)", "na"),
        ("Confluence LONG", f"{conf_l}/100 (thr {p.conf_min}, z {trend_z:+.1f})",
         "ok" if conf_l >= p.conf_min else "na"),
        ("Confluence SHORT", f"{conf_s}/100", "ok" if conf_s >= p.conf_min else "na"),
        ("Vol percentile", f"{vol_rank.iloc[-1]:.0f}"
         + ("  SHOCK" if vol_shock else ""), "warn" if vol_shock else "na"),
        ("Ceiling / Floor", f"C {ceil_d:.1f}%  F {floor_d:.1f}%"
         + ("  ใกล้เพดาน" if ceil_near else "  ใกล้พื้น" if floor_near else ""),
         "warn" if (ceil_near or floor_near) else "na"),
        ("งบ/XD (กรอกเอง)", "อยู่ในหน้าต่างเหตุการณ์" if in_blk
         else ("ปิดใช้" if not p.use_blk else "clear (fail-open)"),
         "warn" if in_blk else "na"),
        (f"Tape (FRM) ×{frm_ratio:.1f}", ("PUMP signature" if frm_pump else
         "SPIKE" if frm_spike else "clear")
         + ("  [veto ON]" if p.use_frm_veto else "  [veto off]"),
         "warn" if frm_hit else "na"),
        ("HTF Weekly EMA40", "n/a" if htf_up is None else
         ("UP" if htf_up else "DOWN") + "  (info only)",
         "na" if htf_up is None else ("ok" if htf_up else "bad")),
        ("ระยะถึง trigger", ("เหนือ 20-bar high แล้ว" if above else
         "หลุด 20-bar low แล้ว" if below else
         (f"{dist_l:.1f} ATR ถึง 20-bar high" if regime_up and dist_l is not None
          else f"{dist_s:.1f} ATR ถึง 20-bar low" if dist_s is not None else "-")),
         "ok" if (above or below) else "na"),
        ("Accum watch", f"{votes}/4" + ("  WATCH" if acc_watch else "")
         + "  (display only, grade C)", "warn" if acc_watch else "na"),
    ]
    return {"ok": True, "rows": rows, "status": status,
            "ready_l": bool(ready_l), "ready_s": bool(ready_s),
            "conf_l": int(conf_l), "conf_s": int(conf_s),
            "regime": "UP" if regime_up else ("DOWN" if regime_dn else "FLAT"),
            "swing_hi": float(swing_hi.iloc[-1]), "swing_lo": float(swing_lo.iloc[-1]),
            "atr": float(atrv.iloc[-1]), "prof": prof}
