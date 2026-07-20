# -*- coding: utf-8 -*-
"""
gold.py — XAU Research Trend Pullback v6.4 (port จาก Pine v6 ของผู้ใช้)

หลักการ port อย่างซื่อสัตย์:
- คงกติกาต้นฉบับทุกข้อ รวม 'ระดับหลักฐาน' ที่ผู้เขียนสคริปต์จัดไว้เอง:
  Tier A (เปิด): regime SMA200+slope, DXY trend veto, pullback entry,
                 confluence score, vol-shock gate, gap gate, cost gate
  Tier B (ปิดค่าตั้งต้น — ต้อง validate ก่อน): US10Y soft gate,
                 JPY carry-unwind veto, weekly EMA gate, ER hard gate
  Tier C (พังหลัง 2022 — จงใจไม่โค้ด): real-yield inversion, gold-DXY inverse
- จุดที่ *เข้มกว่า* TradingView engine: backtest นี้หัก spread ไป-กลับใน
  กำไรทุกเทรด (Pine ใช้เป็นแค่ gate) และแยกบัญชี swap ให้เห็น net หลัง swap
- จุดที่ต่างจากต้นฉบับ (บอกตรงๆ): ใช้ GC=F (ฟิวเจอร์สทอง) แทน spot XAUUSD
  เพราะเป็นแหล่งฟรีที่เสถียร — ระดับราคา/ATR ใกล้เคียง แต่ basis ต่างเล็กน้อย;
  session filter ไม่มีผลบน daily (ตามต้นฉบับ: ใช้เฉพาะ intraday)
- เป้า validate ของต้นฉบับ (คงไว้): 30-50 เทรดใน journal, PF > 1.5 หลัง swap
  — ชื่อรุ่น v6.4 บอกว่าผ่านการปรับหลายรอบ ค่า DSR จึงควรกรอก trials สูง
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import engine as E


# ---------------------------------------------------------------------------
# Parameters (ค่าตั้งต้น = ต้นฉบับ v6.4 ทุกตัว)
# ---------------------------------------------------------------------------

@dataclass
class GoldParams:
    # 1) Regime
    reg_sma_len: int = 200
    slope_bk: int = 5
    use_reg_exit: bool = False
    # 2) Entry
    entry_mode: str = "Pullback"          # Pullback / Breakout / Both
    pb_ema_len: int = 21
    st_ema_len: int = 50
    rsi_len: int = 14
    rsi_floor_l: float = 40.0
    rsi_ceil_s: float = 60.0
    use_brk_conf: bool = False
    bos_len: int = 20
    # 3) Confluence score
    use_score: bool = True
    mom_len: int = 126
    er_len: int = 20
    er_min: float = 0.30
    use_er_gate: bool = False             # Tier B
    conf_min: int = 40
    # 4) Cross-asset
    use_dxy: bool = True                  # Tier A
    dxy_len: int = 50
    use_y10: bool = False                 # Tier B
    y10_len: int = 50
    use_carry: bool = False               # Tier B
    jpy_bk: int = 3
    jpy_thr: float = 2.5
    vix_thr: float = 25.0
    use_htf_w: bool = False               # Tier B
    w_ema_len: int = 40
    # 5) Risk gates
    use_vol_shock: bool = True
    vol_pc: float = 90.0
    max_gap_atr: float = 1.5
    max_cost_r: float = 10.0
    # 8) Stop/trail
    atr_len: int = 14
    stop_mult: float = 1.8
    tr_len: int = 22
    tr_mult: float = 3.0
    use_tp1: bool = False
    tp1_r: float = 1.5
    tp1_qty_pct: float = 50.0
    use_fin: bool = False
    fin_r: float = 3.0
    # 9) Sizing
    risk_pct: float = 1.0
    spread_cents: float = 25.0            # ไป-กลับ (US cents/oz)
    use_vt: bool = True
    vt_thr: float = 80.0
    vt_target: float = 1.2
    use_dd_cut: bool = True
    dd_trig: float = 10.0
    # 10) Swap (ลบ = ต้นทุน) — ตั้งตามโบรกจริงของผู้ใช้
    swap_long_oz: float = -0.76
    swap_short_oz: float = 0.30
    trip_wed: bool = True
    # 11) Brakes
    use_kill: bool = True
    kill_pct: float = 6.0
    max_loss_streak: int = 5
    max_per_day: int = 2
    max_trades_month: int = 20


TIER_NOTE = {
    "A": "Tier A — เปิดใช้ (systematic filter)",
    "B": "Tier B — ปิดค่าตั้งต้น (ต้อง validate ก่อนเปิด)",
    "C": "Tier C — จงใจไม่โค้ด (ความสัมพันธ์พังหลังปี 2022)",
}


# ---------------------------------------------------------------------------
# Indicators แบบ Pine-faithful (RMA/Wilder)
# ---------------------------------------------------------------------------

def rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = rma(d.clip(lower=0), n)
    dn = rma(-d.clip(upper=0), n)
    denom = up + dn
    # เทียบเท่า Wilder RSI = 100·up/(up+dn) — เคสขึ้นล้วน (dn=0) ได้ 100 ตาม Pine
    return pd.Series(np.where(denom > 0, 100.0 * up / denom, np.nan),
                     index=close.index)


def atr_wilder(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c1 = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h - l, (h - c1).abs(), (l - c1).abs()], axis=1).max(axis=1)
    return rma(tr, n)


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    num = (close - close.shift(n)).abs()
    den = close.diff().abs().rolling(n).sum()
    return (num / den.replace(0, np.nan)).fillna(0.0)


def pct_rank(series: pd.Series, n: int = 252) -> pd.Series:
    return series.rolling(n, min_periods=max(60, n // 3)).apply(
        lambda w: float((w <= w[-1]).mean() * 100.0), raw=True)


def _trend_state(close: pd.Series, sma_len: int, slope_bk: int):
    """(up, dn) ตามนิยาม Pine: ใช้ค่า *แท่งก่อนหน้า* (lookahead[1]) กันรีเพนต์"""
    sma = close.rolling(sma_len).mean()
    sma_now, sma_old = sma.shift(1), sma.shift(1 + slope_bk)
    c = close.shift(1)
    up = (c > sma_now) & (sma_now > sma_old)
    dn = (c < sma_now) & (sma_now < sma_old)
    return up.fillna(False), dn.fillna(False)


# ---------------------------------------------------------------------------
# สร้างเฟรมตัวชี้วัด + เงื่อนไขทุก gate (ต่อวัน)
# ---------------------------------------------------------------------------

def compute_frame(xau: pd.DataFrame, dxy_close=None, y10_close=None,
                  usdjpy_close=None, vix_close=None,
                  p: GoldParams = None) -> pd.DataFrame:
    p = p or GoldParams()
    df = xau.copy()
    c, o, h, l = df["Close"], df["Open"], df["High"], df["Low"]

    # regime (self, ค่ายืนยันของเมื่อวานตามต้นฉบับ)
    reg_sma = c.rolling(p.reg_sma_len).mean().shift(1)
    reg_sma_old = c.rolling(p.reg_sma_len).mean().shift(1 + p.slope_bk)
    df["reg_sma"] = reg_sma
    df["regime_up"] = ((c > reg_sma) & (reg_sma > reg_sma_old)).fillna(False)
    df["regime_dn"] = ((c < reg_sma) & (reg_sma < reg_sma_old)).fillna(False)

    # chart tools
    df["pb_ema"] = c.ewm(span=p.pb_ema_len, adjust=False).mean()
    df["st_ema"] = c.ewm(span=p.st_ema_len, adjust=False).mean()
    df["rsi"] = rsi_wilder(c, p.rsi_len)
    atr = atr_wilder(df, p.atr_len)
    df["atr"] = atr
    df["day_atr"] = atr.shift(1)            # dayAtr = ATR[1] ต่อต้นฉบับ
    df["stop_dist"] = df["day_atr"] * p.stop_mult
    atr_pct = (atr / c * 100.0).fillna(0.0)
    df["atr_pct"] = atr_pct
    df["vol_rank"] = pct_rank(atr_pct, 252)
    df["vol_shock"] = p.use_vol_shock & (df["vol_rank"] > p.vol_pc)
    df["eff_r"] = efficiency_ratio(c, p.er_len)
    df["er_ok"] = (~p.use_er_gate) | (df["eff_r"] >= p.er_min)

    # confluence score
    mom_norm = (c - c.shift(p.mom_len)) / (atr + 1e-10)
    hh252 = h.rolling(252).max()
    lo252 = l.rolling(252).min()
    df["score_l"] = ((mom_norm > 0) * 40 + (df["eff_r"] >= p.er_min) * 30
                     + ((c / hh252) >= 0.90) * 30)
    df["score_s"] = ((mom_norm < 0) * 40 + (df["eff_r"] >= p.er_min) * 30
                     + ((lo252 / c) >= 0.90) * 30)
    df["score_ok_l"] = (~p.use_score) | (df["score_l"] >= p.conf_min)
    df["score_ok_s"] = (~p.use_score) | (df["score_s"] >= p.conf_min)

    # gap gate (open วันนี้ เทียบ close เมื่อวาน หน่วย dayATR)
    gap = (o - c.shift(1)) / df["day_atr"].replace(0, np.nan)
    df["gap_ok_l"] = (p.max_gap_atr <= 0) | (gap <= p.max_gap_atr) | gap.isna()
    df["gap_ok_s"] = (p.max_gap_atr <= 0) | (-gap <= p.max_gap_atr) | gap.isna()

    # cost gate
    rt_cost = p.spread_cents / 100.0
    cost_pct_r = rt_cost / df["stop_dist"].replace(0, np.nan) * 100.0
    df["cost_pct_r"] = cost_pct_r
    df["cost_ok"] = (p.max_cost_r <= 0) | (cost_pct_r <= p.max_cost_r) | cost_pct_r.isna()

    # cross-asset — Tier A: DXY
    def _ok_flags(close_ext, use, length):
        if close_ext is None or not use:
            t = pd.Series(True, index=df.index)
            return t, t, pd.Series(False, index=df.index), pd.Series(False, index=df.index), False
        s = pd.Series(close_ext).reindex(df.index).ffill()
        up, dn = _trend_state(s, length, p.slope_bk)
        return (~up), (~dn), up, dn, True

    df["dxy_ok_l"], df["dxy_ok_s"], df["dxy_up"], df["dxy_dn"], dxy_has = \
        _ok_flags(dxy_close, p.use_dxy, p.dxy_len)
    df["y10_ok_l"], df["y10_ok_s"], df["y10_up"], df["y10_dn"], y10_has = \
        _ok_flags(y10_close, p.use_y10, p.y10_len)

    # Tier B: carry-unwind stress (JPY surge + VIX)
    if p.use_carry and usdjpy_close is not None and vix_close is not None:
        j = pd.Series(usdjpy_close).reindex(df.index).ffill().shift(1)
        jr = (j / j.shift(p.jpy_bk) - 1.0) * 100.0
        v = pd.Series(vix_close).reindex(df.index).ffill().shift(1)
        df["jpy_ret"] = jr
        df["carry_stress"] = ((jr <= -p.jpy_thr) & (v >= p.vix_thr)).fillna(False)
    else:
        df["jpy_ret"] = np.nan
        df["carry_stress"] = False

    # Tier B: weekly EMA gate (สร้างจากราคา daily ตัวเอง)
    if p.use_htf_w:
        wc = c.resample("W-FRI").last()
        we = wc.ewm(span=p.w_ema_len, adjust=False).mean()
        w_ok_l = (wc.shift(1) > we.shift(1)).reindex(df.index).ffill()
        w_ok_s = (wc.shift(1) < we.shift(1)).reindex(df.index).ffill()
        df["w_ok_l"] = w_ok_l.fillna(True)
        df["w_ok_s"] = w_ok_s.fillna(True)
    else:
        df["w_ok_l"] = True
        df["w_ok_s"] = True

    # entry triggers
    pull_l = ((c > df["st_ema"]) & (l <= df["pb_ema"]) & (c > df["pb_ema"])
              & (c > o) & (df["rsi"] >= p.rsi_floor_l))
    pull_s = ((c < df["st_ema"]) & (h >= df["pb_ema"]) & (c < df["pb_ema"])
              & (c < o) & (df["rsi"] <= p.rsi_ceil_s))
    if p.use_brk_conf:
        pull_l &= c > h.shift(1)
        pull_s &= c < l.shift(1)
    swing_hi = h.shift(1).rolling(p.bos_len).max()
    swing_lo = l.shift(1).rolling(p.bos_len).min()
    bos_up = (c > swing_hi) & (c.shift(1) <= swing_hi.shift(1))
    bos_dn = (c < swing_lo) & (c.shift(1) >= swing_lo.shift(1))
    mode_p = p.entry_mode in ("Pullback", "Both")
    mode_b = p.entry_mode in ("Breakout", "Both")
    df["trig_l"] = (mode_p & pull_l) | (mode_b & bos_up)
    df["trig_s"] = (mode_p & pull_s) | (mode_b & bos_dn)
    df["swing_hi"], df["swing_lo"] = swing_hi, swing_lo

    gates = ((~df["vol_shock"]) & (~df["carry_stress"]) & df["cost_ok"]
             & df["er_ok"])
    df["long_cond"] = (df["regime_up"] & df["trig_l"] & df["score_ok_l"]
                       & gates & df["gap_ok_l"] & df["dxy_ok_l"]
                       & df["y10_ok_l"] & df["w_ok_l"])
    df["short_cond"] = (df["regime_dn"] & df["trig_s"] & df["score_ok_s"]
                        & gates & df["gap_ok_s"] & df["dxy_ok_s"]
                        & df["y10_ok_s"] & df["w_ok_s"])
    df.attrs["dxy_has"] = dxy_has
    df.attrs["y10_has"] = y10_has
    return df


# ---------------------------------------------------------------------------
# สถานะวันล่าสุด (mirror ตาราง dashboard ของต้นฉบับ + checklist โปร่งใส)
# ---------------------------------------------------------------------------

def state_today(fr: pd.DataFrame, p: GoldParams = None,
                equity: float = 10000.0) -> dict:
    p = p or GoldParams()
    r = fr.iloc[-1]
    reg = "UP" if r["regime_up"] else ("DOWN" if r["regime_dn"] else "MIXED")
    if r["vol_shock"]:
        status = "Vol shock — งดเข้าใหม่"
    elif bool(r.get("carry_stress")):
        status = "Carry stress — งดเข้าใหม่"
    elif not r["cost_ok"]:
        status = "Cost gate — spread แพงเทียบ 1R"
    elif reg == "MIXED":
        status = "Regime flat — ไม่มีฝั่งให้เทรด"
    elif not r["er_ok"]:
        status = "Chop (ER ต่ำ)"
    elif reg == "UP" and not r["dxy_ok_l"]:
        status = "DXY veto (ดอลลาร์ขาขึ้นยืนยัน)"
    elif reg == "DOWN" and not r["dxy_ok_s"]:
        status = "DXY veto (ดอลลาร์ขาลงยืนยัน)"
    elif bool(r["long_cond"]) or bool(r["short_cond"]):
        status = "TRIGGER — เงื่อนไขเข้าครบเมื่อปิดแท่งนี้"
    else:
        status = "Scanning — รอ pullback ในฝั่ง regime"
    side = "LONG" if reg == "UP" else ("SHORT" if reg == "DOWN" else None)
    checklist = []

    def add(name, ok, tier, detail=""):
        checklist.append({"เงื่อนไข": name, "ผ่าน": bool(ok), "Tier": tier,
                          "รายละเอียด": detail})

    add("Regime SMA200+slope มีทิศ", reg != "MIXED", "A",
        f"{reg} (ปิดเทียบ SMA200 ของเมื่อวาน)")
    if side == "LONG":
        add("Pullback trigger (แตะ EMA21 แล้วปิดกลับ)", r["trig_l"], "A",
            f"RSI {r['rsi']:.0f} ≥ {p.rsi_floor_l:.0f}")
        add(f"Score ≥ {p.conf_min}", r["score_ok_l"], "A",
            f"score L = {int(r['score_l'])}")
        add("DXY ไม่ขึ้นยืนยัน (veto)", r["dxy_ok_l"], "A",
            "up" if r["dxy_up"] else ("down" if r["dxy_dn"] else "flat/n-a")
            if fr.attrs.get("dxy_has") else "n/a — gate ผ่านอัตโนมัติ")
        add("Gap ≤ 1.5 dayATR", r["gap_ok_l"], "A")
    elif side == "SHORT":
        add("Pullback trigger (แตะ EMA21 แล้วปิดกลับ)", r["trig_s"], "A",
            f"RSI {r['rsi']:.0f} ≤ {p.rsi_ceil_s:.0f}")
        add(f"Score ≥ {p.conf_min}", r["score_ok_s"], "A",
            f"score S = {int(r['score_s'])}")
        add("DXY ไม่ลงยืนยัน (veto)", r["dxy_ok_s"], "A",
            "n/a — gate ผ่านอัตโนมัติ" if not fr.attrs.get("dxy_has") else "")
        add("Gap ≤ 1.5 dayATR", r["gap_ok_s"], "A")
    add(f"Vol-shock (ATR% เกิน P{p.vol_pc:.0f})", not r["vol_shock"], "A",
        f"vol rank {r['vol_rank']:.0f}")
    add("Cost ≤ 10% ของ 1R", r["cost_ok"], "A",
        f"{r['cost_pct_r']:.1f}% ของ 1R" if r["cost_pct_r"] == r["cost_pct_r"] else "")
    add("US10Y soft gate", r["y10_ok_l"] if side == "LONG" else r["y10_ok_s"],
        "B", "ปิดอยู่" if not p.use_y10 else "")
    add("JPY carry-unwind veto", not bool(r.get("carry_stress")), "B",
        "ปิดอยู่" if not p.use_carry else f"JPY {r.get('jpy_ret', float('nan')):+.1f}%/3d")
    add("Weekly EMA agreement", r["w_ok_l"] if side == "LONG" else r["w_ok_s"],
        "B", "ปิดอยู่" if not p.use_htf_w else "")

    stop_dist = float(r["stop_dist"]) if r["stop_dist"] == r["stop_dist"] else float("nan")
    vt_mult = 1.0
    if p.use_vt and r["vol_rank"] == r["vol_rank"] and r["vol_rank"] > p.vt_thr \
            and r["atr_pct"] > 0:
        vt_mult = max(0.4, min(1.0, p.vt_target / float(r["atr_pct"])))
    qty = (equity * p.risk_pct / 100.0 * vt_mult / stop_dist
           if stop_dist and stop_dist > 0 else float("nan"))
    entry = float(r["Close"])
    plan = None
    if side and stop_dist == stop_dist:
        sl = entry - stop_dist if side == "LONG" else entry + stop_dist
        plan = {"side": side, "ref_close": round(entry, 2),
                "stop_dist": round(stop_dist, 2), "sl": round(sl, 2),
                "trail": f"Chandelier {p.tr_len} แท่ง − {p.tr_mult}×ATR (ไม่มี TP ตายตัว)",
                "qty_oz": round(qty, 2) if qty == qty else None,
                "size_mult": round(vt_mult, 2)}
    return {"regime": reg, "status": status,
            "score_l": int(r["score_l"]), "score_s": int(r["score_s"]),
            "rsi": round(float(r["rsi"]), 1) if r["rsi"] == r["rsi"] else None,
            "atr_pct": round(float(r["atr_pct"]), 2),
            "vol_rank": round(float(r["vol_rank"]), 0)
            if r["vol_rank"] == r["vol_rank"] else None,
            "checklist": checklist, "plan": plan,
            "triggered": bool(r["long_cond"] or r["short_cond"])}


# ---------------------------------------------------------------------------
# Backtest: สัญญาณปิดแท่ง t → เข้า open(t+1), trail chandelier, swap แยกบัญชี
# ---------------------------------------------------------------------------

def backtest(fr: pd.DataFrame, p: GoldParams = None,
             equity0: float = 10000.0) -> dict:
    p = p or GoldParams()
    o = fr["Open"].to_numpy(float)
    h = fr["High"].to_numpy(float)
    l = fr["Low"].to_numpy(float)
    c = fr["Close"].to_numpy(float)
    atr = fr["atr"].to_numpy(float)
    longc = fr["long_cond"].to_numpy(bool)
    shortc = fr["short_cond"].to_numpy(bool)
    stopd = fr["stop_dist"].to_numpy(float)
    volrk = fr["vol_rank"].to_numpy(float)
    atrpc = fr["atr_pct"].to_numpy(float)
    idx = fr.index
    weekday = idx.weekday  # Mon=0 ... Thu=3
    month = idx.to_period("M")

    rt_cost = p.spread_cents / 100.0
    eq = equity0
    peak_eq_month = equity0
    loss_streak = 0
    month_trades = 0
    cur_month = month[0] if len(month) else None
    pos = 0
    qty = 0.0
    entry_px = risk = trail = np.nan
    entry_i = -1
    swap_open = 0.0
    trades = []
    eq_curve = np.full(len(fr), np.nan)
    halted_days = 0

    def unreal(i):
        return (c[i] - entry_px) * qty * pos if pos != 0 else 0.0

    for i in range(len(fr)):
        if cur_month is not None and month[i] != cur_month:
            cur_month = month[i]
            peak_eq_month = eq + unreal(i - 1 if i else i)
            loss_streak = 0
            month_trades = 0
        # swap accrual: ถือมาจากแท่งก่อน (triple ถ้าวันนี้พฤหัส = ข้ามคืนพุธ)
        if pos != 0 and i > entry_i:
            nights = 3 if (p.trip_wed and weekday[i] == 3) else 1
            rate = p.swap_long_oz if pos > 0 else p.swap_short_oz
            swap_open += nights * rate * qty
        # exit ด้วย trail/base stop
        if pos != 0:
            if pos > 0:
                chand = np.nanmax(h[max(0, i - p.tr_len + 1):i + 1]) - atr[i] * p.tr_mult
                base = entry_px - risk
                trail = max(trail, base, chand) if trail == trail else max(base, chand)
                hit = l[i] <= trail
                px = min(o[i], trail) if o[i] < trail else trail
                reg_exit = p.use_reg_exit and c[i] < fr["reg_sma"].iloc[i]
            else:
                chand = np.nanmin(l[max(0, i - p.tr_len + 1):i + 1]) + atr[i] * p.tr_mult
                base = entry_px + risk
                trail = min(trail, base, chand) if trail == trail else min(base, chand)
                hit = h[i] >= trail
                px = max(o[i], trail) if o[i] > trail else trail
                reg_exit = p.use_reg_exit and c[i] > fr["reg_sma"].iloc[i]
            if hit or reg_exit:
                exit_px = c[i] if (reg_exit and not hit) else px
                gross = (exit_px - entry_px) * qty * pos
                pnl = gross - rt_cost * qty          # หัก spread (เข้มกว่า Pine)
                eq += pnl
                trades.append({
                    "เข้า": idx[entry_i].date(), "ออก": idx[i].date(),
                    "ทิศ": "LONG" if pos > 0 else "SHORT",
                    "ราคาเข้า": round(entry_px, 2), "ราคาออก": round(exit_px, 2),
                    "R": round((exit_px - entry_px) * pos / risk, 2),
                    "กำไร$ (หัก spread)": round(pnl, 2),
                    "swap$ (ประมาณ)": round(swap_open, 2),
                    "สุทธิหลัง swap$": round(pnl + swap_open, 2),
                    "วันถือ": int(i - entry_i),
                })
                loss_streak = loss_streak + 1 if pnl < 0 else 0
                pos = 0
                qty = 0.0
                swap_open = 0.0
                trail = np.nan
        # entry จากสัญญาณแท่งก่อนหน้า (next-bar open)
        if pos == 0 and i > 0 and (longc[i - 1] or shortc[i - 1]):
            live_eq = eq
            dd_pct = ((peak_eq_month - live_eq) / peak_eq_month * 100.0
                      if peak_eq_month > 0 else 0.0)
            halted = p.use_kill and (dd_pct >= p.kill_pct
                                     or loss_streak >= p.max_loss_streak)
            if halted:
                halted_days += 1
            if (not halted) and month_trades < p.max_trades_month:
                j = i - 1
                r_ = stopd[j]
                if r_ == r_ and r_ > 0:
                    vt = 1.0
                    if p.use_vt and volrk[j] == volrk[j] and volrk[j] > p.vt_thr \
                            and atrpc[j] > 0:
                        vt = max(0.4, min(1.0, p.vt_target / atrpc[j]))
                    ddm = 0.5 if (p.use_dd_cut and dd_pct >= p.dd_trig) else 1.0
                    lsm = 0.6 if loss_streak >= 2 else 1.0
                    smult = max(0.2, vt * ddm * lsm)
                    q = eq * p.risk_pct / 100.0 * smult / r_
                    q = math.floor(q * 100) / 100.0
                    if q > 0:
                        pos = 1 if longc[j] else -1
                        qty = q
                        entry_px = o[i]
                        risk = r_
                        entry_i = i
                        trail = np.nan
                        swap_open = 0.0
                        month_trades += 1
        # mark-to-market equity + monthly peak
        mtm = eq + unreal(i)
        eq_curve[i] = mtm
        peak_eq_month = max(peak_eq_month, mtm)

    eqs = pd.Series(eq_curve, index=idx).dropna()
    tdf = pd.DataFrame(trades)
    out = {"trades": tdf, "equity": eqs, "n": len(tdf),
           "halted_days": halted_days}
    if len(tdf):
        pnl = tdf["กำไร$ (หัก spread)"]
        net_after = tdf["สุทธิหลัง swap$"]
        wins = int((pnl > 0).sum())
        gp = float(pnl[pnl > 0].sum())
        gl = float(-pnl[pnl < 0].sum())
        gp2 = float(net_after[net_after > 0].sum())
        gl2 = float(-net_after[net_after < 0].sum())
        lo, hi = E.wilson_ci(wins, len(tdf))
        ret = eqs.pct_change().dropna()
        out.update({
            "wins": wins, "win_rate": wins / len(tdf), "ci": (lo, hi),
            "pf_pre_swap": (gp / gl) if gl > 0 else float("inf"),
            "pf_after_swap": (gp2 / gl2) if gl2 > 0 else float("inf"),
            "expectancy_r": float(tdf["R"].mean()),
            "avg_hold": float(tdf["วันถือ"].mean()),
            "swap_total": float(tdf["swap$ (ประมาณ)"].sum()),
            "net_profit": float(pnl.sum()),
            "net_after_swap": float(net_after.sum()),
            "max_dd": float((eqs / eqs.cummax() - 1).min()),
            "psr": E.probabilistic_sharpe_ratio(ret) if len(ret) > 30
            else float("nan"),
        })
    return out


def validation_verdict(bt: dict) -> tuple:
    """ตัดสินตามเป้าของต้นฉบับเอง: 30-50 เทรด + PF หลัง swap > 1.5"""
    n = bt.get("n", 0)
    if n == 0:
        return "fail", "ยังไม่มีเทรดเลยในช่วงข้อมูลนี้ — ตัดสินอะไรไม่ได้"
    pf = bt.get("pf_after_swap", float("nan"))
    if n < 30:
        return "fail", (f"มี {n} เทรด (< 30 ตามเป้า validate ของสคริปต์เอง) — "
                        "ตัวเลขยังเป็นเสียงรบกวน ใช้ประกอบเท่านั้น ห้ามสรุป")
    msg = (f"n={n} (เข้าเป้า 30-50) | PF หลัง swap = "
           + ("∞" if not np.isfinite(pf) else f"{pf:.2f}")
           + f" เทียบเป้า > 1.5")
    if np.isfinite(pf) and pf > 1.5:
        return "warn", msg + " — ผ่านเป้าใน backtest แต่เป็น in-sample: " \
                             "ต้อง journal จริงต่อ ห้ามถือเป็นข้อพิสูจน์"
    return "fail", msg + " — ไม่ผ่านเป้าของสคริปต์เอง"
