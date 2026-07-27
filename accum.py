# -*- coding: utf-8 -*-
"""ACCUMULATION-FOOTPRINT WATCH + SQUEEZE — นิยาม "ชุดเดียว" ใช้ร่วมกันทั้ง
หุ้นไทย (set_swing.py) และทองคำ (gold.py) เพื่อไม่ให้สองที่คำนวณเพี้ยนจากกัน.

ที่มาของสูตร: SET Swing v5.13 (Pine) หัวข้อ
  * "SQUEEZE PRECONDITION (TTM-style)"       บรรทัด ~1193-1201
  * "ACCUMULATION-FOOTPRINT WATCH (v5.8)"    บรรทัด ~1508-1542

ป้ายกำกับของ "ต้นฉบับเอง" (ยกมาตรง ๆ ไม่ตกแต่ง ไม่ขยายความ):
  * accumulation watch = **DISPLAY ONLY, grade C proxy**
    "NEVER enters, exits, sizes or gates"
    และต้นฉบับระบุเองว่า เครื่องมือสาย OBV/AD หลักฐาน weak/mixed,
    Wyckoff เป็น anecdotal, และออร์เดอร์ VWAP/POV ที่ทำดี ๆ **ตรวจไม่เจอ**
  * squeeze edge "decayed to ~1 of 14 index markets after 2001"
    (Fang-Jacobsen-Qin, JPM 2017) → ต้นฉบับปิด useSqz เป็นค่าตั้งต้นตั้งแต่ v5.8

ข้อจำกัดที่ยังไม่ถูก validate (แจ้งไว้ตรงนี้ ไม่ซ่อน):
  - ยังไม่มีการทดสอบเชิงสถิติว่า "แท่งที่ขึ้นสะสม" ให้ผลตอบแทนต่างจากแท่งอื่น
    ทั้งบนหุ้นไทยและบนทองคำ — ตารางนี้จึงเป็น "คิวเฝ้าดู" ไม่ใช่สัญญาณ
  - บนทองคำยิ่งอ่อนกว่าหุ้น เพราะวอลุ่มที่ใช้เป็นวอลุ่ม proxy (ดู volume_quality)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# ค่าตั้งต้น = ตรงกับ input ของ Pine v5.13 หัวข้อ 18
# --------------------------------------------------------------------------
ACC_LEN_DEF = 20          # accLen   — Footprint window (bars)
ACC_FLAT_DEF = 2.0        # accFlat  — Max net price move in ATR units
ACC_RATIO_DEF = 1.25      # accRatio — Min up-vol / down-vol ratio
ACC_CLV_THR = 0.10        # ค่าคงที่ในสคริปต์ (ไม่ใช่ input)
ACC_ACT_THR = 0.7         # ค่าคงที่ในสคริปต์ (ไม่ใช่ input)
ACC_POS_MAX = 0.65        # ค่าคงที่ในสคริปต์ (ไม่ใช่ input)
ACC_VOTES_MIN = 3         # ต้อง >= 3 จาก 4
ACC_PERSIST = 2           # ต้องติดกันกี่แท่งจึงพิมพ์เครื่องหมาย (accShow)
VOL_BASE_LEN = 100        # len100Eff — ฐานวอลุ่มเทียบ
SQ_LEN = 20               # len20Eff — BB/KC length
SQ_BB_MULT = 2.0
SQ_KC_MULT = 1.5

VOTE_NAMES = ["ราคานิ่ง", "วอลุ่มขาซื้อเด่น", "ปิดค่อนบน", "ตลาดไม่ตาย"]
VOTE_KEYS = ["acc_flat_ok", "acc_press_ok", "acc_clv_ok", "acc_act_ok"]
VOTE_NEEDS_VOLUME = [False, True, False, True]   # โหวตข้อ 2 และ 4 ใช้วอลุ่ม

MARKER_NOTE = (
    "เครื่องหมายบนชาร์ต Pine: สี่เหลี่ยมเหลืองแถวล่าง = accShow "
    "(โหวต ≥3/4 ติดกัน 2 แท่ง + อยู่ใต้ trigger + ครึ่งล่างของกรอบ) · "
    "วงกลมฟ้าแถวล่าง = squeezeOn (แท่งที่บีบตัวอยู่จริง ๆ เท่านั้น)"
)


# --------------------------------------------------------------------------
def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def atr_wilder(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c1 = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h - l, (h - c1).abs(), (l - c1).abs()], axis=1).max(axis=1)
    return _rma(tr, n)


def squeeze_frame(df: pd.DataFrame, sq_len: int = SQ_LEN,
                  bb_mult: float = SQ_BB_MULT,
                  kc_mult: float = SQ_KC_MULT) -> pd.DataFrame:
    """TTM squeeze: BB(20,2) อยู่ข้างใน KC(20,1.5×ATR) → squeezeOn.

    ตรงกับ Pine: bbBasis=sma, bbDev=2*ta.stdev (biased/population),
    kcMid=ema, kcRng=1.5*ta.atr(20).
    """
    c = df["Close"]
    out = pd.DataFrame(index=df.index)
    bb_basis = c.rolling(sq_len).mean()
    bb_dev = bb_mult * c.rolling(sq_len).std(ddof=0)      # Pine ta.stdev = population
    kc_mid = c.ewm(span=sq_len, adjust=False).mean()
    kc_rng = kc_mult * atr_wilder(df, sq_len)
    out["squeeze_on"] = ((bb_basis + bb_dev < kc_mid + kc_rng)
                         & (bb_basis - bb_dev > kc_mid - kc_rng)).fillna(False)
    pos = np.arange(len(df), dtype=float)
    last = pd.Series(np.where(out["squeeze_on"].to_numpy(), pos, np.nan),
                     index=df.index).ffill()
    out["bars_sq"] = pos - last          # NaN = ยังไม่เคยเกิด squeeze ในข้อมูลที่มี
    return out


def volume_quality(v: pd.Series, n: int = VOL_BASE_LEN) -> pd.Series:
    """สัดส่วนแท่งที่มีวอลุ่มใช้ได้จริง (ไม่ NaN และ > 0) ใน n แท่งล่าสุด.

    ใช้ตัดสินว่าโหวตที่พึ่งวอลุ่ม (ข้อ 2, 4) "วัดได้" หรือ "วัดไม่ได้"
    — สำคัญมากกับสินทรัพย์อย่างทองคำที่วอลุ่มเป็น proxy หรือไม่มีเลย
    """
    ok = v.notna() & (v > 0)
    return ok.rolling(n, min_periods=1).mean()


def accumulation_frame(df: pd.DataFrame, atr: pd.Series, swing_hi: pd.Series,
                       acc_len: int = ACC_LEN_DEF,
                       acc_flat: float = ACC_FLAT_DEF,
                       acc_ratio: float = ACC_RATIO_DEF,
                       vol_base_len: int = VOL_BASE_LEN,
                       use_acc: bool = True) -> pd.DataFrame:
    """โหวต 4 ข้อของ accumulation-footprint watch — ตรงตาม Pine v5.13.

    1) ราคานิ่ง      |close - close[accLen]| <= accFlat * ATR
    2) วอลุ่มขาซื้อเด่น  sum(vol วันบวก) >= accRatio * sum(vol วันลบ) และวันลบ > 0
    3) ปิดค่อนบน      sma(CLV, accLen) >= 0.10
    4) ตลาดไม่ตาย     sma(vol, accLen) / max(sma(vol, 100), 1) >= 0.7
    บริบท (accCtx)   close < swingHi  และ  posInRng <= 0.65
    accHot  = use_acc และ ctx และ โหวต >= 3
    accShow = accHot ติดกัน 2 แท่ง  ← ตัวนี้คือสี่เหลี่ยมเหลืองบนชาร์ต
    """
    c, h, l = df["Close"], df["High"], df["Low"]
    v = df["Volume"] if "Volume" in df.columns \
        else pd.Series(np.nan, index=df.index, dtype=float)
    out = pd.DataFrame(index=df.index)

    up_vol = v.where(c > c.shift(1), 0.0).rolling(acc_len).sum()
    dn_vol = v.where(c < c.shift(1), 0.0).rolling(acc_len).sum()
    rng_hl = h - l
    clv_raw = pd.Series(np.where(rng_hl > 0, ((c - l) - (h - c)) / rng_hl, 0.0),
                        index=df.index)
    clv_avg = clv_raw.rolling(acc_len).mean()
    vol_act = v.rolling(acc_len).mean() / v.rolling(vol_base_len).mean().clip(lower=1.0)
    rng_hi_a = h.rolling(acc_len).max()
    rng_lo_a = l.rolling(acc_len).min()

    out["pos_in_rng"] = pd.Series(
        np.where(rng_hi_a > rng_lo_a, (c - rng_lo_a) / (rng_hi_a - rng_lo_a), 0.5),
        index=df.index)
    out["acc_flat_ok"] = ((c - c.shift(acc_len)).abs() <= acc_flat * atr).fillna(False)
    out["acc_press_ok"] = ((dn_vol > 0) & (up_vol >= acc_ratio * dn_vol)).fillna(False)
    out["acc_clv_ok"] = (clv_avg >= ACC_CLV_THR).fillna(False)
    out["acc_act_ok"] = (vol_act >= ACC_ACT_THR).fillna(False)
    out["acc_votes"] = sum(out[k].astype(int) for k in VOTE_KEYS)
    ctx = ((c < swing_hi) & (out["pos_in_rng"] <= ACC_POS_MAX)).fillna(False)
    out["acc_ctx"] = ctx
    out["acc_hot"] = bool(use_acc) & ctx & (out["acc_votes"] >= ACC_VOTES_MIN)
    out["acc_show"] = out["acc_hot"] & out["acc_hot"].shift(1).fillna(False)

    # ค่าดิบไว้ตรวจสอบบนจอ (ไม่ได้ใช้ตัดสินใจ — ใช้ให้ผู้ใช้เทียบกับ TradingView ได้)
    out["_net_move_atr"] = (c - c.shift(acc_len)).abs() / atr.replace(0, np.nan)
    out["_updn_ratio"] = up_vol / dn_vol.replace(0, np.nan)
    out["_clv_avg"] = clv_avg
    out["_vol_act"] = vol_act
    out["vol_quality"] = volume_quality(v, vol_base_len)
    return out


def acc_squeeze(df: pd.DataFrame, acc_len: int = ACC_LEN_DEF,
                acc_flat: float = ACC_FLAT_DEF,
                acc_ratio: float = ACC_RATIO_DEF,
                atr_len: int = 14, bos_len: int = 20,
                use_acc: bool = True,
                vol_base_len: int = VOL_BASE_LEN) -> pd.DataFrame:
    """ชุดเต็ม (squeeze + accumulation) สำหรับสินทรัพย์ที่ยังไม่มี frame อยู่ก่อน
    เช่นทองคำ. หุ้นไทยเรียกผ่าน set_swing.compute_frame ซึ่งใช้ฟังก์ชันย่อย
    ชุดเดียวกันนี้ (ATR/swing_hi คำนวณจากพารามิเตอร์ของกลยุทธ์นั้น ๆ)
    """
    atr = atr_wilder(df, atr_len)
    swing_hi = df["High"].shift(1).rolling(bos_len).max()
    out = accumulation_frame(df, atr, swing_hi, acc_len, acc_flat, acc_ratio,
                             vol_base_len, use_acc)
    out["swing_hi"] = swing_hi
    out["atr"] = atr
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"]
            if c in df.columns]
    return pd.concat([df[keep], out, squeeze_frame(df)], axis=1)


# --------------------------------------------------------------------------
# ตารางตรวจสอบทีละข้อ — ให้ผู้ใช้ทาบกับหน้าจอ TradingView ได้ตรง ๆ
# --------------------------------------------------------------------------
def audit_rows(fr: pd.DataFrame, i: int = -1, acc_len: int = ACC_LEN_DEF,
               acc_flat: float = ACC_FLAT_DEF,
               acc_ratio: float = ACC_RATIO_DEF,
               vol_ok: bool = True) -> list[dict]:
    """คืนรายการ 4 โหวต + บริบท พร้อม 'ค่าที่วัดได้' และ 'เกณฑ์' ของแท่งที่ i."""
    r = fr.iloc[i]

    def num(x, nd=2, suf=""):
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return "—"
        return "—" if xf != xf else f"{xf:.{nd}f}{suf}"

    rows = [
        {"ข้อ": "1) ราคานิ่ง", "ค่าที่วัดได้": num(r.get("_net_move_atr"), 2, " ATR"),
         "เกณฑ์": f"≤ {acc_flat} ATR", "ผ่าน": bool(r.get("acc_flat_ok", False)),
         "ใช้วอลุ่ม": "ไม่"},
        {"ข้อ": "2) วอลุ่มขาซื้อเด่น", "ค่าที่วัดได้": num(r.get("_updn_ratio"), 2, "×"),
         "เกณฑ์": f"≥ {acc_ratio}× (และวันลบต้อง > 0)",
         "ผ่าน": bool(r.get("acc_press_ok", False)), "ใช้วอลุ่ม": "ใช่"},
        {"ข้อ": "3) ปิดค่อนบน (CLV)", "ค่าที่วัดได้": num(r.get("_clv_avg"), 3),
         "เกณฑ์": f"≥ {ACC_CLV_THR}", "ผ่าน": bool(r.get("acc_clv_ok", False)),
         "ใช้วอลุ่ม": "ไม่"},
        {"ข้อ": "4) ตลาดไม่ตาย", "ค่าที่วัดได้": num(r.get("_vol_act"), 2, "×"),
         "เกณฑ์": f"≥ {ACC_ACT_THR}× ฐาน {VOL_BASE_LEN} แท่ง",
         "ผ่าน": bool(r.get("acc_act_ok", False)), "ใช้วอลุ่ม": "ใช่"},
        {"ข้อ": "บริบท: อยู่ใต้ trigger", "ค่าที่วัดได้": num(r.get("swing_hi"), 2),
         "เกณฑ์": f"close {num(r.get('Close'), 2)} ต้องต่ำกว่า",
         "ผ่าน": bool(r.get("Close", np.nan) < r.get("swing_hi", np.nan)),
         "ใช้วอลุ่ม": "ไม่"},
        {"ข้อ": "บริบท: ครึ่งล่างของกรอบ",
         "ค่าที่วัดได้": num(float(r.get("pos_in_rng", np.nan)) * 100, 0, "%"),
         "เกณฑ์": f"≤ {ACC_POS_MAX * 100:.0f}%",
         "ผ่าน": bool(r.get("pos_in_rng", 1.0) <= ACC_POS_MAX), "ใช้วอลุ่ม": "ไม่"},
    ]
    if not vol_ok:
        for k in (1, 3):
            rows[k]["ผ่าน"] = None
            rows[k]["ค่าที่วัดได้"] = "วัดไม่ได้ (วอลุ่มไม่น่าเชื่อถือ)"
    return rows


def votes_measurable(vol_ok: bool) -> int:
    """จำนวนโหวตที่ 'วัดได้จริง' — 4 ถ้ามีวอลุ่มใช้ได้, 2 ถ้าไม่มี."""
    return 4 if vol_ok else 2


def status_label(acc_show: bool, acc_hot: bool, sq_on: bool,
                 bars_sq: float, sq_win: int = 6) -> tuple[str, bool]:
    """คืน (ป้ายสถานะ, มีเครื่องหมายบนชาร์ต Pine หรือไม่).

    ความซื่อสัตย์: มีแค่ 2 สถานะเท่านั้นที่ "มีเครื่องหมายจริงบนชาร์ต" คือ
    acc_show (สี่เหลี่ยมเหลือง) และ squeeze_on (วงกลมฟ้า).
    สถานะที่เหลือเป็นของหน้าจอนี้เอง ไม่มีอะไรให้ทาบบนชาร์ต.
    """
    recent = (bars_sq == bars_sq) and (0 < bars_sq <= sq_win)
    if acc_show and sq_on:
        return "🟣 สะสม + สควีซพร้อมกัน", True
    if sq_on:
        return "🔵 สควีซอยู่ (บีบตัว)", True
    if acc_show:
        return "🟡 สะสม (footprint)", True
    if acc_hot:
        return "⚪ สะสมแท่งแรก (ยังไม่ครบ 2 แท่ง)", False
    if recent:
        return "🟠 เพิ่งคลายสควีซ (≤%d แท่ง)" % sq_win, False
    return "—", False
