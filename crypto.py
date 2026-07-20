# -*- coding: utf-8 -*-
"""
crypto.py — พอร์ต Crypto Research Toolkit v6 (Pine) + สรุปหลักฐานจากรายงาน
"Cryptocurrency and Bitcoin: A Balanced, Evidence-Based Report (2008-2026)"

ตามต้นฉบับเป๊ะ: นี่คือ 'เครื่องมือบริบท' — **ไม่ใช่ระบบเข้าออก** ทุกตัวชี้วัดเป็น
heuristic จากตัวอย่างประวัติศาสตร์เล็ก (~4 วัฏจักร) เกณฑ์คาลิเบรตกับ BTC daily
เท่านั้น (ไม่ใช้กับเหรียญอื่น/intraday) | ตัวเลข on-chain ในกล่องหลักฐานเป็น
snapshot จากรายงาน (ต้นปี 2026) ไม่ใช่ค่าดึงสด
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import gold as G

HALVING_2024 = pd.Timestamp("2024-04-20")


@dataclass
class CryptoParams:
    reg_len: int = 200
    mayer_hot: float = 2.4
    mayer_val: float = 0.8
    pi_fast: int = 111
    pi_slow: int = 350
    rsi_len: int = 14
    vol_len: int = 30
    bars_year: int = 365


def compute(df: pd.DataFrame, p: CryptoParams = None) -> pd.DataFrame:
    p = p or CryptoParams()
    fr = df.copy()
    c, hgh = fr["Close"], fr["High"]
    fr["reg_sma"] = c.rolling(p.reg_len).mean()
    fr["mayer"] = c / fr["reg_sma"].replace(0, np.nan)
    fr["pi_fast"] = c.rolling(p.pi_fast).mean()
    fr["pi_slow2"] = 2.0 * c.rolling(p.pi_slow).mean()
    fr["pi_cross"] = (fr["pi_fast"] > fr["pi_slow2"]) & \
        (fr["pi_fast"].shift(1) <= fr["pi_slow2"].shift(1))
    fr["rsi"] = G.rsi_wilder(c, p.rsi_len)
    logret = np.log(c / c.shift(1))
    fr["ann_vol"] = logret.rolling(p.vol_len).std() * math.sqrt(p.bars_year) * 100
    fr["ath"] = hgh.cummax()
    fr["dd_ath"] = (c / fr["ath"] - 1.0) * 100
    fr["bull"] = c > fr["reg_sma"]
    return fr


def state(fr: pd.DataFrame, is_btc: bool, p: CryptoParams = None) -> dict:
    p = p or CryptoParams()
    r = fr.iloc[-1]
    mayer = float(r["mayer"]) if r["mayer"] == r["mayer"] else None
    zone = None
    if mayer is not None:
        zone = ("OVERHEATED" if mayer > p.mayer_hot else
                "VALUE" if mayer < p.mayer_val else "NEUTRAL")
    pi_gap = None
    if is_btc and r["pi_slow2"] == r["pi_slow2"] and r["pi_slow2"] > 0:
        pi_gap = float((r["pi_fast"] / r["pi_slow2"] - 1) * 100)
    recent_cross = bool(fr["pi_cross"].tail(30).any()) if is_btc else None
    halv_days = int((fr.index[-1].normalize() - HALVING_2024).days) if is_btc else None
    return {"price": float(r["Close"]),
            "regime": "BULL" if bool(r["bull"]) else "BEAR",
            "mayer": None if mayer is None else round(mayer, 2),
            "mayer_zone": zone,
            "rsi": round(float(r["rsi"]), 1) if r["rsi"] == r["rsi"] else None,
            "ann_vol": round(float(r["ann_vol"]), 1)
            if r["ann_vol"] == r["ann_vol"] else None,
            "dd_ath": round(float(r["dd_ath"]), 1),
            "ath": round(float(r["ath"]), 0),
            "pi_gap": None if pi_gap is None else round(pi_gap, 1),
            "pi_recent_cross": recent_cross,
            "halv_days": halv_days}


ALT_CAVEAT = ("เกณฑ์ Mayer 2.4/0.8, Pi Cycle และวันหลัง halving คาลิเบรตกับ "
              "**BTC รายวันเท่านั้น** — สำหรับ ETH/SOL แสดงเฉพาะ regime/RSI/"
              "ความผันผวน/ระยะจาก ATH และค่า Mayer เป็น 'ตัวเลขดิบ' ห้ามใช้โซน")

# ---------------------------------------------------------------------------
# สรุปหลักฐานจากรายงานของผู้ใช้ (snapshot กลางปี 2026 — ไม่ใช่ข้อมูลสด)
# ---------------------------------------------------------------------------
EVIDENCE_TWO_SIDES = [
    {"ประเด็น": "Diversifier",
     "ฝั่งสนับสนุน": "จัดสรร 1-5% เพิ่ม Sharpe อย่างมีนัย (หลายงานอิสระ; "
     "5% ใน 60/40 → Sharpe 0.63→1.15 ช่วง 2014-23)",
     "ฝั่งค้าน": "correlation กับหุ้น 'พุ่งขึ้น' ตอน stress — จึงไม่ใช่ "
     "crisis hedge / ไม่ใช่ทองดิจิทัลในวิกฤต"},
    {"ประเด็น": "มูลค่าพื้นฐาน",
     "ฝั่งสนับสนุน": "scarcity 21M + network effect + censorship-resistance",
     "ฝั่งค้าน": "งานวิชาการหลายชิ้นสรุป 'fundamental value = 0' และพฤติกรรม"
     "ฟองสบู่ซ้ำ ≥4 ครั้ง (Wheatley/Sornette)"},
    {"ประเด็น": "โมเดลราคา",
     "ฝั่งสนับสนุน": "on-chain (MVRV-Z, NUPL, SOPR) ชี้โซนสุดขั้วได้หยาบๆ",
     "ฝั่งค้าน": "S2F ล้มเหลวชัด (~$500k vs ราคาจริง ~$70k) | Metcalfe โดนวิจารณ์ "
     "spurious regression"},
    {"ประเด็น": "วัฏจักร halving",
     "ฝั่งสนับสนุน": "peak ต.ค. 2025 ยัง 'เข้าแพตเทิร์น' 12-18 เดือนหลัง halving; "
     "NUPL เย็นลงแบบคลาสสิก",
     "ฝั่งค้าน": "2025 = ปีแรกหลัง halving ที่ปิดแดง; ETF/มหภาคใหญ่กว่า supply "
     "ของเหมืองแล้ว — 'อ่อนแรงลง ไม่ตายสนิท'"},
    {"ประเด็น": "DCA vs Lump-sum",
     "ฝั่งสนับสนุน": "Lump-sum ชนะเชิงคณิต ~66% ของเวลา",
     "ฝั่งค้าน": "DCA ชนะเชิงพฤติกรรม + drawdown ต่ำกว่า และชนะจริงในตลาดหมี "
     "2018-19, 2022 — เป็นกลยุทธ์ที่คนทำได้จริง"},
]

REPORT_SNAPSHOT = [
    ("Peak ล่าสุด", "~$126,000 (ต.ค. 2025) → แก้ตัว 40-50% เหลือ $60-80k กลางปี 2026"),
    ("MVRV Z-Score", "~1.32 (ม.ค. 2026) = โซน 'fair value' — ห่างจาก ~7 ที่ยอด 2017/2021"),
    ("NUPL", "~19% (ก.พ. 2026) — เย็นลงจากยอด ต.ค. 2025"),
    ("Drawdown ต่อวัฏจักร", "หดลง ~94% → ~84% → ~77% (สอดคล้อง maturation)"),
    ("ETF era", "IBIT แตะ $100B AUM เร็วสุดในประวัติศาสตร์ ถือ >800k BTC (~3.8% supply)"),
]

BENCH_BULL = [
    "ETF net inflow กลับมาต่อเนื่อง",
    "MVRV Z ทรงในแถบ 1-3 (fair value) ขณะราคาสร้างฐาน",
    "กฎหมาย market-structure สหรัฐชัด (คลายแรงกดกำกับดูแล)",
]
BENCH_BEAR = [
    "MVRV Z พุ่ง >6-7 หรือ NUPL >0.75 (โซน euphoria = ความเสี่ยงยอด)",
    "Stablecoin หลัก (USDT) หลุด peg หรือเจอ run",
    "หมุดหมาย quantum-cryptanalysis ที่ย่น Q-Day อย่างมีนัย",
    "Correlation กับหุ้นค้างสูงระหว่าง risk-off ใหญ่ (ล้มธีสิส diversification)",
]

RISKS = [
    "สแกม/ฉ้อโกง $17B ในปี 2025 (ยอดเฉลี่ยต่อเคส +253%, AI ทำให้กำไรต่อเคส ×4.5)",
    "USDT: S&P ลดคะแนนเสถียรภาพเหลือ 5 (weak) พ.ย. 2025 — สินทรัพย์เสี่ยงในทุนสำรอง 24%",
    "Leverage: liquidation cascade ~$19B ในวันเดียว (10 ต.ค. 2025)",
    "ประวัติ custodian ล้ม: Mt.Gox, FTX, Celsius ฯลฯ — 'not your keys, not your coins'",
    "Quantum: ไทม์ไลน์หดลง (งานวิจัย 2026 ชี้ใช้ qubits น้อยกว่าที่คิด) — ความเสี่ยงระยะยาวจริง",
]

TOOL_DISCLAIMER = (
    "Toolkit นี้คือกรอบอ่านบริบท (regime / valuation proxy / top-risk heuristic / "
    "vol / DD / อายุวัฏจักร) — ไม่พยากรณ์ ไม่ใช่คำแนะนำ | Pi Cycle = ความน่าเชื่อถือ "
    "'ผสม' ตามรายงาน | ข้อเสนอเชิงหลักฐานจากรายงาน: ขนาด 1-5% ของพอร์ต, "
    "ไซส์ให้รอด drawdown 80%, ใช้ DCA อัตโนมัติสู้จิตวิทยา, เลี่ยง leverage/เหรียญเล็ก")
