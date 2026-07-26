# -*- coding: utf-8 -*-
"""
gold_council.py — "Gold Council / War Room" สำหรับ XAU (ทำตามภาพ แต่แก้จุดที่ทำให้เข้าใจผิด)

สิ่งที่ทำตามภาพ:
- 10 specialists แบ่ง 3 กลุ่ม (TECHNICAL / FUNDAMENTAL / RISK)
- แถบคะแนนรายคน, ป้าย BUY/NEUTRAL/SELL, กล่อง CHIEF VERDICT, RISK CONTROL gate
- กระดานนับเสียง BUY / NEUTRAL / SELL

สิ่งที่ **จงใจไม่ทำตามภาพ** และเหตุผล (สำคัญกว่าหน้าตา):

1) ภาพแสดง "BUY 4.68 / NEUTRAL 1.25 / SELL 0.00" — คะแนนถ่วงน้ำหนักทศนิยม 2 ตำแหน่ง
   จากความเห็น LLM ให้ความรู้สึกว่าเป็นการวัดที่ละเอียด ทั้งที่น้ำหนัก (×2) ตั้งเอง
   และความเห็นแต่ละคนไม่อิสระต่อกัน → โมดูลนี้แสดง **จำนวนนับเป็นจำนวนเต็ม**
   (เช่น 6 BUY / 4 NEUTRAL / 0 SELL) และถ้าเปิดถ่วงน้ำหนักจะติดป้ายว่า
   "น้ำหนักที่ผู้ใช้ตั้งเอง ไม่ใช่ค่าที่ validate แล้ว ไม่ใช่ความน่าจะเป็น"

2) ภาพแสดง "CONFIDENCE 40%" — ตัวเลขความมั่นใจที่ LLM เขียนเองไม่ใช่ความน่าจะเป็น
   ที่ calibrate แล้ว → เราไม่แสดงเป็น % เดี่ยว แต่แสดง (ก) เช็คลิสต์กติกา v6.4
   ที่ผ่านจริงกี่ข้อจากกี่ข้อ (คำนวณด้วย Python) และ (ข) ช่วง conf ต่ำ-สูงของทีม

3) ภาพแสดง "VALIDATED ✓ PASS" โดยไม่บอกว่า validate กับอะไร → ที่นี่ PASS/VETO
   คำนวณด้วย **Python ล้วน** จากเกตจริงของ gold.py (vol shock, cost gate, regime,
   ER, DXY veto) + R:R ขั้นต่ำ ไม่ใช่ AI ตรวจตัวเอง

4) **สภาที่ปรึกษาสร้างไม้เข้าเองไม่ได้** — verdict คำนวณตามกฎที่ประกาศล่วงหน้า:
   ถ้ากติกา v6.4 ยังไม่ครบ = NO TRADE เสมอ ต่อให้ทีมเชียร์ BUY ทั้งสภา
   สภามีอำนาจ "เพิ่มความระมัดระวัง" ได้อย่างเดียว ไม่มีอำนาจ "อนุมัติเข้า"

5) specialist ที่ **ไม่มีแหล่งข้อมูลจริงในระบบนี้** จะถูกบังคับให้งดออกเสียง
   ไม่ใช่ให้ LLM เดา (ดู SPECIALISTS: have=False)
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

import stock_meeting as SM

VERSION = "v1.0"

LEANS = ("BUY", "NEUTRAL", "SELL")

# ---------------------------------------------------------------------------
# ทะเบียน specialist + แหล่งข้อมูลจริง + เกรดหลักฐาน
# grade: A = คำนวณจากราคาโดยตรง มีนิยามชัด | B = คำนวณได้แต่เป็น proxy
#        C = proxy อ่อน | D = เป็นการเล่าเรื่อง | — = ไม่มีข้อมูลในระบบ
# ---------------------------------------------------------------------------
SPECIALISTS: list[dict] = [
    {"id": "trend", "th": "Trend", "group": "TECHNICAL", "grade": "A",
     "have": True, "source": "gold.py: regime_up/dn, reg_sma + slope",
     "role": "ทิศทางหลักตาม regime และความชันของ SMA — เทรนด์จริงหรือเด้งในกรอบ",
     "caveat": ""},
    {"id": "structure", "th": "Structure/SMC", "group": "TECHNICAL", "grade": "C",
     "have": True, "source": "คำนวณ swing high/low + BOS จาก OHLC ในโมดูลนี้",
     "role": "โครงสร้างราคา: higher-high/higher-low, break of structure",
     "caveat": "ระบบนี้**ไม่มีข้อมูล order flow / liquidity / order book** — "
               "'SMC' ที่นี่คือโครงสร้างราคาล้วน ไม่ใช่ smart-money footprint จริง"},
    {"id": "momentum", "th": "Momentum", "group": "TECHNICAL", "grade": "A",
     "have": True, "source": "RSI (Wilder), Efficiency Ratio",
     "role": "แรงส่ง: RSI อยู่โซนไหน ER บอกว่าเป็นเทรนด์หรือ chop",
     "caveat": ""},
    {"id": "volatility", "th": "Volatility", "group": "TECHNICAL", "grade": "A",
     "have": True, "source": "ATR%, vol_rank (percentile 252 แท่ง)",
     "role": "ความผันผวนเทียบอดีต — สูงเกินจนคุมความเสี่ยงไม่ได้หรือไม่",
     "caveat": ""},
    {"id": "volume", "th": "Volume/Liq", "group": "TECHNICAL", "grade": "C",
     "have": True, "source": "Volume ของสัญลักษณ์ที่เลือกเท่านั้น",
     "role": "ปริมาณเทียบค่าเฉลี่ย — ยืนยันการเคลื่อนไหวหรือไม่",
     "caveat": "PAXG = ปริมาณเทรด**โทเคน** ไม่ใช่ปริมาณตลาดทอง · GC=F = "
               "ฟิวเจอร์ส COMEX เท่านั้น ไม่ครอบคลุม spot OTC ซึ่งเป็นตลาดหลัก "
               "→ อ่านเป็นสัญญาณยืนยันไม่ได้เต็มปาก"},
    {"id": "news", "th": "News/Macro", "group": "FUNDAMENTAL", "grade": "—",
     "have": False, "source": "ไม่มี",
     "role": "ข่าว/มหภาคที่กระทบทองรายวัน",
     "caveat": "**ระบบนี้ไม่มีฟีดข่าวทอง** (data_sources.py มีแค่ FRED + ราคา) "
               "→ บทนี้ถูกบังคับให้งดออกเสียง ไม่ให้ LLM เดาข่าวจากความจำ "
               "ซึ่งจะเป็นข้อมูลเก่าและอาจผิด"},
    {"id": "session", "th": "Session", "group": "FUNDAMENTAL", "grade": "B",
     "have": True, "source": "คำนวณจากเวลาจริง (Asia / London / NY overlap)",
     "role": "ช่วงตลาด — สภาพคล่องและพฤติกรรมต่างกันตามช่วง",
     "caveat": "คำนวณด้วย Python ไม่ต้องใช้ LLM · เป็นบริบท ไม่ใช่สัญญาณเข้า",
     "llm": False},
    {"id": "sentiment", "th": "Sentiment", "group": "FUNDAMENTAL", "grade": "—",
     "have": False, "source": "ไม่มี",
     "role": "การวางสถานะของตลาด (COT, ETF flow, positioning)",
     "caveat": "**ไม่มี COT / ETF flow / positioning ในระบบ** → บังคับงดออกเสียง"},
    {"id": "correlation", "th": "Correlation", "group": "FUNDAMENTAL", "grade": "B",
     "have": True, "source": "DXY(50), US10Y(50), USDJPY, VIX จาก gold.py",
     "role": "ดอลลาร์/ยีลด์หนุนหรือต้านทอง — DXY veto ติดหรือไม่",
     "caveat": "ความสัมพันธ์ทอง-DXY ไม่คงที่ตลอดเวลา (สลับเครื่องหมายได้ในบางช่วง)"},
    {"id": "pattern", "th": "Pattern", "group": "RISK", "grade": "D",
     "have": True, "source": "ราคา OHLC ล่าสุด (ให้ LLM บรรยาย)",
     "role": "รูปแบบราคาที่มองเห็น",
     "caveat": "**เกรด D** — LLM บรรยายรูปแบบจากตัวเลขคือการเล่าเรื่อง ไม่ใช่ "
               "detector ที่ backtest ได้ ห้ามใช้เป็นเหตุผลหลัก "
               "(เช่น คำว่า 'double bottom' เกิดขึ้นได้เสมอเมื่อมองย้อนหลัง)"},
]

NO_DATA = [s["id"] for s in SPECIALISTS if not s["have"]]
DEFAULT_PANEL = [s["id"] for s in SPECIALISTS]
GROUPS = ["TECHNICAL", "FUNDAMENTAL", "RISK"]


def spec(sid: str) -> dict:
    return next((s for s in SPECIALISTS if s["id"] == sid), {})


# ---------------------------------------------------------------------------
# ตัวเลขที่คำนวณเองด้วย Python (ไม่ผ่าน LLM)
# ---------------------------------------------------------------------------

def session_of(ts) -> dict:
    """ช่วงตลาดจาก UTC — คำนวณตรง ๆ ไม่ต้องถาม AI

    Asia ~00-07 UTC, London ~07-16, NY ~12-21, overlap London/NY ~12-16
    """
    try:
        t = pd.Timestamp(ts)
        h = int(t.tz_convert("UTC").hour) if t.tzinfo else int(t.hour)
    except Exception:
        return {"ช่วง": "ไม่ทราบ", "overlap": False, "หมายเหตุ": "อ่านเวลาไม่ได้"}
    if 12 <= h < 16:
        return {"ช่วง": "London/NY overlap", "overlap": True,
                "หมายเหตุ": "สภาพคล่องสูงสุดของวัน ความผันผวนมักสูงตาม"}
    if 7 <= h < 12:
        return {"ช่วง": "London", "overlap": False, "หมายเหตุ": "สภาพคล่องดี"}
    if 16 <= h < 21:
        return {"ช่วง": "NY (หลัง London ปิด)", "overlap": False,
                "หมายเหตุ": "สภาพคล่องลดลงหลัง London ปิด"}
    return {"ช่วง": "Asia / ปลายวัน", "overlap": False,
            "หมายเหตุ": "สภาพคล่องบางกว่า — สเปรดมักกว้างขึ้น"}


def swing_structure(df: pd.DataFrame, n: int = 10) -> dict:
    """โครงสร้างราคาจาก fractal swing แบบง่าย — นิยามชัด ตรวจซ้ำได้

    ไม่ใช่ 'SMC' ตามสำนักไหน แต่เป็น higher-high/higher-low ที่วัดได้
    """
    if df is None or len(df) < n * 3:
        return {"พอข้อมูล": False}
    h, l = df["High"], df["Low"]
    hh = h[(h == h.rolling(2 * n + 1, center=True).max())].dropna()
    ll = l[(l == l.rolling(2 * n + 1, center=True).min())].dropna()
    if len(hh) < 2 or len(ll) < 2:
        return {"พอข้อมูล": False}
    last_h, prev_h = float(hh.iloc[-1]), float(hh.iloc[-2])
    last_l, prev_l = float(ll.iloc[-1]), float(ll.iloc[-2])
    c = float(df["Close"].iloc[-1])
    if last_h > prev_h and last_l > prev_l:
        st = "higher-high + higher-low (โครงสร้างขาขึ้น)"
    elif last_h < prev_h and last_l < prev_l:
        st = "lower-high + lower-low (โครงสร้างขาลง)"
    else:
        st = "ผสม (ไม่มีโครงสร้างชัด)"
    return {"พอข้อมูล": True, "โครงสร้าง": st,
            "swing_high_ล่าสุด": round(last_h, 2),
            "swing_high_ก่อนหน้า": round(prev_h, 2),
            "swing_low_ล่าสุด": round(last_l, 2),
            "swing_low_ก่อนหน้า": round(prev_l, 2),
            "BOS_ขึ้น": bool(c > last_h), "BOS_ลง": bool(c < last_l),
            "หมายเหตุ": f"fractal {n} แท่งซ้าย-ขวา · ยืนยันย้อนหลัง {n} แท่ง "
                        "(swing ล่าสุดอาจเปลี่ยนได้เมื่อมีแท่งใหม่)"}


def volume_state(df: pd.DataFrame, n: int = 20) -> dict:
    if df is None or "Volume" not in df.columns or len(df) < n + 1:
        return {"พอข้อมูล": False}
    v = df["Volume"].astype(float)
    if float(v.tail(n).sum()) <= 0:
        return {"พอข้อมูล": False, "หมายเหตุ": "ไม่มีค่า volume ในชุดข้อมูลนี้"}
    cur, avg = float(v.iloc[-1]), float(v.tail(n).mean())
    return {"พอข้อมูล": True, "volume_ล่าสุด": round(cur, 0),
            "ค่าเฉลี่ย_20": round(avg, 0),
            "เท่าของค่าเฉลี่ย": round(cur / avg, 2) if avg > 0 else None}


# ---------------------------------------------------------------------------
# เกตความเสี่ยง — Python ล้วน (นี่คือ "RISK CONTROL / VALIDATED" ตัวจริง)
# ---------------------------------------------------------------------------

def risk_gate(state: dict, min_rr: float = 2.0,
              spread_c: float = 25.0) -> dict:
    """คืน {'pass': bool, 'veto': [...], 'checks': [...], 'rr': float|None}

    ทุกข้อมาจากตัวเลข engine ไม่มี LLM เกี่ยวข้อง
    """
    checks, veto = [], []

    def add(name, ok, detail=""):
        checks.append({"ชื่อ": name, "ผ่าน": bool(ok), "รายละเอียด": detail})
        if not ok:
            veto.append(name)

    status = str(state.get("status", ""))
    hard = ("Vol shock", "Carry stress", "Cost gate", "DXY veto")
    for hkey in hard:
        add(f"ไม่ติด {hkey}", hkey not in status,
            status if hkey in status else "")
    add("มี regime ให้เทรด (ไม่ใช่ MIXED)",
        state.get("regime") in ("UP", "DOWN"), f"regime={state.get('regime')}")
    add("กติกา v6.4 ครบ (มีสัญญาณจริง)", bool(state.get("triggered")),
        "ถ้ายังไม่ครบ = ห้ามเข้าไม่ว่าสภาจะเห็นอย่างไร")

    plan = state.get("plan") or {}
    rr = None
    sd = plan.get("stop_dist")
    if sd:
        # R:R เทียบระยะ stop กับ spread ไป-กลับ (¢/oz → $/oz)
        cost = float(spread_c) / 100.0
        rr = round(float(sd) / cost, 2) if cost > 0 else None
        add(f"stop กว้างพอเทียบต้นทุน (≥ {min_rr}×)",
            rr is not None and rr >= float(min_rr),
            f"stop {sd} $/oz ÷ spread {cost:.2f} $/oz = {rr}×")
    else:
        add(f"stop กว้างพอเทียบต้นทุน (≥ {min_rr}×)", False,
            "ยังไม่มีแผน stop เพราะยังไม่มีสัญญาณ")

    return {"pass": not veto, "veto": veto, "checks": checks, "rr": rr}


# ---------------------------------------------------------------------------
# บริบทที่ส่งให้ LLM — ตัวเลขทั้งหมดจาก engine
# ---------------------------------------------------------------------------

def build_context(symbol: str, timeframe: str, df: pd.DataFrame,
                  state: dict, gate: dict, extra: dict | None = None) -> dict:
    r = df.iloc[-1]
    ctx = {
        "สัญลักษณ์": symbol, "timeframe": timeframe,
        "เวลาแท่งล่าสุด": str(df.index[-1]),
        "ราคาปิดล่าสุด": round(float(r["Close"]), 2),
        "engine_v6_4": {
            "regime": state.get("regime"), "status": state.get("status"),
            "score_long": state.get("score_l"), "score_short": state.get("score_s"),
            "rsi": state.get("rsi"), "atr_pct": state.get("atr_pct"),
            "vol_rank_percentile": state.get("vol_rank"),
            "มีสัญญาณตามกติกา": state.get("triggered"),
            "แผน": state.get("plan"),
            "เช็คลิสต์": state.get("checklist"),
        },
        "เกตความเสี่ยง_คำนวณด้วย_python": {
            "ผ่าน": gate["pass"], "ที่ติด": gate["veto"],
            "stop_ต่อ_spread": gate.get("rr"),
        },
        "โครงสร้างราคา": swing_structure(df),
        "ปริมาณ": volume_state(df),
        "ช่วงตลาด": session_of(df.index[-1]),
        "ข้อมูลที่ระบบนี้ไม่มี": {
            "ข่าวทองรายวัน": "ไม่มีฟีด",
            "COT_positioning_ETF_flow": "ไม่มี",
            "order_flow_liquidity_orderbook": "ไม่มี",
            "bid_ask_realtime": "ไม่มี (ข้อมูลเป็นแท่งปิดย้อนหลัง)",
        },
    }
    if extra:
        ctx.update(extra)
    return ctx


IRON_RULES_GOLD = (
    "กติกาเหล็ก (ห้ามละเมิด): (1) ใช้เฉพาะตัวเลขใน context — ห้ามคำนวณใหม่ "
    "ห้ามตั้งราคาเป้า ห้ามอ้างข่าวจากความจำ (ระบบนี้ไม่มีฟีดข่าว) "
    "(2) ทุกบทต้อง **อ้างชื่อตัวเลขที่ใช้** จาก context ทุกครั้ง "
    "(3) บทที่ context บอกว่าไม่มีข้อมูล ต้องตอบ NEUTRAL และเขียนเหตุผลว่า "
    "'ไม่มีข้อมูลในระบบ' ห้ามเดา "
    "(4) lean มีแค่ BUY / NEUTRAL / SELL + conf 0-100 + เหตุผล ≤ 20 คำ "
    "(5) สภา **ไม่มีอำนาจอนุมัติการเข้าไม้** — ถ้ากติกา v6.4 ยังไม่ครบ "
    "ผลลัพธ์คือ NO TRADE เสมอ ความเห็นของคุณใช้เพิ่มความระมัดระวังเท่านั้น "
    "(6) ตอบภาษาไทย กระชับ")


def build_council_prompt(panel_ids: list[str], context_json: str) -> str:
    rows = []
    for sid in panel_ids:
        s = spec(sid)
        if not s or s.get("llm") is False:
            continue
        tag = ("[ไม่มีข้อมูลในระบบ — ต้องตอบ NEUTRAL + 'ไม่มีข้อมูล']"
               if not s["have"] else f"[เกรดหลักฐาน {s['grade']} · {s['source']}]")
        line = f"- {s['th']}: {s['role']} {tag}"
        if s.get("caveat"):
            line += f"\n  ข้อควรระวังที่ต้องพูดถึง: {s['caveat']}"
        rows.append(line)
    return (
        f"คุณจะเล่นบทผู้เชี่ยวชาญ {len(rows)} คนต่อไปนี้ทีละคน:\n"
        + "\n".join(rows) + "\n\n" + IRON_RULES_GOLD + "\n\n"
        "รูปแบบ: บรรทัดละหนึ่งบท `ชื่อบท | LEAN | conf | ตัวเลขที่อ้าง | เหตุผลสั้น`\n"
        "แล้วปิดท้ายด้วยบล็อก JSON เดียวใน ```json ...``` เท่านั้น:\n"
        '{"specialists": {"trend": {"lean": "BUY|NEUTRAL|SELL", "conf": 0-100, '
        '"อ้างอิง": "ชื่อตัวเลขใน context", "เหตุผล": "สั้น"}}, '
        '"ข้อขัดแย้ง": ["ประเด็นที่บทต่าง ๆ มองไม่ตรงกัน"], '
        '"ความเสี่ยงหลัก": ["ข้อ"]}\n'
        "ห้ามมีข้อความหลังบล็อก JSON\n\n"
        f"context (ตัวเลขทั้งหมดจาก engine v6.4):\n{context_json}")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _norm_lean(s) -> str:
    s = str(s or "").strip().upper()
    for v in LEANS:
        if v in s:
            return v
    return "NEUTRAL"


def parse_council(text: str) -> tuple[str, dict | None]:
    if not text:
        return "", None
    m = list(re.finditer(r"```json\s*(\{.*?\})\s*```", text, re.S))
    raw, analysis = None, text
    if m:
        raw, analysis = m[-1].group(1), text[: m[-1].start()].strip()
    else:
        depth, start = 0, None
        for j, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = j
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    raw = text[start: j + 1]
        if raw:
            analysis = text[: text.rfind(raw)].strip()
    if not raw:
        return analysis, None
    try:
        d = json.loads(raw)
    except Exception:
        return analysis, None
    out = {}
    valid = {s["id"] for s in SPECIALISTS}
    for sid, v in (d.get("specialists") or {}).items():
        sid = str(sid).strip().lower()
        if sid not in valid or not isinstance(v, dict):
            continue
        lean = _norm_lean(v.get("lean"))
        # บังคับ: บทที่ไม่มีข้อมูล ต้องเป็น NEUTRAL เสมอ ถึง LLM จะตอบอย่างอื่น
        if sid in NO_DATA:
            lean = "NEUTRAL"
        out[sid] = {"lean": lean, "conf": SM._clamp_conf(v.get("conf")),
                    "อ้างอิง": str(v.get("อ้างอิง", ""))[:120],
                    "เหตุผล": str(v.get("เหตุผล", ""))[:200]}
    return analysis, {
        "specialists": out,
        "ข้อขัดแย้ง": [str(x)[:200] for x in (d.get("ข้อขัดแย้ง") or [])][:8],
        "ความเสี่ยงหลัก": [str(x)[:200] for x in (d.get("ความเสี่ยงหลัก") or [])][:8],
    }


# ---------------------------------------------------------------------------
# กระดานนับเสียง — จำนวนเต็ม ไม่ใช่คะแนนถ่วงทศนิยม
# ---------------------------------------------------------------------------

def tally(parsed: dict, weights: dict | None = None) -> dict:
    sp = (parsed or {}).get("specialists") or {}
    counts = {k: 0 for k in LEANS}
    abstain, confs = [], []
    for sid, v in sp.items():
        counts[v["lean"]] += 1
        confs.append(int(v.get("conf", 0)))
        if sid in NO_DATA:
            abstain.append(spec(sid)["th"])
    out = {"counts": counts, "n": len(sp), "งดออกเสียง_ไม่มีข้อมูล": abstain,
           "conf_ต่ำสุด": min(confs) if confs else None,
           "conf_สูงสุด": max(confs) if confs else None,
           "ถ่วงน้ำหนัก": None}
    if weights:
        w = {k: 0.0 for k in LEANS}
        for sid, v in sp.items():
            w[v["lean"]] += float(weights.get(sid, 1.0))
        out["ถ่วงน้ำหนัก"] = {k: round(x, 2) for k, x in w.items()}
    return out


WEIGHT_WARNING = (
    "คะแนนถ่วงน้ำหนักนี้เป็น **น้ำหนักที่ผู้ใช้ตั้งเอง ไม่ได้ผ่านการ validate** "
    "และความเห็นแต่ละบทมาจากโมเดลเดียวกัน จึงไม่อิสระต่อกัน — "
    "ตัวเลขนี้ **ไม่ใช่ความน่าจะเป็น** และการมีทศนิยมไม่ได้แปลว่าละเอียดขึ้น")


# ---------------------------------------------------------------------------
# CHIEF VERDICT — กฎประกาศล่วงหน้า คำนวณด้วย Python
# ---------------------------------------------------------------------------

def chief_verdict(state: dict, gate: dict, tal: dict) -> dict:
    """ลำดับการตัดสินที่ล็อกไว้ (สภาแทรกแซงไม่ได้):

    1. ติดเกตแข็ง → NO TRADE
    2. กติกา v6.4 ยังไม่ครบ → NO TRADE (ต่อให้สภาเชียร์ทั้งสภา)
    3. กติกาครบ + เกตผ่าน → ตามกติกา (LONG/SHORT) แต่ถ้าสภาเสียงแตก
       หรือมีบทค้านฝั่งเดียวกับสัญญาณ → ลดขนาด
    """
    counts = tal.get("counts", {})
    n_rated = sum(counts.values())
    plan = state.get("plan") or {}
    side = plan.get("side")

    if not gate["pass"]:
        return {"verdict": "NO TRADE", "reason": "ติดเกตความเสี่ยง: "
                + ", ".join(gate["veto"]), "action": "ไม่เข้าไม้ใหม่",
                "override": "สภาเปลี่ยนผลข้อนี้ไม่ได้"}
    if not state.get("triggered") or not side:
        return {"verdict": "NO TRADE", "reason":
                f"กติกา v6.4 ยังไม่ครบ (status: {state.get('status')})",
                "action": "รอให้เงื่อนไขครบ",
                "override": "สภาเปลี่ยนผลข้อนี้ไม่ได้ — ความเห็นเชิงบวก"
                            "ไม่ใช่เหตุผลให้เข้าก่อนกติกา"}

    want = "BUY" if side == "LONG" else "SELL"
    against = counts.get("SELL" if want == "BUY" else "BUY", 0)
    agree = counts.get(want, 0)
    if against > 0:
        act = "ลดขนาด (มีบทค้านทิศสัญญาณ)"
    elif n_rated and agree < max(1, n_rated // 3):
        act = "ลดขนาด (เสียงหนุนน้อย ส่วนใหญ่เป็นกลาง)"
    else:
        act = "ทำตามกติกา v6.4 เต็มขนาดที่คำนวณไว้"
    return {"verdict": f"ตามกติกา · {side}", "reason":
            f"เกตผ่านครบ และ engine มีสัญญาณ {side}",
            "action": act,
            "override": "สภาปรับได้แค่ขนาด ไม่ใช่การมี/ไม่มีไม้"}


DISCLAIMER = (
    "สภานี้ = **โมเดลภาษาเล่นหลายบท** ความเห็นไม่อิสระทางสถิติ เสียงข้างมาก"
    "ไม่เพิ่มความน่าจะเป็นที่จะถูก · verdict และเกตคำนวณด้วย Python จากกติกา "
    "v6.4 ไม่ใช่จากมติสภา · ระบบไม่มีข่าว/COT/order flow → บทเหล่านั้นงดออกเสียง · "
    "ข้อมูลเป็นแท่งปิดย้อนหลัง ไม่ใช่ราคา real-time และไม่มี bid/ask · "
    "ไม่ใช่คำแนะนำการลงทุน")
