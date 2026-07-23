# -*- coding: utf-8 -*-
"""
stock_meeting.py — ห้องประชุม AI สำหรับ "หุ้นที่มีสัญญาณ" (โซน SET)

ความซื่อสัตย์ที่ต้องแสดงบน UI ทุกครั้ง:
1. "ทีม AI หลายบทบาท" = โมเดลเดียวเล่นหลายบท — ความเห็นไม่อิสระทางสถิติ
   การที่หลายบทเห็นตรงกันไม่ได้เพิ่มความน่าจะเป็นถูกของสัญญาณ
2. กติกาเหล็ก: AI ห้ามคิดเลข/ตั้งราคาเป้า/เดาข่าว — ลงมติได้แค่
   ตาม (เข้าเมื่อกติกา v5.13 ครบเท่านั้น) / งด / ค้าน ต่อหุ้นที่ engine คัดมา
   และคำสั่งจำกัดที่ ทำตามกติกา / ข้าม / ลดขนาด — ไม่มีคำสั่งซื้อขายจริง
3. ไม่มีฟีดข่าวรายหุ้นในระบบนี้ — ทุกบทต้องประเมินจากตัวเลขใน context เท่านั้น
   ถ้าข้อมูลไม่พอ ต้องพูดว่าไม่พอ
4. มติ = ความเห็นเชิงคุณภาพ ไม่ใช่คำแนะนำการลงทุน

โมดูลนี้ pure (สร้าง prompt / parse ผลลัพธ์) — การเรียก API อยู่ใน app.py
"""
from __future__ import annotations

import json
import re

VERSION = "v5.13"

PERSONAS: list[dict] = [
    {"id": "macro", "th": "มหภาค/Flow",
     "role": "อ่าน Global Overlay, Fund Flow สถาบัน/ต่างชาติ (streak, z-score), "
             "โซน Market Context — ประเมินว่าลมหนุนหรือต้าน"},
    {"id": "trend", "th": "เทรนด์",
     "role": "อ่าน Regime (หุ้น+ดัชนี vs SMA200), ระยะห่าง trigger, "
             "โครงสร้าง BOS — เทรนด์จริงหรือแค่เด้ง"},
    {"id": "quant", "th": "ควอนต์",
     "role": "อ่าน ConfL, ผล backtest ต่อหุ้น (n, win-rate CI, PF) — "
             "ต้องท้วงทุกครั้งที่ตัวอย่างเล็กหรือหลักฐานอ่อน"},
    {"id": "contra", "th": "สวนฝูง",
     "role": "หาเหตุค้าน: vol-shock rank, ตำแหน่งในกรอบสูงไป, ceiling ใกล้, "
             "ทุกคนเห็นตรงกันเกิน — ต้องแย้งเสียงข้างมากอย่างน้อยหนึ่งประเด็น"},
    {"id": "technic", "th": "เทคนิคอล",
     "role": "อ่าน checklist v5.13 รายข้อ, สถานะ squeeze/สะสม (จำไว้ว่า "
             "accumulation เป็น proxy เกรด C แสดงผลเท่านั้น), สถานะ PB map"},
]
DEFAULT_PANEL = ["macro", "trend", "quant", "contra", "technic"]
CHAIR_TH = "หัวหน้าทีม"

VOTE_OPTIONS = ("ตาม", "งด", "ค้าน")
ORDER_OPTIONS = ("ทำตามกติกา", "ข้าม", "ลดขนาด")

IRON_RULES = (
    "กติกาเหล็ก (ห้ามละเมิด): (1) ใช้เฉพาะตัวเลขใน context — ห้ามคำนวณใหม่/"
    "ประมาณ/ตั้งราคาเป้า/เดาข่าว (ระบบนี้ไม่มีฟีดข่าวรายหุ้น) "
    "(2) มติต่อหุ้นมีแค่ ตาม (=เข้าเมื่อเงื่อนไข v5.13 ครบเท่านั้น) / งด / ค้าน "
    "+ conf 0-100 + เหตุผล ≤ 20 คำ (3) ถ้าข้อมูลไม่พอ ให้บอกว่าไม่พอ "
    "(4) 'accumulation' เป็น proxy เกรด C แสดงผลเท่านั้น ห้ามใช้เป็นเหตุผลหลัก "
    "(5) ห้ามให้คำแนะนำการลงทุนเฉพาะบุคคล (6) ตอบภาษาไทย กระชับ")


def build_round1_prompt(panel_ids: list[str], context_json: str) -> str:
    ps = [p for p in PERSONAS if p["id"] in panel_ids]
    roles = "\n".join(f"- {p['th']}: {p['role']}" for p in ps)
    return (f"คุณจะเล่นบทนักวิเคราะห์ {len(ps)} คนต่อไปนี้ทีละคน:\n{roles}\n\n"
            f"{IRON_RULES}\n\n"
            "รูปแบบต่อคน: ชื่อบท แล้วบรรทัดมติหนึ่งบรรทัดต่อหุ้น "
            "`TICKER มติ conf | เหตุผลสั้น` ตามด้วยข้อสังเกต ≤ 2 ประโยค "
            "(สวนฝูงต้องแย้งเสียงข้างมากอย่างน้อย 1 ประเด็น, "
            "ควอนต์ต้องระบุขนาดตัวอย่างทุกครั้ง)\n\n"
            f"context (ตัวเลขทั้งหมดจาก engine v5.13):\n{context_json}")


def build_round2_prompt() -> str:
    return ("รอบโต้แย้ง: ให้แต่ละบทชี้จุดที่ขัดแย้งกับบทอื่น 1-2 ประโยค "
            "และบอกว่าเปลี่ยนมติหรือไม่เพราะอะไร (เปลี่ยนได้ถ้ายอมรับเหตุผล) "
            + IRON_RULES)


def build_chair_prompt() -> str:
    return (f"บท{CHAIR_TH}: สรุปการประชุมเป็น 3 ส่วนตามลำดับ — "
            "1) เสียงลูกทีม (ย่อ) 2) ข้อขัดแย้ง + คุณเลือกเชื่อฝั่งไหนเพราะอะไร "
            "3) ตัดสิน\n"
            "จากนั้นปิดท้ายด้วยบล็อก JSON เดียวใน ```json ...``` เท่านั้น "
            "โครงสร้าง:\n"
            '{"votes": {"TICKER": {"มติ": "ตาม|งด|ค้าน", "conf": 0-100, '
            '"เหตุผล": "สั้น"}}, '
            '"ขัดแย้ง": ["ประเด็น"], '
            '"คำสั่ง": [{"หุ้น": "TICKER", "คำสั่ง": "ทำตามกติกา|ข้าม|ลดขนาด", '
            '"เงื่อนไข": "สั้น"}], "conf_รวม": 0-100}\n'
            "ห้ามมีข้อความหลังบล็อก JSON | " + IRON_RULES)


# ---------------------------------------------------------------------------
# Parser — ทนต่อ AI ที่ตอบไม่ตรงรูปแบบ (คืน None ถ้า parse ไม่ได้ ไม่โยน)
# ---------------------------------------------------------------------------

def _clamp_conf(x) -> int:
    try:
        return max(0, min(100, int(round(float(x)))))
    except (TypeError, ValueError):
        return 0


def _norm_vote(s) -> str:
    s = str(s or "").strip()
    for v in VOTE_OPTIONS:
        if v in s:
            return v
    return "งด"


def _norm_order(s) -> str:
    s = str(s or "").strip()
    for v in ORDER_OPTIONS:
        if v in s:
            return v
    return "ข้าม"


def parse_chair(text: str) -> tuple[str, dict | None]:
    """แยก (บทวิเคราะห์ก่อน JSON, dict ที่ normalize แล้ว หรือ None)"""
    if not text:
        return "", None
    m = list(re.finditer(r"```json\s*(\{.*?\})\s*```", text, re.S))
    raw, analysis = None, text
    if m:
        raw = m[-1].group(1)
        analysis = text[: m[-1].start()].strip()
    else:  # fallback: หา {...} ก้อนใหญ่สุดท้าย
        i = text.rfind("{")
        depth = 0
        start = None
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
        _ = i
    if not raw:
        return analysis, None
    try:
        d = json.loads(raw)
    except Exception:
        return analysis, None
    votes = {}
    for tk, v in (d.get("votes") or {}).items():
        if not isinstance(v, dict):
            continue
        votes[str(tk).upper()] = {
            "มติ": _norm_vote(v.get("มติ")),
            "conf": _clamp_conf(v.get("conf")),
            "เหตุผล": str(v.get("เหตุผล", ""))[:160],
        }
    orders = []
    for o in (d.get("คำสั่ง") or []):
        if not isinstance(o, dict):
            continue
        orders.append({"หุ้น": str(o.get("หุ้น", "")).upper(),
                       "คำสั่ง": _norm_order(o.get("คำสั่ง")),
                       "เงื่อนไข": str(o.get("เงื่อนไข", ""))[:160]})
    out = {"votes": votes,
           "ขัดแย้ง": [str(x)[:200] for x in (d.get("ขัดแย้ง") or [])][:8],
           "คำสั่ง": orders,
           "conf_รวม": _clamp_conf(d.get("conf_รวม"))}
    return analysis, out


def vote_style(vote: str) -> str:
    """สี badge สำหรับ st.markdown — ตาม=เขียว งด=เทา ค้าน=แดง"""
    return {"ตาม": "green", "ค้าน": "red"}.get(vote, "gray")


DISCLAIMER = (
    "มติ = ความเห็นเชิงคุณภาพจาก **โมเดลเดียวเล่นหลายบท** (ไม่อิสระทางสถิติ — "
    "เสียงเอกฉันท์ไม่ได้เพิ่มความน่าจะเป็นถูก) · ไม่มีฟีดข่าวรายหุ้นในระบบ · "
    "ไม่ใช่คำแนะนำการลงทุน — การเข้าจริงยึดกติกา v5.13 + วินัย stop เท่านั้น")
