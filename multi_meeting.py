# -*- coding: utf-8 -*-
"""
multi_meeting.py — ห้องประชุม AI แบบ "หลายค่ายจริง" (Gemini / Groq / OpenRouter)

ต่างจาก stock_meeting.py อย่างไร:
  stock_meeting = โมเดลเดียวเล่นหลายบท (ยังใช้ได้ ถ้ามี key เจ้าเดียว)
  multi_meeting = โมเดลคนละค่ายวิเคราะห์ **แยกกัน ไม่เห็นคำตอบกัน** แล้วเราเอามาเทียบ

หลักซื่อสัตย์ที่บังคับในโมดูลนี้:
1. ตารางเทียบมติคำนวณด้วย Python ล้วน — **ไม่ให้ LLM เป็นคนสรุปว่าใครถูก**
   (ถ้าให้ LLM สรุป มันจะเกลี่ยความเห็นจนความขัดแย้งหายไป ซึ่งคือส่วนที่มีค่าที่สุด)
2. ไม่มีการคำนวณ "ความมั่นใจรวม" จากการโหวต — เพราะโมเดลไม่อิสระทางสถิติ
   การเอา conf มาเฉลี่ยแล้วเรียกว่าความน่าจะเป็น คือการสร้างความมั่นใจเทียม
   เราแสดงแค่ "ใครว่าอะไร" + "ตรงกันหรือไม่" + "ช่วง conf ต่ำสุด-สูงสุด"
3. ธงที่ต้องเด่นที่สุดบนจอคือ **ข้อขัดแย้ง** ไม่ใช่เสียงข้างมาก
4. โมเดลที่เรียกไม่สำเร็จ = ไม่นับ และต้องแสดงว่าขาดไปกี่เจ้า
   (2 เจ้าเห็นตรงกัน โดยเจ้าที่สามพัง ≠ เอกฉันท์)

ประหยัดโควตา: ออกแบบให้ **1 call ต่อโมเดล** (ไม่ใช่ 3 รอบเหมือนของเดิม)
เพราะ free tier ทุกเจ้าจำกัด requests/นาที — 3 โมเดล × 3 รอบ = 9 call ชน 429 ง่าย
รอบ "ผู้ตัดสิน" เป็นออปชัน (+1 call) และห้ามใช้ตัดสินใจแทนกติกา v5.13
"""
from __future__ import annotations

import stock_meeting as SM

VERSION = "v1.0"

# มติ/คำสั่งใช้ชุดเดียวกับ stock_meeting เพื่อให้ parser และ UI ใช้ซ้ำได้
VOTE_OPTIONS = SM.VOTE_OPTIONS
ORDER_OPTIONS = SM.ORDER_OPTIONS


def build_solo_prompt(panel_ids: list[str], context_json: str) -> str:
    """prompt เดียวจบต่อหนึ่งโมเดล: วิเคราะห์ + ลงมติ + ปิดท้ายด้วย JSON

    ตัวโมเดลไม่เห็นคำตอบของเจ้าอื่นโดยตั้งใจ — เพื่อไม่ให้เกิด anchoring
    """
    ps = [p for p in SM.PERSONAS if p["id"] in panel_ids] or SM.PERSONAS
    roles = "\n".join(f"- {p['th']}: {p['role']}" for p in ps)
    return (
        "คุณเป็นทีมวิเคราะห์อิสระหนึ่งทีม (ไม่เห็นความเห็นของทีมอื่น) "
        f"ให้ไล่มุมมองต่อไปนี้ให้ครบก่อนสรุป:\n{roles}\n\n"
        f"{SM.IRON_RULES}\n\n"
        "ข้อกำหนดเพิ่ม: ต้องมีอย่างน้อย 1 ประเด็นที่ **ค้าน** สัญญาณ "
        "(ถ้าหาไม่เจอ ให้บอกตรง ๆ ว่าหาไม่เจอ ห้ามแต่ง) และทุกครั้งที่อ้าง "
        "ผล backtest ต้องระบุจำนวนเทรด (n) กำกับ\n\n"
        "รูปแบบคำตอบ: บทวิเคราะห์สั้น ≤ 12 บรรทัด แล้วปิดท้ายด้วยบล็อก JSON "
        "เดียวใน ```json ...``` เท่านั้น ห้ามมีข้อความหลังบล็อก JSON\n"
        '{"votes": {"TICKER": {"มติ": "ตาม|งด|ค้าน", "conf": 0-100, '
        '"เหตุผล": "สั้น"}}, '
        '"ขัดแย้ง": ["ประเด็นที่ค้านสัญญาณ"], '
        '"คำสั่ง": [{"หุ้น": "TICKER", "คำสั่ง": "ทำตามกติกา|ข้าม|ลดขนาด", '
        '"เงื่อนไข": "สั้น"}], "conf_รวม": 0-100}\n\n'
        f"context (ตัวเลขทั้งหมดจาก engine v5.13):\n{context_json}"
    )


def build_referee_prompt(disagreements: list[str], context_json: str) -> str:
    """รอบผู้ตัดสิน (ออปชัน) — ให้ดูเฉพาะ 'จุดที่เห็นต่าง' เท่านั้น

    ห้ามให้ผู้ตัดสินออกมติแทน: หน้าที่คือชี้ว่าข้อมูลชุดไหนจะชี้ขาดข้อขัดแย้งได้
    """
    items = "\n".join(f"- {d}" for d in disagreements) or "- (ไม่มีข้อขัดแย้ง)"
    return (
        "บทผู้ตัดสิน: ทีมวิเคราะห์หลายทีมให้มติต่างกันในประเด็นด้านล่าง\n"
        f"{items}\n\n"
        "หน้าที่คุณ **ไม่ใช่** การเลือกข้างหรือออกมติใหม่ แต่คือ:\n"
        "1) แต่ละข้อขัดแย้ง เกิดจากทีมมองตัวเลขคนละตัว หรือมองตัวเดียวกันคนละแบบ\n"
        "2) ต้องมีข้อมูลอะไรเพิ่มถึงจะชี้ขาดได้ (ระบุชื่อตัวเลขที่ต้องดู)\n"
        "3) ถ้าข้อมูลที่มีอยู่ชี้ขาดไม่ได้ ให้บอกว่าชี้ขาดไม่ได้\n"
        "ตอบไทย ≤ 10 บรรทัด ไม่ต้องมี JSON | " + SM.IRON_RULES + "\n\n"
        f"context:\n{context_json}"
    )


def parse_solo(text: str) -> tuple[str, dict | None]:
    """ใช้ parser เดียวกับ stock_meeting (โครงสร้าง JSON เหมือนกัน)"""
    return SM.parse_chair(text)


# ---------------------------------------------------------------------------
# ตารางเทียบมติ — คำนวณด้วย Python ล้วน
# ---------------------------------------------------------------------------

def collect(results: list[dict]) -> dict:
    """รวมผลจากหลายโมเดล

    results: [{"label": "Gemini/gemini-2.5-flash", "ok": bool,
               "parsed": dict|None, "error": str, "analysis": str}, ...]
    คืน: {"ok_labels", "failed", "tickers", "votes", "conflicts"}
    """
    ok_labels, failed = [], []
    votes: dict[str, dict[str, dict]] = {}   # ticker -> label -> {มติ, conf, เหตุผล}
    conflicts: list[str] = []
    for r in results:
        label = str(r.get("label") or r.get("provider") or "?")
        pr = r.get("parsed")
        if not r.get("ok") or not isinstance(pr, dict) or not pr.get("votes"):
            failed.append({"label": label,
                           "error": str(r.get("error")
                                        or "ตอบกลับไม่ตรงรูปแบบ JSON")[:300]})
            continue
        ok_labels.append(label)
        for tk, v in pr["votes"].items():
            votes.setdefault(str(tk).upper(), {})[label] = v
        for c in (pr.get("ขัดแย้ง") or []):
            conflicts.append(f"[{label}] {c}")
    tickers = sorted(votes.keys())
    return {"ok_labels": ok_labels, "failed": failed, "tickers": tickers,
            "votes": votes, "conflicts": conflicts[:24]}


def agreement_rows(bundle: dict) -> list[dict]:
    """แถวสรุปต่อหุ้น — ไม่มีการเฉลี่ย conf เป็นความน่าจะเป็น

    สถานะ: "เห็นต่าง" / "เห็นตรง (n เจ้า)" / "ตอบไม่ครบ"
    conf แสดงเป็นช่วง min-max ไม่ใช่ค่าเฉลี่ย เพื่อไม่ให้ดูเหมือนสถิติ
    """
    n_ok = len(bundle.get("ok_labels", []))
    rows = []
    for tk in bundle.get("tickers", []):
        per = bundle["votes"][tk]
        vs = [v["มติ"] for v in per.values()]
        confs = [int(v.get("conf", 0)) for v in per.values()]
        uniq = sorted(set(vs))
        if len(per) < n_ok:
            status = f"ตอบไม่ครบ ({len(per)}/{n_ok} เจ้า)"
        elif len(uniq) > 1:
            status = "⚠️ เห็นต่าง"
        else:
            status = f"เห็นตรง ({len(per)} เจ้า)"
        rows.append({
            "หุ้น": tk,
            "สถานะ": status,
            "มติ": " / ".join(f"{lb.split('/')[0]}:{v['มติ']}"
                              for lb, v in per.items()),
            "conf (ต่ำ-สูง)": (f"{min(confs)}-{max(confs)}" if confs else "—"),
            "เห็นต่าง": len(uniq) > 1,
            "เหตุผล": " | ".join(f"{lb.split('/')[0]}: {v.get('เหตุผล', '')}"
                                 for lb, v in per.items())[:400],
        })
    rows.sort(key=lambda r: (not r["เห็นต่าง"], r["หุ้น"]))
    return rows


def disagreement_list(bundle: dict) -> list[str]:
    """ข้อความบรรยายจุดที่เห็นต่าง — ใช้ป้อนรอบผู้ตัดสิน"""
    out = []
    for r in agreement_rows(bundle):
        if r["เห็นต่าง"]:
            out.append(f"{r['หุ้น']}: {r['มติ']} — {r['เหตุผล'][:220]}")
    return out


def headline(bundle: dict) -> str:
    """บรรทัดสรุปที่ต้องขึ้นบนสุด — เน้นความไม่ครบและความขัดแย้งก่อนเสมอ"""
    n_ok = len(bundle.get("ok_labels", []))
    n_fail = len(bundle.get("failed", []))
    rows = agreement_rows(bundle)
    n_diff = sum(1 for r in rows if r["เห็นต่าง"])
    parts = [f"ตอบสำเร็จ {n_ok} เจ้า"]
    if n_fail:
        parts.append(f"**พัง {n_fail} เจ้า** (มติที่เหลือจึงไม่ใช่เอกฉันท์ครบทีม)")
    if n_ok < 2:
        parts.append("**เหลือเจ้าเดียว — เทียบข้ามค่ายไม่ได้แล้ว**")
    if n_diff:
        parts.append(f"**เห็นต่าง {n_diff}/{len(rows)} ตัว — อ่านตรงนี้ก่อน**")
    elif rows and n_ok >= 2:
        parts.append(f"เห็นตรงกันทุกตัว ({len(rows)}) — "
                     "ระวัง: ความเหมือนอาจมาจากข้อมูลเทรนที่ทับซ้อนกัน")
    return " · ".join(parts)


DISCLAIMER = (
    "มติจาก **โมเดลต่างค่าย** (อิสระกว่าโมเดลเดียวเล่นหลายบท แต่ยังไม่อิสระจริง — "
    "คลังเทรนทับซ้อน, context และ prompt ชุดเดียวกัน → error สหสัมพันธ์) · "
    "เสียงเอกฉันท์ **ไม่เพิ่ม** ความน่าจะเป็นที่สัญญาณจะถูก · "
    "ไม่มีฟีดข่าวรายหุ้นในระบบ · ระบบนี้ไม่คำนวณ 'ความมั่นใจรวม' จากการโหวต "
    "โดยตั้งใจ · ไม่ใช่คำแนะนำการลงทุน — การเข้าจริงยึดกติกา v5.13 + วินัย stop"
)
