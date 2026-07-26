# -*- coding: utf-8 -*-
"""
pf_holdings.py — พอร์ตหุ้นที่ถืออยู่ (5 บัญชี) + ปันผล + แผนแก้หุ้นติด

ปรัชญาของโมดูลนี้: **ไม่สร้างคำแนะนำใหม่ที่ระบบไม่มีหลักฐานรองรับ**
ทุกคำว่า "ถือ/ตัด/ซื้อเพิ่ม" ต้องสืบย้อนไปที่ตัวเลขจาก engine v5.13 ที่มีอยู่แล้ว
หรือเป็นเลขคณิตล้วนที่ตรวจซ้ำได้ ถ้าเรื่องไหนไม่มีข้อมูล → เขียนว่าไม่มี

สามข้อที่ต้องพูดตรง ๆ ก่อน (แสดงบน UI ด้วย):

1) **"ควรขายทำกำไรที่ราคาเท่าไร" — ระบบนี้ตอบไม่ได้ และไม่ควรตอบ**
   กติกา v5.13 ออกด้วย Chandelier trailing stop **ไม่มี take-profit ตายตัว**
   การไปตั้งเป้าราคาให้จึงเป็นการเพิ่มกฎที่ไม่เคย backtest → เราแสดง
   "ระดับ stop ตามสูตรเดียวกับระบบ" แทน แล้วให้ราคาวิ่งจนหลุด trail เอง

2) **"ซื้อเฉลี่ยขาลง" ไม่มีหลักฐานว่าเพิ่มผลตอบแทน** — มันลดต้นทุนเฉลี่ยบนกระดาษ
   แต่เพิ่มขนาดสถานะและความกระจุกตัวในหุ้นที่ราคากำลังบอกว่าคุณคิดผิด
   โมดูลนี้ยอมคำนวณให้ (ผู้ใช้ขอ) แต่จะแสดงเงินที่ต้องใส่เพิ่ม น้ำหนักพอร์ตที่โต
   และคำถามคัดกรองที่ต้องตอบก่อน — ไม่มีการบอกว่า "ราคานี้น่าซื้อเฉลี่ย"

3) **"เวลาไหนดีที่สุด" — ไม่มีคำตอบที่เชื่อถือได้** เอกสารในโปรเจกต์สรุปว่า
   ฤดูกาลของ SET (January / Monday effect) "มีจริงแต่ขนาดเล็กและไม่เสถียร"
   → เราไม่แปลงเป็นสัญญาณเข้า จุดอ้างอิงเดียวที่ไม่ใช่การเดาคือเงื่อนไขเข้า
   ของ v5.13 เอง (บักเก็ต 🟢) ซึ่งแปลว่า "ตัดสินใหม่เหมือนซื้อครั้งแรก"

ปันผล/XD อ้างอิงเอกสารในโปรเจกต์ (เกรดหลักฐาน B):
   บุคคลธรรมดาไทยถูกหัก ณ ที่จ่าย 10% หรือเลือกนำไปรวมคำนวณภาษีปลายปี
   แล้วใช้เครดิตภาษีเงินปันผลได้ → ภาษีที่แท้จริงขึ้นกับฐานภาษีของแต่ละคน
   ราคาที่ลดลงวัน XD ของหุ้นใหญ่ไทย "มักน้อยกว่าเงินปันผลเต็มจำนวน"
   แต่ **ต้องวัดเป็นรายตัว** และส่วนต่างมักถูกกลืนโดยต้นทุนรอบละ ~0.32%/ข้าง
   เว้นแต่ yield จะสูง (>4-5%)
"""
from __future__ import annotations

import io
import math
from datetime import date, datetime

import numpy as np
import pandas as pd

# ต้นทุนต่อข้างของ v5.13: (คอม 0.15 × VAT 1.07 + ค่าธรรมเนียมตลาด 0.007 + สเปรด 0.15)%
VERSION = "1.3"   # 1.3 = เปลี่ยนชื่อไฟล์เป็น pf_holdings.py (เลี่ยงปัญหาอัปทับไฟล์เดิม)

COST_SIDE_PCT = (0.15 * 1.07 + 0.007 + 0.15)
WHT_DIVIDEND = 0.10          # หัก ณ ที่จ่ายมาตรฐานของบุคคลธรรมดาไทย
MAX_ACCOUNTS = 5

COLUMNS = ["บัญชี", "หุ้น", "จำนวนหุ้น", "ราคาต้นทุน", "วันที่ซื้อ",
           "ปันผลต่อหุ้น", "หมายเหตุ"]

NUM_COLS = ["จำนวนหุ้น", "ราคาต้นทุน", "ปันผลต่อหุ้น"]
DATE_COLS = ["วันที่ซื้อ"]

# คอลัมน์ที่เคยมีแล้วเลิกเก็บ — ถ้าเจอในไฟล์เก่าจะแจ้งว่าถูกตัด ไม่ลบเงียบ ๆ
DROPPED_COLS = ["วันที่ XD"]


# ---------------------------------------------------------------------------
# โครงข้อมูล + CSV
# ---------------------------------------------------------------------------

def empty_df() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS})
    for c in NUM_COLS:
        df[c] = pd.Series(dtype="float")
    return df


def normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """บังคับชนิดข้อมูล + คืนรายการปัญหาที่พบ (ไม่เงียบ ไม่ทิ้งแถวโดยไม่บอก)"""
    problems: list[str] = []
    if df is None or len(df) == 0:
        return empty_df(), problems
    d = df.copy()
    for c in DROPPED_COLS:
        if c in d.columns and d[c].notna().any():
            problems.append(f"ไฟล์นี้มีคอลัมน์ '{c}' ซึ่งเลิกเก็บแล้ว — "
                            "ข้อมูลส่วนนั้นถูกตัดออก (ไม่ได้ใช้ในการคำนวณอยู่แล้ว)")
    for c in COLUMNS:
        if c not in d.columns:
            d[c] = np.nan
            problems.append(f"ไม่มีคอลัมน์ '{c}' — เติมเป็นค่าว่าง")
    d = d[COLUMNS]
    d["หุ้น"] = (d["หุ้น"].astype(str).str.upper().str.strip()
                 .str.replace(".BK", "", regex=False))
    for c in NUM_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in DATE_COLS:
        d[c] = pd.to_datetime(d[c], errors="coerce").dt.date
    d["บัญชี"] = pd.to_numeric(d["บัญชี"], errors="coerce").fillna(1).astype(int)
    bad_acc = d["บัญชี"].between(1, MAX_ACCOUNTS)
    if (~bad_acc).any():
        problems.append(f"บัญชีต้องเป็น 1-{MAX_ACCOUNTS} — "
                        f"{int((~bad_acc).sum())} แถวถูกปรับเป็น 1")
        d.loc[~bad_acc, "บัญชี"] = 1
    d["หมายเหตุ"] = d["หมายเหตุ"].fillna("").astype(str)

    blank = d["หุ้น"].isin(["", "NAN", "NONE"])
    if blank.any():
        problems.append(f"ตัดออก {int(blank.sum())} แถวที่ไม่มีชื่อหุ้น")
        d = d[~blank]
    bad_qty = d["จำนวนหุ้น"].isna() | (d["จำนวนหุ้น"] <= 0)
    if bad_qty.any():
        problems.append(f"{int(bad_qty.sum())} แถวมีจำนวนหุ้นว่าง/≤0 — "
                        "ยังเก็บไว้แต่คำนวณมูลค่าไม่ได้")
    bad_cost = d["ราคาต้นทุน"].isna() | (d["ราคาต้นทุน"] <= 0)
    if bad_cost.any():
        problems.append(f"{int(bad_cost.sum())} แถวมีราคาต้นทุนว่าง/≤0 — "
                        "คำนวณกำไร/ขาดทุนไม่ได้")
    return d.reset_index(drop=True), problems


def to_csv(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in DATE_COLS:
        if c in d.columns:
            d[c] = d[c].astype(str).replace({"NaT": "", "None": "", "nan": ""})
    return d.to_csv(index=False)


def from_csv(src) -> tuple[pd.DataFrame, list[str]]:
    try:
        if isinstance(src, (bytes, bytearray)):
            src = io.StringIO(src.decode("utf-8-sig"))
        elif isinstance(src, str):
            src = io.StringIO(src)
        d, probs = normalize(pd.read_csv(src))
        if len(d) == 0 and not probs:
            probs = ["อ่านไฟล์ได้แต่ไม่พบแถวที่ใช้ได้เลย — "
                     "ตรวจว่าไฟล์เป็น CSV จริงและมีคอลัมน์ 'หุ้น'"]
        return d, probs
    except Exception as e:
        return empty_df(), [f"อ่านไฟล์ CSV ไม่ได้: {e}"]


# ---------------------------------------------------------------------------
# ระดับ stop ตามสูตรเดียวกับ v5.13 (ไม่ประดิษฐ์กฎใหม่)
# ---------------------------------------------------------------------------

def pick_frame(pool: dict, ticker: str):
    """หา OHLCV ของหุ้นจาก pool ไม่ว่า key จะมี .BK หรือไม่

    เขียนเป็นฟังก์ชันแยกเพราะเคยพลาด: `pool.get(f"{t}.BK") or pool.get(t)`
    ทำให้ pandas เรียก DataFrame.__bool__ แล้วโยน ValueError
    ("The truth value of a DataFrame is ambiguous") → ห้ามใช้ or/and/if
    กับ DataFrame ต้องเทียบ `is None` เท่านั้น
    """
    if not pool or not ticker:
        return None
    t = str(ticker).upper().replace(".BK", "").strip()
    for key in (f"{t}.BK", t):
        df = pool.get(key)
        if df is not None and len(df) > 0:
            return df
    return None


def chandelier_stop(fr: pd.DataFrame, tr_len: int = 22,
                    tr_mlt: float = 3.0) -> float:
    """Chandelier ฝั่ง long = สูงสุด tr_len แท่ง − tr_mlt × ATR

    ใช้สูตรและค่าตั้งต้นเดียวกับ SwingParams ของ v5.13 เป๊ะ
    หมายเหตุความซื่อสัตย์: ในระบบจริง trail จะ "ขยับขึ้นอย่างเดียว" นับจากวันเข้า
    ค่าที่คำนวณตรงนี้เป็น **ค่า ณ วันนี้เท่านั้น** ไม่ได้ไล่ประวัติจากวันที่คุณซื้อ
    → ถ้าเข้ามานานแล้ว stop จริงตามแผนเดิมอาจสูงกว่านี้
    """
    if fr is None or len(fr) < tr_len + 2:
        return float("nan")
    try:
        hi = float(fr["High"].tail(tr_len).max())
        atr = float(fr["atr"].iloc[-1])
        if not (hi == hi and atr == atr and atr > 0):
            return float("nan")
        return round(hi - tr_mlt * atr, 2)
    except Exception:
        return float("nan")


STOP_NOTE = ("stop นี้คำนวณจากสูตร Chandelier ของ v5.13 **ณ วันนี้** "
             "ไม่ได้ไล่ประวัติจากวันที่คุณซื้อจริง — ถ้าถือมานาน stop ตามแผนเดิม"
             "อาจสูงกว่านี้ ให้ยึดแผนเดิมเป็นหลัก")


# ---------------------------------------------------------------------------
# รวมข้อมูลพอร์ต
# ---------------------------------------------------------------------------

def enrich(df: pd.DataFrame, last_price: dict, engine: dict | None = None,
           stops: dict | None = None) -> pd.DataFrame:
    """เติมราคาล่าสุด/มูลค่า/กำไรขาดทุน/น้ำหนัก — เลขคณิตล้วน"""
    engine, stops = engine or {}, stops or {}
    if df is None or len(df) == 0:
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        tk = str(r["หุ้น"])
        px = last_price.get(tk)
        qty = float(r["จำนวนหุ้น"]) if pd.notna(r["จำนวนหุ้น"]) else float("nan")
        cost = float(r["ราคาต้นทุน"]) if pd.notna(r["ราคาต้นทุน"]) else float("nan")
        mv = qty * px if (px is not None and qty == qty) else float("nan")
        cb = qty * cost if (qty == qty and cost == cost) else float("nan")
        pl = mv - cb if (mv == mv and cb == cb) else float("nan")
        e = engine.get(tk, {})
        rows.append({
            "บัญชี": int(r["บัญชี"]), "หุ้น": tk,
            "จำนวนหุ้น": qty, "ราคาต้นทุน": cost,
            "ราคาล่าสุด": px, "มูลค่าตลาด": mv, "ต้นทุนรวม": cb,
            "กำไร/ขาดทุน": pl,
            "%": round(pl / cb * 100, 2) if (pl == pl and cb and cb > 0) else None,
            "Regime": e.get("Regime"), "บักเก็ต": e.get("บักเก็ต"),
            "stop ระบบ": stops.get(tk),
            "ปันผลต่อหุ้น": (float(r["ปันผลต่อหุ้น"])
                             if pd.notna(r["ปันผลต่อหุ้น"]) else None),
            "หมายเหตุ": r["หมายเหตุ"],
        })
    out = pd.DataFrame(rows)
    tot = out["มูลค่าตลาด"].sum(skipna=True)
    out["น้ำหนัก %"] = (out["มูลค่าตลาด"] / tot * 100).round(2) if tot else None
    return out


def summary(enriched: pd.DataFrame) -> dict:
    if enriched is None or len(enriched) == 0:
        return {}
    mv = float(enriched["มูลค่าตลาด"].sum(skipna=True))
    cb = float(enriched["ต้นทุนรวม"].sum(skipna=True))
    top = enriched.dropna(subset=["น้ำหนัก %"]).nlargest(1, "น้ำหนัก %")
    return {"มูลค่าตลาดรวม": round(mv, 2), "ต้นทุนรวม": round(cb, 2),
            "กำไร/ขาดทุน": round(mv - cb, 2),
            "%": round((mv - cb) / cb * 100, 2) if cb > 0 else None,
            "จำนวนตัว": int(enriched["หุ้น"].nunique()),
            "กระจุกสูงสุด": (f"{top.iloc[0]['หุ้น']} "
                             f"{top.iloc[0]['น้ำหนัก %']}%") if len(top) else None}


# ---------------------------------------------------------------------------
# ปันผล / XD — เลขคณิตที่ตรวจซ้ำได้ ไม่ใช่การทำนาย
# ---------------------------------------------------------------------------

def dividend_view(qty: float, cost: float, price: float, div_ps: float,
                  xd: date | None, today: date | None = None,
                  wht: float = WHT_DIVIDEND,
                  drop_ratio: float | None = None) -> dict:
    """คืนตัวเลขปันผล + **จุดคุ้มทุนที่แน่นอนทางเลขคณิต**

    ตรรกะ: ถือข้าม XD ได้ปันผลสุทธิ D×(1−ภาษี) แต่ราคาลง D×k (k = drop ratio)
           ผลสุทธิ = D×(1−ภาษี) − D×k → คุ้มก็ต่อเมื่อ k < (1−ภาษี)
    ที่ภาษี 10% → **คุ้มก็ต่อเมื่อราคาลงหลัง XD น้อยกว่า 90% ของเงินปันผล**
    (พารามิเตอร์ xd เป็นออปชัน — ตั้งแต่ v1.2 ไฟล์พอร์ตไม่เก็บวัน XD แล้ว
     การคำนวณจุดคุ้มทุนไม่ต้องใช้วันที่ ใช้แค่จำนวนเงินปันผลกับภาษี)
    ค่า k ต้องวัดเป็นรายตัวจากประวัติ — ระบบนี้ไม่มีข้อมูลนั้น จึงให้ผู้ใช้ใส่เอง
    """
    out: dict = {"มีข้อมูล": False}
    if not (div_ps and div_ps > 0 and qty and qty > 0):
        out["เหตุผล"] = "ยังไม่ได้กรอกปันผลต่อหุ้น หรือจำนวนหุ้น"
        return out
    today = today or date.today()
    gross = qty * div_ps
    net = gross * (1 - wht)
    be_ratio = 1 - wht
    out.update({
        "มีข้อมูล": True,
        "ปันผลรวม (ก่อนภาษี)": round(gross, 2),
        "หักภาษี 10%": round(gross * wht, 2),
        "ปันผลสุทธิ": round(net, 2),
        "yield on cost %": round(div_ps / cost * 100, 2) if cost and cost > 0 else None,
        "yield ราคาปัจจุบัน %": round(div_ps / price * 100, 2) if price and price > 0 else None,
        "จุดคุ้มทุน": f"คุ้มก็ต่อเมื่อราคาหลัง XD ลง **น้อยกว่า "
                      f"{be_ratio * 100:.0f}%** ของเงินปันผล "
                      f"(≤ {div_ps * be_ratio:.4f} บาท/หุ้น)",
        "ถ้าราคาลงเท่าปันผลพอดี": round(-gross * wht, 2),
    })
    if xd:
        d = (xd - today).days
        out["วันถึง XD"] = d
        out["สถานะ XD"] = ("ผ่าน XD ไปแล้ว" if d < 0
                           else ("XD วันนี้" if d == 0 else f"อีก {d} วัน"))
    if drop_ratio is not None:
        eff = gross * (be_ratio - float(drop_ratio))
        out["drop ratio ที่ใส่เอง"] = drop_ratio
        out["ผลสุทธิที่คาดตาม ratio นี้"] = round(eff, 2)
        out["สรุปตาม ratio ที่ใส่"] = ("คุ้ม (แต่เป็นค่าที่คุณใส่เอง ไม่ใช่ค่าที่วัดจากระบบ)"
                                       if eff > 0 else "ไม่คุ้ม")
    return out


DIVIDEND_NOTE = (
    "**สิ่งที่ระบบนี้ทำได้จริง:** คำนวณเลขคณิตของจุดคุ้มทุนเท่านั้น "
    "**สิ่งที่ทำไม่ได้:** ทำนายว่าราคาจะลงเท่าไรหลัง XD — ระบบไม่มีข้อมูลประวัติ "
    "ปันผล/วัน XD ย้อนหลังรายตัว · เอกสารในโปรเจกต์ (เกรดหลักฐาน B) ระบุว่า "
    "ราคาที่ลดวัน XD ของหุ้นใหญ่ไทย *มัก* น้อยกว่าปันผลเต็มจำนวน แต่ "
    "**ต้องวัดเป็นรายตัว** และส่วนต่างมักถูกกลืนด้วยต้นทุนรอบละ "
    f"~{COST_SIDE_PCT:.2f}%/ข้าง เว้นแต่ yield จะสูงเกิน 4-5% · "
    "ภาษี 10% เป็นการหัก ณ ที่จ่ายมาตรฐาน — ถ้าคุณเลือกนำไปรวมคำนวณปลายปีและใช้ "
    "เครดิตภาษีเงินปันผล ภาษีจริงจะต่างออกไปตามฐานภาษีของคุณ (ปรับช่องภาษีได้)")


# ---------------------------------------------------------------------------
# จัดกลุ่มการกระทำ — ทุกกรณีสืบย้อนไป engine ได้
# ---------------------------------------------------------------------------

def action_for(price, cost, stop, regime, bucket) -> dict:
    """คืน {'action', 'เหตุผล', 'ที่มา'} — ไม่มีเป้าราคาขายทำกำไร โดยตั้งใจ"""
    if price is None or price != price:
        return {"action": "ไม่มีราคา", "เหตุผล": "ดึงราคาล่าสุดไม่ได้",
                "ที่มา": "—"}
    if stop == stop and stop is not None and price < stop:
        # ระวังการตีความ: Chandelier ที่คำนวณ "ณ วันนี้" อ้างจากสูงสุด 22 แท่งล่าสุด
        # ถ้าหุ้นเพิ่งย่อจากยอด ค่านี้จะอยู่เหนือราคาได้ทั้งที่คุณยังกำไร
        # → ห้ามเขียนว่า "หลุด stop ของคุณ" เพราะ stop จริงขึ้นกับวันที่คุณเข้า
        return {"action": "⛔ ต่ำกว่าระดับ trail อ้างอิงวันนี้",
                "เหตุผล": f"ราคา {price:.2f} < trail อ้างอิง {stop:.2f} "
                          "(ย่อจากยอด 22 แท่งเกิน 3×ATR) — "
                          "**ไม่ได้แปลว่าหลุด stop ตามแผนของคุณ** "
                          "stop จริงขึ้นกับวันที่คุณเข้าและระดับที่ trail ไต่ขึ้นมา",
                "ที่มา": "Chandelier v5.13 (ค่าวันนี้ ไม่ใช่ค่าจากวันที่คุณซื้อ)"}
    if regime == "DOWN":
        return {"action": "🔻 ระบบไม่สนับสนุนให้ถือ",
                "เหตุผล": "regime ขาลง — กติกา v5.13 ไม่เข้าฝั่ง long ในสภาวะนี้",
                "ที่มา": "regime_dn ของ engine"}
    if bucket and str(bucket).startswith("🟢"):
        return {"action": "🟢 ระบบมีสัญญาณเข้าวันนี้",
                "เหตุผล": "เงื่อนไขเข้าครบ — ถ้าจะเพิ่ม ให้คิดเป็น 'ไม้ใหม่' "
                          "ตามกติกาและขนาดความเสี่ยงเดิม ไม่ใช่การเฉลี่ยขาลง",
                "ที่มา": "บักเก็ต v5.13"}
    pl = ((price - cost) / cost * 100) if (cost and cost > 0) else float("nan")
    if pl == pl and pl < 0:
        return {"action": "🟠 ติดลบ แต่ยังไม่หลุด stop",
                "เหตุผล": f"ขาดทุน {pl:.1f}% · ยังอยู่เหนือ stop → "
                          "กติกาคือถือตามแผนเดิม ไม่ใช่เติมของ",
                "ที่มา": "stop ระบบ + ต้นทุนของคุณ"}
    return {"action": "🟡 ถือต่อตาม trail",
            "เหตุผล": "ยังไม่หลุด stop และไม่มีสัญญาณใหม่ — "
                      "ระบบ **ไม่มีเป้าราคาขายทำกำไร** ออกด้วย trail เท่านั้น",
            "ที่มา": "Chandelier v5.13"}


NO_TP_NOTE = (
    "**ทำไมไม่มีช่อง 'ราคาขายทำกำไร':** กติกา v5.13 ออกด้วย Chandelier "
    "trailing stop เท่านั้น ไม่มี take-profit ตายตัว — ผล backtest ทั้งหมด "
    "(win rate, PF, expectancy) มาจากการออกแบบ trail ถ้าเติมเป้าราคาเข้าไป "
    "ตัวเลขเหล่านั้นใช้อ้างอิงไม่ได้อีกต่อไป ระบบจึงไม่ตั้งเป้าให้")


# ---------------------------------------------------------------------------
# หุ้นติด: เลขคณิตของการเฉลี่ย + คำเตือนที่ต้องอ่าน
# ---------------------------------------------------------------------------

def breakeven_gain_pct(price: float, cost: float) -> float:
    """ราคาต้องขึ้นกี่ % จากราคาปัจจุบัน ถึงจะกลับมาเท่าทุน (ยังไม่รวมค่าคอม)"""
    if not (price and cost and price > 0):
        return float("nan")
    return round((cost / price - 1) * 100, 2)


def average_down_plan(cur_qty: float, cur_cost: float, add_qty: float,
                      add_price: float, port_value: float | None = None) -> dict:
    """เลขคณิตล้วน — ไม่มีการบอกว่าควรทำหรือไม่ควรทำ"""
    if not all(x and x > 0 for x in (cur_qty, cur_cost, add_qty, add_price)):
        return {"ok": False, "เหตุผล": "ต้องกรอกจำนวน/ราคาที่เป็นบวกทั้งหมด"}
    new_qty = cur_qty + add_qty
    new_cost = (cur_qty * cur_cost + add_qty * add_price) / new_qty
    capital = add_qty * add_price
    fee = capital * COST_SIDE_PCT / 100.0
    before = breakeven_gain_pct(add_price, cur_cost)
    after = breakeven_gain_pct(add_price, new_cost)
    old_val, new_val = cur_qty * add_price, new_qty * add_price
    out = {
        "ok": True,
        "ต้นทุนเฉลี่ยใหม่": round(new_cost, 4),
        "จำนวนหุ้นใหม่": round(new_qty, 0),
        "เงินที่ต้องใส่เพิ่ม": round(capital, 2),
        "ค่าธรรมเนียมซื้อ": round(fee, 2),
        "เดิมต้องขึ้น % ถึงเท่าทุน": before,
        "หลังเฉลี่ยต้องขึ้น %": after,
        # ตัวเลขที่คนมักเข้าใจผิด: ขาดทุนที่ยังไม่รับรู้ **ไม่เปลี่ยน** ณ วินาทีที่ซื้อ
        # (ซื้อที่ราคาตลาด = ขาดทุนเพิ่ม 0) สิ่งที่เปลี่ยนจริงคือความไวต่อการลงต่อ
        "ขาดทุนที่ยังไม่รับรู้ ณ ราคานี้ (ก่อน)": round(cur_qty * (add_price - cur_cost), 2),
        "ขาดทุนที่ยังไม่รับรู้ ณ ราคานี้ (หลัง)": round(new_qty * (add_price - new_cost), 2),
        "ถ้าลงต่ออีก 10% ขาดทุนเพิ่ม (ก่อน)": round(-cur_qty * add_price * 0.10, 2),
        "ถ้าลงต่ออีก 10% ขาดทุนเพิ่ม (หลัง)": round(-new_qty * add_price * 0.10, 2),
    }
    if port_value and port_value > 0:
        out["น้ำหนักในพอร์ต ก่อน %"] = round(old_val / port_value * 100, 2)
        out["น้ำหนักในพอร์ต หลัง %"] = round(
            new_val / (port_value + capital) * 100, 2)
    return out


AVERAGE_DOWN_WARNING = (
    "**ก่อนกดเฉลี่ย — สิ่งที่เลขข้างบนไม่ได้บอก:** ต้นทุนเฉลี่ยที่ลดลงเป็นเพียง"
    "ตัวเลขบัญชี **มันไม่ได้ทำให้หุ้นตัวนี้มีโอกาสขึ้นมากกว่าเดิมแม้แต่นิดเดียว** · "
    "สิ่งที่เปลี่ยนจริงคือ เงินที่เสี่ยงมากขึ้น และน้ำหนักพอร์ตที่กระจุกในตัวที่ "
    "ราคากำลังบอกว่าคุณคิดผิด · **ขาดทุนที่ยังไม่รับรู้ ณ ราคานี้ไม่ได้ลดลงเลย** "
    "(ซื้อที่ราคาตลาด = ขาดทุนเพิ่ม 0 บาททันที) สิ่งที่โตขึ้นคือ**ความไวต่อการลงต่อ** "
    "— ลงอีก 10% เจ็บเป็นสัดส่วนกับจำนวนหุ้นใหม่ ไม่ใช่จำนวนเดิม")

AVERAGE_DOWN_CHECKLIST = [
    "ถ้าวันนี้ยังไม่มีหุ้นตัวนี้เลย จะซื้อที่ราคานี้ด้วยเหตุผลอะไร (เขียนออกมาเป็นข้อ)",
    "regime ของหุ้นตัวนี้เป็นขาขึ้นหรือขาลงตาม engine",
    "ถ้าเติมแล้วยังลงต่ออีก 20% จะยังถือไหวไหม เงินก้อนนี้จำเป็นต้องใช้เมื่อไร",
    "น้ำหนักหลังเติมเกินเพดานที่ตั้งไว้ต่อตัวหรือยัง",
    "เหตุผลที่ซื้อครั้งแรกยังเป็นจริงอยู่ไหม หรือเปลี่ยนไปแล้ว",
]

TIMING_NOTE = (
    "**เรื่อง 'ควรเติมตอนไหน':** ไม่มีคำตอบที่เชื่อถือได้ในระบบนี้ — "
    "เอกสารในโปรเจกต์สรุปว่าฤดูกาลของ SET (January / Monday effect) "
    "'มีจริงแต่ขนาดเล็กและไม่เสถียร' และงานวิจัยที่เผยแพร่แล้วผลตอบแทน "
    "หายไปราวครึ่งหนึ่ง (McLean & Pontiff 2016) → ไม่แปลงเป็นสัญญาณเข้า · "
    "จุดอ้างอิงเดียวที่ไม่ใช่การเดาคือเงื่อนไขเข้าของ v5.13 เอง (บักเก็ต 🟢) "
    "ซึ่งแปลว่า 'ตัดสินใจใหม่เหมือนซื้อครั้งแรก' ไม่ใช่ 'เฉลี่ยของเดิม'")

DISCLAIMER = (
    "ตัวเลขทั้งหมดเป็นเลขคณิตจากข้อมูลที่คุณกรอก + ราคาจาก yfinance "
    "(อาจล่าช้าและไม่รวมสิทธิประโยชน์/การปรับพาร์) · การจัดกลุ่ม ถือ/ตัด/เข้าใหม่ "
    "อ้างกติกา v5.13 เท่านั้น ไม่ใช่การประเมินมูลค่ากิจการ · "
    "ระบบไม่มีข้อมูลงบการเงินหรือประวัติปันผล — ปันผลต่อหุ้นคุณกรอกเอง · "
    "ไม่ใช่คำแนะนำการลงทุน")
