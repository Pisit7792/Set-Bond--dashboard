# -*- coding: utf-8 -*-
"""
flows.py — ยอดซื้อขายสุทธิรายกลุ่มนักลงทุน SET (ตรรกะล้วน ทดสอบได้)

ที่มาข้อมูล: ผู้ใช้คัดจากเว็บ ตลท. (Investor Type) มาเก็บใน CSV เอง
คอลัมน์: วันที่ (d/m/YYYY), Institute, Foreign, Retail (สุทธิ ล้านบาท),
Set Index (ระดับดัชนี), Set Change (จุด)

ความซื่อสัตย์ของโมดูลนี้:
1. ตรวจสมดุลอัตโนมัติ: ถ้า 3 กลุ่มรวมกัน ≈ 0 แปลว่าไฟล์จัดกลุ่มครบถ้วน
   (Prop ถูกนับรวมในกลุ่มใดกลุ่มหนึ่งแล้ว — ข้อมูลจริงของผู้ใช้เป็นแบบนี้);
   ถ้าไม่ ≈ 0 จะติดธงเตือนว่ามีกลุ่มหายไป ไม่เดาแทน
2. หลักฐานงานวิจัยไทย: flow ต่างชาติสัมพันธ์กับผลตอบแทน 'วันเดียวกัน' แรง
   แต่พลังพยากรณ์ล่วงหน้าอ่อนและสั้น (คงอยู่ราว 1-2 เดือน) — จอนี้จึงเป็น
   'บริบท' ไม่ใช่สัญญาณ และไม่ถูกผูกเข้า overlay/backtest
3. การเขียนไฟล์บน Streamlit Cloud คงอยู่แค่ถึงรอบ restart ของเครื่อง —
   ความถาวรจริงคือดาวน์โหลดแล้ว commit กลับ GitHub (แอปย้ำเรื่องนี้บนจอ)
"""
from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd

DATE_COL = "date"
NET_COLS = ["Institute", "Foreign", "Retail"]
TH_NAMES = {"Institute": "สถาบันในประเทศ", "Foreign": "ต่างชาติ",
            "Retail": "รายย่อยในประเทศ"}
IDX_COL, CHG_COL = "Set Index", "Set Change"
# header เดิมของผู้ใช้ (มีช่องว่างรอบ วันที่) — คงไว้ตอนบันทึกเพื่อความเข้ากันได้
ORIG_HEADER = "    วันที่    ,Institute,Foreign,Retail,Set Index,Set Change"

EVIDENCE_NOTE = (
    "หลักฐาน (เกรดปานกลาง): flow ต่างชาติสัมพันธ์กับผลตอบแทน SET *วันเดียวกัน* "
    "อย่างมีนัย แต่ใช้พยากรณ์วันพรุ่งนี้ได้อ่อนมาก และแรงซื้อ/ขายมักคงอยู่เพียง "
    "~1-2 เดือน — ใช้เป็นบริบทประกอบ ไม่ใช่สัญญาณเข้าออก")


def _clean_cols(cols) -> list:
    return [str(c).replace("\ufeff", "").strip() for c in cols]


def load_flow_csv(path_or_buf):
    """อ่าน CSV ของผู้ใช้ -> (DataFrame เรียงวันเก่า→ใหม่ index=วันที่, issues list)

    ทนทานต่อ: BOM, CRLF, ช่องว่างรอบหัวคอลัมน์, แถวว่าง, วันที่ d/m/YYYY,
    ค่าที่มี comma คั่นหลัก, แถวซ้ำ (เก็บค่าแรกที่พบ = แถวบนสุด/ใหม่สุดของไฟล์เดิม)
    """
    issues = []
    df = pd.read_csv(path_or_buf, encoding="utf-8-sig", dtype=str,
                     skip_blank_lines=True)
    df.columns = _clean_cols(df.columns)
    date_src = None
    for c in df.columns:
        if "วันที่" in c or c.lower() == "date":
            date_src = c
            break
    if date_src is None:
        return pd.DataFrame(), ["ไม่พบคอลัมน์วันที่"]
    missing = [c for c in NET_COLS if c not in df.columns]
    if missing:
        return pd.DataFrame(), [f"ไม่พบคอลัมน์: {missing}"]

    out = pd.DataFrame()
    out[DATE_COL] = pd.to_datetime(df[date_src].astype(str).str.strip(),
                                   format="%d/%m/%Y", errors="coerce")
    for c in NET_COLS + [IDX_COL, CHG_COL]:
        if c in df.columns:
            out[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", "", regex=False)
                .str.strip(), errors="coerce")
        else:
            out[c] = np.nan
            issues.append(f"ไม่มีคอลัมน์ {c} (ปล่อยว่าง)")
    n0 = len(out)
    out = out.dropna(subset=[DATE_COL])
    out = out.dropna(subset=NET_COLS, how="all")
    if len(out) < n0:
        issues.append(f"ข้ามแถวว่าง/วันที่อ่านไม่ได้ {n0 - len(out)} แถว")
    dup = out[DATE_COL].duplicated(keep="first")
    if dup.any():
        issues.append(f"วันที่ซ้ำ {int(dup.sum())} แถว — เก็บค่าบนสุดของไฟล์")
        out = out[~dup]
    out = out.set_index(DATE_COL).sort_index()
    if len(out):
        bal = float(out[NET_COLS].sum(axis=1).abs().median())
        if bal > 1.0:
            issues.append(f"ผลรวม 3 กลุ่มไม่เป็นศูนย์ (มัธยฐาน {bal:,.0f} ลบ.) — "
                          "อาจมีกลุ่มนักลงทุนหายไปจากไฟล์")
    return out, issues


def save_flow_csv(df: pd.DataFrame, path: str) -> None:
    """บันทึกกลับรูปแบบเดิมของผู้ใช้: BOM + CRLF + วันที่ d/m/YYYY + ใหม่→เก่า"""
    d = df.sort_index(ascending=False)
    lines = [ORIG_HEADER]
    for ts, row in d.iterrows():
        def f(v, nd=2):
            return "" if pd.isna(v) else f"{float(v):.{nd}f}"
        lines.append(f"{ts.day}/{ts.month}/{ts.year},"
                     f"{f(row['Institute'])},{f(row['Foreign'])},"
                     f"{f(row['Retail'])},{f(row.get(IDX_COL))},"
                     f"{f(row.get(CHG_COL))}")
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write("\r\n".join(lines) + "\r\n")


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    d = df.sort_index(ascending=False)
    buf.write(ORIG_HEADER + "\r\n")
    for ts, row in d.iterrows():
        def f(v):
            return "" if pd.isna(v) else f"{float(v):.2f}"
        buf.write(f"{ts.day}/{ts.month}/{ts.year},{f(row['Institute'])},"
                  f"{f(row['Foreign'])},{f(row['Retail'])},"
                  f"{f(row.get(IDX_COL))},{f(row.get(CHG_COL))}\r\n")
    return buf.getvalue().encode("utf-8-sig")


def append_or_update(df: pd.DataFrame, date, institute: float, foreign: float,
                     retail: float, set_index=None, set_change=None,
                     overwrite: bool = False):
    """เพิ่ม/แก้ข้อมูลหนึ่งวัน -> (df ใหม่, ok, msg)"""
    ts = pd.Timestamp(date).normalize()
    exists = ts in df.index
    if exists and not overwrite:
        return df, False, (f"วันที่ {ts:%d/%m/%Y} มีอยู่แล้ว — ติ๊ก 'เขียนทับ' "
                           "ถ้าต้องการแก้ไข")
    row = {"Institute": float(institute), "Foreign": float(foreign),
           "Retail": float(retail),
           IDX_COL: (np.nan if set_index in (None, "") else float(set_index)),
           CHG_COL: (np.nan if set_change in (None, "") else float(set_change))}
    d2 = df.copy()
    d2.loc[ts] = row
    d2 = d2.sort_index()
    act = "แก้ไข" if exists else "เพิ่ม"
    return d2, True, f"{act}ข้อมูลวันที่ {ts:%d/%m/%Y} สำเร็จ (รวม {len(d2)} วัน)"


# ---------------------------------------------------------------------------
# Analytics (บริบท — ไม่ใช่สัญญาณ)
# ---------------------------------------------------------------------------

def roll_sum(df: pd.DataFrame, w: int) -> pd.DataFrame:
    return df[NET_COLS].rolling(w).sum()


def streak(series: pd.Series) -> int:
    """นับวันติดต่อกันของฝั่งเดียวกันล่าสุด: +n ซื้อสุทธิ n วันติด / -n ขาย"""
    s = pd.Series(series).dropna()
    if not len(s):
        return 0
    sign = 1 if s.iloc[-1] > 0 else (-1 if s.iloc[-1] < 0 else 0)
    if sign == 0:
        return 0
    n = 0
    for v in s.iloc[::-1]:
        if (v > 0) == (sign > 0) and v != 0:
            n += 1
        else:
            break
    return sign * n


def zscore_full(series: pd.Series) -> float:
    s = pd.Series(series).dropna()
    if len(s) < 60:
        return float("nan")
    sd = s.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return float("nan")
    return float((s.iloc[-1] - s.mean()) / sd)


def flow_summary(df: pd.DataFrame) -> pd.DataFrame:
    """ตารางสรุปต่อกลุ่ม: วันล่าสุด, สะสม 5/20/60 วัน, z ของวันล่าสุด, streak"""
    rows = []
    for c in NET_COLS:
        s = df[c].dropna()
        rows.append({
            "กลุ่ม": TH_NAMES[c],
            "วันล่าสุด (ลบ.)": round(float(s.iloc[-1]), 1) if len(s) else np.nan,
            "สะสม 5 วัน": round(float(s.tail(5).sum()), 0),
            "สะสม 20 วัน": round(float(s.tail(20).sum()), 0),
            "สะสม 60 วัน": round(float(s.tail(60).sum()), 0),
            "z วันล่าสุด": round(zscore_full(s), 2),
            "ซื้อ/ขายติดกัน (วัน)": streak(s),
        })
    return pd.DataFrame(rows)


def same_day_corr(df: pd.DataFrame, window: int = 250) -> dict:
    """สหสัมพันธ์ 'วันเดียวกัน' ระหว่าง net ต่างชาติ กับ Set Change (จุด)
    — ตัวเลขนี้มักสูง และคือเหตุผลที่คนเข้าใจผิดว่า flow 'นำ' ราคา"""
    d = df[["Foreign", CHG_COL]].dropna().tail(window)
    if len(d) < 60:
        return {"ok": False}
    same = float(d["Foreign"].corr(d[CHG_COL]))
    lead = df[["Foreign", CHG_COL]].dropna()
    lead = pd.DataFrame({"f": lead["Foreign"].shift(1),
                         "r": lead[CHG_COL]}).dropna().tail(window)
    nextd = float(lead["f"].corr(lead["r"])) if len(lead) >= 60 else float("nan")
    return {"ok": True, "n": len(d), "same_day": same, "next_day": nextd}
