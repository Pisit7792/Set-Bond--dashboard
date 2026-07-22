# -*- coding: utf-8 -*-
"""
model_history.py — ประวัติคะแนน 6 โมเดล (รายวัน) + สมุดบันทึก snapshot จริง

นิยามความซื่อสัตย์ของสองเส้นที่ต่างกัน (แสดงบน UI ด้วย):

1) daily_history() = "คำนวณย้อนหลังด้วยโค้ด+ข้อมูลชุดปัจจุบัน"
   - percentile แบบ expanding: ณ วัน t แต่ละ component ใช้เฉพาะข้อมูล ≤ t
     → ไม่มี look-ahead ในตัวสูตร และค่าจุดสุดท้ายเท่ากับคะแนนหน้าโมเดลวันนี้
   - แต่ FRED บาง series ถูก revise ย้อนหลัง + มี publication lag (เช่น CPI, เงินฝาก)
     → เส้นนี้ "ไม่เท่ากับสิ่งที่จะเห็นสด ณ วันนั้นจริง" เสมอไป
   - ต่างจาก models6.score_history (รายสัปดาห์ W-FRI): ตัวนั้น resample ก่อนค่อยทำ
     percentile ส่วนตัวนี้ทำ percentile บนข้อมูลความถี่เดิมแล้ว forward-fill รายวัน
     (component รายเดือน/รายสัปดาห์จึงถือค่าเดิมจนกว่าจะมีข้อมูลใหม่ — ตรงกับสิ่งที่
     score_models เห็น ณ วันนั้น)

2) snapshot log = track record จริง บันทึกวันต่อวันด้วยปุ่มในแอป
   - หนึ่งแถวต่อวัน อัปเดตทับได้เฉพาะ "วันล่าสุด" — ห้ามแก้วันในอดีต (append-only)
   - โหมด DEMO ห้ามบันทึก (กันข้อมูลสังเคราะห์ปนเข้า track record จริง)

หมายเหตุที่แก้ความเข้าใจก่อนหน้า: ในระบบนี้ "ข่าว" ไม่ได้อยู่ในคะแนนโมเดล
(ข่าวใช้ trigger ห้องประชุม AI เท่านั้น) — เส้นย้อนหลังจึงครอบคลุมคะแนนเต็ม
"""
from __future__ import annotations

import base64
import io
import math
from datetime import date

import numpy as np
import pandas as pd

import models6 as M

MIN_N = 60          # ให้ตรงกับ models6._pct_rank(min_n=60)
FFILL_LIMIT_D = 45  # ถือค่า percentile เดิมได้ไม่เกิน 45 วัน (กันข้อมูลตายลากยาว)


# ---------------------------------------------------------------------------
# 1) ประวัติคะแนนรายวัน (คำนวณย้อนหลัง)
# ---------------------------------------------------------------------------

def _expanding_pct_tail(s: pd.Series, need_from: pd.Timestamp,
                        invert: bool) -> pd.Series:
    """expanding percentile เฉพาะจุดสังเกตตั้งแต่ need_from (บวกจุดล่าสุดก่อนหน้า
    หนึ่งจุดไว้เป็นฐาน ffill) — O(k·n) แทน O(n²) เพื่อความเร็วบนข้อมูลรายวันหลายปี"""
    s = pd.Series(s).dropna()
    if len(s) < MIN_N:
        return pd.Series(dtype=float)
    vals = s.to_numpy()
    idx = s.index
    pos = np.searchsorted(idx.values, np.datetime64(need_from))
    pos = max(0, min(pos, len(s) - 1))
    if pos > 0:
        pos -= 1  # จุดฐานก่อนหน้าไว้ ffill เข้ากริด
    out_i, out_v = [], []
    for j in range(pos, len(s)):
        if j + 1 < MIN_N:
            continue  # ยังไม่ครบขั้นต่ำ — เหมือน _pct_rank คืน NaN
        p = float((vals[: j + 1] <= vals[j]).mean() * 100.0)
        out_i.append(idx[j])
        out_v.append(100.0 - p if invert else p)
    return pd.Series(out_v, index=pd.DatetimeIndex(out_i))


def daily_history(data: dict, key: str, tail_days: int = 30,
                  end: pd.Timestamp | None = None) -> pd.Series:
    """เส้นคะแนนรายวันย้อนหลัง tail_days วันทำการของโมเดล key.

    ณ วันทำการ t: คะแนน = ค่าเฉลี่ย (equal weights) ของ percentile ล่าสุดที่มี
    ของแต่ละ component (ข้อมูล ≤ t เท่านั้น) — จุดสุดท้าย = score_models วันนี้
    """
    spec = M.MODEL_DEFS[key]
    comp_frames = {}
    ends = []
    for name, fn, invert in spec["components"]:
        try:
            s = fn(data)
        except Exception:
            s = None
        if s is None:
            continue
        s = pd.Series(s).dropna()
        if len(s) < MIN_N:
            continue
        ends.append(s.index[-1])
        comp_frames[name] = (s, invert)
    if not comp_frames:
        return pd.Series(dtype=float, name=key)

    last = pd.Timestamp(end) if end is not None else max(ends)
    grid = pd.bdate_range(end=last.normalize(), periods=tail_days)
    need_from = grid[0] - pd.Timedelta(days=FFILL_LIMIT_D)

    cols = {}
    for name, (s, invert) in comp_frames.items():
        p = _expanding_pct_tail(s, need_from, invert)
        if p.empty:
            continue
        cols[name] = p.reindex(p.index.union(grid)).ffill(
            limit=None).reindex(grid)
        # จำกัดอายุค่า ffill: ถ้าจุดสังเกตล่าสุด ≤ t ห่างเกิน FFILL_LIMIT_D วัน → NaN
        last_obs = p.index.to_series().reindex(
            p.index.union(grid)).ffill().reindex(grid)
        age_days = np.asarray((grid - pd.DatetimeIndex(last_obs)).days,
                              dtype="float64")
        cols[name] = cols[name].where(age_days <= FFILL_LIMIT_D)
    if not cols:
        return pd.Series(dtype=float, name=key)
    hist = pd.DataFrame(cols).mean(axis=1).dropna()
    hist.name = key
    return hist


def all_daily_history(data: dict, tail_days: int = 30) -> pd.DataFrame:
    """DataFrame คอลัมน์ = โมเดล (คีย์อังกฤษ), แถว = วันทำการ tail_days วัน"""
    cols = {}
    for k in M.MODEL_DEFS:
        h = daily_history(data, k, tail_days)
        if len(h):
            cols[k] = h
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# 2) สมุดบันทึก snapshot (track record จริง)
# ---------------------------------------------------------------------------

SNAP_FILE = "model_history.csv"
SNAP_COLS = ["date"] + list(M.MODEL_DEFS) + ["composite", "mode"]


def empty_snap() -> pd.DataFrame:
    return pd.DataFrame(columns=SNAP_COLS)


def load_snapshots(path_or_buf) -> tuple[pd.DataFrame, list[str]]:
    """อ่านไฟล์บันทึก — คืน (df เรียงวัน, รายการปัญหาที่พบ) ไม่โยน exception"""
    problems: list[str] = []
    try:
        df = pd.read_csv(path_or_buf)
    except FileNotFoundError:
        return empty_snap(), ["ยังไม่มีไฟล์บันทึก (จะสร้างเมื่อกดบันทึกครั้งแรก)"]
    except Exception as e:
        return empty_snap(), [f"อ่านไฟล์ไม่ได้: {e}"]
    for c in SNAP_COLS:
        if c not in df.columns:
            df[c] = np.nan if c not in ("date", "mode") else ""
            problems.append(f"ไฟล์ขาดคอลัมน์ {c} — เติมว่างให้")
    df = df[SNAP_COLS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    bad = df["date"].isna().sum()
    if bad:
        problems.append(f"ตัดแถววันที่เสีย {bad} แถว")
        df = df.dropna(subset=["date"])
    dup = df.duplicated("date").sum()
    if dup:
        problems.append(f"พบวันซ้ำ {dup} แถว — เก็บแถวล่าสุดของวันนั้น")
        df = df.drop_duplicates("date", keep="last")
    return df.sort_values("date").reset_index(drop=True), problems


def append_snapshot(df: pd.DataFrame, d: date, scores: dict[str, float],
                    composite: float, mode: str) -> tuple[pd.DataFrame, bool, str]:
    """เพิ่ม/อัปเดตบันทึกของวัน d — กติกา append-only:
    - d ต้องไม่เก่ากว่าวันล่าสุดในไฟล์ (ห้ามย้อนแก้อดีต)
    - ถ้า d = วันล่าสุดที่มีอยู่ → ทับได้ (refresh ระหว่างวัน)
    - mode == "demo" → ปฏิเสธ (กันข้อมูลสังเคราะห์ปน track record)
    คืน (df ใหม่, สำเร็จ?, ข้อความ)"""
    if str(mode).lower().startswith("demo"):
        return df, False, "โหมด DEMO บันทึกไม่ได้ — track record ต้องมาจากข้อมูลจริงเท่านั้น"
    if len(df):
        last = max(df["date"])
        if d < last:
            return df, False, f"ห้ามบันทึกย้อนหลัง (วันล่าสุดในไฟล์คือ {last})"
    row = {"date": d, "composite": _r1(composite), "mode": mode}
    for k in M.MODEL_DEFS:
        row[k] = _r1(scores.get(k, float("nan")))
    df2 = df[df["date"] != d].copy()
    df2 = pd.concat([df2, pd.DataFrame([row])], ignore_index=True)
    return df2.sort_values("date").reset_index(drop=True), True, \
        f"บันทึกของวัน {d} แล้ว ({len(df2)} วันสะสม)"


def _r1(x) -> float:
    try:
        return round(float(x), 1)
    except (TypeError, ValueError):
        return float("nan")


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# 3) GitHub commit (แบบเดียวกับหน้า Fund Flow — แยกเป็นฟังก์ชัน pure ทดสอบได้)
# ---------------------------------------------------------------------------

def github_put_file(owner: str, repo: str, branch: str, path: str,
                    content: bytes, token: str, message: str,
                    get=None, put=None) -> tuple[bool, str]:
    """PUT ไฟล์ขึ้น GitHub Contents API (สร้าง/ทับ) — คืน (สำเร็จ?, ข้อความ/URL).
    get/put ฉีดแทนได้เพื่อเทสต์ออฟไลน์; ปกติใช้ requests"""
    if not token.strip():
        return False, "ต้องใส่ token ก่อน"
    if get is None or put is None:
        import requests as _rq  # import ในฟังก์ชัน — เทสต์ออฟไลน์ไม่แตะเน็ต
        get = get or _rq.get
        put = put or _rq.put
    api = (f"https://api.github.com/repos/{owner.strip()}/"
           f"{repo.strip()}/contents/{path.strip()}")
    hd = {"Authorization": f"Bearer {token.strip()}",
          "Accept": "application/vnd.github+json"}
    try:
        sha = None
        r0 = get(api, headers=hd, params={"ref": branch.strip()}, timeout=20)
        if getattr(r0, "status_code", None) == 200:
            sha = r0.json().get("sha")
        payload = {"message": message,
                   "content": base64.b64encode(content).decode(),
                   "branch": branch.strip()}
        if sha:
            payload["sha"] = sha
        r1 = put(api, headers=hd, json=payload, timeout=30)
        if getattr(r1, "status_code", None) in (200, 201):
            url = r1.json().get("commit", {}).get("html_url", "")
            return True, url
        try:
            msg = str(r1.json().get("message", ""))[:200]
        except Exception:
            msg = str(getattr(r1, "text", ""))[:200]
        return False, f"GitHub ตอบ {getattr(r1, 'status_code', '?')}: {msg}"
    except Exception as e:
        return False, f"เชื่อมต่อ GitHub ไม่ได้: {e}"


# ---------------------------------------------------------------------------
# ข้อความความซื่อสัตย์สำหรับ UI (จุดเดียว — ใช้ซ้ำได้)
# ---------------------------------------------------------------------------

RECOMPUTE_CAVEAT = (
    "เส้น 'คำนวณย้อนหลัง' ใช้สูตร expanding percentile (ณ วัน t ใช้ข้อมูล ≤ t "
    "เท่านั้น — ไม่มี look-ahead ในสูตร) แต่คำนวณจาก **ข้อมูลชุดปัจจุบัน**: FRED "
    "บาง series ถูก revise ย้อนหลังและมี publication lag จึง *ไม่เท่ากับ* "
    "สิ่งที่จะเห็นสด ณ วันนั้นจริง 100% — track record จริงคือเส้น 'บันทึกสด' "
    "ที่สะสมจากปุ่มบันทึกรายวันเท่านั้น | หมายเหตุ: ข่าวไม่ได้อยู่ในคะแนนโมเดล "
    "ของระบบนี้ (ข่าวใช้เรียกห้องประชุมเท่านั้น) เส้นนี้จึงครอบคลุมคะแนนเต็ม")

SNAP_CAVEAT = (
    "สมุดบันทึกเป็น append-only: ทับได้เฉพาะวันล่าสุด ห้ามแก้อดีต · โหมด DEMO "
    "บันทึกไม่ได้ · ไฟล์บน Streamlit Cloud หายเมื่อเครื่อง restart — กด commit "
    "ขึ้น GitHub เพื่อเก็บถาวร (จะ redeploy อัตโนมัติ ~1-2 นาที)")
