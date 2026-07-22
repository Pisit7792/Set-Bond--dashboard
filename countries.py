# -*- coding: utf-8 -*-
"""
countries.py — ชั้นข้อมูล/ตรรกะสำหรับแท็บ "รายประเทศ" (yield 10 ปี + spread เทียบ US + คะแนนเสี่ยง)

หลักซื่อสัตย์ที่บังคับใช้ในโมดูลนี้ (แสดงบนหน้าจอด้วย):
1) แหล่งข้อมูลฟรีมีจำกัดจริง:
   - US: FRED DGS10 รายวัน (ยืนยันแล้ว)
   - India: FRED/OECD INDIRLTLT01STM รายเดือน ช้า ~1-2 เดือน (ยืนยันแล้ว 22 ก.ค. 2026)
   - Indonesia / South Africa / Poland: series ตามแพตเทิร์น OECD — สถานะ "ยังไม่ยืนยัน"
     โค้ดจะลองดึงจริงตอน deploy และรายงานผลตรง ๆ ผ่านปุ่ม "ตรวจแหล่งข้อมูล"
   - Russia / Turkey / Brazil / Malaysia / Philippines: ไม่พบ API ฟรีที่เชื่อถือได้
     → แสดง "ไม่มีข้อมูล" หรือให้ผู้ใช้กรอกเอง (ติดป้าย MANUAL + วันที่อ้างอิงเสมอ)
2) "คะแนนความเสี่ยงประเทศ" เป็น heuristic สูตรเปิดเผย (ดู FORMULA_TEXT)
   ยังไม่ผ่าน validation เชิงประจักษ์ใด ๆ — ใช้จัดเรียง/เปรียบเทียบ ไม่ใช่สัญญาณเทรด
   และตัวเลขจะไม่ตรงกับแอปอื่นที่ใช้สูตรต่างกัน
3) โหมดออฟไลน์/ดึงไม่สำเร็จ → ใช้ค่า DEMO ที่ติดป้ายชัดเจน ไม่ปลอมเป็นข้อมูลสด

ไม่พึ่งไลบรารีภายนอก (stdlib เท่านั้น) — ทดสอบออฟไลน์ได้ทั้งไฟล์
"""
from __future__ import annotations

import csv
import io
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable, Optional

# ---------------------------------------------------------------- ค่าคงที่

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
FETCH_TIMEOUT_S = 12

# ป้ายความถี่/ความสดที่ใช้บน UI
FREQ_LABEL = {"D": "รายวัน", "M": "รายเดือน (ช้า ~1-2 เดือน)"}

RISK_TIERS = [(0, 25, "เสี่ยงต่ำ"), (25, 50, "เสี่ยงปานกลาง"), (50, 101, "เสี่ยงสูง")]

FORMULA_TEXT = (
    "คะแนนเสี่ยง 0-100 (heuristic เปิดเผย ยังไม่ validate):\n"
    "• ระดับ yield (0-40): 0 แต้มที่ ≤2% ไต่เชิงเส้นถึง 40 แต้มที่ ≥15%\n"
    "• spread เทียบ US10Y (0-40): 0 แต้มที่ ≤0bps ไต่เชิงเส้นถึง 40 แต้มที่ ≥3,000bps\n"
    "• แนวโน้ม 3 เดือน (0-20): 0 แต้มถ้า yield ไม่ขึ้น ไต่เชิงเส้นถึง 20 แต้มที่ +150bps\n"
    "เกณฑ์: <25 เสี่ยงต่ำ · 25-49 ปานกลาง · ≥50 เสี่ยงสูง\n"
    "หมายเหตุ: เป็นสูตรของระบบนี้เอง ตัวเลขไม่เท่ากับแอป/สำนักอื่น และไม่ใช่สัญญาณซื้อขาย"
)

# ---------------------------------------------------------------- โครงประเทศ


@dataclass(frozen=True)
class CountrySpec:
    code: str            # ISO2 ที่ใช้แสดง
    name_th: str
    flag: str
    fred_series: Optional[str]   # None = ไม่มีแหล่งฟรี
    freq: str                    # "D" หรือ "M" (มีผลเมื่อ fred_series ไม่ใช่ None)
    verified: bool               # True = ยืนยันแล้วว่า series มีจริง ณ 22 ก.ค. 2026
    note: str = ""


COUNTRIES: list[CountrySpec] = [
    CountrySpec("US", "สหรัฐฯ (ฐานเทียบ)", "🇺🇸", "DGS10", "D", True,
                "FRED รายวัน — ใช้เป็นฐานคำนวณ spread"),
    CountrySpec("IN", "อินเดีย", "🇮🇳", "INDIRLTLT01STM", "M", True,
                "OECD ผ่าน FRED รายเดือน (ยืนยัน: ก.พ. 2026 = 6.78%)"),
    CountrySpec("ID", "อินโดนีเซีย", "🇮🇩", "IDNIRLTLT01STM", "M", False,
                "แพตเทิร์น OECD — ต้องยืนยันตอน deploy"),
    CountrySpec("ZA", "แอฟริกาใต้", "🇿🇦", "ZAFIRLTLT01STM", "M", False,
                "แพตเทิร์น OECD — ต้องยืนยันตอน deploy"),
    CountrySpec("PL", "โปแลนด์", "🇵🇱", "IRLTLT01PLM156N", "M", False,
                "สมาชิก OECD — ต้องยืนยันตอน deploy"),
    CountrySpec("BR", "บราซิล", "🇧🇷", None, "M", False,
                "ไม่พบ API ฟรีที่เชื่อถือได้ — กรอกเองได้"),
    CountrySpec("TR", "ตุรกี", "🇹🇷", None, "M", False,
                "ไม่พบ API ฟรีที่เชื่อถือได้ — กรอกเองได้"),
    CountrySpec("MY", "มาเลเซีย", "🇲🇾", None, "M", False,
                "ไม่พบ API ฟรีที่เชื่อถือได้ — กรอกเองได้"),
    CountrySpec("PH", "ฟิลิปปินส์", "🇵🇭", None, "M", False,
                "ไม่พบ API ฟรีที่เชื่อถือได้ — กรอกเองได้"),
    CountrySpec("RU", "รัสเซีย", "🇷🇺", None, "M", False,
                "OECD หยุดเผยแพร่หลังปี 2022 — กรอกเองได้"),
]


def spec_by_code(code: str) -> CountrySpec:
    for c in COUNTRIES:
        if c.code == code:
            return c
    raise KeyError(code)


# ---------------------------------------------------------------- ดึงข้อมูล FRED

Fetcher = Callable[[str], str]  # รับ URL คืนข้อความ CSV


def _default_fetcher(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "set-bond-dashboard/1.0"})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as r:
        return r.read().decode("utf-8", errors="replace")


def fetch_fred_series(series_id: str, fetcher: Optional[Fetcher] = None
                      ) -> list[tuple[date, float]]:
    """ดึง fredgraph.csv (ไม่ต้องใช้ API key) → รายการ (วันที่, ค่า) เรียงเก่า→ใหม่
    ข้ามค่า '.' (missing) — โยน exception ตรง ๆ ถ้าดึง/parse ไม่ได้ (UI จับไปแสดง)"""
    text = (fetcher or _default_fetcher)(FRED_CSV_URL.format(sid=series_id))
    rows: list[tuple[date, float]] = []
    rdr = csv.reader(io.StringIO(text))
    header = next(rdr, None)
    if not header or len(header) < 2:
        raise ValueError(f"FRED CSV ผิดรูปแบบ ({series_id})")
    for line in rdr:
        if len(line) < 2 or line[1].strip() in {".", ""}:
            continue
        try:
            d = datetime.strptime(line[0].strip(), "%Y-%m-%d").date()
            v = float(line[1])
        except ValueError:
            continue
        rows.append((d, v))
    if not rows:
        raise ValueError(f"FRED ไม่มีข้อมูลใช้ได้ ({series_id})")
    rows.sort(key=lambda t: t[0])
    return rows


def latest_and_trend(obs: list[tuple[date, float]], lookback_days: int = 95
                     ) -> tuple[date, float, Optional[float]]:
    """คืน (วันที่ล่าสุด, ค่า, การเปลี่ยนแปลง ~3 เดือนเป็น bps หรือ None ถ้าข้อมูลไม่พอ)"""
    last_d, last_v = obs[-1]
    cutoff = last_d - timedelta(days=lookback_days)
    base: Optional[float] = None
    for d, v in obs:
        if d <= cutoff:
            base = v
        else:
            break
    chg_bps = None if base is None else round((last_v - base) * 100.0, 1)
    return last_d, last_v, chg_bps


# ---------------------------------------------------------------- คำนวณ

def spread_bps(country_yield: float, us_yield: float) -> float:
    return round((country_yield - us_yield) * 100.0, 1)


def _lin(x: float, x0: float, x1: float, pts: float) -> float:
    if x <= x0:
        return 0.0
    if x >= x1:
        return pts
    return (x - x0) / (x1 - x0) * pts


def risk_components(y: float, spr_bps: float, chg3m_bps: Optional[float]) -> dict:
    """สูตรตรงตาม FORMULA_TEXT — คืนส่วนประกอบทุกตัวเพื่อแสดงบนหน้าจอ"""
    level = round(_lin(y, 2.0, 15.0, 40.0), 1)
    spread = round(_lin(spr_bps, 0.0, 3000.0, 40.0), 1)
    trend = 0.0 if chg3m_bps is None else round(_lin(chg3m_bps, 0.0, 150.0, 20.0), 1)
    total = round(min(100.0, level + spread + trend), 1)
    tier = next(t for lo, hi, t in RISK_TIERS if lo <= total < hi)
    return {"level_pts": level, "spread_pts": spread, "trend_pts": trend,
            "trend_known": chg3m_bps is not None, "total": total, "tier": tier}


def staleness_label(last_d: date, today: date, freq: str) -> str:
    days = (today - last_d).days
    if freq == "D":
        return "สดวันนี้" if days <= 1 else f"ล่าสุด {last_d.isoformat()} (ช้า {days} วัน)"
    months = max(0, round(days / 30.4))
    return f"ข้อมูลเดือน {last_d.strftime('%m/%Y')} (ช้า ~{months} เดือน)"


# ---------------------------------------------------------------- กรอกเอง (manual)

def validate_manual(y: float, asof: str, today: date) -> tuple[bool, str]:
    """ตรวจค่าที่ผู้ใช้กรอกเอง — ต้องมีวันที่อ้างอิง และค่าอยู่ในช่วงสมเหตุผล"""
    try:
        d = datetime.strptime(asof.strip(), "%Y-%m-%d").date()
    except ValueError:
        return False, "วันที่อ้างอิงต้องเป็น YYYY-MM-DD"
    if d > today:
        return False, "วันที่อ้างอิงเป็นอนาคตไม่ได้"
    if not (0.0 < y < 60.0):
        return False, "yield ต้องอยู่ระหว่าง 0-60%"
    return True, ""


# ---------------------------------------------------------------- DEMO (ออฟไลน์)

# ค่าอ้างอิงจากภาพตัวอย่างของผู้ใช้ (ก.ค. 2026) — ใช้เฉพาะโหมด DEMO ติดป้ายเสมอ
DEMO_ASOF = date(2026, 7, 22)
DEMO_YIELDS: dict[str, float] = {
    "US": 4.57, "IN": 6.78, "ID": 7.27, "ZA": 8.68, "PL": 5.58,
    "BR": 14.69, "TR": 31.96, "MY": 3.65, "PH": 7.37, "RU": 16.75,
}


@dataclass
class CountryRow:
    spec: CountrySpec
    source: str                 # "FRED" | "MANUAL" | "DEMO" | "NONE"
    y: Optional[float] = None
    asof: Optional[date] = None
    chg3m_bps: Optional[float] = None
    spread: Optional[float] = None
    risk: dict = field(default_factory=dict)
    fresh_label: str = ""
    error: str = ""


def build_rows(fetched: dict[str, list[tuple[date, float]]],
               manual: dict[str, dict],
               today: date,
               demo: bool = False) -> list[CountryRow]:
    """ประกอบข้อมูลทุกประเทศเป็นแถว — ลำดับความสำคัญ: FRED จริง > MANUAL > (DEMO ถ้าเปิด) > NONE
    fetched: {code: obs} เฉพาะที่ดึงสำเร็จ · manual: {code: {"y":float,"asof":"YYYY-MM-DD"}}"""
    us_y: Optional[float] = None
    if "US" in fetched:
        us_y = fetched["US"][-1][1]
    elif demo:
        us_y = DEMO_YIELDS["US"]

    rows: list[CountryRow] = []
    for spec in COUNTRIES:
        r = CountryRow(spec=spec, source="NONE")
        if spec.code in fetched:
            obs = fetched[spec.code]
            d, v, chg = latest_and_trend(obs)
            r.source, r.y, r.asof, r.chg3m_bps = "FRED", v, d, chg
            r.fresh_label = staleness_label(d, today, spec.freq)
        elif spec.code in manual:
            m = manual[spec.code]
            ok, err = validate_manual(float(m["y"]), str(m["asof"]), today)
            if ok:
                r.source, r.y = "MANUAL", float(m["y"])
                r.asof = datetime.strptime(str(m["asof"]), "%Y-%m-%d").date()
                r.fresh_label = f"กรอกเอง ณ {r.asof.isoformat()} — ยังไม่ได้ตรวจสอบอิสระ"
            else:
                r.error = err
        elif demo:
            r.source, r.y, r.asof = "DEMO", DEMO_YIELDS[spec.code], DEMO_ASOF
            r.fresh_label = "DEMO — ไม่ใช่ข้อมูลจริง"
        if r.y is not None and us_y is not None:
            r.spread = 0.0 if spec.code == "US" else spread_bps(r.y, us_y)
            r.risk = risk_components(r.y, max(0.0, r.spread), r.chg3m_bps)
        rows.append(r)
    return rows


def source_check_report(fetcher: Optional[Fetcher] = None) -> list[tuple[str, str]]:
    """ปุ่ม 'ตรวจแหล่งข้อมูล': ลองดึงทุก series จริง แล้วรายงานผลตรง ๆ ต่อประเทศ
    (ใช้ตอน deploy เพื่อยืนยัน series ที่ยัง unverified)"""
    out: list[tuple[str, str]] = []
    for spec in COUNTRIES:
        if not spec.fred_series:
            out.append((spec.code, "ไม่มีแหล่งฟรี — " + spec.note))
            continue
        try:
            obs = fetch_fred_series(spec.fred_series, fetcher)
            d, v, _ = latest_and_trend(obs)
            out.append((spec.code, f"OK: {spec.fred_series} ล่าสุด {d.isoformat()} = {v:.2f}%"))
        except Exception as e:  # รายงาน error ตรง ๆ ไม่กลบ
            out.append((spec.code, f"FAIL: {spec.fred_series} — {e}"))
    return out
