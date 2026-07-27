# -*- coding: utf-8 -*-
"""แสดง "วันที่+เวลาของข้อมูล" ให้ทุกแท็บ — เพื่อตรวจว่าที่เห็นบนจอเป็นปัจจุบันจริง

สิ่งที่โมดูลนี้แยกให้ชัด (คนละเรื่องกัน อย่าปนกัน):
  1. **เวลาแท่งล่าสุด**  — มาจากตัวข้อมูลเอง เชื่อได้ 100%
  2. **แท่งล่าสุดปิดแล้วหรือยัง** — เดาจากเวลาทำการของตลาด (heuristic ไม่ใช่
     ข้อมูลจากตลาดจริง) ถ้าเดาผิดจะบอกว่าเดา ไม่ใช่ยืนยัน
  3. **เวลาที่โหลดข้อมูล** — เวลาที่ cache ของชุดนั้นถูกสร้าง คลาดจากเวลายิง
     API จริงได้ระดับวินาที-นาที (ชุด SET100 ใช้เวลาโหลด ~1-2 นาที)
  4. **อายุ cache / เหลืออีกกี่นาทีจะรีเฟรช** — คำนวณจาก TTL ที่ตั้งไว้

ข้อจำกัดที่ไม่ปิดบัง:
  - yfinance เป็นข้อมูล "สิ้นวัน" ที่อาจล่าช้า 15-20 นาทีหรือมากกว่า และแท่ง
    ของวันปัจจุบันระหว่างเวลาทำการเป็น **แท่งที่ยังไม่จบ** ค่าจะเปลี่ยนได้อีก
  - โมดูลนี้ไม่รู้ว่าตลาดหยุดวันไหน (วันหยุดพิเศษ ตลท./NYSE ไม่ได้ฝังไว้)
    จึงนับ "อายุข้อมูล" เป็นวันปฏิทินและวันทำการโดยประมาณเท่านั้น
"""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

import pandas as pd

try:                                    # py3.9+ มีมาให้ ถ้าไม่มีถอยไป UTC+7 คงที่
    from zoneinfo import ZoneInfo
    TZ_TH = ZoneInfo("Asia/Bangkok")
    TZ_NY = ZoneInfo("America/New_York")
    _HAS_TZDB = True
except Exception:                       # pragma: no cover - สภาพแวดล้อมไม่มี tzdata
    TZ_TH = timezone(timedelta(hours=7))
    TZ_NY = timezone(timedelta(hours=-5))
    _HAS_TZDB = False

FMT = "%d/%m/%Y %H:%M:%S"
FMT_SHORT = "%d/%m/%Y %H:%M"

# ตลาด: (timezone, เวลาปิด, ทำงานเสาร์-อาทิตย์ไหม, ชื่อ)
MARKETS = {
    "SET": (TZ_TH, time(16, 30), False, "ตลท. (จ-ศ 10:00-16:30 น.)"),
    "US": (TZ_NY, time(16, 0), False, "สหรัฐฯ (จ-ศ 09:30-16:00 ET)"),
    "COMEX": (TZ_NY, time(17, 0), False, "COMEX (ปิดชำระ ~17:00 ET)"),
    "24H": (timezone.utc, time(0, 0), True, "24/7 (แท่งวันปิดที่ 00:00 UTC)"),
    "NONE": (TZ_TH, time(23, 59), True, "ไม่ผูกกับตลาดใด"),
}


def now_th() -> datetime:
    return datetime.now(TZ_TH)


def fmt(dt: datetime | None, short: bool = False) -> str:
    if dt is None:
        return "—"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ_TH)
    return dt.astimezone(TZ_TH).strftime(FMT_SHORT if short else FMT) + " น."


def last_index(obj) -> pd.Timestamp | None:
    """ดึงเวลาแท่งล่าสุดจาก DataFrame / Series / DatetimeIndex / dict ของ DataFrame"""
    try:
        if obj is None:
            return None
        if isinstance(obj, dict):
            ts = [last_index(v) for v in obj.values()]
            ts = [t for t in ts if t is not None]
            return max(ts) if ts else None
        idx = obj.index if hasattr(obj, "index") else obj
        if len(idx) == 0:
            return None
        return pd.Timestamp(idx[-1])
    except Exception:
        return None


def bar_is_closed(last_ts, market: str = "SET", now: datetime | None = None) -> bool | None:
    """แท่งล่าสุด 'น่าจะ' ปิดแล้วหรือยัง — คืน None ถ้าประเมินไม่ได้.

    heuristic ล้วน: ถ้าวันที่ของแท่ง = วันนี้ (ตามเวลาตลาด) และยังไม่ถึงเวลาปิด
    ให้ถือว่า **ยังไม่ปิด**. ไม่รู้จักวันหยุดพิเศษ.
    """
    if last_ts is None:
        return None
    tz, close_t, _wknd, _lbl = MARKETS.get(market, MARKETS["NONE"])
    now = (now or datetime.now(tz)).astimezone(tz)
    d = pd.Timestamp(last_ts).date()
    if d < now.date():
        return True
    if d > now.date():
        return None                     # ข้อมูลล้ำหน้า — ผิดปกติ ไม่เดา
    if market == "24H":
        return False                    # แท่งของ "วันนี้" ใน 24/7 ยังไม่จบเสมอ
    return now.time() >= close_t


def age_text(last_ts, now: datetime | None = None) -> str:
    """อายุข้อมูลเทียบเวลาไทยตอนนี้ (วันปฏิทิน + ประมาณวันทำการ)"""
    if last_ts is None:
        return "—"
    now = now or now_th()
    d0 = pd.Timestamp(last_ts).date()
    days = (now.date() - d0).days
    if days <= 0:
        return "วันนี้"
    biz = int(pd.bdate_range(d0, now.date()).size) - 1
    if days == 1:
        return "1 วัน (เมื่อวาน)"
    return f"{days} วันปฏิทิน (~{max(biz, 0)} วันทำการ)"


def staleness(last_ts, market: str = "SET", max_biz_days: int = 2,
              now: datetime | None = None) -> str:
    """คืน 'ok' | 'warn' | 'bad' — เกณฑ์ตรงไปตรงมา ไม่ตีความเกิน"""
    if last_ts is None:
        return "bad"
    now = now or now_th()
    biz = int(pd.bdate_range(pd.Timestamp(last_ts).date(), now.date()).size) - 1
    if market == "24H":
        cal = (now.date() - pd.Timestamp(last_ts).date()).days
        return "ok" if cal <= 1 else ("warn" if cal <= 3 else "bad")
    if biz <= max_biz_days:
        return "ok"
    return "warn" if biz <= max_biz_days + 3 else "bad"


def ttl_text(loaded_at: datetime | None, ttl_sec: int,
             now: datetime | None = None) -> str:
    if loaded_at is None or not ttl_sec:
        return "—"
    now = now or now_th()
    if loaded_at.tzinfo is None:
        loaded_at = loaded_at.replace(tzinfo=TZ_TH)
    used = (now - loaded_at.astimezone(TZ_TH)).total_seconds()
    left = max(0.0, ttl_sec - used)
    return (f"โหลดมาแล้ว {int(used // 60)} นาที · "
            f"cache หมดอายุอีก ~{int(left // 60)} นาที")


# --------------------------------------------------------------------------
def describe(name: str, data, market: str = "SET",
             loaded_at: datetime | None = None, ttl_sec: int = 3600,
             note: str = "") -> dict:
    """สรุปสถานะข้อมูล 1 ชุด (ใช้ทั้งใน UI และในเทสต์)"""
    ts = last_index(data)
    closed = bar_is_closed(ts, market, None)
    lvl = staleness(ts, market)
    return {
        "ชุดข้อมูล": name,
        "แท่งล่าสุด": ("—" if ts is None else
                        (str(ts.date()) if ts.hour == 0 and ts.minute == 0
                         else ts.strftime("%d/%m/%Y %H:%M"))),
        "อายุ": age_text(ts),
        "แท่งปิดแล้ว?": ("ปิดแล้ว" if closed is True else
                          ("ยังไม่ปิด (ค่ายังเปลี่ยนได้)" if closed is False
                           else "ประเมินไม่ได้")),
        "โหลดเมื่อ": fmt(loaded_at),
        "cache": ttl_text(loaded_at, ttl_sec),
        "ตลาดอ้างอิง": MARKETS.get(market, MARKETS["NONE"])[3],
        "ระดับ": lvl,
        "หมายเหตุ": note,
    }


def render(st, items: list[dict], compact: bool = True) -> None:
    """วาดแถบเวลาข้อมูลบนหน้าเพจ — เรียกท้าย/ต้นทุกหน้าได้เลย

    items = ผลลัพธ์จาก describe() หลายชุด
    """
    if not items:
        return
    seen, uniq = set(), []
    for it in items:                    # กันซ้ำ ถ้าหน้าหนึ่งลงทะเบียนชุดเดิมหลายรอบ
        if it["ชุดข้อมูล"] in seen:
            continue
        seen.add(it["ชุดข้อมูล"])
        uniq.append(it)
    items = uniq
    worst = "ok"
    for it in items:
        if it["ระดับ"] == "bad":
            worst = "bad"
        elif it["ระดับ"] == "warn" and worst != "bad":
            worst = "warn"
    icon = {"ok": "🟢", "warn": "🟡", "bad": "🔴"}[worst]
    head = " · ".join(
        f"**{it['ชุดข้อมูล']}** แท่งล่าสุด {it['แท่งล่าสุด']} ({it['อายุ']})"
        for it in items[:3])
    live = [it["ชุดข้อมูล"] for it in items
            if it["แท่งปิดแล้ว?"].startswith("ยังไม่ปิด")]
    tail = (f" · ⚠️ {', '.join(live)}: แท่งล่าสุด**ยังไม่ปิด** ค่าที่เห็นเปลี่ยนได้อีก"
            if live else "")
    st.caption(f"{icon} เวลาเครื่อง {fmt(now_th())} (เวลาไทย) · {head}{tail}")
    if compact:
        with st.expander("🕒 รายละเอียดความสดของข้อมูลหน้านี้"):
            _table(st, items)
    else:
        _table(st, items)


def _table(st, items: list[dict]) -> None:
    icons = {"ok": "🟢", "warn": "🟡", "bad": "🔴"}
    df = pd.DataFrame([{
        "": icons[it["ระดับ"]], "ชุดข้อมูล": it["ชุดข้อมูล"],
        "แท่งล่าสุด": it["แท่งล่าสุด"], "อายุ": it["อายุ"],
        "สถานะแท่ง": it["แท่งปิดแล้ว?"], "โหลดเมื่อ": it["โหลดเมื่อ"],
        "cache": it["cache"], "ตลาด": it["ตลาดอ้างอิง"],
        "หมายเหตุ": it["หมายเหตุ"],
    } for it in items])
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.caption(
        "อ่านค่าอย่างระวัง: **แท่งล่าสุด** มาจากตัวข้อมูลจริง เชื่อได้ · "
        "**สถานะแท่ง** เป็นการเดาจากเวลาทำการตลาด ไม่ได้เช็คกับตลาดจริงและ"
        "ไม่รู้จักวันหยุดพิเศษ · **โหลดเมื่อ** คือเวลาที่ cache ชุดนั้นถูกสร้าง "
        "(ชุด SET100 ใช้เวลาโหลด 1-2 นาที เวลาจริงที่ยิง API จึงเร็วกว่านี้เล็กน้อย) · "
        "ราคาจาก yfinance เป็นข้อมูลสิ้นวันที่อาจล่าช้า — ไม่ใช่ราคาเรียลไทม์")
