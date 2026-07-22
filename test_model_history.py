# -*- coding: utf-8 -*-
"""เทสต์ออฟไลน์: model_history.py (ใช้ demo_bundle สังเคราะห์ — ไม่แตะเน็ต)"""
from datetime import date

import numpy as np
import pandas as pd

import data_sources as D
import model_history as MH
import models6 as M

ok = fail = 0


def check(name, cond, extra=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  ✓ {name}")
    else:
        fail += 1
        print(f"  ✗ {name} {extra}")


b = D.demo_bundle()
DATA = dict(b["fred"])
DATA["MOVE"] = b["move"]
SC = M.score_models(DATA)

# ---------------------------------------------------------------- daily_history
H = MH.all_daily_history(DATA, tail_days=30)
check("hist_has_all_models", set(H.columns) == set(M.MODEL_DEFS), str(list(H.columns)))
check("hist_30_rows", len(H) == 30, str(len(H)))
check("hist_bounds_0_100", bool(((H.dropna() >= 0) & (H.dropna() <= 100)).all().all()))
check("hist_index_business_daily", bool(H.index.is_monotonic_increasing)
      and (H.index[-1] - H.index[0]).days <= 45)

# จุดสุดท้ายของเส้นย้อนหลังต้องเท่าคะแนนหน้าโมเดลวันนี้ (นิยามเดียวกัน)
mismatch = {}
for k in M.MODEL_DEFS:
    hv = H[k].dropna()
    sv = SC[k]["score"]
    if len(hv) and sv == sv:
        if abs(round(float(hv.iloc[-1]), 1) - sv) > 0.11:
            mismatch[k] = (round(float(hv.iloc[-1]), 1), sv)
check("last_point_equals_score_models", not mismatch, str(mismatch))

# ffill: component รายเดือน (CPI) ต้องทำให้เส้น inflation_oil "ถือค่า" ระหว่างเดือนได้
h_inf = MH.daily_history(DATA, "inflation_oil", tail_days=30)
check("monthly_comp_still_daily_line", len(h_inf) >= 25, str(len(h_inf)))

# series สั้นกว่า MIN_N ต้องถูกข้าม (ไม่พังและไม่มั่วค่า)
short = {"DGS10": pd.Series(np.linspace(4, 4.5, 30),
                            index=pd.bdate_range(end="2026-07-21", periods=30))}
check("short_series_skipped", MH.daily_history(short, "yield_shock").empty
      or MH.daily_history(short, "yield_shock").isna().all())

# FFILL_LIMIT: ข้อมูลตายนาน 60 วัน → ค่าท้ายเส้นต้องเป็น NaN/หาย ไม่ลากยาวเงียบๆ
idx_old = pd.bdate_range(end=pd.Timestamp("2026-05-20"), periods=400)
stale = {"BAMLH0A0HYM2": pd.Series(np.linspace(3, 5, 400), index=idx_old),
         "STLFSI4": pd.Series(np.linspace(-0.5, 1.0, 400), index=idx_old)}
h_stale = MH.daily_history(stale, "credit_crisis", tail_days=30,
                           end=pd.Timestamp("2026-07-21"))
check("stale_data_not_dragged", h_stale.empty or
      h_stale.reindex(pd.bdate_range(end="2026-07-21", periods=5)).isna().all())

# ---------------------------------------------------------------- snapshots
df0 = MH.empty_snap()
sc_flat = {k: SC[k]["score"] for k in M.MODEL_DEFS}
df1, ok1, msg1 = MH.append_snapshot(df0, date(2026, 7, 21), sc_flat, 44.2, "live")
check("snap_append_ok", ok1 and len(df1) == 1, msg1)
df2, ok2, _ = MH.append_snapshot(df1, date(2026, 7, 22), sc_flat, 45.1, "live")
check("snap_second_day", ok2 and len(df2) == 2)
df3, ok3, _ = MH.append_snapshot(df2, date(2026, 7, 22),
                                 {k: 50.0 for k in M.MODEL_DEFS}, 50.0, "live")
check("snap_same_day_overwrite", ok3 and len(df3) == 2
      and float(df3[df3["date"] == date(2026, 7, 22)]["composite"].iloc[0]) == 50.0)
df4, ok4, msg4 = MH.append_snapshot(df3, date(2026, 7, 20), sc_flat, 40.0, "live")
check("snap_refuse_past", (not ok4) and len(df4) == 2 and "ห้าม" in msg4, msg4)
df5, ok5, msg5 = MH.append_snapshot(df3, date(2026, 7, 23), sc_flat, 40.0, "demo")
check("snap_refuse_demo", (not ok5) and "DEMO" in msg5, msg5)

# CSV ไป-กลับ + ความทนไฟล์เพี้ยน
import io
rt, probs = MH.load_snapshots(io.BytesIO(MH.to_csv_bytes(df3)))
check("snap_csv_roundtrip", len(rt) == 2 and list(rt.columns) == MH.SNAP_COLS, str(probs))
messy = "date,bank_run\n2026-07-21,55\nnotadate,10\n2026-07-21,60\n"
rt2, probs2 = MH.load_snapshots(io.StringIO(messy))
check("snap_tolerates_messy_file", len(rt2) == 1
      and float(rt2["bank_run"].iloc[0]) == 60.0 and len(probs2) >= 2, str(probs2))

# ---------------------------------------------------------------- github helper
class _Resp:
    def __init__(self, code, js):
        self.status_code, self._js, self.text = code, js, str(js)
    def json(self):
        return self._js

calls = {}
def fake_get(api, **kw):
    calls["get"] = (api, kw)
    return _Resp(200, {"sha": "abc123"})
def fake_put(api, **kw):
    calls["put"] = (api, kw)
    return _Resp(200, {"commit": {"html_url": "https://x/commit/1"}})

okg, msg = MH.github_put_file("o", "r", "main", "model_history.csv",
                              b"date\n", "tok", "log", get=fake_get, put=fake_put)
check("gh_ok_url", okg and msg.startswith("https://"), msg)
check("gh_sha_forwarded", calls["put"][1]["json"].get("sha") == "abc123")
check("gh_b64_content", calls["put"][1]["json"]["content"] == "ZGF0ZQo=")
check("gh_api_path", calls["put"][0].endswith("/repos/o/r/contents/model_history.csv"))

def fake_put_fail(api, **kw):
    return _Resp(422, {"message": "Invalid request"})
okf, msgf = MH.github_put_file("o", "r", "main", "p", b"x", "tok", "m",
                               get=fake_get, put=fake_put_fail)
check("gh_fail_reported", (not okf) and "422" in msgf, msgf)
okn, msgn = MH.github_put_file("o", "r", "main", "p", b"x", "  ", "m",
                               get=fake_get, put=fake_put)
check("gh_empty_token_refused", not okn)

# ข้อความความซื่อสัตย์ต้องมีสาระสำคัญครบ
check("caveat_mentions_revision", "revise" in MH.RECOMPUTE_CAVEAT
      and "look-ahead" in MH.RECOMPUTE_CAVEAT)
check("caveat_mentions_news_fact", "ข่าว" in MH.RECOMPUTE_CAVEAT)
check("snap_caveat_appendonly", "append-only" in MH.SNAP_CAVEAT
      and "DEMO" in MH.SNAP_CAVEAT)

print(f"\n== {ok} passed, {fail} failed ==")
raise SystemExit(1 if fail else 0)
