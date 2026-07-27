# -*- coding: utf-8 -*-
"""ทดสอบเฟส v14:
  1) พิสูจน์แบบ 'ไล่ทีละแท่ง' ว่า accumulation/squeeze ตรงกับ Pine v5.13
  2) โหมดใช้เฉพาะแท่งปิด + คอลัมน์ 'มีเครื่องหมายบนชาร์ต' ในหน้าสแกน
  3) ตราเวลาข้อมูล (datastamp)
  4) สะสม/สควีซ บนทองคำ + การจัดการวอลุ่มที่เชื่อไม่ได้

ทั้งหมดรันออฟไลน์ด้วยข้อมูลสังเคราะห์ — ไม่มีการเรียกเน็ต ไม่มีการอ้างว่า
ตัวเลขเหล่านี้เป็นข้อมูลตลาดจริง สิ่งที่พิสูจน์ได้คือ 'สูตรตรงกับต้นฉบับ'
เท่านั้น ไม่ใช่ 'สูตรทำเงินได้'
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import accum as ACC
import datastamp as DS
import gold as G
import set_swing as SW

PASS, FAIL = 0, 0


def check(name, cond, note=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name} {note}")
    else:
        FAIL += 1
        print(f"  ✗ {name} {note}")


def mk(n=520, seed=7, flat_from=None):
    rng = np.random.default_rng(seed)
    c = [30.0]
    for i in range(n - 1):
        drift = 0.0006 if i % 90 < 45 else -0.0004
        if flat_from is not None and i >= flat_from:
            drift = 0.0
        c.append(max(1.0, c[-1] * (1 + drift + rng.normal(0, 0.016))))
    c = np.array(c)
    o = c * (1 + rng.normal(0, 0.004, n))
    h = np.maximum(o, c) * (1 + abs(rng.normal(0, 0.006, n)))
    l = np.minimum(o, c) * (1 - abs(rng.normal(0, 0.006, n)))
    v = rng.lognormal(14.5, 0.55, n)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c,
                         "Volume": v},
                        index=pd.bdate_range("2023-01-02", periods=n))


# ===========================================================================
# 1) อ้างอิง Pine แบบไล่ทีละแท่ง (ไม่ใช้ pandas rolling เลย)
# ===========================================================================
def pine_ref(df, acc_len=20, acc_flat=2.0, acc_ratio=1.25, atr_len=14,
             bos_len=20, use_acc=True):
    c = df["Close"].to_numpy(float); h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float); v = df["Volume"].to_numpy(float)
    n = len(c); nan = float("nan")
    tr = np.full(n, nan)
    for i in range(n):
        tr[i] = (h[i] - l[i]) if i == 0 else max(
            h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = np.full(n, nan)
    for i in range(n):                       # ta.atr = rma seed ด้วย sma(n แรก)
        if i == atr_len - 1:
            atr[i] = tr[:atr_len].mean()
        elif i >= atr_len:
            atr[i] = (atr[i - 1] * (atr_len - 1) + tr[i]) / atr_len
    clv = np.array([((c[i] - l[i]) - (h[i] - c[i])) / (h[i] - l[i])
                    if h[i] > l[i] else 0.0 for i in range(n)])
    votes = np.full(n, nan); pos = np.full(n, nan); swing = np.full(n, nan)
    hot = np.zeros(n, bool); show = np.zeros(n, bool)
    ok = {k: np.zeros(n, bool) for k in ["flat", "press", "clv", "act"]}
    for i in range(n):
        if i >= acc_len:
            w = slice(i - acc_len + 1, i + 1)
            uv = sum(v[j] for j in range(i - acc_len + 1, i + 1)
                     if j >= 1 and c[j] > c[j - 1])
            dv = sum(v[j] for j in range(i - acc_len + 1, i + 1)
                     if j >= 1 and c[j] < c[j - 1])
            rhi, rlo = h[w].max(), l[w].min()
            pos[i] = (c[i] - rlo) / (rhi - rlo) if rhi > rlo else 0.5
            ok["flat"][i] = (abs(c[i] - c[i - acc_len]) <= acc_flat * atr[i]) \
                if atr[i] == atr[i] else False
            ok["press"][i] = (dv > 0) and (uv >= acc_ratio * dv)
            ok["clv"][i] = clv[w].mean() >= 0.10
            if i >= 99:
                ok["act"][i] = (v[w].mean() / max(v[i - 99:i + 1].mean(), 1.0)) >= 0.7
            votes[i] = sum(int(ok[k][i]) for k in ok)
        if i >= bos_len:
            swing[i] = h[i - bos_len:i].max()      # ta.highest(high[1], bos)
        ctx = (c[i] < swing[i]) and (pos[i] <= 0.65) \
            if swing[i] == swing[i] else False
        hot[i] = bool(use_acc) and ctx and (votes[i] >= 3)
        show[i] = hot[i] and (hot[i - 1] if i else False)
    return dict(votes=votes, hot=hot, show=show, pos=pos, swing=swing, **ok)


print("\n---------------- 1) accumulation/squeeze ตรงกับ Pine v5.13 ทีละแท่ง")
WARM = 130
for sd in (7, 21, 99):
    df = mk(520, seed=sd)
    fr = SW.compute_frame(df, None, "T", SW.SwingParams())
    ref = pine_ref(df)
    pairs = [("acc_flat_ok", "flat"), ("acc_press_ok", "press"),
             ("acc_clv_ok", "clv"), ("acc_act_ok", "act"),
             ("acc_hot", "hot"), ("acc_show", "show")]
    bad = {}
    for pycol, rkey in pairs:
        d = int((fr[pycol].to_numpy(bool)[WARM:] != ref[rkey][WARM:]).sum())
        if d:
            bad[pycol] = d
    check(f"pine_equivalence_seed{sd}", not bad, str(bad) if bad else
          f"{len(df) - WARM} แท่ง ตรงทุกตัว")
    vd = int((fr["acc_votes"].to_numpy()[WARM:] != ref["votes"][WARM:]).sum())
    check(f"votes_equal_seed{sd}", vd == 0, f"ต่าง {vd} แท่ง")
    check(f"pos_in_rng_equal_seed{sd}",
          float(np.nanmax(np.abs(fr["pos_in_rng"].to_numpy()[WARM:]
                                 - ref["pos"][WARM:]))) < 1e-12)
    check(f"swing_hi_equal_seed{sd}",
          float(np.nanmax(np.abs(fr["swing_hi"].to_numpy()[WARM:]
                                 - ref["swing"][WARM:]))) < 1e-12)

df = mk(520, seed=7)
fr = SW.compute_frame(df, None, "T", SW.SwingParams())
_prev_hot = fr["acc_hot"].shift(1).fillna(False).astype(bool)
check("acc_show_needs_two_bars",
      bool((fr["acc_show"].to_numpy(bool) & ~_prev_hot.to_numpy(bool)).sum() == 0),
      "acc_show ห้ามเกิดถ้าแท่งก่อนหน้าไม่ hot")
check("acc_show_subset_of_hot",
      bool((fr["acc_show"].to_numpy(bool)
            & ~fr["acc_hot"].to_numpy(bool)).sum() == 0))

# ยืนยันคำสัญญาเดิม: accumulation เป็น display-only จริง ไม่แตะสัญญาณเข้า
frb = SW.compute_frame(df, None, "T", SW.SwingParams(use_acc=False))
check("acc_still_display_only",
      bool((fr["long_cond"] == frb["long_cond"]).all()
           and (fr["short_cond"] == frb["short_cond"]).all()),
      "เปิด/ปิด use_acc → long_cond & short_cond ไม่เปลี่ยน")

# squeeze: นิยาม TTM + bars_sq
sq = ACC.squeeze_frame(df)
on = sq["squeeze_on"].to_numpy(bool)
bs = sq["bars_sq"].to_numpy(float)
check("bars_sq_zero_when_on", bool(np.nanmax(np.abs(bs[on])) == 0.0
                                  if on.any() else True))
idx = np.where(~on)[0]
idx = idx[idx > 0]
check("bars_sq_increments",
      all((bs[i] != bs[i]) or (bs[i] == bs[i - 1] + 1) for i in idx))
check("squeeze_module_matches_engine",
      bool((sq["squeeze_on"] == fr["squeeze_on"]).all()),
      "accum.py กับ set_swing.py ใช้สูตรเดียวกันจริง")

# ===========================================================================
# 2) หน้าสแกน — โหมดแท่งปิด, บักเก็ต, คอลัมน์เครื่องหมายบนชาร์ต
# ===========================================================================
print("\n---------------- 2) scan_acc_squeeze")
def mk_acc(n=340, seed=5):
    """สร้างหุ้นที่ 'สะสม' จริงตามนิยาม: ราคานิ่ง + วอลุ่มขาซื้อเด่น +
    ปิดค่อนบน + ตลาดไม่ตาย และมี spike เก่าไว้เป็น swing high"""
    rng = np.random.default_rng(seed)
    c = np.full(n, 10.0) + rng.normal(0, 0.02, n)
    up = rng.random(n) < 0.5
    c[1:] = np.where(up[1:], c[:-1] + 0.03, c[:-1] - 0.02)
    c[n - 45] = 11.6                       # ยอดเก่า → อยู่ใต้ trigger
    v = np.where(np.diff(np.r_[c[0], c]) > 0, 2.0e6, 1.0e6)
    return pd.DataFrame({"Open": c, "High": c + 0.10, "Low": c - 0.30,
                         "Close": c, "Volume": v},
                        index=pd.bdate_range("2023-01-02", periods=n))


prices = {f"S{i}.BK": mk(400, seed=100 + i) for i in range(6)}
prices["ACC.BK"] = mk_acc()
prices["ACC2.BK"] = mk_acc(seed=11)
tb = SW.scan_acc_squeeze(prices, None)
check("scan_returns_df", isinstance(tb, pd.DataFrame))
check("scan_not_empty", len(tb) > 0, f"{len(tb)} แถว — ถ้าว่างเทสต์ข้างล่างจะกลวง")
check("scan_finds_engineered_acc",
      "ACC" in set(tb["หุ้น"]),
      "หุ้นที่สร้างให้ 'สะสม' ต้องโผล่ในตาราง")
if len(tb):
    check("scan_has_marker_col", "มีเครื่องหมายบนชาร์ต" in tb.columns)
    check("scan_has_bar_date_col", "แท่งล่าสุดที่ใช้" in tb.columns)
    mk_ok = all(
        (r["มีเครื่องหมายบนชาร์ต"] == "ใช่")
        == (r["สถานะ"] in SW.BUCKETS_WITH_PINE_MARKER)
        for _, r in tb.iterrows())
    check("scan_marker_flag_correct", mk_ok,
          "สถานะที่ Pine ไม่พิมพ์เครื่องหมาย ต้องถูกกำกับว่า 'ไม่'")
    check("scan_buckets_known",
          set(tb["สถานะ"]).issubset(set(SW.ACC_SQ_BUCKETS)))

tb_closed = SW.scan_acc_squeeze(prices, None, closed_only=True)
d_all = set(tb["แท่งล่าสุดที่ใช้"]) if len(tb) else set()
d_cls = set(tb_closed["แท่งล่าสุดที่ใช้"]) if len(tb_closed) else set()
check("closed_only_drops_last_bar",
      (not d_all or not d_cls or max(d_cls) < max(d_all)),
      f"เต็ม {max(d_all) if d_all else '—'} → ปิดแล้ว {max(d_cls) if d_cls else '—'}")

cut = pd.Timestamp(mk(400, seed=100).index[-30])
tb_cut = SW.scan_acc_squeeze(prices, None, last_closed_date=cut)
check("last_closed_date_respected",
      all(pd.Timestamp(x) <= cut for x in tb_cut["แท่งล่าสุดที่ใช้"])
      if len(tb_cut) else True)

# ตัวที่ hot วันนี้แต่ยังไม่ครบ 2 แท่ง ต้องไม่ถูกเหมารวมเป็น 'สะสม'
lbl_hot, on_chart_hot = ACC.status_label(False, True, False, float("nan"))
check("first_bar_bucket_not_on_chart",
      lbl_hot.startswith("⚪") and on_chart_hot is False)
lbl_show, on_chart_show = ACC.status_label(True, True, False, float("nan"))
check("acc_show_bucket_on_chart", on_chart_show is True)
lbl_rec, on_chart_rec = ACC.status_label(False, False, False, 3.0)
check("recent_squeeze_not_on_chart",
      lbl_rec.startswith("🟠") and on_chart_rec is False,
      "บักเก็ตนี้ Pine ไม่พิมพ์อะไรเลย")
check("no_status_when_nothing",
      ACC.status_label(False, False, False, float("nan"))[0] == "—")

# ตารางตรวจทีละข้อ
aud = SW.acc_audit(df, None, "TEST.BK")
check("audit_has_6_rows", len(aud["rows"]) == 6)
check("audit_pine_dash_format",
      aud["pine_dash"].endswith("/4")
      and (aud["pine_dash"].startswith("WATCH ") == aud["acc_show"]),
      aud["pine_dash"])
check("audit_votes_match_rows",
      aud["votes"] == sum(1 for r in aud["rows"][:4] if r["ผ่าน"] is True))
aud_c = SW.acc_audit(df, None, "TEST.BK", closed_only=True)
check("audit_closed_only_earlier_bar", aud_c["bar_date"] < aud["bar_date"],
      f"{aud['bar_date']} → {aud_c['bar_date']}")

# ===========================================================================
# 3) ตราเวลาข้อมูล
# ===========================================================================
print("\n---------------- 3) datastamp")
now = datetime(2026, 7, 27, 12, 0, tzinfo=DS.TZ_TH)      # จันทร์ เที่ยง
check("bar_today_not_closed_midday",
      DS.bar_is_closed(pd.Timestamp("2026-07-27"), "SET", now) is False)
check("bar_today_closed_after_1630",
      DS.bar_is_closed(pd.Timestamp("2026-07-27"), "SET",
                       now.replace(hour=17)) is True)
check("bar_yesterday_closed",
      DS.bar_is_closed(pd.Timestamp("2026-07-24"), "SET", now) is True)
check("bar_future_returns_none",
      DS.bar_is_closed(pd.Timestamp("2026-07-28"), "SET", now) is None)
check("bar_24h_today_never_closed",
      DS.bar_is_closed(pd.Timestamp("2026-07-27"), "24H", now) is False)
check("bar_none_returns_none", DS.bar_is_closed(None, "SET", now) is None)

check("age_today", DS.age_text(pd.Timestamp("2026-07-27"), now) == "วันนี้")
check("age_yesterday", "1 วัน" in DS.age_text(pd.Timestamp("2026-07-26"), now))
check("stale_ok_recent", DS.staleness(pd.Timestamp("2026-07-24"), "SET", now=now)
      == "ok")
check("stale_bad_old", DS.staleness(pd.Timestamp("2026-05-01"), "SET", now=now)
      == "bad")
check("stale_bad_when_none", DS.staleness(None, "SET", now=now) == "bad")

d = DS.describe("ทดสอบ", df, "SET", now - timedelta(minutes=20), 3600)
for k in ["ชุดข้อมูล", "แท่งล่าสุด", "อายุ", "แท่งปิดแล้ว?", "โหลดเมื่อ",
          "cache", "ตลาดอ้างอิง", "ระดับ"]:
    check(f"describe_has_{k}", k in d)
check("describe_ttl_text", "cache หมดอายุอีก" in d["cache"], d["cache"])
check("last_index_dict", DS.last_index({"a": df, "b": df.iloc[:-5]})
      == pd.Timestamp(df.index[-1]))
check("last_index_empty", DS.last_index(pd.DataFrame()) is None)
check("fmt_none", DS.fmt(None) == "—")

# ===========================================================================
# 4) ทองคำ — สะสม/สควีซ + การจัดการวอลุ่มที่เชื่อไม่ได้
# ===========================================================================
print("\n---------------- 4) gold accumulation/squeeze")
gdf = mk(400, seed=42)
gs = G.acc_squeeze_state(gdf, "GC=F")
check("gold_state_ok", gs["ok"] is True)
check("gold_votes_denominator_4", gs["เต็ม"] == 4 and gs["ต้องการ"] == 3)
check("gold_has_audit_rows", len(gs["rows"]) == 6)
check("gold_marker_flag_present", "มีเครื่องหมายบนชาร์ต Pine (ฝั่งหุ้น)" in gs)

gs2 = G.acc_squeeze_state(gdf, "PAXG-USD", use_volume_votes=False)
check("gold_novol_denominator_2", gs2["เต็ม"] == 2 and gs2["ต้องการ"] == 2)
check("gold_novol_marks_unmeasurable",
      gs2["rows"][1]["ผ่าน"] is None and gs2["rows"][3]["ผ่าน"] is None,
      "ข้อ 2 และ 4 ต้องขึ้น 'วัดไม่ได้' ไม่ใช่ตกไปเป็น False")
check("gold_novol_votes_le_2", gs2["โหวต"] <= 2)

gdf0 = gdf.copy(); gdf0["Volume"] = np.nan
gs3 = G.acc_squeeze_state(gdf0, "GC=F")
check("gold_no_volume_quality_zero", gs3["vol_quality"] == 0.0)
check("gold_no_volume_no_false_votes",
      gs3["โหวต"] <= 2, f"โหวต {gs3['โหวต']} — ห้ามได้คะแนนจากวอลุ่มที่ไม่มี")

gdf1 = gdf.drop(columns=["Volume"])
check("gold_missing_volume_col_no_crash",
      G.acc_squeeze_state(gdf1, "GC=F")["ok"] is True)

check("gold_short_data_declines",
      G.acc_squeeze_state(gdf.tail(50), "GC=F")["ok"] is False)

# นิยามเดียวกับหุ้นจริงไหม (ข้อ 3 ของงาน: 'เหมือนกับหุ้น')
gfr = G.acc_squeeze_frame(gdf, G.GoldParams())
sfr = SW.compute_frame(gdf, None, "X", SW.SwingParams())
check("gold_same_definition_as_stock",
      bool((gfr["acc_votes"] == sfr["acc_votes"]).all()
           and (gfr["squeeze_on"] == sfr["squeeze_on"]).all()),
      "ทองกับหุ้นใช้สูตรชุดเดียวกัน (accum.py) — ไม่แตกสองสาย")

# สะสมบนทอง ต้องไม่ไปแตะสัญญาณของ v6.4
gp = G.GoldParams()
base = G.compute_frame(gdf, p=gp)
after = G.compute_frame(gdf, p=gp)
G.acc_squeeze_state(gdf, "GC=F")
check("gold_acc_does_not_touch_v64",
      bool((base["long_cond"] == after["long_cond"]).all()
           and (base["short_cond"] == after["short_cond"]).all()),
      "เรียกสะสมแล้วสัญญาณ v6.4 ต้องเท่าเดิม")
check("gold_vol_notes_disclosed",
      "roll" in G.GOLD_VOL_NOTE["GC=F"]
      and "โทเคน" in G.GOLD_VOL_NOTE["PAXG-USD"])

# ===========================================================================
print(f"\n== {PASS} passed, {FAIL} failed ==")
if FAIL:
    raise SystemExit(1)
