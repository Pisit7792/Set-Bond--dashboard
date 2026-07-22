# -*- coding: utf-8 -*-
"""เทสต์ออฟไลน์: countries.py + worldmon.py (ไม่ใช้เน็ต — ใช้ fetcher จำลอง)"""
from datetime import date

import countries as C
import worldmon as W

ok = fail = 0


def check(name, cond, extra=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  ✓ {name}")
    else:
        fail += 1
        print(f"  ✗ {name} {extra}")


TODAY = date(2026, 7, 22)

# ---------------------------------------------------------------- config
codes = [c.code for c in C.COUNTRIES]
check("codes_unique", len(codes) == len(set(codes)))
check("us_daily_verified", C.spec_by_code("US").freq == "D" and C.spec_by_code("US").verified)
check("in_monthly_verified", C.spec_by_code("IN").verified and C.spec_by_code("IN").freq == "M")
no_src = [c.code for c in C.COUNTRIES if c.fred_series is None]
check("no_free_source_honest", set(no_src) == {"BR", "TR", "MY", "PH", "RU"}, str(no_src))
check("ru_note_discontinued", "หยุดเผยแพร่" in C.spec_by_code("RU").note)
check("unverified_flagged", all((not c.verified) for c in C.COUNTRIES
                                if c.fred_series and c.code in {"ID", "ZA", "PL"}))
check("freq_labels_exist", all(c.freq in C.FREQ_LABEL for c in C.COUNTRIES))

# ---------------------------------------------------------------- FRED parse
CSV = ("DATE,XXX\n2026-01-01,4.10\n2026-02-01,.\n2026-03-01,4.30\n"
       "2026-04-01,4.40\n2026-05-01,bad\n2026-06-01,4.55\n2026-07-01,4.57\n")
obs = C.fetch_fred_series("XXX", fetcher=lambda url: CSV)
check("parse_skips_missing", len(obs) == 5, str(len(obs)))
check("parse_sorted", obs[0][0] < obs[-1][0])
d, v, chg = C.latest_and_trend(obs)
check("latest_value", v == 4.57 and d == date(2026, 7, 1))
# cutoff = 1 ก.ค. − 95 วัน = 28 มี.ค. → ฐาน = ค่าล่าสุดที่ ≤ cutoff คือ 1 มี.ค. (4.30)
# ดังนั้น 4.57 − 4.30 = +27 bps (นิยาม: เทียบค่า ณ/ก่อน ~3 เดือนที่แล้ว)
check("trend_3m_bps", chg == 27.0, str(chg))

try:
    C.fetch_fred_series("EMPTY", fetcher=lambda url: "DATE,Y\n2026-01-01,.\n")
    check("empty_raises", False)
except ValueError:
    check("empty_raises", True)

# ---------------------------------------------------------------- คำนวณ
check("spread_math", C.spread_bps(6.78, 4.57) == 221.0)

r_low = C.risk_components(3.6, 0.0, -10.0)      # มาเลเซียประมาณนี้
r_mid = C.risk_components(7.3, 270.0, 20.0)
r_hi = C.risk_components(31.9, 2739.0, 120.0)   # ตุรกีประมาณนี้
check("risk_monotonic_total", r_low["total"] < r_mid["total"] < r_hi["total"],
      f"{r_low['total']},{r_mid['total']},{r_hi['total']}")
check("risk_tiers", r_low["tier"] == "เสี่ยงต่ำ" and r_hi["tier"] == "เสี่ยงสูง",
      f"{r_low['tier']}/{r_mid['tier']}/{r_hi['tier']}")
check("risk_cap_100", C.risk_components(50.0, 5000.0, 500.0)["total"] == 100.0)
check("risk_floor_0", C.risk_components(1.0, -50.0, None)["total"] == 0.0)
check("trend_unknown_zero", C.risk_components(5.0, 100.0, None)["trend_pts"] == 0.0
      and not C.risk_components(5.0, 100.0, None)["trend_known"])
comp = C.risk_components(6.78, 221.0, 0.0)
check("components_sum", comp["total"] ==
      round(comp["level_pts"] + comp["spread_pts"] + comp["trend_pts"], 1))

# ---------------------------------------------------------------- ความสด
check("stale_daily_fresh", C.staleness_label(TODAY, TODAY, "D") == "สดวันนี้")
lbl = C.staleness_label(date(2026, 2, 28), TODAY, "M")
check("stale_monthly_honest", "ช้า" in lbl and "02/2026" in lbl, lbl)

# ---------------------------------------------------------------- manual
check("manual_ok", C.validate_manual(7.4, "2026-07-20", TODAY)[0])
check("manual_future_rejected", not C.validate_manual(7.4, "2026-08-01", TODAY)[0])
check("manual_range_rejected", not C.validate_manual(75.0, "2026-07-20", TODAY)[0]
      and not C.validate_manual(-1.0, "2026-07-20", TODAY)[0])
check("manual_baddate_rejected", not C.validate_manual(5.0, "20/07/2026", TODAY)[0])

# ---------------------------------------------------------------- build_rows
us_obs = [(date(2026, 4, 1), 4.40), (date(2026, 7, 21), 4.57)]
in_obs = [(date(2025, 11, 1), 6.54), (date(2026, 2, 1), 6.78)]
rows = C.build_rows({"US": us_obs, "IN": in_obs},
                    {"TR": {"y": 31.96, "asof": "2026-07-20"},
                     "BR": {"y": 200.0, "asof": "2026-07-20"}},  # ตั้งใจให้ไม่ผ่าน
                    TODAY, demo=False)
by = {r.spec.code: r for r in rows}
check("rows_all_countries", len(rows) == len(C.COUNTRIES))
check("us_is_base", by["US"].spread == 0.0 and by["US"].source == "FRED")
check("in_spread_221", by["IN"].spread == 221.0, str(by["IN"].spread))
check("in_risk_present", by["IN"].risk.get("tier") in {"เสี่ยงต่ำ", "เสี่ยงปานกลาง", "เสี่ยงสูง"})
check("manual_used_and_labeled", by["TR"].source == "MANUAL"
      and "ยังไม่ได้ตรวจสอบ" in by["TR"].fresh_label)
check("manual_invalid_becomes_error", by["BR"].source == "NONE" and by["BR"].error != "")
check("no_data_no_fake", by["MY"].source == "NONE" and by["MY"].y is None)

rows_demo = C.build_rows({}, {}, TODAY, demo=True)
byd = {r.spec.code: r for r in rows_demo}
check("demo_labeled", all("DEMO" in r.fresh_label for r in rows_demo))
check("demo_covers_all", all(r.y is not None for r in rows_demo))
check("demo_tr_high_risk", byd["TR"].risk["tier"] == "เสี่ยงสูง", str(byd["TR"].risk))
check("demo_my_low_risk", byd["MY"].risk["tier"] == "เสี่ยงต่ำ", str(byd["MY"].risk))

# ---------------------------------------------------------------- source check (จำลอง)
def fake_fetcher(url):
    if "DGS10" in url or "INDIRLTLT01STM" in url:
        return "DATE,Y\n2026-06-01,4.50\n2026-07-01,4.57\n"
    raise OSError("series not found")

rep = dict(C.source_check_report(fetcher=fake_fetcher))
check("srccheck_ok_reported", rep["US"].startswith("OK"), rep["US"])
check("srccheck_fail_reported", rep["ID"].startswith("FAIL"), rep["ID"])
check("srccheck_none_honest", rep["MY"].startswith("ไม่มีแหล่งฟรี"))

# ---------------------------------------------------------------- worldmon
check("wm_default_finance", W.DEFAULT_VARIANT in W.VARIANTS)
check("wm_urls_https", all(u.startswith("https://") for _, u in W.VARIANTS.values()))
check("wm_urls_unique", len({u for _, u in W.VARIANTS.values()}) == len(W.VARIANTS))
check("wm_unknown_falls_back", W.variant_url("nope") == W.VARIANTS[W.DEFAULT_VARIANT][1])
check("wm_snippet_has_iframe_and_link",
      "components.iframe" in W.APP_PY_SNIPPET and "link_button" in W.APP_PY_SNIPPET)
check("wm_caveats_honest", any("ปฏิเสธการฝัง" in c for c in W.CAVEATS)
      and any("validate ไม่ได้" in c for c in W.CAVEATS)
      and any("ไม่นำตัวเลข" in c for c in W.CAVEATS))
check("wm_no_api_dependency", "api.worldmonitor" not in (W.APP_PY_SNIPPET + str(W.VARIANTS)))

print(f"\n== {ok} passed, {fail} failed ==")
raise SystemExit(1 if fail else 0)
