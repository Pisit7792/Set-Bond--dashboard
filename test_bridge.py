"""ทดสอบ bridge (Global->SET overlay): python3 test_bridge.py"""
import numpy as np
import pandas as pd

import bridge as B

ok = fail = 0
def check(name, cond, detail=""):
    global ok, fail
    if cond: ok += 1; print(f"PASS  {name} {detail}")
    else: fail += 1; print(f"FAIL  {name} {detail}")

NAN = float("nan")

# 1) overlay rules — ประกาศไว้ต้องทำงานตามนั้นเป๊ะ
r = B.overlay_state(30.0, {"yield_shock": 30, "credit_crisis": 20, "bank_run": 10,
                           "recovery": 80})
check("calm_level0", r["level"] == 0, r["label"])
check("calm_recovery_ignored", r["level"] == 0)  # recovery สูงไม่ใช่ความเสี่ยง flow

r = B.overlay_state(62.0, {"yield_shock": 30})
check("composite60_level1", r["level"] == 1 and "62" in r["reasons"][0])

r = B.overlay_state(40.0, {"credit_crisis": 55.0})
check("model50_level1", r["level"] == 1)

r = B.overlay_state(40.0, {"credit_crisis": 70.0})
check("model65_level2", r["level"] == 2)

r = B.overlay_state(80.0, {})
check("composite75_level2", r["level"] == 2)

r = B.overlay_state(40.0, {"yield_shock": 52.0, "bank_run": 51.0})
check("pair50_level2", r["level"] == 2, str(r["reasons"]))

r = B.overlay_state(NAN, {})
check("nodata_none", r["level"] is None and "ไม่มีข้อมูล" in r["label"])

r = B.overlay_state(NAN, {"credit_crisis": 70.0})
check("nan_composite_model_still_works", r["level"] == 2)

for lvl_scores in [(10.0, {}), (62.0, {}), (80.0, {})]:
    rr = B.overlay_state(*lvl_scores)
    check(f"guidance_{lvl_scores[0]:.0f}", len(rr["guidance"]) > 20
          and len(rr["disclaimer"]) > 50)
r = B.overlay_state(80.0, {})
check("guidance_not_sell_order", "ไม่ใช่คำสั่งขาย" in r["guidance"])
check("disclaimer_not_validated", "validate" in r["disclaimer"])

# 2) ตารางส่งผ่าน — ทุกแถวต้องมีเกรดหลักฐาน
check("g2s_rows", len(B.GLOBAL_TO_SET) >= 5)
check("g2s_grades", all(row.get("เกรดหลักฐาน") for row in B.GLOBAL_TO_SET))

# 3) THB context
idx = pd.bdate_range("2024-01-01", periods=300)
# บาทอ่อน 'เร็ว': นิ่ง 279 วัน แล้วอ่อน ~3.6% ใน 21 วันสุดท้าย (>เกณฑ์ 2%/เดือน)
weak = pd.Series(np.r_[np.full(279, 33.0), np.linspace(33.0, 34.2, 21)], index=idx)
t = B.thb_context(weak)
check("thb_ok", t["ok"] and t["chg_1m"] > 2.0, f"1m={t['chg_1m']:.2f}%")
check("thb_weak_text", "อ่อนเร็ว" in t["text"])
strong = pd.Series(np.r_[np.full(279, 36.0), np.linspace(36.0, 34.9, 21)], index=idx)
t2 = B.thb_context(strong)
check("thb_strong_text", "แข็งเร็ว" in t2["text"])
mild = pd.Series(np.linspace(33, 36.5, 300), index=idx)  # ~0.7%/เดือน = ปกติ
check("thb_mild_normal", "ปกติ" in B.thb_context(mild)["text"])
check("thb_short_notok", B.thb_context(pd.Series([35.0]*10))["ok"] is False)

# 4) Thai scenarios + glossary
check("set_scen_rows", len(B.SET_SCENARIOS) >= 4)
check("set_scen_grades", all(r.get("เกรดหลักฐาน") for r in B.SET_SCENARIOS))
check("set_glossary_wrn", all(all(d.get(f) for f in ("what","read","not"))
                              for d in B.SET_GLOSSARY.values()))
check("set_glossary_size", len(B.SET_GLOSSARY) >= 6)

print(f"\n== {ok} passed, {fail} failed ==")
raise SystemExit(1 if fail else 0)
