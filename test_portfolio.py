# -*- coding: utf-8 -*-
"""ทดสอบออฟไลน์: portfolio.py — รันด้วย python3 test_portfolio.py"""
from __future__ import annotations

import sys
from datetime import date

import numpy as np
import pandas as pd

import portfolio as PF

PASS = FAIL = 0
FAILED: list[str] = []


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        FAILED.append(name)


# ------------------------------------------------------------ โครงข้อมูล
check("pf_columns_8", len(PF.COLUMNS) == 8)
check("pf_empty_has_cols", list(PF.empty_df().columns) == PF.COLUMNS)
check("pf_max_accounts_5", PF.MAX_ACCOUNTS == 5)
check("pf_cost_matches_v513", abs(PF.COST_SIDE_PCT - 0.3175) < 1e-9)

_raw = pd.DataFrame({
    "บัญชี": [1, 9, "2"], "หุ้น": ["ptt.bk", "AOT", ""],
    "จำนวนหุ้น": [1000, "500", 100], "ราคาต้นทุน": [35.0, 60.0, 1.0],
    "วันที่ซื้อ": ["2025-01-05", "ไม่ใช่วันที่", None],
    "ปันผลต่อหุ้น": [2.0, None, None], "วันที่ XD": ["2026-03-01", None, None],
    "หมายเหตุ": [None, "x", None]})
_d, _pb = PF.normalize(_raw)
check("pf_norm_uppercases_and_strips_bk", _d.iloc[0]["หุ้น"] == "PTT")
check("pf_norm_drops_blank_ticker", len(_d) == 2)
check("pf_norm_reports_drop", any("ไม่มีชื่อหุ้น" in p for p in _pb))
check("pf_norm_clamps_account", int(_d.iloc[1]["บัญชี"]) == 1)
check("pf_norm_reports_clamp", any("บัญชีต้องเป็น 1-5" in p for p in _pb))
check("pf_norm_coerces_numeric", float(_d.iloc[1]["จำนวนหุ้น"]) == 500.0)
check("pf_norm_bad_date_to_nat", pd.isna(_d.iloc[1]["วันที่ซื้อ"]))
check("pf_norm_missing_col_reported",
      any("ไม่มีคอลัมน์" in p for p in
          PF.normalize(pd.DataFrame({"หุ้น": ["A"]}))[1]))
check("pf_norm_empty_ok", len(PF.normalize(pd.DataFrame())[0]) == 0)

_csv = PF.to_csv(_d)
_back, _ = PF.from_csv(_csv)
check("pf_csv_roundtrip_rows", len(_back) == len(_d))
check("pf_csv_roundtrip_ticker", _back.iloc[0]["หุ้น"] == "PTT")
check("pf_csv_roundtrip_date", _back.iloc[0]["วันที่ XD"] == date(2026, 3, 1))
check("pf_csv_roundtrip_no_nat_text", "NaT" not in _csv)
check("pf_csv_bad_input", PF.from_csv(b"\x00\x01")[1] != [])

# ----------------------------------------------------------------- enrich
_px = {"PTT": 40.0, "AOT": 50.0}
_eng = {"PTT": {"Regime": "UP", "บักเก็ต": "🟢 ควรซื้อ (สัญญาณวันนี้)"},
        "AOT": {"Regime": "DOWN"}}
_st = {"PTT": 36.0, "AOT": 55.0}
_en = PF.enrich(_d, _px, _eng, _st)
check("pf_enrich_mv", float(_en.iloc[0]["มูลค่าตลาด"]) == 40000.0)
check("pf_enrich_pl", float(_en.iloc[0]["กำไร/ขาดทุน"]) == 5000.0)
check("pf_enrich_pct", float(_en.iloc[0]["%"]) == round(5000 / 35000 * 100, 2))
check("pf_enrich_weight_sums_100",
      abs(float(_en["น้ำหนัก %"].sum()) - 100.0) < 0.05)
check("pf_enrich_missing_price_is_nan",
      not (PF.enrich(_d, {}, {}, {})["มูลค่าตลาด"].notna().any()))
_s = PF.summary(_en)
check("pf_summary_totals", _s["มูลค่าตลาดรวม"] == 65000.0)
check("pf_summary_pl", _s["กำไร/ขาดทุน"] == round(65000 - 65000.0, 2))
check("pf_summary_empty", PF.summary(pd.DataFrame()) == {})

# --------------------------------- ค้นหา frame (บั๊ก DataFrame truthiness)
_pool = {"PTT.BK": pd.DataFrame({"Close": [1.0, 2.0]}),
         "AOT": pd.DataFrame({"Close": [3.0]}),
         "EMPTY.BK": pd.DataFrame({"Close": []})}
check("pf_pick_with_bk", PF.pick_frame(_pool, "PTT") is not None)
check("pf_pick_without_bk", PF.pick_frame(_pool, "AOT") is not None)
check("pf_pick_input_has_bk", PF.pick_frame(_pool, "ptt.bk") is not None)
check("pf_pick_missing_is_none", PF.pick_frame(_pool, "ZZZ") is None)
check("pf_pick_empty_frame_is_none", PF.pick_frame(_pool, "EMPTY") is None)
check("pf_pick_empty_pool", PF.pick_frame({}, "PTT") is None)
check("pf_pick_blank_ticker", PF.pick_frame(_pool, "") is None)
try:
    _ = bool(PF.pick_frame(_pool, "PTT") is not None)
    _ok_no_valueerror = True
except ValueError:
    _ok_no_valueerror = False
check("pf_pick_never_raises_valueerror", _ok_no_valueerror)

# ------------------------------------------------------------ stop v5.13
_n = 100
_idx = pd.bdate_range("2025-01-01", periods=_n)
_fr = pd.DataFrame({"High": np.full(_n, 100.0), "atr": np.full(_n, 2.0)},
                   index=_idx)
check("pf_chandelier_formula", PF.chandelier_stop(_fr, 22, 3.0) == 94.0)
check("pf_chandelier_short_data",
      not (PF.chandelier_stop(_fr.head(5)) == PF.chandelier_stop(_fr.head(5))))
check("pf_chandelier_zero_atr",
      not (PF.chandelier_stop(pd.DataFrame({"High": np.full(_n, 1.0),
                                            "atr": np.zeros(_n)}, index=_idx))
           == PF.chandelier_stop(pd.DataFrame({"High": np.full(_n, 1.0),
                                               "atr": np.zeros(_n)}, index=_idx))))
check("pf_stop_note_admits_not_from_entry", "ไม่ได้ไล่ประวัติ" in PF.STOP_NOTE)

# ------------------------------------------------------------- ปันผล/XD
_dv = PF.dividend_view(1000, 35.0, 40.0, 2.0, date(2026, 3, 1),
                       today=date(2026, 2, 1))
check("pf_div_gross", _dv["ปันผลรวม (ก่อนภาษี)"] == 2000.0)
check("pf_div_net_after_10pct", _dv["ปันผลสุทธิ"] == 1800.0)
check("pf_div_yield_on_cost", _dv["yield on cost %"] == round(2 / 35 * 100, 2))
check("pf_div_days_to_xd", _dv["วันถึง XD"] == 28)
check("pf_div_breakeven_is_90pct", "90%" in _dv["จุดคุ้มทุน"])
check("pf_div_loss_equals_tax_if_full_drop",
      _dv["ถ้าราคาลงเท่าปันผลพอดี"] == -200.0)
check("pf_div_no_data_when_blank",
      not PF.dividend_view(1000, 35.0, 40.0, 0, None)["มีข้อมูล"])
_dv2 = PF.dividend_view(1000, 35.0, 40.0, 2.0, None, drop_ratio=0.95)
check("pf_div_user_ratio_unfavourable",
      _dv2["สรุปตาม ratio ที่ใส่"] == "ไม่คุ้ม")
_dv3 = PF.dividend_view(1000, 35.0, 40.0, 2.0, None, drop_ratio=0.70)
check("pf_div_user_ratio_favourable",
      "คุ้ม" in _dv3["สรุปตาม ratio ที่ใส่"]
      and "ไม่ใช่ค่าที่วัดจากระบบ" in _dv3["สรุปตาม ratio ที่ใส่"])
_dv4 = PF.dividend_view(1000, 35.0, 40.0, 2.0, None, wht=0.0)
check("pf_div_zero_tax_breakeven_100", "100%" in _dv4["จุดคุ้มทุน"])
check("pf_div_past_xd", PF.dividend_view(1, 1, 1, 1, date(2020, 1, 1),
                                         today=date(2026, 1, 1))["สถานะ XD"]
      == "ผ่าน XD ไปแล้ว")
check("pf_div_note_cites_cost_and_measurement",
      "ต้องวัดเป็นรายตัว" in PF.DIVIDEND_NOTE and "0.32" in PF.DIVIDEND_NOTE)

# ----------------------------------------------------------- การจัดกลุ่ม
_a_stop = PF.action_for(30.0, 35.0, 32.0, "UP", None)
check("pf_action_below_trail", "ต่ำกว่าระดับ trail" in _a_stop["action"])
check("pf_action_does_not_claim_your_stop_hit",
      "ไม่ได้แปลว่าหลุด stop ตามแผนของคุณ" in _a_stop["เหตุผล"])
_a_win = PF.action_for(40.0, 35.0, 42.0, "UP", None)
check("pf_action_profitable_but_below_trail_not_mislabelled",
      "ต่ำกว่าระดับ trail" in _a_win["action"]
      and "ไม่ใช่ค่าจากวันที่คุณซื้อ" in _a_win["ที่มา"])
check("pf_action_regime_down",
      "ไม่สนับสนุน" in PF.action_for(40.0, 35.0, 30.0, "DOWN", None)["action"])
check("pf_action_signal_today",
      "สัญญาณเข้าวันนี้" in PF.action_for(40.0, 35.0, 30.0, "UP",
                                          "🟢 ควรซื้อ (สัญญาณวันนี้)")["action"])
check("pf_action_signal_says_new_leg_not_average",
      "ไม่ใช่การเฉลี่ยขาลง" in PF.action_for(40.0, 35.0, 30.0, "UP",
                                             "🟢 x")["เหตุผล"])
check("pf_action_underwater_no_add",
      "ไม่ใช่เติมของ" in PF.action_for(32.0, 35.0, 30.0, "UP", None)["เหตุผล"])
check("pf_action_hold_says_no_tp",
      "ไม่มีเป้าราคาขายทำกำไร" in PF.action_for(40.0, 35.0, 30.0, "UP",
                                                None)["เหตุผล"])
check("pf_action_no_price", PF.action_for(None, 35.0, 30.0, "UP",
                                          None)["action"] == "ไม่มีราคา")
check("pf_action_every_case_has_source",
      all(PF.action_for(*a)["ที่มา"] for a in
          [(30.0, 35.0, 32.0, "UP", None), (40.0, 35.0, 30.0, "DOWN", None),
           (40.0, 35.0, 30.0, "UP", "🟢 x"), (32.0, 35.0, 30.0, "UP", None),
           (40.0, 35.0, 30.0, "UP", None)]))
check("pf_no_tp_note_explains_backtest",
      "ใช้อ้างอิงไม่ได้" in PF.NO_TP_NOTE)

# ------------------------------------------------------------- หุ้นติด
check("pf_breakeven_pct", PF.breakeven_gain_pct(80.0, 100.0) == 25.0)
check("pf_breakeven_zero_when_flat", PF.breakeven_gain_pct(100.0, 100.0) == 0.0)
_ap = PF.average_down_plan(1000, 100.0, 1000, 80.0, port_value=200000.0)
check("pf_avg_new_cost", _ap["ต้นทุนเฉลี่ยใหม่"] == 90.0)
check("pf_avg_capital", _ap["เงินที่ต้องใส่เพิ่ม"] == 80000.0)
check("pf_avg_fee_uses_v513_cost",
      abs(_ap["ค่าธรรมเนียมซื้อ"] - 80000 * PF.COST_SIDE_PCT / 100) < 0.01)
check("pf_avg_breakeven_drops", _ap["หลังเฉลี่ยต้องขึ้น %"] < _ap["เดิมต้องขึ้น % ถึงเท่าทุน"])
check("pf_avg_unrealised_loss_unchanged_at_purchase",
      _ap["ขาดทุนที่ยังไม่รับรู้ ณ ราคานี้ (หลัง)"]
      == _ap["ขาดทุนที่ยังไม่รับรู้ ณ ราคานี้ (ก่อน)"])
check("pf_avg_downside_sensitivity_doubles",
      _ap["ถ้าลงต่ออีก 10% ขาดทุนเพิ่ม (หลัง)"]
      == 2 * _ap["ถ้าลงต่ออีก 10% ขาดทุนเพิ่ม (ก่อน)"])
check("pf_avg_weight_rises",
      _ap["น้ำหนักในพอร์ต หลัง %"] > _ap["น้ำหนักในพอร์ต ก่อน %"])
check("pf_avg_rejects_bad_input",
      not PF.average_down_plan(0, 100.0, 100, 80.0)["ok"])
check("pf_avg_warning_says_no_better_odds",
      "ไม่ได้ทำให้หุ้นตัวนี้มีโอกาสขึ้นมากกว่าเดิม" in PF.AVERAGE_DOWN_WARNING)
check("pf_avg_warning_corrects_loss_myth",
      "ไม่ได้ลดลงเลย" in PF.AVERAGE_DOWN_WARNING
      and "ความไวต่อการลงต่อ" in PF.AVERAGE_DOWN_WARNING)
check("pf_avg_checklist_5", len(PF.AVERAGE_DOWN_CHECKLIST) == 5)
check("pf_timing_note_refuses_to_time",
      "ไม่มีคำตอบที่เชื่อถือได้" in PF.TIMING_NOTE
      and "McLean" in PF.TIMING_NOTE)
check("pf_disclaimer_honest",
      "ไม่ใช่คำแนะนำการลงทุน" in PF.DISCLAIMER
      and "ระบบไม่มีข้อมูลงบการเงิน" in PF.DISCLAIMER)

print(f"\n{PASS} ผ่าน / {FAIL} ตก  (รวม {PASS + FAIL})")
if FAILED:
    print("ตก:", ", ".join(FAILED))
sys.exit(1 if FAIL else 0)
