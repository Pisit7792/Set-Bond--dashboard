# -*- coding: utf-8 -*-
"""ทดสอบออฟไลน์: gold_council.py
รันด้วย: python3 test_gold_council.py   (ไม่ต้องต่อเน็ต ไม่ต้องมี API key)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import gold_council as GC

PASS = FAIL = 0
FAILED: list[str] = []


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        FAILED.append(name)


# ------------------------------------------------------------- ทะเบียนบท
check("gc_ten_specialists", len(GC.SPECIALISTS) == 10)
check("gc_groups_valid",
      {s["group"] for s in GC.SPECIALISTS} == {"TECHNICAL", "FUNDAMENTAL", "RISK"})
check("gc_no_data_is_news_and_sentiment", set(GC.NO_DATA) == {"news", "sentiment"})
check("gc_every_spec_has_source_or_caveat",
      all(s["source"] and (s["have"] or s["caveat"]) for s in GC.SPECIALISTS))
check("gc_pattern_is_grade_d", GC.spec("pattern")["grade"] == "D")
check("gc_smc_admits_no_orderflow", "order flow" in GC.spec("structure")["caveat"])
check("gc_volume_admits_not_whole_market",
      "spot OTC" in GC.spec("volume")["caveat"])
check("gc_session_computed_not_llm", GC.spec("session").get("llm") is False)

# --------------------------------------------------------------- prompt
_p = GC.build_council_prompt(GC.DEFAULT_PANEL, '{"x":1}')
check("gc_prompt_excludes_session_from_llm", "Session:" not in _p)
check("gc_prompt_marks_no_data", "ไม่มีข้อมูลในระบบ" in _p)
check("gc_prompt_requires_citation", "อ้างชื่อตัวเลขที่ใช้" in _p)
check("gc_iron_rule_council_cannot_enter",
      "ไม่มีอำนาจอนุมัติการเข้าไม้" in GC.IRON_RULES_GOLD)
check("gc_iron_rule_no_news_memory", "ห้ามอ้างข่าวจากความจำ" in GC.IRON_RULES_GOLD)

# ---------------------------------------------------------------- parser
_raw = ('บทวิเคราะห์...\n```json\n{"specialists": {'
        '"trend": {"lean": "BUY", "conf": 70, "อ้างอิง": "regime", "เหตุผล": "ก"},'
        '"news": {"lean": "BUY", "conf": 90, "อ้างอิง": "ข่าว", "เหตุผล": "ข"},'
        '"sentiment": {"lean": "SELL", "conf": 80, "อ้างอิง": "COT", "เหตุผล": "ค"},'
        '"ผิด": {"lean": "BUY", "conf": 50}},'
        '"ข้อขัดแย้ง": ["a"], "ความเสี่ยงหลัก": ["b"]}\n```')
_ana, _pr = GC.parse_council(_raw)
check("gc_parse_ok", _pr is not None and "trend" in _pr["specialists"])
check("gc_parse_drops_unknown_id", "ผิด" not in _pr["specialists"])
check("gc_parse_forces_neutral_for_news",
      _pr["specialists"]["news"]["lean"] == "NEUTRAL")
check("gc_parse_forces_neutral_for_sentiment",
      _pr["specialists"]["sentiment"]["lean"] == "NEUTRAL")
check("gc_parse_keeps_analysis", "บทวิเคราะห์" in _ana)
check("gc_parse_bad_json_none", GC.parse_council("ไม่มี json")[1] is None)
check("gc_parse_empty", GC.parse_council("")[1] is None)
check("gc_lean_norm_defaults_neutral", GC._norm_lean("ไม่รู้") == "NEUTRAL")

# ----------------------------------------------------------------- tally
_t = GC.tally(_pr)
check("gc_tally_counts_are_int",
      all(isinstance(v, int) for v in _t["counts"].values()))
check("gc_tally_neutral_includes_forced", _t["counts"]["NEUTRAL"] == 2)
check("gc_tally_lists_abstain", set(_t["งดออกเสียง_ไม่มีข้อมูล"])
      == {"News/Macro", "Sentiment"})
check("gc_tally_conf_is_range", _t["conf_ต่ำสุด"] == 70 and _t["conf_สูงสุด"] == 90)
check("gc_tally_no_weight_by_default", _t["ถ่วงน้ำหนัก"] is None)
_tw = GC.tally(_pr, weights={"trend": 2.0})
check("gc_tally_weight_optional", _tw["ถ่วงน้ำหนัก"]["BUY"] == 2.0)
check("gc_weight_warning_says_not_probability",
      "ไม่ใช่ความน่าจะเป็น" in GC.WEIGHT_WARNING)

# ------------------------------------------------------------- risk gate
_ok_state = {"regime": "UP", "status": "พร้อม", "triggered": True,
             "plan": {"side": "LONG", "stop_dist": 20.0}}
_g = GC.risk_gate(_ok_state, min_rr=2.0, spread_c=25.0)
check("gc_gate_pass_when_clean", _g["pass"])
check("gc_gate_rr_computed", _g["rr"] == 80.0)
check("gc_gate_veto_on_vol_shock",
      not GC.risk_gate(dict(_ok_state, status="Vol shock — งดเข้าใหม่"))["pass"])
check("gc_gate_veto_on_cost",
      not GC.risk_gate(dict(_ok_state, status="Cost gate — spread แพงเทียบ 1R"))["pass"])
check("gc_gate_veto_on_mixed_regime",
      not GC.risk_gate(dict(_ok_state, regime="MIXED"))["pass"])
check("gc_gate_veto_when_no_signal",
      not GC.risk_gate(dict(_ok_state, triggered=False, plan=None))["pass"])
check("gc_gate_veto_when_stop_too_tight",
      not GC.risk_gate(dict(_ok_state, plan={"side": "LONG", "stop_dist": 0.2}),
                       min_rr=2.0, spread_c=25.0)["pass"])

# --------------------------------------------------------- chief verdict
_all_buy = {"counts": {"BUY": 10, "NEUTRAL": 0, "SELL": 0}}
_v1 = GC.chief_verdict(dict(_ok_state, triggered=False, plan=None),
                       GC.risk_gate(dict(_ok_state, triggered=False, plan=None)),
                       _all_buy)
check("gc_verdict_no_trade_when_rules_incomplete", _v1["verdict"] == "NO TRADE")
check("gc_verdict_council_cannot_override", "เปลี่ยนผลข้อนี้ไม่ได้" in _v1["override"])
_shock = dict(_ok_state, status="Vol shock — งดเข้าใหม่")
check("gc_verdict_no_trade_on_hard_veto",
      GC.chief_verdict(_shock, GC.risk_gate(_shock), _all_buy)["verdict"] == "NO TRADE")
_v3 = GC.chief_verdict(_ok_state, _g, {"counts": {"BUY": 6, "NEUTRAL": 4, "SELL": 0}})
check("gc_verdict_follows_engine_when_clean", _v3["verdict"] == "ตามกติกา · LONG")
check("gc_verdict_full_size_when_no_counter", "เต็มขนาด" in _v3["action"])
_v4 = GC.chief_verdict(_ok_state, _g, {"counts": {"BUY": 5, "NEUTRAL": 3, "SELL": 2}})
check("gc_verdict_reduces_size_on_counter_vote", "ลดขนาด" in _v4["action"])
_v5 = GC.chief_verdict(_ok_state, _g, {"counts": {"BUY": 1, "NEUTRAL": 9, "SELL": 0}})
check("gc_verdict_reduces_size_on_weak_support", "ลดขนาด" in _v5["action"])

# ------------------------------------------------------- ตัวเลขที่คำนวณเอง
check("gc_session_overlap", GC.session_of(pd.Timestamp("2026-07-20 13:00"))["overlap"])
check("gc_session_asia",
      "Asia" in GC.session_of(pd.Timestamp("2026-07-20 02:00"))["ช่วง"])
check("gc_session_london",
      GC.session_of(pd.Timestamp("2026-07-20 09:00"))["ช่วง"] == "London")
check("gc_session_bad_input", GC.session_of("ไม่ใช่เวลา")["ช่วง"] == "ไม่ทราบ")

_n = 200
_idx = pd.bdate_range("2025-01-01", periods=_n)


def _mk(trend: float) -> pd.DataFrame:
    """คลื่นไซน์ + เทรนด์ — มี swing จริงให้ fractal จับได้ (เส้นตรงล้วนไม่มี swing)"""
    t = np.arange(_n, dtype=float)
    c = 100.0 + trend * t + 8.0 * np.sin(t / 8.0)
    return pd.DataFrame({"Open": c, "High": c + 1.0, "Low": c - 1.0,
                         "Close": c, "Volume": np.full(_n, 1000.0)}, index=_idx)


_up = _mk(0.5)
_ss = GC.swing_structure(_up)
check("gc_structure_detects_uptrend",
      _ss.get("พอข้อมูล") and "ขาขึ้น" in _ss.get("โครงสร้าง", ""))
check("gc_structure_detects_downtrend",
      "ขาลง" in GC.swing_structure(_mk(-0.5)).get("โครงสร้าง", ""))
check("gc_structure_flat_is_mixed_or_insufficient",
      GC.swing_structure(_mk(0.0)).get("โครงสร้าง", "ผสม") in
      ("ผสม (ไม่มีโครงสร้างชัด)", "ผสม"))
check("gc_structure_straight_line_has_no_swing",
      not GC.swing_structure(pd.DataFrame(
          {"Open": np.linspace(1, 2, _n), "High": np.linspace(1, 2, _n),
           "Low": np.linspace(1, 2, _n), "Close": np.linspace(1, 2, _n)},
          index=_idx))["พอข้อมูล"])
check("gc_structure_short_data",
      not GC.swing_structure(_up.head(5))["พอข้อมูล"])
check("gc_structure_warns_repaint", "อาจเปลี่ยนได้" in _ss.get("หมายเหตุ", ""))

check("gc_volume_ok", GC.volume_state(_up)["พอข้อมูล"])
check("gc_volume_missing_col",
      not GC.volume_state(_up.drop(columns=["Volume"]))["พอข้อมูล"])
_zero = _up.copy(); _zero["Volume"] = 0.0
check("gc_volume_all_zero", not GC.volume_state(_zero)["พอข้อมูล"])

# --------------------------------------------------------------- context
_ctx = GC.build_context("PAXG-USD", "1d", _up, _ok_state, _g)
check("gc_ctx_declares_missing_data",
      "ข่าวทองรายวัน" in _ctx["ข้อมูลที่ระบบนี้ไม่มี"]
      and "bid_ask_realtime" in _ctx["ข้อมูลที่ระบบนี้ไม่มี"])
check("gc_ctx_has_engine_block", _ctx["engine_v6_4"]["regime"] == "UP")
check("gc_ctx_gate_is_python", _ctx["เกตความเสี่ยง_คำนวณด้วย_python"]["ผ่าน"] is True)

check("gc_disclaimer_honest",
      "ไม่อิสระทางสถิติ" in GC.DISCLAIMER
      and "ไม่ใช่จากมติสภา" in GC.DISCLAIMER
      and "ไม่ใช่คำแนะนำ" in GC.DISCLAIMER)

print(f"\n{PASS} ผ่าน / {FAIL} ตก  (รวม {PASS + FAIL})")
if FAILED:
    print("ตก:", ", ".join(FAILED))
sys.exit(1 if FAIL else 0)
