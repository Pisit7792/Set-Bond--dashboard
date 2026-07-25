# -*- coding: utf-8 -*-
"""ทดสอบออฟไลน์: llm_providers, multi_meeting, quant_evaluation, quant_optimize
รันด้วย: python3 test_multi_selfimprove.py   (ไม่ต้องต่อเน็ต ไม่ต้องมี API key)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import llm_providers as LP
import multi_meeting as MM
import quant_evaluation as QE
import quant_optimize as QO

PASS = FAIL = 0
FAILED: list[str] = []


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        FAILED.append(name)


# ---------------------------------------------------------------- providers
def tr_ok(url, headers, payload, timeout):
    if "generativelanguage" in url:
        return 200, json.dumps(
            {"candidates": [{"content": {"parts": [{"text": "ก"}, {"text": "ข"}]}}]})
    return 200, json.dumps({"choices": [{"message": {"content": "hello"}}]})


check("prov_registry_3", set(LP.ORDER) == {"gemini", "groq", "openrouter"})
for _pv in LP.ORDER:
    _r = LP.chat(_pv, "k", [{"role": "user", "content": "hi"}], transport=tr_ok)
    check(f"prov_ok_{_pv}", _r["ok"] and _r["text"])
check("prov_gemini_concat_parts",
      LP.chat("gemini", "k", [{"role": "user", "content": "x"}],
              transport=tr_ok)["text"] == "กข")
check("prov_no_key", not LP.chat("gemini", "", [{"role": "user", "content": "x"}])["ok"])
check("prov_unknown", "ไม่รู้จัก" in LP.chat("zzz", "k", [{"role": "user", "content": "x"}])["error"])
check("prov_429", "429" in LP.chat("groq", "k", [{"role": "user", "content": "x"}],
                                   transport=lambda *a: (429, "{}"))["error"])
check("prov_404_tells_models_url",
      LP.PROVIDERS["openrouter"]["models_url"] in
      LP.chat("openrouter", "k", [{"role": "user", "content": "x"}],
              transport=lambda *a: (404, "{}"))["error"])
check("prov_network_down",
      "ต่อเซิร์ฟเวอร์ไม่ได้" in LP.chat("groq", "k", [{"role": "user", "content": "x"}],
                                        transport=lambda *a: (0, "boom"))["error"])
check("prov_empty_200",
      not LP.chat("groq", "k", [{"role": "user", "content": "x"}],
                  transport=lambda *a: (200, '{"choices":[{"message":{"content":""}}]}'))["ok"])
check("prov_bad_json",
      "อ่าน JSON" in LP.chat("groq", "k", [{"role": "user", "content": "x"}],
                             transport=lambda *a: (200, "not json"))["error"])
check("prov_role_maps_to_model",
      LP._to_gemini([{"role": "assistant", "content": "a"}])[0]["role"] == "model")
check("prov_honesty_says_not_independent",
      "ยังไม่อิสระจริง" in LP.HONESTY and "ไม่เท่ากับ" in LP.HONESTY)
_many = LP.chat_many([{"provider": p, "api_key": "k"} for p in LP.ORDER],
                     [{"role": "user", "content": "x"}], transport=tr_ok)
check("prov_chat_many_all", len(_many) == 3 and all(r["ok"] for r in _many))
_mix = LP.chat_many([{"provider": "gemini", "api_key": "k"},
                     {"provider": "groq", "api_key": ""}],
                    [{"role": "user", "content": "x"}], transport=tr_ok)
check("prov_one_fail_others_survive", _mix[0]["ok"] and not _mix[1]["ok"])

# ------------------------------------------------------------ multi_meeting
_p = MM.build_solo_prompt(["quant", "contra"], '{"a":1}')
check("mm_prompt_has_iron_rules", "กติกาเหล็ก" in _p)
check("mm_prompt_forces_counterpoint", "ค้าน" in _p and "ห้ามแต่ง" in _p)
check("mm_prompt_single_call_json", "```json" in _p)


def _mk(label, votes, ok=True, err=""):
    return {"label": label, "ok": ok, "error": err,
            "parsed": ({"votes": votes, "ขัดแย้ง": ["x"], "คำสั่ง": [],
                        "conf_รวม": 50} if votes else None)}


_b = MM.collect([
    _mk("Google/g", {"PTT": {"มติ": "ตาม", "conf": 70, "เหตุผล": "a"}}),
    _mk("Groq/l", {"PTT": {"มติ": "ค้าน", "conf": 40, "เหตุผล": "b"}}),
    _mk("OpenRouter/o", None, ok=False, err="429"),
])
check("mm_collect_ok_labels", len(_b["ok_labels"]) == 2)
check("mm_collect_failed_recorded", len(_b["failed"]) == 1)
_rows = MM.agreement_rows(_b)
check("mm_flags_disagreement", _rows[0]["เห็นต่าง"] and "เห็นต่าง" in _rows[0]["สถานะ"])
check("mm_conf_is_range_not_mean", _rows[0]["conf (ต่ำ-สูง)"] == "40-70")
check("mm_no_consensus_score",
      not any("conf_รวม" in r or "คะแนนรวม" in r for r in _rows))
_h = MM.headline(_b)
check("mm_headline_warns_failure", "พัง 1 เจ้า" in _h and "ไม่ใช่เอกฉันท์" in _h)
check("mm_headline_warns_disagreement", "เห็นต่าง" in _h)
_b2 = MM.collect([
    _mk("A/a", {"PTT": {"มติ": "ตาม", "conf": 60, "เหตุผล": "x"}}),
    _mk("B/b", {"PTT": {"มติ": "ตาม", "conf": 62, "เหตุผล": "y"}}),
])
check("mm_unanimous_still_warns", "ทับซ้อน" in MM.headline(_b2))
check("mm_disagreement_list", len(MM.disagreement_list(_b)) == 1
      and MM.disagreement_list(_b2) == [])
check("mm_referee_does_not_vote",
      "ไม่ใช่" in MM.build_referee_prompt(["a"], "{}")
      and "ชี้ขาดไม่ได้" in MM.build_referee_prompt(["a"], "{}"))
check("mm_disclaimer_honest",
      "ไม่อิสระจริง" in MM.DISCLAIMER and "ไม่เพิ่ม" in MM.DISCLAIMER
      and "ไม่ใช่คำแนะนำ" in MM.DISCLAIMER)
check("mm_parse_reuses_sm", MM.parse_solo('```json\n{"votes":{"a":'
                                          '{"มติ":"ตาม","conf":9}}}\n```')[1]
      ["votes"]["A"]["conf"] == 9)
check("mm_parse_bad_returns_none", MM.parse_solo("ไม่มี json")[1] is None)

# --------------------------------------------------------- quant_evaluation
_sp = QE.walk_forward_splits(1500, 5, 0.01)
check("qe_splits_count", len(_sp) == 5)
check("qe_splits_anchored", all(s["train"][0] == 0 for s in _sp))
check("qe_splits_embargo_gap",
      all(s["test"][0] > s["train"][1] for s in _sp))
check("qe_splits_no_overlap",
      all(_sp[i]["test"][1] <= _sp[i + 1]["test"][0] for i in range(len(_sp) - 1)))
check("qe_splits_too_short", QE.walk_forward_splits(80, 5) == [])

_rng = np.random.default_rng(0)
check("qe_pbo_needs_configs", "อย่างน้อย 4" in
      QE.cscv_pbo(_rng.normal(0, 1, (500, 3)))["reason"])
check("qe_pbo_needs_length", "ต่ำกว่าขั้นต่ำ" in
      QE.cscv_pbo(_rng.normal(0, 1, (40, 10)))["reason"])
_null = [QE.cscv_pbo(np.random.default_rng(s).normal(0, .01, (1200, 20)))["pbo"]
         for s in range(12)]
check("qe_pbo_null_centres_on_half", 0.35 < float(np.mean(_null)) < 0.65)
_M = _rng.normal(0, .01, (1200, 20)); _M[:, 3] += 0.004
check("qe_pbo_low_when_real_edge", QE.cscv_pbo(_M)["pbo"] < 0.2)
check("qe_pbo_admits_own_noise", "แกว่ง" in QE.cscv_pbo(_M)["note"])

check("qe_tstat", abs(QE.tstat([0.01] * 50 + [0.011] * 50)) > 3)
check("qe_sharpe_nan_on_flat", not np.isfinite(QE.sharpe([1.0] * 10)))

_good = {"oos_returns": list(np.random.default_rng(1).normal(0.002, 0.008, 900)),
         "n_trials": 50, "n_oos_trades": 400, "net_thb_oos": 90000.0,
         "pbo": 0.10, "regimes": {"ขาขึ้น": 250, "ขาลง/ออกข้าง": 150},
         "cost_stress_net": 30000.0}
_g = QE.gate_verdict(_good)
check("qe_gate_pass_when_all_good", _g["pass"])
check("qe_gate_has_7_checks", len(_g["checks"]) == 7)
_bad = dict(_good, n_trials=500000)
check("qe_gate_more_trials_hurts_dsr",
      QE.gate_verdict(_bad)["dsr"] < _g["dsr"])
check("qe_gate_fails_on_thin_regime",
      not QE.gate_verdict(dict(_good, regimes={"ขาขึ้น": 400, "ขาลง/ออกข้าง": 5}))["pass"])
check("qe_gate_fails_on_high_pbo",
      not QE.gate_verdict(dict(_good, pbo=0.60))["pass"])
check("qe_gate_fails_on_cost_stress",
      not QE.gate_verdict(dict(_good, cost_stress_net=-1.0))["pass"])
check("qe_gate_fails_on_few_trades",
      not QE.gate_verdict(dict(_good, n_oos_trades=20))["pass"])
check("qe_verdict_note_not_oversold",
      "ไม่ใช่" in QE.VERDICT_NOTE and "paper trade" in QE.VERDICT_NOTE)

# ------------------------------------------------------------ quant_optimize
check("qo_space_small", len(QO.SEARCH_SPACE) <= 6)
check("qo_space_size", QO.space_size() > 100)
_led = {"total_trials": 0, "runs": []}
_led = QO.write_ledger(_led, 30, "t1", path="/tmp/_led_test.json")
_led = QO.write_ledger(QO.read_ledger("/tmp/_led_test.json"), 20, "t2",
                       path="/tmp/_led_test.json")
check("qo_ledger_accumulates", _led["total_trials"] == 50)
check("qo_ledger_persists", QO.read_ledger("/tmp/_led_test.json")["total_trials"] == 50)

_idx = pd.bdate_range("2020-01-01", periods=400)
_fr = {"A": pd.DataFrame({"Close": np.linspace(10, 20, 400)}, index=_idx),
       "B": pd.DataFrame({"Close": np.linspace(20, 10, 400)}, index=_idx)}
_reg = QO.proxy_regime(_fr, n=50)
check("qo_regime_labels", set(_reg.dropna().unique()) <= {"ขาขึ้น", "ขาลง/ออกข้าง"})
check("qo_regime_len", len(_reg) == 400)

# ------------------------------------------------ เมนู vs ROUTES (กันบั๊กเดิมซ้ำ)
# รอบก่อนเปลี่ยนชื่อคีย์ใน ROUTES แต่ลืมแก้ ZONES → KeyError ตอนกดหน้านั้น
# เทสต์นี้อ่าน app.py ด้วย AST (ไม่ต้องมี streamlit) แล้วเทียบสองฝั่งให้ตรงกัน
import ast as _ast

_src = open("app.py", encoding="utf-8").read()
_tree = _ast.parse(_src)
_zones = _routes = None
for _n in _tree.body:
    if isinstance(_n, _ast.Assign) and getattr(_n.targets[0], "id", "") == "ZONES":
        _zones = _ast.literal_eval(_n.value)
    if isinstance(_n, _ast.Assign) and getattr(_n.targets[0], "id", "") == "ROUTES":
        _routes = {_k.value for _k in _n.value.keys}
        _fns = {getattr(_v, "id", "") for _v in _n.value.values}
_pages = [p for v in (_zones or {}).values() for p in v]
_defs = {n.name for n in _tree.body if isinstance(n, _ast.FunctionDef)}
check("app_zones_found", bool(_zones) and bool(_routes))
check("app_every_menu_page_has_route", not (set(_pages) - _routes))
check("app_every_route_is_in_menu", not (_routes - set(_pages)))
check("app_no_duplicate_menu_page", len(_pages) == len(set(_pages)))
check("app_every_route_fn_defined", not (_fns - _defs))
check("app_selfimprove_reachable", "🔬 Self-Improve (ผลออฟไลน์)" in _pages)
check("app_meeting_page_name_stable", "AI Meeting หุ้น" in _pages)

print(f"\n{PASS} ผ่าน / {FAIL} ตก  (รวม {PASS + FAIL})")
if FAILED:
    print("ตก:", ", ".join(FAILED))
sys.exit(1 if FAIL else 0)
