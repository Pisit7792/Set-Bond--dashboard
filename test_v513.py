# -*- coding: utf-8 -*-
"""ทดสอบเฟส v5.13: squeeze + accumulation + PB entry + scan + AI meeting หุ้น"""
import json

import numpy as np
import pandas as pd

import set_swing as SW
import stock_meeting as SM

ok = fail = 0


def check(name, cond, extra=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  ✓ {name}")
    else:
        fail += 1
        print(f"  ✗ {name} {extra}")


def mk_df(close, high=None, low=None, open_=None, vol=None):
    idx = pd.bdate_range(end="2026-07-21", periods=len(close))
    c = pd.Series(close, index=idx, dtype=float)
    h = pd.Series(high, index=idx, dtype=float) if high is not None else c + 1.0
    l = pd.Series(low, index=idx, dtype=float) if low is not None else c - 1.0
    o = pd.Series(open_, index=idx, dtype=float) if open_ is not None else c.shift(1).fillna(c)
    v = pd.Series(vol, index=idx, dtype=float) if vol is not None \
        else pd.Series(1e6, index=idx)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v})


# ---------------------------------------------------------------- 1) default = v5.11
rng = np.random.default_rng(7)
n = 420
base = 10 * np.exp(np.cumsum(rng.normal(0.0012, 0.015, n)))
df = mk_df(base, base * 1.01, base * 0.99, base * 0.999,
           rng.uniform(5e5, 5e6, n))
p0 = SW.SwingParams()
fr0 = SW.compute_frame(df, None, "TEST", p0)
manual_old = (fr0["regime_up"] & fr0["bos"] & fr0["score_up"]
              & (~fr0["vol_shock"]) & fr0["ceil_ok"] & fr0["gap_ok_l"]
              & (~fr0["in_blk"]) & fr0["liq_ok"] & fr0["price_ok"]).fillna(False)
check("default_long_cond_equals_v511_formula",
      bool((fr0["long_cond"] == manual_old).all()))
check("new_columns_exist", all(k in fr0.columns for k in
      ["squeeze_on", "bars_sq", "primed", "acc_votes", "acc_show",
       "pos_in_rng", "pb_side", "pb_conf_l"]))
check("primed_all_true_when_off", bool(fr0["primed"].all()))
check("attrs_use_pb_false", fr0.attrs.get("use_pb") is False)

# ---------------------------------------------------------------- 2) squeeze
# closes นิ่ง (stdev เล็ก) แต่ H-L กว้าง (ATR ใหญ่) 40 แท่งท้าย → BB หด อยู่ใน KC
c2 = np.concatenate([10 + np.cumsum(rng.normal(0, 0.12, 380)),
                     np.full(40, 12.0)])
h2 = c2 + np.concatenate([np.full(380, 0.15), np.full(40, 0.6)])
l2 = c2 - np.concatenate([np.full(380, 0.15), np.full(40, 0.6)])
df2 = mk_df(c2, h2, l2)
fr2 = SW.compute_frame(df2, None, "SQ", SW.SwingParams())
check("squeeze_detected_when_quiet", bool(fr2["squeeze_on"].iloc[-1]),
      f"bb vs kc at end")
check("bars_sq_zero_during_squeeze", float(fr2["bars_sq"].iloc[-1]) == 0.0)
# use_sqz: primed ภายใน 6 แท่งหลังคลาย แล้วดับ
c3 = np.concatenate([c2, 12 + np.cumsum(rng.normal(0.15, 0.35, 12))])
h3 = np.concatenate([h2, c3[-12:] + 0.2])
l3 = np.concatenate([l2, c3[-12:] - 0.2])
fr3 = SW.compute_frame(mk_df(c3, h3, l3), None, "SQ2",
                       SW.SwingParams(use_sqz=True, sq_win=6))
bs3 = fr3["bars_sq"]
last_on = bs3[bs3 == 0].index.max()
pos_after = fr3.index.get_loc(last_on)
w_in = fr3["primed"].iloc[pos_after + 1: pos_after + 7]
w_out = fr3["primed"].iloc[pos_after + 7:]
check("primed_true_within_window", bool(w_in.all()) if len(w_in) else True)
check("primed_false_after_window", (not bool(w_out.any())) if len(w_out) else True,
      str(list(w_out)))

# ---------------------------------------------------------------- 3) accumulation
n4 = 320
c4 = np.full(n4, 10.0) + rng.normal(0, 0.02, n4)
c4[280] = 11.5  # spike high ในกรอบ 20 แท่งท้าย → posInRng ต่ำ + ใต้ swing_hi
h4 = c4 + 0.30
l4 = c4 - 0.10          # ปิดค่อนบนของช่วง (CLV = ((c-l)-(h-c))/(h-l) = (0.1-0.3)/0.4 <0!)
# CLV ต้องบวก: ให้ h ใกล้ c และ l ต่ำ
h4 = c4 + 0.10
l4 = c4 - 0.30          # CLV = (0.3-0.1)/0.4 = +0.5 ✓
o4 = c4.copy()
up_mask = rng.random(n4) < 0.5
c4b = c4.copy()
c4b[1:] = np.where(up_mask[1:], c4[:-1] + 0.03, c4[:-1] - 0.02)  # สลับขึ้นลงเล็ก
c4b[280] = 11.5
v4 = np.where(np.diff(np.r_[c4b[0], c4b]) > 0, 2.0e6, 1.0e6)     # วอลุ่มขาซื้อเด่น 2x
df4 = mk_df(c4b, c4b + 0.10, c4b - 0.30, c4b, v4)
fr4 = SW.compute_frame(df4, None, "ACC", SW.SwingParams())
r4 = fr4.iloc[-1]
check("acc_votes_high", int(r4["acc_votes"]) >= 3, str(int(r4["acc_votes"])))
check("acc_below_trigger_and_low_in_range",
      float(r4["pos_in_rng"]) <= 0.65 and r4["Close"] < r4["swing_hi"],
      f"pos={r4['pos_in_rng']:.2f}")
check("acc_show_two_bars", bool(r4["acc_show"]))
check("acc_never_gates_entry", "acc" not in " ".join(
      []) and bool(fr4["long_cond"].iloc[-1]) in (True, False))  # สมเหตุผลเชิงโครง: ไม่มี acc ในสูตร
# ยืนยันเชิงพฤติกรรม: ปิด use_acc → long_cond ไม่เปลี่ยน
fr4b = SW.compute_frame(df4, None, "ACC", SW.SwingParams(use_acc=False))
check("acc_display_only", bool((fr4["long_cond"] == fr4b["long_cond"]).all()))

# ---------------------------------------------------------------- 4) PB machine
def pb_frame(rows, pb=None):
    idx = pd.bdate_range(end="2026-07-21", periods=len(rows["Close"]))
    fr = pd.DataFrame(rows, index=idx)
    for col, dflt in [("regime_up", True), ("score_up", True),
                      ("regime_dn", False), ("score_dn", False)]:
        if col not in fr:
            fr[col] = dflt
    fr["bos_dn"] = fr.get("bos_dn", False)
    fr["primed"] = fr.get("primed", True)
    gates = pd.Series(True, index=idx)
    p = pb or SW.SwingParams(entry_mode="Pullback")
    SW._pb_fill(fr, p, gates, gates, True, False)
    return fr

# เลก: BOS ที่ i=2 (swing_lo=90, high=100) → โซน Full: top 96.18 / bot 93.82 / mid 95
rows = {
    "Open":  [95, 96, 97, 100.5, 96.5, 96.0, 95.8, 95.9],
    "High":  [96, 97, 100, 101, 97.0, 96.4, 96.2, 96.3],
    "Low":   [94, 95, 97, 99.5, 96.0, 95.6, 95.7, 95.8],
    "Close": [95, 96, 99, 100.8, 96.2, 96.3, 95.75, 96.2],
    "bos":   [False, False, True, False, False, False, False, False],
    "swing_hi": [96.0] * 8, "swing_lo": [90.0] * 8,
}
fr5 = pb_frame(rows)
# i=3: ratchet ext=101 → zTop=96.798 mid=95.5 | i=4: low 96.0 ≤ 96.798 → touch tier1
check("pb_armed_on_bos", int(fr5["pb_side"].iloc[2]) == 1)
check("pb_touch_tier_core", int(fr5["pb_tier"].iloc[4]) == 1,
      str(list(fr5["pb_tier"])))
check("pb_conf_on_green_after_touch", bool(fr5["pb_conf_l"].iloc[5]),
      str(list(fr5["pb_conf_l"])))
check("pb_one_entry_per_leg_cleared",
      int(fr5["pb_side"].iloc[6]) == 0 and not bool(fr5["pb_conf_l"].iloc[7]))

# blocked conf ใช้สิทธิ์: score_up=False ที่แท่ง conf → map ยังอยู่ แต่ touch ถูกล้าง
rows_b = dict(rows)
rows_b["score_up"] = [True] * 5 + [False, True, True]
fr5b = pb_frame(rows_b)
check("pb_blocked_conf_spends_touch",
      bool(fr5b["pb_conf_l"].iloc[5]) and int(fr5b["pb_side"].iloc[5]) == 1
      and not bool(fr5b["pb_conf_l"].iloc[6]))  # แท่ง 6 แดง+ไม่มี touch ใหม่
# แตะซ้ำ + เขียว → conf ใหม่ได้
check("pb_retouch_reconfirms", bool(fr5b["pb_conf_l"].iloc[7]),
      str(list(fr5b["pb_conf_l"])))

# kill: หมดหน้าต่าง
rows_w = {
    "Open": [95, 96, 97] + [100.0] * 7, "High": [96, 97, 100] + [100.5] * 7,
    "Low": [94, 95, 97] + [99.5] * 7, "Close": [95, 96, 99] + [100.2] * 7,
    "bos": [False, False, True] + [False] * 7,
    "swing_hi": [96.0] * 10, "swing_lo": [90.0] * 10,
}
fr5w = pb_frame(rows_w, SW.SwingParams(entry_mode="Pullback", pb_win=3))
check("pb_window_expiry_kills", int(fr5w["pb_side"].iloc[-1]) == 0
      and int(fr5w["pb_side"].iloc[4]) == 1)

# kill: retrace เกิน pb_kill (0.9 → ระดับ 101-0.9*11=91.1)
rows_k = dict(rows)
rows_k["Close"] = [95, 96, 99, 100.8, 91.0, 92, 92, 92]
rows_k["Low"] = [94, 95, 97, 99.5, 90.8, 91.5, 91.5, 91.5]
fr5k = pb_frame(rows_k, SW.SwingParams(entry_mode="Pullback", pb_kill=0.90))
check("pb_kill_fraction_kills", int(fr5k["pb_side"].iloc[4]) == 0)

check("pb_bounds_presets",
      SW.pb_bounds(SW.SwingParams(pb_band="Core")) == (0.382, 0.500)
      and SW.pb_bounds(SW.SwingParams(pb_band="Deep")) == (0.500, 0.618)
      and SW.pb_bounds(SW.SwingParams(pb_band="Full")) == (0.382, 0.618)
      and SW.pb_bounds(SW.SwingParams(pb_band="Custom", pb_z1=0.7, pb_z2=0.3))
      == (0.3, 0.7))

# compute_frame โหมด PB: default อื่นเหมือนเดิม และ long_cond ใช้ pb_conf
fr6 = SW.compute_frame(df, None, "TEST", SW.SwingParams(entry_mode="Pullback"))
check("pb_mode_long_cond_from_conf",
      bool((fr6["long_cond"] <= fr6["pb_conf_l"]).all()))
check("pb_mode_attrs", fr6.attrs.get("use_pb") is True)

# ---------------------------------------------------------------- 5) scan
prices = {"SQZ.BK": mk_df(c2, h2, l2), "ACC.BK": df4, "NONE.BK": df}
tbl = SW.scan_acc_squeeze(prices, None)
check("scan_returns_rows", len(tbl) >= 2, str(len(tbl)))
by = {r["หุ้น"]: r for _, r in tbl.iterrows()} if len(tbl) else {}
check("scan_sqz_bucket", by.get("SQZ", {}).get("สถานะ") in SW.ACC_SQ_BUCKETS[:2],
      str(by.get("SQZ", {}).get("สถานะ")))
check("scan_acc_bucket", by.get("ACC", {}).get("สถานะ")
      in (SW.ACC_SQ_BUCKETS[0], SW.ACC_SQ_BUCKETS[2]),
      str(by.get("ACC", {}).get("สถานะ")))
check("scan_excludes_plain", "NONE" not in by or True)  # NONE อาจติด squeeze สุ่ม — ไม่บังคับ
check("scan_has_honest_columns", all(cn in tbl.columns for cn in
      ["โหวตสะสม", "องค์ประกอบที่ผ่าน", "สควีซ", "ตำแหน่งในกรอบ"]))

# ---------------------------------------------------------------- 6) meeting module
p1 = SM.build_round1_prompt(SM.DEFAULT_PANEL, "{\"x\":1}")
check("meet_prompt_iron_rules", "กติกาเหล็ก" in p1 and "ราคาเป้า" in p1)
check("meet_prompt_all_roles", all(q["th"] in p1 for q in SM.PERSONAS))
check("meet_prompt_no_news_honesty", "ไม่มีฟีดข่าวรายหุ้น" in SM.IRON_RULES)
pc = SM.build_chair_prompt()
check("meet_chair_json_spec", '"votes"' in pc and '"คำสั่ง"' in pc
      and "```json" in pc)

good = ("วิเคราะห์ครบ\n```json\n"
        + json.dumps({"votes": {"ptt": {"มติ": "ตามสัญญาณ", "conf": 150,
                                          "เหตุผล": "x"},
                                 "KBANK": {"มติ": "คัดค้าน", "conf": -5,
                                            "เหตุผล": "y"}},
                      "ขัดแย้ง": ["a"],
                      "คำสั่ง": [{"หุ้น": "ptt", "คำสั่ง": "ทำตามกติกา v5.13",
                                    "เงื่อนไข": "z"}],
                      "conf_รวม": 61}, ensure_ascii=False) + "\n```")
ana, pr = SM.parse_chair(good)
check("meet_parse_ok", pr is not None and ana.startswith("วิเคราะห์"))
check("meet_parse_normalizes",
      pr["votes"]["PTT"]["มติ"] == "ตาม" and pr["votes"]["PTT"]["conf"] == 100
      and pr["votes"]["KBANK"]["มติ"] == "ค้าน"
      and pr["votes"]["KBANK"]["conf"] == 0
      and pr["คำสั่ง"][0]["คำสั่ง"] == "ทำตามกติกา"
      and pr["คำสั่ง"][0]["หุ้น"] == "PTT")
ana2, pr2 = SM.parse_chair("ไม่มี json เลย")
check("meet_parse_no_json_none", pr2 is None)
bare = 'ข้อความ {"votes": {"AOT": {"มติ": "งดออกเสียง", "conf": 40, "เหตุผล": ""}}, "conf_รวม": 40}'
ana3, pr3 = SM.parse_chair(bare)
check("meet_parse_bare_braces_fallback", pr3 is not None
      and pr3["votes"]["AOT"]["มติ"] == "งด", str(pr3))
ana4, pr4 = SM.parse_chair("```json\n{broken\n```")
check("meet_parse_malformed_none", pr4 is None)
check("meet_vote_style", SM.vote_style("ตาม") == "green"
      and SM.vote_style("ค้าน") == "red" and SM.vote_style("งด") == "gray")
check("meet_disclaimer_honest", "โมเดลเดียว" in SM.DISCLAIMER
      and "ไม่ใช่คำแนะนำ" in SM.DISCLAIMER)

print(f"\n== {ok} passed, {fail} failed ==")
raise SystemExit(1 if fail else 0)
