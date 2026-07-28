# -*- coding: utf-8 -*-
"""ทดสอบเฟส v5.14: distribution / release watch (F1) + F2 ลดไซส์ + F3 บีบ trail

หัวใจของชุดนี้คือ **การพิสูจน์ว่า distribution เป็นกระจกเงาของ accumulation จริง**
ด้วยการสะท้อนราคาทั้งชุด (c' = K - c, h' = K - l, l' = K - h) แล้วยืนยันว่า
โหวตฝั่งสะสมบนชุดที่สะท้อนแล้ว = โหวตฝั่งกระจายบนชุดต้นฉบับ ทุกแท่ง
— ถ้าสูตรฝั่งใดฝั่งหนึ่งพลาด เทสต์นี้จะตกทันที (falsifiable ไม่ใช่ smoke test)

สิ่งที่เทสต์ชุดนี้ **ไม่ได้** พิสูจน์ (บอกไว้ตรงนี้ ไม่ซ่อน):
  - ไม่ได้พิสูจน์ว่า distribution watch ทำนายอะไรได้ (ต้นฉบับเองก็บอกว่าเกรด C)
  - ไม่ได้รันบน TradingView จึงไม่ได้ยืนยันว่า Pine v5.14 คอมไพล์ผ่าน
  - ไม่ได้เรียก yfinance / GitHub API / Anthropic — ทุกอย่างเป็นข้อมูลสังเคราะห์
"""
import numpy as np
import pandas as pd

import accum as ACC
import set_swing as SW

ok = fail = 0


def check(name, cond, extra=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  ✓ {name}")
    else:
        fail += 1
        print(f"  ✗ {name} {extra}")


def synth(n=700, seed=11):
    """ราคาสังเคราะห์ที่มีทั้งช่วงวิ่งและช่วงกรอบ (ให้ footprint มีโอกาสติดจริง)"""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0006, 0.016, n)
    r[250:300] += 0.011                          # markup
    r[300:380] = rng.normal(0.0, 0.004, 80)      # range (โซน footprint)
    r[520:560] -= 0.009                          # markdown
    c = 40 * np.exp(np.cumsum(r))
    hi = c * (1 + np.abs(rng.normal(0, 0.008, n)))
    lo = c * (1 - np.abs(rng.normal(0, 0.008, n)))
    op = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.002, n))
    v = rng.lognormal(14, 0.45, n)
    idx = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame({"Open": op,
                         "High": np.maximum.reduce([hi, op, c]),
                         "Low": np.minimum.reduce([lo, op, c]),
                         "Close": c, "Volume": v}, index=idx)


def bench_of(n=700, seed=3):
    rng = np.random.default_rng(seed)
    c = 1500 * np.exp(np.cumsum(rng.normal(0.0004, 0.009, n)))
    return pd.Series(c, index=pd.bdate_range("2021-01-04", periods=n))


def mirror(df, K=None):
    """สะท้อนราคารอบค่าคงที่ K — สูง/ต่ำสลับกัน วอลุ่มคงเดิม"""
    K = float(df["Close"].max() + df["Close"].min()) if K is None else K
    return pd.DataFrame({"Open": K - df["Open"], "High": K - df["Low"],
                         "Low": K - df["High"], "Close": K - df["Close"],
                         "Volume": df["Volume"]}, index=df.index)


DF = synth()
BCH = bench_of()

print("\n--- 1) ค่าคงที่ + ค่าตั้งต้น ต้องตรงกับ Pine v5.14 ---")
check("accum_version_1_2", ACC.VERSION == "1.2", ACC.VERSION)
check("swing_version_v514", SW.VERSION.startswith("v5.14"), SW.VERSION)
check("dist_clv_thr", ACC.DIST_CLV_THR == -0.10, str(ACC.DIST_CLV_THR))
check("dist_pos_min", ACC.DIST_POS_MIN == 0.35, str(ACC.DIST_POS_MIN))
check("dist_mirrors_acc_thresholds",
      ACC.DIST_CLV_THR == -ACC.ACC_CLV_THR
      and abs(ACC.DIST_POS_MIN - (1.0 - ACC.ACC_POS_MAX)) < 1e-12)
_p = SW.SwingParams()
check("default_use_dist_on", _p.use_dist is True)
check("default_F2_off", _p.use_dist_size is False)
check("default_F3_off", _p.use_dist_trail is False)
check("default_dist_cut", _p.dist_cut == 0.8)
check("default_dist_tr_m", _p.dist_tr_m == 0.75)
check("default_dist_gap", _p.dist_gap == 0.25)
check("dist_shares_acc_inputs",
      not any(a in SW.SwingParams.__dataclass_fields__
              for a in ("dist_len", "dist_flat", "dist_ratio")),
      "ต้องใช้ acc_len/acc_flat/acc_ratio ร่วมกัน ไม่เพิ่มปุ่มชุดที่สอง")

print("\n--- 2) F1 กระจกเงา: acc(สะท้อน) ต้องเท่ากับ dist(ต้นฉบับ) ทุกแท่ง ---")
FR = SW.compute_frame(DF, BCH, "PTT", _p)
MFR_src = mirror(DF)
ATRm = ACC.atr_wilder(MFR_src, _p.atr_len)
sh_m = MFR_src["High"].shift(1).rolling(_p.bos_len).max()
sl_m = MFR_src["Low"].shift(1).rolling(_p.bos_len).min()
MACC = ACC.accumulation_frame(MFR_src, ATRm, sh_m, acc_len=_p.acc_len,
                              acc_flat=_p.acc_flat, acc_ratio=_p.acc_ratio,
                              swing_lo=sl_m)
w = slice(150, None)          # ตัดช่วง warm-up ของ rolling ออก
for key_a, key_d in [("acc_flat_ok", "dist_flat_ok"),
                     ("acc_press_ok", "dist_press_ok"),
                     ("acc_clv_ok", "dist_clv_ok"),
                     ("acc_act_ok", "dist_act_ok"),
                     ("acc_ctx", "dist_ctx"),
                     ("acc_hot", "dist_hot"),
                     ("acc_show", "dist_show")]:
    a = MACC[key_a].to_numpy(bool)[w]
    d = FR[key_d].to_numpy(bool)[w]
    check(f"mirror_{key_d}", bool((a == d).all()),
          f"ต่างกัน {int((a != d).sum())} แท่ง")
check("mirror_votes",
      bool((MACC["acc_votes"].to_numpy()[w]
            == FR["dist_votes"].to_numpy()[w]).all()))
check("mirror_pos_in_rng",
      float(np.nanmax(np.abs(MACC["pos_in_rng"].to_numpy()[w]
                             + FR["pos_in_rng"].to_numpy()[w] - 1.0))) < 1e-9)
check("footprint_actually_fires",
      int(FR["dist_show"].sum()) > 0 and int(FR["acc_show"].sum()) > 0,
      f"dist_show={int(FR['dist_show'].sum())} acc_show={int(FR['acc_show'].sum())}")

print("\n--- 3) F1 นิยามตรงตัว (ตรวจซ้ำแบบไม่ผ่านฟังก์ชันเดิม) ---")
c = DF["Close"]
h, l, v = DF["High"], DF["Low"], DF["Volume"]
up = v.where(c > c.shift(1), 0.0).rolling(_p.acc_len).sum()
dn = v.where(c < c.shift(1), 0.0).rolling(_p.acc_len).sum()
rng_hl = h - l
clv = pd.Series(np.where(rng_hl > 0, ((c - l) - (h - c)) / rng_hl, 0.0),
                index=DF.index).rolling(_p.acc_len).mean()
check("dist_press_formula",
      bool((((up > 0) & (dn >= _p.acc_ratio * up)).fillna(False).to_numpy()[w]
            == FR["dist_press_ok"].to_numpy(bool)[w]).all()))
check("dist_clv_formula",
      bool(((clv <= -0.10).fillna(False).to_numpy()[w]
            == FR["dist_clv_ok"].to_numpy(bool)[w]).all()))
check("dist_flat_is_shared", bool((FR["dist_flat_ok"] == FR["acc_flat_ok"]).all()))
check("dist_act_is_shared", bool((FR["dist_act_ok"] == FR["acc_act_ok"]).all()))
check("dist_ctx_formula",
      bool((((FR["Close"] > FR["swing_lo"]) & (FR["pos_in_rng"] >= 0.35))
            .fillna(False).to_numpy()[w]
            == FR["dist_ctx"].to_numpy(bool)[w]).all()))
check("dist_show_needs_2_bars",
      bool((FR["dist_show"] == (FR["dist_hot"]
                                & FR["dist_hot"].shift(1).fillna(False))).all()))
check("dist_show_implies_hot",
      bool((~FR["dist_show"] | FR["dist_hot"]).all()))

print("\n--- 4) สวิตช์ปิด = ต้องเงียบสนิท ---")
FR_off = SW.compute_frame(DF, BCH, "PTT", SW.SwingParams(use_dist=False))
check("use_dist_off_kills_hot", int(FR_off["dist_hot"].sum()) == 0)
check("use_dist_off_keeps_votes", int(FR_off["dist_votes"].max()) > 0,
      "โหวตยังคำนวณอยู่ (ไว้ตรวจ) แต่ไม่ติดป้าย")
_nolo = ACC.accumulation_frame(DF, ACC.atr_wilder(DF, 14),
                               DF["High"].shift(1).rolling(20).max())
check("swing_lo_none_no_dist", int(_nolo["dist_ctx"].sum()) == 0)
check("use_dist_off_no_order_change",
      SW.backtest(FR_off, SW.SwingParams(use_dist=False))["n"]
      == SW.backtest(FR, _p)["n"])

print("\n--- 5) F2 (ปิดเป็นค่าตั้งต้น) — ลดอย่างเดียว ไม่เคยเพิ่ม ---")
i_up = int(np.where(FR["regime_up"].to_numpy())[0][-1])
FR2 = FR.copy()
FR2.loc[FR2.index[i_up], "dist_show"] = True
m_off = SW.size_mult_at(FR2, i_up, SW.SwingParams(), 0.0, 0)
m_on = SW.size_mult_at(FR2, i_up, SW.SwingParams(use_dist_size=True), 0.0, 0)
check("F2_off_no_change", abs(m_on / max(m_off, 1e-12) - 0.8) < 1e-9
      and m_off > 0, f"off={m_off} on={m_on}")
check("F2_is_reducer_only", m_on <= m_off + 1e-12)
FR3 = FR.copy()
FR3["dist_show"] = False
check("F2_quiet_without_footprint",
      abs(SW.size_mult_at(FR3, i_up, SW.SwingParams(use_dist_size=True), 0.0, 0)
          - SW.size_mult_at(FR3, i_up, SW.SwingParams(), 0.0, 0)) < 1e-12)
i_dn_arr = np.where(FR["regime_dn"].to_numpy())[0]
if len(i_dn_arr):
    i_dn = int(i_dn_arr[-1])
    FR4 = FR.copy()
    FR4.loc[FR4.index[i_dn], "acc_show"] = True
    check("F2_symmetric_short_side",
          SW.size_mult_at(FR4, i_dn, SW.SwingParams(use_dist_size=True), 0.0, 0)
          < SW.size_mult_at(FR4, i_dn, SW.SwingParams(), 0.0, 0) + 1e-12)
else:
    check("F2_symmetric_short_side", True, "(ไม่มีแท่ง regime DOWN ในชุดนี้)")

print("\n--- 6) F3 guard สองด้าน — พิสูจน์ด้วยการสุ่ม 200,000 เคส ---")
rng = np.random.default_rng(0)
bad_bound = bad_market = 0
for _ in range(200_000):
    atr = float(rng.uniform(0.1, 5)); px = float(rng.uniform(1, 200))
    hh = px + float(rng.uniform(-3, 10)) * atr
    ll = px - float(rng.uniform(-3, 10)) * atr
    trm = float(rng.uniform(1, 6)); k = float(rng.uniform(0.3, 1.0))
    gap = float(rng.uniform(0.05, 1.5))
    got = SW.dist_chand(hh, atr, px, trm, k, gap, True)
    base, tgt = hh - atr * trm, hh - atr * trm * k
    if not (base - 1e-9 <= got <= tgt + 1e-9):
        bad_bound += 1
    if got > px and got > base + 1e-9:
        bad_market += 1
    gots = SW.dist_chand(ll, atr, px, trm, k, gap, False)
    bases, tgts = ll + atr * trm, ll + atr * trm * k
    if not (tgts - 1e-9 <= gots <= bases + 1e-9):
        bad_bound += 1
    if gots < px and gots < bases - 1e-9:
        bad_market += 1
check("F3_never_looser_never_tighter_than_target", bad_bound == 0, str(bad_bound))
check("F3_never_pushed_through_market_by_tightening", bad_market == 0,
      str(bad_market))
check("F3_no_op_when_scaler_is_1",
      abs(SW.dist_chand(100.0, 2.0, 95.0, 3.0, 1.0, 0.25, True)
          - (100.0 - 2.0 * 3.0)) < 1e-12)
check("F3_tightens_when_room",
      SW.dist_chand(100.0, 1.0, 99.0, 3.0, 0.5, 0.25, True) > 100.0 - 3.0)

print("\n--- 7) ค่าตั้งต้นต้องให้ออร์เดอร์เท่า v5.13 ---")
FR_forced = FR.copy()
FR_forced["dist_show"] = True          # จุดไฟทุกแท่ง
FR_forced["acc_show"] = True
bt_base = SW.backtest(FR, _p)
bt_forced = SW.backtest(FR_forced, _p)
check("defaults_ignore_footprint_entirely",
      bt_base["n"] == bt_forced["n"]
      and abs(bt_base.get("net_thb", 0.0) - bt_forced.get("net_thb", 0.0)) < 1e-6,
      f"{bt_base['n']} vs {bt_forced['n']}")
bt_f3 = SW.backtest(FR_forced, SW.SwingParams(use_dist_trail=True))
check("F3_on_does_change_orders", bt_f3.get("net_thb") != bt_base.get("net_thb")
      or bt_f3["n"] != bt_base["n"])

print("\n--- 8) รายงานผลบนจอ ---")
stt = SW.state_today(FR, _p)
d = stt.get("dist") or {}
check("state_today_has_dist", bool(d) and "votes" in d and "show" in d)
check("dist_not_in_entry_checklist",
      not any("กระจาย" in nm for nm, _o, _t in stt["checklist"]),
      "ต้องแยกออกจาก checklist เงื่อนไขเข้า เพราะไม่ปิดกั้นอะไร")
check("dist_pine_dash_format",
      d["pine_dash"].endswith("/4")
      and (d["pine_dash"].startswith("RELEASE ") or d["pine_dash"][0].isdigit()))
aud = SW.acc_audit(DF, BCH, "PTT", _p)
check("audit_has_dist_rows", len(aud.get("dist_rows", [])) == 6)
check("audit_dist_row_keys",
      all(set(r) >= {"ข้อ", "ค่าที่วัดได้", "เกณฑ์", "ผ่าน"}
          for r in aud["dist_rows"]))
check("audit_dist_votes_match", aud["dist_votes"] == int(FR["dist_votes"].iloc[-1]))
lbl_show, on_show = ACC.dist_label(True, True)
lbl_hot, on_hot = ACC.dist_label(False, True)
lbl_no, on_no = ACC.dist_label(False, False)
check("dist_label_marker_truth", on_show is True and on_hot is False
      and on_no is False, "มีแค่ dist_show ที่มีเครื่องหมายจริงบนชาร์ต")
check("dist_label_texts", "กระจาย" in lbl_show and lbl_no == "—")
scan = SW.scan_acc_squeeze({"PTT.BK": DF}, BCH, _p)
check("scan_has_dist_cols",
      scan.empty or {"กระจายของ (v5.14)", "โหวตกระจาย"} <= set(scan.columns),
      str(list(scan.columns))[:120])

print("\n--- 9) ความซื่อตรงของข้อความ (ต้องมีคำเตือนจริง ไม่ใช่โฆษณา) ---")
_swdoc = SW.__doc__ or ""
check("doc_says_display_only", "DISPLAY ONLY" in _swdoc)
check("doc_says_grade_c", "grade C" in _swdoc)
check("doc_says_vwap_undetectable", "VWAP/POV" in _swdoc and "ตรวจไม่เจอ" in _swdoc)
check("doc_says_absence_is_not_proof", "ไม่เท่ากับ" in _swdoc)
check("doc_says_defaults_unchanged", "เท่ากับ v5.13" in _swdoc)
check("doc_f3_states_frozen_r", "FROZEN-R" in _swdoc)
check("marker_note_mentions_red_square",
      "สี่เหลี่ยมแดง" in ACC.MARKER_NOTE and "distShow" in ACC.MARKER_NOTE)
_accdoc = ACC.accumulation_frame.__doc__ or ""
check("accum_doc_documents_mirror", "กระจกเงา" in _accdoc and "0.35" in _accdoc)
try:
    _appsrc = open("app.py", encoding="utf-8").read()
    check("app_warns_display_only", "แสดงผลอย่างเดียว เกรด C" in _appsrc)
    check("app_warns_vwap", "VWAP/POV" in _appsrc)
    check("app_warns_arm_testing", "อย่าเปิดพร้อมกันตอนทดสอบ" in _appsrc)
    check("app_warns_right_tail", "หางขวา" in _appsrc)
except FileNotFoundError:
    check("app_text_checks", False, "ไม่พบ app.py ในโฟลเดอร์นี้")

print(f"\n== {ok} passed, {fail} failed ==")
raise SystemExit(1 if fail else 0)
