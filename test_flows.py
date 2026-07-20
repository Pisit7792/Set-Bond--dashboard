"""ทดสอบ flows.py กับไฟล์จริงของผู้ใช้: python3 test_flows.py"""
import os
import tempfile

import numpy as np
import pandas as pd

import flows as FL

ok = fail = 0
def check(name, cond, detail=""):
    global ok, fail
    if cond: ok += 1; print(f"PASS  {name} {detail}")
    else: fail += 1; print(f"FAIL  {name} {detail}")

# 1) โหลดไฟล์จริง
df, issues = FL.load_flow_csv("Set_update.csv")
check("load_rows>=1100", len(df) >= 1100, f"n={len(df)}")
check("cols_present", all(c in df.columns for c in FL.NET_COLS + [FL.IDX_COL]))
check("index_ascending", df.index.is_monotonic_increasing)
check("index_unique", df.index.is_unique)
check("latest_date_2026", df.index[-1].year == 2026, str(df.index[-1].date()))
check("latest_values", abs(df["Foreign"].iloc[-1] - 5152.86) < 0.01,
      f"={df['Foreign'].iloc[-1]}")
bal = float(df[FL.NET_COLS].sum(axis=1).abs().median())
check("groups_balance_zero", bal < 1.0 and not any("หายไป" in i for i in issues),
      f"median|sum|={bal:.2f}")
check("numeric", df["Institute"].dtype.kind == "f")

# 2) append / duplicate / overwrite
nd = df.index[-1] + pd.offsets.BDay(1)
d2, okk, msg = FL.append_or_update(df, nd, -100, 250.5, -150.5, 1640.0, 0.96)
check("append_ok", okk and len(d2) == len(df) + 1, msg)
check("append_sorted", d2.index.is_monotonic_increasing)
d3, okk2, msg2 = FL.append_or_update(d2, nd, 1, 2, 3)
check("dup_rejected", not okk2 and "เขียนทับ" in msg2)
d4, okk3, _ = FL.append_or_update(d2, nd, 9, 9, 9, overwrite=True)
check("overwrite_ok", okk3 and abs(d4.loc[nd, "Foreign"] - 9) < 1e-9
      and len(d4) == len(d2))

# 3) save -> reload roundtrip (คงรูปแบบเดิม)
tmp = os.path.join(tempfile.gettempdir(), "flow_rt.csv")
FL.save_flow_csv(d2, tmp)
raw = open(tmp, "rb").read()
check("bom_kept", raw[:3] == b"\xef\xbb\xbf")
check("crlf_kept", b"\r\n" in raw)
check("header_kept", FL.ORIG_HEADER.encode("utf-8") in raw)
d5, iss5 = FL.load_flow_csv(tmp)
check("roundtrip_len", len(d5) == len(d2))
check("roundtrip_values",
      float(abs(d5[FL.NET_COLS] - d2[FL.NET_COLS]).max().max()) < 0.01)
b = FL.to_csv_bytes(d2)
check("bytes_export", b[:3] == b"\xef\xbb\xbf" and b"17/7/2026" in b)

# 4) analytics
s = pd.Series([5, 3, -2, 4, 6, 7])
check("streak_pos3", FL.streak(s) == 3)
check("streak_neg", FL.streak(pd.Series([1, -2, -3])) == -2)
check("streak_zero", FL.streak(pd.Series([1, 0])) == 0)
rs = FL.roll_sum(df, 20)
check("rollsum_shape", rs.shape == df[FL.NET_COLS].shape and
      np.isfinite(rs.iloc[-1]["Foreign"]))
summ = FL.flow_summary(df)
check("summary_3rows", len(summ) == 3 and "z วันล่าสุด" in summ.columns)
check("summary_streak_int", summ["ซื้อ/ขายติดกัน (วัน)"].dtype.kind in "i")
z = FL.zscore_full(df["Foreign"])
check("z_finite", np.isfinite(z), f"z={z:.2f}")

# 5) same-day vs next-day corr (ข้อมูลจริง)
cc = FL.same_day_corr(df)
check("corr_ok", cc["ok"] and cc["n"] >= 200)
check("corr_bounded", -1 <= cc["same_day"] <= 1 and
      (cc["next_day"] != cc["next_day"] or -1 <= cc["next_day"] <= 1),
      f"same={cc['same_day']:.2f} next={cc['next_day']:.2f}")

print(f"\n== {ok} passed, {fail} failed ==")
raise SystemExit(1 if fail else 0)
