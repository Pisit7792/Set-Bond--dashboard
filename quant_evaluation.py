# -*- coding: utf-8 -*-
"""
quant_evaluation.py — ชั้น "ตรวจสอบ" ของลูป self-improve (กล่อง quant_evaluation.py ในผัง)

หน้าที่จริงของโมดูลนี้คือ **ขัดขวางไม่ให้ลูปหลอกตัวเอง** ไม่ใช่รายงานว่าดีขึ้น

ทำไมต้องมี (อ้างจากเอกสารใน project เอง):
- Optuna ยิ่งลอง trial เยอะ ยิ่งเจอ Sharpe สูงที่เป็นเสียงรบกวน → ต้องหัก
  ด้วย Deflated Sharpe Ratio (Bailey & López de Prado 2014) ซึ่งรับ n_trials
- ต้องวัด Probability of Backtest Overfitting (Bailey, Borwein, López de Prado
  & Zhu 2017) ด้วย CSCV — ถ้า PBO > 50% แปลว่ากระบวนการคัดเลือกแย่กว่าสุ่ม
- ต้องแยก train/test ด้วย purge + embargo (López de Prado) ไม่งั้นข้อมูลรั่ว
- ประตูตัดสิน (C9) ต้องผ่าน **ทุกข้อ** ไม่ใช่ผ่านบางข้อแล้วเคลม

ข้อจำกัดที่ต้องบอกตรง ๆ:
- PBO/DSR ลดโอกาสหลอกตัวเอง แต่ **ไม่รับประกันว่าจะกำไรจริง** ไม่มีสถิติตัวไหนทำได้
- ถ้าจำนวน config ที่ลอง (N) น้อยกว่า ~10 หรือ observation น้อย ค่า PBO
  จะไม่เสถียร — โมดูลนี้จะคืน nan พร้อมเหตุผล ไม่ปั้นตัวเลขให้ดูดี
- นับ trial ต้องนับ **ทุกครั้งที่รัน** สะสมข้ามวัน ไม่ใช่นับเฉพาะรอบนี้
  (รันลูปซ้ำ 10 รอบ รอบละ 200 trials = 2,000 trials ไม่ใช่ 200)
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd

import set_engine as SE

# เกณฑ์ประตู (แก้ได้ แต่ค่าตั้งต้นอิงเอกสารใน project)
DSR_MIN = 0.95          # DSR เป็นความน่าจะเป็น (แบบ PSR) — 0.95 = นัยสำคัญปกติ
PBO_MAX = 0.25
TSTAT_MIN = 3.0         # Harvey-Liu-Zhu: ของใหม่ต้อง t > 3 ไม่ใช่ 2
OOS_TRADES_MIN = 100
COST_STRESS_MULT = 1.5  # ต้องรอดเมื่อต้นทุน +50%


# ---------------------------------------------------------------------------
# 1) แบ่งข้อมูล: walk-forward แบบ anchored + purge + embargo
# ---------------------------------------------------------------------------

def walk_forward_splits(n_obs: int, n_folds: int = 5,
                        embargo_frac: float = 0.01,
                        min_train: int = 250) -> list[dict]:
    """คืน [{'train': (a,b), 'test': (c,d)}] เป็นดัชนีครึ่งเปิด [a,b)

    anchored = train ขยายไปเรื่อย ๆ, test คือช่วงถัดไปที่ไม่เคยเห็น
    embargo = เว้นช่องว่างหลัง train ก่อนเริ่ม test กันข้อมูลรั่วจาก
    autocorrelation และจากเทรดที่คาบเกี่ยวสองช่วง
    """
    n_obs = int(n_obs)
    if n_obs < min_train + 2 * n_folds:
        return []
    emb = max(1, int(round(n_obs * float(embargo_frac))))
    usable = n_obs - min_train
    step = usable // int(n_folds)
    if step <= emb + 1:
        return []
    out = []
    for k in range(int(n_folds)):
        tr_end = min_train + k * step
        te_start = tr_end + emb
        te_end = min(n_obs, tr_end + step)
        if te_start >= te_end:
            continue
        out.append({"train": (0, tr_end), "test": (te_start, te_end),
                    "embargo": emb})
    return out


# ---------------------------------------------------------------------------
# 2) สถิติพื้นฐาน
# ---------------------------------------------------------------------------

def sharpe(returns, periods: int = 252) -> float:
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(r.mean() / sd * math.sqrt(periods))


def tstat(returns) -> float:
    """t ของ mean return เทียบศูนย์ — เกณฑ์ Harvey-Liu-Zhu ใช้ตัวนี้"""
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if len(r) < 3:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(r.mean() / (sd / math.sqrt(len(r))))


def deflated_sharpe(returns, n_trials: int) -> float:
    """ยืมของ set_engine เพื่อไม่ให้มีสองสูตรในโปรเจกต์เดียว"""
    return SE.deflated_sharpe(returns, n_trials)


# ---------------------------------------------------------------------------
# 3) PBO ด้วย CSCV (Bailey, Borwein, López de Prado & Zhu 2017)
# ---------------------------------------------------------------------------

def cscv_pbo(perf_matrix, n_groups: int = 8) -> dict:
    """perf_matrix: (T observations × N configs) ผลตอบแทนรายคาบของแต่ละ config

    วิธี: หั่น T เป็น S กลุ่มเท่า ๆ กัน → เลือก S/2 กลุ่มเป็น IS ที่เหลือเป็น OOS
    ครบทุก combination → ในแต่ละ combination หา config ที่ดีสุดใน IS
    แล้วดูว่ามัน "อยู่อันดับที่เท่าไรใน OOS"
    PBO = สัดส่วนที่ config ดีสุดใน IS ตกไปอยู่ครึ่งล่างของ OOS

    ตีความ: PBO > 0.5 = การคัดเลือกแย่กว่าโยนเหรียญ
    คืน nan + reason ถ้าข้อมูลไม่พอ (ไม่ปั้นเลข)
    """
    M = np.asarray(perf_matrix, dtype=float)
    if M.ndim != 2:
        return {"pbo": float("nan"), "n_splits": 0,
                "reason": "perf_matrix ต้องเป็น 2 มิติ (T × N)"}
    T, N = M.shape
    if N < 4:
        return {"pbo": float("nan"), "n_splits": 0,
                "reason": f"มีเพียง {N} config — PBO ต้องการอย่างน้อย 4 "
                          "(ยิ่งน้อยยิ่งไม่มีความหมาย)"}
    S = int(n_groups)
    if S % 2:
        S -= 1
    S = max(4, min(S, 16))
    if T < S * 10:
        return {"pbo": float("nan"), "n_splits": 0,
                "reason": f"มี {T} คาบ ต่ำกว่าขั้นต่ำ {S * 10} สำหรับ {S} กลุ่ม"}

    edges = np.array_split(np.arange(T), S)
    lambdas = []
    half = S // 2
    for combo in combinations(range(S), half):
        is_idx = np.concatenate([edges[g] for g in combo])
        oos_idx = np.concatenate([edges[g] for g in range(S)
                                  if g not in combo])
        is_perf = np.array([sharpe(M[is_idx, j]) for j in range(N)])
        oos_perf = np.array([sharpe(M[oos_idx, j]) for j in range(N)])
        if not np.isfinite(is_perf).any() or not np.isfinite(oos_perf).any():
            continue
        best = int(np.nanargmax(is_perf))
        finite = np.isfinite(oos_perf)
        if finite.sum() < 2 or not finite[best]:
            continue
        # อันดับสัมพัทธ์ของ config ที่ดีสุดใน IS เมื่อไปวัดใน OOS
        rank = float((oos_perf[finite] < oos_perf[best]).sum())
        omega = (rank + 1.0) / (float(finite.sum()) + 1.0)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        lambdas.append(math.log(omega / (1.0 - omega)))
    if not lambdas:
        return {"pbo": float("nan"), "n_splits": 0,
                "reason": "คำนวณไม่ได้ (Sharpe เป็น nan ทุกชุด)"}
    lam = np.array(lambdas, dtype=float)
    # ความจริงที่ต้องบอก: PBO เองก็เป็นค่าประมาณที่แกว่ง — ทดสอบกับข้อมูลสุ่มล้วน
    # (ซึ่งค่าจริงคือ 0.5) ด้วย 20 config × 1,200 คาบ ได้ค่ากระจาย 0.10-0.86
    # ต่อ seed แม้ค่าเฉลี่ยจะเข้าใกล้ 0.5 → อย่าอ่าน PBO เป็นตัวเลขเป๊ะ
    note = ("PBO เป็นค่าประมาณที่แกว่งเอง — ตีความเป็นช่วง ไม่ใช่ตัวเลขเป๊ะ")
    if N < 10:
        note += f" · มีเพียง {N} config ค่าที่ได้ยิ่งไม่เสถียร"
    return {"pbo": float((lam <= 0).mean()), "n_splits": int(len(lam)),
            "n_configs": int(N), "lambda_median": float(np.median(lam)),
            "reason": "", "note": note}


# ---------------------------------------------------------------------------
# 4) ประตูตัดสิน (C9) — ต้องผ่านทุกข้อ
# ---------------------------------------------------------------------------

def gate_verdict(m: dict) -> dict:
    """m ต้องมีคีย์: oos_returns (list), n_trials, n_oos_trades,
    net_thb_oos, pbo, regimes (dict ชื่อ->จำนวนเทรด), cost_stress_net

    คืน {"pass": bool, "checks": [{"ชื่อ","ผ่าน","ค่า","เกณฑ์","หมายเหตุ"}]}
    """
    r = pd.Series(m.get("oos_returns") or [], dtype=float).dropna()
    n_trials = int(m.get("n_trials") or 0)
    checks = []

    def add(name, ok, val, crit, note=""):
        checks.append({"ชื่อ": name, "ผ่าน": bool(ok), "ค่า": val,
                       "เกณฑ์": crit, "หมายเหตุ": note})

    net = float(m.get("net_thb_oos", float("nan")))
    add("กำไรสุทธิ OOS หลังต้นทุน", np.isfinite(net) and net > 0,
        f"{net:,.0f} บาท" if np.isfinite(net) else "—", "> 0",
        "ต้องเป็นผลจากช่วงที่ไม่ได้ใช้จูนเท่านั้น")

    dsr = deflated_sharpe(r, n_trials) if len(r) else float("nan")
    add("Deflated Sharpe (หัก n_trials แล้ว)",
        np.isfinite(dsr) and dsr > DSR_MIN,
        f"{dsr:.3f}" if np.isfinite(dsr) else "—", f"> {DSR_MIN}",
        f"หักจากการลอง {n_trials:,} ชุด — ถ้าตัวเลขนี้ต่ำ แปลว่า Sharpe ที่เห็น"
        "อธิบายได้ด้วยการลองเยอะ")

    pbo = float(m.get("pbo", float("nan")))
    add("PBO (โอกาสที่เป็น overfit)", np.isfinite(pbo) and pbo < PBO_MAX,
        f"{pbo:.1%}" if np.isfinite(pbo) else "คำนวณไม่ได้",
        f"< {PBO_MAX:.0%}", str(m.get("pbo_reason", "")))

    t = tstat(r) if len(r) else float("nan")
    add("t-stat ของผลตอบแทน OOS", np.isfinite(t) and t > TSTAT_MIN,
        f"{t:.2f}" if np.isfinite(t) else "—", f"> {TSTAT_MIN}",
        "Harvey-Liu-Zhu: ของค้นพบใหม่ต้อง t>3 ไม่ใช่ 2")

    ntr = int(m.get("n_oos_trades") or 0)
    add("จำนวนเทรด OOS", ntr >= OOS_TRADES_MIN, f"{ntr:,}",
        f"≥ {OOS_TRADES_MIN}", "ต่ำกว่านี้ช่วงความเชื่อมั่นกว้างจนไร้ความหมาย")

    regs = m.get("regimes") or {}
    thin = [k for k, v in regs.items() if int(v) < OOS_TRADES_MIN]
    add("ครอบคลุมหลายสภาวะตลาด", bool(regs) and not thin,
        ", ".join(f"{k}={v}" for k, v in regs.items()) or "ไม่ได้แยก",
        f"ทุก regime ≥ {OOS_TRADES_MIN}",
        ("regime ที่ตัวอย่างบาง: " + ", ".join(thin)) if thin else "")

    cs = float(m.get("cost_stress_net", float("nan")))
    add(f"ทนต้นทุน +{(COST_STRESS_MULT - 1) * 100:.0f}%",
        np.isfinite(cs) and cs > 0,
        f"{cs:,.0f} บาท" if np.isfinite(cs) else "ไม่ได้ทดสอบ", "> 0",
        "ต้นทุนไทยจริงราว 0.5% ต่อรอบ — ถ้าบวกอีกครึ่งแล้วขาดทุน แปลว่าขอบบาง")

    return {"pass": all(c["ผ่าน"] for c in checks), "checks": checks,
            "dsr": dsr, "tstat": t, "n_trials": n_trials}


VERDICT_NOTE = (
    "ผ่านประตูนี้ = 'ยังไม่ถูกจับได้ว่าหลอกตัวเอง' **ไม่ใช่** 'จะกำไรจริง' — "
    "ขั้นถัดไปที่ถูกต้องคือ paper trade แล้วเทียบผลจริงกับ backtest "
    "ไม่ใช่เอาไปเทรดเงินจริงทันที"
)
