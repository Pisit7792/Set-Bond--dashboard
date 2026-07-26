# -*- coding: utf-8 -*-
"""
quant_optimize.py — ตัวรัน "ลูป self-improve" แบบ **ออฟไลน์** (รันบนเครื่องตัวเอง)

ทำไมไม่รันในแอป: Streamlit Cloud ฟรีมี RAM จำกัด (ราว 1GB) และแอปหลับเมื่อไม่มีคนใช้
การรัน Optuna หลายร้อย trial × หลายสิบหุ้น ในแอปจะช้าหรือถูกฆ่ากลางคัน
→ รันไฟล์นี้บนเครื่อง แล้วอัปโหลด optimize_result.json เข้าหน้า "🔬 Self-Improve"

วิธีใช้:
    python quant_optimize.py --prices-dir ./data --out optimize_result.json
    python quant_optimize.py --yf --tickers PTT,AOT,CPALL --period 5y
    python quant_optimize.py --prices-dir ./data --trials 150 --folds 5

สิ่งที่ไฟล์นี้ **จงใจไม่ทำ** (และเหตุผล):
1. ไม่จูนพารามิเตอร์แยกรายหุ้น — López de Prado แนะนำให้สร้างโมเดลระดับ
   universe ทั้งชุด การจูนรายตัวคือ multiple testing คูณจำนวนหุ้น
   (มี --per-ticker ให้ แต่จะขึ้นคำเตือนและคูณ n_trials ตามจำนวนหุ้นจริง)
2. ไม่เลือก "ชุดที่ดีที่สุด" จากข้อมูลทั้งหมด — เลือกจาก train fold เท่านั้น
   แล้ววัดผลใน test fold ที่ไม่เคยเห็น (ค่าที่รายงานคือ OOS ล้วน)
3. ไม่ deploy อัตโนมัติ — ผลลัพธ์คือ "ผู้สมัคร" เท่านั้น (ผังต้นฉบับขั้นที่ 9
   ต่อตรงเข้า robot_trading.py ซึ่งอันตราย: ลูปที่ deploy ตัวเองได้
   จะ deploy ผลของ noise ได้ด้วย)
4. ไม่รีเซ็ตตัวนับ trial — สมุดบัญชี optimize_ledger.json สะสมข้ามการรันทุกครั้ง
   เพราะการรันลูปซ้ำ ๆ คือการลองเพิ่ม ไม่ใช่การเริ่มใหม่ ตัวเลข Deflated Sharpe
   ต้องหักจากยอดสะสม ไม่ใช่ยอดรอบเดียว
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, replace
from datetime import datetime

import numpy as np
import pandas as pd

import quant_evaluation as QE
import set_swing as SW

VERSION = "1.0"
LEDGER = "optimize_ledger.json"

# --- พื้นที่ค้นหาที่ "จดทะเบียนไว้ล่วงหน้า" ---------------------------------
# ตั้งใจให้แคบ: 5 พารามิเตอร์ ไม่ใช่ 30
# ทุกตัวมีเหตุผลเชิงทฤษฎีรองรับ (ระดับความเชื่อมั่นสัญญาณ, ระยะ stop เชิง ATR,
# ระยะ trail, คุณภาพเทรนด์, เพดานความผันผวน) — ไม่ใช่ตัวที่ใส่มาเพราะ "เผื่อดีขึ้น"
SEARCH_SPACE: dict[str, tuple] = {
    "conf_min": ("int", 45, 70, 5),
    "sl_mlt": ("float", 1.5, 3.0, 0.25),
    "tr_mlt": ("float", 2.0, 4.0, 0.5),
    "er_min": ("float", 0.20, 0.40, 0.05),
    "vol_pc": ("float", 80.0, 95.0, 5.0),
}


def space_size() -> int:
    n = 1
    for kind, lo, hi, step in SEARCH_SPACE.values():
        n *= max(1, int(round((hi - lo) / step)) + 1)
    return n


# ---------------------------------------------------------------------------
# สมุดบัญชี trial สะสม
# ---------------------------------------------------------------------------

def read_ledger(path: str = LEDGER) -> dict:
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            return {"total_trials": int(d.get("total_trials", 0)),
                    "runs": list(d.get("runs", []))}
        except Exception:
            pass
    return {"total_trials": 0, "runs": []}


def write_ledger(led: dict, added: int, note: str, path: str = LEDGER) -> dict:
    led["total_trials"] = int(led.get("total_trials", 0)) + int(added)
    led.setdefault("runs", []).append(
        {"เวลา": datetime.now().strftime("%Y-%m-%d %H:%M"),
         "trials": int(added), "หมายเหตุ": note})
    led["runs"] = led["runs"][-200:]
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(led, f, ensure_ascii=False, indent=1)
    except Exception as e:
        print(f"[เตือน] เขียนสมุดบัญชีไม่ได้: {e}", file=sys.stderr)
    return led


# ---------------------------------------------------------------------------
# โหลดราคา
# ---------------------------------------------------------------------------

def load_from_dir(d: str) -> dict:
    out = {}
    for fn in sorted(os.listdir(d)):
        if not fn.lower().endswith(".csv"):
            continue
        tk = os.path.splitext(fn)[0].upper().replace(".BK", "")
        try:
            df = pd.read_csv(os.path.join(d, fn))
            cols = {c.lower(): c for c in df.columns}
            dt = cols.get("date") or df.columns[0]
            df[dt] = pd.to_datetime(df[dt], errors="coerce")
            df = df.dropna(subset=[dt]).set_index(dt).sort_index()
            need = ["Open", "High", "Low", "Close", "Volume"]
            ren = {}
            for n in need:
                src = cols.get(n.lower())
                if src:
                    ren[src] = n
            df = df.rename(columns=ren)
            if all(n in df.columns for n in need):
                out[tk] = df[need].astype(float)
        except Exception as e:
            print(f"[ข้าม] {fn}: {e}", file=sys.stderr)
    return out


def load_from_yf(tickers: list[str], period: str) -> tuple[dict, pd.Series]:
    import set_engine as SE
    prices = SE.load_universe_prices(tickers, period)
    bench = SE.load_benchmark(period)
    return prices, bench


# ---------------------------------------------------------------------------
# ประเมิน config หนึ่งชุดบนช่วงเวลาหนึ่ง
# ---------------------------------------------------------------------------

def _apply(base: SW.SwingParams, cfg: dict) -> SW.SwingParams:
    return replace(base, **{k: v for k, v in cfg.items()
                            if hasattr(base, k)})


def eval_config(frames: dict, cfg: dict, base: SW.SwingParams,
                lo: int, hi: int, cost_mult: float = 1.0) -> dict:
    """รัน backtest ทุกหุ้นในช่วงดัชนี [lo,hi) แล้วรวมเป็นพอร์ตถ่วงเท่ากัน

    หมายเหตุความซื่อสัตย์: การเฉลี่ยผลตอบแทนรายวันของแต่ละหุ้นแบบถ่วงเท่ากัน
    เป็น **การประมาณ** พอร์ตจริง (ไม่ได้จำลองการแย่งเงินทุนระหว่างสัญญาณ
    พร้อมกัน) — ใช้เทียบ config ด้วยกันได้ แต่ห้ามอ่านเป็นผลตอบแทนพอร์ตจริง
    """
    p = _apply(base, cfg)
    if cost_mult != 1.0:
        p = replace(p, comm_side=p.comm_side * cost_mult,
                    spread_e=p.spread_e * cost_mult)
    rets, n_trades, net, entries = [], 0, 0.0, []
    for tk, fr in frames.items():
        sub = fr.iloc[lo:hi]
        if len(sub) < 60:
            continue
        try:
            res = SW.backtest(sub, p)
        except Exception:
            continue
        n_trades += int(res.get("n", 0))
        net += float(res.get("net_thb", 0.0) or 0.0)
        tdf = res.get("trades")
        if tdf is not None and len(tdf) and "เข้า" in tdf.columns:
            entries.extend(pd.to_datetime(tdf["เข้า"], errors="coerce")
                           .dropna().tolist())
        eq = res.get("equity")
        if eq is not None and len(eq) > 2:
            rets.append(eq.pct_change().dropna())
    if not rets:
        return {"returns": pd.Series(dtype=float), "n": 0, "net": 0.0,
                "sharpe": float("nan"), "entries": []}
    port = pd.concat(rets, axis=1).mean(axis=1).dropna()
    return {"returns": port, "n": n_trades, "net": net,
            "sharpe": QE.sharpe(port), "entries": entries}


def proxy_regime(frames: dict, n: int = 200) -> pd.Series:
    """ดัชนีตัวแทน = ค่าเฉลี่ยราคาปิดที่ normalize แล้วของทุกหุ้นใน universe
    regime = ดัชนีตัวแทน > SMA200 (ขาขึ้น) / ไม่เกิน (ขาลง-ออกข้าง)

    เป็น **ตัวแทน** ไม่ใช่ SET index จริง — ใช้เพื่อเช็คว่ากลุ่มตัวอย่างเทรด
    กระจายทั้งสองสภาวะหรือกระจุกอยู่ขาขึ้นล้วน (ซึ่งเป็นกับดักคลาสสิก)
    """
    cols = []
    for tk, fr in frames.items():
        c = fr["Close"].dropna()
        if len(c) > n:
            cols.append(c / c.iloc[0])
    if not cols:
        return pd.Series(dtype=object)
    idx = pd.concat(cols, axis=1).mean(axis=1).dropna()
    return pd.Series(np.where(idx > idx.rolling(n).mean(), "ขาขึ้น", "ขาลง/ออกข้าง"),
                     index=idx.index)


# ---------------------------------------------------------------------------
# ตัวค้นหา: Optuna ถ้ามี ไม่งั้น random search
# ---------------------------------------------------------------------------

def _sample_random(rng: random.Random) -> dict:
    cfg = {}
    for k, (kind, lo, hi, step) in SEARCH_SPACE.items():
        n = int(round((hi - lo) / step))
        v = lo + rng.randint(0, n) * step
        cfg[k] = int(round(v)) if kind == "int" else round(float(v), 4)
    return cfg


def search(frames: dict, base: SW.SwingParams, lo: int, hi: int,
           n_trials: int, seed: int = 7) -> tuple[dict, list[dict]]:
    """คืน (best_cfg, ประวัติ trial ทั้งหมด) — เลือกจาก train window เท่านั้น"""
    history: list[dict] = []
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            cfg = {}
            for k, (kind, a, b, step) in SEARCH_SPACE.items():
                cfg[k] = (trial.suggest_int(k, int(a), int(b), step=int(step))
                          if kind == "int"
                          else trial.suggest_float(k, a, b, step=step))
            r = eval_config(frames, cfg, base, lo, hi)
            history.append({"cfg": cfg, "sharpe": r["sharpe"], "n": r["n"]})
            s = r["sharpe"]
            return -1e9 if not np.isfinite(s) else s

        st = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
        st.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
        best = dict(st.best_params)
    except ImportError:
        rng = random.Random(seed)
        best, best_s = None, -1e18
        for _ in range(int(n_trials)):
            cfg = _sample_random(rng)
            r = eval_config(frames, cfg, base, lo, hi)
            history.append({"cfg": cfg, "sharpe": r["sharpe"], "n": r["n"]})
            s = r["sharpe"]
            if np.isfinite(s) and s > best_s:
                best, best_s = cfg, s
        best = best or _sample_random(random.Random(seed))
    return best, history


# ---------------------------------------------------------------------------
# ลูปหลัก
# ---------------------------------------------------------------------------

def run(frames: dict, base: SW.SwingParams, n_trials: int, folds: int,
        embargo: float, pbo_configs: int, seed: int, note: str) -> dict:
    n_obs = min(len(f) for f in frames.values())
    splits = QE.walk_forward_splits(n_obs, n_folds=folds, embargo_frac=embargo)
    if not splits:
        return {"error": f"ข้อมูลสั้นเกินไป ({n_obs} แท่ง) แบ่ง walk-forward "
                         f"{folds} ช่วงไม่ได้ — ต้องการอย่างน้อยราว "
                         f"{250 + 2 * folds} แท่ง"}

    reg = proxy_regime(frames)
    per_fold, oos_rets, all_hist, oos_entries = [], [], [], []
    tot_trades, tot_net = 0, 0.0
    for i, sp in enumerate(splits, 1):
        tr_lo, tr_hi = sp["train"]
        te_lo, te_hi = sp["test"]
        print(f"  fold {i}/{len(splits)}: train[{tr_lo}:{tr_hi}] "
              f"test[{te_lo}:{te_hi}] embargo={sp['embargo']}")
        best, hist = search(frames, base, tr_lo, tr_hi, n_trials, seed + i)
        all_hist.extend(hist)
        oos = eval_config(frames, best, base, te_lo, te_hi)
        ins = eval_config(frames, best, base, tr_lo, tr_hi)
        oos_rets.append(oos["returns"])
        oos_entries.extend(oos.get("entries") or [])
        tot_trades += int(oos["n"])
        tot_net += float(oos["net"])
        per_fold.append({
            "fold": i, "best_cfg": best,
            "train_sharpe": None if not np.isfinite(ins["sharpe"]) else round(ins["sharpe"], 3),
            "oos_sharpe": None if not np.isfinite(oos["sharpe"]) else round(oos["sharpe"], 3),
            "oos_trades": int(oos["n"]), "oos_net_thb": round(float(oos["net"]), 2),
        })

    oos_all = (pd.concat(oos_rets).sort_index() if oos_rets
               else pd.Series(dtype=float))

    # --- เมทริกซ์สำหรับ PBO: เอา config ที่ลองจริงมาวัดบนช่วงเดียวกันทั้งหมด ---
    seen, uniq = set(), []
    for h in sorted(all_hist, key=lambda x: -(x["sharpe"]
                                              if np.isfinite(x["sharpe"]) else -1e9)):
        key = tuple(sorted(h["cfg"].items()))
        if key not in seen:
            seen.add(key)
            uniq.append(h["cfg"])
        if len(uniq) >= int(pbo_configs):
            break
    cols = []
    for cfg in uniq:
        r = eval_config(frames, cfg, base, 0, n_obs)["returns"]
        if len(r) > 10:
            cols.append(r.rename(json.dumps(cfg, sort_keys=True)))
    pbo = {"pbo": float("nan"), "reason": "ไม่มี config พอสำหรับ PBO"}
    if len(cols) >= 4:
        M = pd.concat(cols, axis=1).dropna()
        pbo = QE.cscv_pbo(M.to_numpy())

    # --- cost stress: ใช้ config ของ fold สุดท้ายกับช่วง OOS สุดท้าย ---
    last = splits[-1]
    cs = eval_config(frames, per_fold[-1]["best_cfg"], base,
                     last["test"][0], last["test"][1],
                     cost_mult=QE.COST_STRESS_MULT)

    regimes: dict[str, int] = {}
    if len(reg) and oos_entries:
        for d in oos_entries:
            try:
                lab = str(reg.asof(pd.Timestamp(d)))
            except Exception:
                continue
            if lab and lab != "nan":
                regimes[lab] = regimes.get(lab, 0) + 1

    led = write_ledger(read_ledger(), len(all_hist), note)
    metrics = {
        "oos_returns": [float(x) for x in oos_all.tolist()],
        "n_trials": int(led["total_trials"]),
        "n_oos_trades": int(tot_trades),
        "net_thb_oos": float(tot_net),
        "pbo": pbo.get("pbo", float("nan")),
        "pbo_reason": pbo.get("reason", ""),
        "regimes": regimes,
        "cost_stress_net": float(cs["net"]),
    }
    gate = QE.gate_verdict(metrics)
    return {
        "เวอร์ชัน": "quant_optimize v1.0",
        "เวลา": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "หุ้น": sorted(frames.keys()),
        "จำนวนแท่ง": int(n_obs),
        "search_space": {k: list(v) for k, v in SEARCH_SPACE.items()},
        "ขนาดพื้นที่ค้นหา": space_size(),
        "trials_รอบนี้": len(all_hist),
        "trials_สะสมทั้งหมด": int(led["total_trials"]),
        "folds": per_fold,
        "pbo": pbo,
        "metrics": {k: v for k, v in metrics.items() if k != "oos_returns"},
        "oos_sharpe_รวม": (None if not np.isfinite(QE.sharpe(oos_all))
                           else round(QE.sharpe(oos_all), 3)),
        "gate": gate,
        "หมายเหตุ": QE.VERDICT_NOTE,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ลูป self-improve แบบออฟไลน์ (ไม่ deploy อัตโนมัติ)")
    ap.add_argument("--prices-dir", help="โฟลเดอร์ CSV รายหุ้น (Date,Open,High,Low,Close,Volume)")
    ap.add_argument("--yf", action="store_true", help="ดึงจาก yfinance แทน")
    ap.add_argument("--tickers", default="", help="เช่น PTT,AOT,CPALL")
    ap.add_argument("--period", default="5y")
    ap.add_argument("--trials", type=int, default=100, help="trial ต่อ fold")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embargo", type=float, default=0.01)
    ap.add_argument("--pbo-configs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="optimize_result.json")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)

    if a.prices_dir:
        prices = load_from_dir(a.prices_dir)
        bench = None
    elif a.yf:
        tks = [t.strip() for t in a.tickers.split(",") if t.strip()]
        if not tks:
            print("ต้องระบุ --tickers เมื่อใช้ --yf", file=sys.stderr)
            return 2
        prices, bench = load_from_yf(tks, a.period)
    else:
        print("ต้องเลือก --prices-dir หรือ --yf", file=sys.stderr)
        return 2
    if not prices:
        print("ไม่พบข้อมูลราคา", file=sys.stderr)
        return 2

    base = SW.SwingParams()
    frames = {}
    for tk, df in prices.items():
        try:
            frames[tk] = SW.compute_frame(df, bench, tk, base)
        except Exception as e:
            print(f"[ข้าม] {tk}: {e}", file=sys.stderr)
    if not frames:
        print("สร้าง frame ไม่ได้เลย", file=sys.stderr)
        return 2

    print(f"หุ้น {len(frames)} ตัว · trial/fold {a.trials} · folds {a.folds}")
    print(f"พื้นที่ค้นหา {space_size():,} ชุด (ลองจริง {a.trials * a.folds:,})")
    res = run(frames, base, a.trials, a.folds, a.embargo, a.pbo_configs,
              a.seed, a.note or f"{len(frames)} หุ้น")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    if res.get("error"):
        print("ผิดพลาด:", res["error"], file=sys.stderr)
        return 1
    g = res["gate"]
    print(f"\nประตูตัดสิน: {'ผ่าน' if g['pass'] else 'ไม่ผ่าน'}")
    for c in g["checks"]:
        print(f"  [{'ผ่าน' if c['ผ่าน'] else 'ตก '}] {c['ชื่อ']}: "
              f"{c['ค่า']} (เกณฑ์ {c['เกณฑ์']})")
    print(f"\nเขียนผลไปที่ {a.out} — อัปโหลดเข้าหน้า 'Self-Improve' ในแอป")
    _ = asdict  # keep import meaningful for future param dumps
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
