# -*- coding: utf-8 -*-
"""
set_engine.py — ชั้นหุ้นไทย SET100 (ตรรกะล้วน ทดสอบได้ — UI/cache อยู่ใน app.py)

หลักการเดียวกับ engine.py ฝั่ง Global:
- ทุกคะแนนมาจากสูตรเปิดเผย, ปัจจัยที่มีหลักฐานรองรับระดับปานกลางขึ้นไปใน SET/EM
- สภาพคล่องเป็น 'ประตู' ไม่ใช่คะแนน | z-score winsorize ±3 | น้ำหนักเท่ากัน
- Backtest หักต้นทุนไทยจริง + กัน look-ahead (ลงมือแท่งถัดไปเสมอ)
- ใช้สถิติร่วมกับ engine.py (Wilson CI, PSR) และเพิ่ม Deflated Sharpe ที่นี่

หมายเหตุรายชื่อ SET100 (ซื่อตรง):
- เรียบเรียงสำหรับรอบ 1 ก.ค. – 31 ธ.ค. 2569 จากประกาศ ตลท. (มิ.ย. 2569):
  SET100 เข้าใหม่: MRDIYT, THAI, THCOM, WHAUP | ออก: JAS, JMART, SISB, SJWD
  SET50 เข้าใหม่ (จึงอยู่ใน SET100 แน่นอน): BCP, MRDIYT, TFG, THAI
- รายชื่อ sync ตรงกับตาราง getProfile ใน SET Swing v5.11 (รายชื่อทางการ) —
  อัปเดตพร้อมกันทุกรอบ ม.ค./ก.ค. และแก้ชั่วคราวได้ในแถบข้างของแอป
- หุ้นเข้าตลาดใหม่ (เช่น MRDIYT, THAI) ประวัติราคาสั้น → หลายปัจจัยจะเป็น
  'ไม่มีข้อมูล' และไม่ถูกคิดคะแนนรวม — นี่คือพฤติกรรมที่ถูกต้อง ไม่ใช่บั๊ก
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

import engine as E

TRADING_DAYS = 252
_EULER = 0.5772156649015329
MIN_TRADES_HARD = 30
MIN_TRADES_GOOD = 100

# ---------------------------------------------------------------------------
# 1) Universe SET100 (แก้ไขได้ — ดู docstring ด้านบน)
# ---------------------------------------------------------------------------
SET100_H2_2026 = [
    # รายชื่อทางการ H2-2026 (มีผล 1 ก.ค.–31 ธ.ค. 69) — sync ตรงจาก
    # ตาราง getProfile ใน SET Swing v5.11 ของผู้ใช้ (ประกาศ ตลท. 17 มิ.ย. 69)
    "AAV", "ADVANC", "AEONTS", "AMATA", "AOT", "AP", "AURA", "AWC", "BA",
    "BAM", "BANPU", "BBL", "BCH", "BCP", "BCPG", "BDMS", "BEM", "BGRIM",
    "BH", "BJC", "BLA", "BTG", "BTS", "CBG", "CCET", "CENTEL", "CHG",
    "CK", "COM7", "CPALL", "CPF", "CPN", "CRC", "DELTA", "DOHOME", "EA",
    "EGCO", "ERW", "GFPT", "GLOBAL", "GPSC", "GULF", "GUNKUL", "HANA", "HMPRO",
    "ICHI", "IRPC", "IVL", "JMT", "JTS", "KBANK", "KCE", "KKP", "KTB",
    "KTC", "LH", "M", "MEGA", "MINT", "MOSHI", "MRDIYT", "MTC", "OR",
    "OSP", "PLANB", "PR9", "PRM", "PTG", "PTT", "PTTEP", "PTTGC", "QH",
    "RATCH", "RCL", "SAWAD", "SCB", "SCC", "SCGP", "SIRI", "SPALI", "SPRC",
    "STA", "STECON", "STGT", "TASCO", "TCAP", "TFG", "THAI", "THCOM", "TIDLOR",
    "TISCO", "TLI", "TOA", "TOP", "TRUE", "TTB", "TU", "VGI", "WHA",
    "WHAUP",
]

BENCHMARK_CANDIDATES = [
    ("^SET.BK", "SET Index"),
    ("TDEX.BK", "TDEX (ETF SET50 — ใช้เป็นตัวแทนดัชนี)"),
]
THB_TICKER = "THB=X"  # USD/THB: ค่าขึ้น = บาทอ่อน


def to_yahoo(symbols) -> list:
    """ชื่อย่อไทย -> สัญลักษณ์ Yahoo (เติม .BK, ตัดซ้ำ, คงลำดับ)"""
    out, seen = [], set()
    for s in symbols:
        s = str(s).strip().upper()
        if not s:
            continue
        if not s.endswith(".BK") and "=" not in s and not s.startswith("^"):
            s = s + ".BK"
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# 2) Data helpers (pure; การ cache ทำที่ app.py)
# ---------------------------------------------------------------------------

def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """คอลัมน์ชั้นเดียว, เวลาไม่มี timezone, เรียงเวลา"""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.loc[:, ~pd.Index(d.columns).duplicated()]
    idx = pd.to_datetime(d.index)
    try:
        if idx.tz is not None:
            idx = idx.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    d.index = idx
    return d.sort_index().dropna(how="all")


def yf_download(tickers, period: str):
    """เรียก yfinance แบบกันเหนียว — เวอร์ชันใหม่เปลี่ยนพารามิเตอร์จะถอยไปพื้นฐาน"""
    import yfinance as yf
    try:
        return yf.download(tickers, period=period, auto_adjust=True,
                           group_by="ticker", threads=True, progress=False)
    except TypeError:
        return yf.download(tickers, period=period, group_by="ticker")


def extract_ticker(raw: pd.DataFrame, t: str) -> pd.DataFrame:
    """ดึงตารางของหุ้น t ไม่ว่า ticker อยู่ระดับบนหรือล่างของ MultiIndex"""
    if isinstance(raw.columns, pd.MultiIndex):
        if t in raw.columns.get_level_values(0):
            return raw[t]
        last = raw.columns.nlevels - 1
        if t in raw.columns.get_level_values(last):
            return raw.xs(t, axis=1, level=last)
        raise KeyError(t)
    return raw


def load_universe_prices(tickers, period: str):
    """-> (dict[ticker -> OHLCV DataFrame], failed list)"""
    out, failed = {}, []
    if not tickers:
        return out, failed
    try:
        raw = yf_download(list(tickers), period)
    except Exception:
        return out, list(tickers)
    if raw is None or len(raw) == 0:
        return out, list(tickers)
    for t in tickers:
        try:
            d = normalize_ohlc(extract_ticker(raw, t))
            if len(d) >= 60 and "Close" in d.columns:
                out[t] = d
            else:
                failed.append(t)
        except Exception:
            failed.append(t)
    return out, failed


def load_single(ticker: str, period: str) -> pd.DataFrame:
    try:
        raw = yf_download(ticker, period)
        if raw is None or len(raw) == 0:
            return pd.DataFrame()
        try:
            sub = extract_ticker(raw, ticker)
        except KeyError:
            sub = raw
        return normalize_ohlc(sub)
    except Exception:
        return pd.DataFrame()


def load_benchmark(period: str):
    """-> (symbol, label, DataFrame) พร้อม fallback"""
    for sym, label in BENCHMARK_CANDIDATES:
        d = load_single(sym, period)
        if len(d) >= 210 and "Close" in d.columns:
            return sym, label, d
    return None, None, pd.DataFrame()


# ---------------------------------------------------------------------------
# 3) Indicators (คำอธิบายภาษาคนอยู่ใน docstring และ GLOSSARY ของ bridge)
# ---------------------------------------------------------------------------

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def rsi_series(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def ann_vol(close: pd.Series, n: int = 60) -> float:
    r = close.pct_change().tail(n).dropna()
    if len(r) < max(20, n // 2):
        return float("nan")
    return float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    e = equity.dropna()
    if len(e) == 0:
        return float("nan")
    return float((e / e.cummax() - 1.0).min())


def trend_score(close: pd.Series, n: int = 200) -> float:
    """ราคา ÷ SMA200 − 1 (บวก = เหนือแนวโน้มระยะยาว)"""
    c = close.dropna()
    if len(c) < n + 5:
        return float("nan")
    m = c.rolling(n).mean().iloc[-1]
    if not np.isfinite(m) or m == 0:
        return float("nan")
    return float(c.iloc[-1] / m - 1.0)


def momentum_6_1(close: pd.Series) -> float:
    """โมเมนตัม ~6 เดือน เว้นเดือนล่าสุด (มาตรฐานงานวิจัย ลด reversal สั้น)"""
    c = close.dropna()
    if len(c) < 130:
        return float("nan")
    return float(c.iloc[-22] / c.iloc[-127] - 1.0)


def residual_momentum(close: pd.Series, bench_close: pd.Series,
                      lookback: int = 252, skip: int = 21) -> float:
    """โมเมนตัม 'ส่วนเกินจากตลาด' (หัก beta) — นิ่งกว่าโมเมนตัมดิบใน SET/เอเชีย"""
    r = close.pct_change()
    m = bench_close.pct_change()
    df = pd.concat([r, m], axis=1, join="inner").dropna()
    need = lookback + skip
    if len(df) < need:
        return float("nan")
    win = df.iloc[-need:-skip] if skip > 0 else df.iloc[-lookback:]
    y = win.iloc[:, 0].to_numpy(dtype=float)
    x = win.iloc[:, 1].to_numpy(dtype=float)
    vx = x.var()
    if vx <= 0:
        return float("nan")
    beta = float(((x - x.mean()) * (y - y.mean())).mean() / vx)
    alpha = float(y.mean() - beta * x.mean())
    res = y - (alpha + beta * x)
    s = res.std(ddof=1)
    if not np.isfinite(s) or s <= 0:
        return float("nan")
    return float(res.mean() / s * math.sqrt(TRADING_DAYS))


def median_turnover_thb(df: pd.DataFrame, n: int = 20) -> float:
    """มูลค่าซื้อขาย/วัน (บาท) ค่ากลาง n วัน — ใช้เป็นประตูสภาพคล่อง (gate)"""
    if "Volume" not in df.columns or "Close" not in df.columns:
        return float("nan")
    v = (df["Close"] * df["Volume"]).tail(n).dropna()
    if len(v) < max(5, n // 2):
        return float("nan")
    return float(v.median())


def cross_z(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return s * np.nan
    return (s - mu) / sd


def winsorize(series: pd.Series, limit: float = 3.0) -> pd.Series:
    return series.clip(lower=-limit, upper=limit)


# ---------------------------------------------------------------------------
# 4) Scoreboard SET100
# ---------------------------------------------------------------------------
FACTOR_COLS = ["z_trend", "z_mom", "z_resmom", "z_lowvol"]
MIN_FACTORS = 3

FACTOR_EXPLAIN = {
    "trend": ("แนวโน้ม (ราคาเทียบ SMA200)",
              "ราคาปิด ÷ ค่าเฉลี่ย 200 วัน − 1 | บวก = เหนือแนวโน้มระยะยาว"),
    "mom": ("โมเมนตัม 6-1 เดือน",
            "ผลตอบแทน ~6 เดือน เว้นเดือนล่าสุด (หลักฐานปานกลางใน SET)"),
    "resmom": ("Residual momentum",
               "โมเมนตัมส่วนที่เกินกว่าตลาดพาไป (หัก beta) — นิ่งกว่าโมเมนตัมดิบ"),
    "vol": ("ความผันผวนต่ำ (กลับด้าน)",
            "ผันผวน 60 วันต่อปี ต่ำได้คะแนนสูง (low-volatility anomaly)"),
}


def build_scoreboard(prices: dict, bench_close: pd.Series,
                     min_turnover_thb: float) -> pd.DataFrame:
    rows = []
    for t, df in prices.items():
        c = df["Close"].dropna()
        rows.append({
            "ticker": t.replace(".BK", ""),
            "close": float(c.iloc[-1]) if len(c) else float("nan"),
            "turnover_thb": median_turnover_thb(df),
            "trend": trend_score(c),
            "mom": momentum_6_1(c),
            "resmom": residual_momentum(c, bench_close),
            "vol": ann_vol(c),
        })
    board = pd.DataFrame(rows)
    if board.empty:
        return board
    board["liq_pass"] = board["turnover_thb"] >= float(min_turnover_thb)
    board["z_trend"] = winsorize(cross_z(board["trend"]))
    board["z_mom"] = winsorize(cross_z(board["mom"]))
    board["z_resmom"] = winsorize(cross_z(board["resmom"]))
    board["z_lowvol"] = winsorize(-cross_z(board["vol"]))
    zs = board[FACTOR_COLS]
    board["n_factors"] = zs.notna().sum(axis=1)
    board["composite"] = zs.mean(axis=1, skipna=True)
    board.loc[board["n_factors"] < MIN_FACTORS, "composite"] = np.nan
    return board.sort_values("composite", ascending=False, na_position="last")


# ---------------------------------------------------------------------------
# 5) ต้นทุนหุ้นไทย (สเปกโปรเจกต์: ต้นทุนต้องมาก่อนทุกอย่าง)
# ---------------------------------------------------------------------------

@dataclass
class ThaiCost:
    commission_pct: float = 0.15   # % ต่อข้าง
    vat_pct: float = 7.0
    exchange_fees_pct: float = 0.007
    half_spread_pct: float = 0.15  # SET50 ~0.10-0.25 / SET51-100 มักกว้างกว่า

    def per_side_pct(self) -> float:
        return (self.commission_pct * (1.0 + self.vat_pct / 100.0)
                + self.exchange_fees_pct + self.half_spread_pct)

    def round_trip_pct(self) -> float:
        return 2.0 * self.per_side_pct()

    def breakdown(self):
        comm_vat = self.commission_pct * (1.0 + self.vat_pct / 100.0)
        return [
            ("ค่าคอมมิชชั่น + VAT", comm_vat,
             f"คอม {self.commission_pct:.3f}% × 1.{int(self.vat_pct):02d}"),
            ("ค่าธรรมเนียมตลาดฯ", self.exchange_fees_pct,
             "SET 0.005% + ชำระราคา 0.001% + กำกับ 0.001% (ประมาณ)"),
            ("Half-spread + slippage", self.half_spread_pct,
             "หุ้นกลาง-เล็กใน SET51-100 spread กว้างกว่า SET50 — ปรับขึ้นตามจริง"),
        ]


# ---------------------------------------------------------------------------
# 6) Validation เพิ่มเติมจาก engine.py: Deflated Sharpe + เกณฑ์ n
# ---------------------------------------------------------------------------

def deflated_sharpe(returns, n_trials: int) -> float:
    """DSR (Bailey & López de Prado 2014): PSR ที่ยกเกณฑ์เป็น 'Sharpe สูงสุด
    ที่คาดจากการสุ่มลอง n_trials ชุด' — ยากันหลอกตัวเองจากการเลือกอันดีสุด
    engine.py มี PSR อยู่แล้ว; ฟังก์ชันนี้เติมส่วน deflation"""
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    T = len(r)
    if T < 30:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    sr = float(r.mean() / sd)
    if n_trials is None or n_trials < 2:
        return E.probabilistic_sharpe_ratio(pd.Series(r), 0.0)
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))
    var_sr = (1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr) / (T - 1)
    if var_sr <= 0:
        return float("nan")
    sr_star = math.sqrt(var_sr) * (
        (1.0 - _EULER) * norm.ppf(1.0 - 1.0 / n_trials)
        + _EULER * norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    return E.probabilistic_sharpe_ratio(pd.Series(r), sr_star)


def sample_verdict(n_trades: int):
    if n_trades < MIN_TRADES_HARD:
        return ("fail",
                f"มีเพียง {n_trades} เทรด (< {MIN_TRADES_HARD}) — ข้อมูลไม่พอทางสถิติ "
                "ตัวเลขทุกตัวยังเป็น 'เสียงรบกวน' ห้ามใช้ตัดสินใจลงทุนจริง")
    if n_trades < MIN_TRADES_GOOD:
        return ("warn",
                f"มี {n_trades} เทรด (ต่ำกว่า {MIN_TRADES_GOOD}) — ดูแนวโน้มได้ "
                "แต่ช่วงความเชื่อมั่นยังกว้าง ควรสะสมให้ครบหลายสภาวะตลาด")
    return ("ok",
            f"มี {n_trades} เทรด — เริ่มพอเชื่อได้ "
            "(ตรวจว่าครอบคลุมทั้งขาขึ้น/ขาลง ไม่ใช่ขาขึ้นล้วน)")


# ---------------------------------------------------------------------------
# 7) Backtest สาธิต (long-only, กัน look-ahead, หักต้นทุนทุกครั้ง)
# ---------------------------------------------------------------------------

def regime_series(bench_close: pd.Series, n: int = 200) -> pd.Series:
    """ดัชนี > SMA200 = 1 — เบรกลด drawdown ไม่ใช่เครื่องจับจังหวะกำไร"""
    b = bench_close.dropna()
    return (b > b.rolling(n).mean()).astype(int)


def sma_cross_backtest(close: pd.Series, fast: int, slow: int,
                       cost_side_pct: float, regime=None):
    px = close.dropna()
    if len(px) < slow + 30:
        return None
    f = px.rolling(fast).mean()
    s = px.rolling(slow).mean()
    raw = (f > s).astype(int)
    if regime is not None and len(regime):
        reg = regime.reindex(px.index).ffill().fillna(0).astype(int)
        raw = raw * reg
    sig = raw.shift(1).fillna(0).astype(int)   # ลงมือแท่งถัดไป
    ret = px.pct_change().fillna(0.0)
    turnover = sig.diff().abs()
    turnover.iloc[0] = abs(int(sig.iloc[0]))
    cost = cost_side_pct / 100.0
    daily = sig * ret - turnover * cost
    equity = (1.0 + daily).cumprod()

    v = sig.to_numpy()
    trades, entry_i = [], None
    for i in range(len(v)):
        prev = v[i - 1] if i > 0 else 0
        if v[i] == 1 and prev == 0:
            entry_i = i
        elif v[i] == 0 and prev == 1 and entry_i is not None:
            gross = float(px.iloc[i] / px.iloc[entry_i] - 1.0)
            trades.append({"วันเข้า": px.index[entry_i].date(),
                           "วันออก": px.index[i].date(),
                           "วันถือ": i - entry_i,
                           "กำไรก่อนต้นทุน %": gross * 100,
                           "กำไรสุทธิ %": (gross - 2 * cost) * 100,
                           "สถานะ": "ปิดแล้ว"})
            entry_i = None
    if entry_i is not None:
        gross = float(px.iloc[-1] / px.iloc[entry_i] - 1.0)
        trades.append({"วันเข้า": px.index[entry_i].date(), "วันออก": None,
                       "วันถือ": len(v) - 1 - entry_i,
                       "กำไรก่อนต้นทุน %": gross * 100,
                       "กำไรสุทธิ %": (gross - cost) * 100,
                       "สถานะ": "ยังถืออยู่"})
    return {"daily": daily, "equity": equity,
            "trades": pd.DataFrame(trades)}


def trade_stats(trades: pd.DataFrame) -> dict:
    """สถิติจาก 'เทรดที่ปิดแล้ว' เท่านั้น"""
    if trades is None or trades.empty:
        return {"n": 0}
    closed = trades[trades["สถานะ"] == "ปิดแล้ว"]
    n = len(closed)
    if n == 0:
        return {"n": 0}
    net = closed["กำไรสุทธิ %"].to_numpy(dtype=float)
    wins = int((net > 0).sum())
    gains = net[net > 0].sum()
    losses = -net[net < 0].sum()
    pf = float(gains / losses) if losses > 0 else float("inf")
    return {"n": n, "wins": wins, "win_rate": wins / n, "profit_factor": pf,
            "avg_win": float(net[net > 0].mean()) if wins else float("nan"),
            "avg_loss": float(net[net < 0].mean()) if wins < n else float("nan"),
            "expectancy": float(net.mean()),
            "avg_hold": float(closed["วันถือ"].mean())}


# ---------------------------------------------------------------------------
# 8) RRG (ภาพรวม ไม่ใช่สัญญาณ — สูตรประมาณ ไม่ใช่ JdK ต้นฉบับ)
# ---------------------------------------------------------------------------

def _roll_z(s: pd.Series, w: int) -> pd.Series:
    m = s.rolling(w).mean()
    sd = s.rolling(w).std()
    return (s - m) / sd


def compute_rrg(prices: dict, bench_close: pd.Series,
                w_ratio: int = 52, w_mom: int = 26, tail: int = 8) -> dict:
    bench_w = bench_close.dropna().resample("W-FRI").last().dropna()
    out = {}
    for t, df in prices.items():
        c = df["Close"].dropna().resample("W-FRI").last().dropna()
        rs = (100.0 * c / bench_w).dropna()
        if len(rs) < w_ratio + w_mom + tail:
            continue
        rr = 100.0 + _roll_z(rs, w_ratio)
        rm = 100.0 + _roll_z(rr.diff(), w_mom)
        d = pd.concat([rr.rename("x"), rm.rename("y")], axis=1).dropna().iloc[-tail:]
        if len(d) >= 2:
            out[t.replace(".BK", "")] = d
    return out


def quadrant_name(x: float, y: float) -> str:
    if x >= 100 and y >= 100:
        return "Leading (นำตลาด)"
    if x >= 100 and y < 100:
        return "Weakening (เริ่มแผ่ว)"
    if x < 100 and y < 100:
        return "Lagging (ตามหลัง)"
    return "Improving (กำลังฟื้น)"


# ---------------------------------------------------------------------------
# 9) ฤดูกาลระดับดัชนี: Turn-of-Month (จงใจไม่มีปฏิทินรายหุ้น — data-mining risk)
# ---------------------------------------------------------------------------

def tom_stats(bench_close: pd.Series, k_last: int = 1, k_first: int = 3) -> dict:
    px = bench_close.dropna()
    r = px.pct_change().dropna()
    if len(r) < 250:
        return {"ok": False}
    df = pd.DataFrame({"r": r.to_numpy()}, index=r.index)
    mkey = df.index.to_period("M")
    df["pos"] = df.groupby(mkey).cumcount()
    sizes = df.groupby(mkey)["r"].transform("size")
    df["from_end"] = sizes - df["pos"]
    is_tom = (df["pos"] < k_first) | (df["from_end"] <= k_last)
    tom_r = df.loc[is_tom, "r"]
    oth_r = df.loc[~is_tom, "r"]
    last_pos = int(df["pos"].iloc[-1])
    last_dom = int(df.index[-1].day)
    if last_pos < k_first:
        now_flag = "อยู่ช่วงต้นเดือน = ในหน้าต่าง TOM (โดยประมาณ)"
    elif last_dom >= 27:
        now_flag = "ใกล้สิ้นเดือน = กำลังเข้าหน้าต่าง TOM (โดยประมาณ)"
    else:
        now_flag = "อยู่นอกหน้าต่าง TOM"
    return {"ok": True,
            "tom_mean": float(tom_r.mean()), "tom_n": int(len(tom_r)),
            "oth_mean": float(oth_r.mean()), "oth_n": int(len(oth_r)),
            "diff_bps": float((tom_r.mean() - oth_r.mean()) * 1e4),
            "now_flag": now_flag,
            "window_text": f"วันทำการสุดท้าย {k_last} + วันแรก {k_first} ของเดือน"}
