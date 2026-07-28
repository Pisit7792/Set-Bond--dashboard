# -*- coding: utf-8 -*-
"""
SET × Bond Crisis Dashboard — Full Version (SET100 + Global context)

โครงสร้าง 3 โซน:
🇹🇭 SET (หุ้นไทย SET100): สแกนคะแนนรวม, กราฟ, RRG, Backtest+ตรวจสอบ, Scenario ไทย,
   TOM, ต้นทุน — พร้อม 'Global Overlay' จากฝั่ง Bond Crisis (กติกาเปิดเผย)
🌍 Global (Bond Crisis v3 เดิมครบ): composite เตือนภัย, 6 โมเดล, สัญญาณ,
   จำลองสถานการณ์, แบงก์รัน, ข่าว, ห้องประชุม AI, มหภาค, Trend โลก
📖 ร่วม: Trade Log & สถิติ, คู่มืออ่านค่า (รวมศัพท์ทุกฝั่ง), ข้อจำกัด & จุดพัฒนา

จุดยืน: ทุกตัวเลขมาพร้อมความไม่แน่นอน — overlay คือวินัยคุมความเสี่ยง
ไม่ใช่เครื่องจับจังหวะ และไม่ถูกผูกเข้า backtest (ยังไม่ผ่าน validation)
รัน: streamlit run app.py
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import bridge as BR
import countries as CO
import data_sources as D
import engine as E
import explain as X
import flows as FL
import gold as G
import meeting as MT
import model_history as MH
import models6 as M
import scenario as SC
import worldmon as WM
import crypto as CR
import set_context as CX
import set_engine as SE
import set_swing as SW
import signals as SG
import stock_meeting as SM
import multi_meeting as MM
import llm_providers as LP
import quant_evaluation as QE
import gold_council as GC
import pf_holdings as PF
import accum as ACC
import datastamp as DS

st.set_page_config(page_title="SET × Bond Crisis", page_icon="🛡️", layout="wide")

DISCLAIMER = ("เครื่องมือเพื่อการศึกษา/วินิจฉัยความเสี่ยง ไม่ใช่คำแนะนำการลงทุน — "
              "P(วิกฤต|สัญญาณเตือน) ≈ 50% ในวรรณกรรม EWS และ <3-5% ของรายย่อย"
              "ทำกำไรจากสัญญาณได้สม่ำเสมอ (Barber et al. 2020)")

ZONES = {
    "🇹🇭 SET (หุ้นไทย)": ["ภาพรวม SET + Overlay", "Fund Flow นักลงทุน",
                          "SET Swing v5.13 + Context", "💼 พอร์ตที่ถืออยู่",
                          "Scan หุ้น Overall", "สแกน Accum+Squeeze",
                          "AI Meeting หุ้น",
                          "สแกน SET100", "กราฟรายตัว",
                          "RRG", "Backtest + ตรวจสอบ", "Scenario ไทย",
                          "ฤดูกาล (TOM)", "ต้นทุนไทย",
                          "🔬 Self-Improve (ผลออฟไลน์)"],
    "🌍 Global (Bond Crisis)": ["ภาพรวมโลก", "โมเดลทำกำไร (6)", "สัญญาณ Global",
                                "จำลองสถานการณ์", "วิกฤตแบงก์รัน", "ข่าวสาร",
                                "ห้องประชุม AI", "ข้อมูลมหภาค", "รายประเทศ (Bond)",
                                "Trend สินทรัพย์โลก", "World Monitor",
                                "ทองคำ XAU (RTP v6.4.1)", "🥇 Gold Council",
                                "คริปโต (BTC/ETH)"],
    "📖 ร่วม": ["Trade Log & สถิติ", "คู่มืออ่านค่า", "ข้อจำกัด & จุดพัฒนา"],
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🛡️ SET × Bond Crisis")
zone = st.sidebar.radio("โซน", list(ZONES), label_visibility="collapsed")
page = st.sidebar.radio("หน้า", ZONES[zone], label_visibility="collapsed")
st.sidebar.divider()

st.sidebar.markdown("**🇹🇭 ฝั่ง SET**")
uni_text = st.sidebar.text_area(
    "รายชื่อ SET100 (แก้ได้ 1 ตัว/บรรทัด)", "\n".join(SE.SET100_H2_2026),
    height=140, help="เรียบเรียงรอบ 1 ก.ค.–31 ธ.ค. 2569 จากประกาศ ตลท. — "
    "ช่อง mid-cap บางตัวอาจคลาดเคลื่อน โปรดตรวจ set.or.th; ตัวที่ดึงไม่ได้ระบบข้ามเอง")
set_period = st.sidebar.selectbox("ช่วงข้อมูล SET", ["2y", "3y", "5y"], 0)
min_turn_m = st.sidebar.number_input("ประตูสภาพคล่อง (ลบ./วัน)", 1.0, 500.0, 20.0, 5.0)
c_comm = st.sidebar.number_input("คอมมิชชั่น %/ข้าง", 0.0, 1.0, 0.15, 0.01)
c_sprd = st.sidebar.number_input("Half-spread+slippage %/ข้าง", 0.0, 2.0, 0.15, 0.05,
                                 help="SET51-100 spread มักกว้างกว่า SET50 — ปรับขึ้นตามจริง")
n_trials = st.sidebar.number_input("ชุดพารามิเตอร์ที่เคยลอง (trials)", 1, 10000, 10,
                                   help="ใช้หักลด Sharpe (DSR) — ยิ่งลองมาก เกณฑ์ยิ่งสูง")
cost = SE.ThaiCost(commission_pct=c_comm, half_spread_pct=c_sprd)

st.sidebar.markdown("**🌍 ฝั่ง Global**")
gmode = st.sidebar.radio("แหล่งข้อมูล Global",
                         ["Live (FRED + yfinance)", "Demo (สังเคราะห์)", "ปิดฝั่ง Global"])
fred_key = ""
if gmode.startswith("Live"):
    fred_key = st.sidebar.text_input("FRED API key", type="password",
                                     help="ฟรี: fred.stlouisfed.org/docs/api/api_key.html")
if st.sidebar.button("🔄 โหลดข้อมูลใหม่ (ล้าง cache)"):
    st.cache_data.clear()
st.sidebar.caption(DISCLAIMER)

# ---------------------------------------------------------------------------
# Cached loaders (ห่อฟังก์ชัน pure ของ set_engine / data_sources)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="โหลดราคาหุ้นไทย SET100 (ครั้งแรก ~2 นาที)...")
def C_set_prices(tickers: tuple, period: str):
    return SE.load_universe_prices(tickers, period)


@st.cache_data(ttl=3600, show_spinner=False)
def C_set_bench(period: str):
    return SE.load_benchmark(period)


@st.cache_data(ttl=3600, show_spinner=False)
def C_single(ticker: str, period: str):
    return SE.load_single(ticker, period)


@st.cache_data(ttl=3600, show_spinner="ดึงข้อมูล FRED...")
def C_fred(key: str):
    return D.fetch_all_fred(key)


@st.cache_data(ttl=3600, show_spinner="ดึงราคาสินทรัพย์โลก...")
def C_global_prices():
    tickers = list(D.YF_ASSETS) + [D.YF_MOVE]
    px = D.fetch_yf_history(tickers)
    move = px[D.YF_MOVE].dropna() if D.YF_MOVE in px else pd.Series(dtype=float)
    assets = px[[c for c in px.columns if c != D.YF_MOVE]]
    return assets, move


FLOW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "Set_update.csv")


def load_flows():
    """อ่านไฟล์ flow ล่าสุดจากดิสก์ทุกครั้ง (ไฟล์เล็ก ~60KB, เปลี่ยนได้ระหว่างวัน)"""
    if not os.path.exists(FLOW_PATH):
        return pd.DataFrame(), ["ยังไม่มีไฟล์ Set_update.csv"]
    return FL.load_flow_csv(FLOW_PATH)


@st.cache_data(ttl=3600, show_spinner="ดึงราคาทองคำ + สินทรัพย์ที่ใช้กรอง...")
def C_gold_bundle(sym: str, period: str, need_carry: bool):
    def _close(d):
        if d is None or d.empty or "Close" not in d.columns:
            return None
        s = d["Close"].dropna()
        return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
    xau = SE.load_single(sym, period)
    dxy = _close(SE.load_single("DX-Y.NYB", period))
    jpy = _close(SE.load_single("USDJPY=X", period)) if need_carry else None
    vix = _close(SE.load_single("^VIX", period)) if need_carry else None
    return xau, dxy, jpy, vix


@st.cache_data(ttl=3600, show_spinner=False)
def C_ctx_bundle(period: str):
    def _cl(t):
        d = SE.load_single(t, period)
        if d.empty or "Close" not in d.columns:
            return None
        s = d["Close"].dropna()
        return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
    return _cl("^VIX"), _cl("^GSPC"), _cl("EEM")


@st.cache_data(ttl=3600, show_spinner="ดึงราคาคริปโต...")
def C_crypto(sym: str):
    return SE.load_single(sym, "10y")


# ---------------------------------------------------------------------------
# โหลด + คำนวณฝั่ง SET (ทำงานเสมอ — ต้องมีอินเทอร์เน็ต)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# v14 — ตราเวลาข้อมูล (ใช้ทุกแท็บ): บอกว่าที่เห็นบนจอสดแค่ไหน
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def C_stamp(tag: str, key: str) -> str:
    """เวลาที่ cache ของชุดข้อมูล (tag,key) ถูกสร้าง — TTL เท่ากับตัวข้อมูล

    ข้อจำกัดที่ไม่ปิดบัง: เป็น 'เวลาที่ cache ถูกสร้าง' ไม่ใช่เวลาที่ yfinance
    ตอบกลับเป๊ะ ๆ และถ้า cache สองชุดถูกล้างไม่พร้อมกันตัวเลขอาจคลาดได้
    ตัวที่เชื่อได้แน่นอนคือ 'แท่งล่าสุด' ซึ่งอ่านจากตัวข้อมูลเอง
    """
    return DS.now_th().isoformat()


DS_ITEMS: list = []


def stamp_add(name: str, data, market: str = "SET", tag: str = "",
              key: str = "", ttl: int = 3600, note: str = ""):
    """ลงทะเบียนชุดข้อมูลของหน้านี้ เพื่อไปโผล่ในแถบ 'ความสดของข้อมูล'"""
    from datetime import datetime as _dt
    loaded = None
    if tag:
        try:
            loaded = _dt.fromisoformat(C_stamp(tag, key))
        except Exception:
            loaded = None
    it = DS.describe(name, data, market, loaded, ttl, note)
    DS_ITEMS.append(it)
    return it


tickers = tuple(SE.to_yahoo(uni_text.splitlines()))
bench_sym, bench_label, bench_df = C_set_bench(set_period)
set_prices, set_failed = C_set_prices(tickers, set_period)
thb = C_single(SE.THB_TICKER, set_period)

if bench_df.empty or not set_prices:
    st.error("โหลดข้อมูลฝั่ง SET ไม่สำเร็จ (ดัชนี/หุ้น) — ตรวจอินเทอร์เน็ตแล้วกด "
             "'โหลดข้อมูลใหม่' | เครื่องต้องออกเน็ตได้เพราะฝั่ง SET ไม่มีโหมด demo")
    st.stop()

bench_close = bench_df["Close"].dropna()
if isinstance(bench_close, pd.DataFrame):
    bench_close = bench_close.iloc[:, 0]
set_regime = SE.regime_series(bench_close)

stamp_add(f"ราคาหุ้นไทย ({len(set_prices)} ตัว)", set_prices, "SET",
          "set_prices", f"{tickers}|{set_period}", 3600,
          "yfinance สิ้นวัน · auto_adjust=True (ปรับปันผล/แตกพาร์ย้อนหลัง)")
stamp_add(f"ดัชนีอ้างอิง {bench_label}", bench_df, "SET",
          "set_bench", str(set_period), 3600, f"สัญลักษณ์ {bench_sym}")
LAST_SET_BAR = DS.last_index(bench_close)
SET_BAR_CLOSED = DS.bar_is_closed(LAST_SET_BAR, "SET")
thb_close = thb["Close"].dropna() if ("Close" in thb.columns and len(thb)) \
    else pd.Series(dtype=float)
if isinstance(thb_close, pd.DataFrame):
    thb_close = thb_close.iloc[:, 0]

CTX_VIX, CTX_SPX, CTX_EEM = C_ctx_bundle(set_period)

# ---------------------------------------------------------------------------
# โหลด + คำนวณฝั่ง Global (ตามโหมด)
# ---------------------------------------------------------------------------
is_demo = gmode.startswith("Demo")
g_on = not gmode.startswith("ปิด")
fred, g_prices, move = {}, pd.DataFrame(), pd.Series(dtype=float)
if g_on:
    if is_demo:
        b = D.demo_bundle()
        fred, g_prices, move = b["fred"], b["prices"], b["move"]
    elif fred_key:
        fred = C_fred(fred_key)
        if fred.get("_errors"):
            st.error("FRED บาง series ดึงไม่ได้: "
                     + json.dumps(fred["_errors"], ensure_ascii=False))
        try:
            g_prices, move = C_global_prices()
        except Exception as e:
            st.warning(f"yfinance ฝั่งโลกใช้ไม่ได้ ({e}) — สัญญาณ/Trend โลกจะว่าง, "
                       "MOVE fallback เป็น VIX")
    else:
        g_on = False  # เลือก Live แต่ยังไม่ใส่ key → ฝั่ง Global ยังไม่ทำงาน

if g_on:
    if is_demo:
        stamp_add("Global (DEMO สังเคราะห์)", g_prices, "NONE", "", "", 0,
                  "⚠️ ตัวเลขสมมติทั้งหมด ไม่ใช่ข้อมูลตลาดจริง — ห้ามใช้ตัดสินใจ")
    else:
        stamp_add("มหภาค FRED", {k: v for k, v in fred.items()
                                 if isinstance(v, pd.Series)}, "NONE",
                  "fred", "live", 3600,
                  "FRED เผยแพร่ล่าช้าตามรอบของแต่ละ series (บางตัวรายเดือน)")
        stamp_add("ราคาสินทรัพย์โลก", g_prices, "US", "gprices", "live", 3600,
                  "yfinance สิ้นวัน — เวลาปิดตลาดสหรัฐฯ ตรงกับ ~03:00-04:00 น. ไทย")


def S(sid: str) -> pd.Series:
    s = fred.get(sid)
    return s if isinstance(s, pd.Series) else pd.Series(dtype=float)


composite = float("nan")
subs, warnings_g, mscores, deltas = {}, [], {}, {}
hist30 = pd.DataFrame()
sig_g = {"signals": [], "conflicts": [], "skipped": []}
spread_latest, rec_prob, inv_days = float("nan"), float("nan"), 0
vol_name, breadth_g = "-", float("nan")
if g_on:
    t10y3m = S("T10Y3M")
    spread_latest = float(t10y3m.iloc[-1]) if len(t10y3m) else float("nan")
    rec_prob = E.recession_probability(spread_latest)
    vol_series = move if len(move) > 100 else S("VIXCLS")
    vol_name = "MOVE" if len(move) > 100 else "VIX (fallback)"
    if len(g_prices) > 250:
        breadth_g = float((g_prices.iloc[-1] <
                           g_prices.rolling(200).mean().iloc[-1]).mean() * 100)
    subs = {"curve": rec_prob * 100 if rec_prob == rec_prob else float("nan"),
            "stress": E.percentile_of_latest(S("STLFSI4")),
            "credit": E.percentile_of_latest(S("BAMLH0A0HYM2")),
            "vol": E.percentile_of_latest(vol_series),
            "breadth": breadth_g}
    composite, _used = E.composite_crisis_score(subs)
    for v in reversed(t10y3m.dropna().values):
        if v < 0:
            inv_days += 1
        else:
            break
    warnings_g = E.build_warnings(composite, spread_latest, subs["stress"],
                                  E.zscore_of_latest(S("BAMLH0A0HYM2")),
                                  subs["vol"], inv_days)
    MDATA = dict(fred)
    MDATA["MOVE"] = move
    mscores = M.score_models(MDATA)

    @st.cache_data(ttl=3600, show_spinner="คำนวณประวัติคะแนนโมเดลโลก...")
    def C_deltas(cache_key: str) -> dict:
        out = {}
        for k in M.MODEL_DEFS:
            try:
                out[k] = round(M.model_delta(M.score_history(MDATA, k)), 1)
            except Exception:
                out[k] = float("nan")
        return out

    ckey = ("demo" if is_demo else "live") + str(
        t10y3m.index[-1].date() if len(t10y3m) else "")
    deltas = C_deltas(ckey)

    @st.cache_data(ttl=3600, show_spinner="คำนวณประวัติคะแนนรายวัน 30 วัน...")
    def C_hist30(cache_key: str) -> pd.DataFrame:
        try:
            return MH.all_daily_history(MDATA, tail_days=30)
        except Exception:
            return pd.DataFrame()

    hist30 = C_hist30(ckey)
    if len(g_prices):
        sig_g = SG.build_signals(mscores, M.ASSET_IMPACT, g_prices, D.YF_ASSETS)

overlay = BR.overlay_state(composite,
                           {k: m["score"] for k, m in mscores.items()}) \
    if g_on else BR.overlay_state(float("nan"), {})


def _score_key(k):
    v = mscores.get(k, {}).get("score", float("nan"))
    return -(v if v == v else -1.0)


MODEL_ORDER = sorted(M.MODEL_DEFS, key=_score_key) if mscores else list(M.MODEL_DEFS)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def pct(x, d=2):
    return "—" if x is None or not np.isfinite(x) else f"{x*100:.{d}f}%"


def num(x, d=2):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def plot(fig):
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.plotly_chart(fig)


def explain_box(title, body):
    with st.expander("ℹ️ " + title):
        st.markdown(body)


def overlay_card():
    lv = overlay["level"]
    box = st.error if lv == 2 else (st.warning if lv == 1 else
                                    (st.info if lv is None else st.success))
    box(f"**Global Overlay: {overlay['label']}** — {overlay['guidance']}\n\n"
        + "\n".join("• " + r for r in overlay["reasons"]))
    st.caption("⚠️ " + overlay["disclaimer"])


# ===========================================================================
# 🇹🇭 SET pages
# ===========================================================================

def page_set_overview():
    reg_on = bool(set_regime.iloc[-1] == 1)
    sma200 = bench_close.rolling(200).mean()
    dist = bench_close.iloc[-1] / sma200.iloc[-1] - 1
    vol20 = bench_close.pct_change().tail(20).std() * np.sqrt(252)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ดัชนีอ้างอิง (" + (bench_label or "-") + ")",
              f"{bench_close.iloc[-1]:,.2f}",
              f"{bench_close.pct_change().iloc[-1]*100:+.2f}% วันล่าสุด")
    c2.metric("Regime SET", "Risk-On 🟢" if reg_on else "Risk-Off 🔴",
              f"ห่าง SMA200 {dist*100:+.1f}%")
    c3.metric("ผันผวน 20 วัน (ต่อปี)", pct(vol20, 1))
    c4.metric("หุ้นในจอ", f"{len(set_prices)}/{len(tickers)}",
              f"ข้าม {len(set_failed)}")
    if set_failed:
        st.caption("ดึงไม่ได้ (ข้าม): "
                   + ", ".join(s.replace(".BK", "") for s in set_failed))
    if bench_sym != "^SET.BK":
        st.info(f"ใช้ {bench_label} แทนดัชนี SET (ดึง ^SET.BK ไม่ได้)")

    st.markdown("#### 🌍→🇹🇭 บริบทโลก (Overlay)")
    overlay_card()
    tctx = BR.thb_context(thb_close)
    if tctx.get("ok"):
        d1, d2, d3 = st.columns(3)
        d1.metric("USD/THB", f"{tctx['last']:.2f}")
        d2.metric("บาทเปลี่ยน 1 เดือน", f"{tctx['chg_1m']:+.1f}%",
                  "ค่าบวก = บาทอ่อน")
        d3.metric("3 เดือน", f"{tctx['chg_3m']:+.1f}%")
        st.caption("💱 " + tctx["text"])
    _f, _fi = load_flows()
    if len(_f):
        _f20 = float(_f["Foreign"].tail(20).sum())
        _stk = FL.streak(_f["Foreign"])
        _dir = "ซื้อ" if _stk > 0 else "ขาย"
        st.caption(f"🌊 ต่างชาติสะสม 20 วัน: **{_f20:+,.0f} ลบ.** "
                   f"({_dir}สุทธิติดกัน {abs(_stk)} วัน, ถึง {_f.index[-1]:%d/%m}) "
                   "— รายละเอียดที่หน้า Fund Flow")
    with st.expander("🔍 ช่องทางส่งผ่าน Global → SET (เกรดหลักฐานกำกับ)"):
        st.dataframe(pd.DataFrame(BR.GLOBAL_TO_SET), hide_index=True)
        st.caption("หลักฐานหลัก: VIX พยากรณ์เงินไหลเข้า SET เป็นลบ / วัฏจักร Fed-"
                   "ดอลลาร์ ↔ flow EM — ระดับ 'ปานกลาง' ทั้งสิ้น ไม่ใช่กฎตายตัว")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bench_close.index, y=bench_close, name="ดัชนี",
                             line=dict(width=1.6)))
    fig.add_trace(go.Scatter(x=sma200.index, y=sma200, name="SMA200",
                             line=dict(width=1.2, dash="dot")))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h"))
    plot(fig)
    st.info("**ข้อมูลที่จอนี้ไม่มี (ตรงๆ):** ยอดซื้อขายสุทธิต่างชาติ/NVDR รายวัน "
            "ไม่มีในแหล่งฟรี — ดูที่เว็บ SET (Investor Type) แล้วใช้เป็นบริบท: "
            "สถิติชี้เงินไหลเข้า SET มักอยู่แค่ 1-2 เดือน ไม่ใช่เทรนด์ยาว")


def page_set_scan():
    st.subheader("สแกน SET100 — จัดลำดับ 'น่าศึกษาต่อ' (ไม่ใช่คำสั่งซื้อ)")
    if overlay["level"] in (1, 2):
        overlay_card()
    board = SE.build_scoreboard(set_prices, bench_close, min_turn_m * 1e6)
    only_liq = st.checkbox("เฉพาะหุ้นที่ผ่านประตูสภาพคล่อง", True)
    show = board[board["liq_pass"]] if only_liq else board
    st.caption(f"ผ่านประตูสภาพคล่อง {int(board['liq_pass'].sum())}/{len(board)} ตัว "
               f"(เกณฑ์ {min_turn_m:.0f} ลบ./วัน) — SET51-100 หลายตัวไม่ผ่านเป็นเรื่องปกติ "
               "และคือเหตุผลที่ประตูนี้มีอยู่")
    disp = show[["ticker", "close", "composite", "z_trend", "z_mom", "z_resmom",
                 "z_lowvol", "n_factors", "turnover_thb", "liq_pass"]].copy()
    disp["turnover_thb"] = disp["turnover_thb"] / 1e6
    st.dataframe(
        disp, height=500, hide_index=True,
        column_config={
            "ticker": "หุ้น",
            "close": st.column_config.NumberColumn("ราคา", format="%.2f"),
            "composite": st.column_config.NumberColumn("คะแนนรวม", format="%.2f"),
            "z_trend": st.column_config.NumberColumn("z แนวโน้ม", format="%.2f"),
            "z_mom": st.column_config.NumberColumn("z โมเมนตัม", format="%.2f"),
            "z_resmom": st.column_config.NumberColumn("z Residual", format="%.2f"),
            "z_lowvol": st.column_config.NumberColumn("z ผันผวนต่ำ", format="%.2f"),
            "n_factors": st.column_config.NumberColumn("ปัจจัยใช้ได้"),
            "turnover_thb": st.column_config.NumberColumn("มูลค่าซื้อขาย (ลบ.)",
                                                          format="%.1f"),
            "liq_pass": st.column_config.CheckboxColumn("ผ่านสภาพคล่อง"),
        })
    explain_box(
        "ปัจจัย 4 ตัวคำนวณอย่างไร และทำไมน้ำหนักเท่ากัน",
        "\n".join(f"- **{a}** — {b}" for a, b in
                  (SE.FACTOR_EXPLAIN[k] for k in ["trend", "mom", "resmom", "vol"]))
        + "\n\n**น้ำหนักเท่ากันโดยเจตนา:** งานวิจัย forecast combination 50+ ปี "
        "พบว่า simple average มักชนะ optimal weights (การ optimize ไป fit "
        "เสียงรบกวน)\n\n**หุ้นใหม่ (MRDIYT, THAI ฯลฯ):** ประวัติสั้น → ปัจจัย"
        "เป็น 'ไม่มีข้อมูล' และไม่คิดคะแนนรวม — พฤติกรรมถูกต้อง ไม่ใช่บั๊ก")
    st.caption("⚠️ คะแนนสูง = ควรเปิดงบ/เช็กลิสต์พื้นฐานของตัวนั้นก่อน — "
               "ไม่มีคะแนนใดแทนการบ้านได้")


def page_set_chart():
    from plotly.subplots import make_subplots
    names = sorted(set_prices.keys())
    pick = st.selectbox("เลือกหุ้น", names,
                        format_func=lambda s: s.replace(".BK", ""))
    tkr = pick.replace(".BK", "")
    d = set_prices[pick]
    swp = SW.SwingParams()
    fr = SW.compute_frame(d, bench_close, tkr, swp)
    r = fr.iloc[-1]
    reg = "UP 🟢" if r["regime_up"] else ("DOWN 🔴" if r["regime_dn"] else "FLAT ⚪")
    dist_t = ((r["swing_hi"] - r["Close"]) / r["atr"]) if r["atr"] > 0 else float("nan")
    st.caption(f"มุมมองแบบสคริปต์ SET Swing v5.13 — Regime {reg} · ConfL "
               f"{int(r['conf_l'])}/100 (thr 55) · trigger {r['swing_hi']:.2f} "
               f"(ห่าง {dist_t:.1f} ATR) · stop อ้างอิง {r['sl_dist']:.2f} บาท (2 ATR)")
    show = fr.tail(300)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.62, 0.20, 0.18], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=show.index, open=show["Open"],
                                 high=show["High"], low=show["Low"],
                                 close=show["Close"], name="ราคา"), row=1, col=1)
    fig.add_trace(go.Scatter(x=show.index, y=show["sma_r"], name="Regime SMA200",
                             line=dict(width=2, color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=show.index, y=show["swing_hi"], name="แนวเบรก 20",
                             line=dict(width=1, color="purple", shape="hv")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=show.index, y=show["swing_lo"], name="แนวหลุด 20",
                             line=dict(width=1, color="maroon", dash="dot",
                                       shape="hv")), row=1, col=1)
    chand = show["High"].rolling(swp.tr_len).max() - show["atr"] * swp.tr_mlt
    fig.add_trace(go.Scatter(x=show.index, y=chand,
                             name="Chandelier 22/3 (อ้างอิง)",
                             line=dict(width=1, color="#888", dash="dash")),
                  row=1, col=1)
    ptsL = show[show["long_cond"]]
    if len(ptsL):
        fig.add_trace(go.Scatter(x=ptsL.index, y=ptsL["Low"] * 0.995,
                                 mode="markers", name="สัญญาณ L",
                                 marker=dict(symbol="triangle-up", size=10,
                                             color="#2c7")), row=1, col=1)
    ptsS = show[show["short_cond"]]
    if len(ptsS):
        fig.add_trace(go.Scatter(x=ptsS.index, y=ptsS["High"] * 1.005,
                                 mode="markers", name="สัญญาณ S",
                                 marker=dict(symbol="triangle-down", size=10,
                                             color="#e55")), row=1, col=1)
    fig.add_trace(go.Scatter(x=show.index, y=show["conf_l"], name="Confluence L",
                             line=dict(width=1.2)), row=2, col=1)
    fig.add_hline(y=swp.conf_min, line_dash="dot", line_width=1, row=2, col=1)
    if "Volume" in show.columns:
        fig.add_trace(go.Bar(x=show.index, y=show["Volume"], name="ปริมาณ",
                             opacity=0.5), row=3, col=1)
    fig.update_layout(height=640, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h"))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    plot(fig)
    explain_box("อ่านจอนี้แบบ v5.13",
                "เส้นส้ม = SMA200 (regime) | เส้นม่วง/แดง hv = แนวเบรก/หลุด 20 แท่ง "
                "(BOS) | เส้นเทา dash = Chandelier 22/3 อ้างอิง (ตอนถือจริง ratchet "
                "ขึ้นอย่างเดียวจากจุดเข้า) | ▲/▼ = แท่งที่เงื่อนไขครบ → ลงมือ *open "
                "แท่งถัดไป* เสมอ | แถวกลาง Confluence L (mom35+ER25+vol20+ใกล้ high "
                "252 = 20) เทียบเส้น 55")

def page_set_rrg():
    st.subheader("RRG — แผนที่ความแข็งแรงเทียบดัชนี (ภาพรวม ไม่ใช่สัญญาณ)")
    st.warning("สูตรเป็นการประมาณ (ไม่ใช่ JdK ต้นฉบับซึ่งเป็นกรรมสิทธิ์) และไม่มี "
               "peer-review ว่า RRG เดี่ยวๆ ทำกำไร — ใช้เป็น situational awareness")
    picks = st.multiselect("เลือกหุ้น (≤ 12 อ่านง่าย)", sorted(set_prices.keys()),
                           default=_overall_default(),
                           format_func=lambda s: s.replace(".BK", ""))
    pts = SE.compute_rrg({t: set_prices[t] for t in picks}, bench_close)
    if not pts:
        st.info("ข้อมูลรายสัปดาห์ไม่พอ — เพิ่มช่วงข้อมูลเป็น 3y/5y")
        return
    fig = go.Figure()
    for x0, x1, y0, y1, colr in [(100, 110, 100, 110, "rgba(60,180,120,0.10)"),
                                 (100, 110, 90, 100, "rgba(230,200,80,0.08)"),
                                 (90, 100, 90, 100, "rgba(220,90,90,0.08)"),
                                 (90, 100, 100, 110, "rgba(90,140,220,0.08)")]:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=colr, line_width=0)
    fig.add_hline(y=100, line_width=1, line_color="#888")
    fig.add_vline(x=100, line_width=1, line_color="#888")
    for t, dd in pts.items():
        fig.add_trace(go.Scatter(x=dd["x"], y=dd["y"], mode="lines+markers",
                                 name=t, line=dict(width=1.4),
                                 marker=dict(size=[5]*(len(dd)-1)+[11])))
        fig.add_annotation(x=float(dd["x"].iloc[-1]), y=float(dd["y"].iloc[-1]),
                           text=t, showarrow=False, yshift=12, font=dict(size=11))
    for tx, ty, lb in [(108.6, 108.8, "LEADING"), (108.6, 91.2, "WEAKENING"),
                       (91.4, 91.2, "LAGGING"), (91.4, 108.8, "IMPROVING")]:
        fig.add_annotation(x=tx, y=ty, text=lb, showarrow=False,
                           font=dict(size=11, color="#9aa"))
    fig.update_layout(height=540, xaxis=dict(range=[90, 110]),
                      yaxis=dict(range=[90, 110]),
                      xaxis_title="RS-Ratio (แข็งแรงเทียบดัชนี)",
                      yaxis_title="RS-Momentum (ความเร่ง)",
                      margin=dict(l=10, r=10, t=30, b=10))
    plot(fig)
    st.dataframe(pd.DataFrame(
        [{"หุ้น": t, "ควอดรันต์": SE.quadrant_name(float(dd["x"].iloc[-1]),
                                                    float(dd["y"].iloc[-1]))}
         for t, dd in pts.items()]), hide_index=True)


def page_set_backtest():
    st.subheader("Backtest สาธิต + ชั้นตรวจสอบความน่าเชื่อถือ")
    st.info("SMA cross เป็นกลยุทธ์ **สาธิตกระบวนการตรวจสอบ** — ไม่ใช่กลยุทธ์แนะนำ "
            "| Overlay จาก Global *ไม่ถูกผูกเข้า backtest* โดยเจตนา "
            "(ยังไม่ผ่าน validation — ความซื่อสัตย์มาก่อนฟีเจอร์)")
    cA, cB, cC, cD = st.columns(4)
    pick = cA.selectbox("หุ้น", sorted(set_prices.keys()),
                        format_func=lambda s: s.replace(".BK", ""), key="btp")
    bt_period = cB.selectbox("ช่วงทดสอบ", ["3y", "5y", "10y", "max"], 1)
    fast = cC.slider("SMA เร็ว", 5, 60, 20)
    slow = cD.slider("SMA ช้า", 30, 250, 50)
    use_reg = st.checkbox("Regime filter (ถือเฉพาะดัชนี > SMA200)", True)
    if slow <= fast:
        st.error("SMA ช้าต้องมากกว่า SMA เร็ว")
        return
    hist = C_single(pick, bt_period)
    if hist.empty or len(hist) < slow + 60:
        st.error("ข้อมูลไม่พอสำหรับช่วง/พารามิเตอร์นี้")
        return
    reg = None
    if use_reg:
        bl = C_single(bench_sym, bt_period)
        if not bl.empty:
            bc = bl["Close"].dropna()
            if isinstance(bc, pd.DataFrame):
                bc = bc.iloc[:, 0]
            reg = SE.regime_series(bc)
    res = SE.sma_cross_backtest(hist["Close"].dropna(), fast, slow,
                                cost.per_side_pct(), reg)
    if res is None:
        st.error("รันไม่สำเร็จ — ลองช่วงข้อมูลยาวขึ้น")
        return
    stats = SE.trade_stats(res["trades"])
    n = stats.get("n", 0)
    level, verdict = SE.sample_verdict(n)
    {"fail": st.error, "warn": st.warning, "ok": st.success}[level]("🧮 " + verdict)
    if n == 0:
        st.warning("ช่วง/พารามิเตอร์นี้ไม่เกิดเทรดที่ปิดเลย")
        return
    lo, hi = E.wilson_ci(stats["wins"], n)
    eq = res["equity"]
    yrs = len(eq) / 252
    cagr = (float(eq.iloc[-1]) ** (1 / yrs) - 1
            if yrs > 0.5 and eq.iloc[-1] > 0 else float("nan"))
    sd_d = res["daily"].std(ddof=1)
    shp = (float(res["daily"].mean() / sd_d * math.sqrt(252))
           if sd_d > 0 else float("nan"))
    p_ = E.probabilistic_sharpe_ratio(res["daily"])
    d_ = SE.deflated_sharpe(res["daily"], int(n_trials))
    pf = stats["profit_factor"]
    pf_txt = ("∞ (ยังไม่มีไม้ขาดทุน — อย่าเพิ่งเชื่อ)"
              if not np.isfinite(pf) else f"{pf:.2f}")
    m = st.columns(4)
    m[0].metric("เทรดปิดแล้ว", f"{n}")
    m[1].metric("Win rate", pct(stats["win_rate"], 1),
                f"95% CI: {lo*100:.0f}–{hi*100:.0f}%")
    m[2].metric("Profit Factor", pf_txt)
    m[3].metric("ค่าคาดหวัง/เทรด", f"{stats['expectancy']:+.2f}%")
    m = st.columns(4)
    m[0].metric("กำไรรวม (หลังต้นทุน)", pct(float(eq.iloc[-1]) - 1, 1))
    m[1].metric("CAGR", pct(cagr, 1))
    m[2].metric("Max Drawdown", pct(SE.max_drawdown(eq), 1))
    m[3].metric("Sharpe/ปี", num(shp))
    m = st.columns(4)
    m[0].metric("PSR", num(p_), "อยากเห็น ≥ 0.95")
    m[1].metric(f"DSR (trials={int(n_trials)})", num(d_), "อยากเห็น ≥ 0.95")
    m[2].metric("เฉลี่ยไม้ชนะ", f"{stats['avg_win']:+.2f}%"
                if np.isfinite(stats["avg_win"]) else "—")
    m[3].metric("เฉลี่ยไม้แพ้", f"{stats['avg_loss']:+.2f}%"
                if np.isfinite(stats["avg_loss"]) else "—")
    figE = go.Figure(go.Scatter(x=eq.index, y=eq, name="Equity"))
    figE.update_layout(height=280, title="เส้นทุนสะสม (หลังหักต้นทุนทุกเทรด)",
                       margin=dict(l=10, r=10, t=40, b=10))
    plot(figE)
    with st.expander("📋 รายการเทรด"):
        st.dataframe(res["trades"], hide_index=True)
    explain_box(
        "ตัวเลขแต่ละตัวแปลว่าอะไร",
        f"- **ต้นทุนที่หัก:** {cost.per_side_pct():.3f}%/ข้าง "
        f"(ไป-กลับ {cost.round_trip_pct():.3f}%)\n"
        "- **กัน look-ahead:** สัญญาณจากปิดวันนี้ → ลงมือปิด *วันถัดไป*\n"
        "- **Wilson CI:** ค่าจริงของ win rate น่าจะอยู่ในช่วงนี้ — กว้าง = ยังสรุปไม่ได้\n"
        "- **PSR/DSR:** ความน่าจะเป็นว่า 'มีของจริง ไม่ใช่โชค' — DSR หักค่าฟลุค"
        "จากการลองหลายชุด\n"
        "- **Survivorship bias:** ใช้รายชื่อปัจจุบันย้อนอดีต → ผลจริงมักแย่กว่านี้\n"
        "- **ขั้นกว่า (ยังไม่ทำ):** walk-forward / purged CV — จำเป็นก่อนใช้เงินจริง")


def page_set_scenario():
    st.subheader("Scenario ไทย — ทิศทาง ไม่ใช่ตัวเลขปลอม")
    st.dataframe(pd.DataFrame(BR.SET_SCENARIOS), hide_index=True)
    st.warning("⚠️ " + BR.SET_SCENARIO_DISCLAIMER)
    st.caption("ฝั่ง Global มี slider เชิงตัวเลข (หน้า 'จำลองสถานการณ์') เพราะมี"
               "เมทริกซ์ความไวที่ประกาศเป็นสมมติฐานเปิดเผย — ฝั่งไทยยังไม่มีเมทริกซ์"
               "ที่ซื่อสัตย์พอ จึงให้เฉพาะทิศทาง")


def page_set_tom():
    st.subheader("ฤดูกาลระดับดัชนี: Turn-of-Month")
    ts = SE.tom_stats(bench_close)
    if not ts.get("ok"):
        st.info("ข้อมูลไม่พอ (ต้องการ ≥ 1 ปี)")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("เฉลี่ย/วัน ช่วง TOM", f"{ts['tom_mean']*1e4:+.1f} bps",
              f"n={ts['tom_n']} วัน")
    c2.metric("วันอื่นๆ", f"{ts['oth_mean']*1e4:+.1f} bps", f"n={ts['oth_n']} วัน")
    c3.metric("ส่วนต่าง", f"{ts['diff_bps']:+.1f} bps/วัน")
    st.info(f"🗓️ {ts['now_flag']} · หน้าต่าง: {ts['window_text']}")
    explain_box("หลักฐาน และสิ่งที่จอนี้จงใจไม่ทำ",
                "- Panyagometh (2016): ผลตอบแทนส่วนเกิน SET กระจุกช่วง TOM — "
                "หลักฐานปานกลาง ระดับดัชนี ใช้เป็น tilt เล็กๆ\n"
                "- ส่วนต่างไม่กี่ bps/วัน **แพ้ต้นทุนเทรดได้ง่าย** — "
                "ไม่ใช่เหตุผลซื้อโดยลำพัง\n"
                "- **จงใจไม่มีปฏิทินซื้อรายหุ้น-รายวัน:** เลือก วันที่×หุ้น จากอดีต "
                "= data-mining ชั้นดี")


def page_set_cost():
    st.subheader("ต้นทุนการเทรดหุ้นไทย — ด่านแรกของทุกระบบ")
    st.dataframe(pd.DataFrame(
        [{"ส่วนประกอบ": a, "%/ข้าง": f"{b:.4f}", "คำอธิบาย": c_}
         for a, b, c_ in cost.breakdown()]), hide_index=True)
    c1, c2 = st.columns(2)
    c1.metric("รวมต่อข้าง", f"{cost.per_side_pct():.3f}%")
    c2.metric("ไป-กลับ (จุดคุ้มทุน)", f"{cost.round_trip_pct():.3f}%")
    thb_size = st.number_input("ขนาดไม้ (บาท)", 1000.0, 1e9, 100000.0, 10000.0)
    st.write(f"ไม้ละ {thb_size:,.0f} บาท → ต้นทุนไป-กลับ ≈ "
             f"**{thb_size*cost.round_trip_pct()/100:,.0f} บาท/รอบ** — "
             f"เดือนละ 10 รอบ = {10*thb_size*cost.round_trip_pct()/100:,.0f} บาท "
             "ที่ต้องชนะก่อนถึงจะ 'เริ่ม' กำไร")
    st.warning("SET51-100 spread มักกว้างกว่า SET50 — ถ้าเทรดกลุ่มนี้ ปรับช่อง "
               "half-spread ในแถบข้างขึ้นตามจริง แล้วดูจุดคุ้มทุนใหม่")


# ===========================================================================
# 🌍 Global pages (Bond Crisis v3)
# ===========================================================================

def need_global() -> bool:
    if not g_on:
        st.info("ฝั่ง Global ยังไม่ทำงาน — เลือกโหมด **Live + ใส่ FRED key** "
                "(ฟรี) หรือ **Demo** ในแถบข้าง | ฝั่ง SET ใช้งานได้ตามปกติ")
        return True
    if is_demo:
        st.warning("⚠️ **DEMO DATA (สังเคราะห์)** — ห้ามใช้ตัดสินใจใดๆ", icon="⚠️")
    return False


def model_bar():
    keys = [k for k in MODEL_ORDER][::-1]
    vals = [mscores[k]["score"] for k in keys]
    ths = [mscores[k]["th"] for k in keys]
    f = go.Figure(go.Bar(x=vals, y=ths, orientation="h",
                         text=[f"{v:.1f}" for v in vals], textposition="outside"))
    f.update_layout(height=300, xaxis_range=[0, 100],
                    margin=dict(l=10, r=40, t=10, b=10))
    return f


def page_g_overview():
    if need_global():
        return
    st.markdown("#### 📋 บทสรุปภาษาคน")
    st.info(X.plain_summary(composite, subs, spread_latest, rec_prob,
                            inv_days, len(warnings_g), is_demo, vol_name))
    st.caption("บทสรุปเป็น rule-based (ไม่ใช่ AI แต่ง) — อ้างเฉพาะตัวเลขที่คำนวณจริง")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Crisis Score", f"{composite:.0f}/100"
                  if composite == composite else "n/a")
        for w in warnings_g:
            st.write({1: "🟥", 2: "🟧", 3: "🟨"}[w["tier"]] + " " + w["msg"])
        if not warnings_g:
            st.success("ไม่มีเตือนเข้าเกณฑ์ (ความเงียบ ≠ ปลอดภัย)")
        st.metric("สัญญาณ Global เข้าเกณฑ์", len(sig_g["signals"]),
                  f"ขัดแย้ง {len(sig_g['conflicts'])} | กรองออก {len(sig_g['skipped'])}")
        st.markdown("**ผลต่อฝั่ง SET:**")
        overlay_card()
    with c2:
        st.markdown("**6 โมเดล (คะแนนสภาพแวดล้อม 0-100)**")
        plot(model_bar())
        st.caption("Δ สัปดาห์: " + ", ".join(
            f"{mscores[k]['th']} {deltas.get(k, float('nan')):+.1f}"
            for k in MODEL_ORDER))


def page_g_models():
    if need_global():
        return
    # --- ประวัติคะแนนโมเดล (30 วัน): เส้นคำนวณย้อนหลัง + สมุดบันทึกสด ---
    st.subheader("📈 ประวัติคะแนนโมเดล (30 วัน)")
    if hist30.empty:
        st.info("ยังคำนวณเส้นย้อนหลังไม่ได้ (ข้อมูลไม่พอ/ดึงไม่สำเร็จ)")
    else:
        fig = go.Figure()
        for k in [c for c in M.MODEL_DEFS if c in hist30.columns]:
            fig.add_trace(go.Scatter(x=hist30.index, y=hist30[k],
                                     mode="lines", name=M.MODEL_DEFS[k]["th"]))
        fig.add_hline(y=SG.SIGNAL_THRESHOLD, line_dash="dash", line_color="#888",
                      annotation_text=f"เกณฑ์สัญญาณ ({SG.SIGNAL_THRESHOLD:.0f})",
                      annotation_position="top left")
        fig.add_hline(y=BR.CAUTION_MODEL, line_dash="dot", line_color="#e6a23c",
                      annotation_text=f"overlay ระวัง — โมเดลเสี่ยง ({BR.CAUTION_MODEL:.0f})",
                      annotation_position="bottom left")
        fig.add_hline(y=BR.RISKOFF_MODEL, line_dash="dot", line_color="#e05252",
                      annotation_text=f"overlay ตึงตัว ({BR.RISKOFF_MODEL:.0f})",
                      annotation_position="top right")
        fig.update_layout(height=380, yaxis_range=[0, 100],
                          legend=dict(orientation="h", y=-0.25),
                          margin=dict(l=10, r=10, t=30, b=10))
        plot(fig)
        st.caption("⚠️ " + MH.RECOMPUTE_CAVEAT)
        explain_box("เส้นนี้คำนวณอย่างไร / ต่างจาก Δสัปดาห์ตรงไหน",
                    "ณ วันทำการ t คะแนน = ค่าเฉลี่ย percentile ของ component ที่มี "
                    "โดยแต่ละ component เทียบกับ **ประวัติของตัวเองถึงวัน t เท่านั้น** "
                    "(expanding) — จุดขวาสุดจึงเท่ากับคะแนนในการ์ดด้านล่างวันนี้\n\n"
                    "ส่วน Δสัปดาห์ในการ์ดใช้อนุกรมรายสัปดาห์ (W-FRI) ของ models6 — "
                    "คนละความถี่ ตัวเลขจึงต่างกันได้เล็กน้อย ไม่ใช่บั๊ก")
    with st.expander("💾 สมุดบันทึกสด (track record จริง — สะสมวันต่อวัน)"):
        st.caption(MH.SNAP_CAVEAT)
        snap_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 MH.SNAP_FILE)
        sdf, sprobs = MH.load_snapshots(snap_path)
        for p_ in sprobs:
            st.caption("ℹ️ " + p_)
        mode_now = "demo" if is_demo else "live"
        today_d = datetime.now().date()
        if st.button(f"📌 บันทึกคะแนนวันนี้ ({today_d}) — โหมด {mode_now.upper()}"):
            scores_now = {k: mscores.get(k, {}).get("score", float("nan"))
                          for k in M.MODEL_DEFS}
            sdf2, okb, msgb = MH.append_snapshot(sdf, today_d, scores_now,
                                                 composite, mode_now)
            if okb:
                try:
                    with open(snap_path, "wb") as f_:
                        f_.write(MH.to_csv_bytes(sdf2))
                    sdf = sdf2
                    st.success("✅ " + msgb)
                except Exception as e_:
                    st.error(f"เขียนไฟล์ไม่ได้: {e_}")
            else:
                st.error("⛔ " + msgb)
        if len(sdf):
            show = sdf.copy()
            show.columns = (["วันที่"] + [M.MODEL_DEFS[k]["th"] for k in M.MODEL_DEFS]
                            + ["composite", "โหมด"])
            st.dataframe(show.tail(30), hide_index=True)
            if len(sdf) >= 2:
                fig2 = go.Figure()
                for k in M.MODEL_DEFS:
                    fig2.add_trace(go.Scatter(
                        x=pd.to_datetime(sdf["date"]), y=sdf[k],
                        mode="lines+markers", name=M.MODEL_DEFS[k]["th"]))
                fig2.update_layout(height=300, yaxis_range=[0, 100],
                                   legend=dict(orientation="h", y=-0.3),
                                   margin=dict(l=10, r=10, t=10, b=10))
                plot(fig2)
            st.download_button("⬇️ ดาวน์โหลด " + MH.SNAP_FILE,
                               MH.to_csv_bytes(sdf), file_name=MH.SNAP_FILE,
                               mime="text/csv")
            st.markdown("**☁️ เก็บถาวรขึ้น GitHub** (แบบเดียวกับหน้า Fund Flow)")
            g1_, g2_ = st.columns(2)
            gh_owner = g1_.text_input("Owner", "Pisit7792", key="mh_owner")
            gh_repo = g2_.text_input("Repo", "Set-Bond--dashboard", key="mh_repo")
            g3_, g4_ = st.columns(2)
            gh_branch = g3_.text_input("Branch", "main", key="mh_branch")
            gh_path = g4_.text_input("Path ไฟล์ใน repo", MH.SNAP_FILE, key="mh_path")
            gh_tok = st.text_input("GitHub token", type="password", key="mh_tok")
            if st.button("🚀 Commit ไฟล์บันทึกขึ้น GitHub"):
                okc, msgc = MH.github_put_file(
                    gh_owner, gh_repo, gh_branch, gh_path,
                    MH.to_csv_bytes(sdf), gh_tok,
                    f"update {MH.SNAP_FILE} ({today_d})")
                if okc:
                    st.success("✅ Commit สำเร็จ"
                               + (f" — [ดู commit]({msgc})" if msgc else ""))
                else:
                    st.error(msgc)
        else:
            st.caption("ยังไม่มีบันทึก — กดปุ่มด้านบนเพื่อเริ่ม track record วันแรก")
    st.divider()
    st.caption("คะแนนจากสูตรเปิดเผย (percentile ตัวชี้วัดจริง) — ตารางได้/เสีย"
               "ประโยชน์คือแนวโน้มอดีต *ไม่ใช่กฎตายตัว*")
    for k in MODEL_ORDER:
        m = mscores[k]
        imp = M.ASSET_IMPACT[k]
        d = deltas.get(k, float("nan"))
        with st.expander(f"{m['th']} — {m['score']:.1f}/100"
                         + (f"  (Δสัปดาห์ {d:+.1f})" if d == d else ""),
                         expanded=(k == MODEL_ORDER[0])):
            st.write(m["desc"])
            cols = st.columns(3)
            cols[0].markdown("**ส่วนประกอบ (percentile)**")
            for name, v in m["components"].items():
                cols[0].write(f"- {name}: {v:.0f}")
            for miss in m["missing"]:
                cols[0].write(f"- {miss}: ⚠️ ไม่มีข้อมูล")
            cols[1].markdown("**ได้ประโยชน์ (อดีต)**")
            for a in imp["benefit"]:
                cols[1].write("🟢 " + a)
            cols[2].markdown("**เสียประโยชน์ (อดีต)**")
            for a in imp["lose"]:
                cols[2].write("🔴 " + a)
            st.warning("ข้อจำกัด: " + imp["note"])
    st.caption(f"เกณฑ์สัญญาณ: ≥ {SG.SIGNAL_THRESHOLD:.0f} | "
               f"เกณฑ์เรียกประชุม: Δ ≥ {MT.DELTA_TRIGGER_PTS:.0f} จุด/สัปดาห์")


def page_g_signals():
    if need_global():
        return
    st.error(SG.DISCLAIMER, icon="⚠️")
    st.caption(f"กติกาประกาศล่วงหน้า: โมเดล ≥ {SG.SIGNAL_THRESHOLD:.0f} → ตารางได้/เสีย "
               f"| LONG เหนือ / SHORT ใต้ 200DMA | SL {SG.SL_ATR:.0f}×ATR, "
               f"TP {SG.TP_ATR:.0f}×ATR (R:R 1:2) | ขัดแย้ง → ไม่ออกสัญญาณ")
    if not len(g_prices):
        st.info("ไม่มีข้อมูลราคาโลก (yfinance)")
        return
    if sig_g["signals"]:
        df = pd.DataFrame(sig_g["signals"])[
            ["asset", "side", "model", "strength", "entry", "sl", "tp", "rr",
             "atr14≈", "rsi14", "mom12m%", "dist_200dma%", "trend"]]
        df.columns = ["สินทรัพย์", "ทิศ", "โมเดล", "แข็งแรง", "เข้า", "SL", "TP",
                      "R:R", "ATR≈", "RSI", "โมเมนตัม12ด%", "ห่าง200DMA%", "แนวโน้ม"]
        st.dataframe(df, hide_index=True)
        st.download_button("⬇️ Signal Journal (CSV)",
                           SG.journal_csv(sig_g["signals"],
                                          str(datetime.now().date())),
                           "signal_journal.csv", "text/csv")
        st.info("วินัยหลักฐาน: กรอก pnl เมื่อปิดออเดอร์ แล้วอัปโหลดที่ "
                "'Trade Log & สถิติ' — ห้ามตัดสินระบบก่อนครบ 100 เทรด")
    else:
        st.write("ไม่มีสัญญาณเข้าเกณฑ์ตอนนี้")
    if sig_g["conflicts"]:
        st.markdown("**โมเดลขัดแย้ง (ไม่เลือกข้างเอง):**")
        st.dataframe(pd.DataFrame(sig_g["conflicts"]), hide_index=True)
    if sig_g["skipped"]:
        with st.expander(f"ถูกกรองออก ({len(sig_g['skipped'])})"):
            st.dataframe(pd.DataFrame(sig_g["skipped"]), hide_index=True)


def page_g_scenario():
    if need_global():
        return
    st.caption("ค่าความไวเป็น 'สมมติฐานการออกแบบ' อิงทิศทางประวัติศาสตร์ — "
               "*ไม่ใช่* ค่าที่ประมาณจากข้อมูลจริง")
    left, right = st.columns(2)
    vals = {}
    with left:
        st.markdown("**ปรับสถานการณ์สมมติ**")
        if st.button("Reset ทุกตัว"):
            for k, *_r in SC.SLIDERS:
                st.session_state.pop(f"sc_{k}", None)
            st.rerun()
        for k, th, u, lo, hi, step, dflt in SC.SLIDERS:
            vals[k] = st.slider(f"{th} ({u})", float(lo), float(hi),
                                float(st.session_state.get(f"sc_{k}", dflt)),
                                float(step), key=f"sc_{k}")
    base = {k: mscores[k]["score"] for k in M.MODEL_DEFS}
    res = SC.apply_scenario(base, vals)
    with right:
        st.markdown("**ผลต่อคะแนนโมเดล**")
        for i, k in enumerate(sorted(res, key=lambda k: -res[k]["new"]), 1):
            r = res[k]
            color = "🟢" if r["delta"] > 0.05 else ("🔴" if r["delta"] < -0.05 else "⚪")
            st.write(f"#{i} **{mscores[k]['th']}** {r['base']:.1f} → "
                     f"**{r['new']:.1f}** {color} {r['delta']:+.1f}")
            st.progress(min(1.0, r["new"] / 100.0))
        st.info("นี่คือ what-if ไม่ใช่คำพยากรณ์ — ดูสินทรัพย์ได้/เสียของโมเดลที่ขยับ"
                "ในหน้า 'โมเดลทำกำไร' และ *ทิศทางฝั่งไทย* ในหน้า Scenario ไทย")
    with st.expander("🔍 เมทริกซ์ความไว (โปร่งใส — แก้ได้ใน scenario.py)"):
        st.dataframe(SC.sensitivity_table())


def page_g_bankrun():
    if need_global():
        return
    m = mscores["bank_run"]
    st.metric("คะแนนโมเดลแบงก์รัน", f"{m['score']:.1f}/100"
              if m["score"] == m["score"] else "n/a",
              f"Δสัปดาห์ {deltas.get('bank_run', float('nan')):+.1f}")
    st.write("ส่วนประกอบ: " + " | ".join(f"{n}: {v:.0f}"
             for n, v in m["components"].items()))
    charts = [("DPSACBW027SBOG", "เงินฝากธนาคารพาณิชย์"),
              ("BORROW", "ยอดกู้จาก Fed"), ("RRPONTSYD", "Reverse Repo (RRP)")]
    cols = st.columns(3)
    for (sid, title), c in zip(charts, cols):
        s = S(sid)
        if len(s):
            f = go.Figure(go.Scatter(x=s.index, y=s.values))
            f.update_layout(title=title, height=250,
                            margin=dict(l=10, r=10, t=40, b=10))
            with c:
                plot(f)
    st.warning("ตรงๆ: สาเหตุจริงของแบงก์รัน (unrealized loss ใน HTM, สัดส่วนเงินฝาก"
               "ไม่คุ้มครอง) เป็นข้อมูล *รายไตรมาส* ใน filings — แดชบอร์ดรายวันจับได้"
               "แค่ 'อาการ' (เงินฝากไหล, กู้ Fed, repo ตึง) ซึ่งมาช้ากว่าเหตุ")
    st.markdown("**บทเรียนประวัติศาสตร์ (เทียบเหตุการณ์จริง):**")
    st.dataframe(pd.DataFrame(E.HISTORICAL_EPISODES), hide_index=True)


def page_g_news():
    if need_global():
        return
    st.caption("จับคู่ข่าว→โมเดลด้วยกฎคำสำคัญเปิดเผย — โปร่งใสแต่หยาบ: "
               "พาดหัวเชิงปฏิเสธ/เสียดสีจับผิดได้ และระบบ *ไม่ให้คะแนน sentiment*")
    feeds = st.text_area("RSS feeds (บรรทัดละ 1 URL)",
                         "https://www.federalreserve.gov/feeds/press_all.xml",
                         height=70)
    if st.button("ดึงและวิเคราะห์หัวข้อข่าว"):
        try:
            import feedparser
        except ImportError:
            st.error("ต้องติดตั้งก่อน: pip install feedparser")
            return
        max_sev, rows = 0, []
        for url in [u.strip() for u in feeds.splitlines() if u.strip()]:
            try:
                d = feedparser.parse(url)
                for e_ in d.entries[:12]:
                    title = e_.get("title", "")
                    c = MT.classify_news(title)
                    max_sev = max(max_sev, c["severity"])
                    rows.append({"หัวข้อ": title,
                                 "โมเดล": ", ".join(M.MODEL_DEFS[mm]["th"]
                                                    for mm in c["models"]) or "—",
                                 "ความรุนแรง": c["severity"],
                                 "ลิงก์": e_.get("link", "")})
            except Exception as ex:
                st.error(f"ดึง {url} ไม่ได้: {ex}")
        st.session_state["news_max_sev"] = max_sev
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.write(f"ความรุนแรงสูงสุด: **{max_sev}** "
                     f"(เกณฑ์เรียกประชุม ≥ {MT.NEWS_TRIGGER_SEVERITY})")
    with st.expander("🔍 กฎคำสำคัญ (แก้ได้ใน meeting.py)"):
        st.write({M.MODEL_DEFS[k]["th"]: v for k, v in MT.MODEL_KEYWORDS.items()})
        st.write({f"ระดับ {lvl}": kws for lvl, kws in MT.SEVERITY_RULES})


def page_g_meeting():
    if need_global():
        return
    st.caption("**ก่อนใช้:** 'AI หลายตัว' = โมเดลเดียวเล่นหลายบท ความเห็นไม่อิสระ"
               "ทางสถิติ | กติกาเหล็ก: ห้ามคิดเลข/ตั้งราคาเป้าเอง — ลงมติได้แค่ "
               "เห็นด้วย/คัดค้าน/งด ต่อสัญญาณที่ engine คำนวณแล้ว")
    ev_txt = st.text_input("เวลาตัวเลขสำคัญ (ISO เช่น 2026-07-20 19:30 คั่นด้วย ;)", "")
    events = []
    for tok in [t.strip() for t in ev_txt.split(";") if t.strip()]:
        try:
            events.append(datetime.fromisoformat(tok))
        except ValueError:
            st.warning(f"อ่านเวลาไม่ได้: {tok}")
    trig = MT.should_convene(deltas, st.session_state.get("news_max_sev", 0),
                             events, datetime.now())
    if trig["convene"]:
        st.warning("เงื่อนไขเปิดประชุมทำงาน:\n"
                   + "\n".join("- " + r for r in trig["reasons"]))
    else:
        st.success("ยังไม่เข้าเงื่อนไขอัตโนมัติ — เปิดเองได้ด้านล่าง")
    ids = [p["id"] for p in MT.PERSONAS if p["id"] != "chair"]
    labels = {p["id"]: p["th"] for p in MT.PERSONAS}
    panel = st.multiselect("ผู้เข้าประชุม (มากขึ้น = token มากขึ้น ไม่ใช่ความเห็น"
                           "อิสระมากขึ้น)", ids, default=MT.DEFAULT_PANEL,
                           format_func=lambda i: labels[i])
    api_key = st.text_input("Anthropic API key", type="password")
    context = {
        "as_of": str(datetime.now().date()),
        "data_mode": "DEMO" if is_demo else "LIVE",
        "crisis_score": None if composite != composite else round(composite, 1),
        "set_overlay": {"label": overlay["label"], "reasons": overlay["reasons"]},
        "model_scores": {m["th"]: m["score"] for m in mscores.values()},
        "model_deltas_wk": {mscores[k]["th"]: deltas.get(k) for k in deltas},
        "warnings": [w["msg"] for w in warnings_g],
        "signals_pending": [{k: s_[k] for k in ("asset", "side", "model",
                                                "strength", "entry", "sl", "tp",
                                                "rsi14", "trend")}
                            for s_ in sig_g["signals"]],
        "conflicts": [c["asset"] for c in sig_g["conflicts"]],
    }
    with st.expander("ข้อมูลที่ส่งให้ AI (โปร่งใส)"):
        st.code(json.dumps(context, ensure_ascii=False, indent=2))
    if st.button("🏛️ เปิดประชุม (3 API calls)"):
        if not api_key:
            st.error("ต้องใส่ Anthropic API key")
        elif not sig_g["signals"] and not warnings_g:
            st.info("ไม่มีสัญญาณ/เตือนให้ลงมติ — ประชุมไปก็ไม่มีวาระ")
        else:
            try:
                import anthropic
            except ImportError:
                st.error("ต้องติดตั้งก่อน: pip install anthropic")
                return
            client = anthropic.Anthropic(api_key=api_key)
            msgs = []
            try:
                with st.spinner("รอบ 1: แถลงมุมมอง..."):
                    msgs.append({"role": "user", "content": MT.build_round1_prompt(
                        panel, json.dumps(context, ensure_ascii=False))})
                    r1 = client.messages.create(model="claude-sonnet-4-6",
                                                max_tokens=1800, messages=msgs)
                    t1 = "".join(b.text for b in r1.content if b.type == "text")
                    msgs.append({"role": "assistant", "content": t1})
                with st.spinner("รอบ 2: โต้แย้ง..."):
                    msgs.append({"role": "user", "content": MT.build_round2_prompt()})
                    r2 = client.messages.create(model="claude-sonnet-4-6",
                                                max_tokens=1200, messages=msgs)
                    t2 = "".join(b.text for b in r2.content if b.type == "text")
                    msgs.append({"role": "assistant", "content": t2})
                with st.spinner("ประธานสรุปมติ..."):
                    msgs.append({"role": "user", "content": MT.build_chair_prompt()})
                    r3 = client.messages.create(model="claude-sonnet-4-6",
                                                max_tokens=1000, messages=msgs)
                    t3 = "".join(b.text for b in r3.content if b.type == "text")
                st.session_state["meeting"] = [("รอบ 1 — แถลงมุมมอง", t1),
                                               ("รอบ 2 — โต้แย้ง", t2),
                                               ("มติประธาน", t3)]
            except Exception as e_:
                st.error(f"เรียก API ไม่สำเร็จ: {e_}")
    for title, body in st.session_state.get("meeting", []):
        with st.expander(title, expanded=(title == "มติประธาน")):
            st.markdown(body)
    if st.session_state.get("meeting"):
        st.caption("มติ = ความเห็นเชิงคุณภาพจากโมเดลเดียวเล่นหลายบท — "
                   "ไม่เพิ่มความน่าจะเป็นถูกของสัญญาณ และไม่ใช่คำแนะนำการลงทุน")


def page_g_macro():
    if need_global():
        return
    tC, tS, tR = st.tabs(["Yield Curve", "Stress Monitor", "Regime (HMM)"])
    with tC:
        c1, c2, c3 = st.columns(3)
        c1.metric("10Y-3M", f"{spread_latest:+.2f}%"
                  if spread_latest == spread_latest else "n/a",
                  "INVERTED" if spread_latest < 0 else "ปกติ",
                  delta_color="inverse" if spread_latest < 0 else "normal")
        t2 = S("T10Y2Y")
        c2.metric("10Y-2Y", f"{float(t2.iloc[-1]):+.2f}%" if len(t2) else "n/a")
        c3.metric("P(recession 12 ด.)",
                  f"{rec_prob:.0%}" if rec_prob == rec_prob else "n/a")
        t10 = S("T10Y3M")
        if len(t10):
            f = go.Figure()
            f.add_scatter(x=t10.index, y=t10.values, name="10Y-3M")
            if len(t2):
                f.add_scatter(x=t2.index, y=t2.values, name="10Y-2Y")
            f.add_hline(y=0, line_dash="dot")
            f.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
            plot(f)
        with st.expander("💡 อธิบายแบบง่าย"):
            st.markdown(X.curve_explainer(rec_prob, spread_latest, inv_days))
    with tS:
        rows = []
        for sid in ["STLFSI4", "NFCI", "BAMLH0A0HYM2", "BAMLC0A0CM", "VIXCLS",
                    "DCOILWTICO", "T5YIE"]:
            s_ = S(sid)
            if len(s_):
                rows.append({"ตัวชี้วัด": D.FRED_SERIES[sid],
                             "ล่าสุด": round(float(s_.iloc[-1]), 2),
                             "z": round(E.zscore_of_latest(s_), 2),
                             "pct": round(E.percentile_of_latest(s_), 0)})
        if len(move) > 100:
            rows.append({"ตัวชี้วัด": "MOVE",
                         "ล่าสุด": round(float(move.iloc[-1]), 1),
                         "z": round(E.zscore_of_latest(move), 2),
                         "pct": round(E.percentile_of_latest(move), 0)})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("ตัวชี้วัดเหล่านี้ coincident — เทอร์โมมิเตอร์ ไม่ใช่เครื่องพยากรณ์")
        with st.expander("💡 percentile / z-score อ่านยังไง"):
            st.markdown(X.macro_explainer(subs.get("stress", float("nan")),
                                          E.zscore_of_latest(S("STLFSI4"))))
    with tR:
        y10 = S("DGS10")
        if len(y10) > 400:
            dy = y10.resample("W-FRI").last().diff().dropna()
            try:
                res = E.fit_hmm_2state(dy.values)
                hv = res.high_vol_state
                p_now = float(res.filtered[-1, hv])
                st.metric("P(High-vol regime) — real-time", f"{p_now:.0%}")
                f = go.Figure()
                f.add_scatter(x=dy.index, y=res.smoothed[:, hv],
                              name="smoothed (ย้อนหลัง)")
                f.add_scatter(x=dy.index, y=res.filtered[:, hv],
                              name="filtered (real-time)", line=dict(dash="dot"))
                f.update_layout(height=300, yaxis_range=[0, 1],
                                margin=dict(l=10, r=10, t=20, b=10))
                plot(f)
                with st.expander("💡 อธิบายแบบง่าย"):
                    st.markdown(X.regime_explainer(p_now, res.sigma[1 - hv] * 100,
                                                   res.sigma[hv] * 100))
            except Exception as e_:
                st.error(f"HMM ไม่ converge: {e_}")
        else:
            st.info("ข้อมูล 10Y ไม่พอ")


def page_g_trend():
    if need_global():
        return
    st.caption("บริบทแนวโน้มโลก — ไม่ใช่สัญญาณซื้อขาย")
    if len(g_prices) > 260:
        ma200 = g_prices.rolling(200).mean()
        mom12 = g_prices.pct_change(252)
        vol20 = g_prices.pct_change().rolling(20).std() * math.sqrt(252) * 100
        rows = []
        for t in g_prices.columns:
            c = g_prices[t].dropna()
            if len(c) < 260:
                continue
            rows.append({"สินทรัพย์": D.YF_ASSETS.get(t, t),
                         "ราคา": round(float(c.iloc[-1]), 2),
                         "เทียบ 200DMA": "เหนือ" if c.iloc[-1] >= ma200[t].iloc[-1]
                         else "ใต้",
                         "โมเมนตัม 12ด.": f"{mom12[t].iloc[-1]:+.1%}",
                         "Vol 20d/ปี": f"{vol20[t].iloc[-1]:.0f}%"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        if breadth_g == breadth_g:
            st.metric("Risk-off breadth (% ใต้ 200DMA)", f"{breadth_g:.0f}%")
        with st.expander("💡 อ่านตารางนี้ยังไง"):
            st.markdown(X.trend_explainer(breadth_g))
    else:
        st.info("ไม่มีข้อมูลราคาโลก")


# ===========================================================================
# 📖 Shared pages
# ===========================================================================

def page_tradelog():
    st.caption("อัปโหลด CSV ที่มีคอลัมน์ `pnl` — ใช้ได้ทั้ง journal ฝั่ง Global "
               "และบันทึกเทรด SET ของคุณ (ระบบเดียว มาตรฐานเดียว)")
    up = st.file_uploader("ไฟล์ trade log (.csv)", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as e_:
            st.error(f"อ่านไฟล์ไม่ได้: {e_}")
            return
        if "pnl" not in df.columns:
            st.error("ไม่พบคอลัมน์ 'pnl' — ต้องมีอย่างน้อยคอลัมน์นี้ "
                     "(กำไร/ขาดทุนต่อเทรด หน่วยอะไรก็ได้แต่ต้องสม่ำเสมอ)")
            return
        r = E.trade_log_report(df)
        c = st.columns(5)
        c[0].metric("จำนวนเทรด (มี pnl)", r["n"])
        c[1].metric("Win rate", f"{r['win_rate']:.0%}" if r["n"] else "—",
                    f"CI95: {r['ci_low']:.0%}–{r['ci_high']:.0%}"
                    if r["n"] and r["ci_low"] == r["ci_low"] else "")
        pf = r["profit_factor"]
        c[2].metric("Profit Factor",
                    "∞" if (isinstance(pf, float) and math.isinf(pf))
                    else (f"{pf:.2f}" if pf == pf else "—"))
        c[3].metric("Expectancy/เทรด",
                    f"{r['expectancy']:+.2f}" if r["expectancy"] == r["expectancy"]
                    else "—")
        c[4].metric("PSR", f"{r['psr']:.0%}" if r["psr"] == r["psr"] else "n/a")
        st.info("🧮 " + r["verdict"])
        st.markdown("**💡 แปลผลแบบภาษาคน**")
        st.markdown(X.interpret_trade_log(r))
    with st.expander("ต้องมีกี่เทรดถึงเชื่อ win rate ได้"):
        moe = st.slider("ยอมรับความคลาดเคลื่อน (± จุดเปอร์เซ็นต์)", 1, 15, 5)
        st.write(f"ที่ความเชื่อมั่น 95% และ win rate แถว 50%: ต้องการประมาณ "
                 f"**{E.required_n(moe/100.0):,} เทรด**")
        st.caption("นี่คือเหตุผลที่กติกา 'ห้ามตัดสินระบบก่อน 100 เทรด' มีอยู่ — "
                   "และ 100 ก็ยังให้ความแม่นแค่ ±10 จุดเท่านั้น")


def page_glossary():
    st.subheader("คู่มืออ่านค่า — ทุกศัพท์จากทั้งสองฝั่งในที่เดียว")
    st.caption("แต่ละคำมี 3 ส่วน: คืออะไร / วิธีอ่าน / **สิ่งที่มัน *ไม่ได้* บอก** "
               f"(Global {len(X.GLOSSARY)} คำ + SET/Bridge {len(BR.SET_GLOSSARY)} คำ)")
    q = st.text_input("ค้นหาคำศัพท์", "")
    merged = {}
    merged.update({"🌍 " + k: v for k, v in X.GLOSSARY.items()})
    merged.update({"🇹🇭 " + k: v for k, v in BR.SET_GLOSSARY.items()})
    shown = 0
    for term, d in merged.items():
        blob = term + d.get("what", "") + d.get("read", "") + d.get("not", "")
        if q.strip() and q.strip().lower() not in blob.lower():
            continue
        shown += 1
        with st.expander(term):
            st.markdown(f"**คืออะไร:** {d['what']}")
            st.markdown(f"**วิธีอ่าน:** {d['read']}")
            st.markdown(f"**สิ่งที่มัน *ไม่ได้* บอก:** {d['not']}")
    if q.strip() and shown == 0:
        st.info("ไม่พบคำที่ค้นหา — ลองคำสั้นลง")


def page_limits():
    st.subheader("ข้อจำกัด & จุดพัฒนา — อ่านให้จบก่อนใช้")
    st.markdown(
        "#### สิ่งที่ระบบนี้ทำได้จริง\n"
        "- รวมบริบทโลก (FRED/ราคาโลก) กับหุ้นไทย SET100 ไว้จอเดียว โดยทุกตัวเลข"
        "มาจากสูตรเปิดเผย + แสดงความไม่แน่นอนเสมอ\n"
        "- Overlay Global→SET ตามกติกาประกาศล่วงหน้า (วินัยคุมขนาดความเสี่ยง)\n"
        "- ชั้นตรวจสอบครบ: ต้นทุนไทยจริง, Wilson CI, PSR, Deflated Sharpe, "
        "เกณฑ์ n ขั้นต่ำ, กัน look-ahead\n\n"
        "#### สิ่งที่ทำไม่ได้ (และไม่มีเครื่องมือไหนทำได้)\n"
        "- ทำนายราคา / รับประกันกำไร / แทนการอ่านงบและเช็กลิสต์พื้นฐาน\n\n"
        "#### ข้อจำกัดข้อมูล (ตรงไปตรงมา)\n"
        "- yfinance = สิ้นวัน/ดีเลย์ | ไม่มี NVDR·foreign flow รายวัน (ไม่มี API ฟรี)\n"
        "- รายชื่อ SET100 อัปเดตมือทุกรอบ (ม.ค./ก.ค.) — backtest ด้วยรายชื่อ"
        "ปัจจุบันมี survivorship bias ผลจริงมักแย่กว่าที่เห็น\n"
        "- Overlay ยังเป็น rule-based ที่ *ไม่ผ่าน validation เชิงประจักษ์* — "
        "จึงไม่ถูกผูกเข้า backtest โดยเจตนา\n"
        "- ATR ฝั่ง Global ประมาณจากราคาปิด (ไม่มี high/low)\n\n"
        "#### จุดพัฒนาต่อ (เรียงตามผลตอบแทนต่อแรง)\n"
        "1. ~~หน้า import flow รายกลุ่มนักลงทุน~~ ✅ **ทำแล้ว** (หน้า Fund Flow: "
        "อ่าน/เพิ่มรายวัน/ดาวน์โหลด + สถิติ same-day vs next-day จากข้อมูลจริง)\n"
        "2. **Validate overlay ด้วย event study** — วัด lead-lag ระหว่าง composite/"
        "โมเดลเสี่ยง กับผลตอบแทน SET ล่วงหน้า (ระวัง: เหตุการณ์วิกฤตมีน้อย, n เล็ก)\n"
        "3. **Walk-forward / purged CV** สำหรับ backtest — จำเป็นก่อนใช้เงินจริง\n"
        "4. **Point-in-time constituents** แก้ survivorship bias (ข้อมูลไม่ฟรี)\n"
        "5. **Per-stock spread จริง** แทนค่าคงที่ — สำคัญมากกับ SET51-100\n"
        "6. ดึง OHLC เต็มฝั่ง Global — ✅ ทำแล้วสำหรับทองคำ (หน้า XAU ใช้ ATR "
        "จาก high/low จริง); สินทรัพย์โลกตัวอื่นยังใช้ราคาปิด\n"
        "7. **Calibration log ห้องประชุม AI** — บันทึกมติเทียบผลจริง สะสมหลักฐาน"
        "ว่าห้องประชุมช่วยจริงไหม\n"
        "8. **Stale-data detector** — เตือนเมื่อ ticker ใดหยุดอัปเดตเงียบๆ\n"
        "9. **Cache ราคาแบบ parquet** — โหลด 100 ตัวครั้งแรก ~2 นาที ลดได้มาก\n"
        "10. **Auto-อัปเดต universe จากเว็บ SET** — ทำได้แต่เปราะ (scraping) "
        "จึงยังเลือกวิธีแก้มือที่โปร่งใสกว่า")
    st.divider()
    st.caption(DISCLAIMER)




def page_set_flow():
    st.subheader("Fund Flow รายกลุ่มนักลงทุน — บริบท ไม่ใช่สัญญาณ")
    fdf, issues = load_flows()
    if fdf.empty:
        st.warning("ยังไม่มีไฟล์ Set_update.csv ในโฟลเดอร์แอป — อัปโหลดเพื่อเริ่มต้น")
        up0 = st.file_uploader("อัปโหลด Set_update.csv", type=["csv"])
        if up0 is not None:
            d0, iss0 = FL.load_flow_csv(up0)
            if d0.empty:
                st.error("อ่านไฟล์ไม่ได้: " + "; ".join(iss0))
            else:
                FL.save_flow_csv(d0, FLOW_PATH)
                st.success(f"บันทึกแล้ว {len(d0)} วัน — รีเฟรชหน้า")
                st.rerun()
        return
    for i in issues:
        st.caption("ℹ️ " + i)
    last = fdf.index[-1]
    _mt = None
    try:
        from datetime import datetime as _dt
        _mt = _dt.fromtimestamp(os.path.getmtime(FLOW_PATH))
    except Exception:
        _mt = None
    DS_ITEMS.append(DS.describe("Fund Flow (Set_update.csv)", fdf, "SET",
                                _mt, 0,
                                "ไฟล์ในเครื่อง — ไม่ได้ดึงอัตโนมัติ ต้องเติมข้อมูลเอง"))
    st.caption(f"ข้อมูล {len(fdf)} วันทำการ: {fdf.index[0]:%d/%m/%Y} → "
               f"**{last:%d/%m/%Y}** (สุทธิ หน่วยล้านบาท) · "
               f"ไฟล์ถูกแก้ล่าสุด {DS.fmt(_mt)}")
    _fl_age = DS.staleness(last, "SET")
    if _fl_age != "ok":
        st.warning(f"⚠️ ข้อมูล flow ล่าสุดคือ {last:%d/%m/%Y} ({DS.age_text(last)}) "
                   "— ยังไม่ได้เติมข้อมูลวันล่าสุดหรือเปล่า? ตัวเลขบนหน้านี้"
                   "สะท้อนแค่ถึงวันนั้น")

    st.dataframe(FL.flow_summary(fdf), hide_index=True)
    cc = FL.same_day_corr(fdf)
    if cc.get("ok"):
        st.info(f"📐 จากข้อมูลของคุณเอง (n={cc['n']} วันล่าสุด): สหสัมพันธ์ flow "
                f"ต่างชาติกับการเปลี่ยนแปลงดัชนี **วันเดียวกัน r={cc['same_day']:.2f}** "
                f"แต่พยากรณ์ **วันถัดไป r={cc['next_day']:.2f}** — "
                "flow จึงเป็น 'กระจกสะท้อนวันนี้' มากกว่า 'เข็มทิศพรุ่งนี้'")
    st.warning("⚠️ " + FL.EVIDENCE_NOTE + " | จอนี้ *ไม่ถูกผูก* เข้า overlay/backtest "
               "จนกว่าจะผ่าน validation")

    win_map = {"3 เดือน": 63, "6 เดือน": 126, "1 ปี": 250, "ทั้งหมด": len(fdf)}
    wsel = st.selectbox("ช่วงกราฟ", list(win_map), index=2)
    sub = fdf.tail(win_map[wsel])
    figc = go.Figure()
    for c_ in FL.NET_COLS:
        figc.add_trace(go.Scatter(x=sub.index, y=sub[c_].cumsum(),
                                  name=FL.TH_NAMES[c_], line=dict(width=1.5)))
    figc.update_layout(title="ยอดสะสมสุทธิ (เริ่มนับศูนย์ที่ต้นช่วงที่เลือก)",
                       height=320, margin=dict(l=10, r=10, t=40, b=10),
                       legend=dict(orientation="h"))
    plot(figc)
    r20 = fdf["Foreign"].rolling(20).sum().reindex(sub.index)
    figb = go.Figure(go.Bar(x=sub.index, y=r20, name="ต่างชาติสะสม 20 วัน"))
    figb.add_hline(y=0, line_width=1)
    figb.update_layout(title="ต่างชาติ: ยอดสะสมเคลื่อนที่ 20 วันทำการ (ลบ.)",
                       height=260, margin=dict(l=10, r=10, t=40, b=10))
    plot(figb)
    with st.expander("📋 ข้อมูลรายวันล่าสุด (15 แถว)"):
        st.dataframe(fdf.sort_index(ascending=False).head(15)
                     .rename(columns=FL.TH_NAMES))

    st.markdown("#### ➕ เพิ่ม / แก้ไขข้อมูลรายวัน")
    today = pd.Timestamp.today().normalize()
    dsel = st.date_input("วันที่ (ค่าตั้งต้น = วันล่าสุดที่มีข้อมูล)",
                         value=last.date(), max_value=today.date(), key="fl_d")
    ts = pd.Timestamp(dsel)
    exists = ts in fdf.index
    if exists:
        row = fdf.loc[ts]
        st.caption(f"📌 มีข้อมูลวันนี้แล้ว — ฟอร์มแสดงค่าเดิม แก้แล้วกดบันทึกเพื่อ"
                   f"เขียนทับ | Institute {row['Institute']:+,.2f} · "
                   f"Foreign {row['Foreign']:+,.2f} · Retail {row['Retail']:+,.2f}")
        d_in, d_fo, d_re = (float(row["Institute"]), float(row["Foreign"]),
                            float(row["Retail"]))
        d_ix = float(row[FL.IDX_COL]) if row[FL.IDX_COL] == row[FL.IDX_COL] else 0.0
        d_ch = float(row[FL.CHG_COL]) if row[FL.CHG_COL] == row[FL.CHG_COL] else 0.0
    else:
        st.caption("🆕 ยังไม่มีข้อมูลวันที่นี้ — กรอกใหม่ได้เลย")
        d_in = d_fo = d_re = d_ix = d_ch = 0.0
    k = str(dsel)
    with st.form("flow_add"):
        c3, c4, c5 = st.columns(3)
        v_in = c3.number_input("Institute (ลบ.)", value=d_in, step=50.0,
                               format="%.2f", key=f"fi_{k}")
        v_fo = c4.number_input("Foreign (ลบ.)", value=d_fo, step=50.0,
                               format="%.2f", key=f"ff_{k}")
        v_re = c5.number_input("Retail (ลบ.)", value=d_re, step=50.0,
                               format="%.2f", key=f"fr_{k}")
        c6, c7 = st.columns(2)
        v_ix = c6.number_input("Set Index (0 = ไม่กรอก)", value=d_ix, step=1.0,
                               format="%.2f", key=f"fx_{k}")
        v_ch = c7.number_input("Set Change (จุด)", value=d_ch, step=0.5,
                               format="%.2f", key=f"fc_{k}")
        sub_btn = st.form_submit_button(
            "💾 บันทึกลง CSV" + (" (เขียนทับวันเดิม)" if exists else ""))
    if sub_btn:
        ssum = v_in + v_fo + v_re
        if abs(ssum) > 1.0:
            st.warning(f"3 กลุ่มรวม = {ssum:+,.2f} ลบ. (ไฟล์เดิมรวม ≈ 0 เสมอ) — "
                       "บันทึกตามที่กรอก โปรดตรวจตัวเลขอีกครั้ง")
        d2, okk, msg = FL.append_or_update(
            fdf, ts, v_in, v_fo, v_re,
            set_index=(None if v_ix == 0 else v_ix),
            set_change=(None if (v_ch == 0 and v_ix == 0) else v_ch),
            overwrite=True)
        if okk:
            try:
                FL.save_flow_csv(d2, FLOW_PATH)
                st.success("✅ " + msg)
            except Exception as e_:
                st.error(f"เขียนไฟล์ไม่ได้ ({e_}) — ใช้ปุ่มดาวน์โหลดแทน")
            st.rerun()
        else:
            st.error(msg)
    st.download_button("⬇️ ดาวน์โหลด Set_update.csv (ฉบับล่าสุด)",
                       FL.to_csv_bytes(fdf), "Set_update.csv", "text/csv")
    with st.expander("☁️ อัปโหลดขึ้น GitHub (commit ทับไฟล์ใน repo)"):
        st.caption("ใช้ GitHub token ที่มีสิทธิ์เขียน (Fine-grained: Contents → "
                   "Read and write เฉพาะ repo นี้) — token ใช้ครั้งนี้ ไม่ถูกเก็บ | "
                   "⚠️ commit สำเร็จจะทำให้ Streamlit Cloud redeploy อัตโนมัติ "
                   "(แอปรีสตาร์ต ~1-2 นาที ข้อมูลบนจอหายชั่วคราว)")
        g1, g2 = st.columns(2)
        gh_owner = g1.text_input("Owner", "Pisit7792")
        gh_repo = g2.text_input("Repo", "Set-Bond--dashboard")
        g3, g4 = st.columns(2)
        gh_branch = g3.text_input("Branch", "main")
        gh_path = g4.text_input("Path ไฟล์ใน repo", "Set_update.csv")
        gh_tok = st.text_input("GitHub token", type="password")
        if st.button("🚀 Commit ไฟล์ล่าสุดขึ้น GitHub"):
            if not gh_tok.strip():
                st.error("ต้องใส่ token ก่อน")
            else:
                import base64
                import requests as _rq
                api = (f"https://api.github.com/repos/{gh_owner.strip()}/"
                       f"{gh_repo.strip()}/contents/{gh_path.strip()}")
                hd = {"Authorization": f"Bearer {gh_tok.strip()}",
                      "Accept": "application/vnd.github+json"}
                try:
                    sha = None
                    r0 = _rq.get(api, headers=hd,
                                 params={"ref": gh_branch.strip()}, timeout=20)
                    if r0.status_code == 200:
                        sha = r0.json().get("sha")
                    payload = {
                        "message": f"update Set_update.csv (ถึง {last:%d/%m/%Y})",
                        "content": base64.b64encode(
                            FL.to_csv_bytes(fdf)).decode(),
                        "branch": gh_branch.strip()}
                    if sha:
                        payload["sha"] = sha
                    r1 = _rq.put(api, headers=hd, json=payload, timeout=30)
                    if r1.status_code in (200, 201):
                        url = r1.json().get("commit", {}).get("html_url", "")
                        st.success("✅ Commit สำเร็จ"
                                   + (f" — [ดู commit]({url})" if url else ""))
                    else:
                        st.error(f"GitHub ตอบ {r1.status_code}: "
                                 f"{str(r1.json().get('message', r1.text))[:200]}")
                except Exception as e_:
                    st.error(f"เชื่อมต่อ GitHub ไม่ได้: {e_}")
    st.info("💾 ความถาวร: บนเครื่องตัวเอง = ถาวรทันที | บน Streamlit Cloud ไฟล์อยู่"
            "จนเครื่อง restart — ทางถาวรคือปุ่ม GitHub ข้างบน (แนะนำ) หรือ"
            "ดาวน์โหลดแล้วอัปทับเองบนเว็บ GitHub")

def page_gold():
    if not _guard(["gold", "accum"], "หน้าทองคำ (v6.4.1)"):
        return
    st.subheader("ทองคำ — XAU Research Trend Pullback v6.4.1 (พอร์ตจากสคริปต์ของคุณ)")
    st.warning("**อ่านก่อนใช้:** (1) ใช้ **GC=F ฟิวเจอร์ส** แทน spot XAUUSD — "
               "ระดับ/ATR ใกล้เคียงแต่ basis ต่างเล็กน้อย (2) ผล backtest เป็น "
               "in-sample และชื่อรุ่น v6.4 บ่งว่าผ่านการปรับหลายรอบ → DSR ตั้ง "
               "trials สูงไว้ก่อน (3) Tier C ของสคริปต์ (real-yield, gold-DXY "
               "inverse) **จงใจไม่โค้ด** ตามต้นฉบับ (4) เป้า validate ของสคริปต์เอง: "
               "**30-50 เทรดจริงใน journal, PF > 1.5 หลัง swap** (5) v6.4.1 เพิ่ม "
               "squeeze gate + accumulation/distribution footprint — ทั้งคู่**ปิด/display-only เป็นค่าตั้งต้น** ผลจึงเท่า v6.4 เดิมทุกแท่ง")
    gsrc = st.radio("แหล่งข้อมูลราคา",
                    ["PAXG-USD — โทเคนอิงทอง เทรดทุกวัน 24/7 (แสดง/คำนวณทุกวันตามที่ขอ)",
                     "GC=F — ฟิวเจอร์ส COMEX (จันทร์-ศุกร์)"], 0, horizontal=True)
    gsym = "PAXG-USD" if gsrc.startswith("PAXG") else "GC=F"
    st.caption("ตรงไปตรงมา: PAXG = โทเคนอิงทองจริง 1 oz ราคาแนบ spot มาก "
               "(อาจมี premium/discount เล็กน้อย) — โหมดนี้อินดิเคเตอร์คำนวณบน"
               "แท่งจริง 7 วัน/สัปดาห์ ATR/SMA จึงต่างจากฟิวเจอร์สเล็กน้อย")
    c1, c2, c3, c4 = st.columns(4)
    g_period = c1.selectbox("ช่วงข้อมูล", ["3y", "5y", "10y"], 1)
    spread_c = c2.number_input("Spread ไป-กลับ (¢/oz)", 0.0, 200.0, 25.0, 5.0)
    swap_l = c3.number_input("Swap ฝั่ง Long ($/oz/คืน)", -5.0, 5.0, -0.76, 0.01)
    swap_s = c4.number_input("Swap ฝั่ง Short ($/oz/คืน)", -5.0, 5.0, 0.30, 0.01)
    c5, c6, c7, c8 = st.columns(4)
    use_carry = c5.checkbox("Tier B: JPY carry veto", False)
    use_er = c6.checkbox("Tier B: ER hard gate", False)
    use_w = c7.checkbox("Tier B: Weekly EMA gate", False)
    use_y10 = c8.checkbox("Tier B: US10Y gate (ต้องเปิดฝั่ง Global Live)", False)
    g_trials = st.number_input("DSR trials (v6.4.1 → แนะนำ ≥ 30)", 1, 5000, 30)
    st.caption("**v6.4.1 (Pine v4.1) — Tape context** · ค่าตั้งต้นทั้งสองสวิตช์ "
               "= ปิด ตามต้นฉบับ ซึ่งให้ผลเท่ากับ v6.4 เดิมทุกแท่ง")
    c9, c10, c11 = st.columns(3)
    trust_vol = c9.checkbox("trustVol — เชื่อวอลุ่ม (+โหวตวอลุ่ม 2 ข้อ)", False,
                            help="ต้นฉบับปิดเป็นค่าตั้งต้น เหตุผลของต้นฉบับ: บน "
                                 "CFD ทองคำ volume คือจำนวน tick ไม่ใช่วอลุ่มจริง "
                                 "· ปิด = ใช้เฉพาะโหวตราคาล้วน 2 ข้อ และต้องผ่านทั้งคู่")
    sqz_gate = c10.checkbox("useSqzGate — ให้ squeeze เป็นประตูเข้า", False,
                            help="ต้นฉบับปิดเป็นค่าตั้งต้น (edge decayed) · "
                                 "เปิดแล้ว backtest ด้านล่างจะเปลี่ยนตามด้วย")
    g_closed = c11.checkbox("ใช้เฉพาะแท่งที่ปิดแล้ว (ส่วนบริบทเทป)", True,
                            help="PAXG เดิน 7 วัน/สัปดาห์ แท่ง 'วันนี้' ไม่เคยจบจริง")
    if sqz_gate:
        st.warning("⚠️ เปิด squeeze gate = **เปลี่ยนกติกาเข้า** ไม่ใช่แค่การแสดงผล "
                   "— ตัวเลข backtest ด้านล่างจะไม่ใช่ค่าตั้งต้นของต้นฉบับอีกต่อไป")

    p = G.GoldParams(spread_cents=spread_c, swap_long_oz=swap_l,
                     swap_short_oz=swap_s, use_carry=use_carry,
                     use_er_gate=use_er, use_htf_w=use_w, use_y10=use_y10,
                     trust_vol=trust_vol, use_sqz_gate=sqz_gate)
    xau, dxy, jpy, vix = C_gold_bundle(gsym, g_period, use_carry)
    stamp_add(f"ราคาทองคำ ({gsym})", xau,
              "24H" if gsym == "PAXG-USD" else "COMEX",
              "gold", f"{gsym}|{g_period}|{use_carry}", 3600,
              "PAXG = โทเคนอิงทอง เดิน 7 วัน/สัปดาห์ · GC=F = ฟิวเจอร์ส "
              "COMEX ซีรีส์ต่อเนื่อง (มี roll)")
    if xau.empty or len(xau) < 320:
        st.error(f"ดึงราคาทอง ({gsym}) ไม่ได้/สั้นเกิน — ลองสลับแหล่งข้อมูลหรือตรวจเน็ต")
        return
    y10s = S("DGS10") if (use_y10 and g_on and not is_demo) else None
    if use_y10 and (y10s is None or not len(y10s)):
        st.caption("US10Y gate: ไม่มีข้อมูล (ฝั่ง Global ไม่ได้เปิด Live) — "
                   "gate ผ่านอัตโนมัติแบบ fail-open ตามต้นฉบับ")
        y10s = None
    fr = G.compute_frame(xau, dxy_close=dxy, y10_close=y10s,
                         usdjpy_close=jpy, vix_close=vix, p=p)
    stt = G.state_today(fr, p)
    seven_g = bool((fr.index.weekday >= 5).any())
    if dxy is None:
        st.caption("DXY ดึงไม่ได้ — veto ผ่านอัตโนมัติ (fail-open ตามต้นฉบับ)")

    m = st.columns(4)
    m[0].metric("Regime (SMA200+slope)", stt["regime"])
    m[1].metric("Score L / S", f"{stt['score_l']} / {stt['score_s']}",
                "เกณฑ์ ≥ 40")
    m[2].metric("RSI(14)", stt["rsi"] if stt["rsi"] is not None else "—")
    m[3].metric("ATR% ของราคา", f"{stt['atr_pct']:.2f}%",
                f"vol rank {stt['vol_rank']:.0f}"
                if stt["vol_rank"] is not None else "")
    st.markdown(f"**สถานะวันนี้:** {stt['status']}")
    if stt["triggered"]:
        st.success("เงื่อนไขเข้า *ครบเมื่อปิดแท่งล่าสุด* — ตามกติกา ลงมือที่ "
                   "**ราคาเปิดแท่งถัดไป** เท่านั้น (ไม่ไล่ราคา)")
    if stt["plan"]:
        pl = stt["plan"]
        st.info(f"แผนอ้างอิงฝั่ง **{pl['side']}** (ต่อทุน $10,000, เสี่ยง 1%): "
                f"stop {pl['stop_dist']}$ → ~{pl['qty_oz']} oz "
                f"(size mult {pl['size_mult']}x) | SL เริ่ม ~{pl['sl']} | "
                f"{pl['trail']}")
    ck = pd.DataFrame(stt["checklist"])
    ck["ผ่าน"] = ck["ผ่าน"].map({True: "✅", False: "❌"})
    st.dataframe(ck, hide_index=True)

    show = fr.tail(250)
    figg = go.Figure()
    figg.add_trace(go.Candlestick(x=show.index, open=show["Open"],
                                  high=show["High"], low=show["Low"],
                                  close=show["Close"], name=gsym))
    figg.add_trace(go.Scatter(x=show.index, y=show["reg_sma"], name="SMA200[1]",
                              line=dict(width=1.4, color="#888")))
    figg.add_trace(go.Scatter(x=show.index, y=show["pb_ema"], name="EMA21",
                              line=dict(width=1, dash="dot")))
    figg.add_trace(go.Scatter(x=show.index, y=show["st_ema"], name="EMA50",
                              line=dict(width=1, dash="dash")))
    for cond, mark, col in [("long_cond", "triangle-up", "#2c7"),
                            ("short_cond", "triangle-down", "#e55")]:
        pts = show[show[cond]]
        if len(pts):
            figg.add_trace(go.Scatter(x=pts.index, y=pts["Close"],
                                      mode="markers", name=cond,
                                      marker=dict(symbol=mark, size=9,
                                                  color=col)))
    figg.update_layout(height=460, xaxis_rangeslider_visible=False,
                       margin=dict(l=10, r=10, t=30, b=10),
                       legend=dict(orientation="h"))
    if not seven_g:
        figg.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    plot(figg)

    # ------------------------------------------------------------------
    # v6.4.1: Tape context — squeeze gate + accumulation/distribution
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("#### 🟡 จุดสะสม / 🟣 จุดกระจาย / 🔵 จุดสควีซ (v6.4.1)")
    st.info(
        "**พอร์ตตรงจากสคริปต์ของคุณเอง** — *XAU Research Trend Pullback v4.1* "
        "หัวข้อ 12) Tape context · " + G.TAPE_DISCLOSURE)
    if trust_vol:
        st.warning("⚠️ เปิด trustVol แล้ว — บริบทของแอปนี้ต่างจากที่ต้นฉบับ"
                   f"สมมติไว้ (CFD): **{gsym}** — " + G.GOLD_VOL_NOTE.get(gsym, ""))

    tp = G.tape_state(xau, gsym, p, closed_only=g_closed)
    if not tp.get("ok"):
        st.info(tp.get("เหตุผล", "คำนวณไม่ได้"))
    else:
        ga = st.columns(4)
        ga[0].metric("สถานะเทป", tp["สถานะ"] if tp["สถานะ"] != "—"
                     else "ไม่เข้าเงื่อนไข")
        ga[1].metric("โหวต สะสม / กระจาย",
                     f"{tp['โหวตสะสม']} / {tp['โหวตกระจาย']} จาก {tp['เต็ม']}",
                     f"ต้องการ ≥{tp['ต้องการ']}", delta_color="off")
        ga[2].metric("Squeeze", "ON" if tp["squeeze_on"] else
                     (f"คลาย {tp['bars_sq']} แท่ง"
                      if tp["bars_sq"] is not None else "ไม่เคยเกิด"))
        ga[3].metric("ตำแหน่งในกรอบ 20 แท่ง",
                     f"{tp['pos_in_rng'] * 100:.0f}%"
                     if tp["pos_in_rng"] is not None else "—",
                     "สะสม ≤65% · กระจาย ≥35%", delta_color="off")
        st.caption(f"แถว **Squeeze / tape** ที่ควรเห็นบน TradingView: "
                   f"`{tp['pine_cell']}` · แท่งที่ใช้คำนวณ **{tp['bar_date']}** · "
                   f"คุณภาพวอลุ่ม {tp['vol_quality'] * 100:.0f}% "
                   f"(สัดส่วนแท่งที่มีวอลุ่ม > 0 ใน 100 แท่ง)")
        tabA, tabD = st.tabs(["ตรวจทีละข้อ — ฝั่งสะสม",
                              "ตรวจทีละข้อ — ฝั่งกระจาย"])
        for _tab, _rows in ((tabA, tp["rows_acc"]), (tabD, tp["rows_dist"])):
            with _tab:
                _df = pd.DataFrame(_rows)
                _df["ผ่าน"] = _df["ผ่าน"].map({True: "✅", False: "❌",
                                                None: "—"})
                st.dataframe(_df, hide_index=True, use_container_width=True)

        gfr = tp["frame"]
        gshow = gfr.tail(260)
        figa = go.Figure()
        figa.add_trace(go.Candlestick(x=gshow.index, open=gshow["Open"],
                                      high=gshow["High"], low=gshow["Low"],
                                      close=gshow["Close"], name=gsym))
        for _col, _nm, _sym, _c, _off in [
                ("squeeze_on", "สควีซ (บีบตัว)", "circle", "#2ac7e0", 0.985),
                ("acc_show", "สะสม (display only)", "square", "#e8c22a", 0.97)]:
            _p = gshow[gshow[_col]]
            if len(_p):
                figa.add_trace(go.Scatter(
                    x=_p.index, y=_p["Low"] * _off, mode="markers", name=_nm,
                    marker=dict(symbol=_sym, size=7, color=_c)))
        _pd = gshow[gshow["dist_show"]]
        if len(_pd):
            figa.add_trace(go.Scatter(
                x=_pd.index, y=_pd["High"] * 1.03, mode="markers",
                name="กระจาย (display only)",
                marker=dict(symbol="square", size=8, color="#d13fd1")))
        figa.update_layout(height=380, xaxis_rangeslider_visible=False,
                           margin=dict(l=10, r=10, t=30, b=10),
                           legend=dict(orientation="h"))
        if not seven_g:
            figa.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        plot(figa)
        st.caption(
            f"260 แท่งล่าสุด: สะสม {int(gshow['acc_show'].sum())} · "
            f"กระจาย {int(gshow['dist_show'].sum())} · "
            f"สควีซ {int(gshow['squeeze_on'].sum())} แท่ง — "
            "**จำนวนที่ขึ้นบ่อยไม่ได้แปลว่าใช้ได้** ยังไม่มีการทดสอบว่าแท่ง"
            "เหล่านี้ให้ผลตอบแทนต่างจากแท่งอื่น")
        with st.expander("ทำไม footprint ถึงไม่ถูกเอาไปใช้เข้าออร์เดอร์"):
            st.markdown(
                "- ต้นฉบับ v4.1 เขียนกำกับเองว่า footprint "
                "**\"NEVER enter, exit, size, or gate\"** และให้ **เกรด C**\n"
                "- ต้นฉบับเลือกปิด `trustVol` เพราะบน CFD ทองคำ volume "
                "คือจำนวน tick ไม่ใช่วอลุ่มจริง — เมื่อปิด เหลือโหวตราคาล้วน "
                "2 ข้อ (ราคานิ่ง + CLV) และต้องผ่านทั้งคู่\n"
                "- ฝั่ง squeeze: ต้นฉบับปิด `useSqzGate` เพราะระบุว่า edge "
                "เดี่ยว ๆ decayed (ตรงกับข้อสรุปฝั่งหุ้น v5.13 ซึ่งอ้าง "
                "Fang-Jacobsen-Qin, JPM 2017)\n"
                "- ในแอปนี้ข้อมูลไม่ใช่ CFD: `GC=F` เป็นวอลุ่มฟิวเจอร์สจริง "
                "แต่มี roll artifact · `PAXG-USD` เป็นวอลุ่มโทเคน — "
                "**ทั้งคู่ยังไม่แนะนำให้เปิด trustVol**")

    st.markdown("#### Backtest (สัญญาณปิดแท่ง → เข้า open แท่งถัดไป, "
                "หัก spread ทุกเทรด, แยกบัญชี swap"
                + (" · โหมด 7 วัน: swap 1 คืน/แท่ง = รวมสัปดาห์เทียบเท่า 5 คืน+triple พุธ" if seven_g else "") + ")")
    bt = G.backtest(fr, p)
    lvl, vmsg = G.validation_verdict(bt)
    {"fail": st.error, "warn": st.warning, "ok": st.success}[lvl]("🧮 " + vmsg)
    if bt["n"] > 0:
        lo, hi = bt["ci"]
        ret = bt["equity"].pct_change().dropna()
        dsr = (SE.deflated_sharpe(ret, int(g_trials))
               if len(ret) > 30 else float("nan"))
        b1 = st.columns(4)
        b1[0].metric("เทรด (ปิดแล้ว)", bt["n"],
                     f"ข้ามเพราะเบรก {bt['halted_days']} ครั้ง")
        b1[1].metric("Win rate", f"{bt['win_rate']:.0%}",
                     f"95% CI {lo*100:.0f}–{hi*100:.0f}%")
        b1[2].metric("PF ก่อน swap",
                     "∞" if not np.isfinite(bt["pf_pre_swap"])
                     else f"{bt['pf_pre_swap']:.2f}")
        b1[3].metric("PF หลัง swap",
                     "∞" if not np.isfinite(bt["pf_after_swap"])
                     else f"{bt['pf_after_swap']:.2f}", "เป้าสคริปต์ > 1.5")
        b2 = st.columns(4)
        b2[0].metric("Expectancy", f"{bt['expectancy_r']:+.2f} R")
        b2[1].metric("กำไรสุทธิ (หัก spread)", f"{bt['net_profit']:+,.0f} $",
                     f"swap รวม {bt['swap_total']:+,.0f} $")
        b2[2].metric("สุทธิหลัง swap", f"{bt['net_after_swap']:+,.0f} $")
        b2[3].metric("Max DD | PSR | DSR",
                     f"{bt['max_dd']:.0%} | {num(bt['psr'])} | {num(dsr)}")
        fige = go.Figure(go.Scatter(x=bt["equity"].index, y=bt["equity"],
                                    name="Equity $"))
        fige.update_layout(height=260,
                           title="เส้นทุน mark-to-market รายวัน (เริ่ม $10,000)",
                           margin=dict(l=10, r=10, t=40, b=10))
        plot(fige)
        with st.expander(f"📋 รายการเทรด ({bt['n']})"):
            st.dataframe(bt["trades"], hide_index=True)
        st.download_button("⬇️ Journal เทรดทอง (CSV)",
                           bt["trades"].to_csv(index=False)
                           .encode("utf-8-sig"),
                           "xau_rtp_journal.csv", "text/csv")
    explain_box(
        "กติกาเต็มของ v6.4 ที่พอร์ตมา (และจุดที่ต่างจากต้นฉบับ)",
        "**Tier A (เปิด):** regime = ปิด > SMA200 เมื่อวาน + slope 5 แท่ง | "
        "entry pullback: ราคา > EMA50, low แตะ EMA21 แล้วปิดกลับเหนือ, "
        "ปิด > เปิด, RSI ≥ 40 | score ≥ 40 (mom126 = 40, ER ≥ 0.30 = 30, "
        "ใกล้ high 252 วัน = 30) | DXY trend veto | vol-shock gate P90 | "
        "gap ≤ 1.5 dayATR | cost ≤ 10% ของ 1R\n\n"
        "**Stop/Trail:** เริ่ม 1.8×dayATR → Chandelier 22 แท่ง −3×ATR "
        "(ratchet ไม่ถอยหลัง) ไม่มี TP ตายตัว (ปล่อย runner)\n\n"
        "**Sizing:** เสี่ยง 1% × ตัวลด (vol-target เหนือ P80, DD ≥ 10% → ×0.5, "
        "แพ้ติด ≥ 2 → ×0.6, พื้น 0.2) | เบรก: DD เดือน ≥ 6% หรือแพ้ติด 5 → "
        "หยุดเดือนนั้น, เพดาน 20 เทรด/เดือน\n\n"
        "**Tier B (ปิดตั้งต้น):** US10Y gate, JPY+VIX carry veto, Weekly EMA, "
        "ER hard gate — เปิดได้ข้างบนแต่ต้อง validate เอง\n\n"
        "**ต่างจากต้นฉบับ (ตรงๆ):** ใช้ GC=F แทน spot | backtest หัก spread "
        "ในกำไรจริง (Pine ใช้เป็นแค่ gate) — ผลที่นี่จึง *เข้มกว่า* TradingView")




def page_swing():
    if not _guard(["set_swing", "accum"], "หน้า SET Swing"):
        return
    st.subheader("SET Swing v5.14 + Market/Stock Context — คณิตเดียวกับ Pine")
    st.caption("พอร์ตจากสคริปต์ของคุณแบบ same math, same defaults (Long only, "
               "risk 0.5%, BOS 20, score≥55, stop 2 ATR → Chandelier 22/3, "
               "TP1 OFF, kill 6%/แพ้ติด 5, เพดาน 6 เทรด/เดือน, โปรไฟล์ SET100 "
               "ทางการ H2-2026) | v5.13 port แล้ว: squeeze precondition (default "
               "OFF — ต้นฉบับปิดเองตั้งแต่ v5.8 เหตุ edge decayed post-2001), "
               "accumulation watch (display only เกรด C), PB entry (default OFF "
               "= breakout เดิมเป๊ะ) | **v5.14 ใหม่**: distribution/release watch "
               "(กระจกเงาของ accumulation — display only เกรด C เหมือนกัน) + "
               "F2 ลดไซส์/F3 บีบ trail ซึ่ง**ปิดทั้งคู่** จึงยังได้ออร์เดอร์เท่า "
               "v5.13 เป๊ะ | ที่ยังไม่ port: flow/FX/event/skew/HTF/"
               "ER-gate/breadth ฯลฯ (ต้นฉบับก็ปิด)")
    mc = CX.market_context(bench_close,
                           vix_close=CTX_VIX, usdthb_close=thb_close,
                           spx_close=CTX_SPX, eem_close=CTX_EEM)
    if mc.get("ok"):
        zc = st.success if "BUY" in mc["zone"] else \
            (st.error if "SELL" in mc["zone"] else st.info)
        zc(f"**Market Context: {mc['score']:+d} ({mc['zone']})** — "
           + " | ".join(f"{k} {v:+d}" for k, v in mc["parts"].items()))
        st.caption(f"🧭 {mc['align']}  ·  แรงผลักต่างชาติ (proxy) = {mc['press']:+d} "
                   f"{mc['press_votes']}"
                   + (f"  ·  ⚠️ ผันผวนดัชนี rank {mc['rv_rank']:.0f} (display)"
                      if mc["vol_risk"] else ""))
        if mc["flow_na"]:
            st.caption("ℹ️ sFlow (CMF ดัชนี) = 0 เพราะ yfinance ไม่มี volume ของ "
                       "^SET.BK — ใช้หน้า Fund Flow (ข้อมูลจริง) แทน")
        for cline in mc["calendar"]:
            st.caption("🗓️ " + cline)
        st.caption("⚠️ " + mc["note"])
    st.markdown("#### 📋 จัดลำดับทั้ง SET100 ตามกติกา v5.13 (Long only)")
    rank = C_swing_rank(set_period, tuple(sorted(set_prices)),
                        str(bench_close.index[-1].date()))
    if rank.empty:
        st.info("จัดลำดับไม่ได้ (ข้อมูลไม่พอ)")
    else:
        cnt = rank["บักเก็ต"].value_counts()
        mR = st.columns(4)
        for _i, _b in enumerate(SW.BUCKET_ORDER):
            mR[_i].metric(_b, int(cnt.get(_b, 0)))
        fsel = st.selectbox("กรองบักเก็ต", ["ทั้งหมด"] + SW.BUCKET_ORDER, 0)
        shr = rank if fsel == "ทั้งหมด" else rank[rank["บักเก็ต"] == fsel]
        st.dataframe(shr, hide_index=True, height=360)
        st.caption("⚠️ 🔴 = ไม่เข้าไม้ใหม่ — *ไม่ใช่คำสั่งขายของที่ถืออยู่* "
                   "(ของเดิมยึด stop ตามแผน) | จัดคิวทำการบ้าน ไม่ใช่คำแนะนำลงทุน")
    st.divider()
    names = sorted(set_prices.keys())
    c1, c2, c3 = st.columns([1.2, 1, 1.4])
    pick = c1.selectbox("หุ้น (SET100)", names,
                        format_func=lambda s: s.replace(".BK", ""), key="swpick")
    surv = c2.checkbox("ติดธง Cash Balance/Trading Alert (เช็ก set.or.th)", False)
    blk_txt = c3.text_input("วันงบ/XD (YYYY-MM-DD คั่น , — fail-open)", "")
    blk_dates = [x.strip() for x in blk_txt.split(",") if x.strip()]
    tkr = pick.replace(".BK", "")
    scp = CX.StockCtxParams()
    sc = CX.stock_context(set_prices[pick], bench_close, tkr, scp,
                          surv_flag=surv, blackout_dates=blk_dates)
    if not sc.get("ok"):
        st.error(sc.get("status", "ข้อมูลไม่พอ"))
        return
    met = "context met" in sc["status"]
    (st.success if met else st.warning)(
        f"**Stock Context:** {sc['status']}  ·  Regime {sc['regime']}  ·  "
        f"Conf L {sc['conf_l']} / S {sc['conf_s']} (thr 55)")
    lvmap = {"ok": "🟢", "bad": "🔴", "warn": "🟡", "na": "⚪"}
    st.dataframe(pd.DataFrame(
        [{"": lvmap[lv], "รายการ": a, "ค่า": b} for a, b, lv in sc["rows"]]),
        hide_index=True, height=420)
    st.divider()
    st.markdown("#### สถานะกลยุทธ์ Swing v5.14 (วันนี้)")
    eqty = st.number_input("ทุนจำลอง (บาท)", 10000.0, 1e9, 1_000_000.0, 50000.0)
    with st.expander("⚙️ ตัวเลือก v5.13/v5.14 (ค่าตั้งต้น = พฤติกรรม v5.11 เป๊ะ)"):
        e1, e2, e3 = st.columns(3)
        emode = e1.selectbox("Entry trigger",
                             ["Breakout (default)", "Pullback to zone (PB)"], 0)
        pband = e2.selectbox("PB zone preset",
                             ["Full 0.382-0.618 (default)", "Core 0.382-0.500",
                              "Deep 0.500-0.618"], 0)
        pkill = e3.slider("PB kill (สัดส่วน retrace)", 0.50, 1.00, 1.00, 0.01)
        e4, e5 = st.columns(2)
        pwin = e4.number_input("PB window (แท่ง)", 3, 100, 20, 1)
        usesq = e5.checkbox("เปิด squeeze precondition "
                            "(ต้นฉบับปิดตั้งแต่ v5.8 — edge decayed post-2001)",
                            False)
        if emode.startswith("Pull"):
            st.caption("⚠️ คำต้นฉบับตรง ๆ: PB คือ *execution tactic* ไม่ใช่ edge "
                       "ที่มีหลักฐาน — เลเวล fib ไม่พยากรณ์อะไร และเลกที่แรงสุด"
                       "มักไม่ย่อ (จะพลาด breakout ที่ดีที่สุดบางตัว) — "
                       "เทียบสองโหมดผ่าน backtest ก่อนเชื่อ")
        st.markdown("**v5.14 — distribution / release (ทั้งคู่ปิดเป็นค่าตั้งต้น)**")
        e6, e7 = st.columns(2)
        f2on = e6.checkbox("F2: ลดขนาดไม้ใหม่เมื่อ footprint ฝั่งตรงข้ามติด "
                           "(ลดอย่างเดียว)", False)
        f3on = e7.checkbox("F3: บีบ Chandelier trail เมื่อ distribution ติด "
                           "(ตอนถือ long)", False)
        if f2on or f3on:
            st.caption("⚠️ เปิดแล้ว backtest จะ**ไม่เท่า v5.13** อีกต่อไป — ต้อง "
                       "เทียบเป็น arm แยกกัน (อย่าเปิดพร้อมกันตอนทดสอบ) หัก"
                       "ต้นทุนไป-กลับ ~0.5% แล้วดู net/trade กับ DSR · "
                       "F3 บีบแล้วคลายไม่ได้จนจบไม้ (trail เป็น running max) และ"
                       "เอนจินนี้กินกำไรจากหางขวาของเทรนด์ ตัวช่วยออกทุกชนิด"
                       "ตัดหางขวาทิ้ง — ถ้าผลแย่ลง คำตอบที่ถูกคือปิดทิ้ง")
    swp = SW.SwingParams(
        surv_flag=surv,
        entry_mode="Pullback" if emode.startswith("Pull") else "Breakout",
        pb_band=("Core" if pband.startswith("Core") else
                 "Deep" if pband.startswith("Deep") else "Full"),
        pb_win=int(pwin), pb_kill=float(pkill), use_sqz=usesq,
        use_dist_size=bool(f2on), use_dist_trail=bool(f3on))
    fr = SW.compute_frame(set_prices[pick], bench_close, tkr, swp, blk_dates)
    stt = SW.state_today(fr, swp, eqty)
    if stt["triggered"]:
        st.success("🎯 " + stt["entry_note"])
    else:
        st.info(stt["entry_note"])
    ck = pd.DataFrame([{"": "✅" if okk else "❌", "เงื่อนไข": nm, "หมายเหตุ": dt}
                       for nm, okk, dt in stt["checklist"]])
    st.dataframe(ck, hide_index=True)
    _d = stt.get("dist") or {}
    if _d.get("on"):
        _dtxt = (f"🔴 **กระจายของ (release) ติดป้าย** — โหวต {_d['votes']}/4"
                 if _d.get("show") else
                 (f"◻️ กระจายของ: แท่งแรก (ยังไม่ครบ 2 แท่ง) — โหวต {_d['votes']}/4"
                  if _d.get("hot") else f"⚪ กระจายของ: ไม่ติด — โหวต {_d['votes']}/4"))
        if _d.get("pass_names"):
            _dtxt += " · ผ่าน: " + " + ".join(_d["pass_names"])
        _dtxt += (f" · แถวบน Pine = `{_d['pine_dash']}`")
        (st.warning if _d.get("show") else st.caption)(_dtxt)
        st.caption("ℹ️ **แสดงผลอย่างเดียว เกรด C — ไม่เข้า ไม่ออก ไม่ปิดกั้น "
                   "ไม่ปรับไซส์** (เว้นแต่คุณเปิด F2/F3 เอง) · ต้นฉบับ v5.14 "
                   "ระบุเองว่า Wyckoff distribution เป็น *plausible mechanism, "
                   "unproven as a signal* และ **ออร์เดอร์ VWAP/POV ที่รันดี ๆ "
                   "ตรวจไม่เจอจาก OHLCV** → “ไม่มีเครื่องหมาย” ไม่เท่ากับ "
                   "“ไม่มีคนขาย”"
                   + ("  ·  ⚠️ F2 เปิดอยู่: ไซส์ไม้ใหม่ถูกลด"
                      if _d.get("size_trim") else "")
                   + ("  ·  ⚠️ F3 เปิดอยู่: trail ถูกบีบ"
                      if _d.get("trail_on") else ""))
    e = stt["eff"]
    st.caption(f"โปรไฟล์ {e['sector']} V{e['vol_tier']} L{e['liq_tier']} → "
               f"พื้นสภาพคล่อง {e['liq_min']:.0f} ลบ., เพดานไซส์ {e['liq_max']:.1f}% "
               f"ของ ADV, VT target {e['vt_tgt']:.1f}% | stop {stt['sl_dist']} บาท "
               f"(2 ATR) | lot {stt['lot']} | size mult {stt['size_mult']}x → "
               f"~{stt['board_qty']:,} หุ้น ที่ risk 0.5%")
    with st.expander("🧪 Backtest v5.14 (ย้อนหลังช่วงข้อมูลที่โหลด)"):
        bt = SW.backtest(fr, swp, eqty)
        lvl, vmsg = SE.sample_verdict(bt["n"])
        {"fail": st.error, "warn": st.warning, "ok": st.success}[lvl]("🧮 " + vmsg)
        if bt["n"]:
            lo, hi = bt["ci"]
            ret = bt["equity"].pct_change().dropna()
            dsr = SE.deflated_sharpe(ret, int(n_trials)) if len(ret) > 30 \
                else float("nan")
            b1 = st.columns(4)
            b1[0].metric("เทรดปิดแล้ว", bt["n"])
            b1[1].metric("Win rate", f"{bt['win_rate']:.0%}",
                         f"CI {lo*100:.0f}–{hi*100:.0f}%")
            b1[2].metric("Profit Factor", "∞" if not np.isfinite(bt["pf"])
                         else f"{bt['pf']:.2f}")
            b1[3].metric("Expectancy", f"{bt['expectancy_r']:+.2f} R")
            b2 = st.columns(4)
            b2[0].metric("กำไรสุทธิ", f"{bt['net_thb']:+,.0f} ฿",
                         "หักต้นทุนไทยทุกข้าง")
            b2[1].metric("Max DD", f"{bt['max_dd']:.0%}")
            b2[2].metric("PSR", num(bt["psr"]))
            b2[3].metric(f"DSR (trials={int(n_trials)})", num(dsr))
            fge = go.Figure(go.Scatter(x=bt["equity"].index, y=bt["equity"]))
            fge.update_layout(height=240, margin=dict(l=10, r=10, t=20, b=10))
            plot(fge)
            st.dataframe(bt["trades"], hide_index=True)
        st.caption("⚠️ หุ้นเดียว in-sample + survivorship bias | งบ/XD ใช้วันที่"
                   "กรอกเอง (Pine ใช้ฟีดจริง) | self-test ledger ของ Market "
                   "Context ยังไม่ port — ความต่างจาก TradingView ที่เข้มกว่า: "
                   "หักต้นทุนไทยเต็มทุกข้างในกำไร")


def page_crypto():
    st.subheader("คริปโต — Research Toolkit v6 + รายงาน 2008-2026")
    st.error("**ไม่ใช่ระบบเข้าออก** — กรอบอ่านบริบทจาก heuristic ตัวอย่างเล็ก "
             "(~4 วัฏจักร) | เกณฑ์คาลิเบรตกับ BTC daily เท่านั้น | ตัวเลข "
             "on-chain ในกล่องหลักฐาน = snapshot จากรายงาน ไม่ใช่ค่าดึงสด",
             icon="⚠️")
    coin = st.selectbox("เหรียญ", ["BTC-USD", "ETH-USD", "SOL-USD"], 0)
    is_btc = coin == "BTC-USD"
    d = C_crypto(coin)
    stamp_add(f"ราคาคริปโต ({coin})", d, "24H", "crypto", str(coin),
              3600, "เดิน 24/7 — แท่งของวันปัจจุบันไม่เคยปิดจริง")
    if d.empty or len(d) < 380:
        st.error("ดึงราคาไม่ได้/ประวัติสั้นเกิน — ลองใหม่")
        return
    frc = CR.compute(d)
    stc = CR.state(frc, is_btc)
    m = st.columns(4)
    m[0].metric("ราคา", f"${stc['price']:,.0f}",
                f"ATH ${stc['ath']:,.0f}")
    m[1].metric("Regime (SMA200)", stc["regime"])
    if is_btc:
        m[2].metric("Mayer ×", f"{stc['mayer']}", stc["mayer_zone"]
                    + " (hot>2.4 / value<0.8)")
    else:
        m[2].metric("Mayer × (ดิบ)", f"{stc['mayer']}",
                    "เกณฑ์โซน = BTC เท่านั้น")
    m[3].metric("จาก ATH", f"{stc['dd_ath']:+.1f}%")
    m = st.columns(4)
    m[0].metric("RSI(14)", stc["rsi"])
    m[1].metric("Realized vol/ปี (30d)", f"{stc['ann_vol']:.0f}%")
    if is_btc:
        m[2].metric("Pi Cycle gap", f"{stc['pi_gap']:+.1f}%",
                    ("CROSS ใน 30 วัน!" if stc["pi_recent_cross"]
                     else "fast ยังใต้ 2×slow") + " · mixed reliability")
        m[3].metric("หลัง Halving 2024", f"+{stc['halv_days']} วัน",
                    "peak เดิมมัก +12-18 เดือน")
    else:
        m[2].metric("Pi / Halving", "—", CR.ALT_CAVEAT[:38] + "…")
    show = frc.tail(500)
    fg = go.Figure()
    fg.add_trace(go.Scatter(x=show.index, y=show["Close"], name=coin,
                            line=dict(width=1.4)))
    fg.add_trace(go.Scatter(x=show.index, y=show["reg_sma"], name="SMA200",
                            line=dict(width=1.2, dash="dot", color="orange")))
    if is_btc:
        fg.add_trace(go.Scatter(x=show.index, y=show["pi_fast"],
                                name="Pi fast 111", line=dict(width=1)))
        fg.add_trace(go.Scatter(x=show.index, y=show["pi_slow2"],
                                name="2×SMA350", line=dict(width=1, dash="dash")))
        px_ = show[show["pi_cross"]]
        if len(px_):
            fg.add_trace(go.Scatter(x=px_.index, y=px_["Close"], mode="markers",
                                    name="Pi cross", marker=dict(symbol="x",
                                    size=10, color="red")))
    fg.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                     legend=dict(orientation="h"))
    plot(fg)
    if not is_btc:
        st.warning("⚠️ " + CR.ALT_CAVEAT)
    with st.expander("⚖️ หลักฐานสองด้าน (จากรายงานที่แนบ)", expanded=True):
        st.dataframe(pd.DataFrame(CR.EVIDENCE_TWO_SIDES), hide_index=True)
    with st.expander("📌 Snapshot สำคัญจากรายงาน (กลางปี 2026 — ไม่ใช่ live)"):
        st.dataframe(pd.DataFrame(CR.REPORT_SNAPSHOT,
                                  columns=["ตัวชี้วัด", "ค่า ณ รายงาน"]),
                     hide_index=True)
    with st.expander("🎯 เกณฑ์เปลี่ยนมุมมอง (ตามรายงาน)"):
        st.markdown("**เอียง Bull ถ้า:**\n"
                    + "\n".join("- " + x for x in CR.BENCH_BULL)
                    + "\n\n**เอียง Bear ถ้า:**\n"
                    + "\n".join("- " + x for x in CR.BENCH_BEAR))
    with st.expander("🧨 ความเสี่ยงเชิงโครงสร้าง"):
        for r_ in CR.RISKS:
            st.markdown("- " + r_)
    st.caption("⚠️ " + CR.TOOL_DISCLAIMER)




@st.cache_data(ttl=1800, show_spinner="สแกน Swing ทั้ง SET100 (ครั้งแรก ~15-30 วิ)...")
def C_swing_rank(period: str, tickers_key: tuple, bench_key: str):
    return SW.rank_universe(set_prices, bench_close)


def _overall_default():
    picks = st.session_state.get("overall_picks")
    if picks:
        lst = [t for t in picks if t in set_prices][:10]
        if lst:
            return lst
    try:
        rank = C_swing_rank(set_period, tuple(sorted(set_prices)),
                            str(bench_close.index[-1].date()))
        good = rank[rank["บักเก็ต"].isin(SW.BUCKET_ORDER[:2])]["หุ้น"].tolist()
        lst = [t + ".BK" for t in good if t + ".BK" in set_prices][:10]
        if lst:
            return lst
    except Exception:
        pass
    return sorted(set_prices.keys())[:8]


def page_overall():
    st.subheader("Scan หุ้น Overall — Swing v5.13 × ปัจจัย 4 ตัว")
    st.caption("กติกาเปิดเผย: 'น่าลงทุนวันนี้' = ผ่านประตูสภาพคล่อง + Swing บักเก็ต "
               "🟢/🟡 + คะแนนปัจจัยรวม (composite z) ≥ 0 — ทั้งหมดคือคิวทำการบ้าน "
               "ไม่ใช่คำสั่งซื้อ และเกณฑ์รวมนี้ยังไม่ผ่าน validation เชิงประจักษ์")
    rank = C_swing_rank(set_period, tuple(sorted(set_prices)),
                        str(bench_close.index[-1].date()))
    board = SE.build_scoreboard(set_prices, bench_close, min_turn_m * 1e6)
    if rank.empty or board.empty:
        st.error("ข้อมูลไม่พอสำหรับจัดลำดับ")
        return
    j = rank.merge(board[["ticker", "composite", "n_factors", "liq_pass"]],
                   left_on="หุ้น", right_on="ticker", how="left") \
            .drop(columns="ticker")
    j["composite"] = pd.to_numeric(j["composite"], errors="coerce").round(2)
    good = j[j["บักเก็ต"].isin(SW.BUCKET_ORDER[:2])
             & j["liq_pass"].fillna(False) & (j["composite"] >= 0)]
    sig = good[good["บักเก็ต"] == SW.BUCKET_ORDER[0]]
    near = good[good["บักเก็ต"] == SW.BUCKET_ORDER[1]]
    st.session_state["overall_picks"] = [t + ".BK" for t in good["หุ้น"]][:12]
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 สัญญาณวันนี้", len(sig))
    c2.metric("🟡 ใกล้จุด (≤1 ATR)", len(near))
    c3.metric("ผ่านเกณฑ์รวม", len(good))
    if len(sig):
        st.markdown("**🟢 เงื่อนไขครบเมื่อปิดแท่งล่าสุด — ตามกติกาเข้า open แท่งถัดไป:**")
        st.dataframe(sig.drop(columns=["บักเก็ต"]), hide_index=True)
    if len(near):
        st.markdown("**🟡 ใกล้จุดซื้อ — เฝ้ารอเบรกจริง อย่าดักหน้า:**")
        st.dataframe(near.drop(columns=["บักเก็ต"]), hide_index=True)
    if not len(good):
        st.info("วันนี้ไม่มีตัวเข้าเกณฑ์รวม — เป็นเรื่องปกติของระบบเบรกเอาต์ "
                "(กติกาบอกให้รอ ไม่ใช่ให้หาเรื่องเข้า)")
    with st.expander("ตารางรวมทั้ง SET100 (Swing × ปัจจัย)"):
        st.dataframe(j, hide_index=True, height=460)
    st.caption("⚠️ composite z ≥ 0 = อยู่ครึ่งบนของตะกร้า ณ วันนี้เท่านั้น "
               "ไม่ใช่หลักฐานผลตอบแทน | เวิร์กโฟลว์: หน้านี้คัดตัว → Stock Context "
               "เช็กโครงสร้าง → รอกติกา Swing จริง | หุ้นชุดนี้ถูกใช้เป็นค่าตั้งต้น"
               "ของหน้า RRG ด้วย")


# ===========================================================================
# 🇹🇭 สแกน Accum+Squeeze (v5.13) — รายการเฝ้าดู ไม่ใช่สัญญาณซื้อ
# ===========================================================================

@st.cache_data(ttl=3600, show_spinner="สแกน Accumulation + Squeeze ทั้ง SET100...")
def C_accsq(period: str, tickers_key: tuple, bench_key: str,
            closed_only: bool = False):
    return SW.scan_acc_squeeze(set_prices, bench_close, closed_only=closed_only)


def page_set_accsq():
    if not _guard(["set_swing", "accum"], "หน้าสแกน Accum+Squeeze"):
        return
    st.caption("รายการเฝ้าดูจากสคริปต์ v5.13 — **ไม่ใช่สัญญาณซื้อ**: ต้นฉบับ"
               "ติดป้ายเองว่า accumulation watch เป็น *display only เกรด C* "
               "(\"NEVER enters, exits, sizes or gates\") และ squeeze edge "
               "*decayed post-2001* (เหตุที่ต้นฉบับปิด useSqz ตั้งแต่ v5.8) — "
               "ใช้จัดคิวทำการบ้าน แล้วรอเงื่อนไขเต็มของกติกา")

    live_bar = (SET_BAR_CLOSED is False)
    cA, cB = st.columns([2, 3])
    closed_only = cA.checkbox("ใช้เฉพาะแท่งที่ปิดแล้ว (โหมดเทียบ TradingView)",
                              value=live_bar,
                              help="ตัดแท่งวันปัจจุบันที่ยังวิ่งไม่จบออกก่อนคำนวณ "
                                   "— ค่าจะตรงกับชาร์ตตอนสิ้นวันมากขึ้น")
    if live_bar and not closed_only:
        cB.warning("⚠️ แท่งวันนี้ยังไม่ปิด — ตัวเลขจะขยับไปเรื่อยจนตลาดปิด "
                   "และจะไม่ตรงกับชาร์ต")

    tbl = C_accsq(set_period, tuple(sorted(set_prices)),
                  str(bench_close.index[-1].date()), closed_only)
    if tbl.empty:
        st.info("วันนี้ไม่มีตัวที่เข้าเงื่อนไขสะสม/สควีซ (หรือข้อมูลไม่พอ)")
        return

    st.error(
        "**อ่านก่อนเทียบกับชาร์ต — จุดที่เคยทำให้ 'ดูไม่ตรง':**\n\n"
        "1. Pine v5.13 พิมพ์เครื่องหมายบนชาร์ตแค่ **2 อย่าง** คือ "
        "สี่เหลี่ยมเหลือง (`accShow`) และวงกลมฟ้า (`squeezeOn`) — สถานะ "
        "**⚪ สะสมแท่งแรก** และ **🟠 เพิ่งคลายสควีซ** เป็นของหน้าจอนี้เอง "
        "**ไม่มีอะไรให้ทาบบนชาร์ต** (ดูคอลัมน์ 'มีเครื่องหมายบนชาร์ต')\n"
        "2. หุ้นที่โหวต 3-4/4 **วันนี้วันแรก** จะยังไม่ขึ้นสี่เหลี่ยมเหลือง เพราะ "
        "ต้นฉบับบังคับ `accShow = accHot and accHot[1]` (ติดกัน 2 แท่ง) — "
        "แต่ dashboard บน TradingView โชว์เลขโหวตให้เห็นตั้งแต่แท่งแรก "
        "จึงดูเหมือนแอปหาย ทั้งที่ตรงตามกติกา\n"
        "3. ราคาที่ใช้มาจาก **yfinance auto_adjust=True** (ปรับปันผล/แตกพาร์"
        "ย้อนหลัง) ส่วน TradingView ปรับคนละแบบตามการตั้งค่าของคุณ → "
        "แท่งรอบวัน XD ให้ผลต่างกันได้")

    cnt = tbl["สถานะ"].value_counts()
    mR = st.columns(len(SW.ACC_SQ_BUCKETS))
    for _i, _b in enumerate(SW.ACC_SQ_BUCKETS):
        mR[_i].metric(_b.split(" ", 1)[0], int(cnt.get(_b, 0)),
                      _b.split(" ", 1)[1], delta_color="off")
    n_chart = int((tbl["มีเครื่องหมายบนชาร์ต"] == "ใช่").sum())
    st.caption(f"ในตารางนี้ **{n_chart} จาก {len(tbl)} ตัว** มีเครื่องหมายจริงบน"
               f"ชาร์ต Pine · ที่เหลืออีก {len(tbl) - n_chart} ตัวเป็นสถานะที่"
               "หน้าจอนี้เพิ่มเอง · " + ACC.MARKER_NOTE)

    fsel = st.selectbox("กรองสถานะ", ["ทั้งหมด", "เฉพาะที่มีเครื่องหมายบนชาร์ต"]
                        + SW.ACC_SQ_BUCKETS, 0)
    if fsel == "ทั้งหมด":
        shw = tbl
    elif fsel.startswith("เฉพาะ"):
        shw = tbl[tbl["มีเครื่องหมายบนชาร์ต"] == "ใช่"]
    else:
        shw = tbl[tbl["สถานะ"] == fsel]
    st.dataframe(shw, hide_index=True, height=380)

    # ---------------- ตรวจสอบทีละข้อ เทียบกับ TradingView ได------------------
    with st.expander("🔍 ตรวจทีละข้อ — ทาบกับ dashboard บน TradingView"):
        pick = st.selectbox("หุ้นที่จะตรวจ", list(tbl["หุ้น"]), 0,
                            key="accsq_audit_pick")
        key = pick + ".BK"
        if key in set_prices:
            aud = SW.acc_audit(set_prices[key], bench_close, key,
                               closed_only=closed_only)
            k1, k2, k3 = st.columns(3)
            k1.metric("แถว Accum/conf ที่ควรเห็นบน TradingView", aud["pine_dash"])
            k2.metric("แท่งที่ใช้คำนวณ", aud["bar_date"])
            k3.metric("Squeeze", "ON" if aud["squeeze_on"] else
                      (f"คลาย {aud['bars_sq']} แท่ง"
                       if aud["bars_sq"] is not None else "ไม่เคยเกิด"))
            adf = pd.DataFrame(aud["rows"])
            adf["ผ่าน"] = adf["ผ่าน"].map({True: "✅", False: "❌", None: "—"})
            st.dataframe(adf, hide_index=True, use_container_width=True)
            if "dist_rows" in aud:
                st.markdown(f"**ฝั่งกระจายของ (v5.14) — แถว `Distrib (release)` "
                            f"บน TradingView ควรเป็น `{aud['dist_pine_dash']}`**")
                ddf = pd.DataFrame(aud["dist_rows"])
                ddf["ผ่าน"] = ddf["ผ่าน"].map({True: "✅", False: "❌", None: "—"})
                st.dataframe(ddf, hide_index=True, use_container_width=True)
                st.caption("ข้อ 1 และ 4 ใช้เกณฑ์ร่วมกับฝั่งสะสมโดยตั้งใจ "
                           "(ต้นฉบับไม่เพิ่ม input ชุดที่สองให้ตัวชี้วัดเกรด C) "
                           "· ต่างกันแค่ข้อ 2 กับ 3 ที่กลับด้าน")
            st.caption(
                f"ถ้าตัวเลขในคอลัมน์ 'ค่าที่วัดได้' ต่างจาก TradingView อย่างมี"
                "นัย แปลว่า **ข้อมูลดิบต่างกัน** (Yahoo vs ตลท.) ไม่ใช่สูตรต่างกัน "
                "— สูตรถูกตรวจแบบไล่ทีละแท่งแล้วว่าตรงกับ Pine v5.14 "
                "(ดู test_v14.py / test_v514.py) · ถ้าตรงกันแต่สถานะไม่ตรง "
                "ให้ดูข้อ 1-2 ด้านบน")
        else:
            st.caption("ไม่พบราคาของตัวนี้ในชุดที่โหลดมา")

    picks = st.multiselect("เลือกส่งเข้าห้องประชุม AI",
                           list(tbl["หุ้น"]), default=list(tbl["หุ้น"][:4]))
    if st.button("🏛️ ส่งชื่อไปหน้า 'AI Meeting หุ้น'"):
        st.session_state["meet_pick"] = picks
        st.success(f"ส่ง {len(picks)} ตัวแล้ว — เปิดหน้า 'AI Meeting หุ้น' "
                   "จากเมนูซ้าย")
    explain_box("โหวตสะสม 4 ข้อคืออะไร (นิยามตรงจากสคริปต์)",
                "1) **ราคานิ่ง**: ขยับสุทธิ 20 แท่ง ≤ 2 ATR · "
                "2) **วอลุ่มขาซื้อเด่น**: วอลุ่มวันบวก ≥ 1.25× วันลบ · "
                "3) **ปิดค่อนบน**: ค่าเฉลี่ย CLV ≥ +0.10 · "
                "4) **ตลาดไม่ตาย**: วอลุ่มเฉลี่ย 20 ≥ 0.7× ฐาน 100 แท่ง\n\n"
                "ต้อง ≥3/4 **สองแท่งติด** และอยู่ใต้ trigger + ครึ่งล่างของกรอบ "
                "(ตำแหน่ง ≤ 65%) จึงขึ้น 'สะสม'\n\n"
                "**ฝั่งกระจายของ (v5.14) = กระจกเงาเป๊ะ**: ข้อ 1 และ 4 เหมือนกัน, "
                "ข้อ 2 กลับเป็น *วอลุ่มวันลบ ≥ 1.25× วันบวก*, ข้อ 3 กลับเป็น "
                "*CLV เฉลี่ย ≤ −0.10*, บริบทกลับเป็น *เหนือ breakdown + "
                "ครึ่งบนของกรอบ (≥ 35%)* — ใช้ input ชุดเดียวกัน\n\n"
                "เหตุผลเชิงทฤษฎี: square-root impact law บอกว่า metaorder ทิ้ง"
                "รอยเท้าวอลุ่มหลายวัน — แต่คำเตือนของต้นฉบับ: เครื่องมือสาย "
                "OBV/AD หลักฐาน weak/mixed, Wyckoff เป็น anecdotal และออร์เดอร์ "
                "VWAP/POV ที่ทำดี ๆ **ตรวจไม่เจอ** — จึงเป็นแค่รายการเฝ้าดู")


# ===========================================================================
# 🇹🇭 AI Meeting หุ้น — วิเคราะห์หุ้นที่มีสัญญาณ (โมเดลเดียวเล่นหลายบท)
# ===========================================================================

def _stk_context(tickers: list[str]) -> dict:
    """ประกอบ context ต่อหุ้นจาก engine จริง (โปร่งใส — โชว์ทั้งก้อนบนจอ)"""
    fdf, _ = load_flows()
    flow = {}
    if len(fdf):
        try:
            fs = FL.flow_summary(fdf)
            flow = {str(r["กลุ่ม"]): {"สะสม20วัน_ลบ": r.get("สะสม 20 วัน"),
                                       "streak_วัน": r.get("ซื้อ/ขายติดกัน (วัน)"),
                                       "z_วันล่าสุด": r.get("z วันล่าสุด")}
                    for _, r in fs.iterrows()} if len(fs) else {}
        except Exception:
            flow = {}
    ctx = {"ณ_วันที่": str(datetime.now().date()),
           "กติกา": "SET Swing v5.13 (Long only, ค่าตั้งต้น)",
           "ข้อมูล": "ราคาสิ้นวัน yfinance — ไม่มีฟีดข่าวรายหุ้น",
           "global_overlay": {"label": overlay["label"],
                              "reasons": overlay["reasons"][:4]},
           "fund_flow": flow, "หุ้น": {}}
    for tk in tickers:
        key = tk if tk.endswith(".BK") else tk + ".BK"
        df = set_prices.get(key)
        if df is None or len(df) < 260:
            ctx["หุ้น"][tk] = {"error": "ข้อมูลไม่พอ"}
            continue
        p_ = SW.SwingParams()
        fr = SW.compute_frame(df, bench_close, tk, p_)
        r = fr.iloc[-1]
        stt = SW.state_today(fr, p_)
        fails = [nm for nm, okk, _d in stt["checklist"] if not okk]
        atr_ok = r["atr"] == r["atr"] and r["atr"] > 0
        dist = float((r["swing_hi"] - r["Close"]) / r["atr"]) if atr_ok else None
        try:
            bt = SW.backtest(fr, p_)
            btd = ({"เทรด": bt["n"], "win_rate": round(bt["win_rate"], 2),
                    "CI95": [round(x, 2) for x in bt["ci"]],
                    "PF": (None if not np.isfinite(bt["pf"])
                           else round(bt["pf"], 2)),
                    "expectancy_R": round(bt["expectancy_r"], 2)}
                   if bt["n"] else {"เทรด": 0})
        except Exception:
            btd = {"error": "คำนวณไม่ได้"}
        ctx["หุ้น"][tk] = {
            "ราคา": round(float(r["Close"]), 2),
            "regime": stt["regime"],
            "ConfL": int(r["conf_l"]) if r["conf_l"] == r["conf_l"] else None,
            "ห่าง_trigger_ATR": None if dist is None else round(dist, 2),
            "เงื่อนไขที่ยังไม่ผ่าน": fails,
            "squeeze_on": bool(r["squeeze_on"]),
            "squeeze_คลายมาแล้ว_แท่ง": (int(r["bars_sq"])
                                          if r["bars_sq"] == r["bars_sq"] else None),
            "สะสม": {"โหวต": f"{int(r['acc_votes'])}/4",
                      "ติดป้ายสะสม": bool(r["acc_show"]),
                      "ตำแหน่งในกรอบ_pct": (round(float(r["pos_in_rng"]) * 100)
                                              if r["pos_in_rng"] == r["pos_in_rng"]
                                              else None),
                      "หมายเหตุ": "proxy เกรด C — display only"},
            "กระจายของ": {"โหวต": f"{int(r['dist_votes'])}/4",
                           "ติดป้ายกระจาย": bool(r["dist_show"]),
                           "หมายเหตุ": "v5.14 กระจกเงาของสะสม — proxy เกรด C "
                                        "display only · VWAP/POV ที่รันดีตรวจไม่เจอ "
                                        "→ ไม่ติดป้าย ≠ ไม่มีคนขาย"},
            "vol_rank": (round(float(r["vol_rank"]))
                         if r["vol_rank"] == r["vol_rank"] else None),
            "ADV20_ลบ": (round(float(r["liq_val"]) / 1e6, 1)
                          if r["liq_val"] == r["liq_val"] else None),
            "โปรไฟล์": stt["eff"]["sector"],
            "backtest_หุ้นนี้": btd,
        }
    return ctx


def _meet_provider_box(pv: str, ns: str = "mm") -> dict | None:
    """กล่องกรอก key/model ต่อผู้ให้บริการหนึ่งเจ้า — คืน selection ถ้าเปิดใช้"""
    spec = LP.PROVIDERS[pv]
    on = st.checkbox(spec["th"], value=True, key=f"{ns}_on_{pv}")
    if not on:
        return None
    c1, c2 = st.columns([1, 1])
    with c1:
        key = st.text_input("API key", type="password", key=f"{ns}_key_{pv}",
                            help=f"ขอ key ฟรีที่ {spec['keys_url']}")
    with c2:
        mdl = st.text_input("model id", value=spec["default_model"],
                            key=f"{ns}_mdl_{pv}",
                            help=f"รายชื่อโมเดลเปลี่ยนบ่อย เช็คที่ {spec['models_url']}")
    st.caption(f"{spec['note']} · โควตาจริงดูที่ {spec['limits_url']}")
    if not key.strip():
        return None
    return {"provider": pv, "api_key": key.strip(), "model": mdl.strip()}


def page_stock_meeting():
    st.caption("**ก่อนใช้:** " + LP.HONESTY)
    mode = st.radio("โหมด", ["หลายค่าย (ฟรี — Gemini/Groq/OpenRouter)",
                             "Anthropic เจ้าเดียวเล่นหลายบท (แบบเดิม)"],
                    horizontal=False, key="mm_mode")
    multi = mode.startswith("หลายค่าย")

    names = sorted(k.replace(".BK", "") for k in set_prices)
    dflt = [t for t in st.session_state.get("meet_pick", []) if t in names][:6]
    if not dflt:
        try:
            rk = C_swing_rank(set_period, tuple(sorted(set_prices)),
                              str(bench_close.index[-1].date()))
            dflt = list(rk[rk["บักเก็ต"].isin(SW.BUCKET_ORDER[:2])]["หุ้น"])[:4]
        except Exception:
            dflt = []
    picks = st.multiselect("หุ้นที่จะเข้าวาระ (แนะนำ ≤ 6 — token/เวลาเพิ่มตามจำนวน)",
                           names, default=dflt)
    panel = st.multiselect("มุมมองที่ต้องไล่ให้ครบ", [q["id"] for q in SM.PERSONAS],
                           default=SM.DEFAULT_PANEL,
                           format_func=lambda i: next(q["th"] for q in SM.PERSONAS
                                                      if q["id"] == i))
    if not picks:
        st.info("ยังไม่เลือกหุ้น — เลือกเอง หรือกดส่งชื่อจากหน้า "
                "'สแกน Accum+Squeeze' / ใช้บักเก็ต 🟢🟡 อัตโนมัติ")
        return
    ctx = _stk_context(picks)
    cj = json.dumps(ctx, ensure_ascii=False)
    with st.expander("ข้อมูลที่ส่งให้ AI (โปร่งใส — ทั้งหมดจาก engine)"):
        st.code(json.dumps(ctx, ensure_ascii=False, indent=2))

    # ------------------------------------------------------------------
    if multi:
        st.markdown("#### ผู้ให้บริการ (ยิงคนละ 1 call — ประหยัดโควตา free tier)")
        sels = []
        for pv in LP.ORDER:
            with st.container(border=True):
                s = _meet_provider_box(pv)
                if s:
                    sels.append(s)
        ref = st.checkbox("เพิ่มรอบ 'ผู้ตัดสิน' เฉพาะจุดที่เห็นต่าง (+1 call)",
                          value=False, key="mm_ref")
        if len(sels) < 2:
            st.warning("เปิดใช้ + ใส่ key อย่างน้อย 2 เจ้า ถึงจะเทียบข้ามค่ายได้ "
                       "(เจ้าเดียวคือกลับไปเป็นความเห็นเดี่ยว)")
        if st.button(f"🏛️ เปิดประชุม ({len(sels) + (1 if ref else 0)} calls)",
                     disabled=not sels):
            msgs = [{"role": "user",
                     "content": MM.build_solo_prompt(panel, cj)}]
            results = []
            prog = st.progress(0.0)
            for i, s in enumerate(sels):
                with st.spinner(f"ถาม {LP.PROVIDERS[s['provider']]['th']} ..."):
                    r = LP.chat(s["provider"], s["api_key"], msgs,
                                model=s["model"], max_tokens=2200)
                ana, parsed = MM.parse_solo(r.get("text", ""))
                results.append({
                    "label": f"{LP.PROVIDERS[s['provider']]['th'].split(' ')[0]}"
                             f"/{r.get('model', '')}",
                    "ok": r["ok"], "error": r.get("error", ""),
                    "analysis": ana, "parsed": parsed,
                    "latency_s": r.get("latency_s", 0)})
                prog.progress((i + 1) / max(1, len(sels)))
            prog.empty()
            bundle = MM.collect(results)
            ref_txt = ""
            dis = MM.disagreement_list(bundle)
            if ref and dis and sels:
                s0 = sels[0]
                with st.spinner("รอบผู้ตัดสิน (ดูเฉพาะจุดที่เห็นต่าง)..."):
                    rr = LP.chat(s0["provider"], s0["api_key"],
                                 [{"role": "user",
                                   "content": MM.build_referee_prompt(dis, cj)}],
                                 model=s0["model"], max_tokens=1200)
                ref_txt = rr["text"] if rr["ok"] else f"(ผู้ตัดสินล้ม: {rr['error']})"
            st.session_state.setdefault("mm_meetings", []).insert(0, {
                "เวลา": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "หุ้น": picks, "results": results, "bundle": bundle,
                "ผู้ตัดสิน": ref_txt})

        for gi, rec in enumerate(st.session_state.get("mm_meetings", [])):
            b = rec["bundle"]
            with st.expander(f"{rec['เวลา']} · {', '.join(rec['หุ้น'])}",
                             expanded=(gi == 0)):
                st.markdown("**" + MM.headline(b) + "**")
                for f in b["failed"]:
                    st.error(f"{f['label']} — {f['error']}")
                rows = MM.agreement_rows(b)
                if rows:
                    st.markdown("**ตารางเทียบมติ** (เรียงจุดที่เห็นต่างขึ้นก่อน)")
                    st.dataframe(pd.DataFrame(rows).drop(columns=["เห็นต่าง"]),
                                 use_container_width=True, hide_index=True)
                    n_d = sum(1 for r in rows if r["เห็นต่าง"])
                    if n_d:
                        st.warning(f"⚠️ {n_d} ตัวที่โมเดลต่างค่ายเห็นไม่ตรงกัน — "
                                   "นี่คือส่วนที่มีข้อมูลมากที่สุด ไม่ใช่ส่วนที่ควรข้าม")
                if b["conflicts"]:
                    st.markdown("**ประเด็นค้านที่แต่ละเจ้ายกมา**")
                    for c in b["conflicts"]:
                        st.markdown(f"- {c}")
                if rec.get("ผู้ตัดสิน"):
                    st.markdown("**ผู้ตัดสิน (ชี้ว่าต้องดูตัวเลขอะไรเพิ่ม — ไม่ออกมติ)**")
                    st.markdown(rec["ผู้ตัดสิน"])
                for r in rec["results"]:
                    with st.expander(f"บทวิเคราะห์เต็ม — {r['label']} "
                                     f"({r.get('latency_s', 0)}s)"):
                        st.markdown(r.get("analysis") or "—")
                st.caption("⚠️ " + MM.DISCLAIMER)
        if st.session_state.get("mm_meetings"):
            st.download_button(
                "⬇️ ดาวน์โหลดประวัติ (JSON)",
                json.dumps(st.session_state["mm_meetings"],
                           ensure_ascii=False, indent=1, default=str),
                file_name="multi_meetings.json", mime="application/json")
        return

    # ------------------------------------------------------------------
    # โหมดเดิม: Anthropic เจ้าเดียวเล่นหลายบท (3 รอบ)
    st.caption("โหมดนี้ = โมเดลเดียวเล่นหลายบท ความเห็นไม่อิสระทางสถิติ")
    api_key = st.text_input("Anthropic API key", type="password",
                            key="stk_meet_key")
    if st.button("🏛️ เปิดประชุม (3 API calls)"):
        if not api_key:
            st.error("ต้องใส่ Anthropic API key")
        else:
            try:
                import anthropic
            except ImportError:
                st.error("ต้องติดตั้งก่อน: pip install anthropic")
                return
            client = anthropic.Anthropic(api_key=api_key)
            msgs = []
            try:
                with st.spinner("รอบ 1: ลูกทีมแถลง + ลงมติรายหุ้น..."):
                    msgs.append({"role": "user",
                                 "content": SM.build_round1_prompt(panel, cj)})
                    r1 = client.messages.create(model="claude-sonnet-4-6",
                                                max_tokens=2000, messages=msgs)
                    t1 = "".join(b.text for b in r1.content if b.type == "text")
                    msgs.append({"role": "assistant", "content": t1})
                with st.spinner("รอบ 2: โต้แย้งข้อขัดแย้ง..."):
                    msgs.append({"role": "user",
                                 "content": SM.build_round2_prompt()})
                    r2 = client.messages.create(model="claude-sonnet-4-6",
                                                max_tokens=1200, messages=msgs)
                    t2 = "".join(b.text for b in r2.content if b.type == "text")
                    msgs.append({"role": "assistant", "content": t2})
                with st.spinner("หัวหน้าทีมสรุป + กระดานมติ..."):
                    msgs.append({"role": "user",
                                 "content": SM.build_chair_prompt()})
                    r3 = client.messages.create(model="claude-sonnet-4-6",
                                                max_tokens=1600, messages=msgs)
                    t3 = "".join(b.text for b in r3.content if b.type == "text")
                ana, parsed = SM.parse_chair(t3)
                rec = {"เวลา": datetime.now().strftime("%Y-%m-%d %H:%M"),
                       "หุ้น": picks, "รอบ1": t1, "รอบ2": t2,
                       "หัวหน้าทีม": ana or t3, "parsed": parsed}
                st.session_state.setdefault("stock_meetings", []).insert(0, rec)
            except Exception as e_:
                st.error(f"เรียก API ไม่สำเร็จ: {e_}")
    hist = st.session_state.get("stock_meetings", [])
    if not hist:
        return
    st.markdown(f"#### ประวัติการประชุม ({len(hist)}) — อยู่ใน session นี้เท่านั้น")
    for gi, rec in enumerate(hist):
        with st.expander(f"{rec['เวลา']} · {', '.join(rec['หุ้น'])}",
                         expanded=(gi == 0)):
            pr = rec.get("parsed")
            if pr and pr.get("votes"):
                st.markdown("**กระดานมติ**")
                vcols = st.columns(min(4, max(1, len(pr["votes"]))))
                for vi, (tk, v) in enumerate(pr["votes"].items()):
                    with vcols[vi % len(vcols)]:
                        st.markdown(f":{SM.vote_style(v['มติ'])}[**{tk} · "
                                    f"{v['มติ']} {v['conf']}**]")
                        st.caption(v["เหตุผล"] or "—")
            st.markdown("**บทวิเคราะห์หัวหน้าทีม**"
                        + (f" (conf {pr['conf_รวม']})" if pr else ""))
            st.markdown(rec["หัวหน้าทีม"])
            if pr:
                if pr.get("ขัดแย้ง"):
                    st.markdown("**ข้อขัดแย้ง:** " + " · ".join(pr["ขัดแย้ง"]))
                if pr.get("คำสั่ง"):
                    st.markdown("**คำสั่ง (จำกัด 3 แบบ)**")
                    for o in pr["คำสั่ง"]:
                        st.markdown(f"- :{'green' if o['คำสั่ง'] == 'ทำตามกติกา' else 'orange' if o['คำสั่ง'] == 'ลดขนาด' else 'gray'}"
                                    f"[{o['หุ้น']} → **{o['คำสั่ง']}**] {o['เงื่อนไข']}")
            else:
                st.caption("⚠️ หัวหน้าทีมไม่ส่ง JSON ตามรูปแบบ — แสดงข้อความดิบแทน "
                           "(กดเปิดประชุมใหม่ได้)")
            with st.expander("ข้อมูลที่เสนอในที่ประชุม (รอบ 1-2 เต็ม)"):
                st.markdown("**รอบ 1 — ลูกทีมแถลง**\n\n" + rec["รอบ1"])
                st.markdown("---\n**รอบ 2 — โต้แย้ง**\n\n" + rec["รอบ2"])
    st.download_button("⬇️ ดาวน์โหลดประวัติ (JSON)",
                       json.dumps(hist, ensure_ascii=False, indent=1),
                       file_name="stock_meetings.json", mime="application/json")
    st.caption("⚠️ " + SM.DISCLAIMER)


# ===========================================================================
# 🔬 Self-Improve — อ่านผลลูปที่รันออฟไลน์ (ไม่รันในแอป โดยตั้งใจ)
# ===========================================================================
LOOP_STAGES = [
    ("1-2. Run Backtest + Trade Logs", "✅ ทำ", "set_swing.backtest ต่อหุ้น"),
    ("3-4. Audit Performance / Assessment", "✅ ทำ",
     "quant_evaluation: DSR, PBO (CSCV), t-stat, สภาวะตลาด"),
    ("5-6. Optuna tuning → Best Params", "✅ ทำ (ออฟไลน์)",
     "quant_optimize: พื้นที่ค้นหา 5 พารามิเตอร์ จดทะเบียนล่วงหน้า"),
    ("7-8. Validation Backtest", "⚠️ แก้จากผังเดิม",
     "ผังเดิมวัดผลบนข้อมูลชุดเดิม = ยืนยันตัวเอง เราเปลี่ยนเป็น walk-forward "
     "วัดบน fold ที่ยังไม่เคยเห็น + purge/embargo"),
    ("9. Deploy updated params → robot_trading", "❌ ไม่ทำโดยตั้งใจ",
     "ลูปที่ deploy ตัวเองได้ จะ deploy ผลของ noise ได้ด้วย — "
     "หยุดที่ 'ผู้สมัคร' ให้คนอนุมัติ แล้ว paper trade ก่อน"),
]


def page_self_improve():
    st.caption("ลูปนี้ **รันนอกแอป** (`python quant_optimize.py`) แล้วอัปผลมาดูที่นี่ — "
               "Streamlit Cloud ฟรีมีหน่วยความจำจำกัดและแอปหลับเมื่อไม่มีคนใช้ "
               "การรัน Optuna หลายร้อย trial ในแอปมักถูกฆ่ากลางคัน")
    with st.expander("ผังในภาพ vs สิ่งที่ระบบนี้ทำจริง", expanded=True):
        st.dataframe(pd.DataFrame(
            [{"ขั้นตอนในผัง": a, "สถานะ": b, "รายละเอียด": c}
             for a, b, c in LOOP_STAGES]),
            use_container_width=True, hide_index=True)
        st.warning("จุดที่ต้องเข้าใจก่อนใช้: ลูปแบบนี้ **โดยธรรมชาติคือเครื่องผลิต "
                   "overfitting** ยิ่งวนยิ่งเจอ 'ผลดีขึ้น' ที่เป็นเสียงรบกวน "
                   "ตัวนับ trial จึงสะสมข้ามการรันทุกครั้ง และ Deflated Sharpe "
                   "จะ **ยากขึ้น** ทุกครั้งที่คุณวนซ้ำ — นั่นคือพฤติกรรมที่ถูกต้อง "
                   "ไม่ใช่บั๊ก")

    st.markdown("#### วิธีรัน")
    st.code("pip install optuna            # ไม่ติดตั้งก็ได้ จะใช้ random search แทน\n"
            "python quant_optimize.py --prices-dir ./data --trials 100 --folds 5\n"
            "# หรือ\n"
            "python quant_optimize.py --yf --tickers PTT,AOT,CPALL --period 5y",
            language="bash")

    up = st.file_uploader("อัปโหลด optimize_result.json", type=["json"])
    if up is None:
        st.info("ยังไม่มีไฟล์ผล — รันคำสั่งด้านบนแล้วอัป optimize_result.json มาที่นี่")
        return
    try:
        res = json.load(up)
    except Exception as e:
        st.error(f"อ่านไฟล์ไม่ได้: {e}")
        return
    if res.get("error"):
        st.error(res["error"])
        return

    g = res.get("gate") or {}
    ok = bool(g.get("pass"))
    (st.success if ok else st.error)(
        ("✅ ผ่านประตูตัดสินทุกข้อ" if ok else "❌ ไม่ผ่านประตูตัดสิน")
        + f" · trial สะสมทั้งหมด {res.get('trials_สะสมทั้งหมด', 0):,} "
          f"(รอบนี้ {res.get('trials_รอบนี้', 0):,})")
    if ok:
        st.warning(QE.VERDICT_NOTE)

    c = st.columns(4)
    c[0].metric("หุ้น", len(res.get("หุ้น", [])))
    c[1].metric("แท่งข้อมูล", f"{res.get('จำนวนแท่ง', 0):,}")
    c[2].metric("OOS Sharpe รวม", res.get("oos_sharpe_รวม") or "—")
    c[3].metric("พื้นที่ค้นหา", f"{res.get('ขนาดพื้นที่ค้นหา', 0):,} ชุด")

    st.markdown("#### ประตูตัดสิน (ต้องผ่านครบทุกข้อ)")
    st.dataframe(pd.DataFrame([
        {"เกณฑ์": ch["ชื่อ"], "ผล": "✅" if ch["ผ่าน"] else "❌",
         "ค่าที่ได้": ch["ค่า"], "ต้องได้": ch["เกณฑ์"],
         "หมายเหตุ": ch["หมายเหตุ"]} for ch in g.get("checks", [])]),
        use_container_width=True, hide_index=True)

    pb = res.get("pbo") or {}
    if pb.get("note"):
        st.caption("PBO: " + pb["note"]
                   + (f" · {pb.get('n_splits', 0)} splits" if pb.get("n_splits") else ""))

    st.markdown("#### ผลรายช่วง walk-forward (train เลือกพารามิเตอร์ / test ไม่เคยเห็น)")
    folds = res.get("folds") or []
    if folds:
        st.dataframe(pd.DataFrame([
            {"ช่วง": f["fold"], "Sharpe (train)": f.get("train_sharpe"),
             "Sharpe (OOS)": f.get("oos_sharpe"),
             "เทรด OOS": f.get("oos_trades"),
             "กำไรสุทธิ OOS": f.get("oos_net_thb"),
             "พารามิเตอร์ที่เลือก": json.dumps(f.get("best_cfg", {}),
                                              ensure_ascii=False)}
            for f in folds]), use_container_width=True, hide_index=True)
        drop = [f for f in folds
                if f.get("train_sharpe") is not None
                and f.get("oos_sharpe") is not None
                and f["oos_sharpe"] < f["train_sharpe"] * 0.5]
        if drop:
            st.warning(f"⚠️ {len(drop)}/{len(folds)} ช่วง ที่ Sharpe ตกเกินครึ่ง"
                       "เมื่อออกนอกช่วงจูน — อาการคลาสสิกของ overfitting")
        cfgs = [json.dumps(f.get("best_cfg", {}), sort_keys=True) for f in folds]
        if len(set(cfgs)) > 1:
            st.info(f"พารามิเตอร์ที่ 'ดีที่สุด' เปลี่ยนไป {len(set(cfgs))} ชุด"
                    f"ในการจูน {len(folds)} ครั้ง — ถ้ามันเปลี่ยนทุกครั้ง "
                    "แปลว่ากำลังจับเสียงรบกวน ไม่ใช่โครงสร้างที่คงอยู่")

    with st.expander("พื้นที่ค้นหาที่จดทะเบียนไว้ (เปลี่ยนได้ใน quant_optimize.py)"):
        st.json(res.get("search_space", {}))
    with st.expander("ไฟล์ผลดิบ"):
        st.json(res)


# ===========================================================================
# 🌍 รายประเทศ (Bond) — yield 10 ปี, spread เทียบ US, คะแนนเสี่ยง (heuristic)
# ===========================================================================

@st.cache_data(ttl=3600, show_spinner="ดึง yield รายประเทศ (FRED fredgraph, ไม่ใช้ key)...")
def C_country_yields() -> tuple[dict, dict]:
    """ดึงทุกประเทศที่มี series — คืน ({code: [(date,val)...]}, {code: error})"""
    fetched, errs = {}, {}
    for spec in CO.COUNTRIES:
        if not spec.fred_series:
            continue
        try:
            fetched[spec.code] = CO.fetch_fred_series(spec.fred_series)
        except Exception as e:
            errs[spec.code] = str(e)
    return fetched, errs


def page_g_countries():
    st.caption("yield พันธบัตรรัฐบาล 10 ปีรายประเทศ + spread เทียบ US — หน้านี้"
               "ดึงข้อมูลเองผ่าน FRED fredgraph (ไม่ใช้ FRED key และทำงานได้แม้"
               "ปิดฝั่ง Global) | คะแนนเสี่ยงเป็น heuristic สูตรเปิดเผย "
               "**ยังไม่ผ่าน validation — ไม่ใช่สัญญาณซื้อขาย**")
    use_demo = st.toggle("โหมด DEMO (ตัวเลขตัวอย่าง — ไม่ใช่ข้อมูลจริง)", value=False)

    # กรอกเองสำหรับประเทศที่ไม่มีแหล่งฟรี
    if "ctry_manual" not in st.session_state:
        st.session_state["ctry_manual"] = {}
    manual = st.session_state["ctry_manual"]
    with st.expander("✍️ กรอก yield เอง (ประเทศที่ไม่มีแหล่งฟรี) — ติดป้าย MANUAL เสมอ"):
        st.caption("ใช้เมื่อคุณมีตัวเลขจากแหล่งที่คุณเชื่อถือเอง (เช่น เว็บข้อมูลบอนด์) "
                   "ระบบจะแสดงวันที่อ้างอิงกำกับ และเตือนว่ายังไม่ได้ตรวจสอบอิสระ | "
                   "ค่าอยู่ในหน้าจอรอบนี้เท่านั้น (session) ไม่ถูกเก็บถาวร")
        m1, m2, m3 = st.columns([2, 1, 1])
        m_code = m1.selectbox("ประเทศ", [c.code for c in CO.COUNTRIES
                                          if c.fred_series is None],
                              format_func=lambda c: f"{CO.spec_by_code(c).flag} "
                                                    f"{CO.spec_by_code(c).name_th}")
        m_y = m2.number_input("Yield 10Y (%)", 0.01, 59.99, 7.00, 0.05)
        m_asof = m3.date_input("ข้อมูล ณ วันที่", value=datetime.now().date(),
                               max_value=datetime.now().date())
        b1, b2 = st.columns(2)
        if b1.button("บันทึกค่า (เฉพาะหน้าจอนี้)"):
            okm, errm = CO.validate_manual(float(m_y), m_asof.isoformat(),
                                           datetime.now().date())
            if okm:
                manual[m_code] = {"y": float(m_y), "asof": m_asof.isoformat()}
                st.success(f"ใส่ค่า {m_code} = {m_y:.2f}% (ณ {m_asof}) แล้ว")
            else:
                st.error(errm)
        if b2.button("ล้างค่าที่กรอกทั้งหมด"):
            manual.clear()
            st.info("ล้างแล้ว")

    fetched, errs = ({}, {}) if use_demo else C_country_yields()
    if errs:
        st.warning("ดึงไม่สำเร็จบางประเทศ (แสดงตรง ๆ ไม่กลบ): "
                   + json.dumps(errs, ensure_ascii=False)[:400])
    rows = CO.build_rows(fetched, manual, datetime.now().date(), demo=use_demo)

    us_row = next(r for r in rows if r.spec.code == "US")
    if us_row.y is None:
        st.error("ยังไม่มีค่า US10Y (ฐานเทียบ) — ดึง FRED ไม่สำเร็จและไม่ได้เปิด DEMO "
                 "จึงคำนวณ spread/คะแนนไม่ได้")
    others = [r for r in rows if r.spec.code != "US"]
    others.sort(key=lambda r: -(r.risk.get("total", -1.0)
                                if r.risk.get("total") is not None else -1.0))

    def _card(r):
        with st.container(border=True):
            head = f"{r.spec.flag} **{r.spec.name_th}** · {r.spec.code}"
            st.markdown(head)
            if r.y is None:
                st.markdown("**— ไม่มีข้อมูล —**")
                st.caption(r.error or r.spec.note)
                return
            tier = r.risk.get("tier", "—")
            color = {"เสี่ยงต่ำ": "green", "เสี่ยงปานกลาง": "orange",
                     "เสี่ยงสูง": "red"}.get(tier, "gray")
            tot = r.risk.get("total")
            c1, c2 = st.columns([1, 1])
            c1.metric("คะแนนเสี่ยง (heuristic)",
                      "—" if tot is None else f"{tot:.0f}")
            c2.metric("10Y yield", f"{r.y:.2f}%",
                      (f"{r.spread:+.0f} bps vs US" if r.spread is not None
                       and r.spec.code != "US" else None), delta_color="off")
            st.markdown(f":{color}[**{tier}**]")
            rk = r.risk
            st.caption(f"ระดับ {rk.get('level_pts', 0):.0f}/40 · "
                       f"spread {rk.get('spread_pts', 0):.0f}/40 · "
                       f"แนวโน้ม3ด. {rk.get('trend_pts', 0):.0f}/20"
                       + ("" if rk.get("trend_known") else " (ไม่มีข้อมูลแนวโน้ม)"))
            src_th = {"FRED": "FRED", "MANUAL": "กรอกเอง", "DEMO": "DEMO",
                      "NONE": "—"}[r.source]
            st.caption(f"แหล่ง: {src_th} · {r.fresh_label}"
                       + (f" · {CO.FREQ_LABEL[r.spec.freq]}"
                          if r.source == "FRED" else ""))

    _card(us_row)
    cols3 = st.columns(3)
    for i, r in enumerate(others):
        with cols3[i % 3]:
            _card(r)

    with st.expander("📐 สูตรคะแนน + สถานะแหล่งข้อมูล (โปร่งใสทั้งหมด)"):
        st.text(CO.FORMULA_TEXT)
        st.dataframe(pd.DataFrame(
            [{"ประเทศ": f"{c.flag} {c.name_th}", "FRED series": c.fred_series or "—",
              "ความถี่": CO.FREQ_LABEL[c.freq] if c.fred_series else "—",
              "ยืนยันแล้ว": "✅" if c.verified else ("⚠️ ยัง" if c.fred_series else "—"),
              "หมายเหตุ": c.note} for c in CO.COUNTRIES]), hide_index=True)
        if st.button("🧪 ตรวจแหล่งข้อมูลตอนนี้ (ลองดึงทุก series แล้วรายงานตรง ๆ)"):
            for code, msg in CO.source_check_report():
                (st.success if msg.startswith("OK") else
                 (st.error if msg.startswith("FAIL") else st.info))(f"{code}: {msg}")
    explain_box("ทำไมบางประเทศเป็นรายเดือน/ไม่มีข้อมูล",
                "yield รายวันของประเทศ EM ไม่มี API ฟรีที่เชื่อถือได้ — ของฟรีที่มีจริง"
                "คือชุด OECD ผ่าน FRED ซึ่งเป็น **รายเดือนและช้า ~1-2 เดือน** "
                "(รัสเซียถูกหยุดเผยแพร่หลังปี 2022) ระบบเลือกแสดงความช้าตรง ๆ "
                "แทนการโชว์ตัวเลขที่ดูสดแต่ตรวจสอบไม่ได้ | spread คิดจาก yield "
                "คนละสกุลเงิน จึงสะท้อนทั้งความเสี่ยงเครดิต เงินเฟ้อ และค่าเงิน "
                "ปนกัน — อย่าอ่านเป็น 'ความเสี่ยงผิดนัด' อย่างเดียว")


# ===========================================================================
# 🌐 World Monitor (บริการภายนอก) — iframe + ปุ่มลิงก์สำรองเสมอ
# ===========================================================================

def page_worldmonitor():
    import streamlit.components.v1 as components
    st.caption("แผนที่สถานการณ์โลกจากโปรเจกต์โอเพนซอร์ส World Monitor "
               "(koala73/worldmonitor, AGPL-3.0) — **ข้อมูลภายนอก ไม่เกี่ยวกับ"
               "คะแนน/โมเดลของระบบนี้**")
    key = st.selectbox("เลือกมุมมอง", list(WM.VARIANTS.keys()),
                       index=list(WM.VARIANTS.keys()).index(WM.DEFAULT_VARIANT),
                       format_func=WM.variant_label)
    url = WM.variant_url(key)
    c1, c2 = st.columns(2)
    c1.link_button("↗ เปิดแท็บใหม่ (ชัวร์สุด)", url, use_container_width=True)
    c2.link_button("ซอร์สโค้ด (GitHub, AGPL-3.0)", WM.GITHUB_URL,
                   use_container_width=True)
    h = st.slider("ความสูงกรอบ (px)", 400, 1400, WM.IFRAME_HEIGHT, 20)
    try:
        components.iframe(url, height=h, scrolling=True)
    except Exception as e:
        st.error(f"ฝัง iframe ไม่ได้: {e} — ใช้ปุ่มเปิดแท็บใหม่ด้านบน")
    with st.expander("ข้อจำกัดที่ต้องรู้ (อ่านก่อนใช้)", expanded=True):
        for c_ in WM.CAVEATS:
            st.markdown(f"- {c_}")



# ===========================================================================
# 🥇 Gold Council — สภาผู้เชี่ยวชาญ XAU (เกต/คำตัดสินคำนวณด้วย Python)
# ===========================================================================
@st.cache_data(ttl=900, show_spinner=False)
def C_gold_tf(sym: str, period: str, interval: str):
    """โหลดแท่งตาม timeframe — intraday ของ yfinance มีเพดานย้อนหลัง"""
    try:
        import yfinance as yf
        raw = yf.download(sym, period=period, interval=interval,
                          auto_adjust=True, group_by="ticker", progress=False)
        if raw is None or len(raw) == 0:
            return pd.DataFrame()
        try:
            sub = SE.extract_ticker(raw, sym)
        except KeyError:
            sub = raw
        return SE.normalize_ohlc(sub)
    except Exception:
        return pd.DataFrame()


GC_LEAN_COLOR = {"BUY": "green", "SELL": "red", "NEUTRAL": "orange"}


def page_gold_council():
    st.subheader("🥇 Gold Council — สภาผู้เชี่ยวชาญ XAU")
    st.error("**อ่านก่อนใช้ — จุดที่ต่างจากภาพต้นแบบโดยตั้งใจ:** "
             "(1) ไม่มีคะแนนถ่วงน้ำหนักทศนิยม (เช่น BUY 4.68) เพราะน้ำหนักตั้งเอง "
             "และความเห็นแต่ละบทมาจากโมเดลเดียวกัน จึงไม่อิสระ — แสดงเป็น**จำนวนนับ** "
             "(2) ไม่มี CONFIDENCE % เดี่ยว เพราะ conf ที่ LLM เขียนเองไม่ใช่ความน่าจะเป็น "
             "ที่ calibrate แล้ว (3) PASS/VETO คำนวณด้วย **Python จากกติกา v6.4** "
             "ไม่ใช่ AI ตรวจตัวเอง (4) **สภาสร้างไม้เข้าเองไม่ได้** — กติกาไม่ครบ = "
             "NO TRADE เสมอ (5) บทที่ระบบไม่มีข้อมูล (News, Sentiment) ถูกบังคับให้งดออกเสียง")

    c1, c2, c3, c4 = st.columns(4)
    gsym = c1.selectbox("สัญลักษณ์", ["PAXG-USD", "GC=F"], 0,
                        help="ไม่มี spot XAUUSD จาก broker/MT5 ในระบบนี้")
    tf = c2.selectbox("Timeframe", ["1d (ตามที่ v6.4 ถูกจูนไว้)", "1h", "4h"], 0)
    interval = {"1d (ตามที่ v6.4 ถูกจูนไว้)": "1d", "1h": "1h", "4h": "4h"}[tf]
    gper = c3.selectbox("ช่วงข้อมูล", ["1y", "2y", "5y"] if interval == "1d"
                        else ["60d", "180d", "365d"], 1)
    min_rr = c4.number_input("Min stop ÷ spread", 1.0, 10.0, 2.0, 0.5)
    spread_c = st.number_input("Spread ไป-กลับ (¢/oz)", 0.0, 200.0, 25.0, 5.0)

    if interval != "1d":
        st.warning(f"⚠️ **{interval} เปลี่ยนความหมายของทุกอินดิเคเตอร์** — SMA200 "
                   f"กลายเป็น 200 แท่ง {interval} ไม่ใช่ 200 วัน และพารามิเตอร์ v6.4 "
                   "ถูกจูนบนแท่งวัน **ผล backtest ของ v6.4 จึงใช้อ้างอิงกับ timeframe นี้"
                   "ไม่ได้** · yfinance ยังจำกัดความยาวข้อมูล intraday (1h ~730 วัน) "
                   "และไม่มี bid/ask — ต่างจากภาพต้นแบบที่ดึงจาก MT5")

    df = C_gold_tf(gsym, gper, interval)
    if df.empty or len(df) < 260:
        st.error(f"โหลดข้อมูลไม่พอ ({len(df)} แท่ง) — ลองเปลี่ยนช่วงข้อมูล/สัญลักษณ์ "
                 "หรือ timeframe เป็น 1d")
        return
    dxy = None
    try:
        d_ = SE.load_single("DX-Y.NYB", "2y")
        if not d_.empty and "Close" in d_.columns:
            s_ = d_["Close"].dropna()
            dxy = s_.iloc[:, 0] if isinstance(s_, pd.DataFrame) else s_
    except Exception:
        pass

    gp = G.GoldParams()
    try:
        fr = G.compute_frame(df, dxy_close=dxy, p=gp)
        stt = G.state_today(fr, gp)
    except Exception as e:
        st.error(f"คำนวณ engine v6.4 ไม่ได้: {e}")
        return
    gate = GC.risk_gate(stt, min_rr=min_rr, spread_c=spread_c)
    sess = GC.session_of(df.index[-1])
    ctx = GC.build_context(gsym, interval, df, stt, gate)

    # ---------------- แถบหัว ----------------
    h = st.columns(5)
    h[0].metric("ราคาปิดแท่งล่าสุด", f"${float(df['Close'].iloc[-1]):,.2f}")
    h[1].metric("Regime", stt.get("regime", "—"))
    h[2].metric("RSI", stt.get("rsi") or "—")
    h[3].metric("ATR%", stt.get("atr_pct") or "—")
    h[4].metric("Vol rank (pct)", stt.get("vol_rank") or "—")
    st.caption(f"แท่งล่าสุด {df.index[-1]} · ช่วงตลาด: {sess['ช่วง']} — "
               f"{sess['หมายเหตุ']} · **ข้อมูลเป็นแท่งปิดย้อนหลัง ไม่ใช่ราคา "
               f"real-time และไม่มี bid/ask**")

    # ---------------- CHIEF VERDICT (คำนวณก่อน ไม่ต้องรอ AI) ----------------
    verdict0 = GC.chief_verdict(stt, gate, {"counts": {}})
    left, right = st.columns([2, 1])
    with left:
        ok = verdict0["verdict"].startswith("ตามกติกา")
        st.markdown(
            f"<div style='border:2px solid {'#2e7d32' if ok else '#8d6e63'};"
            "border-radius:10px;padding:14px 18px;background:#11150f'>"
            "<div style='letter-spacing:3px;font-size:12px;color:#9e9e9e'>"
            "CHIEF VERDICT · คำนวณจากกติกา v6.4 ไม่ใช่จากมติสภา</div>"
            f"<div style='font-size:34px;font-weight:700;color:"
            f"{'#66bb6a' if ok else '#d7b56d'}'>{verdict0['verdict']}</div>"
            f"<div style='color:#cfcfcf'>{verdict0['reason']}</div>"
            f"<div style='color:#9e9e9e;font-size:13px;margin-top:6px'>"
            f"{verdict0['override']}</div></div>", unsafe_allow_html=True)
    with right:
        (st.success if gate["pass"] else st.error)(
            ("✅ RISK GATE: PASS" if gate["pass"] else "⛔ RISK GATE: VETO")
            + (f" · stop÷spread = {gate['rr']}×" if gate.get("rr") else ""))
        st.dataframe(pd.DataFrame([
            {"เกต": c["ชื่อ"], "": "✅" if c["ผ่าน"] else "❌",
             "รายละเอียด": c["รายละเอียด"]} for c in gate["checks"]]),
            use_container_width=True, hide_index=True)

    with st.expander("เช็คลิสต์ v6.4 รายข้อ (ตัวเลขจริงจาก engine)"):
        st.dataframe(pd.DataFrame(stt.get("checklist") or []),
                     use_container_width=True, hide_index=True)
    with st.expander("ข้อมูลที่ส่งให้สภา (โปร่งใส — ทั้งหมดจาก engine)"):
        st.code(json.dumps(ctx, ensure_ascii=False, indent=2, default=str))

    # ---------------- ทะเบียนบท + แหล่งข้อมูล ----------------
    st.markdown("#### THE COUNCIL — 10 บท พร้อมเกรดหลักฐานและแหล่งข้อมูลจริง")
    st.dataframe(pd.DataFrame([
        {"กลุ่ม": s["group"], "บท": s["th"], "เกรด": s["grade"],
         "แหล่งข้อมูลจริง": s["source"],
         "ข้อจำกัด": s["caveat"] or "—"} for s in GC.SPECIALISTS]),
        use_container_width=True, hide_index=True)
    st.warning("บท **News/Macro** และ **Sentiment** ไม่มีแหล่งข้อมูลในระบบนี้ "
               "(ไม่มีฟีดข่าวทอง ไม่มี COT/ETF flow) — โค้ดบังคับให้เป็น NEUTRAL "
               "เสมอแม้ LLM จะตอบอย่างอื่น · **Pattern เกรด D** = การเล่าเรื่อง "
               "ไม่ใช่ detector ที่ backtest ได้")

    panel = st.multiselect("บทที่จะเรียกประชุม",
                           [s["id"] for s in GC.SPECIALISTS],
                           default=GC.DEFAULT_PANEL,
                           format_func=lambda i: GC.spec(i)["th"])

    st.markdown("#### ผู้ให้บริการ (ใช้ key ชุดเดียวกับหน้า AI Meeting)")
    sels = []
    for pv in LP.ORDER:
        with st.container(border=True):
            s = _meet_provider_box(pv)
            if s:
                sels.append(s)
    if st.button(f"⚔️ เปิดสภา ({len(sels)} calls)", disabled=not sels):
        cj = json.dumps(ctx, ensure_ascii=False, default=str)
        msgs = [{"role": "user",
                 "content": GC.build_council_prompt(panel, cj)}]
        runs = []
        for s in sels:
            with st.spinner(f"ถาม {LP.PROVIDERS[s['provider']]['th']} ..."):
                r = LP.chat(s["provider"], s["api_key"], msgs,
                            model=s["model"], max_tokens=2400)
            ana, parsed = GC.parse_council(r.get("text", ""))
            runs.append({"label": LP.PROVIDERS[s["provider"]]["th"].split(" ")[0],
                         "model": r.get("model", ""), "ok": r["ok"],
                         "error": r.get("error", ""), "analysis": ana,
                         "parsed": parsed})
        st.session_state["gc_runs"] = {
            "เวลา": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "สัญลักษณ์": gsym, "tf": interval, "runs": runs,
            "verdict_engine": verdict0}

    rec = st.session_state.get("gc_runs")
    if not rec:
        st.info("ยังไม่ได้เปิดสภา — CHIEF VERDICT ด้านบนคำนวณจาก engine แล้ว "
                "และ**ไม่เปลี่ยนตามมติสภา** สภาใช้เพื่อดูมุมที่อาจมองข้ามเท่านั้น")
        return

    st.markdown(f"#### ผลสภา · {rec['เวลา']} · {rec['สัญลักษณ์']} {rec['tf']}")
    ok_runs = [r for r in rec["runs"] if r["ok"] and r.get("parsed")]
    for r in rec["runs"]:
        if not (r["ok"] and r.get("parsed")):
            st.error(f"{r['label']} — {r['error'] or 'ตอบไม่ตรงรูปแบบ JSON'}")
    if not ok_runs:
        return

    main = ok_runs[0]
    tal = GC.tally(main["parsed"])
    v = GC.chief_verdict(stt, gate, tal)
    cc = st.columns(4)
    cc[0].metric("BUY", tal["counts"]["BUY"])
    cc[1].metric("NEUTRAL", tal["counts"]["NEUTRAL"])
    cc[2].metric("SELL", tal["counts"]["SELL"])
    cc[3].metric("conf ต่ำ-สูง",
                 f"{tal['conf_ต่ำสุด']}-{tal['conf_สูงสุด']}"
                 if tal["conf_ต่ำสุด"] is not None else "—")
    st.caption("นับเป็นจำนวนเต็มโดยตั้งใจ — ไม่แปลงเป็นคะแนนถ่วงทศนิยม เพราะ"
               "ความเห็นแต่ละบทมาจากโมเดลเดียวกัน จึงบวกกันเป็นหลักฐานไม่ได้")
    if tal["งดออกเสียง_ไม่มีข้อมูล"]:
        st.info("งดออกเสียงเพราะไม่มีข้อมูลในระบบ: "
                + ", ".join(tal["งดออกเสียง_ไม่มีข้อมูล"]))
    st.markdown(f"**ผลหลังฟังสภา:** {v['verdict']} · **{v['action']}** "
                f"— {v['override']}")

    for grp in GC.GROUPS:
        ids = [s["id"] for s in GC.SPECIALISTS
               if s["group"] == grp and s["id"] in main["parsed"]["specialists"]]
        if not ids:
            continue
        st.markdown(f"**{grp}**")
        for sid in ids:
            d = main["parsed"]["specialists"][sid]
            s_ = GC.spec(sid)
            st.markdown(
                f":{GC_LEAN_COLOR[d['lean']]}[**{s_['th']} · {d['lean']}**] "
                f"`เกรด {s_['grade']}` conf {d['conf']} — {d['เหตุผล']}")
            st.progress(min(100, max(0, d["conf"])) / 100.0)
            st.caption(f"อ้างอิง: {d['อ้างอิง'] or '— (ไม่ได้ระบุตัวเลข)'}")

    if main["parsed"].get("ข้อขัดแย้ง"):
        st.markdown("**ข้อขัดแย้งในสภา**")
        for x in main["parsed"]["ข้อขัดแย้ง"]:
            st.markdown(f"- {x}")
    if main["parsed"].get("ความเสี่ยงหลัก"):
        st.markdown("**ความเสี่ยงหลักที่สภายก**")
        for x in main["parsed"]["ความเสี่ยงหลัก"]:
            st.markdown(f"- {x}")

    if len(ok_runs) > 1:
        st.markdown("#### เทียบข้ามค่าย (เฉพาะ lean — verdict ไม่เปลี่ยนตามโมเดล)")
        rows = []
        for s_ in GC.SPECIALISTS:
            row = {"บท": s_["th"], "เกรด": s_["grade"]}
            leans = []
            for r in ok_runs:
                lv = (r["parsed"]["specialists"].get(s_["id"]) or {}).get("lean", "—")
                row[r["label"]] = lv
                leans.append(lv)
            row["ตรงกัน"] = "—" if "—" in leans else (
                "✅" if len(set(leans)) == 1 else "⚠️ ต่าง")
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)
        st.caption("โมเดลต่างค่าย = อิสระกว่าโมเดลเดียว แต่ยังไม่อิสระจริง "
                   "(คลังเทรนทับซ้อน context เดียวกัน) — ตรงกันไม่ได้แปลว่าถูก")

    for r in ok_runs:
        with st.expander(f"บทวิเคราะห์เต็ม — {r['label']} / {r['model']}"):
            st.markdown(r.get("analysis") or "—")
    st.caption("⚠️ " + GC.DISCLAIMER)



# ===========================================================================
# 💼 พอร์ตที่ถืออยู่ — 5 บัญชี, CSV, ปันผล/XD, แผนแก้หุ้นติด
# ===========================================================================
def _pool_lookup(pool: dict, ticker: str):
    """หา OHLCV จาก pool — เขียนไว้ใน app.py เองเพื่อไม่ให้พังถ้า portfolio.py
    บน repo เป็นเวอร์ชันเก่า · ห้ามใช้ or/and/if กับ DataFrame (pandas จะโยน
    ValueError) ต้องเทียบ `is None` เท่านั้น"""
    if not pool or not ticker:
        return None
    t = str(ticker).upper().replace(".BK", "").strip()
    for key in (f"{t}.BK", t):
        d = pool.get(key)
        if d is not None and len(d) > 0:
            return d
    return None


# เช็ค "ความสามารถ" ไม่ใช่เลขเวอร์ชัน — บอกได้ตรง ๆ ว่าไฟล์เก่าขาดอะไร
MODULE_NEEDS = {
    "pf_holdings": ("PF", ["VERSION", "COLUMNS", "DROPPED_COLS", "pick_frame",
                           "normalize", "enrich", "summary", "action_for",
                           "dividend_view", "average_down_plan",
                           "chandelier_stop", "to_csv", "from_csv"]),
    "gold_council": ("GC", ["VERSION", "SPECIALISTS", "risk_gate",
                            "chief_verdict", "parse_council", "tally"]),
    "multi_meeting": ("MM", ["VERSION", "build_solo_prompt", "collect",
                             "agreement_rows", "headline"]),
    "llm_providers": ("LP", ["VERSION", "PROVIDERS", "chat", "ORDER"]),
    "quant_evaluation": ("QE", ["VERSION", "gate_verdict", "cscv_pbo",
                                "walk_forward_splits"]),
    "accum": ("ACC", ["VERSION", "squeeze_frame", "accumulation_frame",
                      "status_label", "audit_rows", "footprint_v641",
                      "audit_rows_v641", "MARKER_NOTE",
                      "dist_audit_rows", "dist_label", "DIST_VOTE_KEYS"]),
    "datastamp": ("DS", ["VERSION", "describe", "render", "bar_is_closed",
                         "age_text", "staleness", "last_index", "fmt",
                         "now_th"]),
    "set_swing": ("SW", ["VERSION", "scan_acc_squeeze", "acc_audit",
                         "ACC_SQ_BUCKETS", "BUCKETS_WITH_PINE_MARKER",
                         "compute_frame", "state_today"]),
    "gold": ("G", ["VERSION", "GOLD_VOL_NOTE", "TAPE_DISCLOSURE",
                   "tape_frame", "tape_state", "compute_frame",
                   "state_today", "backtest"]),
}


def _guard(mods: list[str], what: str) -> bool:
    """ถ้าไฟล์บน repo ยังเป็นตัวเก่า -> บอกให้ชัดว่าไฟล์ไหน ขาดอะไร แล้วหยุด

    เหตุผลที่ต้องมี: การอัปไฟล์ทีละหลายตัวผ่านเว็บ GitHub บางครั้ง
    **ไฟล์ใหม่ถูกสร้างแต่ไฟล์เดิมไม่ถูกทับ** ทำให้ app.py ใหม่ไปเรียก
    ฟังก์ชันที่โมดูลเก่ายังไม่มี แล้วได้ TypeError/AttributeError ที่อ่านไม่ออก
    """
    rep = _module_report(only=mods)
    bad = [r for r in rep if not r["ok"]]
    if not bad:
        return True
    st.error(f"**ไฟล์บน repo ยังเป็นเวอร์ชันเก่า — {what} ทำงานไม่ได้**  \n"
             "อัปโหลดไฟล์ที่ระบุด้านล่างทับของเดิม แล้วกด **Reboot app**")
    for r in bad:
        st.write(f"- `{r['ไฟล์']}` · เวอร์ชันที่พบ: **{r['ver']}** · "
                 f"ขาด: `{'`, `'.join(r['ขาด'][:8])}`")
        st.caption(f"python โหลดมาจาก: `{r['path']}`")
    st.info("เช็ก 3 อย่างถ้าอัปแล้วยังขึ้นข้อความนี้: (1) ดู path ด้านบน — "
            "ถ้าไม่ใช่โฟลเดอร์ repo แปลว่ามีไฟล์ชื่อซ้ำบังอยู่ (2) ลบโฟลเดอร์ "
            "`__pycache__` บน repo ทิ้ง (3) เปิดไฟล์บน GitHub แล้วดูบรรทัดบน ๆ "
            "ว่ามีบรรทัด `VERSION = ...` ตรงกับที่ระบุใน README หรือยัง")
    st.caption("ตารางสถานะทุกโมดูลที่ระบบเช็ก:")
    st.dataframe(pd.DataFrame([
        {"ไฟล์": r["ไฟล์"], "เวอร์ชัน": r["ver"],
         "สถานะ": "✅ ครบ" if r["ok"] else "❌ ขาด " + ", ".join(r["ขาด"][:4]),
         "โหลดจาก": r["path"]} for r in _module_report()]),
        use_container_width=True, hide_index=True)
    return False


def _module_report(only: list[str] | None = None) -> list[dict]:
    """คืนสถานะโมดูลที่หน้านั้นใช้จริง พร้อม path ที่ python โหลดมาจริง

    ใช้ 'มีฟังก์ชัน/ตัวแปรที่ต้องใช้ครบไหม' แทนการเทียบเลขเวอร์ชัน
    เพราะเลขเวอร์ชันบอกได้แค่ว่าไม่ตรง ไม่ได้บอกว่าขาดอะไร
    """
    rows = []
    for mod, (alias, needs) in MODULE_NEEDS.items():
        if only is not None and mod not in only:
            continue
        obj = globals().get(alias)
        if obj is None:
            rows.append({"ไฟล์": f"{mod}.py", "ok": False,
                         "ขาด": ["import ไม่ได้"], "path": "—", "ver": "—"})
            continue
        miss = [a for a in needs if not hasattr(obj, a)]
        rows.append({"ไฟล์": f"{mod}.py", "ok": not miss, "ขาด": miss,
                     "path": getattr(obj, "__file__", "—"),
                     "ver": str(getattr(obj, "VERSION", "ไม่มี"))})
    return rows


@st.cache_data(ttl=1800, show_spinner=False)
def C_pf_prices(tickers: tuple, period: str):
    """ราคาหุ้นที่ถือซึ่งไม่ได้อยู่ใน universe SET100"""
    if not tickers:
        return {}, []
    return SE.load_universe_prices([f"{t}.BK" for t in tickers], period)


def _pf_engine_maps(held: list[str]):
    """คืน (ราคาล่าสุด, ข้อมูล engine, stop) ของหุ้นที่ถือ — ใช้ v5.13 เท่านั้น"""
    px, eng, stops, missing = {}, {}, {}, []
    pool = dict(set_prices)
    need = [t for t in held if f"{t}.BK" not in pool and t not in pool]
    if need:
        extra, failed = C_pf_prices(tuple(sorted(need)), set_period)
        pool.update(extra)
        missing = [t.replace(".BK", "") for t in failed]
    try:
        rk = C_swing_rank(set_period, tuple(sorted(set_prices)),
                          str(bench_close.index[-1].date()))
        rk = rk.set_index("หุ้น")
    except Exception:
        rk = pd.DataFrame()
    p = SW.SwingParams()
    for t in held:
        df = _pool_lookup(pool, t)
        if df is None or len(df) < 60:
            if t not in missing:
                missing.append(t)
            continue
        px[t] = round(float(df["Close"].iloc[-1]), 2)
        try:
            fr = SW.compute_frame(df, bench_close, t, p)
            stops[t] = PF.chandelier_stop(fr, p.tr_len, p.tr_mlt)
            eng[t] = {"Regime": ("UP" if bool(fr["regime_up"].iloc[-1])
                                 else ("DOWN" if bool(fr["regime_dn"].iloc[-1])
                                       else "MIXED"))}
        except Exception:
            eng[t] = {}
        if t in rk.index:
            eng.setdefault(t, {})["บักเก็ต"] = rk.loc[t, "บักเก็ต"]
    return px, eng, stops, missing


def page_portfolio():
    _rep = _module_report(only=["pf_holdings"])
    if any(not r["ok"] for r in _rep):
        bad = [r for r in _rep if not r["ok"]]
        st.error("**ไฟล์ `pf_holdings.py` บน repo ยังไม่ใช่ตัวใหม่** — "
                 "อัปโหลดไฟล์นี้ขึ้น repo แล้ว Reboot")
        for r in bad:
            st.write(f"- `{r['ไฟล์']}` · เวอร์ชันที่พบ: **{r['ver']}** · "
                     f"ขาด: {', '.join(r['ขาด'][:6])}")
            st.caption(f"python โหลดมาจาก: `{r['path']}`")
        st.info("ถ้าอัปแล้วยังขึ้นข้อความนี้ ให้ดู path ด้านบน — ถ้าไม่ใช่โฟลเดอร์ "
                "repo ของคุณ แปลว่ามีไฟล์ชื่อซ้ำบังอยู่ · และลองลบโฟลเดอร์ "
                "`__pycache__` บน repo ออกด้วย")
        return
    with st.expander("ตรวจไฟล์โมดูล (ถ้าสงสัยว่าอัปไม่ขึ้น)"):
        st.dataframe(pd.DataFrame([
            {"ไฟล์": r["ไฟล์"], "เวอร์ชัน": r["ver"],
             "สถานะ": "✅ ครบ" if r["ok"] else "❌ ขาด " + ", ".join(r["ขาด"]),
             "โหลดจาก": r["path"]} for r in _module_report()]),
            use_container_width=True, hide_index=True)
    st.warning("**อ่านก่อนใช้ 3 ข้อ:** (1) หน้านี้ **ไม่มีช่อง 'ราคาขายทำกำไร'** "
               "เพราะ v5.13 ออกด้วย trailing stop ไม่มี TP ตายตัว — ถ้าเติมเป้าราคา "
               "ผล backtest ทั้งหมดใช้อ้างอิงไม่ได้ (2) เครื่องคิดเลข 'ซื้อเฉลี่ย' "
               "คำนวณให้ตามที่ขอ แต่**ไม่มีการบอกว่าราคาไหนน่าเฉลี่ย** เพราะไม่มี"
               "หลักฐานรองรับ (3) 'ควรเติมตอนไหน' — ไม่มีคำตอบที่เชื่อถือได้ "
               "ฤดูกาล SET มีจริงแต่เล็กและไม่เสถียร")

    if "pf_df" not in st.session_state:
        st.session_state["pf_df"] = PF.empty_df()

    c1, c2 = st.columns([2, 1])
    up = c1.file_uploader("โหลดไฟล์พอร์ต (CSV)", type=["csv"], key="pf_up")
    if up is not None and c1.button("📥 นำเข้าไฟล์นี้"):
        d, probs = PF.from_csv(up.getvalue())
        st.session_state["pf_df"] = d
        for p_ in probs:
            st.warning(p_)
        st.success(f"นำเข้า {len(d)} แถว")
    if c2.button("➕ เพิ่มแถวว่าง"):
        st.session_state["pf_df"] = pd.concat(
            [st.session_state["pf_df"],
             pd.DataFrame([{c: None for c in PF.COLUMNS}])],
            ignore_index=True)

    st.markdown(f"#### แก้ไขข้อมูล (บัญชี 1-{PF.MAX_ACCOUNTS})")
    st.caption("แก้ในตารางได้เลย · ปันผลต่อหุ้นและวันที่ XD กรอกเองจากประกาศของบริษัท "
               "(ระบบไม่มีฟีดปันผล) · กด 'บันทึกตาราง' แล้วดาวน์โหลดเก็บไว้")
    # รายชื่อหุ้นใน drop-down = universe ที่ตั้งไว้ + ตัวที่มีอยู่ในไฟล์แล้ว
    # (ถ้าถือหุ้นนอก universe ให้เพิ่มชื่อในช่อง "รายชื่อหุ้น" ที่ sidebar ก่อน)
    _uni = {str(t).replace(".BK", "").upper() for t in tickers}
    _have = {str(x).upper() for x in st.session_state["pf_df"]["หุ้น"].dropna()}
    _opts = sorted(_uni | _have)
    edited = st.data_editor(
        st.session_state["pf_df"], num_rows="dynamic",
        use_container_width=True, key="pf_editor",
        column_config={
            "บัญชี": st.column_config.NumberColumn(min_value=1,
                                                   max_value=PF.MAX_ACCOUNTS,
                                                   step=1, default=1),
            "หุ้น": st.column_config.SelectboxColumn(options=_opts,
                                                     help="เลือกจากรายชื่อ · "
                                                          "ถ้าไม่มีตัวที่ถือ ให้เพิ่มใน "
                                                          "'รายชื่อหุ้น' ที่แถบข้างก่อน"),
            "จำนวนหุ้น": st.column_config.NumberColumn(min_value=0.0, step=1.0,
                                                       format="%.2f"),
            "ราคาต้นทุน": st.column_config.NumberColumn(min_value=0.0, step=0.01,
                                                        format="%.4f"),
            "ปันผลต่อหุ้น": st.column_config.NumberColumn(min_value=0.0, step=0.01,
                                                          format="%.4f"),
            "วันที่ซื้อ": st.column_config.DateColumn(),
        })
    st.caption(f"รายชื่อใน drop-down มี {len(_opts)} ตัว (จาก universe ที่ตั้งไว้) · "
               "ช่องราคาและปันผลใส่ทศนิยมได้ถึง 4 ตำแหน่ง")
    b1, b2 = st.columns([1, 2])
    if b1.button("💾 บันทึกตาราง"):
        d, probs = PF.normalize(edited)
        st.session_state["pf_df"] = d
        for p_ in probs:
            st.warning(p_)
        st.success(f"บันทึกแล้ว {len(d)} แถว")
    b2.download_button("⬇️ ดาวน์โหลด CSV",
                       PF.to_csv(st.session_state["pf_df"]),
                       file_name=f"portfolio_{datetime.now():%Y%m%d}.csv",
                       mime="text/csv")

    df = st.session_state["pf_df"]
    if df.empty:
        st.info("ยังไม่มีข้อมูล — กรอกในตารางหรือนำเข้าไฟล์ CSV")
        return

    held = sorted(df["หุ้น"].dropna().unique().tolist())
    with st.spinner("ดึงราคาและคำนวณ engine v5.13..."):
        px, eng, stops, missing = _pf_engine_maps(held)
    if missing:
        st.error("ดึงราคาไม่ได้/ข้อมูลสั้นเกินไป: " + ", ".join(missing)
                 + " — แถวเหล่านี้จะคำนวณมูลค่าไม่ได้")

    en = PF.enrich(df, px, eng, stops)
    accounts = sorted(en["บัญชี"].unique().tolist())
    pick = st.multiselect("เลือกบัญชีที่จะแสดง", accounts, default=accounts)
    view = en[en["บัญชี"].isin(pick)] if pick else en

    s = PF.summary(view)
    if s:
        m = st.columns(5)
        m[0].metric("มูลค่าตลาดรวม", f"{s['มูลค่าตลาดรวม']:,.0f}")
        m[1].metric("ต้นทุนรวม", f"{s['ต้นทุนรวม']:,.0f}")
        m[2].metric("กำไร/ขาดทุน", f"{s['กำไร/ขาดทุน']:,.0f}",
                    f"{s['%']}%" if s.get("%") is not None else None)
        m[3].metric("จำนวนตัว", s["จำนวนตัว"])
        m[4].metric("กระจุกสูงสุด", s.get("กระจุกสูงสุด") or "—")

    st.markdown("#### หุ้นที่ถืออยู่")
    st.dataframe(view, use_container_width=True, hide_index=True,
                 column_config={
                     "มูลค่าตลาด": st.column_config.NumberColumn(format="%.0f"),
                     "ต้นทุนรวม": st.column_config.NumberColumn(format="%.0f"),
                     "กำไร/ขาดทุน": st.column_config.NumberColumn(format="%.0f")})
    st.caption("⚠️ " + PF.STOP_NOTE)

    st.markdown("#### การจัดกลุ่มตามกติกา v5.13")
    acts = []
    for _, r in view.iterrows():
        a = PF.action_for(r["ราคาล่าสุด"], r["ราคาต้นทุน"], r["stop ระบบ"],
                          r.get("Regime"), r.get("บักเก็ต"))
        acts.append({"บัญชี": r["บัญชี"], "หุ้น": r["หุ้น"],
                     "ราคาล่าสุด": r["ราคาล่าสุด"], "%": r["%"],
                     "stop ระบบ": r["stop ระบบ"], "การจัดกลุ่ม": a["action"],
                     "เหตุผล": a["เหตุผล"], "ที่มา": a["ที่มา"]})
    st.dataframe(pd.DataFrame(acts), use_container_width=True, hide_index=True)
    st.info(PF.NO_TP_NOTE)

    # ---------------- ปันผล / XD ----------------
    st.markdown("#### ปันผล / ถือข้าม XD คุ้มไหม")
    wht = st.slider("ภาษีเงินปันผลที่ใช้คำนวณ (%)", 0.0, 35.0, 10.0, 0.5,
                    help="10% = หัก ณ ที่จ่ายมาตรฐาน · ถ้าเลือกรวมคำนวณปลายปี"
                         "และใช้เครดิตภาษีเงินปันผล ให้ใส่ฐานภาษีจริงของคุณ") / 100.0
    dv_rows = []
    for _, r in view.iterrows():
        if not r.get("ปันผลต่อหุ้น"):
            continue
        d = PF.dividend_view(r["จำนวนหุ้น"], r["ราคาต้นทุน"], r["ราคาล่าสุด"],
                             r["ปันผลต่อหุ้น"], None, wht=wht)
        if not d.get("มีข้อมูล"):
            continue
        dv_rows.append({
            "หุ้น": r["หุ้น"], "ปันผล/หุ้น": r["ปันผลต่อหุ้น"],
            "yield ปัจจุบัน %": d.get("yield ราคาปัจจุบัน %"),
            "yield on cost %": d.get("yield on cost %"),
            "ปันผลสุทธิ (บาท)": d.get("ปันผลสุทธิ"),
            "จุดคุ้มทุน": d["จุดคุ้มทุน"].replace("**", "")})
    if dv_rows:
        st.dataframe(pd.DataFrame(dv_rows), use_container_width=True,
                     hide_index=True)
        st.markdown(f"**เกณฑ์ตัดสิน:** ที่ภาษี {wht * 100:.0f}% การถือข้าม XD "
                    f"คุ้มก็ต่อเมื่อราคาหลัง XD ลงน้อยกว่า **{(1 - wht) * 100:.0f}%** "
                    "ของเงินปันผล · ถ้าราคาลงเท่าปันผลพอดี คุณ**ขาดทุนเท่ากับภาษี**")
    else:
        st.caption("ยังไม่มีแถวไหนกรอกปันผลต่อหุ้น")
    st.warning(PF.DIVIDEND_NOTE)

    # ---------------- หุ้นติด ----------------
    st.markdown("#### หุ้นที่ติดอยู่ — เครื่องคิดเลขการเฉลี่ย")
    stuck = view[(view["%"].notna()) & (view["%"] < 0)]
    if stuck.empty:
        st.success("ไม่มีตัวที่ติดลบในบัญชีที่เลือก")
    else:
        st.dataframe(pd.DataFrame([
            {"หุ้น": r["หุ้น"], "ต้นทุน": r["ราคาต้นทุน"],
             "ราคาล่าสุด": r["ราคาล่าสุด"], "ขาดทุน %": r["%"],
             "ต้องขึ้นอีก % ถึงเท่าทุน":
                 PF.breakeven_gain_pct(r["ราคาล่าสุด"], r["ราคาต้นทุน"]),
             "Regime": r.get("Regime"), "stop ระบบ": r["stop ระบบ"]}
            for _, r in stuck.iterrows()]),
            use_container_width=True, hide_index=True)
        tk = st.selectbox("เลือกตัวที่จะคำนวณ", stuck["หุ้น"].tolist())
        row = stuck[stuck["หุ้น"] == tk].iloc[0]
        q1, q2 = st.columns(2)
        def _num(v, dflt=0.0):
            # NaN เป็น truthy ใน Python → `v or 0` ใช้ไม่ได้ ต้องเช็ค NaN ตรง ๆ
            try:
                f = float(v)
                return dflt if f != f else f
            except (TypeError, ValueError):
                return dflt
        add_q = q1.number_input("จำนวนหุ้นที่จะซื้อเพิ่ม", 0.0, 1e9,
                                _num(row["จำนวนหุ้น"]), 100.0)
        add_p = q2.number_input("ราคาที่จะซื้อ", 0.0, 1e6,
                                _num(row["ราคาล่าสุด"]), 0.25)
        plan = PF.average_down_plan(row["จำนวนหุ้น"], row["ราคาต้นทุน"],
                                    add_q, add_p,
                                    port_value=s.get("มูลค่าตลาดรวม"))
        if plan.get("ok"):
            k = st.columns(4)
            k[0].metric("ต้นทุนเฉลี่ยใหม่", plan["ต้นทุนเฉลี่ยใหม่"],
                        f"{plan['ต้นทุนเฉลี่ยใหม่'] - float(row['ราคาต้นทุน']):+.4f}")
            k[1].metric("เงินที่ต้องใส่เพิ่ม", f"{plan['เงินที่ต้องใส่เพิ่ม']:,.0f}")
            k[2].metric("ต้องขึ้น % ถึงเท่าทุน",
                        f"{plan['หลังเฉลี่ยต้องขึ้น %']}%",
                        f"{plan['หลังเฉลี่ยต้องขึ้น %'] - plan['เดิมต้องขึ้น % ถึงเท่าทุน']:+.2f}")
            k[3].metric("ถ้าลงต่ออีก 10% เจ็บเพิ่ม",
                        f"{plan['ถ้าลงต่ออีก 10% ขาดทุนเพิ่ม (หลัง)']:,.0f}",
                        f"{plan['ถ้าลงต่ออีก 10% ขาดทุนเพิ่ม (หลัง)'] - plan['ถ้าลงต่ออีก 10% ขาดทุนเพิ่ม (ก่อน)']:+,.0f}",
                        delta_color="inverse")
            st.caption("ขาดทุนที่ยังไม่รับรู้ ณ ราคานี้ "
                       f"{plan['ขาดทุนที่ยังไม่รับรู้ ณ ราคานี้ (ก่อน)']:,.0f} → "
                       f"{plan['ขาดทุนที่ยังไม่รับรู้ ณ ราคานี้ (หลัง)']:,.0f} บาท "
                       "— **ไม่เปลี่ยน** เพราะซื้อที่ราคาตลาด สิ่งที่โตคือความไวต่อการลงต่อ")
            if "น้ำหนักในพอร์ต หลัง %" in plan:
                st.caption(f"น้ำหนักในพอร์ต {plan['น้ำหนักในพอร์ต ก่อน %']}% → "
                           f"**{plan['น้ำหนักในพอร์ต หลัง %']}%** · "
                           f"ค่าธรรมเนียมซื้อ {plan['ค่าธรรมเนียมซื้อ']:,.0f} บาท")
            st.error(PF.AVERAGE_DOWN_WARNING)
            st.markdown("**คำถามที่ต้องตอบก่อน (ระบบไม่ตอบให้):**")
            for i, q in enumerate(PF.AVERAGE_DOWN_CHECKLIST, 1):
                st.markdown(f"{i}. {q}")
            eng_r = eng.get(tk, {})
            if eng_r.get("Regime") == "DOWN":
                st.error(f"⚠️ engine ระบุ regime ของ {tk} เป็น **ขาลง** — "
                         "กติกา v5.13 ไม่เข้าฝั่ง long ในสภาวะนี้ตั้งแต่ต้น")
            elif str(eng_r.get("บักเก็ต", "")).startswith("🟢"):
                st.info(f"engine มีสัญญาณเข้าใหม่ของ {tk} วันนี้ — ถ้าจะซื้อ "
                        "ให้คิดเป็น**ไม้ใหม่**ที่มี stop ของตัวเอง ไม่ใช่การเฉลี่ยของเดิม")
        else:
            st.caption(plan.get("เหตุผล", ""))
    st.info(PF.TIMING_NOTE)
    st.caption("⚠️ " + PF.DISCLAIMER)


# ===========================================================================
# Routing
# ===========================================================================
ROUTES = {
    "ภาพรวม SET + Overlay": page_set_overview,
    "Fund Flow นักลงทุน": page_set_flow,
    "SET Swing v5.13 + Context": page_swing,
    "💼 พอร์ตที่ถืออยู่": page_portfolio,
    "สแกน Accum+Squeeze": page_set_accsq,
    "AI Meeting หุ้น": page_stock_meeting,
    "🔬 Self-Improve (ผลออฟไลน์)": page_self_improve,
    "Scan หุ้น Overall": page_overall,
    "สแกน SET100": page_set_scan,
    "กราฟรายตัว": page_set_chart,
    "RRG": page_set_rrg,
    "Backtest + ตรวจสอบ": page_set_backtest,
    "Scenario ไทย": page_set_scenario,
    "ฤดูกาล (TOM)": page_set_tom,
    "ต้นทุนไทย": page_set_cost,
    "ภาพรวมโลก": page_g_overview,
    "โมเดลทำกำไร (6)": page_g_models,
    "สัญญาณ Global": page_g_signals,
    "จำลองสถานการณ์": page_g_scenario,
    "วิกฤตแบงก์รัน": page_g_bankrun,
    "ข่าวสาร": page_g_news,
    "ห้องประชุม AI": page_g_meeting,
    "ข้อมูลมหภาค": page_g_macro,
    "รายประเทศ (Bond)": page_g_countries,
    "Trend สินทรัพย์โลก": page_g_trend,
    "World Monitor": page_worldmonitor,
    "ทองคำ XAU (RTP v6.4.1)": page_gold,
    "🥇 Gold Council": page_gold_council,
    "คริปโต (BTC/ETH)": page_crypto,
    "Trade Log & สถิติ": page_tradelog,
    "คู่มืออ่านค่า": page_glossary,
    "ข้อจำกัด & จุดพัฒนา": page_limits,
}

st.title(page)

# --- v14: แถบเวลาข้อมูลด้านบนทุกหน้า (ชุดที่โหลดร่วมกันทุกแท็บ) ---
_top = (f"🕒 **{DS.fmt(DS.now_th(), short=True)}** (เวลาไทย) · "
        f"หุ้นไทยแท่งล่าสุด **{DS_ITEMS[0]['แท่งล่าสุด']}** "
        f"({DS_ITEMS[0]['อายุ']})")
if SET_BAR_CLOSED is False:
    st.warning(_top + " — ⚠️ **แท่งวันนี้ยังไม่ปิด** ค่าที่คำนวณได้ยังเปลี่ยนได้อีก "
               "และจะไม่ตรงกับชาร์ต TradingView ตอนสิ้นวัน")
elif DS_ITEMS[0]["ระดับ"] != "ok":
    st.warning(_top + " — ⚠️ ข้อมูลเก่ากว่าที่ควร ตรวจอินเทอร์เน็ต/กด "
               "'โหลดข้อมูลใหม่' ที่แถบซ้าย")
else:
    st.caption(_top + " · แท่งล่าสุดปิดแล้ว")

ROUTES[page]()
st.divider()
DS.render(st, DS_ITEMS)
st.caption(DISCLAIMER + " | Global: "
           + ("ปิด" if not g_on else ("DEMO ⚠️" if is_demo else "LIVE"))
           + " | SET: LIVE (yfinance, สิ้นวัน)")
