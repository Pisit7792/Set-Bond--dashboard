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
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import bridge as BR
import data_sources as D
import engine as E
import explain as X
import meeting as MT
import models6 as M
import scenario as SC
import set_engine as SE
import signals as SG

st.set_page_config(page_title="SET × Bond Crisis", page_icon="🛡️", layout="wide")

DISCLAIMER = ("เครื่องมือเพื่อการศึกษา/วินิจฉัยความเสี่ยง ไม่ใช่คำแนะนำการลงทุน — "
              "P(วิกฤต|สัญญาณเตือน) ≈ 50% ในวรรณกรรม EWS และ <3-5% ของรายย่อย"
              "ทำกำไรจากสัญญาณได้สม่ำเสมอ (Barber et al. 2020)")

ZONES = {
    "🇹🇭 SET (หุ้นไทย)": ["ภาพรวม SET + Overlay", "สแกน SET100", "กราฟรายตัว",
                          "RRG", "Backtest + ตรวจสอบ", "Scenario ไทย",
                          "ฤดูกาล (TOM)", "ต้นทุนไทย"],
    "🌍 Global (Bond Crisis)": ["ภาพรวมโลก", "โมเดลทำกำไร (6)", "สัญญาณ Global",
                                "จำลองสถานการณ์", "วิกฤตแบงก์รัน", "ข่าวสาร",
                                "ห้องประชุม AI", "ข้อมูลมหภาค", "Trend สินทรัพย์โลก"],
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


# ---------------------------------------------------------------------------
# โหลด + คำนวณฝั่ง SET (ทำงานเสมอ — ต้องมีอินเทอร์เน็ต)
# ---------------------------------------------------------------------------
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
thb_close = thb["Close"].dropna() if ("Close" in thb.columns and len(thb)) \
    else pd.Series(dtype=float)
if isinstance(thb_close, pd.DataFrame):
    thb_close = thb_close.iloc[:, 0]

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


def S(sid: str) -> pd.Series:
    s = fred.get(sid)
    return s if isinstance(s, pd.Series) else pd.Series(dtype=float)


composite = float("nan")
subs, warnings_g, mscores, deltas = {}, [], {}, {}
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
    names = sorted(set_prices.keys())
    pick = st.selectbox("เลือกหุ้น", names, format_func=lambda s: s.replace(".BK", ""))
    d = set_prices[pick]
    c = d["Close"].dropna()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.62, 0.2, 0.18], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"],
                                 low=d["Low"], close=d["Close"], name="ราคา"),
                  row=1, col=1)
    for n_, dash in [(20, "solid"), (50, "dash"), (200, "dot")]:
        fig.add_trace(go.Scatter(x=c.index, y=SE.sma(c, n_), name=f"SMA{n_}",
                                 line=dict(width=1, dash=dash)), row=1, col=1)
    if "Volume" in d.columns:
        fig.add_trace(go.Bar(x=d.index, y=d["Volume"], name="ปริมาณ", opacity=0.5),
                      row=2, col=1)
    fig.add_trace(go.Scatter(x=c.index, y=SE.rsi_series(c), name="RSI(14)",
                             line=dict(width=1)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_width=1, row=3, col=1)
    fig.update_layout(height=620, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h"))
    plot(fig)
    explain_box("อ่านกราฟอย่างไร — และอย่าอ่านเกินจริงอย่างไร",
                "- แท่งเทียน: เขียว = ปิด>เปิด | ไส้ = high/low ของวัน\n"
                "- SMA20/50/200 = แนวโน้มสั้น/กลาง/ยาว\n"
                "- RSI >70 ร้อนแรง / <30 อ่อนแรง — เป็น *บริบท* "
                "(หลักฐานว่า RSI เดี่ยวๆ ทำกำไรหลังต้นทุน: **อ่อน**)\n"
                "- กราฟใช้หา 'จังหวะ' ของหุ้นที่ทำการบ้านพื้นฐานแล้ว "
                "ไม่ใช่เครื่องตัดสินคุณภาพกิจการ")


def page_set_rrg():
    st.subheader("RRG — แผนที่ความแข็งแรงเทียบดัชนี (ภาพรวม ไม่ใช่สัญญาณ)")
    st.warning("สูตรเป็นการประมาณ (ไม่ใช่ JdK ต้นฉบับซึ่งเป็นกรรมสิทธิ์) และไม่มี "
               "peer-review ว่า RRG เดี่ยวๆ ทำกำไร — ใช้เป็น situational awareness")
    picks = st.multiselect("เลือกหุ้น (≤ 12 อ่านง่าย)", sorted(set_prices.keys()),
                           default=sorted(set_prices.keys())[:8],
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
        "1. **หน้า import ยอดซื้อขายต่างชาติ/NVDR แบบ CSV** — ข้อมูลมีฟรีบนเว็บ SET "
        "แต่ไม่มี API; ให้ผู้ใช้วางเองแล้วระบบวิเคราะห์ให้\n"
        "2. **Validate overlay ด้วย event study** — วัด lead-lag ระหว่าง composite/"
        "โมเดลเสี่ยง กับผลตอบแทน SET ล่วงหน้า (ระวัง: เหตุการณ์วิกฤตมีน้อย, n เล็ก)\n"
        "3. **Walk-forward / purged CV** สำหรับ backtest — จำเป็นก่อนใช้เงินจริง\n"
        "4. **Point-in-time constituents** แก้ survivorship bias (ข้อมูลไม่ฟรี)\n"
        "5. **Per-stock spread จริง** แทนค่าคงที่ — สำคัญมากกับ SET51-100\n"
        "6. **ดึง OHLC เต็มฝั่ง Global** ให้ ATR แม่นขึ้น\n"
        "7. **Calibration log ห้องประชุม AI** — บันทึกมติเทียบผลจริง สะสมหลักฐาน"
        "ว่าห้องประชุมช่วยจริงไหม\n"
        "8. **Stale-data detector** — เตือนเมื่อ ticker ใดหยุดอัปเดตเงียบๆ\n"
        "9. **Cache ราคาแบบ parquet** — โหลด 100 ตัวครั้งแรก ~2 นาที ลดได้มาก\n"
        "10. **Auto-อัปเดต universe จากเว็บ SET** — ทำได้แต่เปราะ (scraping) "
        "จึงยังเลือกวิธีแก้มือที่โปร่งใสกว่า")
    st.divider()
    st.caption(DISCLAIMER)


# ===========================================================================
# Routing
# ===========================================================================
ROUTES = {
    "ภาพรวม SET + Overlay": page_set_overview,
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
    "Trend สินทรัพย์โลก": page_g_trend,
    "Trade Log & สถิติ": page_tradelog,
    "คู่มืออ่านค่า": page_glossary,
    "ข้อจำกัด & จุดพัฒนา": page_limits,
}

st.title(page)
ROUTES[page]()
st.divider()
st.caption(DISCLAIMER + " | Global: "
           + ("ปิด" if not g_on else ("DEMO ⚠️" if is_demo else "LIVE"))
           + " | SET: LIVE (yfinance, สิ้นวัน)")
