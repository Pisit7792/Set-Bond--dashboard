# -*- coding: utf-8 -*-
"""รันหน้าจอจริงของ app.py แบบออฟไลน์ ด้วย streamlit/yfinance ปลอม
เป้าหมาย: จับ error ระดับ runtime ที่ py_compile จับไม่ได้ (ชื่อคอลัมน์ผิด,
พารามิเตอร์ API ผิด, ตัวแปรไม่ถูกนิยาม ฯลฯ)

ข้อจำกัดที่ต้องรู้: นี่ไม่ได้พิสูจน์ว่าแอปทำงานถูกบน Streamlit Cloud จริง
มันพิสูจน์แค่ว่า 'โค้ดเดินผ่านโดยไม่ระเบิด' ด้วยข้อมูลสังเคราะห์เท่านั้น
"""
import sys
import types
from datetime import datetime

import numpy as np
import pandas as pd

CALLS = {"error": [], "warning": [], "exception": []}


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _ColCfg:
    def __getattr__(self, n):
        return lambda *a, **k: None


class _El(_Ctx):
    """widget ปลอมที่ 'คืนค่าจริง' สำหรับ input ทุกชนิด (ไม่ใช่คืน object)"""
    column_config = _ColCfg()

    # ---- input widgets: ต้องคืนค่าที่ใช้ต่อได้ ----
    def radio(self, label, opts, index=0, *a, **k):
        opts = list(opts)
        return opts[index if isinstance(index, int) and index < len(opts) else 0] \
            if opts else None

    def selectbox(self, label, opts, index=0, *a, **k):
        opts = list(opts)
        return opts[index if isinstance(index, int) and index < len(opts) else 0] \
            if opts else None

    def multiselect(self, label, opts, default=None, *a, **k):
        return list(default) if default is not None else []

    def checkbox(self, label, value=False, *a, **k):
        return bool(value)

    def toggle(self, label, value=False, *a, **k):
        return bool(value)

    def button(self, *a, **k):
        return False

    def form_submit_button(self, *a, **k):
        return False

    def download_button(self, *a, **k):
        return False

    def number_input(self, label, mn=0.0, mx=1.0, val=0.0, *a, **k):
        return k.get("value", val)

    def slider(self, label, mn=0.0, mx=1.0, val=0.0, *a, **k):
        return k.get("value", val)

    def select_slider(self, label, opts=(), value=None, *a, **k):
        return value if value is not None else (list(opts)[0] if opts else None)

    def text_input(self, label="", value="", *a, **k):
        return value

    def text_area(self, label="", value="", *a, **k):
        return value

    def date_input(self, label="", value=None, *a, **k):
        return value or datetime.now().date()

    def time_input(self, label="", value=None, *a, **k):
        return value

    def file_uploader(self, *a, **k):
        return None

    def color_picker(self, label="", value="#000000", *a, **k):
        return value

    def data_editor(self, data=None, *a, **k):
        return data if data is not None else pd.DataFrame()

    # ---- containers ----
    def columns(self, n, *a, **k):
        n = n if isinstance(n, int) else len(n)
        return [_El() for _ in range(n)]

    def tabs(self, names, *a, **k):
        return [_El() for _ in names]

    def expander(self, *a, **k):
        return _El()

    def container(self, *a, **k):
        return _El()

    def form(self, *a, **k):
        return _El()

    def empty(self, *a, **k):
        return _El()

    def spinner(self, *a, **k):
        return _El()

    def status(self, *a, **k):
        return _El()

    def popover(self, *a, **k):
        return _El()

    def __getattr__(self, name):
        def f(*a, **k):
            if name in CALLS:
                CALLS[name].append(a[0] if a else "")
            return _El()
        return f

    def __call__(self, *a, **k):
        return _El()

    def __iter__(self):
        return iter([_El() for _ in range(8)])

    def __getitem__(self, i):
        return _El()


class _Sidebar(_El):
    pass


class _ST(_El):
    sidebar = _Sidebar()
    session_state = {}
    secrets = {}
    column_config = _ColCfg()

    def cache_data(self, *a, **k):
        if a and callable(a[0]):
            return a[0]

        def deco(fn):
            return fn
        return deco

    cache_resource = cache_data

    def stop(self):
        raise RuntimeError("st.stop() ถูกเรียก — ข้อมูลโหลดไม่ผ่าน")

    def rerun(self):
        raise RuntimeError("st.rerun()")


def _mk(n, seed, base=30.0, vol=0.016):
    rng = np.random.default_rng(seed)
    c = [base]
    for _ in range(n - 1):
        c.append(max(base * 0.2, c[-1] * (1 + rng.normal(0.0003, vol))))
    c = np.array(c)
    o = c * (1 + rng.normal(0, 0.004, n))
    h = np.maximum(o, c) * (1 + abs(rng.normal(0, 0.006, n)))
    l = np.minimum(o, c) * (1 - abs(rng.normal(0, 0.006, n)))
    v = rng.lognormal(14.5, 0.5, n)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c,
                         "Volume": v},
                        index=pd.bdate_range(end=pd.Timestamp.today().normalize(),
                                             periods=n))


class _YF(types.ModuleType):
    @staticmethod
    def download(tickers, period="5y", **k):
        n = 900
        if isinstance(tickers, str):
            tickers = [tickers]
        frames = {}
        for i, t in enumerate(tickers):
            base = 1900.0 if ("GC" in t or "PAXG" in t or "XAU" in t) else 30.0
            frames[t] = _mk(n, 1000 + i, base)
        return pd.concat(frames, axis=1)

    class Ticker:
        def __init__(self, t):
            self.t = t

        def history(self, *a, **k):
            return _mk(900, 7)

        @property
        def info(self):
            return {}


def install_stubs():
    st = _ST()
    sys.modules["streamlit"] = st
    yf = _YF("yfinance")
    sys.modules["yfinance"] = yf
    plt = types.ModuleType("plotly.graph_objects")

    class _Fig:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, n):
            return lambda *a, **k: None
    for nm in ["Figure", "Candlestick", "Scatter", "Bar", "Heatmap", "Table",
               "Indicator", "Pie", "Histogram", "Box"]:
        setattr(plt, nm, _Fig)
    sys.modules["plotly"] = types.ModuleType("plotly")
    sys.modules["plotly.graph_objects"] = plt
    sub = types.ModuleType("plotly.subplots")
    sub.make_subplots = lambda *a, **k: _Fig()
    sys.modules["plotly.subplots"] = sub
    px = types.ModuleType("plotly.express")
    sys.modules["plotly.express"] = px
    fp = types.ModuleType("feedparser")
    fp.parse = lambda *a, **k: types.SimpleNamespace(entries=[])
    sys.modules["feedparser"] = fp
    return st


if __name__ == "__main__":
    st = install_stubs()
    import importlib
    app = importlib.import_module("app")
    print("✓ import app สำเร็จ (โหลดข้อมูล + สร้าง stamp ผ่าน)")
    print(f"  DS_ITEMS ที่ลงทะเบียนตอนโหลด = "
          f"{[i['ชุดข้อมูล'] for i in app.DS_ITEMS]}")
    ok, bad = [], []
    for name, fn in app.ROUTES.items():
        app.DS_ITEMS[:] = app.DS_ITEMS[:2]
        try:
            fn()
            ok.append(name)
        except Exception as e:
            bad.append((name, f"{type(e).__name__}: {e}"))
    print(f"\nหน้าที่เดินผ่าน {len(ok)}/{len(app.ROUTES)}")
    for n, e in bad:
        print(f"  ✗ {n}: {e}")
    if CALLS["exception"]:
        print("st.exception ถูกเรียก:", CALLS["exception"][:3])
    sys.exit(1 if bad else 0)
