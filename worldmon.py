# -*- coding: utf-8 -*-
"""
worldmon.py — ชั้นตรรกะสำหรับหน้า "World Monitor" (แนวทาง A+C: iframe + ปุ่มลิงก์สำรองเสมอ)

ข้อเท็จจริงที่ตรวจสอบแล้ว (22 ก.ค. 2026):
- World Monitor เป็นแอป TypeScript/JS (AGPL-3.0) — รวมโค้ดเข้าแอป Python/Streamlit ตรง ๆ ไม่ได้
- เวอร์ชันเว็บฟรีใช้ได้โดยไม่ต้องมี key (รีเฟรชข้อมูล ~5-15 นาทีบนแผนฟรี)
- การดึงข้อมูลผ่าน API/SDK ต้องมี key จากแผน Pro ($39.99/เดือน) — ผู้ใช้เลือกไม่ใช้
- การฝัง iframe อาจถูกเบราว์เซอร์/เซิร์ฟเวอร์ปลายทางปฏิเสธ (พิสูจน์ได้ตอน deploy เท่านั้น)
  → หน้าเพจนี้จึงแสดงปุ่มลิงก์เปิดแท็บใหม่ "เสมอ" ไม่ว่ากรณีใด

ขอบเขตที่ตั้งใจ: เป็นหน้าต่างดูข้อมูลภายนอกเท่านั้น — ไม่ป้อนตัวเลขจาก World Monitor
เข้า engine/คะแนนของระบบเรา เพราะเป็นข้อมูล heuristic ของบุคคลที่สามที่เรา validate ไม่ได้
"""
from __future__ import annotations

# (label ไทย, URL) — เรียงตามความเกี่ยวข้องกับงานของระบบนี้
VARIANTS: dict[str, tuple[str, str]] = {
    "finance":   ("การเงิน/ตลาด (แนะนำ)", "https://finance.worldmonitor.app"),
    "world":     ("ภูมิรัฐศาสตร์/ภาพรวมโลก", "https://www.worldmonitor.app"),
    "commodity": ("สินค้าโภคภัณฑ์/พลังงาน", "https://commodity.worldmonitor.app"),
    "energy":    ("พลังงาน", "https://energy.worldmonitor.app"),
    "tech":      ("เทคโนโลยี", "https://tech.worldmonitor.app"),
}
DEFAULT_VARIANT = "finance"
IFRAME_HEIGHT = 820  # px — ปรับได้ใน UI

GITHUB_URL = "https://github.com/koala73/worldmonitor"

CAVEATS: list[str] = [
    "บริการภายนอก (AGPL-3.0, koala73/worldmonitor) — ไม่ใช่ข้อมูลของระบบเรา "
    "และตัวเลข/คะแนนของเขา (เช่น CII, AI brief) เรา validate ไม่ได้",
    "ถ้ากรอบด้านบนว่างเปล่า = ปลายทางปฏิเสธการฝัง iframe — ใช้ปุ่มเปิดแท็บใหม่แทน "
    "(ทำงานได้เสมอ)",
    "แผนที่ 3D/WebGL กินทรัพยากรมาก อาจช้าบน iPad — ถ้าหน่วง ให้เปิดแท็บใหม่",
    "แผนฟรีรีเฟรชข้อมูล ~5-15 นาที ไม่ใช่เรียลไทม์ระดับวินาที",
    "ระบบนี้ไม่นำตัวเลขจาก World Monitor เข้าคะแนน/โมเดลใด ๆ — เป็นหน้าต่างดูบริบทเท่านั้น",
]


def variant_url(key: str) -> str:
    """คืน URL ของ variant — ถ้า key ไม่รู้จัก ใช้ค่าเริ่มต้น (ไม่โยน error ใส่ UI)"""
    return VARIANTS.get(key, VARIANTS[DEFAULT_VARIANT])[1]


def variant_label(key: str) -> str:
    return VARIANTS.get(key, VARIANTS[DEFAULT_VARIANT])[0]


# ---------------------------------------------------------------------------
# โค้ดพร้อมวางใน app.py (หลังได้ไฟล์จริงจะรวมให้เอง — เก็บไว้ที่นี่เพื่อความโปร่งใส)
# ---------------------------------------------------------------------------
APP_PY_SNIPPET = '''
# --- หน้า: World Monitor (iframe + ลิงก์สำรอง) ---
import streamlit.components.v1 as components
import worldmon as WM

def page_worldmonitor():
    st.header("🌐 World Monitor (บริการภายนอก)")
    key = st.selectbox("เลือกมุมมอง", list(WM.VARIANTS.keys()),
                       index=list(WM.VARIANTS.keys()).index(WM.DEFAULT_VARIANT),
                       format_func=WM.variant_label)
    url = WM.variant_url(key)

    c1, c2 = st.columns(2)
    c1.link_button("↗ เปิดแท็บใหม่ (ชัวร์สุด)", url, use_container_width=True)
    c2.link_button("ซอร์สโค้ด (GitHub, AGPL-3.0)", WM.GITHUB_URL,
                   use_container_width=True)

    h = st.slider("ความสูงกรอบ (px)", 400, 1400, WM.IFRAME_HEIGHT, 20)
    components.iframe(url, height=h, scrolling=True)

    with st.expander("ข้อจำกัดที่ต้องรู้ (อ่านก่อนใช้)", expanded=True):
        for c in WM.CAVEATS:
            st.markdown(f"- {c}")
'''
