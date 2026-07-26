# -*- coding: utf-8 -*-
"""
llm_providers.py — ชั้นเรียก LLM หลายเจ้าแบบอินเทอร์เฟซเดียว (Gemini / Groq / OpenRouter)

ทำไมต้องมีไฟล์นี้:
ห้องประชุม AI เดิมใช้ "โมเดลเดียวเล่นหลายบท" — ความเห็นไม่อิสระเลย
ไฟล์นี้เปิดให้ใช้โมเดลจาก 3 ค่ายจริง ซึ่ง "อิสระกว่า" แต่ **ยังไม่อิสระจริง**
(ดู HONESTY ด้านล่าง — ข้อความนี้ถูกแสดงบน UI ด้วย ห้ามตัดออก)

ข้อเท็จจริงที่ต้องบอกผู้ใช้ (ตรวจสอบ ก.ค. 2026):
- โควตา free tier ของทุกเจ้า "เปลี่ยนได้โดยไม่แจ้งล่วงหน้า" — Google เคยตัดโควตาฟรี
  ลงมากช่วงธ.ค. 2025 ตัวเลขที่บล็อกต่าง ๆ อ้างขัดกันเอง โมดูลนี้จึง **ไม่ฮาร์ดโค้ด
  ตัวเลขโควตา** เป็นความจริง แต่ลิงก์ไปหน้า docs ทางการให้ผู้ใช้เช็คเอง
- รายชื่อโมเดลฟรีหมุนเวียน ถูกถอดได้ตลอด → model id เป็นช่องกรอกเอง ไม่ล็อกตาย
  ถ้า id ตาย เราโชว์ error ดิบจากเซิร์ฟเวอร์ ไม่กลบ
- free tier ของ Google ระบุว่าอาจนำ prompt ไปปรับปรุงโมเดล → ห้ามส่งข้อมูลลับ
  (เคสนี้ส่งแต่ราคาหุ้น/ตัวเลข engine ซึ่งเป็นข้อมูลสาธารณะ)

ดีไซน์: pure + inject ได้ (transport=) → ทดสอบออฟไลน์ได้โดยไม่ต่อเน็ต
ทุกฟังก์ชัน **ไม่โยน exception** — คืน dict ที่มี ok/error เสมอ
"""
from __future__ import annotations

import json
import time

# ---------------------------------------------------------------------------
# ทะเบียนผู้ให้บริการ
# ---------------------------------------------------------------------------

VERSION = "1.0"

PROVIDERS: dict[str, dict] = {
    "gemini": {
        "th": "Google Gemini (AI Studio)",
        "default_model": "gemini-2.5-flash",
        "keys_url": "https://aistudio.google.com/apikey",
        "limits_url": "https://ai.google.dev/gemini-api/docs/rate-limits",
        "models_url": "https://ai.google.dev/gemini-api/docs/models",
        "note": "ไม่ต้องใช้บัตร · free tier อาจนำ prompt ไปเทรนโมเดล · "
                "เปิด billing เมื่อไหร่ free tier ของโปรเจกต์นั้นหายทันที",
        "style": "gemini",
    },
    "groq": {
        "th": "Groq (LPU)",
        "default_model": "llama-3.3-70b-versatile",
        "keys_url": "https://console.groq.com/keys",
        "limits_url": "https://console.groq.com/docs/rate-limits",
        "models_url": "https://console.groq.com/docs/models",
        "note": "ไม่ต้องใช้บัตร · เร็วที่สุดในสามเจ้า · "
                "โควตานับรวมทั้ง organization (เพิ่ม key ไม่ช่วย)",
        "style": "openai",
        "url": "https://api.groq.com/openai/v1/chat/completions",
    },
    "openrouter": {
        "th": "OpenRouter (โมเดล :free)",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
        "keys_url": "https://openrouter.ai/keys",
        "limits_url": "https://openrouter.ai/docs/api-reference/limits",
        "models_url": "https://openrouter.ai/models?q=free",
        "note": "key เดียวได้หลายค่าย · รายชื่อโมเดล :free หมุนเวียนบ่อย "
                "ถ้าเจอ 404 ให้ไปคัด id ใหม่จากหน้า models",
        "style": "openai",
        "url": "https://openrouter.ai/api/v1/chat/completions",
    },
}

ORDER = ["gemini", "groq", "openrouter"]

HONESTY = (
    "**โมเดลต่างค่ายกัน = อิสระกว่าโมเดลเดียวเล่นหลายบท แต่ยังไม่อิสระจริง** — "
    "ทุกตัวเทรนบนคลังข้อความอินเทอร์เน็ตที่ทับซ้อนกัน อ่าน context ก้อนเดียวกัน "
    "และรับ prompt เดียวกัน → error สหสัมพันธ์กันสูง "
    "ดังนั้น 'เห็นตรงกัน 3/3' **ไม่เท่ากับ** หลักฐานอิสระ 3 ชิ้น "
    "และไม่เพิ่มความน่าจะเป็นที่สัญญาณจะถูก · "
    "จุดที่ควรอ่านคือ **ตรงที่เห็นต่าง** ไม่ใช่คะแนนเสียงข้างมาก"
)

TIMEOUT_S = 90


# ---------------------------------------------------------------------------
# transport เริ่มต้น (requests) — แยกออกมาเพื่อ inject ตอนเทสต์
# ---------------------------------------------------------------------------

def _default_transport(url: str, headers: dict, payload: dict,
                       timeout: float) -> tuple[int, str]:
    """คืน (status_code, body_text) — ไม่ raise"""
    try:
        import requests
    except ImportError:
        return 0, "ไม่มีไลบรารี requests (ใส่ requests ใน requirements.txt)"
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        return r.status_code, r.text
    except Exception as e:  # network / DNS / timeout
        return 0, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# แปลง messages กลาง -> รูปแบบของแต่ละเจ้า
# ---------------------------------------------------------------------------

def _to_gemini(messages: list[dict]) -> list[dict]:
    """Gemini ใช้ role 'user'/'model' และ parts[] (ไม่มี 'assistant')"""
    out = []
    for m in messages:
        role = "model" if m.get("role") == "assistant" else "user"
        out.append({"role": role, "parts": [{"text": str(m.get("content", ""))}]})
    return out


def _extract_gemini(body: dict) -> str:
    cands = body.get("candidates") or []
    if not cands:
        return ""
    parts = ((cands[0].get("content") or {}).get("parts") or [])
    return "".join(str(p.get("text", "")) for p in parts)


def _extract_openai(body: dict) -> str:
    ch = body.get("choices") or []
    if not ch:
        return ""
    msg = ch[0].get("message") or {}
    c = msg.get("content")
    if isinstance(c, list):  # บางเจ้าคืน content เป็น list ของ block
        return "".join(str(b.get("text", "")) for b in c if isinstance(b, dict))
    return str(c or "")


def _server_error(body_text: str, status: int) -> str:
    """ดึงข้อความ error จริงจากเซิร์ฟเวอร์ออกมาโชว์ ไม่กลบด้วยข้อความสวย ๆ"""
    try:
        b = json.loads(body_text)
        e = b.get("error")
        if isinstance(e, dict):
            return f"HTTP {status}: {e.get('message') or e}"
        if e:
            return f"HTTP {status}: {e}"
    except Exception:
        pass
    return f"HTTP {status}: {str(body_text)[:400]}"


# ---------------------------------------------------------------------------
# API หลัก
# ---------------------------------------------------------------------------

def chat(provider: str, api_key: str, messages: list[dict],
         model: str | None = None, max_tokens: int = 2000,
         temperature: float = 0.3, timeout: float = TIMEOUT_S,
         transport=None) -> dict:
    """เรียกโมเดลหนึ่งครั้ง — คืน dict เสมอ ไม่ raise

    คืน: {ok, text, error, provider, model, latency_s, http}
    """
    t0 = time.time()
    spec = PROVIDERS.get(provider)
    base = {"ok": False, "text": "", "error": "", "provider": provider,
            "model": model or (spec or {}).get("default_model", ""),
            "latency_s": 0.0, "http": 0}
    if spec is None:
        base["error"] = f"ไม่รู้จักผู้ให้บริการ '{provider}'"
        return base
    if not str(api_key or "").strip():
        base["error"] = "ยังไม่ได้ใส่ API key"
        return base
    if not messages:
        base["error"] = "ไม่มีข้อความส่ง"
        return base

    mdl = model or spec["default_model"]
    base["model"] = mdl
    tr = transport or _default_transport

    if spec["style"] == "gemini":
        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               f"{mdl}:generateContent")
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        payload = {"contents": _to_gemini(messages),
                   "generationConfig": {"maxOutputTokens": int(max_tokens),
                                        "temperature": float(temperature)}}
    else:
        url = spec["url"]
        headers = {"Authorization": f"Bearer {api_key}",
                   "Content-Type": "application/json"}
        if provider == "openrouter":
            headers["X-Title"] = "SET-Bond Dashboard"
        payload = {"model": mdl,
                   "messages": [{"role": ("assistant"
                                          if m.get("role") == "assistant"
                                          else "user"),
                                 "content": str(m.get("content", ""))}
                                for m in messages],
                   "max_tokens": int(max_tokens),
                   "temperature": float(temperature)}

    status, body_text = tr(url, headers, payload, timeout)
    base["http"] = int(status)
    base["latency_s"] = round(time.time() - t0, 2)

    if status == 0:
        base["error"] = f"ต่อเซิร์ฟเวอร์ไม่ได้ — {body_text}"
        return base
    if status == 429:
        base["error"] = (f"HTTP 429 โควตาหมด/ยิงถี่เกิน ({spec['th']}) — "
                         f"รอแล้วลองใหม่ หรือดูโควตาจริงที่ {spec['limits_url']}")
        return base
    if status == 404:
        base["error"] = (f"HTTP 404 ไม่พบโมเดล '{mdl}' — รายชื่อโมเดลฟรี"
                         f"เปลี่ยนบ่อย เช็ค id ใหม่ที่ {spec['models_url']}")
        return base
    if status >= 400:
        base["error"] = _server_error(body_text, status)
        return base

    try:
        body = json.loads(body_text)
    except Exception as e:
        base["error"] = f"อ่าน JSON ตอบกลับไม่ได้: {e}"
        return base

    text = (_extract_gemini(body) if spec["style"] == "gemini"
            else _extract_openai(body))
    if not str(text).strip():
        base["error"] = ("เซิร์ฟเวอร์ตอบ 200 แต่ไม่มีข้อความ "
                         "(อาจโดน safety filter หรือ max_tokens ต่ำไป)")
        return base
    base["ok"] = True
    base["text"] = text
    return base


def chat_many(selections: list[dict], messages: list[dict],
              max_tokens: int = 2000, temperature: float = 0.3,
              timeout: float = TIMEOUT_S, transport=None,
              on_progress=None) -> list[dict]:
    """ยิง prompt เดียวกันไปหลายเจ้า **ตามลำดับ** (ไม่ขนาน — free tier ชอบ 429)

    selections: [{"provider": "gemini", "api_key": "...", "model": "..."}, ...]
    ตัวที่พังจะคืน ok=False พร้อม error — ไม่ทำให้ทั้งชุดล้ม
    """
    out = []
    for s in selections:
        pv = s.get("provider", "")
        if on_progress:
            try:
                on_progress(pv)
            except Exception:
                pass
        out.append(chat(pv, s.get("api_key", ""), messages,
                        model=s.get("model") or None, max_tokens=max_tokens,
                        temperature=temperature, timeout=timeout,
                        transport=transport))
    return out
