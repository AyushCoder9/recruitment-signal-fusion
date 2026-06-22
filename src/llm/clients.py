"""Provider-agnostic LLM judge clients. Labeling is OFFLINE precompute, so hosted APIs are
fine here (never imported by rank.py). Free stack: Groq (gpt-oss-120b / llama-3.3-70b) +
Gemini Flash. Each client returns a parsed {tier, reasoning, key_factors} dict.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)


class RateLimiter:
    """Thread-safe token bucket. Throttles to ~tpm tokens/min so we never trip Groq's
    per-minute TPM ceiling (the binding free-tier constraint)."""

    def __init__(self, tpm: int, headroom: float = 0.7):
        self.capacity = tpm * headroom
        self.tokens = self.capacity
        self.rate = self.capacity / 60.0   # tokens per second
        self.t = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self, n: float) -> None:
        n = min(n, self.capacity)
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
                self.t = now
                if self.tokens >= n:
                    self.tokens -= n
                    return
                wait = (n - self.tokens) / self.rate
            time.sleep(min(wait, 5.0))


def parse_judgment(text: str) -> dict | None:
    if not text:
        return None
    t = _FENCE.sub("", text.strip())
    try:
        obj = json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None
    try:
        tier = int(obj.get("tier"))
    except Exception:
        return None
    return {
        "tier": max(0, min(4, tier)),
        "reasoning": str(obj.get("reasoning", ""))[:400],
        "key_factors": [str(x)[:60] for x in (obj.get("key_factors") or [])][:6],
    }


_RETRY_AFTER = re.compile(r"try again in ([\d.]+)s", re.I)


class TPDExhausted(Exception):
    """Model's daily token budget (TPD) is spent — stop using it for today."""


class _Retry:
    def __init__(self, max_retries=6, base=2.0):
        self.max_retries, self.base = max_retries, base

    def run(self, fn):
        last = None
        for i in range(self.max_retries):
            try:
                return fn()
            except Exception as e:
                last = e
                msg = str(e)
                # Daily cap exhausted (Groq TPD or Gemini RequestsPerDay) — don't waste
                # retries; bow this model out. Per-MINUTE 429s fall through to backoff.
                daily = ("per day" in msg or "TPD" in msg or "PerDay" in msg
                         or "RequestsPerDay" in msg or "GenerateRequestsPerDayPerProjectPerModel" in msg)
                if daily and "PerMinute" not in msg:
                    raise TPDExhausted(msg[:120]) from e
                m = _RETRY_AFTER.search(msg)
                if m:                                   # honor Groq's exact retry-after
                    wait = float(m.group(1)) + 0.5
                elif any(k in msg for k in ("429", "rate", "quota", "503", "502")):
                    wait = min(30, self.base * (2 ** i) + 1)
                else:
                    wait = self.base * (1.5 ** i)
                time.sleep(wait)
        raise last


class OpenAICompatClient:
    """Any OpenAI-compatible chat endpoint (Cerebras, OpenRouter, ...). Some hosted reasoning
    models (gpt-oss on Cerebras) reject response_format=json_object but emit clean JSON when
    simply asked to; set use_json=False there and lean on parse_judgment's brace-extraction."""

    def __init__(self, base_url: str, api_key_env: str, model: str,
                 temperature: float = 0.0, tpm: int | None = None, use_json: bool = False):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ[api_key_env], base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.retry = _Retry()
        self.limiter = RateLimiter(tpm) if tpm else None
        self.use_json = use_json
        self.max_tokens = 400

    def judge(self, messages: list[dict]) -> dict | None:
        if self.limiter:
            est = sum(len(m["content"]) for m in messages) / 4.0 + self.max_tokens
            self.limiter.acquire(est)

        def _call():
            kw = dict(model=self.model, messages=messages, temperature=self.temperature,
                      max_tokens=self.max_tokens)
            if self.use_json:
                kw["response_format"] = {"type": "json_object"}
            r = self.client.chat.completions.create(**kw)
            content = r.choices[0].message.content
            if not (content or "").strip():     # throttle -> empty body; retry, don't drop
                raise RuntimeError("empty completion (transient throttle)")
            return content
        return parse_judgment(self.retry.run(_call))


class GroqClient:
    def __init__(self, model: str = "openai/gpt-oss-120b", temperature: float = 0.0,
                 tpm: int | None = None):
        from groq import Groq
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model
        self.temperature = temperature
        self.retry = _Retry()
        self.limiter = RateLimiter(tpm) if tpm else None
        self.max_tokens = 320

    def judge(self, messages: list[dict]) -> dict | None:
        if self.limiter:
            est = sum(len(m["content"]) for m in messages) / 4.0 + self.max_tokens
            self.limiter.acquire(est)

        def _call():
            r = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature,
                response_format={"type": "json_object"}, max_tokens=self.max_tokens)
            return r.choices[0].message.content
        return parse_judgment(self.retry.run(_call))


class RequestLimiter:
    """Thread-safe ~rpm requests/min throttle. Gemini free tier caps REQUESTS-per-minute
    (not tokens), so we pace by request count to avoid per-minute 429 storms."""

    def __init__(self, rpm: int, headroom: float = 0.8):
        self.capacity = max(1.0, rpm * headroom)
        self.tokens = self.capacity
        self.rate = self.capacity / 60.0
        self.t = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
                self.t = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rate
            time.sleep(min(wait, 5.0))


class GeminiClient:
    def __init__(self, model: str = "gemini-2.5-flash-lite", temperature: float = 0.0,
                 rpm: int | None = None):
        from google import genai
        self.genai = genai
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = model
        self.temperature = temperature
        self.retry = _Retry()
        self.limiter = RequestLimiter(rpm) if rpm else None

    def judge(self, messages: list[dict]) -> dict | None:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        from google.genai import types
        if self.limiter:
            self.limiter.acquire()

        def _call():
            r = self.client.models.generate_content(
                model=self.model, contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system, temperature=self.temperature,
                    response_mime_type="application/json", max_output_tokens=400,
                    # 2.5 are reasoning models; disable thinking so JSON is clean + not truncated.
                    thinking_config=types.ThinkingConfig(thinking_budget=0)))
            return r.text
        return parse_judgment(self.retry.run(_call))


# Free-tier TPM per Groq model (the binding constraint). Throttle just under these.
GROQ_TPM = {
    "llama-3.3-70b-versatile": 12000,
    "openai/gpt-oss-120b": 8000,
    "openai/gpt-oss-20b": 8000,
    "meta-llama/llama-4-scout-17b-16e-instruct": 30000,
    "llama-3.1-8b-instant": 6000,
}

# Free-tier requests-per-minute per Gemini model (the binding constraint there).
GEMINI_RPM = {
    "gemini-2.5-flash-lite": 15,
    "gemini-2.5-flash": 10,
}


def make_client(spec: str):
    """spec like 'groq:llama-3.3-70b-versatile', 'gemini:gemini-2.5-flash-lite',
    'cerebras:gpt-oss-120b', 'openrouter:meta-llama/llama-3.3-70b-instruct:free'."""
    provider, _, model = spec.partition(":")
    if provider == "groq":
        model = model or "llama-3.3-70b-versatile"
        return GroqClient(model=model, tpm=GROQ_TPM.get(model, 6000))
    if provider == "gemini":
        model = model or "gemini-2.5-flash-lite"
        return GeminiClient(model=model, rpm=GEMINI_RPM.get(model, 10))
    if provider == "cerebras":
        # gpt-oss-120b rejects json_object -> use_json=False (prompt forces JSON).
        return OpenAICompatClient("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY",
                                  model or "gpt-oss-120b", tpm=55000, use_json=False)
    if provider == "openrouter":
        return OpenAICompatClient("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                                  model, tpm=8000, use_json=True)
    raise ValueError(f"unknown provider in spec: {spec}")
