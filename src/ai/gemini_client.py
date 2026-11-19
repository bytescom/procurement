"""Lazy, bounded Google Gen AI SDK client for Gemini 3.8 Flash."""

from __future__ import annotations

import os
import random
import time

from src.ai.prompts import RESPONSE_SCHEMA


MODEL_ID = "gemini-3.8-flash"


class GeminiUnavailable(RuntimeError):
    pass


class GeminiClient:
    def __init__(self, *, api_key: str | None = None, timeout_seconds: float = 30,
                 max_retries: int = 2, client=None):
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(0, int(max_retries))
        self._client = client

    def _get_client(self):
        if not self.api_key:
            raise GeminiUnavailable("GEMINI_API_KEY is not configured.")
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise GeminiUnavailable("Install google-genai to enable Gemini fallback.") from exc
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @staticmethod
    def _retryable(error: Exception) -> bool:
        status = getattr(error, "code", None) or getattr(error, "status_code", None)
        if status is None:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)
        if status is not None:
            try:
                return int(status) in {408, 429, 500, 502, 503, 504}
            except (TypeError, ValueError):
                pass
        return isinstance(error, (TimeoutError, ConnectionError)) or "timeout" in type(error).__name__.casefold()

    def classify(self, prompt: str) -> tuple[dict, dict]:
        client = self._get_client()
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = client.interactions.create(
                    model=MODEL_ID,
                    input=prompt,
                    response_format={"type": "text", "mime_type": "application/json", "schema": RESPONSE_SCHEMA},
                    generation_config={"thinking_level": "low"},
                    timeout=self.timeout_seconds,
                    store=False,
                )
                text = getattr(result, "output_text", None)
                if text is None and isinstance(result, dict):
                    text = result.get("output_text")
                if not text:
                    raise ValueError("Gemini interaction contained no output_text.")
                usage_obj = getattr(result, "usage", None)
                if usage_obj is None and isinstance(result, dict):
                    usage_obj = result.get("usage", {})
                def val(name):
                    if isinstance(usage_obj, dict):
                        return usage_obj.get(name, 0) or 0
                    return getattr(usage_obj, name, 0) or 0
                usage = {"input_tokens": int(val("total_input_tokens")),
                         "output_tokens": int(val("total_output_tokens")),
                         "total_tokens": int(val("total_tokens"))}
                return text, usage
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries or not self._retryable(exc):
                    break
                time.sleep(min(8.0, (0.5 * (2 ** attempt)) + random.uniform(0, 0.25)))
        raise GeminiUnavailable(f"Gemini request failed ({type(last_error).__name__}).") from last_error
