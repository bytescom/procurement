"""Taxonomy-aware Gemini fallback with SQLite cache and bounded request budget."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from contextlib import closing

from src.ai.gemini_client import GeminiClient, GeminiUnavailable, MODEL_ID
from src.ai.prompts import make_prompt
from src.ai.response_parser import parse_response


class GeminiClassifier:
    def __init__(self, *, cache_path="output/cache/gemini.sqlite3", client=None,
                 timeout_seconds=30, max_retries=2, price_input_per_million=0.75,
                 price_output_per_million=3.75):
        self.cache_path = Path(cache_path)
        self.client = client or GeminiClient(timeout_seconds=timeout_seconds, max_retries=max_retries)
        self.price_input = float(price_input_per_million)
        self.price_output = float(price_output_per_million)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.cache_path)) as db:
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute("CREATE TABLE IF NOT EXISTS classifications (cache_key TEXT PRIMARY KEY, response TEXT NOT NULL)")
            self._memory_cache = {key: value for key, value in
                                  db.execute("SELECT cache_key,response FROM classifications").fetchall()}
            db.commit()

    @staticmethod
    def cache_key(record: dict, candidate_paths: list[tuple[str, ...]], level_names: list[str]) -> str:
        context = {"record": record, "taxonomy": candidate_paths, "levels": level_names, "model": MODEL_ID}
        encoded = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def get_cached(self, key: str):
        value = self._memory_cache.get(key)
        return parse_response(value) if value else None

    def _save(self, key: str, response: dict):
        value = json.dumps(response, ensure_ascii=False)
        with closing(sqlite3.connect(self.cache_path)) as db:
            db.execute("INSERT OR REPLACE INTO classifications(cache_key,response) VALUES(?,?)", (key, value))
            db.commit()
        self._memory_cache[key] = value

    def classify(self, record: dict, candidate_paths: list[tuple[str, ...]], level_names: list[str]):
        key = self.cache_key(record, candidate_paths, level_names)
        cached = self.get_cached(key)
        if cached is not None:
            return cached, {"cache_hit": True, "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}
        raw, usage = self.client.classify(make_prompt(record, candidate_paths, level_names))
        parsed = parse_response(raw)
        self._save(key, parsed)
        return parsed, {"cache_hit": False, "usage": usage}

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.price_input + output_tokens * self.price_output) / 1_000_000


__all__ = ["GeminiClassifier", "GeminiUnavailable"]
