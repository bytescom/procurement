"""Shared normalization and evidence-scoring helpers for matching."""

import re
from difflib import SequenceMatcher
from typing import Any

import pandas as pd


def normalize_reference(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    # Numeric-looking reference values are often read as floats by CSV parsers.
    text = re.sub(r"\.0+$", "", text)
    normalized = re.sub(r"[^A-Z0-9]", "", text)
    return normalized or None


def normalize_po_reference(value: Any) -> str | None:
    """Normalize common PO prefixes and PO/item suffixes for matching only."""
    return po_reference_parts(value)[0]


def po_reference_parts(value: Any) -> tuple[str | None, str | None]:
    """Return normalized PO and optional item suffix without losing item evidence."""
    if pd.isna(value):
        return None, None
    text = str(value).strip().upper()
    text = re.sub(r"^PO\s*[#:\-]?\s*", "", text)
    text = re.sub(r"\.0+$", "", text)
    match = re.fullmatch(r"(\d+)/(\d+)", text)
    if match:
        return normalize_reference(match.group(1)), normalize_reference(match.group(2))
    return normalize_reference(text), None


def normalize_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = " ".join(str(value).casefold().split())
    return text or None


def amount_cents(value: Any) -> int | None:
    try:
        number = float(value)
        if pd.isna(number):
            return None
        return int(round(number * 100))
    except (TypeError, ValueError, OverflowError):
        return None


def similarity(left: Any, right: Any) -> float:
    a, b = normalize_text(left), normalize_text(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def get_column(config: dict, dataset: str, semantic: str, fallback: str) -> str:
    return config.get("datasets", {}).get(dataset, {}).get("columns", {}).get(semantic, fallback)
