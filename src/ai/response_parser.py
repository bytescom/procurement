"""Strict parsing of structured Gemini classification responses."""

import json


def parse_response(raw) -> dict:
    if isinstance(raw, dict):
        data = raw
    else:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Gemini returned an empty response.")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Gemini response was not valid JSON.") from exc
    if not isinstance(data, dict):
        raise ValueError("Gemini response must be a JSON object.")
    required = {"status", "transaction_type", "l1", "l2", "l3", "l4", "confidence", "reason"}
    if not required.issubset(data):
        raise ValueError(f"Gemini response is missing required keys: {sorted(required - set(data))}")
    if data["status"] not in {"classified", "uncertain"}:
        raise ValueError("Gemini response status must be classified or uncertain.")
    if data["transaction_type"] not in {"good", "service", "unclear"}:
        raise ValueError("Gemini transaction_type must be good, service, or unclear.")
    try:
        confidence = float(data["confidence"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Gemini confidence must be numeric.") from exc
    if not 0 <= confidence <= 1:
        raise ValueError("Gemini confidence must be between 0 and 1.")
    if not isinstance(data["reason"], str):
        raise ValueError("Gemini reason must be text.")
    for key in ("l1", "l2", "l3", "l4"):
        value = data[key]
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Gemini {key} must be text or null.")
    if data["status"] == "uncertain" and any(data[key] not in (None, "") for key in ("l1", "l2", "l3", "l4")):
        raise ValueError("Uncertain response must not provide category levels.")
    return {**data, "confidence": confidence}
