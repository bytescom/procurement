"""Reusable router for local, semantic, and Gemini category candidates."""

from src.decision_engine.decision_validator import validate_taxonomy_response


def route_decision(*, local_path: tuple[str, ...], local_confidence: float,
                   taxonomy_paths: list[tuple[str, ...]], level_names: list[str],
                   threshold: float, gemini_threshold: float | None = None, semantic: dict | None = None,
                   gemini: dict | None = None, gemini_error: str | None = None) -> dict:
    local_valid = tuple(local_path) in taxonomy_paths
    semantic_path = tuple((semantic or {}).get("path", ()))
    semantic_valid = semantic is not None and semantic_path in taxonomy_paths
    if semantic_valid and local_valid and semantic_path == tuple(local_path) and local_confidence >= threshold:
        return _result(tuple(local_path), local_confidence, "ACCEPT_LOCAL", "local_semantic_agreement")
    if gemini is not None:
        validation = validate_taxonomy_response(gemini, taxonomy_paths, level_names)
        if not validation["valid"]:
            if local_valid:
                return _result(tuple(local_path), local_confidence, "REJECT_INVALID", validation["reason"], fallback=True)
            return _result((), 0.0, "REJECT_INVALID", validation["reason"])
        if validation["status"] == "classified" and float(gemini["confidence"]) >= (gemini_threshold if gemini_threshold is not None else threshold):
            return _result(validation["path"], float(gemini["confidence"]), "ACCEPT_GEMINI", gemini.get("reason", ""))
        if local_valid:
            return _result(tuple(local_path), local_confidence, "FALLBACK_LOCAL", gemini.get("reason", "Gemini uncertain or below threshold."))
        return _result((), 0.0, "NEEDS_REVIEW", gemini.get("reason", "Gemini uncertain."))
    if local_valid:
        status = "ACCEPT_LOCAL" if local_confidence >= threshold else "FALLBACK_LOCAL"
        return _result(tuple(local_path), local_confidence, status, gemini_error or "Gemini fallback not used.")
    return _result((), 0.0, "NEEDS_REVIEW", gemini_error or "No valid local prediction.")


def _result(path, confidence, outcome, reason, fallback=False):
    levels = {f"l{i+1}": path[i] if i < len(path) else "" for i in range(4)}
    return {**levels, "confidence": float(confidence), "decision_outcome": outcome,
            "decision_reason": str(reason)[:500], "fallback_used": bool(fallback or outcome == "FALLBACK_LOCAL")}
