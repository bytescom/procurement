"""Small structured decision audit helper."""


def decision_log(record_id: str, outcome: dict, *, local_confidence: float,
                 gemini_attempted: bool, gemini_cache_hit: bool = False) -> dict:
    return {"record_id": str(record_id), "decision_outcome": outcome["decision_outcome"],
            "decision_reason": outcome["decision_reason"], "local_confidence": float(local_confidence),
            "gemini_attempted": bool(gemini_attempted), "gemini_cache_hit": bool(gemini_cache_hit),
            "fallback_used": bool(outcome["fallback_used"])}
