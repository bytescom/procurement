"""Constrained prompts for taxonomy-only Gemini classification."""

import json


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["classified", "uncertain"]},
        "transaction_type": {"type": "string", "enum": ["good", "service", "unclear"]},
        "l1": {"type": "string"},
        "l2": {"type": "string"},
        "l3": {"type": "string"},
        "l4": {"type": "string"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["status", "transaction_type", "l1", "l2", "l3", "l4", "confidence", "reason"],
}


def make_prompt(record: dict, candidate_paths: list[tuple[str, ...]], level_names: list[str]) -> str:
    taxonomy = [" > ".join(path) for path in candidate_paths]
    return (
        "Classify this procurement transaction using only the supplied category paths. "
        "First decide whether it is a good, service, or unclear. Select an exact path from the candidates; "
        "respect parent-child order and never invent category names. If evidence is insufficient, return "
        "status=uncertain and empty strings for all category levels. Give a short reason and an estimated confidence from 0 to 1. "
        "The confidence is a model estimate, not a calibrated probability.\n\n"
        f"Taxonomy level names: {json.dumps(level_names)}\n"
        f"Candidate taxonomy paths: {json.dumps(taxonomy, ensure_ascii=False)}\n"
        f"Purchase fields: {json.dumps(record, ensure_ascii=False, default=str)}"
    )
