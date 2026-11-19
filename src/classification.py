"""Runtime-taxonomy classification using local lexical evidence only."""

import re

import pandas as pd


def _tokens(value) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {token for token in re.findall(r"[a-z0-9]+", str(value).casefold()) if len(token) > 1}


def load_taxonomy(preprocessed: dict) -> tuple[list[str], list[tuple[str, ...]]]:
    cfg = preprocessed["config"]
    key = next((name for name in cfg["datasets"] if "taxonomy" in name.casefold()), None)
    if key is None:
        raise ValueError("Configuration must include a taxonomy dataset.")
    spec = cfg["datasets"][key]
    df = preprocessed["datasets"][key]
    mappings = spec.get("columns", {})
    level_names = list(cfg.get("taxonomy", {}).get("levels", []))
    columns = [mappings.get(f"l{i+1}") or mappings.get(level_names[i]) for i in range(len(level_names))]
    if not columns or any(c not in df for c in columns):
        raise ValueError(f"Taxonomy level mappings are missing or invalid: {columns}")
    paths = []
    for values in df[columns].itertuples(index=False, name=None):
        path = tuple(str(v).strip() for v in values if not pd.isna(v) and str(v).strip())
        if path and path not in paths:
            paths.append(path)
    if not paths:
        raise ValueError("Taxonomy contains no usable category paths.")
    return level_names, paths


def classify_records(frame: pd.DataFrame, *, text_columns: list[str], taxonomy_paths: list[tuple[str, ...]],
                     level_names: list[str]) -> pd.DataFrame:
    """Return best lexical taxonomy path; unsupported/tied lower levels stay blank."""
    rows = []
    path_tokens = [set().union(*(_tokens(part) for part in path)) for path in taxonomy_paths]
    for _, row in frame.iterrows():
        transaction_tokens = set().union(*(_tokens(row.get(col)) for col in text_columns if col in frame.columns))
        scores = []
        for path, tokens in zip(taxonomy_paths, path_tokens):
            overlap = len(transaction_tokens & tokens)
            score = overlap / len(tokens) if tokens else 0.0
            scores.append(score)
        best = max(scores, default=0.0)
        winners = [path for path, score in zip(taxonomy_paths, scores) if score == best and best > 0]
        if winners:
            depth = min(map(len, winners))
            common = []
            for position in range(depth):
                values = {path[position] for path in winners if len(path) > position}
                if len(values) != 1:
                    break
                common.append(values.pop())
            values = {f"l{i+1}": common[i] if i < len(common) else "" for i in range(len(level_names))}
            status = "CLASSIFIED" if common else "AMBIGUOUS_TAXONOMY"
        else:
            values = {f"l{i+1}": "" for i in range(len(level_names))}
            status = "UNCLASSIFIED_NO_LEXICAL_EVIDENCE"
        rows.append({**values, "confidence": float(best), "classification_status": status,
                     "classification_method": "runtime_taxonomy_token_overlap"})
    return pd.DataFrame(rows, index=frame.index)
