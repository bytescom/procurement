"""Date parsing that preserves source values and distinguishes missing from invalid."""

import pandas as pd


def parse_date_columns(frame: pd.DataFrame, columns: list[str], dataset_name: str,
                       *, dayfirst: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    result = frame.copy(deep=True)
    audit = []
    for column in columns:
        if column not in result.columns:
            raise KeyError(f"Date column {column!r} is missing from dataset {dataset_name!r}.")
        parsed_name = f"{column}_parsed"
        if parsed_name in result.columns:
            raise ValueError(f"Refusing to overwrite existing normalized date column {parsed_name!r}.")
        raw = result[column]
        missing = raw.isna() | raw.astype("string").str.strip().eq("").fillna(False)
        parsed = pd.to_datetime(raw, format="mixed", errors="coerce", dayfirst=dayfirst)
        invalid = ~missing & parsed.isna()
        result[parsed_name] = parsed
        result[f"{column}_missing"] = missing.astype(bool)
        result[f"{column}_invalid"] = invalid.astype(bool)
        audit.append({"dataset": dataset_name, "operation": "parse_date", "column": column,
                      "changed_values": int(parsed.notna().sum()), "missing": int(missing.sum()),
                      "issues": int(invalid.sum()), "details": f"Output: {parsed_name}; missing and invalid flagged separately."})
    return result, audit
