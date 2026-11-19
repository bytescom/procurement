"""General non-destructive text cleanup helpers."""

import re
from typing import Iterable

import pandas as pd


def clean_dataframe(
    frame: pd.DataFrame,
    dataset_name: str,
    *,
    protected_columns: Iterable[str] = (),
    standardize_column_names: bool = False,
) -> tuple[pd.DataFrame, list[dict]]:
    """Return a cleaned copy; changed source values are retained in ``__raw`` columns."""
    result = frame.copy(deep=True)
    audit: list[dict] = []
    if standardize_column_names:
        normalized = [re.sub(r"[^0-9a-zA-Z]+", "_", str(c).strip()).strip("_").lower() for c in result.columns]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"Column-name standardization creates duplicate columns in {dataset_name}.")
        result.columns = normalized

    protected = set(protected_columns)
    null_tokens = {"", "null", "none", "nan", "n/a"}
    for column in result.columns:
        if column in protected or not (pd.api.types.is_object_dtype(result[column]) or
                                       pd.api.types.is_string_dtype(result[column])):
            continue
        original = result[column].copy(deep=True)

        def normalize(value):
            if not isinstance(value, str):
                return value
            stripped = value.strip()
            return pd.NA if stripped.casefold() in null_tokens else stripped

        cleaned = original.map(normalize)
        changed = int((original.astype("string").fillna("<NA>") != cleaned.astype("string").fillna("<NA>")).sum())
        if changed:
            raw_name = f"{column}__raw"
            if raw_name in result.columns:
                raise ValueError(f"Cannot preserve original {column!r}; {raw_name!r} already exists.")
            result[raw_name] = original
            result[column] = cleaned
            audit.append({"dataset": dataset_name, "operation": "trim_text_and_normalize_nulls",
                          "column": column, "changed_values": changed, "issues": 0,
                          "details": f"Original values retained in {raw_name}."})
    return result, audit
