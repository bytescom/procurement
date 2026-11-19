"""Amount, VAT, and FX normalization with per-row quality flags."""

import math
import re
from collections import Counter

import numpy as np
import pandas as pd


def _missing(value) -> bool:
    if value is None or pd.isna(value):
        return True
    return isinstance(value, str) and value.strip().casefold() in {"", "null", "none", "nan", "n/a"}


def _numeric(value):
    if _missing(value):
        return np.nan, True, False
    if isinstance(value, str):
        text = value.strip()
        if "," in text:
            if not re.fullmatch(r"[+-]?(?:\d{1,3}(?:,\d{3})+)(?:\.\d*)?(?:[eE][+-]?\d+)?", text):
                return np.nan, False, True
            text = text.replace(",", "")
        try:
            number = float(text)
        except ValueError:
            return np.nan, False, True
    else:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return np.nan, False, True
    return (number, False, False) if math.isfinite(number) else (np.nan, False, True)


def numeric_values(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    parsed = [_numeric(value) for value in series]
    return (pd.Series([v[0] for v in parsed], index=series.index, dtype="float64"),
            pd.Series([v[1] for v in parsed], index=series.index, dtype=bool),
            pd.Series([v[2] for v in parsed], index=series.index, dtype=bool))


def _audit(dataset, operation, column, changed, issues, details):
    return {"dataset": dataset, "operation": operation, "column": column,
            "changed_values": int(changed), "issues": int(issues), "details": details}


def normalize_erp_amounts(frame: pd.DataFrame, amount_column: str, dataset_name: str,
                          additional_amount_columns: tuple[str, ...] = ()) -> tuple[pd.DataFrame, list[dict]]:
    """ERP values are already net; parse numeric values without applying tax."""
    result = frame.copy(deep=True); audit = []
    for column in (amount_column, *additional_amount_columns):
        numeric, missing, invalid = numeric_values(result[column])
        result[f"{column}_numeric"] = numeric
        result[f"{column}_missing"] = missing
        result[f"{column}_invalid"] = invalid
        audit.append(_audit(dataset_name, "parse_numeric_amount", column, numeric.notna().sum(), invalid.sum(),
                            f"ERP net values retained; no VAT adjustment. Missing: {missing.sum()}."))
        if column == amount_column:
            result["net_amount"] = numeric
            result["negative_amount_suspected_reversal"] = (numeric < 0).fillna(False)
    return result, audit


def _vat_rate(value, default_rate):
    if _missing(value):
        return float(default_rate), True, False
    percent_sign = isinstance(value, str) and "%" in value
    text = value.strip().replace("%", "") if isinstance(value, str) else value
    amount, missing, invalid = _numeric(text)
    if missing or invalid or amount < 0:
        return np.nan, False, True
    rate = amount / 100 if percent_sign or amount > 1 else amount
    return (rate, False, False) if rate <= 1 else (np.nan, False, True)


def normalize_invoice_tool_amounts(frame: pd.DataFrame, gross_column: str, vat_column: str,
                                   dataset_name: str, default_vat_rate: float = 0.15) -> tuple[pd.DataFrame, list[dict]]:
    result = frame.copy(deep=True)
    gross, missing_gross, invalid_gross = numeric_values(result[gross_column])
    rates = [_vat_rate(v, default_vat_rate) for v in result[vat_column]]
    vat = pd.Series([x[0] for x in rates], index=result.index, dtype="float64")
    vat_defaulted = pd.Series([x[1] for x in rates], index=result.index, dtype=bool)
    invalid_vat = pd.Series([x[2] for x in rates], index=result.index, dtype=bool)
    result[f"{gross_column}_numeric"] = gross
    result["vat_rate_normalized"] = vat
    result["vat_rate_defaulted"] = vat_defaulted
    result["invalid_vat_rate"] = invalid_vat
    result["gross_amount_missing"] = missing_gross
    result["invalid_gross_amount"] = invalid_gross
    result["net_amount"] = gross / (1 + vat)
    result["negative_amount_suspected_reversal"] = (gross < 0).fillna(False)
    audit = [_audit(dataset_name, "parse_gross_amount", gross_column, gross.notna().sum(), invalid_gross.sum(),
                    f"Missing: {missing_gross.sum()}"),
             _audit(dataset_name, "normalize_vat_and_calculate_net", vat_column, vat.notna().sum(), invalid_vat.sum(),
                    f"Defaulted missing VAT to {default_vat_rate:.0%}: {vat_defaulted.sum()}; net = gross / (1 + VAT).")]
    return result, audit


def normalize_fx_rates(frame: pd.DataFrame, currency_column: str, month_column: str,
                       rate_column: str, dataset_name: str) -> tuple[pd.DataFrame, list[dict]]:
    result = frame.copy(deep=True)
    result["currency_normalized"] = result[currency_column].astype("string").str.strip().str.upper().replace("", pd.NA)
    parsed_month = pd.to_datetime(result[month_column], format="mixed", errors="coerce")
    result["month_normalized"] = parsed_month.dt.to_period("M").dt.to_timestamp()
    rates, missing, invalid_numeric = numeric_values(result[rate_column])
    invalid = invalid_numeric | (rates <= 0).fillna(False)
    result["units_per_usd_numeric"] = rates
    result["fx_rate_missing"] = missing
    result["fx_rate_invalid"] = invalid
    result["fx_currency_key_invalid"] = result["currency_normalized"].isna()
    result["fx_month_key_invalid"] = result["month_normalized"].isna()
    key_ok = result["currency_normalized"].notna() & result["month_normalized"].notna()
    duplicates = pd.Series(False, index=result.index)
    duplicates.loc[key_ok] = result.loc[key_ok].duplicated(["currency_normalized", "month_normalized"], keep=False)
    result["fx_duplicate_key"] = duplicates
    audit = [_audit(dataset_name, "validate_fx_keys_and_rates", f"{currency_column}/{month_column}/{rate_column}",
                    len(result), result[["fx_rate_missing", "fx_rate_invalid", "fx_currency_key_invalid", "fx_month_key_invalid", "fx_duplicate_key"]].any(axis=1).sum(),
                    f"Rows: {len(result)}; duplicate key rows: {duplicates.sum()}; missing/invalid rates: {(missing | invalid).sum()}.")]
    return result, audit


def normalize_company_b_amounts(frame: pd.DataFrame, fx_frame: pd.DataFrame, *, amount_column: str,
                                currency_column: str, rate_column: str, dataset_name: str,
                                fx_fallback_order: tuple[str, ...] = ("invoice_month", "posting_month")) -> tuple[pd.DataFrame, list[dict]]:
    """USD lines stay unchanged; other lines divide by a valid line or monthly FX rate."""
    result = frame.copy(deep=True)
    currency = result[currency_column].astype("string").str.strip().str.upper()
    result["currency_normalized"] = currency.replace("", pd.NA)
    amount, missing_amount, invalid_amount = numeric_values(result[amount_column])
    line_rate, missing_rate, invalid_numeric = numeric_values(result[rate_column])
    invalid_rate = invalid_numeric | (line_rate <= 0).fillna(False)
    result[f"{amount_column}_numeric"] = amount
    result["currency_rate_numeric"] = line_rate
    result["currency_rate_missing"] = missing_rate
    result["currency_rate_invalid"] = invalid_rate
    result["amount_missing"] = missing_amount
    result["amount_invalid"] = invalid_amount
    result["fx_rate_used"] = np.nan
    result["fx_rate_source"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["conversion_status"] = pd.Series("unresolved", index=result.index, dtype="string")
    result["missing_fx_match"] = False
    result["invalid_fx_match"] = False
    result["duplicate_fx_match"] = False
    usd = result["currency_normalized"].eq("USD")
    result.loc[usd, "conversion_status"] = "usd_no_conversion"
    result.loc[usd, "fx_rate_source"] = "USD"
    non_usd_valid = ~usd & ~missing_rate & ~invalid_rate & result["currency_normalized"].notna()
    result.loc[non_usd_valid, "fx_rate_used"] = line_rate[non_usd_valid]
    result.loc[non_usd_valid, "fx_rate_source"] = "invoice_currency_rate"
    result.loc[non_usd_valid, "conversion_status"] = "converted_line_rate"
    result.loc[~usd & invalid_rate, "conversion_status"] = "invalid_invoice_rate"
    result.loc[~usd & result["currency_normalized"].isna(), "conversion_status"] = "missing_currency"

    # Build one status row per normalized FX key. Duplicate source keys remain in
    # fx_frame and are deliberately unusable rather than arbitrarily selected.
    keyed = fx_frame.loc[fx_frame["currency_normalized"].notna() & fx_frame["month_normalized"].notna()].copy()
    status_rows = []
    for (cur, month), group in keyed.groupby(["currency_normalized", "month_normalized"], sort=False):
        if len(group) != 1: status, fx_rate = "duplicate_fx_key", np.nan
        elif group["fx_rate_invalid"].iloc[0]: status, fx_rate = "invalid_fx_rate", np.nan
        elif group["fx_rate_missing"].iloc[0]: status, fx_rate = "missing_fx_rate", np.nan
        else: status, fx_rate = "valid", float(group["units_per_usd_numeric"].iloc[0])
        status_rows.append({"currency_normalized": cur, "month_normalized": month,
                            "lookup_status": status, "lookup_rate": fx_rate})
    lookup = pd.DataFrame(status_rows, columns=["currency_normalized", "month_normalized", "lookup_status", "lookup_rate"])
    unresolved = result.index[(~usd) & missing_rate & result["currency_normalized"].notna()]
    fallback = {"invoice_month": "invoice_date_parsed", "posting_month": "posting_date_parsed"}
    for month_key in fx_fallback_order:
        date_column = fallback.get(month_key)
        if not date_column or date_column not in result.columns or not len(unresolved): continue
        months = result.loc[unresolved, date_column].dt.to_period("M").dt.to_timestamp()
        candidates = pd.DataFrame({"row_id": unresolved, "currency_normalized": result.loc[unresolved, "currency_normalized"].values,
                                   "month_normalized": months.values})
        has_month = candidates["month_normalized"].notna()
        matched = candidates.loc[has_month].merge(lookup, on=["currency_normalized", "month_normalized"],
                                                   how="left", validate="many_to_one", sort=False)
        if len(matched) != int(has_month.sum()): raise RuntimeError("FX lookup changed invoice row cardinality.")
        matched = matched.set_index("row_id")
        for row_id in matched.index:
            status = matched.loc[row_id, "lookup_status"]
            if pd.isna(status): continue
            if status == "valid":
                result.loc[row_id, "fx_rate_used"] = matched.loc[row_id, "lookup_rate"]
                result.loc[row_id, "fx_rate_source"] = month_key
                result.loc[row_id, "conversion_status"] = "converted_fx_table"
                unresolved = unresolved.drop(row_id)
            else:
                result.loc[row_id, "conversion_status"] = status
                result.loc[row_id, "invalid_fx_match"] = status in {"invalid_fx_rate", "missing_fx_rate"}
                result.loc[row_id, "duplicate_fx_match"] = status == "duplicate_fx_key"
                unresolved = unresolved.drop(row_id)
    result.loc[unresolved, "missing_fx_match"] = True
    result.loc[unresolved, "conversion_status"] = "missing_fx_match"
    result["unresolved_conversion"] = (~usd) & result["fx_rate_used"].isna()
    result["usd_amount"] = np.where(usd, amount, amount / result["fx_rate_used"])
    result.loc[invalid_amount | missing_amount, "usd_amount"] = np.nan
    result["negative_amount_suspected_reversal"] = (amount < 0).fillna(False)
    return result, [_audit(dataset_name, "convert_amount_to_usd", amount_column, result["usd_amount"].notna().sum(),
                           result["unresolved_conversion"].sum(),
                           f"USD unchanged; non-USD divided by selected rate. Unresolved: {result['unresolved_conversion'].sum()}.")]
