"""Configuration-aware preprocessing orchestration for the supplied companies."""

from pathlib import Path
from typing import Any, Union

import pandas as pd

from src.ingestion.loader import load_company_data
from .amounts import (normalize_company_b_amounts, normalize_erp_amounts,
                      normalize_fx_rates, normalize_invoice_tool_amounts)
from .cleaning import clean_dataframe
from .dates import parse_date_columns


_IDENTIFIER_SEMANTICS = {"id", "supplier_id", "item_id", "invoice_number", "invoice_line",
                         "po_number", "po_line", "po_item", "po_reference", "receipt_document",
                         "receipt_item", "contract_number", "vendor_id", "material_number",
                         "line_id", "tool_record_id", "record_id", "currency"}


def _protected_columns(spec: dict) -> set[str]:
    columns = set()
    if spec.get("id_column"):
        columns.add(spec["id_column"])
    for semantic, source in spec.get("columns", {}).items():
        if semantic.lower() in _IDENTIFIER_SEMANTICS or any(
            token in semantic.lower() for token in ("amount", "date", "rate", "vat", "currency", "month")
        ):
            columns.add(source)
    return columns


def preprocess_company_data(
    config_or_loaded: Union[str, Path, dict[str, Any]],
    project_root: Union[str, Path, None] = None,
) -> dict[str, Any]:
    """Clean all configured datasets on copies and return frames plus audit summaries.

    Accepts either a config path/dict or the result of ``load_company_data``.
    Original input frames and source files are never modified.
    """
    if isinstance(config_or_loaded, dict) and "datasets" in config_or_loaded and "config" in config_or_loaded:
        loaded = config_or_loaded
    else:
        loaded = load_company_data(config_or_loaded, project_root=project_root)
    config = loaded["config"]
    source_frames = loaded["datasets"]
    specs = config["datasets"]
    frames: dict[str, pd.DataFrame] = {}
    audit: list[dict] = []
    input_counts = {key: len(df) for key, df in source_frames.items()}

    for key, frame in source_frames.items():
        cleaned, entries = clean_dataframe(frame, key, protected_columns=_protected_columns(specs[key]))
        frames[key] = cleaned
        audit.extend(entries)

    # Date fields are derived from config's semantic mappings; original columns stay intact.
    for key, spec in specs.items():
        dates = [spec.get("columns", {}).get(name) for name in
                 ("posting_date", "invoice_date", "due_date", "paid_date")]
        dates = list(dict.fromkeys(column for column in dates if column and column in frames[key].columns))
        if dates:
            frames[key], entries = parse_date_columns(frames[key], dates, key)
            audit.extend(entries)

    if "erp" in frames and "invoice_tool" in frames:
        erp_spec = specs["erp"]["columns"]
        erp_amount = erp_spec.get("amount", "value_zar")
        unit_cost = erp_spec.get("unit_cost")
        frames["erp"], entries = normalize_erp_amounts(
            frames["erp"], erp_amount, "erp", (unit_cost,) if unit_cost else ()
        )
        audit.extend(entries)

        tool_spec = specs["invoice_tool"]["columns"]
        amount_settings = config.get("tax_settings", {})
        default_vat = float(amount_settings.get("default_vat_rate", 0.15))
        frames["invoice_tool"], entries = normalize_invoice_tool_amounts(
            frames["invoice_tool"], tool_spec.get("gross_amount", "gross_amount"),
            tool_spec.get("vat_rate", "vat_rate"), "invoice_tool", default_vat
        )
        audit.extend(entries)

    if "invoices" in frames and "fx_rates" in frames:
        fx_spec = specs["fx_rates"]["columns"]
        frames["fx_rates"], entries = normalize_fx_rates(
            frames["fx_rates"], fx_spec.get("currency", "currency"),
            fx_spec.get("month", "month"), fx_spec.get("units_per_usd", "units_per_usd"), "fx_rates"
        )
        audit.extend(entries)
        invoice_spec = specs["invoices"]["columns"]
        fx_settings = config.get("currency_settings", {})
        fallback = tuple(fx_settings.get("fx_fallback_order", ["invoice_month", "posting_month"]))
        frames["invoices"], entries = normalize_company_b_amounts(
            frames["invoices"], frames["fx_rates"],
            amount_column=invoice_spec.get("amount", "line_amount"),
            currency_column=invoice_spec.get("currency", "currency"),
            rate_column=invoice_spec.get("currency_rate", "currency_rate"),
            dataset_name="invoices", fx_fallback_order=fallback
        )
        audit.extend(entries)

    row_counts = pd.DataFrame([
        {"dataset": key, "input_rows": count, "output_rows": len(frames[key]),
         "row_difference": len(frames[key]) - count}
        for key, count in input_counts.items()
    ])
    return {
        "config": config,
        "company_id": loaded.get("company_id", config["company"].get("id")),
        "company_name": loaded.get("company_name", config["company"].get("name")),
        "reporting_currency": loaded.get("reporting_currency", config["company"].get("reporting_currency")),
        "datasets": frames,
        "audit_log": pd.DataFrame(audit, columns=["dataset", "operation", "column", "changed_values", "issues", "details"]),
        "row_counts": row_counts,
    }
