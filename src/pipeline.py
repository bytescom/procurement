"""Unattended end-to-end procurement output generation."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.loader import load_company_data
from src.preprocessing import preprocess_company_data
from src.matching import match_company_a_transactions, investigate_company_b_duplicates
from src.spend import decide_company_a_spend, decide_company_b_spend
from src.identity import resolve_supplier_and_item_keys
from src.classification import load_taxonomy
from src.classification_engine import classify_with_decision_engine


OUTPUT_COLUMNS = ["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend",
                  "l1", "l2", "l3", "l4", "confidence"]


def _record_id(company_id, dataset, row, spec):
    raw_id = row[spec.get("columns", {}).get("id", spec.get("id_column", "line_id"))]
    if company_id == "A" and dataset == "erp":
        return f"ERP:{raw_id}"
    if company_id == "A" and dataset == "invoice_tool":
        return f"TOOL:{raw_id}"
    return str(raw_id)


def _transaction_datasets(config):
    if "erp" in config["datasets"] and "invoice_tool" in config["datasets"]:
        return ["erp", "invoice_tool"]
    explicit = config.get("transaction_dataset")
    if explicit:
        return [explicit]
    return [k for k in config["datasets"] if not any(x in k.casefold() for x in ("taxonomy", "fx", "label"))][:1]


def _spend_decisions(data, company_id, transaction_datasets, match_result, duplicate_result):
    if company_id == "A":
        return decide_company_a_spend(data, match_result["records"])
    if company_id == "B":
        return decide_company_b_spend(data)
    decisions = []
    cfg = data["config"]
    for ds in transaction_datasets:
        frame, spec = data["datasets"][ds], cfg["datasets"][ds]
        status_col = spec.get("columns", {}).get("status")
        description_col = spec.get("columns", {}).get("description")
        rules = cfg.get("spend_rules", {})
        valid = {str(v).strip().casefold() for v in rules.get("approved_or_paid_statuses", [])}
        cancelled = {str(v).strip().casefold() for v in rules.get("cancelled_statuses", [])}
        rejected = {str(v).strip().casefold() for v in rules.get("not_approved_statuses", [])}
        non_procurement_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in rules.get("non_procurement_description_patterns", [])
        ]
        for _, row in frame.iterrows():
            rid = _record_id(company_id, ds, row, spec)
            status = str(row.get(status_col, "")).strip().casefold() if status_col else ""
            description = str(row.get(description_col, "")) if description_col else ""
            include, reason, evidence = 1, "", "No configured exclusion rule matched."
            if status_col and status in cancelled:
                include, reason = 0, "CANCELLED"
                evidence = f"Configured status {status!r}."
            elif status_col and status in rejected:
                include, reason = 0, "NOT_APPROVED"
                evidence = f"Configured status {status!r}."
            elif valid and status not in valid:
                include, reason = 0, "NOT_APPROVED"
                evidence = f"Status {status!r} is not configured as approved/paid."
            else:
                matched_pattern = next(
                    (pattern.pattern for pattern in non_procurement_patterns if pattern.search(description)),
                    None,
                )
                if matched_pattern:
                    include, reason, evidence = (
                        0,
                        "NON_PROCUREMENT",
                        f"Description matched configured pattern {matched_pattern!r}.",
                    )
            decisions.append({"record_id": rid, "include": include, "exclusion_reason": reason,
                              "decision_evidence": evidence, "decision_method": "config_rules_or_include_default"})
    return pd.DataFrame(decisions)


def build_company_output(data: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    cfg, cid = data["config"], str(data["company_id"])
    tx_datasets = _transaction_datasets(cfg)
    if not tx_datasets:
        raise ValueError("No transaction dataset is configured.")
    match_result = duplicate_result = None
    if cid == "A" and {"erp", "invoice_tool"}.issubset(data["datasets"]):
        match_result = match_company_a_transactions(data)
    elif cid == "B" and "invoices" in data["datasets"]:
        duplicate_result = investigate_company_b_duplicates(data)
    spend = _spend_decisions(data, cid, tx_datasets, match_result, duplicate_result)
    identities = resolve_supplier_and_item_keys(data, tx_datasets)
    identity = identities["resolutions"].drop_duplicates("record_id")
    spend = spend.drop_duplicates("record_id")
    records = []
    classifier_inputs = []
    amount_audit_rows = []
    for ds in tx_datasets:
        frame, spec = data["datasets"][ds], cfg["datasets"][ds]
        semantic = spec.get("columns", {})
        amount_col = semantic.get("amount") or semantic.get("gross_amount")
        for _, row in frame.iterrows():
            rid = _record_id(cid, ds, row, spec)
            val = row.get("usd_amount") if cid == "B" else row.get("net_amount")
            if val is None or pd.isna(val):
                if amount_col: val = row.get(f"{amount_col}_numeric", row.get(amount_col))
            if val is not None and not pd.isna(val):
                try: val = float(val)
                except (ValueError, TypeError): val = np.nan
            else: val = np.nan
            records.append({"record_id": rid, "dataset": ds, "source_row": row, "normalized_spend": val})
            classifier_inputs.append({"record_id": rid, "dataset": ds, **row.to_dict()})
            amount_audit = {"record_id": rid, "dataset": ds, "source_amount_column": amount_col,
                            "raw_amount": row.get(amount_col) if amount_col else None,
                            "normalized_spend": val}
            for diagnostic in ("currency", "currency_rate", "net_amount", "usd_amount", "conversion_status",
                               "invalid_amount", "invalid_gross_amount", "invalid_vat_rate",
                               "amount_missing", "amount_invalid", "currency_rate_invalid",
                               "unresolved_conversion", "missing_fx_rate", "invalid_fx_match",
                               "duplicate_fx_match"):
                if diagnostic in row.index:
                    amount_audit[diagnostic] = row.get(diagnostic)
            amount_audit_rows.append(amount_audit)
    base = pd.DataFrame([{k: v for k, v in r.items() if k != "source_row"} for r in records])
    base["spend"] = [r["normalized_spend"] for r in records]
    base = base.merge(spend, on="record_id", how="left", validate="one_to_one").merge(
        identity, on="record_id", how="left", validate="one_to_one")
    if base["include"].isna().any() or base["supplier_key"].isna().any() or base["item_key"].isna().any():
        raise ValueError("A transaction is missing its eligibility or entity-resolution result.")
    base.loc[base["include"].eq(0), "spend"] = 0.0

    level_names, paths = load_taxonomy(data)
    detail = pd.DataFrame(classifier_inputs)
    text_semantics = {"description", "material_description", "service_description", "item_id", "uom",
                      "gl_account", "gl_account_description", "region",
                      "supplier_name", "supplier_type", "document_type", "spend_type", "material_type",
                      "cost_centre", "po_type"}
    text_columns = list(dict.fromkeys(
        cfg["datasets"][ds].get("columns", {}).get(name) for ds in tx_datasets for name in text_semantics
        if cfg["datasets"][ds].get("columns", {}).get(name)
    ))
    detail["include"] = base["include"].to_numpy()
    classified, classifier_reports = classify_with_decision_engine(
        detail, text_columns=text_columns, taxonomy_paths=paths, level_names=level_names,
        data=data, config=cfg.get("classification_engine", {}),
        dry_run=bool(cfg.get("_gemini_dry_run", False)))
    base = base.merge(classified, on="record_id", how="left", validate="one_to_one")
    for col in ["l1", "l2", "l3", "l4"]:
        if col not in base: base[col] = ""
    base.loc[base.include.eq(0), ["l1", "l2", "l3", "l4"]] = ""
    base.loc[base.include.eq(0), "confidence"] = 0.0
    out = base[OUTPUT_COLUMNS].copy()
    out["exclusion_reason"] = out["exclusion_reason"].fillna("")
    out["supplier_key"] = out["supplier_key"].fillna("")
    out["item_key"] = out["item_key"].fillna("")
    out[["l1", "l2", "l3", "l4"]] = out[["l1", "l2", "l3", "l4"]].fillna("")
    _validate_output(out)
    reports = {"spend_decisions": spend, "supplier_map": identities["supplier_map"],
               "item_map": identities["item_map"], "row_counts": data["row_counts"],
               "preprocessing_audit": data.get("audit_log", pd.DataFrame()),
               "amount_conversion_audit": pd.DataFrame(amount_audit_rows),
               "classification_diagnostics": classified,
               "classification_metrics": classifier_reports["metrics"],
               "classification_confusion": classifier_reports["confusion"],
               "classification_decisions": classifier_reports["decisions"],
               "gemini_run_metrics": classifier_reports["gemini_run"],
               "classification_split": classifier_reports["split_report"]}
    if match_result:
        reports["transaction_matches"] = match_result["records"]
        reports["tool_duplicate_groups"] = match_result["tool_duplicate_groups"]
        reports["matching_summary"] = match_result["summary"]
    if duplicate_result:
        reports["company_b_duplicate_summary"] = duplicate_result["summary"]
        reports["company_b_duplicate_candidates"] = duplicate_result["candidate_records"]
        reports["company_b_duplicates_by_source"] = duplicate_result["by_source_system"]
    evaluation = _evaluate_reference_labels(data, out,
        allowed_record_ids=classifier_reports.get("split", {}).get("test"))
    if not evaluation.empty: reports["label_evaluation"] = evaluation
    return out, reports


def _validate_output(frame: pd.DataFrame) -> None:
    if list(frame.columns) != OUTPUT_COLUMNS:
        raise ValueError("Output columns do not match the official schema.")
    if frame.record_id.isna().any() or frame.record_id.duplicated().any():
        raise ValueError("Output record_id values must be complete and unique.")
    if not frame.include.isin([0, 1]).all():
        raise ValueError("Output include values must be 0 or 1.")
    included_with_reason = frame.loc[frame.include.eq(1), "exclusion_reason"].astype(str).str.len().gt(0)
    if included_with_reason.any():
        raise ValueError("Included rows must have blank exclusion_reason.")
    excluded_without_reason = frame.loc[frame.include.eq(0), "exclusion_reason"].astype(str).str.len().eq(0)
    if excluded_without_reason.any():
        raise ValueError("Every excluded row needs a reason.")
    if (frame.loc[frame.include.eq(0), "spend"].fillna(0) != 0).any():
        raise ValueError("Excluded rows must have zero spend.")
    known = {"CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE"}
    exclusion_reasons = set(frame.loc[frame.include.eq(0), "exclusion_reason"])
    if not exclusion_reasons.issubset(known):
        raise ValueError("Unknown exclusion reason in output.")
    confidence = pd.to_numeric(frame.confidence, errors="coerce")
    if confidence.notna().any() and not confidence.dropna().between(0, 1).all():
        raise ValueError("Confidence must be between 0 and 1.")


def _evaluate_reference_labels(data: dict, output: pd.DataFrame,
                               allowed_record_ids: set[str] | None = None) -> pd.DataFrame:
    """Evaluate predictions after inference; reference labels never enter decisions."""
    cfg = data["config"]
    label_key = next((k for k in cfg["datasets"] if k in {"labels", "seed_labels"}), None)
    if not label_key:
        return pd.DataFrame()
    spec, labels = cfg["datasets"][label_key], data["datasets"][label_key]
    mapping = spec.get("columns", {})
    id_col = mapping.get("record_id") or mapping.get("id") or spec.get("id_column")
    if not id_col or id_col not in labels:
        return pd.DataFrame()
    label_ids = labels[id_col].astype("string")
    ref = pd.DataFrame({"record_id": label_ids.map(
        lambda x: (str(x) if str(x).startswith("ERP:") else f"ERP:{x}")
        if str(data["company_id"]) == "A" else str(x))})
    rows, levels = [], cfg.get("taxonomy", {}).get("levels", [])
    for i, semantic in enumerate(levels):
        truth_col = mapping.get(semantic)
        pred_col = f"l{i+1}"
        if truth_col and truth_col in labels:
            ref[f"truth_{pred_col}"] = labels[truth_col].astype("string").fillna("")
    joined = ref.merge(output, on="record_id", how="inner", validate="one_to_one")
    taxonomy_joined = joined
    if allowed_record_ids:
        taxonomy_joined = joined[joined.record_id.astype(str).isin(allowed_record_ids)]
    for i, _ in enumerate(levels):
        pred_col, truth_col = f"l{i+1}", f"truth_l{i+1}"
        if truth_col not in joined:
            continue
        mask = taxonomy_joined[truth_col].ne("")
        if mask.any():
            truth_values = taxonomy_joined.loc[mask, truth_col].fillna("")
            pred_values = taxonomy_joined.loc[mask, pred_col].fillna("")
            nonblank = pred_values.ne("")
            rows.append({"evaluation": f"taxonomy_level_{i+1}", "n_labelled": int(mask.sum()),
                         "exact_accuracy": float((pred_values == truth_values).mean()),
                         "prediction_coverage": float(nonblank.mean()),
                         "accuracy_when_predicted": float((pred_values[nonblank] == truth_values[nonblank]).mean()) if nonblank.any() else np.nan,
                         "blank_prediction_pct": float((~nonblank).mean()),
                         "source": "held-out test comparison; test labels excluded from model fit"})
    include_col = mapping.get("include")
    if include_col and include_col in labels:
        truth = pd.Series(pd.to_numeric(labels[include_col], errors="coerce").values, index=ref.record_id)
        preds = output.set_index("record_id")["include"]
        common = truth.index.intersection(preds.index)
        y, p = truth.loc[common], preds.loc[common].astype(int)
        valid = y.notna(); y, p = y[valid].astype(int), p[valid]
        tp, fp = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum())
        fn, tn = int(((y == 1) & (p == 0)).sum()), int(((y == 0) & (p == 0)).sum())
        rows.append({"evaluation": "spend_include_seed_labels", "n_labelled": len(y),
                     "accuracy": float((y == p).mean()) if len(y) else np.nan,
                     "precision_included": tp / (tp + fp) if tp + fp else np.nan,
                     "recall_included": tp / (tp + fn) if tp + fn else np.nan,
                     "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "source": "post-run comparison; seed labels excluded from production inference"})
    return pd.DataFrame(rows)


def run_pipeline(config_path: str, output_dir: str = "output", name: str = "candidate",
                 gemini_dry_run: bool = False) -> tuple[Path, dict]:
    loaded = load_company_data(config_path)
    data = preprocess_company_data(loaded)
    data["config"]["_gemini_dry_run"] = gemini_dry_run
    output, reports = build_company_output(data)
    cid = str(data["company_id"])
    outdir = Path(output_dir); audit_dir = outdir / "audit" / cid
    audit_dir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{name}__{cid}.csv"
    output.to_csv(outfile, index=False)
    for report_name, table in reports.items(): table.to_csv(audit_dir / f"{report_name}.csv", index=False)
    return outfile, {"company_id": cid, "rows": len(output), "included": int(output.include.sum()),
                     "excluded": int(output.include.eq(0).sum()), "unclassified_included": int((output.include.eq(1) & output.l1.eq("")).sum()),
                     "missing_spend": int((output.include.eq(1) & output.spend.isna()).sum()),
                     "output_path": str(outfile), "audit_dir": str(audit_dir)}


def main():
    parser = argparse.ArgumentParser(description="Run the Cantill procurement pipeline for one configured company.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--config", help="Path to a company YAML config")
    target.add_argument("--all", action="store_true", help="Run configs/company_A.yaml and company_B.yaml")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--name", default="candidate", help="Output filename prefix")
    parser.add_argument("--gemini-dry-run", action="store_true",
                        help="Estimate unique Gemini fallback requests and costs without API calls")
    args = parser.parse_args()
    configs = ["configs/company_A.yaml", "configs/company_B.yaml"] if args.all else [args.config]
    for config_path in configs:
        path, summary = run_pipeline(config_path, args.output_dir, args.name,
                                     gemini_dry_run=args.gemini_dry_run)
        print(pd.Series(summary).to_string())


if __name__ == "__main__": main()
