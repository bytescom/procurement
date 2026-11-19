"""Spend eligibility decisions with explicit, auditable exclusion reasons."""

import re

import pandas as pd


ALLOWED = {"CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE"}


def _decision(record_id, include, reason, evidence, method):
    return {"record_id": record_id, "include": int(include), "exclusion_reason": "" if include else reason,
            "decision_evidence": evidence, "decision_method": method}


def decide_company_a_spend(preprocessed: dict, matches: pd.DataFrame) -> pd.DataFrame:
    cfg = preprocessed["config"]
    rules = cfg.get("spend_rules", {}).get("invoice_tool_statuses", {})
    statuses = {k: {str(v).strip().casefold() for v in vals} for k, vals in rules.items()}
    output = []
    tool_spec = cfg["datasets"]["invoice_tool"]["columns"]
    id_col, status_col = tool_spec.get("id", "tool_record_id"), tool_spec.get("status", "status")
    frame = preprocessed["datasets"]["invoice_tool"]
    status_by_id = dict(zip(frame[id_col].astype("string"), frame[status_col]))
    for row in matches.to_dict("records"):
        rid, source = row["record_id"], row["source"]
        if row["duplicate_decision"] == "DUPLICATE":
            output.append(_decision(rid, 0, "DUPLICATE", row["decision_reason"], row["matching_method"] or "transaction_match")); continue
        if source == "ERP":
            output.append(_decision(rid, cfg.get("spend_rules", {}).get("erp_default_include", True),
                                    "", "ERP goods receipt; retained as the canonical record when matched.", "configured_erp_default")); continue
        # Tool row status is looked up from the original preprocessed dataset.
        raw_id = rid.removeprefix("TOOL:")
        raw_status = status_by_id.get(raw_id, "")
        status = "" if pd.isna(raw_status) else str(raw_status).strip().casefold()
        if status in statuses.get("cancelled", set()):
            output.append(_decision(rid, 0, "CANCELLED", f"Invoice tool status={status!r}.", "configured_status"))
        elif status in statuses.get("not_approved", set()):
            output.append(_decision(rid, 0, "NOT_APPROVED", f"Invoice tool status={status!r}.", "configured_status"))
        elif status in statuses.get("approved_or_paid", set()):
            output.append(_decision(rid, 1, "", f"Invoice tool status={status!r}.", "configured_status"))
        else:
            output.append(_decision(rid, 0, "NOT_APPROVED", f"Unrecognized or missing invoice tool status={status!r}; conservatively excluded.", "unknown_status"))
    result = pd.DataFrame(output)
    if not set(result.loc[result.include.eq(0), "exclusion_reason"]).issubset(ALLOWED):
        raise ValueError("Spend decision produced an unsupported exclusion reason.")
    return result


def decide_company_b_spend(preprocessed: dict) -> pd.DataFrame:
    cfg = preprocessed["config"]
    spec = cfg["datasets"]["invoices"]["columns"]
    frame = preprocessed["datasets"]["invoices"]
    rules = cfg.get("spend_rules", {})
    patterns = [re.compile(pattern) for pattern in rules.get("non_procurement_description_patterns", [])]
    desc_col = spec.get("gl_account_description", "gl_account_description")
    rows = []
    for _, row in frame.iterrows():
        desc = str(row.get(desc_col, ""))
        matched = next((pattern.pattern for pattern in patterns if pattern.search(desc)), None)
        rid = str(row[spec.get("id", "line_id")])
        if matched:
            rows.append(_decision(rid, 0, "NON_PROCUREMENT", f"Account description matched configured pattern {matched!r}: {desc!r}.", "configured_account_description"))
        else:
            rows.append(_decision(rid, 1, "", "No configured non-procurement account rule matched; approval state is absent in source.", "default_include_unclassified_status"))
    return pd.DataFrame(rows)
