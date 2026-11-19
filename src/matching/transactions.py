"""Conservative cross-system transaction matching and duplicate investigation."""

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .matching_utils import (amount_cents, get_column, normalize_po_reference,
                             normalize_reference, normalize_text, po_reference_parts, similarity)


def _value(row: pd.Series, column: str | None):
    return row.get(column) if column else None


def _month(value):
    if pd.isna(value):
        return None
    try:
        return pd.Timestamp(value).to_period("M")
    except (TypeError, ValueError):
        return None


def _record_ids(frame: pd.DataFrame, id_col: str, prefix: str = "") -> pd.Series:
    return prefix + frame[id_col].astype("string").fillna("")


def _tool_duplicate_groups(tool: pd.DataFrame, cols: dict) -> tuple[dict[int, str], list[dict]]:
    """Flag only repeated exports with the same complete invoice-line evidence."""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, row in tool.iterrows():
        invoice = normalize_reference(_value(row, cols.get("invoice")))
        supplier = normalize_text(_value(row, cols.get("supplier")))
        amount = amount_cents(_value(row, "net_amount"))
        description = normalize_text(_value(row, cols.get("description")))
        po, po_item = po_reference_parts(_value(row, cols.get("po")))
        date = _month(_value(row, f"{cols.get('date')}_parsed"))
        # Require invoice and supplier identifiers plus amount and line text. Missing
        # fields are not treated as wildcards; the row stays available for review.
        if invoice and supplier and amount is not None and description:
            groups[(invoice, supplier, po, po_item, amount, date, description)].append(i)
    duplicate_rows, evidence = {}, []
    for key, indices in groups.items():
        if len(indices) < 2:
            continue
        # Keep the first export; later identical exports are candidates for DUPLICATE.
        for index in indices[1:]:
            duplicate_rows[index] = str(tool.loc[indices[0], cols["id"]])
        evidence.append({"tool_record_ids": [str(tool.loc[i, cols["id"]]) for i in indices],
                         "method": "identical_normalized_invoice_line", "status": "DUPLICATE_TOOL",
                         "evidence": "Invoice, supplier, PO reference, net amount (cents), month and description agree."})
    return duplicate_rows, evidence


def match_company_a_transactions(preprocessed: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Match Company A ERP and invoice-tool rows using conservative staged evidence.

    Automatic pairs require a unique one-to-one candidate and exact supplier plus
    net amount at currency-cent precision. PO references are normalized only in
    temporary matching keys. Fuzzy candidates are review-only and never excluded.
    """
    config = preprocessed["config"]
    erp, tool = preprocessed["datasets"]["erp"].copy(deep=True), preprocessed["datasets"]["invoice_tool"].copy(deep=True)
    c = lambda ds, semantic, default: get_column(config, ds, semantic, default)
    ec = {"id": c("erp", "id", "line_id"), "po": c("erp", "po_number", "po_number"),
          "po_item": c("erp", "po_item", "po_item"),
          "supplier": c("erp", "supplier_name", "supplier_name"), "date": c("erp", "posting_date", "posting_date"),
          "description": c("erp", "material_description", "material_short_text")}
    tc = {"id": c("invoice_tool", "id", "tool_record_id"), "po": c("invoice_tool", "po_reference", "po_reference"),
          "invoice": c("invoice_tool", "invoice_number", "invoice_number"),
          "supplier": c("invoice_tool", "supplier_name", "supplier_name_as_captured"),
          "date": c("invoice_tool", "invoice_date", "invoice_date"),
          "description": c("invoice_tool", "description", "line_text")}
    erp["_record_id"] = _record_ids(erp, ec["id"], "ERP:")
    tool["_record_id"] = _record_ids(tool, tc["id"], "TOOL:")

    duplicate_tools, duplicate_evidence = _tool_duplicate_groups(tool, tc)
    erp_rows = {i: {"record_id": erp.loc[i, "_record_id"], "source": "ERP", "matched_record_id": None,
                    "match_status": "ERP_ONLY", "matching_method": None, "confidence": 0.0,
                    "matching_evidence": "No accepted cross-system match.", "duplicate_decision": "KEEP",
                    "decision_reason": "No accepted cross-system match.", "matched_record_ids": []} for i in erp.index}
    tool_rows = {i: {"record_id": tool.loc[i, "_record_id"], "source": "TOOL", "matched_record_id": None,
                     "match_status": "TOOL_ONLY", "matching_method": None, "confidence": 0.0,
                     "matching_evidence": "No accepted cross-system match.", "duplicate_decision": "KEEP",
                     "decision_reason": "Potential tool-only spend; defer spend decision.", "matched_record_ids": []} for i in tool.index}

    # Block by normalized PO reference, then require exact supplier and amount
    # evidence before automation. Candidate multiplicity stays explicitly ambiguous.
    erp_by_po, tool_by_po = defaultdict(list), defaultdict(list)
    for i, row in erp.iterrows():
        key = normalize_po_reference(row.get(ec["po"]));
        if key: erp_by_po[key].append(i)
    for j, row in tool.iterrows():
        key = normalize_po_reference(row.get(tc["po"]));
        if key: tool_by_po[key].append(j)
    candidates: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for key in erp_by_po.keys() & tool_by_po.keys():
        for i in erp_by_po[key]:
            erow = erp.loc[i]
            for j in tool_by_po[key]:
                trow = tool.loc[j]
                _, tool_item = po_reference_parts(trow.get(tc["po"]))
                erp_item = normalize_reference(erow.get(ec["po_item"]))
                if tool_item and erp_item and tool_item != erp_item:
                    continue
                supplier_equal = normalize_text(erow.get(ec["supplier"])) == normalize_text(trow.get(tc["supplier"]))
                amounts_equal = amount_cents(erow.get("net_amount")) is not None and amount_cents(erow.get("net_amount")) == amount_cents(trow.get("net_amount"))
                if supplier_equal and amounts_equal:
                    raw_equal = str(erow.get(ec["po"])).strip().casefold() == str(trow.get(tc["po"])).strip().casefold()
                    candidates[j].append((i, "exact_po_supplier_amount" if raw_equal else "normalized_po_supplier_amount"))

    # Detect potential split receipts where several ERP lines for one normalized
    # PO/supplier sum exactly to a tool invoice amount. Keep these review-only.
    erp_grouped = defaultdict(list)
    for i, row in erp.iterrows():
        po = (normalize_po_reference(row.get(ec["po"])), normalize_reference(row.get(ec["po_item"])))
        supplier = normalize_text(row.get(ec["supplier"]))
        cents = amount_cents(row.get("net_amount"))
        if po and supplier and cents is not None:
            erp_grouped[(po, supplier)].append((i, cents))
    split_candidates = {}
    for j, row in tool.iterrows():
        po_base, tool_item = po_reference_parts(row.get(tc["po"]))
        po = (po_base, tool_item)
        supplier = normalize_text(row.get(tc["supplier"]))
        cents = amount_cents(row.get("net_amount"))
        group = erp_grouped.get((po, supplier), []) if po and supplier else []
        if len(group) > 1 and sum(value for _, value in group) == cents:
            split_candidates[j] = [i for i, _ in group]

    # One-to-one acceptance: an edge must be the sole candidate for both endpoints.
    erp_edges = defaultdict(list)
    for j, edges in candidates.items():
        for i, _ in edges: erp_edges[i].append(j)
    accepted = []
    for j, edges in candidates.items():
        if len(edges) == 1 and len(erp_edges[edges[0][0]]) == 1 and j not in duplicate_tools and j not in split_candidates:
            accepted.append((edges[0][0], j, edges[0][1]))
    for i, j, method in accepted:
        evidence = "Unique normalized PO reference; supplier names agree; net amounts agree to cents."
        erp_rows[i].update(matched_record_id=tool.loc[j, "_record_id"], match_status="MATCHED",
                           matching_method=method, confidence=1.0, matching_evidence=evidence)
        erp_rows[i]["matched_record_ids"] = [str(tool.loc[j, "_record_id"])]
        tool_rows[j].update(matched_record_id=erp.loc[i, "_record_id"], match_status="MATCHED",
                            matching_method=method, confidence=1.0, matching_evidence=evidence,
                            duplicate_decision="DUPLICATE", decision_reason="Same purchase retained as ERP record per assessment rule.")
        tool_rows[j]["matched_record_ids"] = [str(erp.loc[i, "_record_id"])]

    paired_erp, paired_tool = {i for i, _, _ in accepted}, {j for _, j, _ in accepted}
    # Review-only PO candidates that could not be uniquely paired.
    for j, edges in candidates.items():
        if j in paired_tool: continue
        erp_ids = [str(erp.loc[i, "_record_id"]) for i, _ in edges]
        if erp_ids:
            tool_rows[j].update(match_status="AMBIGUOUS", matching_method="normalized_po_supplier_amount",
                                confidence=None, matching_evidence=f"Non-unique one-to-one candidates: {erp_ids}")
            tool_rows[j]["matched_record_ids"] = erp_ids
            for i, _ in edges:
                if i not in paired_erp:
                    erp_rows[i].update(match_status="AMBIGUOUS", matching_method="normalized_po_supplier_amount",
                                       confidence=None, matching_evidence=f"Tool candidate(s): {[str(tool.loc[x, '_record_id']) for x in erp_edges[i]]}")
                    erp_rows[i]["matched_record_ids"] = [str(tool.loc[x, "_record_id"]) for x in erp_edges[i]]

    for j, indices in split_candidates.items():
        if j in paired_tool or j in duplicate_tools: continue
        ids = [str(erp.loc[i, "_record_id"]) for i in indices]
        tool_rows[j].update(match_status="AMBIGUOUS", matching_method="split_receipt_sum_review",
                            confidence=None, matching_evidence=f"Multiple ERP lines under the same normalized PO and supplier sum to this tool net amount: {ids}")
        tool_rows[j]["matched_record_ids"] = ids
        for i in indices:
            if i not in paired_erp:
                erp_rows[i].update(match_status="AMBIGUOUS", matching_method="split_receipt_sum_review",
                                   confidence=None, matching_evidence=f"ERP line is part of a possible split receipt for TOOL:{tool.loc[j, tc['id']]}")
                erp_rows[i]["matched_record_ids"] = [str(tool.loc[j, "_record_id"])]

    # Composite stage: unique exact supplier + amount + calendar month, even if
    # PO formatting differs or is absent. This is stricter than fuzzy review.
    composite = defaultdict(list)
    for i, row in erp.iterrows():
        if i in paired_erp: continue
        key = (normalize_text(row.get(ec["supplier"])), amount_cents(row.get("net_amount")),
               _month(row.get(f"{ec['date']}_parsed")))
        if all(part is not None for part in key): composite[key].append(i)
    tool_composite = defaultdict(list)
    for j, row in tool.iterrows():
        if j in paired_tool or j in duplicate_tools: continue
        key = (normalize_text(row.get(tc["supplier"])), amount_cents(row.get("net_amount")),
               _month(row.get(f"{tc['date']}_parsed")))
        if all(part is not None for part in key): tool_composite[key].append(j)
    for key in composite.keys() & tool_composite.keys():
        eis, tjs = composite[key], tool_composite[key]
        if len(eis) == len(tjs) == 1:
            i, j = eis[0], tjs[0]
            evidence = "Unique exact supplier, net amount to cents, and matching calendar month; no shared PO required."
            erp_rows[i].update(matched_record_id=tool.loc[j, "_record_id"], match_status="MATCHED",
                               matching_method="composite_supplier_amount_month", confidence=1.0,
                               matching_evidence=evidence)
            erp_rows[i]["matched_record_ids"] = [str(tool.loc[j, "_record_id"])]
            tool_rows[j].update(matched_record_id=erp.loc[i, "_record_id"], match_status="MATCHED",
                                matching_method="composite_supplier_amount_month", confidence=1.0,
                                matching_evidence=evidence, duplicate_decision="DUPLICATE",
                                decision_reason="Same purchase retained as ERP record per assessment rule.")
            tool_rows[j]["matched_record_ids"] = [str(erp.loc[i, "_record_id"])]
            paired_erp.add(i); paired_tool.add(j)

    # Composite and fuzzy stages generate review candidates only. Blocking by
    # amount cents and month avoids a quadratic all-record comparison; similarity
    # ranks evidence but has no arbitrary acceptance cutoff.
    erp_block = defaultdict(list)
    for i, row in erp.iterrows():
        if i in paired_erp: continue
        amount = amount_cents(row.get("net_amount")); month = _month(row.get(f"{ec['date']}_parsed"))
        if amount is not None and month is not None: erp_block[(amount, month)].append(i)
    for j, row in tool.iterrows():
        if j in paired_tool or j in duplicate_tools: continue
        amount = amount_cents(row.get("net_amount")); month = _month(row.get(f"{tc['date']}_parsed"))
        if amount is None or month is None: continue
        possible = erp_block.get((amount, month), [])
        scored = []
        for i in possible:
            ssim = similarity(erp.loc[i].get(ec["supplier"]), row.get(tc["supplier"]))
            dsim = similarity(erp.loc[i].get(ec["description"]), row.get(tc["description"]))
            if ssim > 0 and dsim > 0:
                scored.append((i, ssim, dsim))
        if scored:
            scored.sort(key=lambda x: (x[1] + x[2]) / 2, reverse=True)
            best = scored[0]
            evidence = [{"erp_record_id": str(erp.loc[i, "_record_id"]), "supplier_similarity": round(s, 4),
                         "description_similarity": round(d, 4), "amount_cents_equal": True,
                         "posting_invoice_month_equal": True} for i, s, d in scored[:5]]
            tool_rows[j].update(match_status="AMBIGUOUS", matching_method="fuzzy_candidate_review",
                                confidence=None,
                                matching_evidence=str(evidence))
            tool_rows[j]["matched_record_ids"] = [str(erp.loc[item[0], "_record_id"]) for item in scored]
            for i, _, _ in scored:
                if erp_rows[i]["match_status"] == "ERP_ONLY":
                    erp_rows[i].update(match_status="AMBIGUOUS", matching_method="fuzzy_candidate_review",
                                       confidence=0.0, matching_evidence=f"Review candidate from {row['_record_id']}; see TOOL candidate evidence.")
                    erp_rows[i]["matched_record_ids"] = [str(tool.loc[j, "_record_id"])]

    for j, matched_id in duplicate_tools.items():
        tool_rows[j].update(match_status="DUPLICATE_TOOL", matching_method="identical_normalized_invoice_line",
                            matched_record_id=f"TOOL:{matched_id}", confidence=1.0,
                            matching_evidence=f"Identical normalized invoice-line signature; first kept export: TOOL:{matched_id}",
                            duplicate_decision="DUPLICATE", decision_reason="Repeated export with matching business fields.")
        tool_rows[j]["matched_record_ids"] = [f"TOOL:{matched_id}"]
    result = pd.DataFrame([*erp_rows.values(), *tool_rows.values()])
    return {"records": result, "tool_duplicate_groups": pd.DataFrame(duplicate_evidence),
            "summary": result.groupby(["source", "match_status"], dropna=False).size().rename("records").reset_index()}


def investigate_company_b_duplicates(preprocessed: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Report duplicate patterns and candidate groups without excluding any line."""
    config = preprocessed["config"]
    frame = preprocessed["datasets"]["invoices"].copy(deep=True)
    c = lambda semantic, default: get_column(config, "invoices", semantic, default)
    id_col, invoice, line = c("id", "line_id"), c("invoice_number", "invoice_number"), c("invoice_line", "invoice_line")
    supplier, source = c("supplier_id", "vendor_id"), c("source_system", "source_system")
    report_rows = []
    for name, subset in {
        "repeated_invoice_number": [invoice],
        "repeated_invoice_and_line": [invoice, line],
        "repeated_source_invoice_line": [source, invoice, line],
        "repeated_supplier_source_invoice_line": [supplier, source, invoice, line],
    }.items():
        mask = frame.duplicated(subset, keep=False)
        report_rows.append({"pattern": name, "key_columns": subset, "candidate_records": int(mask.sum()),
                            "extra_records_if_collapsed": int(frame.duplicated(subset, keep="first").sum()),
                            "automatically_excluded": 0})
    business_cols = [c for c in frame.columns if c != id_col and not c.endswith("__raw")]
    exact = frame.duplicated(business_cols, keep=False)
    frame["exact_duplicate_excluding_id"] = exact
    repeated_invoice = frame.duplicated([invoice], keep=False)
    repeated_invoice_line = frame.duplicated([invoice, line], keep=False)
    repeated_source_line = frame.duplicated([source, invoice, line], keep=False)
    frame["repeated_invoice_number_candidate"] = repeated_invoice
    frame["repeated_invoice_line_candidate"] = repeated_invoice_line
    frame["record_id"] = frame[id_col].astype("string")
    frame["matched_record_id"] = pd.NA
    frame["matched_record_ids"] = [[] for _ in range(len(frame))]
    frame["match_status"] = "REVIEWED_NO_DUPLICATE_EVIDENCE"
    frame.loc[repeated_invoice | repeated_invoice_line | exact, "match_status"] = "AMBIGUOUS"
    frame["matching_method"] = "company_b_duplicate_investigation"
    frame.loc[exact, "matching_method"] = "exact_all_business_columns_except_unique_id"
    frame["confidence"] = pd.NA
    frame["matching_evidence"] = "No duplicate signature detected; retained."
    frame.loc[repeated_invoice, "matching_evidence"] = "Invoice number repeats; multiple invoice lines may be legitimate. Retained for review."
    frame.loc[repeated_source_line, "matching_evidence"] = "Source, invoice number and line number repeat; inspect fields before deciding. Retained."
    frame.loc[exact, "matching_evidence"] = "All business fields match after excluding unique line_id; retained as review candidate."
    frame["duplicate_decision"] = "KEEP"
    frame.loc[repeated_invoice | repeated_invoice_line | exact, "duplicate_decision"] = "REVIEW_ONLY"
    frame["decision_reason"] = "Company B has a single invoice-line source; no row is automatically removed."
    candidate_records = frame.loc[repeated_invoice | repeated_invoice_line | exact].copy()
    report_rows.append({"pattern": "exact_duplicate_excluding_line_id", "key_columns": business_cols,
                        "candidate_records": int(exact.sum()), "extra_records_if_collapsed": int(frame.duplicated(business_cols, keep="first").sum()),
                        "automatically_excluded": 0})
    # Per-system repeated invoice-line candidates help expose ERP-specific patterns.
    by_source = pd.DataFrame([
        {source: system, "rows": len(group),
         "repeated_invoice_line_records": int(group.duplicated([invoice, line], keep=False).sum()),
         "exact_duplicate_records": int(group[business_cols].duplicated(keep=False).sum())}
        for system, group in frame.groupby(source, dropna=False)
    ])
    return {"summary": pd.DataFrame(report_rows), "candidate_records": candidate_records,
            "by_source_system": by_source, "records": frame}
