"""Conservative supplier/item keys with reviewable record-level maps."""

import hashlib
import re

import pandas as pd

from src.matching.matching_utils import normalize_text, normalize_reference, normalize_po_reference


def _key(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:14]
    return f"{prefix}_{digest}"


def _present(value) -> bool:
    return not pd.isna(value) and bool(str(value).strip())


def _record_id(company_id: str, dataset: str, raw_id) -> str:
    if company_id == "A" and dataset == "erp":
        return f"ERP:{raw_id}"
    if company_id == "A" and dataset == "invoice_tool":
        return f"TOOL:{raw_id}"
    return str(raw_id)


def resolve_supplier_and_item_keys(preprocessed: dict, transaction_datasets: list[str]) -> dict[str, pd.DataFrame]:
    cfg, cid = preprocessed["config"], str(preprocessed.get("company_id", "C"))
    source_records = []
    specs = cfg["datasets"]
    def col(ds, semantic):
        return specs[ds].get("columns", {}).get(semantic)
    for ds in transaction_datasets:
        frame = preprocessed["datasets"][ds]
        spec = specs[ds]
        id_col = col(ds, "id") or spec.get("id_column")
        supplier_id_col = col(ds, "supplier_id") or col(ds, "vendor_id")
        supplier_name_col = col(ds, "supplier_name")
        item_id_col = col(ds, "item_id") or col(ds, "material_number")
        description_cols = [col(ds, name) for name in ("material_description", "service_description", "description")]
        description_cols = [c for c in description_cols if c]
        uom_col = col(ds, "uom")
        for _, row in frame.iterrows():
            raw_id = row.get(id_col)
            rid = _record_id(cid, ds, raw_id)
            sid = row.get(supplier_id_col) if supplier_id_col else None
            sname = row.get(supplier_name_col) if supplier_name_col else None
            description = next((row.get(c) for c in description_cols if _present(row.get(c))), None)
            iid = row.get(item_id_col) if item_id_col else None
            uom = row.get(uom_col) if uom_col else None
            source_records.append({"dataset": ds, "record_id": rid, "raw_supplier_id": sid,
                                   "raw_supplier_name": sname, "supplier_norm": normalize_text(sname),
                                   "raw_item_id": iid, "raw_description": description,
                                   "description_norm": normalize_text(description), "uom": uom,
                                   "supplier_id_present": _present(sid), "item_id_present": _present(iid)})
    records = pd.DataFrame(source_records)
    # Exact name aliases across A's two systems are linked only when the ERP
    # name identifies one vendor ID. No fuzzy legal-entity merges are attempted.
    name_to_ids = {}
    if "erp" in transaction_datasets:
        erp_rows = records[records.dataset.eq("erp")]
        for name, group in erp_rows[erp_rows.supplier_norm.notna()].groupby("supplier_norm"):
            ids = {str(x).strip() for x in group.raw_supplier_id if _present(x)}
            if len(ids) == 1: name_to_ids[name] = next(iter(ids))
    supplier_keys, supplier_methods, supplier_confidence = [], [], []
    for r in records.itertuples(index=False):
        if r.supplier_id_present:
            sid = normalize_reference(r.raw_supplier_id) or str(r.raw_supplier_id).strip()
            supplier_keys.append(_key(f"SUP{cid}", sid)); supplier_methods.append("source_supplier_id"); supplier_confidence.append(1.0)
        elif r.dataset == "invoice_tool" and r.supplier_norm in name_to_ids:
            supplier_keys.append(_key(f"SUP{cid}", name_to_ids[r.supplier_norm])); supplier_methods.append("exact_name_to_unique_erp_vendor"); supplier_confidence.append(0.95)
        elif r.supplier_norm:
            supplier_keys.append(_key(f"SUP{cid}", r.supplier_norm)); supplier_methods.append("normalized_exact_name"); supplier_confidence.append(0.75)
        else:
            supplier_keys.append(""); supplier_methods.append("unresolved_missing_supplier"); supplier_confidence.append(0.0)
    records["supplier_key"] = supplier_keys
    records["supplier_match_method"] = supplier_methods
    records["supplier_confidence"] = supplier_confidence

    # A material ID is a stable item key. Without one, only exact normalized
    # description + supplier + unit forms a shared key; uncertain fuzzy links
    # are deliberately left separate for human review.
    desc_to_keys = {}
    item_keys, item_methods, item_confidence = [], [], []
    for r in records.itertuples(index=False):
        if r.item_id_present:
            key = _key(f"ITEM{cid}", normalize_reference(r.raw_item_id) or str(r.raw_item_id).strip())
            if r.description_norm and r.supplier_key:
                evidence = f"{r.supplier_key}|{r.description_norm}|{normalize_text(r.uom) or ''}"
                desc_to_keys.setdefault(evidence, key)
            method, confidence = "source_material_id", 1.0
        elif r.description_norm and r.supplier_key:
            uom = normalize_text(r.uom) or ""
            evidence = f"{r.supplier_key}|{r.description_norm}|{uom}"
            key = desc_to_keys.setdefault(evidence, _key(f"ITEM{cid}", evidence))
            method, confidence = ("exact_supplier_description", 0.8) if r.dataset == "invoice_tool" else ("normalized_description", 0.75)
        else:
            key, method, confidence = "", "unresolved_missing_item_evidence", 0.0
        item_keys.append(key); item_methods.append(method); item_confidence.append(confidence)
    records["item_key"] = item_keys
    records["item_match_method"] = item_methods
    records["item_confidence"] = item_confidence
    supplier_map = records[["record_id", "dataset", "raw_supplier_id", "raw_supplier_name", "supplier_norm",
                            "supplier_key", "supplier_match_method", "supplier_confidence"]].copy()
    item_map = records[["record_id", "dataset", "raw_item_id", "raw_description", "description_norm", "uom",
                        "item_key", "item_match_method", "item_confidence"]].copy()
    resolutions = records[["record_id", "supplier_key", "item_key", "supplier_match_method", "supplier_confidence",
                           "item_match_method", "item_confidence"]].copy()
    return {"resolutions": resolutions, "supplier_map": supplier_map, "item_map": item_map}
