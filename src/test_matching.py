"""Controlled and real-data checks for transaction matching safeguards."""

import unittest
from pathlib import Path

import pandas as pd

from src.matching import investigate_company_b_duplicates, match_company_a_transactions
from src.matching.matching_utils import normalize_po_reference, po_reference_parts
from src.preprocessing import preprocess_company_data


ROOT = Path(__file__).resolve().parent.parent


def tiny_a(erp_rows, tool_rows):
    config = {"datasets": {
        "erp": {"columns": {"id": "line_id", "po_number": "po_number", "supplier_name": "supplier_name",
                             "posting_date": "posting_date", "material_description": "description"}},
        "invoice_tool": {"columns": {"id": "tool_record_id", "po_reference": "po_reference", "invoice_number": "invoice_number",
                                      "supplier_name": "supplier_name", "invoice_date": "invoice_date", "description": "description"}}
    }}
    return {"config": config, "datasets": {"erp": pd.DataFrame(erp_rows), "invoice_tool": pd.DataFrame(tool_rows)}}


class CompanyAMatchingTests(unittest.TestCase):
    def test_po_prefix_and_item_suffix_normalization(self):
        self.assertEqual(normalize_po_reference("PO# 4510085877"), "4510085877")
        self.assertEqual(normalize_po_reference("4510085877/10.0"), "4510085877")
        self.assertEqual(po_reference_parts("4510085877/10.0"), ("4510085877", "10"))

    def test_possible_one_to_many_split_is_review_only(self):
        erp = [
            {"line_id": "E1", "po_number": "PO-1", "supplier_name": "Acme", "posting_date_parsed": pd.Timestamp("2024-01-10"), "description": "bolt", "net_amount": 60.00},
            {"line_id": "E2", "po_number": "PO-1", "supplier_name": "Acme", "posting_date_parsed": pd.Timestamp("2024-01-11"), "description": "bolt", "net_amount": 40.00},
        ]
        tool = [{"tool_record_id": "T1", "po_reference": "PO1", "invoice_number": "INV1", "supplier_name": "Acme",
                 "invoice_date_parsed": pd.Timestamp("2024-01-12"), "description": "bolts", "net_amount": 100.00}]
        records = match_company_a_transactions(tiny_a(erp, tool))["records"].set_index("record_id")
        self.assertEqual(records.loc["TOOL:T1", "match_status"], "AMBIGUOUS")
        self.assertEqual(records.loc["TOOL:T1", "matching_method"], "split_receipt_sum_review")
        self.assertEqual(records.loc["TOOL:T1", "duplicate_decision"], "KEEP")

    def test_exact_normalized_composite_fuzzy_and_duplicate_outcomes(self):
        erp = [
            {"line_id": "E1", "po_number": "PO-100", "supplier_name": "Acme", "posting_date_parsed": pd.Timestamp("2024-01-10"), "description": "steel bolt", "net_amount": 100.00},
            {"line_id": "E2", "po_number": "PO-200", "supplier_name": "North Ltd", "posting_date_parsed": pd.Timestamp("2024-02-12"), "description": "copper wire", "net_amount": 200.00},
            {"line_id": "E3", "po_number": None, "supplier_name": "Gamma Co", "posting_date_parsed": pd.Timestamp("2024-03-05"), "description": "pump seal", "net_amount": 300.00},
            {"line_id": "E4", "po_number": None, "supplier_name": "Delta Supply", "posting_date_parsed": pd.Timestamp("2024-04-05"), "description": "hydraulic hose", "net_amount": 400.00},
            {"line_id": "E5", "po_number": "PO-500", "supplier_name": "Twin", "posting_date_parsed": pd.Timestamp("2024-05-01"), "description": "filter", "net_amount": 500.00},
            {"line_id": "E6", "po_number": "PO-500", "supplier_name": "Twin", "posting_date_parsed": pd.Timestamp("2024-05-02"), "description": "filter", "net_amount": 500.00},
        ]
        tool = [
            {"tool_record_id": "T1", "po_reference": "PO-100", "invoice_number": "INV1", "supplier_name": "Acme", "invoice_date_parsed": pd.Timestamp("2024-01-10"), "description": "steel bolt", "net_amount": 100.00},
            {"tool_record_id": "T2", "po_reference": "PO 200", "invoice_number": "INV2", "supplier_name": "North Ltd", "invoice_date_parsed": pd.Timestamp("2024-02-12"), "description": "copper wire", "net_amount": 200.00},
            {"tool_record_id": "T3", "po_reference": None, "invoice_number": "INV3", "supplier_name": "Gamma Co", "invoice_date_parsed": pd.Timestamp("2024-03-25"), "description": "pump seal", "net_amount": 300.00},
            {"tool_record_id": "T4", "po_reference": None, "invoice_number": "INV4", "supplier_name": "Delta Supplies", "invoice_date_parsed": pd.Timestamp("2024-04-25"), "description": "hydraulic hose", "net_amount": 400.00},
            {"tool_record_id": "T5", "po_reference": "PO500/10.0", "invoice_number": "INV5", "supplier_name": "Twin", "invoice_date_parsed": pd.Timestamp("2024-05-01"), "description": "filter", "net_amount": 500.00},
            {"tool_record_id": "T6", "po_reference": "PO500/10.0", "invoice_number": "INV5", "supplier_name": "Twin", "invoice_date_parsed": pd.Timestamp("2024-05-01"), "description": "filter", "net_amount": 500.00},
            {"tool_record_id": "T7", "po_reference": "PO500/20.0", "invoice_number": "INV5", "supplier_name": "Twin", "invoice_date_parsed": pd.Timestamp("2024-05-01"), "description": "filter", "net_amount": 500.00},
        ]
        result = match_company_a_transactions(tiny_a(erp, tool))
        records = result["records"].set_index("record_id")
        self.assertEqual(records.loc["ERP:E1", "matching_method"], "exact_po_supplier_amount")
        self.assertEqual(records.loc["ERP:E2", "matching_method"], "normalized_po_supplier_amount")
        self.assertEqual(records.loc["ERP:E3", "matching_method"], "composite_supplier_amount_month")
        self.assertEqual(records.loc["TOOL:T4", "match_status"], "AMBIGUOUS")
        self.assertEqual(records.loc["TOOL:T5", "match_status"], "AMBIGUOUS")
        self.assertEqual(records.loc["TOOL:T6", "match_status"], "DUPLICATE_TOOL")
        self.assertNotEqual(records.loc["TOOL:T7", "match_status"], "DUPLICATE_TOOL")
        self.assertEqual(len(records), len(erp) + len(tool))
        self.assertEqual(records.index.nunique(), len(records))
        self.assertEqual(records.loc["TOOL:T1", "duplicate_decision"], "DUPLICATE")

    def test_real_company_a_complete_coverage_and_label_independence(self):
        data = preprocess_company_data("configs/company_A.yaml", project_root=ROOT)
        original_counts = {k: len(v) for k, v in data["datasets"].items()}
        result = match_company_a_transactions(data)
        records = result["records"]
        self.assertEqual(len(records), original_counts["erp"] + original_counts["invoice_tool"])
        self.assertTrue(records["record_id"].is_unique)
        self.assertEqual(set(records["match_status"]), {"MATCHED", "ERP_ONLY", "TOOL_ONLY", "AMBIGUOUS", "DUPLICATE_TOOL"})
        self.assertEqual(len(data["datasets"]["erp"]), original_counts["erp"])
        self.assertEqual(len(data["datasets"]["invoice_tool"]), original_counts["invoice_tool"])


class CompanyBDuplicateTests(unittest.TestCase):
    def test_repeated_invoice_number_is_reported_but_not_removed(self):
        data = preprocess_company_data("configs/company_B.yaml", project_root=ROOT)
        before = data["datasets"]["invoices"].copy(deep=True)
        result = investigate_company_b_duplicates(data)
        summary = result["summary"].set_index("pattern")
        self.assertGreater(summary.loc["repeated_invoice_number", "candidate_records"], 0)
        self.assertGreater(summary.loc["repeated_invoice_and_line", "candidate_records"], 0)
        self.assertEqual(int(summary["automatically_excluded"].sum()), 0)
        self.assertEqual(len(result["records"]), len(before))
        pd.testing.assert_frame_equal(data["datasets"]["invoices"], before)


if __name__ == "__main__":
    unittest.main()
