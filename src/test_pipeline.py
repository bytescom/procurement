"""Step 4–10 integration checks against real data and a synthetic new company."""

import unittest
from pathlib import Path

import pandas as pd
from unittest.mock import patch

from src.classification import classify_records
from src.ingestion.loader import validate_output_template
from src.pipeline import OUTPUT_COLUMNS, build_company_output
from src.preprocessing import preprocess_company_data
from src.ingestion.loader import load_company_data


ROOT = Path(__file__).resolve().parent.parent


class PipelineTests(unittest.TestCase):
    def test_template_is_exact(self):
        info = validate_output_template(project_root=ROOT)
        self.assertEqual(info["columns"], OUTPUT_COLUMNS)

    def test_runtime_classifier_does_not_need_company_labels(self):
        frame = pd.DataFrame({"detail": ["steel coil", "unknown"]})
        paths = [("Raw Materials", "Steel", "Coils"), ("Services", "Freight", "Road")]
        result = classify_records(frame, text_columns=["detail"], taxonomy_paths=paths,
                                  level_names=["l1", "l2", "l3"])
        self.assertEqual(result.loc[0, "l1"], "Raw Materials")
        self.assertEqual(result.loc[1, "classification_status"], "UNCLASSIFIED_NO_LEXICAL_EVIDENCE")

    def test_new_company_works_through_generic_pipeline(self):
        config = {
            "company": {"id": "C", "name": "Synthetic C", "reporting_currency": "USD"},
            "taxonomy": {"levels": ["category_l1", "category_l2", "category_l3", "category_l4"]},
            "transaction_dataset": "purchases",
            "datasets": {
                "purchases": {"id_column": "row_id", "columns": {
                    "id": "row_id", "supplier_name": "supplier", "description": "memo",
                    "amount": "amount"}, "required_columns": []},
                "taxonomy": {"columns": {"l1": "l1", "l2": "l2", "l3": "l3", "l4": "l4"}, "required_columns": []},
            },
        }
        loaded = {"config": config, "company_id": "C", "datasets": {
            "purchases": pd.DataFrame({"row_id": ["r1", "r2"], "supplier": ["Acme", "Beta"],
                                       "memo": ["steel coil", "unknown expense"], "amount": [125, 30]}),
            "taxonomy": pd.DataFrame({"l1": ["Materials"], "l2": ["Metals"],
                                      "l3": ["Steel products"], "l4": ["Coils"]}),
        }}
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
            prepared = preprocess_company_data(loaded)
            output, _ = build_company_output(prepared)
        self.assertEqual(output.record_id.tolist(), ["r1", "r2"])
        self.assertEqual(output.include.tolist(), [1, 1])
        self.assertEqual(output.spend.tolist(), [125.0, 30.0])
        self.assertEqual(output.loc[0, "l1"], "Materials")
        self.assertEqual(output.loc[0, "l4"], "Coils")
        self.assertEqual(list(output.columns), OUTPUT_COLUMNS)

    @patch.dict("os.environ", {"GEMINI_API_KEY": ""})
    def test_real_company_outputs_preserve_rows_and_constraints(self):
        expected = {"A": 37029 + 28520, "B": 41028}
        for company, count in expected.items():
            with self.subTest(company=company):
                loaded = load_company_data(f"configs/company_{company}.yaml", project_root=ROOT)
                snapshots = {k: v.copy(deep=True) for k, v in loaded["datasets"].items()}
                prepared = preprocess_company_data(loaded)
                output, reports = build_company_output(prepared)
                self.assertEqual(len(output), count)
                self.assertEqual(list(output.columns), OUTPUT_COLUMNS)
                self.assertFalse(output.record_id.duplicated().any())
                self.assertTrue(output.include.isin([0, 1]).all())
                self.assertTrue(output.loc[output.include.eq(0), "spend"].fillna(0).eq(0).all())
                self.assertTrue(output.loc[output.include.eq(0), "exclusion_reason"].astype(str).str.len().gt(0).all())
                self.assertTrue((prepared["row_counts"].row_difference == 0).all())
                self.assertIn("preprocessing_audit", reports)
                self.assertIn("amount_conversion_audit", reports)
                self.assertIn("classification_diagnostics", reports)
                for key, original in snapshots.items():
                    pd.testing.assert_frame_equal(loaded["datasets"][key], original)
                    self.assertEqual(len(prepared["datasets"][key]), len(original))
                if company == "A":
                    self.assertIn("label_evaluation", reports)
                    self.assertGreater(int(reports["label_evaluation"].iloc[0].n_labelled), 6000)
                    self.assertLess(int(reports["label_evaluation"].iloc[0].n_labelled), 37029)
                else:
                    eval_table = reports["label_evaluation"]
                    self.assertTrue(eval_table.source.str.contains("held-out test|post-run comparison").all())
                    unresolved_amounts = reports["amount_conversion_audit"].loc[
                        reports["amount_conversion_audit"].normalized_spend.isna()]
                    self.assertEqual(len(unresolved_amounts), 3)
                    self.assertTrue(unresolved_amounts.amount_missing.all())
                    self.assertTrue(unresolved_amounts.conversion_status.eq("converted_fx_table").all())


if __name__ == "__main__":
    unittest.main()
