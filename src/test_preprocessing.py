"""Focused tests for non-destructive preprocessing and financial formulas."""

import unittest
from pathlib import Path

import pandas as pd

from src.ingestion.loader import load_company_data
from src.preprocessing.amounts import (normalize_company_b_amounts,
                                       normalize_erp_amounts,
                                       normalize_fx_rates,
                                       normalize_invoice_tool_amounts)
from src.preprocessing.dates import parse_date_columns
from src.preprocessing.cleaning import clean_dataframe
from src.preprocessing.pipeline import preprocess_company_data


ROOT = Path(__file__).resolve().parent.parent


class AmountNormalizationTests(unittest.TestCase):
    def test_vat_formats_missing_default_and_zero(self):
        source = pd.DataFrame({"gross": [115, 115, 115, 115, 100],
                               "vat": [0.15, "15%", "15", None, "0"]})
        result, _ = normalize_invoice_tool_amounts(source, "gross", "vat", "sample")
        self.assertEqual(result["vat_rate_normalized"].tolist(), [0.15, 0.15, 0.15, 0.15, 0.0])
        self.assertEqual(result["net_amount"].round(6).tolist(), [100, 100, 100, 100, 100])
        self.assertTrue(result.loc[3, "vat_rate_defaulted"])
        self.assertEqual(source.columns.tolist(), ["gross", "vat"])

    def test_erp_is_not_taxed_again(self):
        source = pd.DataFrame({"amount": [115, -20]})
        result, _ = normalize_erp_amounts(source, "amount", "erp")
        self.assertEqual(result["net_amount"].tolist(), [115, -20])
        self.assertTrue(result.loc[1, "negative_amount_suspected_reversal"])
        self.assertEqual(source.columns.tolist(), ["amount"])

    def test_invalid_amount_and_vat_are_flagged_not_defaulted(self):
        source = pd.DataFrame({"gross": ["not an amount"], "vat": ["bad rate"]})
        result, _ = normalize_invoice_tool_amounts(source, "gross", "vat", "sample")
        self.assertTrue(result.loc[0, "invalid_gross_amount"])
        self.assertTrue(result.loc[0, "invalid_vat_rate"])
        self.assertTrue(pd.isna(result.loc[0, "net_amount"]))

    def test_invalid_line_rate_is_not_silently_replaced(self):
        fx, _ = normalize_fx_rates(pd.DataFrame({
            "currency": ["EUR"], "month": ["2024-01"], "rate": [1.2]
        }), "currency", "month", "rate", "fx")
        invoice = pd.DataFrame({"amount": [120], "currency": ["EUR"], "rate": [0],
            "invoice_date_parsed": pd.to_datetime(["2024-01-01"]),
            "posting_date_parsed": pd.to_datetime(["2024-01-01"])})
        result, _ = normalize_company_b_amounts(invoice, fx, amount_column="amount",
            currency_column="currency", rate_column="rate", dataset_name="invoices")
        self.assertEqual(result.loc[0, "conversion_status"], "invalid_invoice_rate")
        self.assertTrue(result.loc[0, "unresolved_conversion"])


class GeneralCleaningTests(unittest.TestCase):
    def test_text_cleanup_keeps_raw_values_and_protects_identifiers(self):
        source = pd.DataFrame({"supplier": ["  Acme Ltd  ", "  "], "record_id": [" 001 ", "002"]})
        result, _ = clean_dataframe(source, "sample", protected_columns={"record_id"})
        self.assertEqual(result["supplier"].iloc[0], "Acme Ltd")
        self.assertTrue(pd.isna(result["supplier"].iloc[1]))
        self.assertEqual(result["supplier__raw"].iloc[0], "  Acme Ltd  ")
        self.assertEqual(result["record_id"].iloc[0], " 001 ")
        pd.testing.assert_frame_equal(source, pd.DataFrame({"supplier": ["  Acme Ltd  ", "  "], "record_id": [" 001 ", "002"]}))

    def test_usd_line_fx_line_rate_missing_rate_and_invalid_fx(self):
        fx, _ = normalize_fx_rates(pd.DataFrame({
            "currency": ["EUR", "GBP"], "month": ["2024-01", "2024-01"], "rate": [1.25, 0]
        }), "currency", "month", "rate", "fx")
        invoices = pd.DataFrame({
            "amount": [100, 125, 125, 125], "currency": ["USD", "EUR", "EUR", "GBP"],
            "rate": [None, 1.25, None, None],
            "invoice_date_parsed": pd.to_datetime(["2024-01-15"] * 4),
            "posting_date_parsed": pd.to_datetime(["2024-01-15"] * 4),
        })
        result, _ = normalize_company_b_amounts(invoices, fx, amount_column="amount",
             currency_column="currency", rate_column="rate", dataset_name="invoices")
        self.assertEqual(result["usd_amount"].iloc[:3].tolist(), [100, 100, 100])
        self.assertEqual(result.loc[0, "conversion_status"], "usd_no_conversion")
        self.assertEqual(result.loc[1, "conversion_status"], "converted_line_rate")
        self.assertEqual(result.loc[2, "conversion_status"], "converted_fx_table")
        self.assertTrue(result.loc[3, "invalid_fx_match"])
        self.assertTrue(result.loc[3, "unresolved_conversion"])
        self.assertEqual(len(result), len(invoices))

    def test_duplicate_fx_keys_are_flagged_and_not_selected(self):
        fx, _ = normalize_fx_rates(pd.DataFrame({
            "currency": ["EUR", "EUR"], "month": ["2024-01", "2024-01"], "rate": [1.2, 1.2]
        }), "currency", "month", "rate", "fx")
        self.assertTrue(fx["fx_duplicate_key"].all())
        invoice = pd.DataFrame({"amount": [120], "currency": ["EUR"], "rate": [None],
            "invoice_date_parsed": pd.to_datetime(["2024-01-01"]),
            "posting_date_parsed": pd.to_datetime(["2024-01-01"])})
        result, _ = normalize_company_b_amounts(invoice, fx, amount_column="amount",
            currency_column="currency", rate_column="rate", dataset_name="invoices")
        self.assertTrue(result.loc[0, "duplicate_fx_match"])
        self.assertTrue(result.loc[0, "unresolved_conversion"])


class DateAndPipelineTests(unittest.TestCase):
    def test_dates_distinguish_missing_and_invalid(self):
        frame, _ = parse_date_columns(pd.DataFrame({"date": ["2024-01-02", None, "not a date"]}), ["date"], "x")
        self.assertEqual(frame["date_parsed"].notna().sum(), 1)
        self.assertEqual(int(frame["date_missing"].sum()), 1)
        self.assertEqual(int(frame["date_invalid"].sum()), 1)

    def test_both_company_pipelines_keep_every_dataset_and_source_frame(self):
        for company in ("A", "B"):
            with self.subTest(company=company):
                loaded = load_company_data(f"configs/company_{company}.yaml", project_root=ROOT)
                snapshots = {key: df.copy(deep=True) for key, df in loaded["datasets"].items()}
                result = preprocess_company_data(loaded)
                self.assertEqual(set(result["datasets"]), set(snapshots))
                for key, original in snapshots.items():
                    self.assertEqual(len(result["datasets"][key]), len(original))
                    pd.testing.assert_frame_equal(loaded["datasets"][key], original)
                self.assertTrue((result["row_counts"]["row_difference"] == 0).all())
        a = preprocess_company_data(load_company_data("configs/company_A.yaml", project_root=ROOT))
        b = preprocess_company_data(load_company_data("configs/company_B.yaml", project_root=ROOT))
        self.assertAlmostEqual(a["datasets"]["erp"]["net_amount"].iloc[0],
                               a["datasets"]["erp"]["value_zar_numeric"].iloc[0])
        self.assertEqual(b["datasets"]["invoices"].loc[0, "usd_amount"],
                         b["datasets"]["invoices"].loc[0, "line_amount_numeric"])


if __name__ == "__main__":
    unittest.main()
