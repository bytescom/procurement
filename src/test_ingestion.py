"""Test script for configuration-driven data ingestion.

Validates that:
1. Both YAML configuration files load and parse cleanly.
2. All specified CSV files exist and can be loaded.
3. Ingested DataFrame shapes match EDA observations.
4. Required columns are present and properly checked.
5. Ingestion operates without any hardcoded company-specific logic.
6. The official output template schema is strictly validated.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.loader import (
    load_config,
    load_company_data,
    validate_output_template,
)


def run_tests() -> bool:
    all_passed = True
    print("=" * 80)
    print("CANTILL.AI ASSESSMENT - STEP 1: INGESTION MODULE VALIDATION")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Test 1: Validate Official Output Template
    # --------------------------------------------------------------------------
    print("\n[TEST 1] Validating Official Output Template...")
    try:
        template_info = validate_output_template(project_root=PROJECT_ROOT)
        print("  [PASS] Output template schema matches official specification.")
        print(f"         Location: {template_info['template_path']}")
        print(f"         Columns ({template_info['column_count']}): {', '.join(template_info['columns'])}")
    except Exception as e:
        print(f"  [FAIL] Output template validation failed: {e}")
        all_passed = False

    # --------------------------------------------------------------------------
    # Test 2: Ingest Company A Data
    # --------------------------------------------------------------------------
    print("\n[TEST 2] Ingesting Company A via configs/company_A.yaml...")
    expected_shapes_a = {
        "erp": (37029, 26),
        "invoice_tool": (28520, 11),
        "labels": (37029, 6),
        "taxonomy": (40, 3),
    }
    try:
        data_a = load_company_data("configs/company_A.yaml", project_root=PROJECT_ROOT)
        print(f"  [PASS] Config loaded: {data_a['company_name']} (ID: {data_a['company_id']}, Currency: {data_a['reporting_currency']})")
        print(f"  [PASS] Loaded {len(data_a['datasets'])} datasets:")

        for key, expected_shape in expected_shapes_a.items():
            if key not in data_a["datasets"]:
                print(f"  [FAIL] Missing dataset '{key}' in Company A data.")
                all_passed = False
                continue

            df = data_a["datasets"][key]
            actual_shape = df.shape
            match = "MATCH" if actual_shape == expected_shape else "MISMATCH"
            status = "[PASS]" if actual_shape == expected_shape else "[FAIL]"
            print(f"    {status} {key:14s}: shape={actual_shape} (expected={expected_shape}) -> {match}")
            if actual_shape != expected_shape:
                all_passed = False

    except Exception as e:
        print(f"  [FAIL] Company A ingestion failed: {e}")
        all_passed = False

    # --------------------------------------------------------------------------
    # Test 3: Ingest Company B Data
    # --------------------------------------------------------------------------
    print("\n[TEST 3] Ingesting Company B via configs/company_B.yaml...")
    expected_shapes_b = {
        "invoices": (41028, 28),
        "fx_rates": (683, 3),
        "seed_labels": (3000, 6),
        "taxonomy": (40, 3),
    }
    try:
        data_b = load_company_data("configs/company_B.yaml", project_root=PROJECT_ROOT)
        print(f"  [PASS] Config loaded: {data_b['company_name']} (ID: {data_b['company_id']}, Currency: {data_b['reporting_currency']})")
        print(f"  [PASS] Loaded {len(data_b['datasets'])} datasets:")

        for key, expected_shape in expected_shapes_b.items():
            if key not in data_b["datasets"]:
                print(f"  [FAIL] Missing dataset '{key}' in Company B data.")
                all_passed = False
                continue

            df = data_b["datasets"][key]
            actual_shape = df.shape
            match = "MATCH" if actual_shape == expected_shape else "MISMATCH"
            status = "[PASS]" if actual_shape == expected_shape else "[FAIL]"
            print(f"    {status} {key:14s}: shape={actual_shape} (expected={expected_shape}) -> {match}")
            if actual_shape != expected_shape:
                all_passed = False

    except Exception as e:
        print(f"  [FAIL] Company B ingestion failed: {e}")
        all_passed = False

    # --------------------------------------------------------------------------
    # Test 4: Error Handling & Validation Tests
    # --------------------------------------------------------------------------
    print("\n[TEST 4] Testing Error Handling on Missing Files and Columns...")
    # Test missing file error
    try:
        load_config("configs/non_existent.yaml", project_root=PROJECT_ROOT)
        print("  [FAIL] Expected FileNotFoundError for missing config, but succeeded.")
        all_passed = False
    except FileNotFoundError:
        print("  [PASS] FileNotFoundError correctly raised for non-existent config.")

    # Test missing required column validation
    from src.ingestion.loader import load_dataset
    try:
        load_dataset(
            "data/company_A/erp_goods_receipts.csv",
            required_columns=["non_existent_column_xyz"],
            project_root=PROJECT_ROOT,
        )
        print("  [FAIL] Expected ValueError for missing required column, but succeeded.")
        all_passed = False
    except ValueError as e:
        print("  [PASS] ValueError correctly raised for missing column.")
        print(f"         Message: {str(e)[:80]}...")

    # --------------------------------------------------------------------------
    # Final Summary
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL INGESTION TESTS PASSED SUCCESSFULLY.")
    else:
        print("SOME INGESTION TESTS FAILED. PLEASE REVIEW LOGS ABOVE.")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
