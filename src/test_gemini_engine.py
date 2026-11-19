"""Offline unit tests for Gemini config, routing, validation, retries, and caching."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ai.gemini_client import GeminiClient, GeminiUnavailable, MODEL_ID
from src.ai.gemini_classifier import GeminiClassifier
from src.ai.response_parser import parse_response
from src.decision_engine.decision_router import route_decision
from src.decision_engine.decision_validator import validate_taxonomy_response
from src.classification_engine import classify_with_decision_engine
from src.local_classifier import _split_indices
import pandas as pd


VALID = {"status": "classified", "transaction_type": "good", "l1": "Plants", "l2": "Food", "l3": "Grain", "l4": "Wheat",
         "confidence": 0.91, "reason": "Grain purchase."}
PATHS = [("Plants", "Food", "Grain", "Wheat"), ("Services", "Freight", "Road", "Linehaul")]
LEVELS = ["l1", "l2", "l3", "l4"]


class _Interactions:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _SdkClient:
    def __init__(self, outcomes):
        self.interactions = _Interactions(outcomes)


class _RateLimited(RuntimeError):
    code = 429


class GeminiEngineTests(unittest.TestCase):
    def test_grouped_holdout_has_no_identical_text_leakage(self):
        targets = pd.Series([f"class_{i % 3}" for i in range(60)])
        groups = pd.Series([f"same_text_{i // 3}" for i in range(60)])
        train, validation, test = _split_indices(targets, groups=groups)
        train_groups, validation_groups, test_groups = (set(groups.iloc[idx]) for idx in (train, validation, test))
        self.assertFalse(train_groups & validation_groups)
        self.assertFalse(train_groups & test_groups)
        self.assertFalse(validation_groups & test_groups)

    def test_missing_key_disables_provider_without_sdk_import(self):
        client = GeminiClient(api_key="")
        with self.assertRaisesRegex(GeminiUnavailable, "GEMINI_API_KEY"):
            client.classify("no private data")

    def test_model_and_structured_low_thinking_request_configuration(self):
        sdk = _SdkClient([{"output_text": json.dumps(VALID), "usage": {
            "total_input_tokens": 10, "total_output_tokens": 5, "total_tokens": 15}}])
        client = GeminiClient(api_key="test-secret", client=sdk, max_retries=0)
        text, usage = client.classify("minimal prompt")
        call = sdk.interactions.calls[0]
        self.assertEqual(MODEL_ID, "gemini-3.8-flash")
        self.assertEqual(call["model"], MODEL_ID)
        self.assertEqual(call["generation_config"], {"thinking_level": "low"})
        self.assertEqual(call["response_format"]["mime_type"], "application/json")
        self.assertEqual(call["timeout"], 30)
        self.assertFalse(call["store"])
        self.assertEqual(usage["total_tokens"], 15)
        self.assertEqual(parse_response(text)["l4"], "Wheat")

    def test_retry_transient_timeout_then_success(self):
        sdk = _SdkClient([TimeoutError("timeout"), {"output_text": json.dumps(VALID), "usage": {}}])
        client = GeminiClient(api_key="test-secret", client=sdk, max_retries=1)
        with patch("src.ai.gemini_client.time.sleep"):
            result, _ = client.classify("prompt")
        self.assertEqual(len(sdk.interactions.calls), 2)
        self.assertEqual(parse_response(result)["status"], "classified")

    def test_rate_limit_retries_are_bounded(self):
        sdk = _SdkClient([_RateLimited("quota"), _RateLimited("quota")])
        client = GeminiClient(api_key="test-secret", client=sdk, max_retries=1)
        with patch("src.ai.gemini_client.time.sleep"):
            with self.assertRaisesRegex(GeminiUnavailable, "_RateLimited"):
                client.classify("prompt")
        self.assertEqual(len(sdk.interactions.calls), 2)

    def test_parse_and_taxonomy_validation_reject_bad_or_invalid_response(self):
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            parse_response("not json")
        invalid = {**VALID, "l2": "Wrong parent"}
        self.assertFalse(validate_taxonomy_response(invalid, PATHS, LEVELS)["valid"])
        no_l4 = {**VALID, "l4": "Invented"}
        self.assertFalse(validate_taxonomy_response(no_l4, PATHS, LEVELS[:3])["valid"])
        uncertain = {"status": "uncertain", "transaction_type": "unclear", "l1": None, "l2": None, "l3": None, "l4": None,
                     "confidence": 0.1, "reason": "Unclear"}
        self.assertTrue(validate_taxonomy_response(uncertain, PATHS, LEVELS)["valid"])

    def test_router_accepts_valid_gemini_rejects_invalid_and_falls_back(self):
        accepted = route_decision(local_path=(), local_confidence=0, taxonomy_paths=PATHS,
                                  level_names=LEVELS, threshold=.6, gemini_threshold=.8, gemini=VALID)
        self.assertEqual(accepted["decision_outcome"], "ACCEPT_GEMINI")
        rejected = route_decision(local_path=PATHS[1], local_confidence=.4, taxonomy_paths=PATHS,
                                  level_names=LEVELS, threshold=.6, gemini={**VALID, "l2": "bad"})
        self.assertEqual(rejected["decision_outcome"], "REJECT_INVALID")
        fallback = route_decision(local_path=PATHS[0], local_confidence=.4, taxonomy_paths=PATHS,
                                  level_names=LEVELS, threshold=.6, gemini_error="timeout")
        self.assertEqual(fallback["decision_outcome"], "FALLBACK_LOCAL")

    def test_cache_reuses_identical_taxonomy_aware_requests(self):
        sdk = _SdkClient([{"output_text": json.dumps(VALID), "usage": {"total_input_tokens": 4}}])
        provider = GeminiClient(api_key="test-secret", client=sdk, max_retries=0)
        with tempfile.TemporaryDirectory() as directory:
            classifier = GeminiClassifier(cache_path=Path(directory) / "cache.sqlite3", client=provider)
            first, meta1 = classifier.classify({"memo": "wheat seed"}, PATHS, LEVELS)
            second, meta2 = classifier.classify({"memo": "wheat seed"}, PATHS, LEVELS)
            other_tree = [("Other", "Food", "Grain", "Wheat")]
            self.assertNotEqual(classifier.cache_key({"memo": "wheat seed"}, PATHS, LEVELS),
                                classifier.cache_key({"memo": "wheat seed"}, other_tree, LEVELS))
        self.assertEqual(first, second)
        self.assertFalse(meta1["cache_hit"])
        self.assertTrue(meta2["cache_hit"])
        self.assertEqual(len(sdk.interactions.calls), 1)

    def test_dry_run_deduplicates_requests_and_never_calls_api(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = pd.DataFrame({"record_id": ["x1", "x2"], "dataset": ["tx", "tx"],
                                  "include": [1, 1], "memo": ["unmapped item", "unmapped item"]})
            data = {"company_id": "C", "config": {"company": {"id": "C"}, "datasets": {}},
                    "datasets": {"tx": pd.DataFrame({"row_id": ["x1", "x2"]})}}
            _, reports = classify_with_decision_engine(
                frame, text_columns=["memo"], taxonomy_paths=PATHS, level_names=LEVELS,
                data=data, config={"gemini": {"enabled": True, "max_requests": 1,
                    "cache_path": str(Path(directory) / "cache.sqlite3")}}, dry_run=True)
        run = reports["gemini_run"].iloc[0]
        self.assertEqual(run.status, "dry_run")
        self.assertEqual(int(run.api_requests), 0)
        self.assertEqual(int(run.unique_candidate_requests_estimate), 1)


if __name__ == "__main__":
    unittest.main()
