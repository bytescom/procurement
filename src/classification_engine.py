"""Local-first classification, optional semantic ranking, and Gemini fallback."""

from __future__ import annotations

import os
import json
import re
import time

import numpy as np
import pandas as pd

from src.ai.gemini_classifier import GeminiClassifier
from src.ai.prompts import make_prompt
from src.classification import classify_records
from src.decision_engine.decision_logger import decision_log
from src.decision_engine.decision_router import route_decision
from src.local_classifier import train_local_classifier
from src.local_classifier import _metrics as score_paths
from src.semantic_classifier import predict_semantic


def _label_frame(data, transaction_frames, taxonomy_paths, level_names):
    cfg, cid = data["config"], str(data["company_id"])
    label_key = next((key for key in cfg["datasets"] if key in {"labels", "seed_labels"}), None)
    if label_key is None:
        return None
    spec, labels = cfg["datasets"][label_key], data["datasets"][label_key]
    mapping = spec.get("columns", {})
    id_col = mapping.get("record_id") or mapping.get("id") or spec.get("id_column")
    level_columns = [mapping.get(name) for name in level_names]
    if not id_col or id_col not in labels or any(col not in labels for col in level_columns if col):
        return None
    by_id = {}
    for dataset, frame in transaction_frames.items():
        tx_spec = cfg["datasets"][dataset]
        source_id = tx_spec.get("columns", {}).get("id", tx_spec.get("id_column"))
        for raw_id in frame[source_id].astype(str):
            rid = f"ERP:{raw_id}" if cid == "A" and dataset == "erp" else (
                f"TOOL:{raw_id}" if cid == "A" and dataset == "invoice_tool" else str(raw_id))
            by_id[raw_id] = rid
            by_id[rid] = rid
    rows = []
    include_col = mapping.get("include")
    current_record_ids = set(by_id.values())
    for _, row in labels.iterrows():
        if include_col and include_col in labels:
            try:
                if int(row[include_col]) != 1:
                    continue
            except (TypeError, ValueError):
                continue
        raw_id = str(row[id_col])
        rid = by_id.get(raw_id, raw_id)
        # Seed labels belong to the source files used during development. On a
        # blind or refreshed dataset, unmatched historical IDs must not be
        # passed to the trainer as examples with empty transaction text.
        if rid not in current_record_ids:
            continue
        path = tuple(str(row[col]).strip() for col in level_columns
                     if col and pd.notna(row[col]) and str(row[col]).strip())
        if path and path in taxonomy_paths:
            rows.append({"record_id": rid, "path": path})
    return pd.DataFrame(rows) if rows else None


def _candidate_paths(record: dict, local_path: tuple[str, ...], paths: list[tuple[str, ...]], limit=10):
    text = " ".join(str(value or "") for value in record.values()).casefold()
    tokens = set(re.findall(r"[a-z0-9]+", text))
    scored = []
    for path in paths:
        path_tokens = set(re.findall(r"[a-z0-9]+", " ".join(path).casefold()))
        score = len(tokens & path_tokens) / max(1, len(path_tokens))
        scored.append((score, path))
    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = []
    if tuple(local_path) in paths:
        selected.append(tuple(local_path))
    for _, path in scored:
        if path not in selected:
            selected.append(path)
        if len(selected) >= limit:
            break
    return selected or paths[:limit]


def _record_payload(row: pd.Series, text_columns: list[str]) -> dict:
    payload = {}
    for column in text_columns:
        if column not in row.index:
            continue
        value = row[column]
        if pd.isna(value) or not str(value).strip():
            continue
        payload[column] = str(value)[:500]
    return payload


def classify_with_decision_engine(frame: pd.DataFrame, *, text_columns: list[str],
                                  taxonomy_paths: list[tuple[str, ...]], level_names: list[str],
                                  data: dict, config: dict | None = None, dry_run: bool = False):
    """Return classifications and auditable metrics/request estimates.

    Training uses only the label-training partition. Validation chooses the local
    abstention threshold; the held-out test partition is excluded from fitting.
    Gemini never receives labels or record identifiers.
    """
    cfg = config or {}
    pipeline_start = time.perf_counter()
    lexical = classify_records(frame, text_columns=text_columns, taxonomy_paths=taxonomy_paths,
                               level_names=level_names)
    feature_frame = frame.copy()
    for i in range(len(level_names)):
        feature_frame[f"baseline_l{i+1}"] = lexical[f"l{i+1}"].values
    feature_frame["baseline_confidence"] = lexical.confidence.values
    transaction_frames = {name: data["datasets"][name]
                          for name in frame["dataset"].dropna().unique()
                          if name in data["datasets"]}
    labels = _label_frame(data, transaction_frames, taxonomy_paths, level_names)
    model_result = None
    if labels is not None:
        model_result = train_local_classifier(feature_frame, text_columns, labels,
                                              taxonomy_paths, level_names)
    if model_result is not None:
        local = model_result.predictions.copy()
        metrics = model_result.evaluation.copy()
        confusion = model_result.confusion.copy()
        split = model_result.split
        threshold = model_result.threshold
        model_status = "trained_on_train_split"
    else:
        local = lexical.copy()
        local["local_confidence"] = pd.to_numeric(local.confidence, errors="coerce").fillna(0.0)
        metrics = pd.DataFrame()
        confusion = pd.DataFrame()
        split = {"train": set(), "validation": set(), "test": set()}
        threshold = float(cfg.get("local_confidence_threshold", 0.25))
        model_status = "taxonomy_only_no_training_labels"

    semantic_enabled = bool(cfg.get("semantic_enabled", False))
    semantic = None
    if semantic_enabled:
        texts = frame[text_columns].fillna("").astype(str).agg(" ".join, axis=1).tolist()
        semantic = predict_semantic(texts, taxonomy_paths)
    if semantic is not None:
        for idx, result in zip(local.index, semantic):
            local.loc[idx, "semantic_path"] = "\x1f".join(result["path"])
            local.loc[idx, "semantic_confidence"] = result["confidence"]

    gemini_cfg = cfg.get("gemini", {})
    enabled = bool(gemini_cfg.get("enabled", True))
    api_key = os.environ.get("GEMINI_API_KEY")
    request_budget = max(0, int(gemini_cfg.get("max_requests", 50)))
    client = None
    if enabled and (api_key or dry_run):
        client = GeminiClassifier(
            cache_path=gemini_cfg.get("cache_path", "output/cache/gemini.sqlite3"),
            timeout_seconds=gemini_cfg.get("timeout_seconds", 30),
            max_retries=gemini_cfg.get("max_retries", 2),
            price_input_per_million=gemini_cfg.get("price_input_per_million", 0.75),
            price_output_per_million=gemini_cfg.get("price_output_per_million", 3.75))
    gemini_threshold = float(gemini_cfg.get("confidence_threshold", 0.75))
    rows, logs = [], []
    calls = requests_sent = input_tokens = output_tokens = cache_hits = candidate_requests = 0
    estimated_input_tokens = estimated_output_tokens = 0
    elapsed = 0.0
    request_cache = {}
    request_error_cache = {}
    dry_run_keys = set()
    for idx, row in frame.iterrows():
        local_row = local.loc[idx]
        local_path = tuple(str(local_row.get(f"l{i+1}", "")) for i in range(len(level_names)))
        local_path = tuple(value for value in local_path if value)
        local_conf = float(local_row.get("local_confidence", local_row.get("confidence", 0.0)) or 0.0)
        semantic_result = None
        if semantic is not None:
            semantic_result = {"path": semantic[idx]["path"], "confidence": semantic[idx]["confidence"]}
        record = _record_payload(row, text_columns)
        candidates = _candidate_paths(record, local_path, taxonomy_paths)
        gemini_response, gemini_error, cache_hit, attempted = None, None, False, False
        should_fallback = bool(row.get("include", 1)) and (
            not local_path or local_conf < threshold or
            (semantic_result is not None and tuple(semantic_result["path"]) != local_path))
        cache_key = client.cache_key(record, candidates, level_names) if client and should_fallback else None
        if client and should_fallback:
            cached = client.get_cached(cache_key)
            if cached is not None:
                gemini_response, cache_hit = cached, True
                cache_hits += 1
            elif dry_run:
                if cache_key not in dry_run_keys and len(dry_run_keys) < request_budget:
                    dry_run_keys.add(cache_key)
                    estimated_input_tokens += max(1, len(make_prompt(record, candidates, level_names)) // 4)
                    estimated_output_tokens += 120
                gemini_error = "DRY_RUN"
            elif api_key and cache_key in request_cache:
                gemini_response, cache_hit = request_cache[cache_key], True
                cache_hits += 1
            elif api_key and cache_key in request_error_cache:
                gemini_error = request_error_cache[cache_key]
            elif api_key and requests_sent < request_budget:
                attempted = True
                candidate_requests += 1
                requests_sent += 1
                start = time.perf_counter()
                try:
                    gemini_response, usage = client.classify(record, candidates, level_names)
                    request_cache[cache_key] = gemini_response
                    calls += 1
                    input_tokens += usage["usage"]["input_tokens"]
                    output_tokens += usage["usage"]["output_tokens"]
                except Exception as exc:
                    gemini_error = f"{type(exc).__name__}: {exc}"
                    request_error_cache[cache_key] = gemini_error
                elapsed += time.perf_counter() - start
            else:
                gemini_error = "request_budget_exhausted" if api_key else "missing_api_key"
        outcome = route_decision(local_path=local_path, local_confidence=local_conf,
                                 taxonomy_paths=taxonomy_paths, level_names=level_names,
                                 threshold=threshold, gemini_threshold=gemini_threshold,
                                 semantic=semantic_result,
                                 gemini=gemini_response, gemini_error=gemini_error)
        rows.append({"record_id": row.get("record_id"), **outcome,
                     "classification_method": "decision_router", "classification_status": outcome["decision_outcome"],
                     "local_confidence": local_conf, "gemini_confidence": gemini_response.get("confidence") if gemini_response else np.nan,
                     "gemini_transaction_type": gemini_response.get("transaction_type", "") if gemini_response else "",
                     "gemini_reason": gemini_response.get("reason", "") if gemini_response else "",
                     "gemini_cache_hit": cache_hit, "gemini_attempted": attempted,
                     "semantic_confidence": semantic_result.get("confidence") if semantic_result else np.nan})
        logs.append(decision_log(str(row.get("record_id", idx)), outcome,
                                 local_confidence=local_conf, gemini_attempted=attempted,
                                 gemini_cache_hit=cache_hit))
    if dry_run:
        candidate_requests = min(request_budget, len(dry_run_keys))
    decisions_frame = pd.DataFrame(rows, index=frame.index)
    evaluation_status = [
        {"model": "existing_lexical", "status": "executed"},
        {"model": "tfidf_logistic", "status": "executed" if model_result else "not_available_no_labels"},
        {"model": "sentence_transformers", "status": "executed" if semantic is not None else "not_installed_or_disabled"},
        {"model": "local_plus_gemini", "status": "executed" if requests_sent else ("dry_run_only" if dry_run else "not_run_missing_api_key_or_no_fallback")},
    ]
    if labels is not None and split.get("test"):
        heldout = labels[labels.record_id.astype(str).isin(split["test"])].copy()
        if len(heldout):
            prediction_by_id = decisions_frame.set_index("record_id")
            weight_column = next((column for column in ("usd_amount", "net_amount", "spend", "amount")
                                  if column in frame), None)
            heldout_weights = (pd.to_numeric(frame.set_index("record_id").reindex(
                heldout.record_id.astype(str))[weight_column], errors="coerce").abs().fillna(0).tolist()
                if weight_column else None)
            truth = heldout.path.map(lambda path: json.dumps(list(path), ensure_ascii=False, separators=(",", ":"))).tolist()
            pred_paths, confs = [], []
            for rid in heldout.record_id.astype(str):
                record = prediction_by_id.loc[rid]
                path = tuple(str(record.get(f"l{i+1}", "")) for i in range(len(level_names)))
                pred_paths.append(json.dumps([part for part in path if part], ensure_ascii=False, separators=(",", ":")))
                confs.append(float(record.get("confidence", 0.0) or 0.0))
            if requests_sent:
                new_rows, new_confusion = score_paths(truth, pred_paths, confs,
                    "local_plus_gemini", level_names, weights=heldout_weights)
                for row in new_rows:
                    row.update(threshold=None, split="held_out_test", training_rows=len(split["train"]),
                               validation_rows=len(split["validation"]), spend_weighted_accuracy=np.nan,
                               processing_seconds=round(time.perf_counter() - pipeline_start, 3))
                metrics = pd.concat([metrics, pd.DataFrame(new_rows)], ignore_index=True)
                confusion = pd.concat([confusion, pd.DataFrame(new_confusion)], ignore_index=True)
            if semantic is not None:
                position_by_id = {str(rid): i for i, rid in enumerate(frame["record_id"].astype(str))}
                sem_results = [semantic[position_by_id[rid]] for rid in heldout.record_id.astype(str)]
                sem_paths = [json.dumps(list(result["path"]), ensure_ascii=False, separators=(",", ":"))
                             for result in sem_results]
                sem_conf = [result["confidence"] for result in sem_results]
                sem_rows, sem_confusion = score_paths(truth, sem_paths, sem_conf,
                    "sentence_transformers", level_names, weights=heldout_weights)
                for row in sem_rows:
                    row.update(threshold=None, split="held_out_test", training_rows=len(split["train"]),
                               validation_rows=len(split["validation"]), spend_weighted_accuracy=np.nan,
                               processing_seconds=round(time.perf_counter() - pipeline_start, 3))
                metrics = pd.concat([metrics, pd.DataFrame(sem_rows)], ignore_index=True)
                confusion = pd.concat([confusion, pd.DataFrame(sem_confusion)], ignore_index=True)
    if not metrics.empty:
        metrics["processing_seconds"] = round(time.perf_counter() - pipeline_start, 3)
    split_report = pd.DataFrame([{"partition": name, "records": len(values)}
                                 for name, values in split.items()])
    estimated_cost = client.estimate_cost(input_tokens, output_tokens) if client else 0.0
    dry_run_cost = client.estimate_cost(estimated_input_tokens, estimated_output_tokens) if client else 0.0
    run_metrics = pd.DataFrame([{
        "model": "gemini-3.8-flash", "status": "dry_run" if dry_run else ("executed" if requests_sent else ("disabled_or_no_key" if not api_key else "no_fallback_needed")),
        "rows": len(frame), "low_confidence_candidates": sum(
            (local["local_confidence"].fillna(0) < threshold).tolist()) if "local_confidence" in local else 0,
        "unique_candidate_requests_estimate": candidate_requests, "api_requests": requests_sent,
        "cache_hits": cache_hits, "input_tokens": input_tokens, "output_tokens": output_tokens,
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "estimated_cost_usd": estimated_cost,
        "dry_run_estimated_cost_usd": dry_run_cost, "processing_seconds": elapsed,
        "model_status": model_status, "local_threshold": threshold,
        "gemini_threshold": gemini_threshold,
        "semantic_status": "enabled" if semantic is not None else "not_available_or_disabled",
        "evaluation_status": ";".join(f"{item['model']}={item['status']}" for item in evaluation_status),
        "total_classification_seconds": round(time.perf_counter() - pipeline_start, 3),
        "pricing_input_usd_per_million": gemini_cfg.get("price_input_per_million", 0.75),
        "pricing_output_usd_per_million": gemini_cfg.get("price_output_per_million", 3.75),
    }])
    return (pd.DataFrame(rows, index=frame.index), {"metrics": metrics, "confusion": confusion,
            "decisions": pd.DataFrame(logs), "gemini_run": run_metrics, "split": split,
            "split_report": split_report})
