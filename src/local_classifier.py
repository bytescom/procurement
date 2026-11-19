"""Label-trained local taxonomy classifier with a leakage-safe holdout split."""

from __future__ import annotations

from dataclasses import dataclass
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline


@dataclass
class LocalModelResult:
    predictions: pd.DataFrame
    evaluation: pd.DataFrame
    confusion: pd.DataFrame
    split: dict[str, set[str]]
    threshold: float
    status: str


def _text(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    present = [column for column in columns if column in frame]
    if not present:
        return pd.Series("", index=frame.index, dtype="string")
    return frame[present].fillna("").astype(str).agg(" ".join, axis=1).str.replace(r"\s+", " ", regex=True).str.strip()


def _split_indices(target: pd.Series, groups: pd.Series | None = None, seed: int = 42):
    indices = np.arange(len(target))
    if groups is not None and groups.nunique() >= 3:
        try:
            outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            train_val, test = next(outer.split(indices, target, groups=groups))
            inner = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
            train, validation = next(inner.split(train_val, target.iloc[train_val],
                                                  groups=groups.iloc[train_val]))
            return train_val[train], train_val[validation], test
        except ValueError:
            pass
    counts = target.value_counts()
    stratify = target if len(counts) > 1 and counts.min() >= 3 else None
    try:
        train_val, test = train_test_split(indices, test_size=0.2, random_state=seed,
                                           stratify=stratify)
        train_target = target.iloc[train_val]
        counts = train_target.value_counts()
        stratify_train = train_target if len(counts) > 1 and counts.min() >= 2 else None
        train, validation = train_test_split(train_val, test_size=0.125, random_state=seed,
                                             stratify=stratify_train)
        return train, validation, test
    except ValueError:
        train_val, test = train_test_split(indices, test_size=0.2, random_state=seed)
        train, validation = train_test_split(train_val, test_size=0.125, random_state=seed)
        return train, validation, test


def _estimator():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=120_000,
                        sublinear_tf=True, strip_accents="unicode", dtype=np.float32),
        LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs", C=4.0),
    )


def _path_class(path: tuple[str, ...]) -> str:
    return json.dumps(list(path), ensure_ascii=False, separators=(",", ":"))


def _class_path(value: str) -> tuple[str, ...]:
    return tuple(json.loads(value))


def _metrics(y_true: list[str], y_pred: list[str], y_conf: list[float], model: str,
             level_names: list[str], threshold: float | None = None,
             weights: list[float] | None = None) -> tuple[list[dict], list[dict]]:
    rows, matrices = [], []
    for depth, level in enumerate(level_names):
        truth = [json.loads(value)[depth] if len(json.loads(value)) > depth else "" for value in y_true]
        pred = [json.loads(value)[depth] if len(json.loads(value)) > depth else "" for value in y_pred]
        accepted = [bool(value) and (threshold is None or conf >= threshold)
                    for value, conf in zip(pred, y_conf)]
        correct = [accepted[i] and pred[i] == truth[i] for i in range(len(truth))]
        confusion_pred = [pred[i] if accepted[i] else "" for i in range(len(truth))]
        raw_correct = [pred[i] == truth[i] for i in range(len(truth))]
        spend_weighted = np.nan
        raw_spend_weighted = np.nan
        if weights is not None and len(weights) == len(truth) and sum(max(0.0, w) for w in weights) > 0:
            normalized_weights = [max(0.0, w) for w in weights]
            spend_weighted = float(np.average(correct, weights=normalized_weights))
            raw_spend_weighted = float(np.average(raw_correct, weights=normalized_weights))
        from sklearn.metrics import confusion_matrix
        labels = sorted(set(truth) | set(confusion_pred))
        matrix = confusion_matrix(truth, confusion_pred, labels=labels)
        for i, label in enumerate(labels):
            for j, predicted_label in enumerate(labels):
                if matrix[i, j]:
                    matrices.append({"model": model, "level": level, "actual": label,
                                     "predicted": predicted_label, "count": int(matrix[i, j])})
        rows.append({"model": model, "level": level, "n_test": len(truth),
                     "accuracy": float(np.mean(correct)),
                     "raw_accuracy_before_abstention": float(np.mean(raw_correct)),
                     "prediction_coverage": float(np.mean(accepted)),
                     "accuracy_when_predicted": float(np.mean([correct[i] for i in range(len(truth)) if accepted[i]]))
                     if any(accepted) else np.nan,
                     "spend_weighted_accuracy": spend_weighted,
                     "raw_spend_weighted_accuracy_before_abstention": raw_spend_weighted,
                     "blank_prediction_pct": float(np.mean([not accepted[i] for i in range(len(truth))]))})
    return rows, matrices


def train_local_classifier(features: pd.DataFrame, text_columns: list[str], labels: pd.DataFrame,
                           taxonomy_paths: list[tuple[str, ...]], level_names: list[str],
                           *, record_id_column: str = "record_id", seed: int = 42) -> LocalModelResult | None:
    """Train on an internal 70% split; validation calibrates abstention, test stays held out."""
    taxonomy_set = set(taxonomy_paths)
    usable = labels[labels["path"].map(lambda path: tuple(path) in taxonomy_set)].copy().reset_index(drop=True)
    if usable.empty:
        return None
    usable["class"] = usable.path.map(lambda value: _path_class(tuple(value)))
    texts_by_id = features.set_index(record_id_column)
    texts_by_id.index = texts_by_id.index.astype(str)
    usable[record_id_column] = usable[record_id_column].astype(str)
    label_texts = _text(texts_by_id.reindex(usable[record_id_column]), text_columns).reset_index(drop=True)
    # Ignore labels without a matching transaction row and rows with no usable
    # text. With fewer than 30 usable labels, the caller falls back to the
    # taxonomy-only classifier instead of fitting an empty vocabulary.
    has_text = label_texts.str.strip().ne("")
    usable = usable.loc[has_text].reset_index(drop=True)
    label_texts = label_texts.loc[has_text].reset_index(drop=True)
    if len(usable) < 30 or usable.path.map(tuple).nunique() < 2:
        return None
    train_idx, val_idx, test_idx = _split_indices(usable["class"], groups=label_texts, seed=seed)
    y = usable["class"].reset_index(drop=True)
    train_model = _estimator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_model.fit(label_texts.iloc[train_idx], y.iloc[train_idx])
    val_prob = train_model.predict_proba(label_texts.iloc[val_idx])
    val_class_idx = val_prob.argmax(axis=1)
    val_classes = train_model.classes_[val_class_idx]
    val_conf = val_prob.max(axis=1)
    val_truth = y.iloc[val_idx].to_numpy()
    candidates = sorted(set([0.0, *np.quantile(val_conf, np.linspace(0, 0.9, 19)).tolist()]))
    choices = []
    for cutoff in candidates:
        mask = val_conf >= cutoff
        if mask.sum() >= max(10, int(0.2 * len(val_idx))):
            choices.append((float(np.mean(val_classes[mask] == val_truth[mask])),
                            float(mask.mean()), float(cutoff)))
    threshold = max(choices, key=lambda item: (item[0], item[1]))[2] if choices else 0.0

    # Validation examples are available for final fit; test examples remain unseen.
    final_idx = np.concatenate([train_idx, val_idx])
    model = _estimator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(label_texts.iloc[final_idx], y.iloc[final_idx])
    test_prob = model.predict_proba(label_texts.iloc[test_idx])
    test_classes = model.classes_[test_prob.argmax(axis=1)]
    test_conf = test_prob.max(axis=1)
    test_record_ids = usable.iloc[test_idx][record_id_column].astype(str).tolist()
    feature_by_id = features.set_index(record_id_column)
    weight_col = next((col for col in ("usd_amount", "net_amount", "spend", "amount") if col in features), None)
    test_weights = None
    if weight_col:
        test_weights = pd.to_numeric(feature_by_id.reindex(test_record_ids)[weight_col], errors="coerce").abs().fillna(0).tolist()
    metric_rows, matrix_rows = _metrics(y.iloc[test_idx].tolist(), test_classes.tolist(),
                                        test_conf.tolist(), "tfidf_logistic", level_names,
                                        threshold=threshold, weights=test_weights)
    for row in metric_rows:
        row.update(threshold=threshold, split="held_out_test", training_rows=len(train_idx),
                   validation_rows=len(val_idx))
    test_ids = set(usable.iloc[test_idx][record_id_column].astype(str))
    val_ids = set(usable.iloc[val_idx][record_id_column].astype(str))
    train_ids = set(usable.iloc[train_idx][record_id_column].astype(str))

    texts = _text(features, text_columns)
    all_prob = model.predict_proba(texts)
    best_idx = all_prob.argmax(axis=1)
    all_classes = model.classes_[best_idx]
    all_conf = all_prob.max(axis=1)
    predicted_paths = [_class_path(value) for value in all_classes]
    output = pd.DataFrame(index=features.index)
    for i in range(4):
        output[f"l{i+1}"] = [path[i] if i < len(path) else "" for path in predicted_paths]
    output["confidence"] = all_conf
    output["classification_status"] = np.where(all_conf >= threshold, "CLASSIFIED", "LOW_CONFIDENCE_LOCAL")
    output["classification_method"] = "tfidf_logistic_train_split"
    output["local_confidence"] = all_conf

    # Compare the existing lexical baseline on this exact held-out test split.
    baseline_cols = [f"baseline_l{i+1}" for i in range(len(level_names))]
    if all(column in features for column in baseline_cols):
        feature_by_id = features.set_index(record_id_column)
        baseline_paths = []
        baseline_conf = []
        for rid in test_record_ids:
            row = feature_by_id.loc[rid]
            path = tuple(str(row[column]) for column in baseline_cols if pd.notna(row[column]) and str(row[column]).strip())
            baseline_paths.append(_path_class(path))
            baseline_conf.append(float(row.get("baseline_confidence", 0.0) or 0.0))
        baseline_rows, baseline_matrix = _metrics(y.iloc[test_idx].tolist(), baseline_paths,
                                                   baseline_conf, "existing_lexical", level_names,
                                                   weights=test_weights)
        for row in baseline_rows:
            row.update(threshold=0.0, split="held_out_test", training_rows=len(train_idx),
                       validation_rows=len(val_idx))
        metric_rows = baseline_rows + metric_rows
        matrix_rows = baseline_matrix + matrix_rows
    metrics = pd.DataFrame(metric_rows)
    confusion = pd.DataFrame(matrix_rows)
    return LocalModelResult(output, metrics, confusion,
                            {"train": train_ids, "validation": val_ids, "test": test_ids},
                            threshold, "trained")
