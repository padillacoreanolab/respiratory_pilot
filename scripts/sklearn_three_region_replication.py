"""Replicate the reference nested logistic models with scikit-learn.

This is a public-library validation of the custom NumPy optimizer in
junwoozy/three-region-logistic-regression. It preserves the same features,
labels, subject roles, L2 candidates, validation selection rule, and final
six-subject refit. The custom objective's lambda maps to scikit-learn as
``C = 1 / (n_rows * lambda)``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


ID_COLUMNS = ["subject", "recording", "event", "partner", "trial", "condition"]
L2_CANDIDATES = (0.0001, 0.001, 0.01, 0.1, 1.0)


@dataclass(frozen=True)
class Fold:
    split_name: str
    train_subjects: tuple[str, ...]
    validation_subject: str
    test_subject: str


def build_wide(csv_path: Path, condition: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_csv(csv_path)
    for column in ("subject", "partner", "condition"):
        data[column] = data[column].astype(str)
    if condition is not None:
        data = data[data["condition"].eq(condition)].copy()
    tables = []
    for metric in ("power", "coherence"):
        wide = data[data["metric"].eq(metric)].pivot_table(
            index=ID_COLUMNS,
            columns=["band", "region_or_pair"],
            values="value",
            aggfunc="first",
        )
        wide.columns = [f"{band}__{metric}__{region}" for band, region in wide.columns]
        tables.append(wide.reset_index())
    result = tables[0].merge(tables[1], on=ID_COLUMNS, validate="one_to_one")
    result["interaction_id"] = result[ID_COLUMNS].astype(str).agg("||".join, axis=1)
    features = [c for c in result if "__power__" in c or "__coherence__" in c]
    return result.reset_index(drop=True), features


def subject_folds(subjects: list[str], seed: int) -> list[Fold]:
    ordered = tuple(sorted(set(map(str, subjects))))
    rng = np.random.default_rng(seed)
    validation_order = np.asarray(ordered, dtype=object)
    while True:
        validation_order = rng.permutation(validation_order)
        if all(t != v for t, v in zip(ordered, validation_order)):
            break
    folds = []
    for i, (test, validation) in enumerate(zip(ordered, validation_order), 1):
        validation = str(validation)
        train = tuple(s for s in ordered if s not in {test, validation})
        folds.append(Fold(f"subject_{i:02d}_{test}", train, validation, test))
    return folds


def fit_model(x: np.ndarray, y: np.ndarray, l2: float):
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        penalty="l2",
        C=1.0 / (len(y) * l2),
        solver="lbfgs",
        fit_intercept=True,
        max_iter=10000,
        tol=1e-12,
        random_state=42,
    ).fit(scaler.transform(x), y)
    return scaler, model


def select_l2(x_train, y_train, x_val, y_val) -> tuple[float, list[dict]]:
    candidates = []
    for l2 in L2_CANDIDATES:
        scaler, model = fit_model(x_train, y_train, l2)
        probability = model.predict_proba(scaler.transform(x_val))[:, 1]
        prediction = (probability >= 0.5).astype(int)
        candidates.append(
            {
                "l2": l2,
                "validation_auc": roc_auc_score(y_val, probability),
                "validation_balanced_accuracy": balanced_accuracy_score(y_val, prediction),
                "validation_loss": log_loss(y_val, probability, labels=[0, 1]),
            }
        )
    best = max(
        candidates,
        key=lambda r: (
            r["validation_auc"],
            r["validation_balanced_accuracy"],
            -r["validation_loss"],
            r["l2"],
        ),
    )
    return float(best["l2"]), candidates


def run_task(csv_path: Path, task: str, condition: str | None, seed: int, output: Path):
    wide, features = build_wide(csv_path, condition)
    if task == "familiarity":
        y = wide["condition"].map({"cagemate": 0, "novel": 1}).to_numpy(int)
    else:
        y = wide["partner"].map({"A": 0, "B": 1}).to_numpy(int)
    x = wide[features].to_numpy(float)
    subject = wide["subject"].to_numpy(str)
    metrics, predictions, coefficients, tuning = [], [], [], []
    for fold in subject_folds(subject.tolist(), seed):
        train = np.flatnonzero(np.isin(subject, fold.train_subjects))
        validation = np.flatnonzero(subject == fold.validation_subject)
        test = np.flatnonzero(subject == fold.test_subject)
        chosen, candidate_rows = select_l2(x[train], y[train], x[validation], y[validation])
        final = np.sort(np.concatenate([train, validation]))
        scaler, model = fit_model(x[final], y[final], chosen)
        probability = model.predict_proba(scaler.transform(x[test]))[:, 1]
        predicted = (probability >= 0.5).astype(int)
        metrics.append(
            {
                "task": task,
                "split_name": fold.split_name,
                "validation_subject": fold.validation_subject,
                "test_subject": fold.test_subject,
                "selected_l2": chosen,
                "sklearn_C_final_fit": model.C,
                "test_accuracy": float(np.mean(predicted == y[test])),
                "test_balanced_accuracy": balanced_accuracy_score(y[test], predicted),
                "test_auc": roc_auc_score(y[test], probability),
                "test_loss": log_loss(y[test], probability, labels=[0, 1]),
            }
        )
        for local_i, row_i in enumerate(test):
            predictions.append(
                {
                    "task": task,
                    "split_name": fold.split_name,
                    "interaction_id": wide.iloc[row_i]["interaction_id"],
                    "subject": wide.iloc[row_i]["subject"],
                    "condition": wide.iloc[row_i]["condition"],
                    "partner": wide.iloc[row_i]["partner"],
                    "actual_label": int(y[row_i]),
                    "predicted_label": int(predicted[local_i]),
                    "predicted_probability_class_1": float(probability[local_i]),
                }
            )
        for feature, value in zip(features, model.coef_[0]):
            coefficients.append(
                {
                    "task": task,
                    "split_name": fold.split_name,
                    "feature": feature,
                    "standardized_coefficient": float(value),
                    "selected_l2": chosen,
                }
            )
        for row in candidate_rows:
            tuning.append({"task": task, "split_name": fold.split_name, **row, "selected": row["l2"] == chosen})
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(output / f"{task}_fold_results.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / f"{task}_fold_predictions.csv", index=False)
    pd.DataFrame(coefficients).to_csv(output / f"{task}_coefficients.csv", index=False)
    pd.DataFrame(tuning).to_csv(output / f"{task}_l2_selection.csv", index=False)
    return pd.DataFrame(metrics), pd.DataFrame(predictions)


def compare_reference(task: str, predictions: pd.DataFrame, reference_root: Path) -> dict:
    reference_file = reference_root / task / "csv" / f"{task}_fold_predictions.csv"
    if not reference_file.exists():
        return {}
    reference = pd.read_csv(reference_file)
    joined = predictions.merge(
        reference[["split_name", "interaction_id", "predicted_label", "predicted_probability_B"]],
        on=["split_name", "interaction_id"],
        suffixes=("_sklearn", "_custom"),
        validate="one_to_one",
    )
    difference = joined["predicted_probability_class_1"] - joined["predicted_probability_B"]
    return {
        "task": task,
        "n_predictions": int(len(joined)),
        "label_agreement": float(np.mean(joined["predicted_label_sklearn"] == joined["predicted_label_custom"])),
        "mean_absolute_probability_difference": float(np.mean(np.abs(difference))),
        "maximum_absolute_probability_difference": float(np.max(np.abs(difference))),
        "probability_correlation": float(np.corrcoef(joined["predicted_probability_class_1"], joined["predicted_probability_B"])[0, 1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sklearn_three_region_replication"))
    parser.add_argument("--reference-results", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    specifications = [
        ("cagemate", "identity", "cagemate"),
        ("novel", "identity", "novel"),
        ("combined", "identity", None),
        ("familiarity", "familiarity", None),
    ]
    summaries, comparisons = [], []
    for name, task_type, condition in specifications:
        metrics, predictions = run_task(args.csv, name if task_type == "identity" else "familiarity", condition, args.seed, args.output_dir)
        summaries.append(
            {
                "task": name,
                "mean_test_accuracy": metrics.test_accuracy.mean(),
                "mean_test_balanced_accuracy": metrics.test_balanced_accuracy.mean(),
                "mean_test_auc": metrics.test_auc.mean(),
                "sem_test_auc": metrics.test_auc.std(ddof=1) / np.sqrt(len(metrics)),
            }
        )
        if args.reference_results:
            comparison = compare_reference(name, predictions, args.reference_results)
            if comparison:
                comparisons.append(comparison)
    pd.DataFrame(summaries).to_csv(args.output_dir / "sklearn_model_summary.csv", index=False)
    if comparisons:
        pd.DataFrame(comparisons).to_csv(args.output_dir / "custom_vs_sklearn_comparison.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    if comparisons:
        print("\nCustom versus scikit-learn:\n" + pd.DataFrame(comparisons).to_string(index=False))


if __name__ == "__main__":
    main()
