"""Classify cagemate (0) versus novel (1) from power and coherence.

The input is expected to be the long-format safe_balanced_3_regions.csv file.
Partner identity (A/B) is included as a nuisance covariate, and all held-out
predictions use leave-one-subject-out cross-validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


OBS_KEYS = ["subject", "recording", "event", "partner", "trial", "condition"]


def load_wide(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    raw = pd.read_csv(csv_path)
    required = set(OBS_KEYS + ["band", "metric", "region_or_pair", "value"])
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    raw = raw.loc[raw["metric"].isin(["power", "coherence"])].copy()
    raw["feature"] = (
        raw["metric"].astype(str)
        + "__"
        + raw["region_or_pair"].astype(str)
        + "__"
        + raw["band"].astype(str)
    )
    if raw.duplicated(OBS_KEYS + ["feature"]).any():
        raise ValueError("Duplicate neural feature rows found within an observation")

    wide = raw.pivot(index=OBS_KEYS, columns="feature", values="value").reset_index()
    wide.columns.name = None
    wide["target"] = wide["condition"].map({"cagemate": 0, "novel": 1})
    if wide["target"].isna().any():
        bad = sorted(wide.loc[wide["target"].isna(), "condition"].unique())
        raise ValueError(f"Unexpected condition labels: {bad}")
    feature_cols = sorted(c for c in wide.columns if c.startswith(("power__", "coherence__")))
    return wide, feature_cols


def make_model(feature_cols: list[str], include_identity: bool) -> Pipeline:
    transformers = [
        (
            "network",
            Pipeline(
                [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
            ),
            feature_cols,
        )
    ]
    if include_identity:
        transformers.append(
            (
                "identity",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                ["partner"],
            )
        )
    prep = ColumnTransformer(transformers, remainder="drop")
    return Pipeline(
        [
            ("prep", prep),
            (
                "logit",
                LogisticRegression(
                    penalty="l2", C=1.0, solver="liblinear", max_iter=5000, random_state=7
                ),
            ),
        ]
    )


def loso_predictions(
    data: pd.DataFrame, feature_cols: list[str], include_identity: bool
) -> pd.DataFrame:
    logo = LeaveOneGroupOut()
    out = []
    y = data["target"].to_numpy()
    groups = data["subject"].astype(str).to_numpy()
    for train, test in logo.split(data, y, groups):
        model = make_model(feature_cols, include_identity)
        model.fit(data.iloc[train], y[train])
        probability = model.predict_proba(data.iloc[test])[:, 1]
        fold = data.iloc[test][OBS_KEYS + ["target"]].copy()
        fold["prob_novel"] = probability
        fold["predicted"] = (probability >= 0.5).astype(int)
        out.append(fold)
    return pd.concat(out, ignore_index=True)


def score(pred: pd.DataFrame) -> dict[str, object]:
    y = pred["target"].to_numpy()
    yhat = pred["predicted"].to_numpy()
    return {
        "n_observations": int(len(pred)),
        "n_subjects": int(pred["subject"].nunique()),
        "accuracy": float(accuracy_score(y, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "roc_auc": float(roc_auc_score(y, pred["prob_novel"])),
        "confusion_matrix_rows_true_0_1": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    }


def coefficient_table(data: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    model = make_model(feature_cols, include_identity=True).fit(data, data["target"])
    names = model.named_steps["prep"].get_feature_names_out()
    beta = model.named_steps["logit"].coef_[0]
    table = pd.DataFrame({"term": names, "standardized_coefficient": beta})
    table["absolute_coefficient"] = table["standardized_coefficient"].abs()
    return table.sort_values("absolute_coefficient", ascending=False, ignore_index=True)


def identity_only_predictions(data: pd.DataFrame) -> pd.DataFrame:
    # A transparent nuisance-only benchmark, evaluated with the identical folds.
    out = []
    logo = LeaveOneGroupOut()
    y = data["target"].to_numpy()
    groups = data["subject"].astype(str).to_numpy()
    for train, test in logo.split(data, y, groups):
        model = Pipeline(
            [
                ("identity", OneHotEncoder(drop="first", handle_unknown="ignore")),
                ("logit", LogisticRegression(solver="liblinear", random_state=7)),
            ]
        )
        model.fit(data.iloc[train][["partner"]], y[train])
        p = model.predict_proba(data.iloc[test][["partner"]])[:, 1]
        fold = data.iloc[test][OBS_KEYS + ["target"]].copy()
        fold["prob_novel"] = p
        fold["predicted"] = (p >= 0.5).astype(int)
        out.append(fold)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/network_partner_logistic"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data, features = load_wide(args.csv)
    adjusted = loso_predictions(data, features, include_identity=True)
    network_only = loso_predictions(data, features, include_identity=False)
    identity_only = identity_only_predictions(data)

    metrics = {
        "outcome_coding": {"cagemate": 0, "novel": 1},
        "cross_validation": "leave-one-subject-out",
        "identity_control": "partner A/B included as nuisance covariate",
        "n_network_features": len(features),
        "network_plus_identity": score(adjusted),
        "network_only": score(network_only),
        "identity_only": score(identity_only),
    }
    adjusted.to_csv(args.output_dir / "loso_predictions.csv", index=False)
    coefficient_table(data, features).to_csv(args.output_dir / "full_model_coefficients.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(args.output_dir / "network_features.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
