"""Training, persistence and scoring for the credit risk model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "german_credit_data.csv"
MODEL_PATH = ROOT / "models" / "credit_model.joblib"

NUMERIC = [
    "month_duration", "credit_amount", "payment_to_income_ratio", "residence_since",
    "age", "n_credits", "n_guarantors",
]
CATEGORICAL = [
    "status_account", "credit_history", "purpose", "status_savings", "years_employment",
    "secondary_obligor", "collateral", "other_installment_plans", "housing", "job", "telephone",
]
FEATURES = NUMERIC + CATEGORICAL
# Protected attributes are dropped so they cannot influence lending decisions.
EXCLUDED = ["status_and_sex", "is_foreign_worker"]
TARGET = "target"

# Probability-of-default cut-offs for the risk bands.
THRESHOLDS = {"low_below": 0.35, "high_from": 0.60}


@dataclass
class TrainedModel:
    pipeline: Pipeline
    metrics: dict
    rows: int
    version: str = MODEL_VERSION


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    return Pipeline([("pre", pre), ("clf", clf)])


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y = (df[TARGET] == "bad").astype(int)
    return df[FEATURES], y


def train(path: Path = DATA_PATH, seed: int = 42) -> TrainedModel:
    X, y = load_data(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_auc = cross_val_score(build_pipeline(), X_train, y_train, cv=cv, scoring="roc_auc").mean()

    holdout = build_pipeline().fit(X_train, y_train)
    proba = holdout.predict_proba(X_test)[:, 1]
    metrics = {
        "cv_roc_auc": round(float(cv_auc), 3),
        "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 3),
        "test_recall_default": round(float(recall_score(y_test, proba >= 0.5)), 3),
    }
    # Final model uses all rows; the metrics above come from held-out data.
    final = build_pipeline().fit(X, y)
    return TrainedModel(pipeline=final, metrics=metrics, rows=len(X))


def save(model: TrainedModel, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_or_train(path: Path = MODEL_PATH) -> TrainedModel:
    if path.exists():
        model = joblib.load(path)
        if getattr(model, "version", None) == MODEL_VERSION:
            return model
    model = train()
    save(model, path)
    return model


def risk_band(p: float) -> str:
    if p < THRESHOLDS["low_below"]:
        return "Low"
    if p < THRESHOLDS["high_from"]:
        return "Medium"
    return "High"


RECOMMENDATION = {"Low": "Approve", "Medium": "Refer to credit officer", "High": "Decline"}


def explain(model: TrainedModel, X: pd.DataFrame, top_n: int = 3) -> list[list[tuple[str, float]]]:
    """Per-applicant contributions to the log-odds of default, grouped by original feature."""
    pre = model.pipeline.named_steps["pre"]
    coef = model.pipeline.named_steps["clf"].coef_[0]
    Z = pre.transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    names = pre.get_feature_names_out()
    owner = []
    for n in names:
        kind, rest = n.split("__", 1)
        owner.append(rest if kind == "num" else next(c for c in CATEGORICAL if rest.startswith(c + "_")))
    contrib = pd.DataFrame(Z * coef, columns=names).T.groupby(owner).sum().T
    out = []
    for _, row in contrib.iterrows():
        top = row[row > 0].sort_values(ascending=False).head(top_n)
        out.append([(f, round(float(v), 3)) for f, v in top.items()])
    return out


def score(model: TrainedModel, X: pd.DataFrame) -> list[dict]:
    X = X[FEATURES]
    proba = model.pipeline.predict_proba(X)[:, 1]
    factors = explain(model, X)
    results = []
    for p, f in zip(proba, factors, strict=True):
        band = risk_band(float(p))
        results.append(
            {
                "probability_of_default": round(float(p), 4),
                "risk_band": band,
                "recommendation": RECOMMENDATION[band],
                "top_risk_factors": [{"feature": n, "contribution": c} for n, c in f],
                "model_version": model.version,
            }
        )
    return results
