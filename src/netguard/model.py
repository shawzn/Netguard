from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from xgboost import XGBClassifier


@dataclass
class TrainedExperiment:
    name: str
    model: XGBClassifier
    best_iteration: int
    feature_importances: list[dict[str, float | str]]


def build_xgboost_classifier(
    random_state: int = 42,
    early_stopping_rounds: int = 30,
    **overrides: Any,
) -> XGBClassifier:
    """Create a standardized XGBClassifier for controlled A/B comparison."""
    params: dict[str, Any] = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "early_stopping_rounds": early_stopping_rounds,
        "random_state": random_state,
        "n_jobs": -1,
    }
    params.update(overrides)
    return XGBClassifier(**params)


def train_experiment(
    experiment_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    random_state: int = 42,
) -> TrainedExperiment:
    """Fit XGBoost on the given training split using X_val for early stopping."""
    model = build_xgboost_classifier(random_state=random_state)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    importances = model.feature_importances_
    ranked_features = sorted(
        [
            {"feature": name, "importance": round(float(score), 6)}
            for name, score in zip(feature_names, importances)
        ],
        key=lambda item: float(item["importance"]),
        reverse=True,
    )

    best_iter = getattr(model, "best_iteration", model.n_estimators - 1)
    return TrainedExperiment(
        name=experiment_name,
        model=model,
        best_iteration=int(best_iter),
        feature_importances=ranked_features,
    )


def save_model_artifact(model: XGBClassifier, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model_artifact(path: str | Path) -> XGBClassifier:
    return joblib.load(path)
