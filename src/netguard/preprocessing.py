from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

from netguard.cleaning import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERICAL_COLUMNS,
)


@dataclass
class PreparedDatasets:
    X_train_baseline: np.ndarray
    y_train_baseline: np.ndarray
    X_train_smotenc: np.ndarray
    y_train_smotenc: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


class NetGuardPreprocessor:
    """Leak-free preprocessing pipeline supporting both Baseline and SMOTENC."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.cat_cols = CATEGORICAL_COLUMNS
        self.num_cols = NUMERICAL_COLUMNS
        self.feature_cols = FEATURE_COLUMNS

        self.pre_smote_transformer = ColumnTransformer(
            transformers=[
                (
                    "cat_ord",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=-1,
                    ),
                    self.cat_cols,
                ),
                ("num_scale", RobustScaler(), self.num_cols),
            ],
            remainder="drop",
        )

        self.post_smote_encoder = ColumnTransformer(
            transformers=[
                (
                    "cat_ohe",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    [0, 1, 2],
                ),
            ],
            remainder="passthrough",
        )

        self.is_fitted_: bool = False
        self.output_feature_names_: list[str] = []

    def fit(self, X_train: pd.DataFrame) -> NetGuardPreprocessor:
        X_ordered = X_train[self.feature_cols]
        X_stage1 = self.pre_smote_transformer.fit_transform(X_ordered)
        self.post_smote_encoder.fit(X_stage1)

        ord_encoder: OrdinalEncoder = self.pre_smote_transformer.named_transformers_["cat_ord"]
        ohe_encoder: OneHotEncoder = self.post_smote_encoder.named_transformers_["cat_ohe"]

        cat_feature_names: list[str] = []
        for col_idx, col_name in enumerate(self.cat_cols):
            raw_cats = ord_encoder.categories_[col_idx]
            used_codes = ohe_encoder.categories_[col_idx]
            for code in used_codes:
                code_int = int(code)
                cat_label = raw_cats[code_int] if 0 <= code_int < len(raw_cats) else "unknown"
                cat_feature_names.append(f"{col_name}__{cat_label}")

        self.output_feature_names_ = cat_feature_names + list(self.num_cols)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("NetGuardPreprocessor must be fitted before transform().")
        X_ordered = X[self.feature_cols]
        X_stage1 = self.pre_smote_transformer.transform(X_ordered)
        return np.asarray(self.post_smote_encoder.transform(X_stage1), dtype=np.float32)

    def fit_resample_train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.fit(X_train)
        X_ordered = X_train[self.feature_cols]
        y_arr = np.asarray(y_train, dtype=np.int64)

        X_stage1 = self.pre_smote_transformer.transform(X_ordered)
        X_baseline = np.asarray(
            self.post_smote_encoder.transform(X_stage1), dtype=np.float32
        )

        t0 = time.perf_counter()
        print("  - Running SMOTENC on training split (multi-core)...", flush=True)
        nn_parallel = NearestNeighbors(n_neighbors=5, n_jobs=-1)
        smotenc = SMOTENC(
            categorical_features=[0, 1, 2],
            k_neighbors=nn_parallel,
            random_state=self.random_state,
        )
        X_stage1_res, y_res = smotenc.fit_resample(X_stage1, y_arr)
        X_smotenc = np.asarray(
            self.post_smote_encoder.transform(X_stage1_res), dtype=np.float32
        )
        elapsed = time.perf_counter() - t0
        print(f"  - SMOTENC completed in {elapsed:.1f}s", flush=True)

        return X_baseline, y_arr, X_smotenc, np.asarray(y_res, dtype=np.int64)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> NetGuardPreprocessor:
        return joblib.load(path)


def prepare_nsl_kdd_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_size: float = 0.2,
    random_state: int = 42,
    preprocessor_save_path: str | Path | None = None,
) -> tuple[PreparedDatasets, NetGuardPreprocessor]:
    X_full = train_df[FEATURE_COLUMNS]
    y_full = train_df["target"]

    X_train, X_val, y_train, y_val = train_test_split(
        X_full,
        y_full,
        test_size=val_size,
        stratify=y_full,
        random_state=random_state,
    )

    preprocessor = NetGuardPreprocessor(random_state=random_state)
    X_train_base, y_train_base, X_train_smote, y_train_smote = (
        preprocessor.fit_resample_train(X_train, y_train)
    )

    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(test_df[FEATURE_COLUMNS])

    if preprocessor_save_path is not None:
        preprocessor.save(preprocessor_save_path)

    datasets = PreparedDatasets(
        X_train_baseline=X_train_base,
        y_train_baseline=y_train_base,
        X_train_smotenc=X_train_smote,
        y_train_smotenc=y_train_smote,
        X_val=X_val_proc,
        y_val=np.asarray(y_val, dtype=np.int64),
        X_test=X_test_proc,
        y_test=np.asarray(test_df["target"], dtype=np.int64),
        feature_names=preprocessor.output_feature_names_,
    )
    return datasets, preprocessor
