from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from netguard.cleaning import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERICAL_COLUMNS,
    clean_nsl_kdd,
)
from netguard.model import load_model_artifact
from netguard.preprocessing import NetGuardPreprocessor

RAW_41_COLUMNS: list[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]


@dataclass
class ScanSummary:
    total_flows: int
    predicted_attacks: int
    predicted_normal: int
    attack_ratio: float
    avg_attack_probability: float
    results_df: pd.DataFrame


class NetGuardPredictor:
    """Production inference wrapper combining NetGuardPreprocessor and XGBoost."""

    def __init__(
        self,
        models_dir: str | Path = "artifacts/models",
        model_variant: str = "xgb_smotenc",
    ) -> None:
        self.models_dir = Path(models_dir)
        self.model_variant = model_variant
        self.preprocessor = NetGuardPreprocessor.load(
            self.models_dir / "preprocessor.joblib"
        )
        self.model = load_model_artifact(self.models_dir / f"{model_variant}.joblib")

    @staticmethod
    def _normalize_input_df(raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()
        # Handle headerless NSL-KDD slices (41, 42, or 43 columns)
        if not set(FEATURE_COLUMNS).issubset(df.columns):
            if df.shape[1] == 41:
                df.columns = RAW_41_COLUMNS
            elif df.shape[1] == 42:
                df.columns = RAW_41_COLUMNS + ["label"]
            elif df.shape[1] == 43:
                df.columns = RAW_41_COLUMNS + ["label", "difficulty"]
            else:
                missing = sorted(set(FEATURE_COLUMNS) - set(df.columns))
                raise ValueError(
                    f"CSV schema mismatch. Expected 41 NSL-KDD feature columns, missing: {missing[:5]}"
                )

        return clean_nsl_kdd(df, target_col="label", drop_difficulty=True)

    @staticmethod
    def _classify_risk(prob: float, threshold: float) -> str:
        if prob >= max(0.85, threshold):
            return "CRITICAL"
        if prob >= threshold:
            return "HIGH"
        if prob >= threshold * 0.6:
            return "MEDIUM"
        return "LOW"

    def predict_dataframe(
        self,
        raw_df: pd.DataFrame,
        threshold: float = 0.5,
        max_rows: int | None = None,
    ) -> ScanSummary:
        if raw_df.empty:
            raise ValueError("Uploaded CSV is empty.")

        sliced_df = raw_df.head(max_rows).copy() if max_rows is not None else raw_df.copy()
        cleaned_df = self._normalize_input_df(sliced_df)

        X_proc = self.preprocessor.transform(cleaned_df[FEATURE_COLUMNS])
        probs = self.model.predict_proba(X_proc)[:, 1]
        preds = (probs >= threshold).astype(int)

        output = pd.DataFrame(
            {
                "flow_id": np.arange(1, len(cleaned_df) + 1),
                "prediction": np.where(preds == 1, "ATTACK", "NORMAL"),
                "attack_probability": np.round(probs, 4),
                "risk_level": [self._classify_risk(float(p), threshold) for p in probs],
                "protocol_type": cleaned_df["protocol_type"].values,
                "service": cleaned_df["service"].values,
                "flag": cleaned_df["flag"].values,
                "src_bytes": cleaned_df["src_bytes"].astype(int).values,
                "dst_bytes": cleaned_df["dst_bytes"].astype(int).values,
                "count": cleaned_df["count"].astype(int).values,
                "serror_rate": np.round(cleaned_df["serror_rate"].values, 3),
            }
        )

        if "attack_type" in cleaned_df.columns:
            output["ground_truth_label"] = cleaned_df["attack_type"].values
            output["ground_truth_family"] = cleaned_df["attack_category"].values
            output["is_correct"] = np.where(
                preds == cleaned_df["target"].values, "MATCH", "MISMATCH"
            )

        total = len(output)
        n_attacks = int((preds == 1).sum())
        n_normal = total - n_attacks

        return ScanSummary(
            total_flows=total,
            predicted_attacks=n_attacks,
            predicted_normal=n_normal,
            attack_ratio=round(n_attacks / total, 4) if total else 0.0,
            avg_attack_probability=round(float(np.mean(probs)), 4) if total else 0.0,
            results_df=output,
        )
