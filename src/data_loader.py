from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
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

TARGET_COLUMN = "label"
METADATA_COLUMN = "difficulty"

ALL_COLUMNS = FEATURE_COLUMNS + [
    TARGET_COLUMN,
    METADATA_COLUMN,
]


def load_nsl_kdd(file_path: str | Path) -> pd.DataFrame:
    """
    Load NSL-KDD data and assign column names.

    Returns:
        DataFrame containing 41 features, original label,
        difficulty metadata, and binary target.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        header=None,
        names=ALL_COLUMNS,
    )

    if df.shape[1] != 43:
        raise ValueError(
            f"Expected 43 columns, got {df.shape[1]}"
        )

    if df.isna().any().any():
        raise ValueError(
            "Dataset contains missing values."
        )

    if not df[TARGET_COLUMN].isin(
        df[TARGET_COLUMN].unique()
    ).all():
        raise ValueError("Invalid target values.")

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.rstrip(".")
    )

    df["binary_label"] = (
        df[TARGET_COLUMN] != "normal"
    ).astype("int8")

    return df


def split_features_target(
    df: pd.DataFrame,
):
    """
    Separate model features and binary target.

    Difficulty and original label are excluded
    from model features.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df["binary_label"].copy()

    return X, y


def generate_quality_report(
    df: pd.DataFrame,
) -> dict:
    """Generate a basic dataset quality summary."""
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "feature_count": len(FEATURE_COLUMNS),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "attack_count": int(df["binary_label"].sum()),
        "normal_count": int(
            (df["binary_label"] == 0).sum()
        ),
        "original_label_distribution": (
            df[TARGET_COLUMN]
            .value_counts()
            .to_dict()
        ),
    }
