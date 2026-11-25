from __future__ import annotations

import pandas as pd

CATEGORICAL_COLUMNS: list[str] = ["protocol_type", "service", "flag"]

NUMERICAL_COLUMNS: list[str] = [
    "duration",
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

FEATURE_COLUMNS: list[str] = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS

ATTACK_CATEGORY_MAP: dict[str, str] = {
    "normal": "Normal",
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "apache2": "DoS",
    "udpstorm": "DoS",
    "processtable": "DoS",
    "mailbomb": "DoS",
    "satan": "Probe",
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "mscan": "Probe",
    "saint": "Probe",
    "guess_passwd": "R2L",
    "ftp_write": "R2L",
    "imap": "R2L",
    "phf": "R2L",
    "multihop": "R2L",
    "warezmaster": "R2L",
    "warezclient": "R2L",
    "spy": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "snmpguess": "R2L",
    "snmpgetattack": "R2L",
    "httptunnel": "R2L",
    "sendmail": "R2L",
    "named": "R2L",
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "rootkit": "U2R",
    "perl": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
    "ps": "U2R",
}


def map_attack_family(label: str) -> str:
    """Map raw NSL-KDD attack label into 5 high-level categories."""
    cleaned = str(label).strip().rstrip(".").lower()
    return ATTACK_CATEGORY_MAP.get(cleaned, "Unknown_Attack")


def clean_nsl_kdd(
    df: pd.DataFrame,
    target_col: str = "label",
    drop_difficulty: bool = True,
) -> pd.DataFrame:
    """Standardize schema, dtypes, and target labels without removing outliers."""
    cleaned = df.copy()

    if drop_difficulty and "difficulty" in cleaned.columns:
        cleaned = cleaned.drop(columns=["difficulty"])

    for col in CATEGORICAL_COLUMNS:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    for col in NUMERICAL_COLUMNS:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="raise").astype("float64")

    if target_col in cleaned.columns:
        raw_labels = (
            cleaned[target_col]
            .astype(str)
            .str.strip()
            .str.rstrip(".")
            .str.lower()
        )
        cleaned["attack_type"] = raw_labels
        cleaned["attack_category"] = raw_labels.map(map_attack_family)
        cleaned["target"] = (raw_labels != "normal").astype("int64")

    return cleaned
