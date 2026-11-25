from __future__ import annotations

from pathlib import Path

import pandas as pd

from netguard.cleaning import FEATURE_COLUMNS, clean_nsl_kdd
try:
    from netguard.data_loader import load_nsl_kdd
except ImportError:
    from data_loader import load_nsl_kdd  # type: ignore[no-redef]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    test_path = project_root / "data" / "raw" / "KDDTest+.txt"
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_test = load_nsl_kdd(test_path)
    df = clean_nsl_kdd(raw_test)

    # Stratified 20-row sample covering Normal + major attack families
    quotas = {
        "Normal": 8,
        "DoS": 5,
        "Probe": 3,
        "R2L": 3,
        "U2R": 1,
    }
    samples: list[pd.DataFrame] = []
    for family, n in quotas.items():
        subset = df[df["attack_category"] == family]
        take = min(n, len(subset))
        if take > 0:
            samples.append(subset.sample(n=take, random_state=42))

    sample_20 = pd.concat(samples, ignore_index=True).sample(frac=1.0, random_state=42)

    # Save version with ground-truth label (for live demo verification) & pure 41-feature version
    demo_cols = FEATURE_COLUMNS + ["label"]
    sample_20[demo_cols].to_csv(out_dir / "sample_scanner_20rows.csv", index=False)
    sample_20[FEATURE_COLUMNS].to_csv(out_dir / "sample_scanner_20rows_unlabeled.csv", index=False)

    print(f"Generated {len(sample_20)}-row sample CSVs in {out_dir}:")
    print(sample_20["attack_category"].value_counts().to_dict())


if __name__ == "__main__":
    main()
