from src.data_loader import (
    load_nsl_kdd,
    split_features_target,
    generate_quality_report,
)


DATASETS = {
    "train": "data/raw/KDDTrain+.txt",
    "test": "data/raw/KDDTest+.txt",
}


def main():
    for name, path in DATASETS.items():
        print(f"\n{'=' * 50}")
        print(f"DATASET: {name.upper()}")
        print("=" * 50)

        df = load_nsl_kdd(path)
        X, y = split_features_target(df)
        report = generate_quality_report(df)

        print("Dataset shape:", df.shape)
        print("Feature shape:", X.shape)
        print("Target shape:", y.shape)

        print("\nQuality report:")
        for key, value in report.items():
            if key != "original_label_distribution":
                print(f"{key}: {value}")

        print("\nBinary label distribution:")
        print(y.value_counts().sort_index())

        print("\nOriginal labels (top 10):")
        print(df["label"].value_counts().head(10))

        assert X.shape[1] == 41
        assert len(X) == len(y)
        assert "difficulty" not in X.columns
        assert "label" not in X.columns
        assert set(y.unique()).issubset({0, 1})

    print("\nAll dataset loader tests passed.")


if __name__ == "__main__":
    main()
