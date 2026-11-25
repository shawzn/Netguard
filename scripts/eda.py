from pathlib import Path
import json

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import (
    load_nsl_kdd,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

TRAIN_PATH = Path("data/raw/KDDTrain+.txt")
TEST_PATH = Path("data/raw/KDDTest+.txt")

FIGURE_DIR = Path("reports/figures")
METRIC_DIR = Path("reports/metrics")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def save_class_distribution(train, test):
    frames = []

    for name, df in [("Train", train), ("Test", test)]:
        counts = (
            df["binary_label"]
            .value_counts(normalize=True)
            .sort_index()
            .rename({0: "Normal", 1: "Attack"})
        )

        for label, proportion in counts.items():
            frames.append({
                "dataset": name,
                "class": label,
                "proportion": proportion,
            })

    result = pd.DataFrame(frames)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=result,
        x="class",
        y="proportion",
        hue="dataset",
    )
    plt.title("Normal vs Attack Distribution")
    plt.ylabel("Proportion")
    plt.xlabel("Class")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "binary_class_distribution.png", dpi=150)
    plt.close()

    result.to_csv(
        METRIC_DIR / "binary_class_distribution.csv",
        index=False,
    )


def save_attack_distribution(train, test):
    frames = []

    for name, df in [("Train", train), ("Test", test)]:
        counts = (
            df[TARGET_COLUMN]
            .value_counts()
            .drop("normal", errors="ignore")
            .head(15)
        )

        for label, count in counts.items():
            frames.append({
                "dataset": name,
                "attack_type": label,
                "count": int(count),
            })

    result = pd.DataFrame(frames)

    plt.figure(figsize=(13, 7))
    sns.barplot(
        data=result,
        x="count",
        y="attack_type",
        hue="dataset",
    )
    plt.title("Top 15 Attack Types")
    plt.xlabel("Number of records")
    plt.ylabel("Attack type")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "attack_distribution.png", dpi=150)
    plt.close()

    result.to_csv(
        METRIC_DIR / "attack_distribution.csv",
        index=False,
    )


def save_categorical_distribution(train):
    categorical = ["protocol_type", "service", "flag"]

    for column in categorical:
        counts = (
            train[column]
            .value_counts()
            .head(15)
            .reset_index()
        )
        counts.columns = [column, "count"]

        plt.figure(figsize=(11, 6))
        sns.barplot(
            data=counts,
            x="count",
            y=column,
        )
        plt.title(f"Top Categories: {column}")
        plt.tight_layout()
        plt.savefig(
            FIGURE_DIR / f"categorical_{column}.png",
            dpi=150,
        )
        plt.close()

        counts.to_csv(
            METRIC_DIR / f"categorical_{column}.csv",
            index=False,
        )


def save_numeric_distributions(train):
    numeric_columns = train[FEATURE_COLUMNS].select_dtypes(
        include="number"
    ).columns.tolist()

    selected = [
        "duration",
        "src_bytes",
        "dst_bytes",
        "count",
        "srv_count",
        "serror_rate",
    ]

    selected = [
        column for column in selected
        if column in numeric_columns
    ]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(15, 9),
    )

    for column, ax in zip(selected, axes.flatten()):
        sns.histplot(
            data=train,
            x=column,
            hue="binary_label",
            bins=40,
            element="step",
            stat="density",
            common_norm=False,
            ax=ax,
        )
        ax.set_title(column)

    fig.suptitle("Selected Numerical Feature Distributions")
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "numeric_distributions.png",
        dpi=150,
    )
    plt.close(fig)


def save_correlation_heatmap(train):
    numeric = train[FEATURE_COLUMNS].select_dtypes(
        include="number"
    )

    correlation = numeric.corr()

    plt.figure(figsize=(18, 14))
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        linewidths=0.1,
    )
    plt.title("Numerical Feature Correlation")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "correlation_heatmap.png",
        dpi=150,
    )
    plt.close()

    correlation.to_csv(
        METRIC_DIR / "correlation_matrix.csv"
    )

    pairs = (
        correlation.abs()
        .where(
            pd.DataFrame(
                True,
                index=correlation.index,
                columns=correlation.columns,
            ).where(
                ~pd.DataFrame(
                    correlation.index.values[:, None]
                    == correlation.columns.values[None, :],
                    index=correlation.index,
                    columns=correlation.columns,
                ),
                False,
            )
        )
        .stack()
        .sort_values(ascending=False)
    )

    pairs = pairs[~pairs.index.map(lambda x: x[0] > x[1])]

    top_pairs = [
        {
            "feature_1": pair[0],
            "feature_2": pair[1],
            "absolute_correlation": float(value),
        }
        for pair, value in pairs.head(20).items()
    ]

    with open(
        METRIC_DIR / "top_correlations.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(top_pairs, file, indent=4)


def main():
    print("Loading NSL-KDD datasets...")

    train = load_nsl_kdd(TRAIN_PATH)
    test = load_nsl_kdd(TEST_PATH)

    print("Training shape:", train.shape)
    print("Testing shape:", test.shape)

    print("\nGenerating EDA outputs...")

    save_class_distribution(train, test)
    save_attack_distribution(train, test)
    save_categorical_distribution(train)
    save_numeric_distributions(train)
    save_correlation_heatmap(train)

    summary = {
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_count": len(FEATURE_COLUMNS),
        "categorical_features": [
            "protocol_type",
            "service",
            "flag",
        ],
        "numeric_feature_count": len(
            train[FEATURE_COLUMNS].select_dtypes(
                include="number"
            ).columns
        ),
        "train_binary_distribution": {
            str(k): int(v)
            for k, v in train["binary_label"].value_counts().items()
        },
        "test_binary_distribution": {
            str(k): int(v)
            for k, v in test["binary_label"].value_counts().items()
        },
    }

    with open(
        METRIC_DIR / "eda_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\nEDA completed successfully.")
    print("Figures:", FIGURE_DIR)
    print("Metrics:", METRIC_DIR)


if __name__ == "__main__":
    main()
