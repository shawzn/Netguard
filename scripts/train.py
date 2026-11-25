from __future__ import annotations

import json
from pathlib import Path

from netguard.cleaning import clean_nsl_kdd
try:
    from netguard.data_loader import load_nsl_kdd
except ImportError:
    from data_loader import load_nsl_kdd  # type: ignore[no-redef]
from netguard.evaluation import evaluate_binary_classifier
from netguard.model import save_model_artifact, train_experiment
from netguard.preprocessing import prepare_nsl_kdd_splits


def resolve_dataset_path(raw_dir: Path, stem: str) -> Path:
    """Locate KDDTrain+ or KDDTest+ regardless of .txt or .csv extension."""
    candidates = [
        raw_dir / f"{stem}.txt",
        raw_dir / f"{stem}.csv",
        raw_dir / stem,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {stem} in {raw_dir}. Checked: {[c.name for c in candidates]}"
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    models_dir = project_root / "artifacts" / "models"
    metrics_dir = project_root / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_path = resolve_dataset_path(raw_dir, "KDDTrain+")
    test_path = resolve_dataset_path(raw_dir, "KDDTest+")

    print(f"[1/5] Loading datasets:\n  - Train: {train_path}\n  - Test : {test_path}")
    raw_train = load_nsl_kdd(train_path)
    raw_test = load_nsl_kdd(test_path)

    print("[2/5] Cleaning schema and normalizing labels...")
    train_df = clean_nsl_kdd(raw_train)
    test_df = clean_nsl_kdd(raw_test)

    print("[3/5] Running leak-free split (80/20) and preprocessing + SMOTENC...")
    datasets, _ = prepare_nsl_kdd_splits(
        train_df=train_df,
        test_df=test_df,
        val_size=0.2,
        random_state=42,
        preprocessor_save_path=models_dir / "preprocessor.joblib",
    )
    print(
        f"  - Baseline train shape : {datasets.X_train_baseline.shape} "
        f"(Normal={(datasets.y_train_baseline == 0).sum()}, Attack={(datasets.y_train_baseline == 1).sum()})"
    )
    print(
        f"  - SMOTENC train shape  : {datasets.X_train_smotenc.shape} "
        f"(Normal={(datasets.y_train_smotenc == 0).sum()}, Attack={(datasets.y_train_smotenc == 1).sum()})"
    )
    print(f"  - Validation shape     : {datasets.X_val.shape}")
    print(f"  - External test shape  : {datasets.X_test.shape}")

    print("[4/5] Training Experiment A (XGBoost Baseline) & Experiment B (XGBoost + SMOTENC)...")
    exp_a = train_experiment(
        experiment_name="xgb_baseline",
        X_train=datasets.X_train_baseline,
        y_train=datasets.y_train_baseline,
        X_val=datasets.X_val,
        y_val=datasets.y_val,
        feature_names=datasets.feature_names,
        random_state=42,
    )
    exp_b = train_experiment(
        experiment_name="xgb_smotenc",
        X_train=datasets.X_train_smotenc,
        y_train=datasets.y_train_smotenc,
        X_val=datasets.X_val,
        y_val=datasets.y_val,
        feature_names=datasets.feature_names,
        random_state=42,
    )

    save_model_artifact(exp_a.model, models_dir / "xgb_baseline.joblib")
    save_model_artifact(exp_b.model, models_dir / "xgb_smotenc.joblib")

    print("[5/5] Evaluating on Validation and External KDDTest+...")
    results = {
        "dataset_shapes": {
            "train_baseline": list(datasets.X_train_baseline.shape),
            "train_smotenc": list(datasets.X_train_smotenc.shape),
            "validation": list(datasets.X_val.shape),
            "external_test": list(datasets.X_test.shape),
        },
        "experiments": {
            "xgb_baseline": {
                "best_iteration": exp_a.best_iteration,
                "validation": evaluate_binary_classifier(
                    exp_a.model, datasets.X_val, datasets.y_val
                ),
                "external_test": evaluate_binary_classifier(
                    exp_a.model,
                    datasets.X_test,
                    datasets.y_test,
                    attack_categories=test_df["attack_category"],
                ),
                "top_20_features": exp_a.feature_importances[:20],
            },
            "xgb_smotenc": {
                "best_iteration": exp_b.best_iteration,
                "validation": evaluate_binary_classifier(
                    exp_b.model, datasets.X_val, datasets.y_val
                ),
                "external_test": evaluate_binary_classifier(
                    exp_b.model,
                    datasets.X_test,
                    datasets.y_test,
                    attack_categories=test_df["attack_category"],
                ),
                "top_20_features": exp_b.feature_importances[:20],
            },
        },
    }

    report_path = metrics_dir / "experiment_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved models to {models_dir} and metrics to {report_path}\n")
    print("=== EXTERNAL KDDTest+ SUMMARY ===")
    for exp_key in ("xgb_baseline", "xgb_smotenc"):
        m = results["experiments"][exp_key]["external_test"]
        cm = m["confusion_matrix"]
        print(
            f"{exp_key:14s} | Acc={m['accuracy']:.4f} | Prec={m['precision']:.4f} | "
            f"Rec={m['recall']:.4f} | F1={m['f1_score']:.4f} | ROC-AUC={m['roc_auc']:.4f} | "
            f"PR-AUC={m['pr_auc']:.4f} | FPR={m['fpr']:.4f} | CM(TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']})"
        )


if __name__ == "__main__":
    main()
