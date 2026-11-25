from pathlib import Path
import pandas as pd
from netguard.predict import NetGuardPredictor


def test_end_to_end_20_row_scanner() -> None:
    sample_path = Path("data/processed/sample_scanner_20rows.csv")
    assert sample_path.exists(), "Sample 20-row CSV missing."

    df = pd.read_csv(sample_path)
    predictor = NetGuardPredictor(
        models_dir="artifacts/models",
        model_variant="xgb_smotenc",
    )
    summary = predictor.predict_dataframe(df, threshold=0.5, max_rows=20)

    assert summary.total_flows == 20
    assert summary.predicted_attacks + summary.predicted_normal == 20
    assert {"flow_id", "prediction", "attack_probability", "risk_level"}.issubset(
        summary.results_df.columns
    )
