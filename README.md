# NetGuard — Network Intrusion Detection System (NSL-KDD)

NetGuard is an end-to-end Machine Learning Network Intrusion Detection System (NIDS) built on the **NSL-KDD** benchmark dataset. It evaluates the impact of **SMOTENC** on **XGBoost** classification under strict leak-free experimental controls, paired with a **Streamlit** operational dashboard and 20-row CSV packet scanner.

## 1. Repository Structure

```text
NetGuard/
├── app/
│   └── streamlit_app.py          # Multi-tab Streamlit dashboard & 20-row CSV scanner
├── artifacts/
│   └── models/
│       ├── preprocessor.joblib   # Fitted leak-free NetGuardPreprocessor
│       ├── xgb_baseline.joblib   # Experiment A model artifact
│       └── xgb_smotenc.joblib    # Experiment B model artifact
├── data/
│   ├── raw/                      # KDDTrain+.txt and KDDTest+.txt
│   └── processed/                # sample_scanner_20rows.csv (demo batch)
├── reports/
│   ├── figures/                  # EDA & evaluation plots
│   ├── metrics/                  # eda_summary.json, top_correlations.json, experiment_results.json
│   └── NetGuard_Presentation.pptx
├── scripts/
│   ├── eda.py                    # Exploratory data analysis
│   ├── train.py                  # Leak-free split, SMOTENC resampling, training & evaluation
│   ├── generate_sample_csv.py    # Creates 20-row stratified test sample for Streamlit
│   └── build_ppt.py              # Generates presentation deck from experiment_results.json
└── src/netguard/
    ├── data_loader.py            # Schema & label validation for NSL-KDD
    ├── cleaning.py               # Type normalization & 5-family attack mapping
    ├── preprocessing.py          # Two-stage Ordinal -> RobustScaler + SMOTENC -> OHE pipeline
    ├── model.py                  # Controlled XGBoost classifier factory
    ├── evaluation.py             # Binary NIDS metrics & per-attack-family breakdown
    └── predict.py                # Inference engine for batch CSV scanning
```

## 2. Leak-Free Methodology

1. **External Test Isolation:** `KDDTest+` (`22,544` rows) is strictly isolated and never used for fitting transformers, resampling, or early stopping.
2. **Stratified Split:** `KDDTrain+` (`125,973` rows) is split 80/20 into Training (`100,778` rows) and Validation (`25,195` rows) prior to any transformation.
3. **Categorical-Safe Resampling:**
   - **Stage 1 (Fit on `X_train` only):** `OrdinalEncoder` maps `protocol_type`, `service`, and `flag` to integer codes while `RobustScaler` scales the 38 numerical features without clipping extreme attack spikes.
   - **Resampling (`X_train` only):** `SMOTENC` synthesizes minority attack instances (`46,904` -> `53,874`) while preserving discrete categorical integrity.
   - **Stage 2 (Fit on `X_train` only):** `OneHotEncoder(handle_unknown="ignore")` expands the 3 categorical features into `121` total model features.

## 3. Experimental Results (`KDDTest+`)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A: XGBoost Baseline** | 78.96% | 96.97% | 65.07% | 0.7788 | 0.9661 | 0.9667 | 2.69% |
| **Exp B: XGBoost + SMOTENC** | **79.40%** | **96.98%** | **65.85%** | **0.7844** | **0.9667** | **0.9670** | 2.71% |

## 4. Quickstart

```bash
# 1. Train & evaluate Experiment A vs Experiment B
PYTHONPATH=src python -m scripts.train

# 2. Generate 20-row CSV scanner test file
PYTHONPATH=src python -m scripts.generate_sample_csv

# 3. Launch Streamlit Dashboard
streamlit run app/streamlit_app.py --server.port 8501
```
# Netguard
