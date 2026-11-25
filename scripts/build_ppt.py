from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt


def add_bullet_slide(prs: Presentation, title: str, bullets: list[str]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = title
    tf = slide.placeholders[1].text_frame
    tf.clear()
    for idx, line in enumerate(bullets):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(18)
        p.space_after = Pt(10)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_path = project_root / "reports" / "metrics" / "experiment_results.json"
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    exp_a = results["experiments"]["xgb_baseline"]["external_test"]
    exp_b = results["experiments"]["xgb_smotenc"]["external_test"]

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Slide 1: Title
    title_slide = prs.slides.add_slide(prs.slide_layouts[0])
    title_slide.shapes.title.text = "NetGuard: Network Intrusion Detection System"
    title_slide.placeholders[1].text = (
        "Controlled Evaluation of XGBoost Baseline vs. XGBoost + SMOTENC on NSL-KDD\n"
        "End-to-End ML Pipeline & Streamlit Threat Scanner"
    )

    # Slide 2: Problem & Objectives
    add_bullet_slide(
        prs,
        "1. Background & Project Objectives",
        [
            "Modern network traffic exhibits heavy class skew and unseen attack signatures.",
            "Objective 1: Build a leak-free ML intrusion detection pipeline on NSL-KDD.",
            "Objective 2: Quantify the impact of SMOTENC resampling on XGBoost detection.",
            "Objective 3: Evaluate generalization on unseen external traffic (KDDTest+).",
            "Objective 4: Deploy an interactive Streamlit dashboard & 20-row CSV scanner.",
        ],
    )

    # Slide 3: Dataset & EDA Findings
    add_bullet_slide(
        prs,
        "2. Dataset Characteristics & EDA Findings",
        [
            "KDDTrain+: 125,973 rows (53.46% Normal vs 46.54% Attack).",
            "KDDTest+: 22,544 rows (43.07% Normal vs 56.93% Attack) — includes novel attacks.",
            "41 raw features: 3 categorical (protocol_type, service, flag) + 38 numerical.",
            "Heavy right-skewness in duration, src_bytes, dst_bytes -> outliers preserved.",
            "3 high-correlation clusters (|r| > 0.90) handled natively by tree-based XGBoost.",
        ],
    )

    # Slide 4: Leak-Free Architecture
    add_bullet_slide(
        prs,
        "3. Leak-Free Preprocessing & SMOTENC Architecture",
        [
            "Strict isolation: KDDTest+ is never touched during fitting or resampling.",
            "Stratified 80/20 split on KDDTrain+ -> 100,778 Train / 25,195 Validation.",
            "Stage 1 (Fit on Train): OrdinalEncoder (cat) + RobustScaler (num).",
            "SMOTENC (Train only): Balances Attack class from 46,904 to 53,874 samples.",
            "Stage 2 (Fit on Train): OneHotEncoder expands features to 121 dimensions.",
        ],
    )

    # Slide 5: Comparison Table Slide
    table_slide = prs.slides.add_slide(prs.slide_layouts[5])
    table_slide.shapes.title.text = "4. External Test Results: Baseline vs. SMOTENC (KDDTest+)"
    rows, cols = 8, 4
    table_shape = table_slide.shapes.add_table(
        rows, cols, Inches(1.0), Inches(1.6), Inches(11.3), Inches(4.8)
    )
    table = table_shape.table

    headers = ["Metric", "Exp A: XGBoost Baseline", "Exp B: XGBoost + SMOTENC", "Delta (B - A)"]
    for col_idx, h in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(30, 58, 138)
        cell.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
        cell.text_frame.paragraphs[0].font.bold = True

    metric_rows = [
        ("Accuracy", f"{exp_a['accuracy']*100:.2f}%", f"{exp_b['accuracy']*100:.2f}%", f"{(exp_b['accuracy']-exp_a['accuracy'])*100:+.2f}%"),
        ("Precision", f"{exp_a['precision']*100:.2f}%", f"{exp_b['precision']*100:.2f}%", f"{(exp_b['precision']-exp_a['precision'])*100:+.2f}%"),
        ("Recall (TPR)", f"{exp_a['recall']*100:.2f}%", f"{exp_b['recall']*100:.2f}%", f"{(exp_b['recall']-exp_a['recall'])*100:+.2f}%"),
        ("F1-Score", f"{exp_a['f1_score']:.4f}", f"{exp_b['f1_score']:.4f}", f"{exp_b['f1_score']-exp_a['f1_score']:+.4f}"),
        ("ROC-AUC", f"{exp_a['roc_auc']:.4f}", f"{exp_b['roc_auc']:.4f}", f"{exp_b['roc_auc']-exp_a['roc_auc']:+.4f}"),
        ("PR-AUC", f"{exp_a['pr_auc']:.4f}", f"{exp_b['pr_auc']:.4f}", f"{exp_b['pr_auc']-exp_a['pr_auc']:+.4f}"),
        ("False Positive Rate", f"{exp_a['fpr']*100:.2f}%", f"{exp_b['fpr']*100:.2f}%", f"{(exp_b['fpr']-exp_a['fpr'])*100:+.2f}%"),
    ]
    for r_idx, row_vals in enumerate(metric_rows, start=1):
        for c_idx, val in enumerate(row_vals):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            cell.text_frame.paragraphs[0].font.size = Pt(16)

    # Slide 6: Key Analytical Takeaways
    cm_a = exp_a["confusion_matrix"]
    cm_b = exp_b["confusion_matrix"]
    add_bullet_slide(
        prs,
        "5. Key Analytical Takeaways",
        [
            f"SMOTENC caught +101 additional attacks on KDDTest+ (TP: {cm_a['tp']:,} -> {cm_b['tp']:,}).",
            f"Precision remained ~97.0% with minimal FPR shift ({exp_a['fpr']*100:.2f}% -> {exp_b['fpr']*100:.2f}%).",
            "High ROC-AUC (0.9667) vs Recall (65.85%) indicates unseen R2L/U2R attacks score below 0.50.",
            "Interactive threshold tuning in Streamlit allows SOC analysts to trade FPR for higher Recall.",
        ],
    )

    # Slide 7: Streamlit System & Demo
    add_bullet_slide(
        prs,
        "6. Streamlit Dashboard & Live 20-Row Scanner",
        [
            "Tab 1 — 20-Row CSV Intrusion Scanner: Batch threat scoring, risk badges, and CSV export.",
            "Tab 2 — Experiment A vs B Evaluation: Confusion matrices, family breakdown, and top features.",
            "Tab 3 — Dataset & EDA Insights: Partition drift and high-correlation diagnostics.",
            "Live Demo Ready: Uses serialized preprocessor.joblib + xgb_smotenc.joblib artifacts.",
        ],
    )

    out_ppt = project_root / "reports" / "NetGuard_Presentation.pptx"
    prs.save(out_ppt)
    print(f"Successfully generated presentation: {out_ppt}")


if __name__ == "__main__":
    main()
