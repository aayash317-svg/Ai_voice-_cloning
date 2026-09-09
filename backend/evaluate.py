"""
Evaluation and Visualization Engine conforming to training_file.md Section 10 & 21.
Computes Accuracy, Precision, Recall, F1, ROC-AUC, EER, FAR, FRR,
and generates all required visual artifacts in results/.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from backend.config import RESULTS_DIR


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Compute Equal Error Rate (EER) where False Acceptance Rate (FAR) == False Rejection Rate (FRR).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr

    # Find the threshold where FAR (fpr) and FRR (fnr) intersect
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])

    return eer, eer_threshold, fpr, fnr


def evaluate_and_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path = RESULTS_DIR,
    report_title: str = "Voice Integrity Baseline Model Evaluation Report"
) -> Dict[str, Any]:
    """
    Compute full evaluation metrics and generate all required plots in results/.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0  # False alarm (genuine flagged as spoof)
    frr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0  # Missed spoof (spoof flagged as genuine)

    eer, eer_thresh, fpr_arr, fnr_arr = compute_eer(y_true, y_prob)

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "equal_error_rate_eer": round(eer, 4),
        "eer_threshold": round(eer_thresh, 4),
        "false_acceptance_rate_far": round(far, 4),
        "false_rejection_rate_frr": round(frr, 4),
        "confusion_matrix": {
            "true_genuine": int(tn),
            "false_spoof_alarm": int(fp),
            "false_genuine_missed": int(fn),
            "true_spoof_detected": int(tp)
        },
        "sample_counts": {
            "total_test_samples": len(y_true),
            "genuine_count": int(np.sum(y_true == 0)),
            "spoof_count": int(np.sum(y_true == 1))
        }
    }

    # 1. Save metrics.json
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # 2. Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Genuine (0)", "Spoof (1)"],
        yticklabels=["Genuine (0)", "Spoof (1)"]
    )
    plt.title("Confusion Matrix — Baseline Random Forest")
    plt.xlabel("Predicted Label")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    # 3. ROC Curve Plot
    plt.figure(figsize=(6, 5))
    plt.plot(fpr_arr, 1 - fnr_arr, color="#1f77b4", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Chance")
    plt.scatter([eer], [1 - eer], color="red", s=50, zorder=5, label=f"EER = {eer * 100:.2f}%")
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (1 - FRR)")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=200)
    plt.close()

    # 4. Genuine vs Spoof Probability Distribution Plot
    plt.figure(figsize=(7, 5))
    genuine_probs = y_prob[y_true == 0]
    spoof_probs = y_prob[y_true == 1]
    plt.hist(genuine_probs, bins=30, alpha=0.6, color="green", label="Genuine Audio (Ground Truth)")
    plt.hist(spoof_probs, bins=30, alpha=0.6, color="red", label="Spoof / Cloned Audio (Ground Truth)")
    plt.axvline(0.5, color="black", linestyle="--", label="Decision Threshold (0.5)")
    plt.xlabel("Predicted Spoof Probability")
    plt.ylabel("Sample Count")
    plt.title("Predicted Probability Distribution by Class")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "prob_distribution.png", dpi=200)
    plt.close()

    # 5. Training & Evaluation Text Report
    report_text = f"""============================================================
{report_title.upper()}
============================================================

1. EVALUATION METRICS SUMMARY:
   * Accuracy               : {acc * 100:.2f}%
   * Precision              : {prec * 100:.2f}%
   * Recall                 : {rec * 100:.2f}%
   * F1-Score               : {f1 * 100:.2f}%
   * ROC-AUC Score          : {roc_auc:.4f}
   * Equal Error Rate (EER) : {eer * 100:.2f}% (Threshold: {eer_thresh:.4f})
   * False Acceptance Rate  : {far * 100:.2f}% (Bona fide falsely rejected)
   * False Rejection Rate   : {frr * 100:.2f}% (Spoof missed as bona fide)

2. CONFUSION MATRIX:
   * True Genuine Speech Detected   : {tn}
   * False Spoof Alarms (Type I)   : {fp}
   * Missed Spoofs (Type II)        : {fn}
   * True Spoofed Audio Detected    : {tp}

3. TEST SET COMPOSITION:
   * Total Test Audio Files         : {len(y_true)}
   * Genuine Human Samples          : {np.sum(y_true == 0)} ({np.sum(y_true == 0)/len(y_true)*100:.1f}%)
   * Synthetic / Spoof Samples      : {np.sum(y_true == 1)} ({np.sum(y_true == 1)/len(y_true)*100:.1f}%)
============================================================
"""
    with open(output_dir / "training_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    return metrics


# Backward compatibility
evaluate_classifier = evaluate_and_plot
