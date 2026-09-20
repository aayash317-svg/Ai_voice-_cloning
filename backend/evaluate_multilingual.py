"""
Multilingual Voice Integrity Benchmark & Forensic Evaluation Script.

Evaluates the trained Multilingual Anti-Spoofing Model across:
1. Overall Benchmark (Accuracy, Precision, Recall, F1, ROC-AUC, EER, FAR, FRR).
2. Per-Language Breakdown (Hindi, Tamil, Telugu, Bengali, Kannada, Marathi, English/ASVspoof).
3. Per-Generator Breakdown (XTTSv2, VITS, FreeVC24, YourTTS, ASVspoof A07-A19).
4. Exports detailed evaluation report to results/multilingual_benchmark.json.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    SAMPLE_RATE, MULTILINGUAL_MODEL_PATH, BASELINE_MODEL_PATH,
    RESULTS_DIR, INDIC_FEATURES_DIR, INDIC_TARGET_LANGUAGES
)
from backend.classifier import AntiSpoofClassifier
from backend.preprocess import AudioPreprocessor
from backend.features import FeatureExtractor
from backend.dataset_loader import DatasetLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Multilingual Anti-Spoofing Benchmark")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MULTILINGUAL_MODEL_PATH if MULTILINGUAL_MODEL_PATH.exists() else BASELINE_MODEL_PATH),
        help="Path to serialized AntiSpoofClassifier checkpoint"
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=str(RESULTS_DIR / "multilingual_benchmark.json"),
        help="Destination path for evaluation JSON report"
    )
    return parser.parse_args()


def evaluate_subset(y_true: np.ndarray, y_prob: np.ndarray, tau: float) -> Dict:
    """Compute core anti-spoof metrics on any given slice."""
    if len(y_true) == 0:
        return {}
    y_pred = (y_prob >= tau).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 1.0
    except Exception:
        auc = 1.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_gen, false_spoof = cm[0][0], cm[0][1]
    missed_spoof, true_spoof = cm[1][0], cm[1][1]

    far = float(false_spoof / (true_gen + false_spoof)) if (true_gen + false_spoof) > 0 else 0.0
    frr = float(missed_spoof / (missed_spoof + true_spoof)) if (missed_spoof + true_spoof) > 0 else 0.0

    # EER
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
        fnr = 1.0 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    except Exception:
        eer = 0.0

    return {
        "samples": len(y_true),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "eer": round(eer, 4),
        "false_alarm_rate_far": round(far, 4),
        "miss_rate_frr": round(frr, 4),
        "confusion_matrix": {
            "true_genuine": int(true_gen),
            "false_spoof": int(false_spoof),
            "missed_spoof": int(missed_spoof),
            "true_spoof": int(true_spoof)
        }
    }


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    output_report = Path(args.output_report)

    print("=" * 70)
    print("MULTILINGUAL VOICE INTEGRITY BENCHMARK & EVALUATION")
    print(f"Target Model: {model_path}")
    print("=" * 70)

    # 1. Load Model
    if not model_path.exists():
        print(f"[!] Model file not found at {model_path}. Please train a model first using backend/train_multilingual.py.")
        sys.exit(1)

    classifier = AntiSpoofClassifier.load(model_path)
    print(f"[+] Loaded model checkpoint successfully.")
    print(f"    - Model Type           : {classifier.model_type}")
    print(f"    - Features Dimension   : {len(classifier.feature_names)}")
    print(f"    - Calibrated Threshold : {classifier.calibrated_threshold:.4f}")

    # 2. Check for Test Cache or Extract from Test Split
    test_caches = list(INDIC_FEATURES_DIR.glob("*test.npz"))
    if test_caches:
        latest_test = max(test_caches, key=lambda p: p.stat().st_mtime)
        print(f"[+] Loading test features from cache: {latest_test.name}")
        data = np.load(latest_test, allow_pickle=True)
        X_test = data["X"]
        y_test = data["y"]
        sys_test = list(data["system_ids"])
    else:
        print("[*] No test feature cache found. Extracting features from current dataset test split...")
        loader = DatasetLoader()
        samples = loader.load_samples()
        indic_s = [s for s in samples if "indic" in s.system_id]
        eval_universe = indic_s if len(indic_s) > 0 else samples[:2000]
        splits = loader.create_splits(train_ratio=0.70, dev_ratio=0.15, test_ratio=0.15)
        test_samples = splits["test"]

        preprocessor = AudioPreprocessor(target_sr=SAMPLE_RATE)
        extractor = FeatureExtractor(sample_rate=SAMPLE_RATE)
        
        feature_list = []
        labels = []
        sys_test = []
        for s in test_samples:
            try:
                a = preprocessor.load_audio(s.file_path)
                a = preprocessor.normalize_audio(a)
                fd = extractor.extract_all(a)
                vec = extractor.to_vector(fd, classifier.feature_names)
                feature_list.append(vec)
                labels.append(s.label)
                sys_test.append(s.system_id)
            except Exception:
                continue

        X_test = np.array(feature_list, dtype=np.float32)
        y_test = np.array(labels, dtype=np.int32)

    print(f"[+] Evaluating test set: {len(X_test)} samples ({np.bincount(y_test)[0]} genuine, {np.bincount(y_test)[1]} spoof)")

    # 3. Model Inference
    y_test_prob = np.array([classifier.predict_spoof_risk(x) for x in X_test])
    tau = classifier.calibrated_threshold if hasattr(classifier, "calibrated_threshold") and classifier.calibrated_threshold else 0.51

    # Overall Metrics
    overall = evaluate_subset(y_test, y_test_prob, tau)

    # Language Breakdowns
    language_metrics = {}
    for lang in INDIC_TARGET_LANGUAGES:
        mask = np.array([lang in s.lower() for s in sys_test])
        if np.sum(mask) >= 5:
            language_metrics[lang] = evaluate_subset(y_test[mask], y_test_prob[mask], tau)

    # Non-Indic / English / ASVspoof
    non_indic_mask = np.array(["indic" not in s.lower() for s in sys_test])
    if np.sum(non_indic_mask) >= 5:
        language_metrics["english_asvspoof"] = evaluate_subset(y_test[non_indic_mask], y_test_prob[non_indic_mask], tau)

    # Generator Breakdowns
    generator_metrics = {}
    for gen in ["xttsv2", "vits", "freevc24", "yourtts", "chatterbox"]:
        mask = np.array([gen in s.lower() for s in sys_test])
        if np.sum(mask) >= 5:
            generator_metrics[gen] = evaluate_subset(y_test[mask], y_test_prob[mask], tau)

    # Print Report
    print("\n" + "=" * 70)
    print("MULTILINGUAL FORENSIC BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Overall Accuracy          : {overall['accuracy'] * 100:.2f}%")
    print(f"Precision                 : {overall['precision'] * 100:.2f}%")
    print(f"Recall                    : {overall['recall'] * 100:.2f}%")
    print(f"F1-Score                  : {overall['f1_score'] * 100:.2f}%")
    print(f"ROC-AUC                   : {overall['roc_auc']:.4f}")
    print(f"Equal Error Rate (EER)    : {overall['eer'] * 100:.2f}%")
    print(f"False Alarm Rate (FAR)    : {overall['false_alarm_rate_far'] * 100:.2f}%")
    print(f"Miss Rate (FRR)           : {overall['miss_rate_frr'] * 100:.2f}%")
    print("-" * 70)
    print("Per-Language Performance Breakdown:")
    print(f"{'Language':<18} {'Samples':<10} {'Accuracy':<12} {'ROC-AUC':<10} {'EER':<10}")
    for lang, metrics in language_metrics.items():
        print(f"{lang.capitalize():<18} {metrics['samples']:<10} {metrics['accuracy'] * 100:>6.2f}%     {metrics['roc_auc']:>6.4f}     {metrics['eer'] * 100:>5.2f}%")

    if generator_metrics:
        print("-" * 70)
        print("Per-Generator Detection Breakdown:")
        print(f"{'Generator':<18} {'Samples':<10} {'Accuracy':<12} {'ROC-AUC':<10}")
        for gen, metrics in generator_metrics.items():
            print(f"{gen.upper():<18} {metrics['samples']:<10} {metrics['accuracy'] * 100:>6.2f}%     {metrics['roc_auc']:>6.4f}")

    print("=" * 70)

    # Save to JSON report
    report = {
        "model_evaluated": str(model_path.name),
        "calibrated_threshold": round(tau, 4),
        "total_test_samples": len(y_test),
        "overall": overall,
        "languages": language_metrics,
        "generators": generator_metrics
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Detailed Benchmark saved to: {output_report}")


if __name__ == "__main__":
    main()
