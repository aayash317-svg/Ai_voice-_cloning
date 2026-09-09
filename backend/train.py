"""
End-to-End Training Pipeline conforming to training_file.md Section 9 & 25.
Loads dataset, extracts features, trains Random Forest baseline,
evaluates on test split, and generates all visual and textual reports.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    ASVSPOOF_ADJUSTED_WAV_DIR,
    BASELINE_MODEL_PATH,
    RESULTS_DIR,
    FEATURES_DIR,
    SAMPLE_RATE
)
from backend.dataset_loader import DatasetLoader, AudioSample
from backend.audio_preprocessing import AudioPreprocessor
from backend.features import FeatureExtractor
from backend.classifier import AntiSpoofClassifier
from backend.evaluate import evaluate_and_plot


def plot_class_distribution(splits_stats: dict, output_path: Path):
    """Plot dataset class distributions across train, dev, and test sets."""
    plt.figure(figsize=(8, 5))
    categories = list(splits_stats.keys())
    genuine_counts = [splits_stats[k]["genuine"] for k in categories]
    spoof_counts = [splits_stats[k]["spoof"] for k in categories]

    x = np.arange(len(categories))
    width = 0.35

    plt.bar(x - width/2, genuine_counts, width, label="Genuine (Human)", color="#2ca02c")
    plt.bar(x + width/2, spoof_counts, width, label="Spoof (Synthetic)", color="#d62728")

    plt.xlabel("Dataset Split")
    plt.ylabel("Audio Sample Count")
    plt.title("Dataset Class Distribution by Split (Stratified)")
    plt.xticks(x, [c.capitalize() for c in categories])
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def extract_features_matrix(
    samples: List[AudioSample],
    preprocessor: AudioPreprocessor,
    extractor: FeatureExtractor,
    cache_path: Optional[Path] = None,
    max_count: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract and validate feature matrix for a list of AudioSamples."""
    if cache_path and cache_path.exists():
        print(f"Loading cached features from: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], list(data["feature_names"])

    target_samples = samples[:max_count] if max_count else samples
    feature_list = []
    labels = []
    feature_names = None

    print(f"Extracting features from {len(target_samples)} audio files...")
    for sample in tqdm(target_samples, desc="Processing Audio"):
        try:
            audio = preprocessor.preprocess_file(sample.file_path)
            feat_dict = extractor.extract_all(audio)

            if feature_names is None:
                feature_names = sorted(feat_dict.keys())

            vec = extractor.to_vector(feat_dict, feature_names)
            feature_list.append(vec)
            labels.append(sample.label)
        except Exception as e:
            print(f"[!] Error processing {sample.file_path.name}: {e}")

    X = np.array(feature_list, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    # Feature validation
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y, feature_names=feature_names)
        print(f"Cached features saved to: {cache_path}")

    return X, y, feature_names


def run_training_pipeline(
    data_dir: Optional[Path] = None,
    max_train: Optional[int] = None,
    max_dev: Optional[int] = None,
    max_test: Optional[int] = None,
    use_cache: bool = True
):
    """Execute complete dataset loading, training, evaluation, and reporting pipeline."""
    total_start = time.time()
    print("=" * 70)
    print("VOICE INTEGRITY VERIFICATION — LIGHTWEIGHT BASELINE TRAINING")
    print("=" * 70)

    # 1. Dataset Loading & Stratified Splitting
    print("\n[Step 1/5] Loading dataset & creating train/dev/test splits...")
    loader = DatasetLoader(data_dir=data_dir)
    loader.load_samples()
    splits = loader.create_splits(train_ratio=0.70, dev_ratio=0.15, test_ratio=0.15)
    split_stats = loader.get_split_stats()

    print(f"Total discovered samples: {len(loader.samples)}")
    for split_name, stats in split_stats.items():
        print(f"  * {split_name.capitalize():5s}: {stats['total']:5d} total | {stats['genuine']:4d} genuine ({stats['genuine_pct']}%) | {stats['spoof']:4d} spoof ({stats['spoof_pct']}%)")

    # Plot class distribution
    plot_class_distribution(split_stats, RESULTS_DIR / "class_distribution.png")

    # 2. Feature Extraction
    print("\n[Step 2/5] Feature Extraction (MFCCs, Spectral, Prosody, Waveform moments)...")
    preprocessor = AudioPreprocessor(target_sr=SAMPLE_RATE)
    extractor = FeatureExtractor(sample_rate=SAMPLE_RATE)

    train_cache = FEATURES_DIR / f"features_train_{max_train or 'full'}.npz" if use_cache else None
    dev_cache = FEATURES_DIR / f"features_dev_{max_dev or 'full'}.npz" if use_cache else None
    test_cache = FEATURES_DIR / f"features_test_{max_test or 'full'}.npz" if use_cache else None

    X_train, y_train, feature_names = extract_features_matrix(splits["train"], preprocessor, extractor, train_cache, max_train)
    X_dev, y_dev, _ = extract_features_matrix(splits["dev"], preprocessor, extractor, dev_cache, max_dev)
    X_test, y_test, _ = extract_features_matrix(splits["test"], preprocessor, extractor, test_cache, max_test)

    print(f"\n[+] Feature Matrices Validated:")
    print(f"    X_train shape: {X_train.shape} (Labels: {np.bincount(y_train)})")
    print(f"    X_dev shape  : {X_dev.shape} (Labels: {np.bincount(y_dev)})")
    print(f"    X_test shape : {X_test.shape} (Labels: {np.bincount(y_test)})")
    print(f"    Feature count: {len(feature_names)} dimensions")

    # 3. Model Training
    print("\n[Step 3/5] Training Random Forest Baseline (n_estimators=200, balanced)...")
    train_start = time.time()
    classifier = AntiSpoofClassifier(model_type="random_forest", model_params={"n_estimators": 200, "random_state": 42})
    classifier.train(X_train, y_train, feature_names=feature_names)
    train_time = round(time.time() - train_start, 2)
    print(f"[+] Training completed in {train_time} seconds.")

    # Save trained model
    classifier.save(BASELINE_MODEL_PATH)
    print(f"[+] Serialized model saved to: {BASELINE_MODEL_PATH}")

    # 4. Evaluation on Dev & Test Set
    print("\n[Step 4/5] Evaluating model on held-out test split (No Data Leakage)...")
    test_start = time.time()
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:, 1]
    test_time = round(time.time() - test_start, 2)

    metrics = evaluate_and_plot(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        output_dir=RESULTS_DIR
    )

    total_time = round(time.time() - total_start, 2)

    # 5. Final Report
    print("\n" + "=" * 70)
    print("FINAL FIRST TASK REPORT — BASELINE EXPERIMENT RESULTS")
    print("=" * 70)
    print(f"1.  Dataset Size               : {len(loader.samples)} audio files ({len(X_train)} train, {len(X_dev)} dev, {len(X_test)} test)")
    print(f"2.  Genuine Samples Count      : {sum(1 for s in loader.samples if s.label == 0)}")
    print(f"3.  Spoof Samples Count        : {sum(1 for s in loader.samples if s.label == 1)}")
    print(f"4.  Feature Count              : {len(feature_names)} features per sample")
    print(f"5.  Training Time              : {train_time} seconds")
    print(f"6.  Test Time                  : {test_time} seconds")
    print(f"7.  Accuracy                   : {metrics['accuracy'] * 100:.2f}%")
    print(f"8.  Precision                  : {metrics['precision'] * 100:.2f}%")
    print(f"9.  Recall                     : {metrics['recall'] * 100:.2f}%")
    print(f"10. F1-Score                   : {metrics['f1_score'] * 100:.2f}%")
    print(f"11. ROC-AUC                    : {metrics['roc_auc']:.4f}")
    print(f"12. Equal Error Rate (EER)     : {metrics['equal_error_rate_eer'] * 100:.2f}% (Threshold: {metrics['eer_threshold']:.4f})")
    print(f"13. False Acceptance Rate (FAR): {metrics['false_acceptance_rate_far'] * 100:.2f}%")
    print(f"14. False Rejection Rate (FRR) : {metrics['false_rejection_rate_frr'] * 100:.2f}%")
    print(f"15. Confusion Matrix           : True Genuine: {metrics['confusion_matrix']['true_genuine']}, False Spoof: {metrics['confusion_matrix']['false_spoof_alarm']}, Missed Spoof: {metrics['confusion_matrix']['false_genuine_missed']}, True Spoof: {metrics['confusion_matrix']['true_spoof_detected']}")
    print(f"16. Artifacts Directory        : {RESULTS_DIR}")
    print(f"17. Total Pipeline Duration    : {total_time}s")
    print("=" * 70 + "\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Voice Integrity Baseline Model")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory path")
    parser.add_argument("--max-train", type=int, default=None, help="Limit training samples for testing")
    parser.add_argument("--max-dev", type=int, default=None, help="Limit dev samples for testing")
    parser.add_argument("--max-test", type=int, default=None, help="Limit test samples for testing")
    parser.add_argument("--no-cache", action="store_true", help="Do not use cached features")

    args = parser.parse_args()
    run_training_pipeline(
        data_dir=Path(args.data_dir) if args.data_dir else None,
        max_train=args.max_train,
        max_dev=args.max_dev,
        max_test=args.max_test,
        use_cache=not args.no_cache
    )


if __name__ == "__main__":
    main()
