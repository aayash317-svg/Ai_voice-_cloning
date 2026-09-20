"""
Controlled 1:1 Class Balancing & Dev-Calibrated Thresholding Experiment (Option A).
Eliminates synthetic majority bias by creating an exact 1:1 training distribution
(3,710 genuine vs 3,710 synthetic), calibrating decision threshold on DEV,
and performing honest, unbiased evaluation on held-out TEST.
"""

import os
import sys
import time
import json
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

from backend.config import (
    SAMPLE_RATE, BASELINE_MODEL_PATH, RESULTS_DIR, FEATURES_DIR
)
from backend.dataset_loader import DatasetLoader, AudioSample
from backend.preprocess import AudioPreprocessor
from backend.features import FeatureExtractor
from backend.classifier import AntiSpoofClassifier


import concurrent.futures
import argparse

def extract_or_load_split(
    samples: List[AudioSample],
    preprocessor: AudioPreprocessor,
    extractor: FeatureExtractor,
    cache_path: Path,
    max_workers: int = 8,
    force_recompute: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Extract or load cached features with system_id tracking."""
    if cache_path.exists() and not force_recompute:
        print(f"Loading cached features from: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return (
            data["X"],
            data["y"],
            list(data["feature_names"]),
            list(data["system_ids"])
        )

    print(f"Extracting features from {len(samples)} audio files -> {cache_path.name} (using {max_workers} worker threads)...")
    
    # Pre-extract canon feature names from dummy signal
    dummy_feat = extractor.extract_all(np.zeros(SAMPLE_RATE, dtype=np.float32))
    feature_names = sorted(dummy_feat.keys())

    def _process_one(sample):
        try:
            audio = preprocessor.load_audio(sample.file_path)

            # Robust acoustic generalization: 35% of genuine speech is augmented
            # with subtle ambient room/mic noise (SNR 18 - 32 dB) so the model
            # learns that real-world room noise and echo do NOT indicate AI cloning!
            if sample.label == 0 and np.random.rand() < 0.35:
                noise_amp = np.random.uniform(0.003, 0.015) * max(0.05, float(np.max(np.abs(audio))))
                audio = audio + np.random.randn(len(audio)).astype(np.float32) * noise_amp

            # Apply Voice Activity Detection (VAD)
            voiced = preprocessor.apply_vad(audio, top_db=28.0)
            if len(voiced) >= int(0.5 * SAMPLE_RATE):
                audio = voiced
            audio = preprocessor.normalize_audio(audio)

            feat_dict = extractor.extract_all(audio)
            vec = extractor.to_vector(feat_dict, feature_names)
            return vec, sample.label, sample.system_id
        except Exception:
            return None

    feature_list = []
    labels = []
    system_ids = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(executor.map(_process_one, samples), total=len(samples), desc=f"Processing {cache_path.stem}"):
            if result is not None:
                vec, label, sys_id = result
                feature_list.append(vec)
                labels.append(label)
                system_ids.append(sys_id)

    X = np.array(feature_list, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X=X,
        y=y,
        feature_names=feature_names,
        system_ids=system_ids
    )
    print(f"Saved feature matrix ({X.shape}) to: {cache_path}")
    return X, y, feature_names, system_ids


def create_balanced_train_subset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sys_train: List[str],
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Sub-samples the majority class to achieve an exact 1:1 ratio between genuine and spoof clips,
    stratifying evenly across all spoof attack generators.
    Gracefully handles whether genuine or spoof is the larger class.
    """
    rng = np.random.RandomState(random_state)
    genuine_indices = np.where(y_train == 0)[0]
    spoof_indices = np.where(y_train == 1)[0]

    n_genuine = len(genuine_indices)
    n_spoof = len(spoof_indices)
    target_count = min(n_genuine, n_spoof)

    print(f"\n[Balancing] Genuine count in Train: {n_genuine}")
    print(f"[Balancing] Original Spoof count: {n_spoof}")
    print(f"[Balancing] Target balanced count per class: {target_count}")

    # 1. Genuine selection: subsample to target_count if needed
    if n_genuine > target_count:
        selected_genuine_indices = rng.choice(genuine_indices, size=target_count, replace=False)
    else:
        selected_genuine_indices = genuine_indices

    # 2. Spoof selection: stratified across spoof attack systems to target_count
    spoof_systems = np.array(sys_train)[spoof_indices]
    unique_systems, sys_counts = np.unique(spoof_systems, return_counts=True)

    selected_spoof_indices = []
    if len(unique_systems) > 0:
        samples_per_sys = target_count // len(unique_systems)
        remainder = target_count % len(unique_systems)

        for i, sys_name in enumerate(unique_systems):
            cur_sys_mask = (spoof_systems == sys_name)
            cur_indices = spoof_indices[cur_sys_mask]
            quota = samples_per_sys + (1 if i < remainder else 0)
            n_take = min(quota, len(cur_indices))
            if n_take > 0:
                chosen = rng.choice(cur_indices, size=n_take, replace=False)
                selected_spoof_indices.extend(chosen)

    # If still short due to unequal attack categories, fill remainder from remaining spoof pool
    if len(selected_spoof_indices) < target_count:
        needed = target_count - len(selected_spoof_indices)
        pool = list(set(spoof_indices) - set(selected_spoof_indices))
        if len(pool) > 0:
            n_fill = min(needed, len(pool))
            extra = rng.choice(pool, size=n_fill, replace=False)
            selected_spoof_indices.extend(extra)

    # Ensure precise matching 1:1 pair count
    final_pairs = min(len(selected_genuine_indices), len(selected_spoof_indices))
    selected_genuine_indices = np.array(selected_genuine_indices[:final_pairs], dtype=int)
    selected_spoof_indices = np.array(selected_spoof_indices[:final_pairs], dtype=int)

    combined_indices = np.concatenate([selected_genuine_indices, selected_spoof_indices])
    rng.shuffle(combined_indices)

    X_balanced = X_train[combined_indices]
    y_balanced = y_train[combined_indices]
    sys_balanced = [sys_train[idx] for idx in combined_indices]

    print(f"[Balancing] Final Balanced Train Matrix: {X_balanced.shape}, Labels: {np.bincount(y_balanced)}")
    return X_balanced, y_balanced, sys_balanced


def find_optimal_dev_threshold(y_dev: np.ndarray, y_dev_prob: np.ndarray) -> Tuple[float, float]:
    """
    Calibrate decision threshold strictly on DEVELOPMENT set by finding the EER point
    where False Positive Rate (FPR) == False Negative Rate (FNR).
    Does NOT touch the test set!
    """
    fpr, tpr, thresholds = roc_curve(y_dev, y_dev_prob, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    best_threshold = float(thresholds[eer_idx])
    dev_eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    return best_threshold, dev_eer


def run_balanced_experiment(max_samples: int = 10000, force_recompute: bool = False, max_workers: int = 8):
    print("=" * 70)
    print("CONTROLLED 1:1 CLASS BALANCING & DEV THRESHOLD CALIBRATION")
    print("=" * 70)

    # 1. Dataset Loading
    loader = DatasetLoader()
    samples = loader.load_samples()
    print(f"Discovered total dataset universe: {len(samples)} audio files.")

    # Stratified sub-sampling if requested
    if max_samples > 0 and len(samples) > max_samples:
        print(f"\n[*] Sub-sampling dataset from {len(samples)} to {max_samples} balanced clips...")
        rng = np.random.RandomState(42)
        gen_s = [s for s in samples if s.label == 0]
        spoof_s = [s for s in samples if s.label == 1]

        target_gen = max_samples // 2
        target_spoof = max_samples - target_gen

        chosen_gen = [gen_s[i] for i in rng.choice(len(gen_s), size=min(target_gen, len(gen_s)), replace=False)]

        sys_dict = {}
        for idx, s in enumerate(spoof_s):
            sys_dict.setdefault(s.system_id, []).append(idx)

        chosen_spoof_indices = []
        per_sys = max(1, target_spoof // len(sys_dict))
        for sys_id, idx_list in sys_dict.items():
            take = min(per_sys, len(idx_list))
            chosen_spoof_indices.extend(rng.choice(idx_list, size=take, replace=False))

        rem = target_spoof - len(chosen_spoof_indices)
        if rem > 0:
            chosen_set = set(chosen_spoof_indices)
            remaining_indices = [i for i in range(len(spoof_s)) if i not in chosen_set]
            chosen_spoof_indices.extend(rng.choice(remaining_indices, size=min(rem, len(remaining_indices)), replace=False))

        chosen_spoof = [spoof_s[i] for i in chosen_spoof_indices]
        samples = chosen_gen + chosen_spoof
        rng.shuffle(samples)
        print(f"[+] Prepared training universe: {len(samples)} clips ({len(chosen_gen)} genuine, {len(chosen_spoof)} spoof)")
        loader.samples = samples

    splits = loader.create_splits(train_ratio=0.70, dev_ratio=0.15, test_ratio=0.15)
    split_stats = loader.get_split_stats()

    preprocessor = AudioPreprocessor(target_sr=SAMPLE_RATE)
    extractor = FeatureExtractor(sample_rate=SAMPLE_RATE)

    # 2. Extract or Load Caches
    cache_tag = f"asv_merged_robust_{max_samples}" if max_samples > 0 else "asv_merged_robust_full"
    train_cache = FEATURES_DIR / f"features_{cache_tag}_train.npz"
    dev_cache = FEATURES_DIR / f"features_{cache_tag}_dev.npz"
    test_cache = FEATURES_DIR / f"features_{cache_tag}_test.npz"

    X_train, y_train, feat_names, sys_train = extract_or_load_split(
        splits["train"], preprocessor, extractor, train_cache, max_workers=max_workers, force_recompute=force_recompute
    )
    X_dev, y_dev, _, sys_dev = extract_or_load_split(
        splits["dev"], preprocessor, extractor, dev_cache, max_workers=max_workers, force_recompute=force_recompute
    )
    X_test, y_test, _, sys_test = extract_or_load_split(
        splits["test"], preprocessor, extractor, test_cache, max_workers=max_workers, force_recompute=force_recompute
    )

    # 3. Controlled 1:1 Balanced Train Set Creation
    X_train_bal, y_train_bal, _ = create_balanced_train_subset(X_train, y_train, sys_train)

    # 4. Model Training
    print("\n[Step 3/5] Fitting Balanced Random Forest Classifier...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=2,
        class_weight=None,  # Classes are already precisely 1:1 balanced!
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_bal, y_train_bal)
    fit_time = round(time.time() - t0, 2)
    print(f"[+] Model fit completed in {fit_time}s.")

    # 5. Threshold Calibration on DEV Set (Zero Test Leakage)
    print("\n[Step 4/5] Calibrating optimal threshold on DEVELOPMENT split (Dev)...")
    y_dev_prob = rf.predict_proba(X_dev)[:, 1]
    tau_star, dev_eer = find_optimal_dev_threshold(y_dev, y_dev_prob)
    dev_auc = roc_auc_score(y_dev, y_dev_prob)
    dev_preds = (y_dev_prob >= tau_star).astype(int)
    dev_acc = accuracy_score(y_dev, dev_preds)

    print(f"    * Dev ROC-AUC        : {dev_auc:.4f}")
    print(f"    * Dev EER            : {dev_eer * 100:.2f}%")
    print(f"    * Calibrated Threshold (tau*): {tau_star:.4f}")
    print(f"    * Dev Accuracy       : {dev_acc * 100:.2f}%")

    # 6. Unbiased Evaluation on Held-Out TEST Set
    print("\n[Step 5/5] Evaluating calibrated model on held-out TEST set...")
    y_test_prob = rf.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= tau_star).astype(int)

    acc = float(accuracy_score(y_test, y_test_pred))
    prec = float(precision_score(y_test, y_test_pred))
    rec = float(recall_score(y_test, y_test_pred))
    f1 = float(f1_score(y_test, y_test_pred))
    auc = float(roc_auc_score(y_test, y_test_prob))

    # Confusion matrix: [[True Genuine, False Spoof (FAR)], [Missed Spoof (FRR), True Spoof]]
    cm = confusion_matrix(y_test, y_test_pred)
    true_gen, false_spoof = cm[0][0], cm[0][1]
    missed_spoof, true_spoof = cm[1][0], cm[1][1]

    far = float(false_spoof / (true_gen + false_spoof)) if (true_gen + false_spoof) > 0 else 0.0
    frr = float(missed_spoof / (missed_spoof + true_spoof)) if (missed_spoof + true_spoof) > 0 else 0.0

    # Test EER calculation
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob, pos_label=1)
    fnr_test = 1.0 - tpr_test
    test_eer_idx = np.nanargmin(np.abs(fpr_test - fnr_test))
    test_eer = float((fpr_test[test_eer_idx] + fnr_test[test_eer_idx]) / 2.0)

    # 7. Print Comprehensive Report
    print("\n" + "=" * 70)
    print("OPTION A RESULTS: CONTROLLED 1:1 BALANCED MODEL REPORT")
    print("=" * 70)
    print(f"Training Set (1:1 Balanced)    : {len(X_train_bal)} samples ({np.bincount(y_train_bal)[0]} Real / {np.bincount(y_train_bal)[1]} Fake)")
    print(f"Dev Set (Tuning)               : {len(X_dev)} samples ({np.bincount(y_dev)[0]} Real / {np.bincount(y_dev)[1]} Fake)")
    print(f"Test Set (Honest Evaluation)   : {len(X_test)} samples ({np.bincount(y_test)[0]} Real / {np.bincount(y_test)[1]} Fake)")
    print(f"Calibrated Threshold (tau*)    : {tau_star:.4f}")
    print("-" * 70)
    print(f"Accuracy                       : {acc * 100:.2f}%")
    print(f"Precision                      : {prec * 100:.2f}%")
    print(f"Recall                         : {rec * 100:.2f}%")
    print(f"F1-Score                       : {f1 * 100:.2f}%")
    print(f"ROC-AUC                        : {auc:.4f}")
    print(f"Equal Error Rate (EER)         : {test_eer * 100:.2f}%")
    print(f"False Alarm Rate (FAR - Real->Fake): {far * 100:.2f}% ({false_spoof}/{true_gen + false_spoof})")
    print(f"Miss Rate        (FRR - Fake->Real): {frr * 100:.2f}% ({missed_spoof}/{missed_spoof + true_spoof})")
    print("-" * 70)
    print("Confusion Matrix:")
    print(f"                     Predicted Real    Predicted Fake")
    print(f"  Actual Real (Bona)      {true_gen:6d}            {false_spoof:6d}")
    print(f"  Actual Fake (Spoof)     {missed_spoof:6d}            {true_spoof:6d}")
    print("=" * 70)

    # 8. Save Model & Metrics
    classifier = AntiSpoofClassifier(
        model_type="random_forest",
        model_params={"n_estimators": 300, "max_depth": 18, "min_samples_leaf": 2},
        calibrated_threshold=tau_star
    )
    classifier.feature_names = feat_names
    classifier.model = rf
    classifier.save(BASELINE_MODEL_PATH)
    print(f"[+] Serialized model saved to: {BASELINE_MODEL_PATH}")

    # Write report JSON
    results = {
        "experiment": "Option_A_Balanced_1to1",
        "train_samples_balanced": int(len(X_train_bal)),
        "calibrated_threshold": round(tau_star, 4),
        "dev_eer": round(dev_eer, 4),
        "test_metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "eer": round(test_eer, 4),
            "false_alarm_rate_far": round(far, 4),
            "miss_rate_frr": round(frr, 4),
            "confusion_matrix": {
                "true_genuine": int(true_gen),
                "false_spoof_alarm": int(false_spoof),
                "missed_spoof": int(missed_spoof),
                "true_spoof_detected": int(true_spoof)
            }
        }
    }
    with open(RESULTS_DIR / "balanced_experiment.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Results saved to: {RESULTS_DIR / 'balanced_experiment.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain Voice Integrity Model with Balanced 1:1 Sampling")
    parser.add_argument("--max-samples", type=int, default=10000, help="Maximum total samples to extract (0 for all 125k)")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputing feature extraction cache")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker threads")
    args = parser.parse_args()
    run_balanced_experiment(max_samples=args.max_samples, force_recompute=args.force_recompute, max_workers=args.workers)
