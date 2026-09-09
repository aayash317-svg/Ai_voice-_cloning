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
from backend.audio_preprocessing import AudioPreprocessor
from backend.features import FeatureExtractor
from backend.classifier import AntiSpoofClassifier


def extract_or_load_split(
    samples: List[AudioSample],
    preprocessor: AudioPreprocessor,
    extractor: FeatureExtractor,
    cache_path: Path
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Extract or load cached features with system_id tracking."""
    if cache_path.exists():
        print(f"Loading cached features from: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return (
            data["X"],
            data["y"],
            list(data["feature_names"]),
            list(data["system_ids"])
        )

    print(f"Extracting features from {len(samples)} audio files -> {cache_path.name}...")
    feature_list = []
    labels = []
    system_ids = []
    feature_names = None

    for sample in tqdm(samples, desc=f"Processing {cache_path.stem}"):
        try:
            audio = preprocessor.load_audio(sample.file_path)
            audio = preprocessor.normalize_amplitude(audio)
            feat_dict = extractor.extract_all(audio)

            if feature_names is None:
                feature_names = sorted(feat_dict.keys())

            vec = extractor.to_vector(feat_dict, feature_names)
            feature_list.append(vec)
            labels.append(sample.label)
            system_ids.append(sample.system_id)
        except Exception as e:
            print(f"[!] Error processing {sample.file_path.name}: {e}")

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
    print(f"Saved feature matrix to: {cache_path}")
    return X, y, feature_names, system_ids


def create_balanced_train_subset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sys_train: List[str],
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Sub-samples the majority spoof class to exactly match genuine count (1:1 ratio),
    stratifying evenly across all spoof attack generators.
    """
    rng = np.random.RandomState(random_state)
    genuine_indices = np.where(y_train == 0)[0]
    spoof_indices = np.where(y_train == 1)[0]

    n_genuine = len(genuine_indices)
    print(f"\n[Balancing] Genuine count in Train: {n_genuine}")
    print(f"[Balancing] Original Spoof count: {len(spoof_indices)}")

    # Stratified sampling across spoof systems
    spoof_systems = np.array(sys_train)[spoof_indices]
    unique_systems, sys_counts = np.unique(spoof_systems, return_counts=True)

    selected_spoof_indices = []
    samples_per_sys = n_genuine // len(unique_systems)
    remainder = n_genuine % len(unique_systems)

    for i, sys_name in enumerate(unique_systems):
        cur_sys_mask = (spoof_systems == sys_name)
        cur_indices = spoof_indices[cur_sys_mask]
        quota = samples_per_sys + (1 if i < remainder else 0)
        n_take = min(quota, len(cur_indices))
        chosen = rng.choice(cur_indices, size=n_take, replace=False)
        selected_spoof_indices.extend(chosen)

    # If still short due to small categories, fill remainder uniformly
    if len(selected_spoof_indices) < n_genuine:
        needed = n_genuine - len(selected_spoof_indices)
        pool = list(set(spoof_indices) - set(selected_spoof_indices))
        extra = rng.choice(pool, size=needed, replace=False)
        selected_spoof_indices.extend(extra)

    selected_spoof_indices = np.array(selected_spoof_indices)
    combined_indices = np.concatenate([genuine_indices, selected_spoof_indices])
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


def run_balanced_experiment():
    print("=" * 70)
    print("OPTION A: CONTROLLED 1:1 CLASS BALANCING & DEV THRESHOLD CALIBRATION")
    print("=" * 70)

    # 1. Dataset Loading
    loader = DatasetLoader()
    samples = loader.load_samples()
    splits = loader.create_splits(train_ratio=0.70, dev_ratio=0.15, test_ratio=0.15)
    split_stats = loader.get_split_stats()

    preprocessor = AudioPreprocessor(target_sr=SAMPLE_RATE)
    extractor = FeatureExtractor(sample_rate=SAMPLE_RATE)

    # 2. Extract or Load Caches
    train_cache = FEATURES_DIR / "features_merged_train.npz"
    dev_cache = FEATURES_DIR / "features_merged_dev.npz"
    test_cache = FEATURES_DIR / "features_merged_test.npz"

    X_train, y_train, feat_names, sys_train = extract_or_load_split(splits["train"], preprocessor, extractor, train_cache)
    X_dev, y_dev, _, sys_dev = extract_or_load_split(splits["dev"], preprocessor, extractor, dev_cache)
    X_test, y_test, _, sys_test = extract_or_load_split(splits["test"], preprocessor, extractor, test_cache)

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
    run_balanced_experiment()
