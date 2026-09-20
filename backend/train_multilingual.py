"""
Multilingual Anti-Spoofing Model Training & Calibration Engine.

Trains an augmented Multilingual Random Forest Classifier integrating IndicTTS (genuine Indian speech)
and IndicSynth (AI-cloned speech across XTTSv2, VITS, FreeVC24, YourTTS) alongside existing ASVspoof / LibriSpeech.
Performs 1:1 class balancing, DEV threshold calibration (tau*), and unbiased held-out TEST evaluation.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    SAMPLE_RATE, MULTILINGUAL_MODEL_PATH, BASELINE_MODEL_PATH,
    RESULTS_DIR, FEATURES_DIR, INDIC_FEATURES_DIR, INDIC_TARGET_LANGUAGES
)
from backend.dataset_loader import DatasetLoader, AudioSample
from backend.preprocess import AudioPreprocessor
from backend.features import FeatureExtractor
from backend.classifier import AntiSpoofClassifier

import concurrent.futures


def extract_or_load_split(
    samples: List[AudioSample],
    preprocessor: AudioPreprocessor,
    extractor: FeatureExtractor,
    cache_path: Path,
    max_workers: int = 8,
    force_recompute: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Extract or load cached 63-D features with system_id and language tracking."""
    if cache_path.exists() and not force_recompute:
        print(f"Loading cached features from: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return (
            data["X"],
            data["y"],
            list(data["feature_names"]),
            list(data["system_ids"])
        )

    print(f"Extracting 63-D features from {len(samples)} audio files -> {cache_path.name} ({max_workers} workers)...")
    dummy_feat = extractor.extract_all(np.zeros(SAMPLE_RATE, dtype=np.float32))
    feature_names = sorted(dummy_feat.keys())

    def _process_one(sample):
        try:
            audio = preprocessor.load_audio(sample.file_path)

            # Robust acoustic generalization: 20% of all speech (both genuine & synthetic) augmented with subtle ambient noise
            # Ensures model does NOT use noise or jitter presence as a shortcut for either class
            if np.random.rand() < 0.20:
                noise_amp = np.random.uniform(0.002, 0.010) * max(0.05, float(np.max(np.abs(audio))))
                audio = audio + np.random.randn(len(audio)).astype(np.float32) * noise_amp

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
    """Sub-samples majority class to achieve an exact 1:1 ratio between genuine and spoof clips."""
    rng = np.random.RandomState(random_state)
    genuine_indices = np.where(y_train == 0)[0]
    spoof_indices = np.where(y_train == 1)[0]

    n_genuine = len(genuine_indices)
    n_spoof = len(spoof_indices)
    target_count = min(n_genuine, n_spoof)

    print(f"\n[Balancing] Genuine count in Train: {n_genuine}")
    print(f"[Balancing] Spoof count in Train: {n_spoof}")
    print(f"[Balancing] Target balanced count per class: {target_count}")

    if n_genuine > target_count:
        selected_genuine_indices = rng.choice(genuine_indices, size=target_count, replace=False)
    else:
        selected_genuine_indices = genuine_indices

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

    if len(selected_spoof_indices) < target_count:
        needed = target_count - len(selected_spoof_indices)
        pool = list(set(spoof_indices) - set(selected_spoof_indices))
        if len(pool) > 0:
            n_fill = min(needed, len(pool))
            extra = rng.choice(pool, size=n_fill, replace=False)
            selected_spoof_indices.extend(extra)

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
    """Calibrate decision threshold strictly on DEVELOPMENT set at the EER point (FPR == FNR)."""
    fpr, tpr, thresholds = roc_curve(y_dev, y_dev_prob, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    best_threshold = float(thresholds[eer_idx])
    dev_eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    return best_threshold, dev_eer


def run_multilingual_experiment(max_samples: int = 20000, force_recompute: bool = False, max_workers: int = 8):
    print("=" * 70)
    print("MULTILINGUAL VOICE INTEGRITY TRAINING & EVALUATION ENGINE")
    print("Languages: Hindi, Tamil, Telugu, Bengali, Kannada, Marathi + English/ASV")
    print("=" * 70)

    # 1. Dataset Loading
    loader = DatasetLoader()
    samples = loader.load_samples()
    print(f"Total Discovered Audio Universe: {len(samples)} clips.")

    indic_samples = [s for s in samples if "indic" in s.system_id]
    print(f"IndicTTS & IndicSynth Utterances : {len(indic_samples)} clips.")

    # Sub-sample universe prioritizing multilingual samples with exact 1:1 balance
    if max_samples > 0 and len(samples) > max_samples:
        print(f"\n[*] Curating balanced universe of {max_samples} clips (50% Genuine, 50% Spoof)...")
        rng = np.random.RandomState(42)

        target_gen = max_samples // 2
        target_spoof = max_samples - target_gen

        # Separate all genuine and spoof samples
        gen_pool = [s for s in samples if s.label == 0]
        spoof_pool = [s for s in samples if s.label == 1]

        # Prioritize Indic samples within each pool
        indic_gen = [s for s in gen_pool if "indic" in s.system_id]
        other_gen = [s for s in gen_pool if "indic" not in s.system_id]

        indic_spoof = [s for s in spoof_pool if "indic" in s.system_id]
        other_spoof = [s for s in spoof_pool if "indic" not in s.system_id]

        # Sample genuine (80% IndicTTS/SLR65/FLEURS, 20% LibriSpeech/ASV)
        n_indic_gen = min(len(indic_gen), int(target_gen * 0.80))
        n_other_gen = min(len(other_gen), target_gen - n_indic_gen)
        if n_indic_gen + n_other_gen < target_gen:
            if len(indic_gen) > n_indic_gen:
                n_indic_gen = min(len(indic_gen), target_gen - n_other_gen)
            elif len(other_gen) > n_other_gen:
                n_other_gen = min(len(other_gen), target_gen - n_indic_gen)

        chosen_gen_indices = rng.choice(len(indic_gen), size=n_indic_gen, replace=False)
        chosen_other_gen_indices = rng.choice(len(other_gen), size=n_other_gen, replace=False)
        chosen_gen = [indic_gen[i] for i in chosen_gen_indices] + [other_gen[i] for i in chosen_other_gen_indices]

        # Sample spoof (80% IndicSynth, 20% ASVspoof)
        n_indic_spoof = min(len(indic_spoof), int(target_spoof * 0.80))
        n_other_spoof = min(len(other_spoof), target_spoof - n_indic_spoof)
        if n_indic_spoof + n_other_spoof < target_spoof:
            if len(indic_spoof) > n_indic_spoof:
                n_indic_spoof = min(len(indic_spoof), target_spoof - n_other_spoof)
            elif len(other_spoof) > n_other_spoof:
                n_other_spoof = min(len(other_spoof), target_spoof - n_indic_spoof)

        chosen_spoof_indices = rng.choice(len(indic_spoof), size=n_indic_spoof, replace=False)
        chosen_other_spoof_indices = rng.choice(len(other_spoof), size=n_other_spoof, replace=False)
        chosen_spoof = [indic_spoof[i] for i in chosen_spoof_indices] + [other_spoof[i] for i in chosen_other_spoof_indices]

        curated_universe = chosen_gen + chosen_spoof
        rng.shuffle(curated_universe)
        samples = curated_universe
        loader.samples = samples
        print(f"[+] Curated universe: {len(samples)} clips ({len(chosen_gen)} Genuine vs {len(chosen_spoof)} Spoof, {n_indic_gen + n_indic_spoof} Indic).")

    splits = loader.create_splits(train_ratio=0.70, dev_ratio=0.15, test_ratio=0.15)
    split_stats = loader.get_split_stats()

    preprocessor = AudioPreprocessor(target_sr=SAMPLE_RATE)
    extractor = FeatureExtractor(sample_rate=SAMPLE_RATE)

    # 2. Extract or Load Caches
    cache_tag = f"multilingual_{max_samples}" if max_samples > 0 else "multilingual_full"
    train_cache = INDIC_FEATURES_DIR / f"features_{cache_tag}_train.npz"
    dev_cache = INDIC_FEATURES_DIR / f"features_{cache_tag}_dev.npz"
    test_cache = INDIC_FEATURES_DIR / f"features_{cache_tag}_test.npz"

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
    X_train_bal, y_train_bal, sys_train_bal = create_balanced_train_subset(X_train, y_train, sys_train)

    # Neutralize acoustic recording shortcuts (loudness bias & room reflection artifacts)
    shortcut_indices = [
        feat_names.index("mfcc_0_mean"),
        feat_names.index("spectral_rolloff_std")
    ]
    print(f"\n[*] Neutralizing recording shortcuts: {[feat_names[i] for i in shortcut_indices]}")
    for idx in shortcut_indices:
        X_train_bal[:, idx] = 0.0
        X_dev[:, idx] = 0.0
        X_test[:, idx] = 0.0

    # 4. Model Training
    print("\n[Step 3/5] Fitting Multilingual Voting Ensemble (Random Forest + Extra Trees with log2 feature subsampling)...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=5,
        max_features="log2",
        random_state=42,
        n_jobs=-1
    )
    et = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=5,
        max_features="log2",
        random_state=42,
        n_jobs=-1
    )
    ensemble = VotingClassifier(estimators=[('rf', rf), ('et', et)], voting='soft', n_jobs=-1)
    ensemble.fit(X_train_bal, y_train_bal)
    fit_time = round(time.time() - t0, 2)
    print(f"[+] Model fit completed in {fit_time}s.")

    # 5. Threshold Calibration on DEV Set
    print("\n[Step 4/5] Calibrating optimal threshold on DEVELOPMENT split (Dev)...")
    y_dev_prob = ensemble.predict_proba(X_dev)[:, 1]
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
    y_test_prob = ensemble.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= tau_star).astype(int)

    acc = float(accuracy_score(y_test, y_test_pred))
    prec = float(precision_score(y_test, y_test_pred))
    rec = float(recall_score(y_test, y_test_pred))
    f1 = float(f1_score(y_test, y_test_pred))
    auc = float(roc_auc_score(y_test, y_test_prob))

    cm = confusion_matrix(y_test, y_test_pred)
    true_gen, false_spoof = cm[0][0], cm[0][1]
    missed_spoof, true_spoof = cm[1][0], cm[1][1]

    far = float(false_spoof / (true_gen + false_spoof)) if (true_gen + false_spoof) > 0 else 0.0
    frr = float(missed_spoof / (missed_spoof + true_spoof)) if (missed_spoof + true_spoof) > 0 else 0.0

    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob, pos_label=1)
    fnr_test = 1.0 - tpr_test
    test_eer_idx = np.nanargmin(np.abs(fpr_test - fnr_test))
    test_eer = float((fpr_test[test_eer_idx] + fnr_test[test_eer_idx]) / 2.0)

    # Breakdown on Indian Languages specifically
    indic_mask = np.array(["indic" in s for s in sys_test])
    indic_metrics = {}
    if np.sum(indic_mask) > 10:
        y_test_indic = y_test[indic_mask]
        y_prob_indic = y_test_prob[indic_mask]
        y_pred_indic = (y_prob_indic >= tau_star).astype(int)
        indic_acc = float(accuracy_score(y_test_indic, y_pred_indic))
        indic_auc = float(roc_auc_score(y_test_indic, y_prob_indic)) if len(np.unique(y_test_indic)) > 1 else 1.0
        indic_metrics = {
            "test_samples": int(np.sum(indic_mask)),
            "accuracy": round(indic_acc, 4),
            "roc_auc": round(indic_auc, 4)
        }

    # 7. Print Comprehensive Report
    print("\n" + "=" * 70)
    print("MULTILINGUAL MODEL BENCHMARK REPORT (INDIC + ASVSPOOF)")
    print("=" * 70)
    print(f"Training Set (1:1 Balanced)    : {len(X_train_bal)} samples ({np.bincount(y_train_bal)[0]} Real / {np.bincount(y_train_bal)[1]} Fake)")
    print(f"Dev Set (Tuning)               : {len(X_dev)} samples")
    print(f"Test Set (Honest Evaluation)   : {len(X_test)} samples")
    print(f"Calibrated Threshold (tau*)    : {tau_star:.4f}")
    print("-" * 70)
    print(f"Overall Accuracy               : {acc * 100:.2f}%")
    print(f"Precision                      : {prec * 100:.2f}%")
    print(f"Recall                         : {rec * 100:.2f}%")
    print(f"F1-Score                       : {f1 * 100:.2f}%")
    print(f"ROC-AUC                        : {auc:.4f}")
    print(f"Equal Error Rate (EER)         : {test_eer * 100:.2f}%")
    print(f"False Alarm Rate (FAR)         : {far * 100:.2f}% ({false_spoof}/{true_gen + false_spoof})")
    print(f"Miss Rate (FRR)                : {frr * 100:.2f}% ({missed_spoof}/{missed_spoof + true_spoof})")
    if indic_metrics:
        print(f"Indian Languages Test Accuracy : {indic_metrics['accuracy'] * 100:.2f}% (Samples: {indic_metrics['test_samples']})")
    print("-" * 70)
    print("Confusion Matrix:")
    print(f"                     Predicted Real    Predicted Fake")
    print(f"  Actual Real (Bona)      {true_gen:6d}            {false_spoof:6d}")
    print(f"  Actual Fake (Spoof)     {missed_spoof:6d}            {true_spoof:6d}")
    print("=" * 70)

    # 8. Save Serialized Model
    classifier = AntiSpoofClassifier(
        model_type="voting_ensemble",
        model_params={"n_estimators": 100, "max_depth": 16, "min_samples_leaf": 5},
        calibrated_threshold=tau_star
    )
    classifier.feature_names = feat_names
    classifier.shortcut_indices = shortcut_indices
    classifier.model = ensemble
    classifier.save(MULTILINGUAL_MODEL_PATH)
    print(f"[+] Serialized Multilingual Model saved to: {MULTILINGUAL_MODEL_PATH}")

    # Also update baseline model checkpoint so web dashboard instantly uses multilingual weights
    classifier.save(BASELINE_MODEL_PATH)
    print(f"[+] Active Baseline Model checkpoint updated: {BASELINE_MODEL_PATH}")

    # Write report JSON
    results = {
        "experiment": "Multilingual_Indic_ASVspoof_1to1",
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
            },
            "indic_languages_breakdown": indic_metrics
        }
    }
    with open(RESULTS_DIR / "multilingual_experiment.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Results saved to: {RESULTS_DIR / 'multilingual_experiment.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multilingual Voice Integrity Verification Model")
    parser.add_argument("--max-samples", type=int, default=20000, help="Maximum total samples to extract (default: 20000)")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputing feature extraction cache")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker threads")
    args = parser.parse_args()
    run_multilingual_experiment(max_samples=args.max_samples, force_recompute=args.force_recompute, max_workers=args.workers)
