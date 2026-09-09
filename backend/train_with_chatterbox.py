"""
Train Balanced Multi-Generator Detector with ChatterboxTTS (PhonemeDF).
Integrates modern Phoneme ChatterboxTTS synthetic speech alongside ASVspoof 2019 (A07-A19)
and LibriSpeech genuine human voices, maintaining strict 1:1 balance and honest evaluation.
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import SAMPLE_RATE, MODELS_DIR, FEATURES_DIR, RESULTS_DIR
from backend.audio_preprocessing import AudioPreprocessor
from backend.features import FeatureExtractor
from backend.classifier import AntiSpoofClassifier


def extract_chatterbox_features(
    chatterbox_dir: Path,
    cache_file: Path,
    n_samples: int = 2000,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Extract or load cached features for ChatterboxTTS samples."""
    if cache_file.exists():
        print(f"[+] Loading cached Chatterbox features from: {cache_file.name}")
        data = np.load(cache_file, allow_pickle=True)
        return data["X"], data["y"], list(data["feature_names"]), list(data["system_ids"])

    print(f"[+] Extracting features from {n_samples} ChatterboxTTS audio files...")
    wav_files = sorted(list(chatterbox_dir.glob("*.wav")))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {chatterbox_dir}")

    rng = np.random.RandomState(random_state)
    selected_files = list(rng.choice(wav_files, size=min(n_samples, len(wav_files)), replace=False))

    preprocessor = AudioPreprocessor(target_sr=SAMPLE_RATE)
    extractor = FeatureExtractor(sample_rate=SAMPLE_RATE)

    feature_list = []
    labels = []
    system_ids = []
    feature_names = None

    for fpath in tqdm(selected_files, desc="Extracting ChatterboxTTS"):
        try:
            audio = preprocessor.load_audio(fpath)
            audio = preprocessor.normalize_amplitude(audio)
            feat_dict = extractor.extract_all(audio)

            if feature_names is None:
                feature_names = sorted(feat_dict.keys())

            vec = extractor.to_vector(feat_dict, feature_names)
            feature_list.append(vec)
            labels.append(1)  # Spoof = 1
            system_ids.append("chatterbox_tts")
        except Exception as e:
            print(f"[!] Error processing {fpath.name}: {e}")

    X = np.array(feature_list, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        X=X,
        y=y,
        feature_names=feature_names,
        system_ids=system_ids
    )
    print(f"[+] Saved Chatterbox features to: {cache_file}")
    return X, y, feature_names, system_ids


def main():
    print("=" * 75)
    print("RETRAINING WITH CHATTERBOX-TTS (PHONEMEDF) & MERGED DATASETS")
    print("=" * 75)

    # 1. Load existing ASVspoof + LibriSpeech cached features
    train_cache = FEATURES_DIR / "features_merged_train.npz"
    dev_cache = FEATURES_DIR / "features_merged_dev.npz"
    test_cache = FEATURES_DIR / "features_merged_test.npz"

    if not (train_cache.exists() and dev_cache.exists() and test_cache.exists()):
        print("[!] Base features not found. Please run backend/train_balanced.py first.")
        return

    print("[1/4] Loading base ASVspoof + LibriSpeech feature matrices...")
    train_data = np.load(train_cache, allow_pickle=True)
    dev_data = np.load(dev_cache, allow_pickle=True)
    test_data = np.load(test_cache, allow_pickle=True)

    X_train_base, y_train_base = train_data["X"], train_data["y"]
    sys_train_base = list(train_data["system_ids"])
    feature_names = list(train_data["feature_names"])

    X_dev_base, y_dev_base = dev_data["X"], dev_data["y"]
    sys_dev_base = list(dev_data["system_ids"])

    X_test_base, y_test_base = test_data["X"], test_data["y"]
    sys_test_base = list(test_data["system_ids"])

    # 2. Extract ChatterboxTTS features (2,000 samples)
    chatterbox_dir = PROJECT_ROOT / "datasets" / "phonemedf_chatterboxtts"
    chatterbox_cache = FEATURES_DIR / "features_chatterbox_2000.npz"
    X_cb, y_cb, _, sys_cb = extract_chatterbox_features(chatterbox_dir, chatterbox_cache, n_samples=2000)

    # Split Chatterbox: 1,400 Train (70%), 300 Dev (15%), 300 Test (15%)
    rng = np.random.RandomState(42)
    indices = np.arange(len(X_cb))
    rng.shuffle(indices)

    cb_train_idx = indices[:1400]
    cb_dev_idx = indices[1400:1700]
    cb_test_idx = indices[1700:]

    X_cb_train, y_cb_train = X_cb[cb_train_idx], y_cb[cb_train_idx]
    sys_cb_train = [sys_cb[i] for i in cb_train_idx]

    X_cb_dev, y_cb_dev = X_cb[cb_dev_idx], y_cb[cb_dev_idx]
    sys_cb_dev = [sys_cb[i] for i in cb_dev_idx]

    X_cb_test, y_cb_test = X_cb[cb_test_idx], y_cb[cb_test_idx]
    sys_cb_test = [sys_cb[i] for i in cb_test_idx]

    # 3. Form Balanced 1:1 Train Split:
    # 3,710 Genuine vs 3,710 Spoof (2,310 ASVspoof + 1,400 ChatterboxTTS)
    print("\n[2/4] Constructing balanced 1:1 training distribution...")
    genuine_mask = (y_train_base == 0)
    spoof_mask = (y_train_base == 1)

    X_real = X_train_base[genuine_mask]
    y_real = y_train_base[genuine_mask]
    sys_real = [sys_train_base[i] for i in np.where(genuine_mask)[0]]
    n_genuine = len(X_real)  # 3,710

    # From base spoofs, take exactly (n_genuine - len(cb_train_idx)) = 3,710 - 1,400 = 2,310
    n_asv_needed = n_genuine - len(cb_train_idx)
    spoof_base_indices = np.where(spoof_mask)[0]
    chosen_asv = rng.choice(spoof_base_indices, size=n_asv_needed, replace=False)

    X_asv_spoof = X_train_base[chosen_asv]
    y_asv_spoof = y_train_base[chosen_asv]
    sys_asv_spoof = [sys_train_base[i] for i in chosen_asv]

    X_train_bal = np.vstack([X_real, X_asv_spoof, X_cb_train])
    y_train_bal = np.concatenate([y_real, y_asv_spoof, y_cb_train])
    sys_train_bal = sys_real + sys_asv_spoof + sys_cb_train

    shuffle_idx = rng.permutation(len(y_train_bal))
    X_train_bal = X_train_bal[shuffle_idx]
    y_train_bal = y_train_bal[shuffle_idx]
    sys_train_bal = [sys_train_bal[i] for i in shuffle_idx]

    print(f"  - Total Balanced Training Set: {len(y_train_bal):,} samples")
    print(f"  - Genuine Human Voices       : {np.sum(y_train_bal == 0):,} (50.0%)")
    print(f"  - Synthetic Spoof Voices     : {np.sum(y_train_bal == 1):,} (50.0%)")
    print(f"    * ASVspoof 2019 (A07-A19)  : {len(X_asv_spoof):,}")
    print(f"    * ChatterboxTTS (PhonemeDF): {len(X_cb_train):,}")

    # Merge Dev & Test sets
    X_dev_merged = np.vstack([X_dev_base, X_cb_dev])
    y_dev_merged = np.concatenate([y_dev_base, y_cb_dev])
    sys_dev_merged = sys_dev_base + sys_cb_dev

    X_test_merged = np.vstack([X_test_base, X_cb_test])
    y_test_merged = np.concatenate([y_test_base, y_cb_test])
    sys_test_merged = sys_test_base + sys_cb_test

    print(f"  - Dev Set (with Chatterbox) : {len(y_dev_merged):,} samples")
    print(f"  - Test Set (with Chatterbox): {len(y_test_merged):,} samples")

    # 4. Train Random Forest
    print("\n[3/4] Training Calibrated Random Forest (100 estimators, max_depth=16)...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    t0 = time.time()
    clf.fit(X_train_bal, y_train_bal)
    train_duration = time.time() - t0
    print(f"[+] RF Model trained in {train_duration:.2f}s")

    # 5. Calibrate on DEV Set
    dev_probs = clf.predict_proba(X_dev_merged)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_dev_merged, dev_probs, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    opt_threshold = float(thresholds[eer_idx])
    dev_eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    dev_auc = float(roc_auc_score(y_dev_merged, dev_probs))

    print(f"\n[+] Dev Set Calibration Results:")
    print(f"    * Optimal Threshold tau* : {opt_threshold:.4f}")
    print(f"    * Dev Equal Error Rate   : {dev_eer * 100:.2f}%")
    print(f"    * Dev ROC-AUC            : {dev_auc:.4f}")

    # 6. Evaluate on TEST Set (Strictly Unbiased)
    print("\n[4/4] Evaluating on Unbiased Held-Out Test Set...")
    test_probs = clf.predict_proba(X_test_merged)[:, 1]
    test_preds = (test_probs >= opt_threshold).astype(int)

    test_acc = float(accuracy_score(y_test_merged, test_preds))
    test_prec = float(precision_score(y_test_merged, test_preds))
    test_rec = float(recall_score(y_test_merged, test_preds))
    test_f1 = float(f1_score(y_test_merged, test_preds))
    test_auc = float(roc_auc_score(y_test_merged, test_probs))

    fpr_t, tpr_t, _ = roc_curve(y_test_merged, test_probs, pos_label=1)
    fnr_t = 1.0 - tpr_t
    test_eer_idx = np.nanargmin(np.abs(fpr_t - fnr_t))
    test_eer = float((fpr_t[test_eer_idx] + fnr_t[test_eer_idx]) / 2.0)

    cm = confusion_matrix(y_test_merged, test_preds)
    tn, fp, fn, tp = cm.ravel()
    far = float(fp / (tn + fp))  # Real -> Fake
    frr = float(fn / (fn + tp))  # Fake -> Real

    print("=" * 75)
    print("FINAL TEST EVALUATION METRICS (WITH CHATTERBOX-TTS)")
    print("=" * 75)
    print(f"  Accuracy                : {test_acc * 100:.2f}%")
    print(f"  Precision               : {test_prec * 100:.2f}%")
    print(f"  Recall                  : {test_rec * 100:.2f}%")
    print(f"  F1 Score                : {test_f1 * 100:.2f}%")
    print(f"  ROC-AUC                 : {test_auc:.4f}")
    print(f"  Equal Error Rate (EER)  : {test_eer * 100:.2f}%")
    print(f"  False Alarm Rate (Real->Fake) : {far * 100:.2f}% ({fp}/{tn+fp})")
    print(f"  Miss Rate (Fake->Real)        : {frr * 100:.2f}% ({fn}/{fn+tp})")
    print("\nConfusion Matrix:")
    print(f"                 Predicted Real   Predicted Fake")
    print(f"  Actual Real         {tn:6d}           {fp:6d}")
    print(f"  Actual Fake         {fn:6d}           {tp:6d}")

    # Benchmark ChatterboxTTS detection rate specifically:
    cb_test_mask = (np.array(sys_test_merged) == "chatterbox_tts")
    cb_preds = test_preds[cb_test_mask]
    cb_acc = float(np.mean(cb_preds == 1))
    cb_mean_prob = float(np.mean(test_probs[cb_test_mask]))
    print(f"\n[+] ChatterboxTTS Specific Test Performance:")
    print(f"    * Detection Rate : {cb_acc * 100:.2f}% ({np.sum(cb_preds == 1)}/{len(cb_preds)})")
    print(f"    * Mean Spoof Prob: {cb_mean_prob:.4f}")

    # Save trained model checkpoint
    model_wrapper = AntiSpoofClassifier(
        model_type="random_forest",
        calibrated_threshold=opt_threshold
    )
    model_wrapper.model = clf
    model_wrapper.feature_names = feature_names
    save_path = MODELS_DIR / "baseline_random_forest.pkl"
    model_wrapper.save(save_path)
    print(f"\n[+] Updated model persisted to: {save_path}")

    # Save detailed JSON report
    report = {
        "dataset": "Merged (ASVspoof 2019 + LibriSpeech + PhonemeDF ChatterboxTTS)",
        "train_samples": len(y_train_bal),
        "test_samples": len(y_test_merged),
        "dev_optimal_threshold": opt_threshold,
        "dev_eer": dev_eer,
        "dev_auc": dev_auc,
        "test_metrics": {
            "accuracy": test_acc,
            "precision": test_prec,
            "recall": test_rec,
            "f1": test_f1,
            "roc_auc": test_auc,
            "eer": test_eer,
            "far_real_to_fake": far,
            "frr_fake_to_real": frr,
            "confusion_matrix": {
                "true_negative_real": int(tn),
                "false_positive_fake": int(fp),
                "false_negative_real": int(fn),
                "true_positive_fake": int(tp)
            },
            "chatterbox_detection_rate": cb_acc,
            "chatterbox_mean_spoof_prob": cb_mean_prob
        }
    }
    with open(RESULTS_DIR / "chatterbox_experiment.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Experiment report written to: {RESULTS_DIR / 'chatterbox_experiment.json'}")


if __name__ == "__main__":
    main()
