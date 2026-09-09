"""
Multi-Generator & Unseen-Attack Generalization Benchmark (Option B).
Evaluates detector performance across individual generative systems:
- Known generators: A07 - A15 (TTS, traditional VC, spectral vocoders)
- Unseen generators: A16 - A19 (concatenative, waveform filtering, neural VC)
- Genuine human speech: nat (studio) & librispeech_human (clean diverse)
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import BASELINE_MODEL_PATH, FEATURES_DIR, RESULTS_DIR
from backend.classifier import AntiSpoofClassifier


# Known generator descriptions from ASVspoof 2019 protocol
GENERATOR_PROFILES = {
    "A07": {"type": "TTS", "acoustic": "Tacotron 2 (Neural)", "vocoder": "WaveRNN", "split": "Known"},
    "A08": {"type": "TTS", "acoustic": "Neural Auto-encoder", "vocoder": "Neural", "split": "Known"},
    "A09": {"type": "TTS", "acoustic": "Tacotron 2", "vocoder": "WORLD Vocoder", "split": "Known"},
    "A10": {"type": "TTS", "acoustic": "Transfer Learning", "vocoder": "WORLD Vocoder", "split": "Known"},
    "A11": {"type": "VC",  "acoustic": "Spectral Mapping", "vocoder": "WORLD Vocoder", "split": "Known"},
    "A12": {"type": "VC",  "acoustic": "Neural Mapping",   "vocoder": "WORLD Vocoder", "split": "Known"},
    "A13": {"type": "VC",  "acoustic": "Moment Matching",  "vocoder": "WORLD Vocoder", "split": "Known"},
    "A14": {"type": "VC",  "acoustic": "CycleGAN VC",      "vocoder": "WORLD Vocoder", "split": "Known"},
    "A15": {"type": "VC",  "acoustic": "StarGAN VC",       "vocoder": "WORLD Vocoder", "split": "Known"},
    "A16": {"type": "TTS", "acoustic": "Waveform Concatenation", "vocoder": "Direct Concatenation", "split": "Unseen"},
    "A17": {"type": "TTS", "acoustic": "Waveform Filtering",     "vocoder": "Linear Prediction",    "split": "Unseen"},
    "A18": {"type": "TTS", "acoustic": "High-Res Vocoder",       "vocoder": "MagPhase Vocoder",     "split": "Unseen"},
    "A19": {"type": "VC",  "acoustic": "Deep Neural VC",         "vocoder": "Neural WaveNet",       "split": "Unseen"},
    "nat": {"type": "Human", "acoustic": "Studio Natural Speech", "vocoder": "Organic Human Vocal Tract", "split": "Genuine"},
    "librispeech_human": {"type": "Human", "acoustic": "LibriSpeech Clean Speech", "vocoder": "Organic Human Vocal Tract", "split": "Genuine"},
}


def run_generator_benchmark(test_cache_path: Path = FEATURES_DIR / "features_merged_test.npz"):
    print("=" * 75)
    print("OPTION B: MULTI-GENERATOR & UNSEEN-ATTACK GENERALIZATION BENCHMARK")
    print("=" * 75)

    if not test_cache_path.exists():
        print(f"[!] Test cache not found at: {test_cache_path}")
        print("Please wait for feature extraction to complete first.")
        return None

    data = np.load(test_cache_path, allow_pickle=True)
    X_test = data["X"]
    y_test = data["y"]
    system_ids = list(data["system_ids"])
    feat_names = list(data["feature_names"])

    classifier = AntiSpoofClassifier.load(BASELINE_MODEL_PATH)
    tau = classifier.calibrated_threshold
    print(f"[+] Loaded Model: {classifier.model_type} (Features: {len(feat_names)})")
    print(f"[+] Calibrated Operating Threshold: tau = {tau:.4f}\n")

    probs = classifier.predict_proba(X_test)[:, 1]
    preds = (probs >= tau).astype(int)

    # Group by system_id
    systems = np.unique(system_ids)
    benchmark_results = {}

    print(f"{'System':<18} | {'Type':<6} | {'Split':<7} | {'Count':<5} | {'Mean Prob':<9} | {'Detection / Acc':<15} | {'Error Rate'}")
    print("-" * 75)

    known_spoof_total = 0
    known_spoof_detected = 0
    unseen_spoof_total = 0
    unseen_spoof_detected = 0
    genuine_total = 0
    genuine_correct = 0

    for sys_id in sorted(systems):
        mask = (np.array(system_ids) == sys_id)
        sub_y = y_test[mask]
        sub_probs = probs[mask]
        sub_preds = preds[mask]
        n_samples = len(sub_y)
        mean_p = float(np.mean(sub_probs))

        profile = GENERATOR_PROFILES.get(sys_id, {"type": "Unknown", "split": "Unknown", "acoustic": "N/A", "vocoder": "N/A"})
        sys_type = profile["type"]
        sys_split = profile["split"]

        if sys_type == "Human":
            # Genuine speech: error = FAR (falsely called spoof)
            n_correct = int(np.sum(sub_preds == 0))
            n_error = int(np.sum(sub_preds == 1))
            acc = float(n_correct / n_samples)
            err_rate = float(n_error / n_samples)
            rate_label = f"FAR: {err_rate*100:5.2f}%"
            perf_label = f"{acc*100:5.2f}% genuine"

            genuine_total += n_samples
            genuine_correct += n_correct
        else:
            # Spoof speech: detection = Recall (% detected as spoof)
            n_detected = int(np.sum(sub_preds == 1))
            n_missed = int(np.sum(sub_preds == 0))
            recall = float(n_detected / n_samples)
            err_rate = float(n_missed / n_samples)
            rate_label = f"FRR: {err_rate*100:5.2f}%"
            perf_label = f"{recall*100:5.2f}% caught"

            if sys_split == "Known":
                known_spoof_total += n_samples
                known_spoof_detected += n_detected
            else:
                unseen_spoof_total += n_samples
                unseen_spoof_detected += n_detected

        benchmark_results[sys_id] = {
            "type": sys_type,
            "split": sys_split,
            "profile": profile,
            "samples": n_samples,
            "mean_spoof_probability": round(mean_p, 4),
            "error_rate": round(err_rate, 4),
            "correct": n_correct if sys_type == "Human" else n_detected
        }

        print(f"{sys_id:<18} | {sys_type:<6} | {sys_split:<7} | {n_samples:5d} | {mean_p:8.4f}  | {perf_label:<15} | {rate_label}")

    print("-" * 75)
    known_recall = (known_spoof_detected / known_spoof_total * 100) if known_spoof_total > 0 else 0.0
    unseen_recall = (unseen_spoof_detected / unseen_spoof_total * 100) if unseen_spoof_total > 0 else 0.0
    human_acc = (genuine_correct / genuine_total * 100) if genuine_total > 0 else 0.0

    print(f"\n[Summary Generalization Scores]:")
    print(f"  * Real Human Speech Accuracy : {human_acc:.2f}% ({genuine_correct}/{genuine_total}) -> FAR: {100 - human_acc:.2f}%")
    print(f"  * Known Spoof Detection Rate : {known_recall:.2f}% ({known_spoof_detected}/{known_spoof_total})")
    print(f"  * Unseen Spoof Detection Rate: {unseen_recall:.2f}% ({unseen_spoof_detected}/{unseen_spoof_total})")
    print(f"  * Generalization Gap         : {abs(known_recall - unseen_recall):.2f}%")
    print("=" * 75)

    summary_data = {
        "calibrated_threshold": round(tau, 4),
        "genuine_accuracy": round(human_acc, 2),
        "genuine_far": round(100 - human_acc, 2),
        "known_spoofs_detected_pct": round(known_recall, 2),
        "unseen_spoofs_detected_pct": round(unseen_recall, 2),
        "generalization_gap_pct": round(abs(known_recall - unseen_recall), 2),
        "systems": benchmark_results
    }

    out_file = RESULTS_DIR / "generator_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[+] Detailed benchmark report saved to: {out_file}\n")
    return summary_data


if __name__ == "__main__":
    run_generator_benchmark()
