"""
Baseline Anti-Spoofing Training & Evaluation Pipeline.
Loads ASVspoof 2019 LA splits, extracts 4-family feature vectors,
trains binary classifier, evaluates EER / FAR / FRR, and saves experiment artifacts.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import ASVSPOOF2019_DIR, EXPERIMENTS_DIR, MODELS_DIR, FEATURES_DIR
from backend.dataset_loader import ASVSpoof2019Loader, AudioSample
from backend.preprocess import AudioPreprocessor
from backend.features import FeatureExtractor
from backend.classifier import AntiSpoofClassifier
from backend.evaluate import evaluate_classifier


def extract_features_for_samples(
    samples: List[AudioSample],
    preprocessor: AudioPreprocessor,
    extractor: FeatureExtractor,
    cache_file: Optional[Path] = None,
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract features for a list of AudioSamples.
    Supports caching to .npz for fast iterative experimentation.
    """
    if cache_file and cache_file.exists():
        print(f"Loading cached features from {cache_file}...")
        data = np.load(cache_file, allow_pickle=True)
        return data["X"], data["y"], list(data["feature_names"])

    valid_samples = [s for s in samples if s.file_path and s.file_path.exists()]
    if max_samples and len(valid_samples) > max_samples:
        # Keep balanced bonafide / spoof ratio when sub-sampling
        bonafide = [s for s in valid_samples if s.label == 0]
        spoof = [s for s in valid_samples if s.label == 1]
        half = max_samples // 2
        valid_samples = bonafide[:half] + spoof[:half]
        print(f"Subsampled {len(valid_samples)} audio files ({len(bonafide[:half])} bonafide, {len(spoof[:half])} spoof).")

    if not valid_samples:
        raise ValueError("No valid audio files found on disk. Please verify dataset download / paths.")

    feature_list = []
    labels = []
    feature_names = None

    print(f"Extracting features for {len(valid_samples)} audio files...")
    for sample in tqdm(valid_samples, desc="Processing Audio"):
        try:
            audio = preprocessor.preprocess_pipeline(sample.file_path)
            feat_dict = extractor.extract_all(audio)
            
            if feature_names is None:
                feature_names = sorted(feat_dict.keys())
                
            vec = extractor.to_vector(feat_dict, feature_names)
            feature_list.append(vec)
            labels.append(sample.label)
        except Exception as e:
            print(f"[!] Error processing {sample.file_path}: {e}")

    X = np.array(feature_list, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, X=X, y=y, feature_names=feature_names)
        print(f"Saved extracted features to cache: {cache_file}")

    return X, y, feature_names


def run_experiment(
    experiment_id: str = "experiment_001",
    model_type: str = "random_forest",
    dataset_dir: Optional[Path] = None,
    max_train_samples: Optional[int] = None,
    max_dev_samples: Optional[int] = None,
    use_cache: bool = True
):
    """Run complete training, evaluation, and logging pipeline for an experiment."""
    start_time = time.time()
    exp_dir = EXPERIMENTS_DIR / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT: {experiment_id} (Model: {model_type})")
    print(f"Output Directory  : {exp_dir}")
    print(f"{'='*60}\n")

    # 1. Dataset Loader & Integrity Verification
    loader = ASVSpoof2019Loader(dataset_dir)
    train_samples = loader.load_split("train")
    dev_samples = loader.load_split("dev")

    is_clean, leakage_info = loader.verify_no_data_leakage()
    print(f"[+] Dataset Split Integrity: {'CLEAN (No Leakage)' if is_clean else 'WARNING (Leakage Detected)'}")
    print(f"    Train Samples: {len(train_samples)} (Bonafide: {sum(1 for s in train_samples if s.label == 0)}, Spoof: {sum(1 for s in train_samples if s.label == 1)})")
    print(f"    Dev Samples  : {len(dev_samples)} (Bonafide: {sum(1 for s in dev_samples if s.label == 0)}, Spoof: {sum(1 for s in dev_samples if s.label == 1)})")

    # 2. Preprocessing & Feature Extraction
    preprocessor = AudioPreprocessor()
    extractor = FeatureExtractor()

    train_cache = FEATURES_DIR / f"asvspoof2019_train_features_{max_train_samples or 'full'}.npz" if use_cache else None
    dev_cache = FEATURES_DIR / f"asvspoof2019_dev_features_{max_dev_samples or 'full'}.npz" if use_cache else None

    X_train, y_train, feature_names = extract_features_for_samples(
        train_samples, preprocessor, extractor, cache_file=train_cache, max_samples=max_train_samples
    )
    X_dev, y_dev, _ = extract_features_for_samples(
        dev_samples, preprocessor, extractor, cache_file=dev_cache, max_samples=max_dev_samples
    )

    print(f"\nFeature matrix shape: Train={X_train.shape}, Dev={X_dev.shape}")
    print(f"Number of feature dimensions: {len(feature_names)}")

    # 3. Train Model
    print(f"\nTraining {model_type} classifier...")
    classifier = AntiSpoofClassifier(model_type=model_type)
    classifier.train(X_train, y_train, feature_names=feature_names)

    # 4. Save Model
    model_path = exp_dir / "model" / f"{model_type}_model.joblib"
    classifier.save(model_path)
    print(f"Saved trained model checkpoint to: {model_path}")

    # 5. Evaluate on Dev Set
    print("\nEvaluating on Development Set...")
    dev_preds = classifier.predict(X_dev)
    dev_probs = classifier.predict_proba(X_dev)[:, 1]

    metrics = evaluate_classifier(
        y_true=y_dev,
        y_pred=dev_preds,
        y_prob=dev_probs,
        output_dir=exp_dir,
        save_plots=True
    )

    # 6. Save Config and Metadata
    config_data = {
        "experiment_id": experiment_id,
        "model_type": model_type,
        "dataset": "ASVspoof2019_LA",
        "train_samples_used": int(len(X_train)),
        "dev_samples_used": int(len(X_dev)),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "leakage_info": leakage_info,
        "duration_seconds": round(time.time() - start_time, 2)
    }

    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT RESULTS: {experiment_id}")
    print(f"{'='*60}")
    print(f"Equal Error Rate (EER) : {metrics['equal_error_rate_eer'] * 100:.2f}%")
    print(f"Accuracy               : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision              : {metrics['precision'] * 100:.2f}%")
    print(f"Recall                 : {metrics['recall'] * 100:.2f}%")
    print(f"F1 Score               : {metrics['f1_score'] * 100:.2f}%")
    print(f"FAR (False Alarm)      : {metrics['false_acceptance_rate_far'] * 100:.2f}%")
    print(f"FRR (Missed Spoof)     : {metrics['false_rejection_rate_frr'] * 100:.2f}%")
    print(f"Metrics & plots saved to: {exp_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Voice Integrity Baseline Training")
    parser.add_argument("--exp-id", type=str, default="experiment_001", help="Experiment ID")
    parser.add_argument("--model", type=str, default="random_forest", choices=["random_forest", "xgboost", "logistic_regression"])
    parser.add_argument("--dataset-dir", type=str, default=None, help="Path to ASVspoof 2019 LA directory")
    parser.add_argument("--max-train", type=int, default=None, help="Subsample train set for fast verification")
    parser.add_argument("--max-dev", type=int, default=None, help="Subsample dev set for fast verification")
    parser.add_argument("--no-cache", action="store_true", help="Do not use cached features")

    args = parser.parse_args()
    run_experiment(
        experiment_id=args.exp_id,
        model_type=args.model,
        dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
        max_train_samples=args.max_train,
        max_dev_samples=args.max_dev,
        use_cache=not args.no_cache
    )


if __name__ == "__main__":
    main()
