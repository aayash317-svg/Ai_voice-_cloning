"""
Training Pipeline for SincNet Deep Raw-Waveform Anti-Spoofing Model (Option C).
Trains on Apple Silicon MPS with batching and early stopping on Dev EER.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import SAMPLE_RATE, MODELS_DIR, RESULTS_DIR
from backend.dataset_loader import DatasetLoader, AudioSample
from backend.audio_preprocessing import AudioPreprocessor
from backend.neural_classifier import SincNetClassifier, get_device, NEURAL_MODEL_PATH


class RawAudioDataset(Dataset):
    """PyTorch Dataset that loads raw mono audio waveforms on the fly."""

    def __init__(self, samples: List[AudioSample], sample_rate: int = 16000, max_seconds: float = 3.0):
        self.samples = samples
        self.sr = sample_rate
        self.target_len = int(sample_rate * max_seconds)
        self.preprocessor = AudioPreprocessor(target_sr=sample_rate)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        try:
            audio = self.preprocessor.load_audio(sample.file_path)
            audio = self.preprocessor.normalize_amplitude(audio)
        except Exception:
            audio = np.zeros(self.target_len, dtype=np.float32)

        # Pad or slice to target length
        if len(audio) < self.target_len:
            audio = np.pad(audio, (0, self.target_len - len(audio)))
        else:
            audio = audio[:self.target_len]

        x = torch.from_numpy(audio).float()
        y = torch.tensor(sample.label, dtype=torch.float32)
        return x, y


def train_neural_model(epochs: int = 5, batch_size: int = 32, lr: float = 1e-3):
    print("=" * 70)
    print("OPTION C: SINC-NET DEEP RAW-WAVEFORM MODEL TRAINING")
    print("=" * 70)

    device = get_device()
    print(f"[+] Active Acceleration Engine: {device}")

    loader = DatasetLoader()
    samples = loader.load_samples()
    splits = loader.create_splits(train_ratio=0.70, dev_ratio=0.15, test_ratio=0.15)

    # Balance train split to 1:1 ratio with 50% ASVspoof and 50% ChatterboxTTS
    rng = np.random.RandomState(42)
    train_genuine = [s for s in splits["train"] if s.label == 0]
    train_spoof = [s for s in splits["train"] if s.label == 1]
    n_bal = len(train_genuine)

    asv_spoof = [s for s in train_spoof if s.system_id != "chatterbox_tts"]
    cb_spoof = [s for s in train_spoof if s.system_id == "chatterbox_tts"]
    n_each = n_bal // 2
    selected_asv = list(rng.choice(asv_spoof, size=min(n_each, len(asv_spoof)), replace=False))
    selected_cb = list(rng.choice(cb_spoof, size=n_bal - len(selected_asv), replace=False))
    selected_spoof = selected_asv + selected_cb

    train_balanced_samples = train_genuine + selected_spoof
    rng.shuffle(train_balanced_samples)

    print(f"[+] Balanced Train Split: {len(train_balanced_samples)} audio clips ({len(train_genuine)} Real / {len(selected_spoof)} Fake [ASV: {len(selected_asv)}, Chatterbox: {len(selected_cb)}])")
    print(f"[+] Dev Split           : {len(splits['dev'])} audio clips")
    print(f"[+] Test Split          : {len(splits['test'])} audio clips")

    train_ds = RawAudioDataset(train_balanced_samples, sample_rate=SAMPLE_RATE)
    dev_ds = RawAudioDataset(splits["dev"], sample_rate=SAMPLE_RATE)
    test_ds = RawAudioDataset(splits["test"], sample_rate=SAMPLE_RATE)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SincNetClassifier(sample_rate=SAMPLE_RATE).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_dev_eer = 1.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / len(train_loader)

        # Evaluate on DEV set
        model.eval()
        dev_probs = []
        dev_targets = []
        with torch.no_grad():
            for x, y in dev_loader:
                x = x.to(device)
                logits = model(x).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                dev_probs.extend(probs)
                dev_targets.extend(y.numpy())

        dev_probs = np.array(dev_probs)
        dev_targets = np.array(dev_targets)

        fpr, tpr, _ = roc_curve(dev_targets, dev_probs, pos_label=1)
        fnr = 1.0 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        dev_eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
        dev_auc = float(roc_auc_score(dev_targets, dev_probs))

        duration = round(time.time() - t0, 1)
        print(f"Epoch {epoch:02d} [{duration}s] | Train Loss: {train_loss:.4f} | Dev AUC: {dev_auc:.4f} | Dev EER: {dev_eer*100:.2f}%")

        if dev_eer < best_dev_eer:
            best_dev_eer = dev_eer
            model.save(NEURAL_MODEL_PATH)
            print(f"  [*] Best checkpoint saved -> Dev EER: {dev_eer*100:.2f}%")

    print(f"\n[+] Neural Training Complete. Model persisted to: {NEURAL_MODEL_PATH}")


if __name__ == "__main__":
    train_neural_model()
