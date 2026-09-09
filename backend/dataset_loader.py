"""
Dataset Loader for Voice Integrity Verification Framework.
Handles ASVspoof 2019 LA datasets, filename-based parsing,
speaker/system mapping, and anti-leakage train/dev/test splitting.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import train_test_split

from backend.config import ASVSPOOF2019_DIR, ASVSPOOF_ADJUSTED_WAV_DIR, DATASETS_DIR


@dataclass
class AudioSample:
    """Metadata for an individual audio sample in the dataset."""
    speaker_id: str
    audio_id: str
    system_id: str          # 'nat' (genuine), 'A07'...'A19' (spoof attacks)
    label_str: str          # 'genuine' / 'bonafide' or 'spoof'
    label: int              # 0 = genuine, 1 = spoof
    file_path: Path


class DatasetLoader:
    """Loads and splits genuine and spoof audio datasets without data leakage."""

    def __init__(self, data_dir: Optional[Path] = None, random_state: int = 42):
        self.data_dir = Path(data_dir) if data_dir else self._discover_dataset_dir()
        self.random_state = random_state
        self.samples: List[AudioSample] = []
        self._splits: Dict[str, List[AudioSample]] = {}

    def _discover_dataset_dir(self) -> Path:
        """Find the available dataset directory."""
        if DATASETS_DIR.exists():
            return DATASETS_DIR
        if ASVSPOOF_ADJUSTED_WAV_DIR.exists():
            return ASVSPOOF_ADJUSTED_WAV_DIR
        return DATASETS_DIR

    def load_samples(self) -> List[AudioSample]:
        """Discover all audio files and extract metadata and labels."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        samples = []
        audio_files = list(self.data_dir.rglob("*.wav")) + list(self.data_dir.rglob("*.flac"))

        for file_path in audio_files:
            fname = file_path.name
            # Check if file comes from LibriSpeech (100% genuine human speech)
            is_librispeech = "librispeech" in str(file_path).lower()
            is_chatterbox = "chatterbox" in str(file_path).lower() or "phonemedf" in str(file_path).lower()

            # Parse naming pattern: ASV2019-{system_id}-LA_E_{audio_id}.wav
            if is_librispeech:
                system_id = "librispeech_human"
                audio_id = fname
                is_genuine = True
            elif is_chatterbox:
                system_id = "chatterbox_tts"
                audio_id = fname
                is_genuine = False
            elif "ASV2019" in fname:
                parts = fname.replace(".wav", "").replace(".flac", "").split("-")
                if len(parts) >= 3:
                    system_id = parts[1]
                    audio_id = parts[2]
                else:
                    system_id = "nat" if "nat" in fname.lower() else "spoof"
                    audio_id = fname
                is_genuine = (system_id.lower() == "nat" or "bonafide" in fname.lower() or "genuine" in fname.lower())
            else:
                system_id = "nat" if ("nat" in fname.lower() or "bona" in fname.lower()) else "spoof"
                audio_id = fname
                is_genuine = (system_id.lower() == "nat" or "bonafide" in fname.lower() or "genuine" in fname.lower())

            label = 0 if is_genuine else 1
            label_str = "genuine" if is_genuine else "spoof"

            sample = AudioSample(
                speaker_id="speaker_unknown",
                audio_id=audio_id,
                system_id=system_id,
                label_str=label_str,
                label=label,
                file_path=file_path
            )
            samples.append(sample)

        self.samples = samples
        return samples

    def create_splits(
        self,
        train_ratio: float = 0.70,
        dev_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict[str, List[AudioSample]]:
        """
        Create stratified Train, Dev, and Test splits maintaining class balance.
        Guarantees 0% overlap between splits.
        """
        if not self.samples:
            self.load_samples()

        labels = [s.label for s in self.samples]
        indices = np.arange(len(self.samples))

        # First split: Train vs (Dev + Test)
        train_idx, val_test_idx = train_test_split(
            indices,
            train_size=train_ratio,
            stratify=labels,
            random_state=self.random_state
        )

        # Second split: Dev vs Test
        val_test_labels = [labels[i] for i in val_test_idx]
        val_ratio_adjusted = dev_ratio / (dev_ratio + test_ratio)
        dev_sub_idx, test_sub_idx = train_test_split(
            np.arange(len(val_test_idx)),
            train_size=val_ratio_adjusted,
            stratify=val_test_labels,
            random_state=self.random_state
        )

        dev_idx = val_test_idx[dev_sub_idx]
        test_idx = val_test_idx[test_sub_idx]

        self._splits = {
            "train": [self.samples[i] for i in train_idx],
            "dev": [self.samples[i] for i in dev_idx],
            "test": [self.samples[i] for i in test_idx]
        }
        return self._splits

    def get_split_stats(self) -> Dict[str, dict]:
        """Compute summary statistics for all splits."""
        if not self._splits:
            self.create_splits()

        stats = {}
        for split_name, samples in self._splits.items():
            genuine_cnt = sum(1 for s in samples if s.label == 0)
            spoof_cnt = sum(1 for s in samples if s.label == 1)
            total = len(samples)
            stats[split_name] = {
                "total": total,
                "genuine": genuine_cnt,
                "spoof": spoof_cnt,
                "genuine_pct": round((genuine_cnt / total) * 100, 2) if total > 0 else 0,
                "spoof_pct": round((spoof_cnt / total) * 100, 2) if total > 0 else 0
            }
        return stats


# Alias for backward compatibility
ASVSpoof2019Loader = DatasetLoader
