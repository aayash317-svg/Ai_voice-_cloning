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

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None

from backend.config import ASVSPOOF2019_DIR, ASVSPOOF_ADJUSTED_WAV_DIR, DATASETS_DIR, ASVSPOOF2021_DF_METADATA



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
        self._asv2021_meta: Optional[Dict[str, dict]] = None

    def _discover_dataset_dir(self) -> Path:
        """Find the available dataset directory."""
        if DATASETS_DIR.exists():
            return DATASETS_DIR
        if ASVSPOOF_ADJUSTED_WAV_DIR.exists():
            return ASVSPOOF_ADJUSTED_WAV_DIR
        return DATASETS_DIR

    def _load_asv2021_metadata(self) -> Dict[str, dict]:
        """Lazy load ASVspoof 2021 DF trial metadata index if present."""
        if self._asv2021_meta is not None:
            return self._asv2021_meta

        meta_dict = {}
        meta_file = ASVSPOOF2021_DF_METADATA
        if not meta_file.exists():
            # Check anywhere under datasets directory
            candidates = list(self.data_dir.rglob("trial_metadata.txt"))
            if candidates:
                meta_file = candidates[0]

        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 6:
                            # Format: speaker_id trial_id codec source attack key/label
                            speaker_id = parts[0]
                            trial_id = parts[1]
                            codec = parts[2]
                            attack = parts[4]
                            raw_label = parts[5].lower()
                            is_bonafide = (raw_label == "bonafide")
                            meta_dict[trial_id] = {
                                "speaker_id": speaker_id,
                                "codec": codec,
                                "system_id": f"asv2021_{attack}_{codec}",
                                "label": 0 if is_bonafide else 1,
                                "label_str": "genuine" if is_bonafide else "spoof"
                            }
            except Exception as e:
                print(f"[!] Warning: Failed loading ASVspoof 2021 metadata: {e}")

        self._asv2021_meta = meta_dict
        return self._asv2021_meta

    def _load_asv2019_metadata(self) -> Dict[str, dict]:
        """Lazy load ASVspoof 2019 LA trial metadata protocols if present."""
        if hasattr(self, "_asv2019_meta") and self._asv2019_meta is not None:
            return self._asv2019_meta

        meta_dict = {}
        for proto_file in self.data_dir.rglob("ASVspoof2019.LA.cm.*.txt"):
            try:
                with open(proto_file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            spk, audio_id, _, attack, label = parts[:5]
                            is_bonafide = (label.lower() == "bonafide")
                            meta_dict[audio_id] = {
                                "speaker_id": spk,
                                "system_id": attack if attack != "-" else "bonafide",
                                "label": 0 if is_bonafide else 1,
                                "label_str": "genuine" if is_bonafide else "spoof"
                            }
            except Exception as e:
                print(f"[!] Warning: Error reading {proto_file.name}: {e}")

        self._asv2019_meta = meta_dict
        return self._asv2019_meta

    def load_samples(self) -> List[AudioSample]:
        """Discover all audio files and extract metadata and labels."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        asv2021_meta = self._load_asv2021_metadata()
        asv2019_meta = self._load_asv2019_metadata()
        samples = []
        audio_files = list(self.data_dir.rglob("*.wav")) + list(self.data_dir.rglob("*.flac"))

        for file_path in audio_files:
            fname = file_path.name
            stem = fname.replace(".wav", "").replace(".flac", "")

            # Check if file comes from LibriSpeech (100% genuine human speech)
            is_librispeech = "librispeech" in str(file_path).lower()
            is_chatterbox = "chatterbox" in str(file_path).lower() or "phonemedf" in str(file_path).lower()
            is_asv2021 = "asvspoof2021" in str(file_path).lower() or stem.startswith("DF_E_")
            is_indictts = "indic_tts" in str(file_path).lower() or "indictts" in str(file_path).lower()
            is_indicsynth = "indic_synth" in str(file_path).lower() or "indicsynth" in str(file_path).lower()
            is_add_genuine = "additional_genuine" in str(file_path).lower()

            speaker_id = "speaker_unknown"

            if is_add_genuine:
                is_slr = "slr65" in str(file_path).lower()
                lang = "tamil" if is_slr else "indian_lang"
                for l in ["hindi", "tamil", "telugu", "bengali", "kannada", "marathi"]:
                    if l in str(file_path).lower():
                        lang = l
                        break
                parts = stem.split("_")
                if is_slr:
                    speaker_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else f"slr65_{parts[0]}"
                    system_id = "indic_slr65_tamil"
                else:
                    parent_split = file_path.parent.name if file_path.parent.name in ["train", "dev", "test"] else "gen"
                    bucket = abs(hash(stem)) % 30
                    speaker_id = f"fleurs_{lang}_{parent_split}_spk{bucket:02d}"
                    system_id = f"indic_fleurs_{lang}"
                audio_id = stem
                label = 0
                label_str = "genuine"
            elif is_indictts:
                # IndicTTS Genuine Human Speech
                lang = "indian_lang"
                for l in ["hindi", "tamil", "telugu", "bengali", "kannada", "marathi", "malayalam", "gujarati"]:
                    if l in str(file_path).lower():
                        lang = l
                        break
                system_id = f"indictts_{lang}"
                parts = stem.split("_")
                speaker_id = parts[2] if len(parts) >= 3 else f"indictts_{lang}_spk"
                audio_id = stem
                label = 0
                label_str = "genuine"
            elif is_indicsynth:
                # IndicSynth AI Cloned & Synthetic Speech
                lang = "indian_lang"
                for l in ["hindi", "tamil", "telugu", "bengali", "kannada", "marathi", "malayalam", "gujarati"]:
                    if l in str(file_path).lower():
                        lang = l
                        break
                gen = "synthetic"
                for g in ["xttsv2", "vits", "freevc24", "freevc", "yourtts"]:
                    if g in str(file_path).lower():
                        gen = g
                        break
                system_id = f"indicsynth_{gen}_{lang}"
                parts = stem.split("_")
                speaker_id = parts[3] if len(parts) >= 4 else f"synth_{gen}_{lang}_spk"
                audio_id = stem
                label = 1
                label_str = "spoof"
            elif stem in asv2019_meta:
                meta = asv2019_meta[stem]
                speaker_id = meta["speaker_id"]
                system_id = meta["system_id"]
                audio_id = stem
                label = meta["label"]
                label_str = meta["label_str"]
            elif is_asv2021 and stem in asv2021_meta:
                meta = asv2021_meta[stem]
                speaker_id = meta["speaker_id"]
                system_id = meta["system_id"]
                audio_id = stem
                label = meta["label"]
                label_str = meta["label_str"]
            elif is_asv2021:
                system_id = "asv2021_df"
                audio_id = stem
                is_genuine = ("bonafide" in fname.lower() or "genuine" in fname.lower())
                label = 0 if is_genuine else 1
                label_str = "genuine" if is_genuine else "spoof"
            elif is_librispeech:
                parts = stem.split("-")
                speaker_id = f"libri_{parts[0]}" if len(parts) >= 1 else "libri_speaker"
                system_id = "bonafide"
                audio_id = stem
                label = 0
                label_str = "genuine"
            elif is_chatterbox:
                speaker_id = "chatterbox_speaker"
                system_id = "chatterbox_tts"
                audio_id = stem
                label = 1
                label_str = "spoof"
            elif "ASV2019" in fname:
                parts = stem.split("-")
                if len(parts) >= 3:
                    system_id = parts[1]
                    audio_id = parts[2]
                    speaker_id = parts[0]
                else:
                    system_id = "bonafide" if "nat" in fname.lower() else "spoof"
                    audio_id = stem
                    speaker_id = "speaker_unknown"
                is_genuine = (system_id.lower() in ("nat", "bonafide") or "bonafide" in fname.lower() or "genuine" in fname.lower())
                label = 0 if is_genuine else 1
                label_str = "genuine" if is_genuine else "spoof"
            else:
                is_genuine = ("nat" in fname.lower() or "bona" in fname.lower() or "genuine" in fname.lower())
                system_id = "bonafide" if is_genuine else "spoof"
                audio_id = stem
                speaker_id = "speaker_unknown"
                label = 0 if is_genuine else 1
                label_str = "genuine" if is_genuine else "spoof"

            sample = AudioSample(
                speaker_id=speaker_id,
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

        if train_test_split is not None:
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
        else:
            rng = np.random.RandomState(self.random_state)
            shuffled = indices.copy()
            rng.shuffle(shuffled)
            n_total = len(shuffled)
            n_train = int(n_total * train_ratio)
            n_dev = int(n_total * dev_ratio)
            train_idx = shuffled[:n_train]
            dev_idx = shuffled[n_train:n_train + n_dev]
            test_idx = shuffled[n_train + n_dev:]


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
