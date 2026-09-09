"""
Audio Preprocessing Pipeline for Voice Integrity Verification Framework.
Handles audio loading, mono conversion, amplitude normalization,
and validation as specified in training_file.md Section 5.
"""

from pathlib import Path
from typing import Generator, Optional, Union
import numpy as np
import librosa
import soundfile as sf

from backend.config import SAMPLE_RATE, WINDOW_DURATION_SEC, WINDOW_HOP_SEC, WINDOW_SAMPLES, HOP_SAMPLES


class AudioPreprocessor:
    """Preprocesses audio waveforms for feature extraction."""

    def __init__(self, target_sr: int = SAMPLE_RATE):
        self.target_sr = target_sr

    def load_audio(self, source: Union[str, Path, bytes, np.ndarray], sr: Optional[int] = None) -> np.ndarray:
        """
        Load audio from file path, raw array, or byte buffer.
        Guarantees 1D mono float32 array resampled to target_sr.
        """
        if isinstance(source, (str, Path)):
            try:
                audio, loaded_sr = librosa.load(str(source), sr=self.target_sr, mono=True)
            except Exception:
                # Fallback to soundfile for faster wav reading
                data, loaded_sr = sf.read(str(source))
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                if loaded_sr != self.target_sr:
                    data = librosa.resample(data.astype(np.float32), orig_sr=loaded_sr, target_sr=self.target_sr)
                audio = data
            return audio.astype(np.float32)

        elif isinstance(source, np.ndarray):
            audio = source.astype(np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr is not None and sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            return audio

        elif isinstance(source, (bytes, bytearray)):
            audio = np.frombuffer(source, dtype=np.int16).astype(np.float32) / 32768.0
            if sr is not None and sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            return audio

        else:
            raise ValueError(f"Unsupported audio source type: {type(source)}")

    def normalize_amplitude(self, audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """Normalize amplitude to target peak level to prevent scale discrepancies."""
        max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
        if max_val > 1e-6:
            return (audio / max_val) * target_peak
        return audio

    def remove_silence(self, audio: np.ndarray, top_db: float = 30.0) -> np.ndarray:
        """Trim silence intervals using energy-based threshold."""
        if len(audio) == 0:
            return audio
        intervals = librosa.effects.split(audio, top_db=top_db)
        if len(intervals) == 0:
            return audio
        return np.concatenate([audio[s:e] for s, e in intervals])

    def preprocess_file(self, file_path: Union[str, Path], trim_silence: bool = False) -> np.ndarray:
        """Full file preprocessing pipeline: load -> mono -> normalize."""
        y = self.load_audio(file_path)
        y = self.normalize_amplitude(y)
        if trim_silence:
            y = self.remove_silence(y)
        return y

    def generate_sliding_windows(
        self,
        audio: np.ndarray,
        window_size: int = WINDOW_SAMPLES,
        hop_size: int = HOP_SAMPLES
    ) -> Generator[np.ndarray, None, None]:
        """Generate overlapping sliding windows (e.g. 3.0s window with 1.0s hop)."""
        if len(audio) < window_size:
            padded = np.pad(audio, (0, window_size - len(audio)), mode='constant')
            yield padded
            return

        for start_idx in range(0, len(audio) - window_size + 1, hop_size):
            yield audio[start_idx : start_idx + window_size]


# Alias for backward compatibility
preprocess_pipeline = AudioPreprocessor().preprocess_file
