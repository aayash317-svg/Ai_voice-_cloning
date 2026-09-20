"""
Audio Preprocessing Pipeline for Voice Integrity Verification Framework.
Handles audio loading, resampling, peak/RMS normalization,
Voice Activity Detection (VAD), and sliding window segmentation.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

from backend.config import SAMPLE_RATE, WINDOW_DURATION_SEC, WINDOW_HOP_SEC, WINDOW_SAMPLES, HOP_SAMPLES


class AudioPreprocessor:
    """Preprocesses audio streams and files for feature extraction."""

    def __init__(self, target_sr: int = SAMPLE_RATE):
        self.target_sr = target_sr

    def load_audio(self, source: Union[str, Path, bytes, np.ndarray], sr: Optional[int] = None) -> np.ndarray:
        """
        Load audio from file path, raw array, or byte buffer.
        Guarantees 1D mono float32 array resampled to target_sr.
        """
        if isinstance(source, (str, Path)):
            # 1. Try soundfile (fastest for WAV/FLAC/OGG)
            try:
                data, loaded_sr = sf.read(str(source))
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                if loaded_sr != self.target_sr:
                    data = librosa.resample(data.astype(np.float32), orig_sr=loaded_sr, target_sr=self.target_sr)
                return data.astype(np.float32)
            except Exception:
                pass

            # 2. Try imageio_ffmpeg for M4A, AAC, MP4, WebM
            try:
                import imageio_ffmpeg, tempfile, subprocess, os
                exe = imageio_ffmpeg.get_ffmpeg_exe()
                if exe:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                        tmp_out_path = tmp_out.name
                    res = subprocess.run([exe, "-y", "-i", str(source), "-ar", str(self.target_sr), "-ac", "1", "-f", "wav", tmp_out_path], capture_output=True)
                    if res.returncode == 0 and os.path.exists(tmp_out_path):
                        data, loaded_sr = sf.read(tmp_out_path)
                        try: os.unlink(tmp_out_path)
                        except Exception: pass
                        if data.ndim > 1:
                            data = np.mean(data, axis=1)
                        return data.astype(np.float32)
            except Exception:
                pass

            audio, loaded_sr = librosa.load(str(source), sr=self.target_sr, mono=True)
            return audio.astype(np.float32)

        elif isinstance(source, np.ndarray):
            audio = source.astype(np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr is not None and sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            return audio

        elif isinstance(source, (bytes, bytearray)):
            # Decode raw PCM 16-bit or bytes
            audio = np.frombuffer(source, dtype=np.int16).astype(np.float32) / 32768.0
            if sr is not None and sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            return audio

        else:
            raise ValueError(f"Unsupported audio source type: {type(source)}")

    def normalize_audio(self, audio: np.ndarray, target_peak: float = 0.95, min_speech_peak: float = 0.03) -> np.ndarray:
        """
        Normalize audio amplitude to target peak level to ensure consistent signal levels.
        Guards against amplifying low-level room noise/silence into full-scale synthetic hiss.
        """
        if len(audio) == 0:
            return audio
        max_val = float(np.max(np.abs(audio)))
        if max_val < min_speech_peak:
            # Ambient noise / room silence - do NOT boost noise floor
            return audio
        if max_val > 1e-6:
            return (audio / max_val) * target_peak
        return audio

    def apply_vad(self, audio: np.ndarray, top_db: float = 30.0) -> np.ndarray:
        """
        Trim leading, trailing, and long internal non-speech silence intervals.
        """
        if len(audio) == 0:
            return audio

        intervals = librosa.effects.split(audio, top_db=top_db)
        if len(intervals) == 0:
            return audio

        # Concatenate active speech segments
        active_segments = [audio[start:end] for start, end in intervals]
        return np.concatenate(active_segments)

    def generate_sliding_windows(
        self,
        audio: np.ndarray,
        window_size: int = WINDOW_SAMPLES,
        hop_size: int = HOP_SAMPLES
    ) -> Generator[np.ndarray, None, None]:
        """
        Generate overlapping sliding windows (e.g. 3.0s window with 1.0s hop).
        Pads shorter audio segments to at least window_size.
        """
        if len(audio) < window_size:
            # Pad with zeros or repeat
            padded = np.pad(audio, (0, window_size - len(audio)), mode='constant')
            yield padded
            return

        for start_idx in range(0, len(audio) - window_size + 1, hop_size):
            yield audio[start_idx : start_idx + window_size]

    def preprocess_pipeline(self, source: Union[str, Path, np.ndarray], apply_vad_filter: bool = True) -> np.ndarray:
        """Full preprocessing pipeline: Load -> Resample -> Normalize -> VAD."""
        y = self.load_audio(source)
        y = self.normalize_audio(y)
        if apply_vad_filter:
            y = self.apply_vad(y)
        return y
