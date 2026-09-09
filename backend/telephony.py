"""
Telephony Processing & Degradation Simulation Subsystem
Emulates PSTN/VoIP/Cellular channel characteristics (8kHz, G.711 companding, bandpass 300-3400Hz).
Matches ASVspoof 2021 telephony degradation specifications.
"""

import numpy as np
from scipy import signal
from typing import Optional

from backend.app.audio.preprocessing import resample_audio


class TelephonyProcessor:
    """Simulates and processes telephony transmission channels."""

    @staticmethod
    def apply_telephony_bandpass(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Applies 300Hz - 3400Hz Butterworth bandpass filter representing POTS telephony."""
        if len(audio) < 100:
            return audio
        lowcut = 300.0
        highcut = min(3400.0, (sr / 2.0) - 100.0)
        sos = signal.butter(4, [lowcut, highcut], btype="bandpass", fs=sr, output="sos")
        filtered = signal.sosfilt(sos, audio)
        return filtered.astype(np.float32)

    @staticmethod
    def mu_law_quantize(audio: np.ndarray, mu: int = 255) -> np.ndarray:
        """Emulates ITU-T G.711 mu-law 8-bit logarithmic companding."""
        if len(audio) == 0:
            return audio
        # Normalize to [-1.0, 1.0]
        x = np.clip(audio, -1.0, 1.0)
        # Mu-law compression
        companded = np.sign(x) * np.log(1.0 + mu * np.abs(x)) / np.log(1.0 + mu)
        # 8-bit quantization (256 discrete levels)
        quantized = np.round((companded + 1.0) * 127.5) / 127.5 - 1.0
        # Inverse mu-law expansion
        expanded = np.sign(quantized) * (1.0 / mu) * ((1.0 + mu) ** np.abs(quantized) - 1.0)
        return expanded.astype(np.float32)

    @staticmethod
    def a_law_quantize(audio: np.ndarray, a: float = 87.6) -> np.ndarray:
        """Emulates ITU-T G.711 A-law 8-bit companding used in European/Indian telephony."""
        if len(audio) == 0:
            return audio
        x = np.clip(audio, -1.0, 1.0)
        abs_x = np.abs(x)
        companded = np.zeros_like(x)
        
        # Piecewise A-law curve
        idx1 = abs_x < (1.0 / a)
        idx2 = ~idx1
        companded[idx1] = (a * abs_x[idx1]) / (1.0 + np.log(a))
        companded[idx2] = (1.0 + np.log(a * abs_x[idx2])) / (1.0 + np.log(a))
        companded = np.sign(x) * companded

        # 8-bit quantization
        quantized = np.round((companded + 1.0) * 127.5) / 127.5 - 1.0
        abs_q = np.abs(quantized)
        expanded = np.zeros_like(quantized)
        
        # Inverse A-law
        idx1_inv = abs_q < (1.0 / (1.0 + np.log(a)))
        idx2_inv = ~idx1_inv
        expanded[idx1_inv] = (abs_q[idx1_inv] * (1.0 + np.log(a))) / a
        expanded[idx2_inv] = np.exp(abs_q[idx2_inv] * (1.0 + np.log(a)) - 1.0) / a
        expanded = np.sign(quantized) * expanded

        return expanded.astype(np.float32)

    def simulate_pstn_channel(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        codec: str = "a_law",
        add_channel_noise: bool = True,
    ) -> np.ndarray:
        """
        Full telephony degradation pipeline:
        1. Resample to 8kHz telephone bandwidth
        2. Apply 300Hz-3400Hz bandpass filter
        3. Quantize with G.711 compander (A-law or mu-law)
        4. Optionally inject line noise (-35 dB)
        5. Upsample back to 16kHz for model inference
        """
        # Step 1: Downsample to 8000 Hz
        audio_8k = resample_audio(audio, sr, 8000)

        # Step 2: Bandpass filter
        filtered = self.apply_telephony_bandpass(audio_8k, 8000)

        # Step 3: Companding
        if codec == "mu_law":
            quantized = self.mu_law_quantize(filtered)
        else:
            quantized = self.a_law_quantize(filtered)

        # Step 4: Line noise
        if add_channel_noise and len(quantized) > 0:
            noise = np.random.normal(0, 0.005, size=len(quantized)).astype(np.float32)
            quantized = np.clip(quantized + noise, -1.0, 1.0)

        # Step 5: Upsample back to 16000 Hz
        output_16k = resample_audio(quantized, 8000, 16000)
        return output_16k.astype(np.float32)
