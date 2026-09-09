"""
Neural Raw-Waveform Anti-Spoofing Classifier (Option C).
Implements SincNet-inspired learnable parameterized bandpass filters
followed by temporal residual convolutions, self-attention pooling,
and binary classification head with Apple Silicon MPS hardware acceleration.
"""

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Ensure project root is on sys.path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import SAMPLE_RATE, MODELS_DIR

NEURAL_MODEL_PATH = MODELS_DIR / "neural_sincnet.pt"


def get_device() -> torch.device:
    """Select best available device: Apple Silicon MPS -> CUDA -> CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SincConv1d(nn.Module):
    """
    Parameterized Sinc-based bandpass filter convolution layer.
    Directly operates on raw audio samples and learns bandpass cutoffs.
    """

    def __init__(self, out_channels: int = 64, kernel_size: int = 251, sample_rate: int = 16000):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Initialize bandpass cutoffs linearly in Mel scale
        low_hz = 30.0
        high_hz = sample_rate / 2.0 - (sample_rate / kernel_size)
        mel_low = 2595.0 * math.log10(1.0 + low_hz / 700.0)
        mel_high = 2595.0 * math.log10(1.0 + high_hz / 700.0)
        mels = np.linspace(mel_low, mel_high, out_channels + 1)
        hz = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

        band_low = hz[:-1]
        band_width = np.diff(hz)

        self.f1 = nn.Parameter(torch.Tensor(band_low).view(-1, 1))
        self.band_width = nn.Parameter(torch.Tensor(band_width).view(-1, 1))

        # Half-window time axis
        t = torch.linspace(1, (kernel_size - 1) / 2, steps=int((kernel_size - 1) / 2)) / sample_rate
        self.register_buffer("t", t.view(1, -1))

        # Hamming window
        window = 0.54 - 0.46 * torch.cos(2.0 * math.pi * torch.arange(kernel_size) / float(kernel_size))
        self.register_buffer("window", window.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, 1, samples]"""
        f1 = torch.abs(self.f1)
        f2 = f1 + torch.abs(self.band_width)

        f1_t = 2.0 * math.pi * f1 * self.t
        f2_t = 2.0 * math.pi * f2 * self.t

        # Sinc bandpass formula
        band_pass_left = (torch.sin(f2_t) - torch.sin(f1_t)) / (math.pi * self.t * self.sample_rate)
        band_pass_center = 2.0 * (f2 - f1) / self.sample_rate
        band_pass_right = torch.flip(band_pass_left, dims=[-1])

        filters = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=-1)
        filters = filters * self.window.view(1, -1)
        filters = filters.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(x, filters, stride=4, padding=self.kernel_size // 2)


class SincNetClassifier(nn.Module):
    """
    Lightweight Deep Raw-Audio Anti-Spoofing Architecture.
    SincConv1d -> 3x ConvBlocks -> Attentive Statistics Pooling -> Dense -> Spoof Prob.
    """

    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = sample_rate

        self.sinc_conv = SincConv1d(out_channels=48, kernel_size=129, sample_rate=sample_rate)
        self.bn0 = nn.BatchNorm1d(48)

        # 3 Downsampling Residual Blocks
        self.conv1 = nn.Conv1d(48, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # Attentive Statistics Pooling
        self.att_conv = nn.Conv1d(128, 128, kernel_size=1)
        self.classifier_head = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # Single binary logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: [batch, samples] or [batch, 1, samples]
        Output: logits [batch, 1]
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)

        h = F.leaky_relu(self.bn0(self.sinc_conv(x)), 0.2)
        h = F.leaky_relu(self.bn1(self.conv1(h)), 0.2)
        h = F.leaky_relu(self.bn2(self.conv2(h)), 0.2)
        h = F.leaky_relu(self.bn3(self.conv3(h)), 0.2)

        # Attention weights
        w = F.softmax(self.att_conv(h), dim=-1)
        mean = torch.sum(h * w, dim=-1)
        var = torch.sum((h ** 2) * w, dim=-1) - (mean ** 2)
        std = torch.sqrt(torch.clamp(var, min=1e-5))

        pooled = torch.cat([mean, std], dim=-1)
        logits = self.classifier_head(pooled)
        return logits

    def predict_spoof_prob(self, audio_array: np.ndarray, device: Optional[torch.device] = None) -> float:
        """Convenience method for single audio array inference."""
        if device is None:
            device = get_device()
        self.eval()
        self.to(device)

        # Standardize length to 3.0 seconds (48,000 samples)
        target_len = self.sample_rate * 3
        if len(audio_array) < target_len:
            audio_array = np.pad(audio_array, (0, target_len - len(audio_array)))
        else:
            audio_array = audio_array[:target_len]

        tensor = torch.from_numpy(audio_array).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logit = self.forward(tensor)
            prob = torch.sigmoid(logit).item()
        return float(prob)

    def save(self, filepath: Path = NEURAL_MODEL_PATH):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "sample_rate": self.sample_rate
        }, filepath)

    @classmethod
    def load(cls, filepath: Path = NEURAL_MODEL_PATH, device: Optional[torch.device] = None) -> "SincNetClassifier":
        if device is None:
            device = get_device()
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(sample_rate=checkpoint.get("sample_rate", 16000))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
