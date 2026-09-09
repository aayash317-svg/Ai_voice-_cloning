"""
Configuration module for Voice Integrity Verification Framework.
Provides centralized path management, audio processing parameters,
risk engine thresholds, and security settings.
"""

from pathlib import Path
import os

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset paths
DATASETS_DIR = Path(os.getenv("VOICE_DATASETS_DIR", BASE_DIR / "datasets"))
ASVSPOOF_ADJUSTED_WAV_DIR = DATASETS_DIR / "ASVspoof2019" / "ASVSpoof2019-sets-adjusted" / "data" / "wav"
ASVSPOOF2019_DIR = DATASETS_DIR / "ASVspoof2019" / "LA"

# Output & Artifact Directories
MODELS_DIR = Path(os.getenv("VOICE_MODELS_DIR", BASE_DIR / "models"))
RESULTS_DIR = Path(os.getenv("VOICE_RESULTS_DIR", BASE_DIR / "results"))
FEATURES_DIR = Path(os.getenv("VOICE_FEATURES_DIR", BASE_DIR / "features"))
EXPERIMENTS_DIR = Path(os.getenv("VOICE_EXPERIMENTS_DIR", BASE_DIR / "experiments"))
LOGS_DIR = Path(os.getenv("VOICE_LOGS_DIR", BASE_DIR / "logs"))

# Model checkpoints
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_random_forest.pkl"

# Audio Processing Parameters
SAMPLE_RATE = 16000               # Standard 16 kHz sample rate for anti-spoofing
WINDOW_DURATION_SEC = 3.0         # 3.0s sliding analysis window
WINDOW_HOP_SEC = 1.0              # 1.0s hop for overlapping continuous analysis
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION_SEC)
HOP_SAMPLES = int(SAMPLE_RATE * WINDOW_HOP_SEC)

# Feature Extraction Settings
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 80
N_MFCC = 20

# Feature Signal Weights in Risk Engine
ACOUSTIC_WEIGHT = 0.30
SPECTRAL_WEIGHT = 0.35
PROSODY_WEIGHT = 0.20
SPEAKER_WEIGHT = 0.15

# Risk Engine Thresholds (0 - 100)
# 0-30: LOW RISK, 31-60: MEDIUM RISK, 61-80: HIGH RISK, 81-100: CRITICAL RISK
RISK_THRESHOLD_LOW = 30
RISK_THRESHOLD_MEDIUM = 60
RISK_THRESHOLD_HIGH = 80
RISK_THRESHOLD_CRITICAL = 95

# Privacy & Security
AUDIO_BUFFER_MAX_SECONDS = 5.0
ENCRYPTION_KEY_ENV = "VOICE_ENCRYPTION_KEY"
AUDIT_CHAIN_FILE = LOGS_DIR / "audit_chain.json"

# Ensure all runtime directories exist
for directory in [DATASETS_DIR, MODELS_DIR, RESULTS_DIR, FEATURES_DIR, EXPERIMENTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
