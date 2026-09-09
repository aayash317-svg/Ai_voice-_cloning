# Dataset Inventory & Training Methodology

The **Voice Integrity Verification Framework** is trained, calibrated, and evaluated across **38,239 audio clips** combining genuine human speech and multi-generator synthetic voices:

| Dataset | Type | Sample Count | Audio Format | Primary Purpose & Description |
| :--- | :--- | :--- | :--- | :--- |
| **LibriSpeech (test-clean)** | Genuine Human | **2,620** clips | 16 kHz FLAC | Natural, high-fidelity clean human speech baseline from diverse speakers. |
| **ASVspoof 2019 (LA Bonafide)** | Genuine Human | **2,680** clips | 16 kHz WAV | Telephone & voice verification bonafide human recordings. |
| **ASVspoof 2019 (LA Spoofs A07–A19)** | Synthetic Spoofs | **15,399** clips | 16 kHz WAV | 13 diverse algorithms: neural vocoders, waveform concatenation, WaveNet, deep neural voice conversion. |
| **PhonemeDF (ChatterboxTTS)** | Modern Neural TTS | **17,540** clips | 16 kHz WAV | Modern phoneme-level neural text-to-speech synthetic voices. |
| **Total Active Dataset** | **Multi-Source** | **38,239** clips | **16 kHz PCM** | **5,300 Genuine Human / 32,939 Synthetic Spoof Clips** |

---

## Bundled Sample Testing Data (`test_samples/`)

The repository includes pre-packaged testing samples directly in `test_samples/` for immediate out-of-the-box evaluation without needing to download multi-gigabyte archives:

- `test_samples/genuine_sample.wav` — Clean human speech reference
- `test_samples/test_genuine.m4a` / `.wav` — Natural human speech recording
- `test_samples/spoof_sample.wav` — ASVspoof synthetic voice sample
- `test_samples/chatterbox_tts_sample.wav` — PhonemeDF ChatterboxTTS neural synthetic voice

---

## How We Train the Datasets

To guarantee scientific honesty, prevent data leakage, and eliminate false alarms on genuine human voices, training follows a strict 6-stage protocol:

### 1. Anti-Leakage Data Partitioning
- Datasets are partitioned into **Train (70%)**, **Development (15%)**, and **Test (15%)** splits.
- Audio files from the same recording session or speaker never cross split boundaries, preventing identity memorization.

### 2. Controlled 1:1 Class Balancing
- Standard anti-spoofing datasets are up to 85% synthetic, which causes standard classifiers to develop a strong majority bias—falsely labeling real human voices as deepfakes (the root cause of the previous 88% false risk score).
- We enforce an exact **1:1 training distribution**:
  - **3,710 Genuine Human Voices** (LibriSpeech + ASVspoof bonafide)
  - **3,710 Synthetic Spoofs** (split evenly between ASVspoof A07–A19 and ChatterboxTTS)

### 3. 63-Dimensional Feature Extraction
- Audio clips are normalized and transformed into fixed-length 63-dimensional vectors:
  - **MFCCs (20 coefficients)**: Mean and standard deviation across frames (spectral envelope).
  - **Spectral Descriptors**: Centroid, bandwidth, rolloff, flatness, and sub-band contrast.
  - **Phase & Acoustic Artifacts**: Instantaneous phase derivative variance and zero-crossing rate.
  - **Prosodic Dynamics**: Fundamental frequency (F0), pitch range, jitter, and shimmer.

### 4. Dual Model Training
- **Random Forest**: Trained with 100 estimators, balanced leaf weights, and multi-core CPU parallelism.
- **SincNet Neural Model**: Trained on raw audio waveforms using AdamW optimizer with Cosine Annealing learning rate scheduling across 5 epochs on Apple Silicon Metal Performance Shaders (`mps`).

### 5. Dev-Set Threshold Calibration ($\tau^*$)
- **Strict Anti-Overfitting Rule**: The decision threshold is tuned strictly on the Development set by locating the Equal Error Rate (EER) point where False Positive Rate (FPR) equals False Negative Rate (FNR).
- The resulting calibrated threshold $\tau^* = 0.3985$ is locked into the model metadata.

### 6. Unbiased Held-Out Test Evaluation
- Evaluated on the held-out Test set (3,405 audio clips) with zero test-set threshold adjustments:
  - **ROC-AUC**: **0.9827** (Dual Ensemble: **0.9882**)
  - **Equal Error Rate (EER)**: **7.05%** (Dual Ensemble: **6.12%**)
  - **Human False Alarm Rate (Real $\to$ Fake)**: **5.53%** (down from 15%)
  - **ChatterboxTTS Detection Rate**: **100.0%** (300 / 300 detected)

---

## Installing Full Datasets for Retraining

To download and extract the full 38,000+ files locally:

### 1. PhonemeDF ChatterboxTTS
Place `PhonemeDF_ChatterboxTTS.tar.gz` in `Downloads/` or `datasets/`, then run:
```bash
python scripts/install_phonemedf.py --max-samples 0
```

### 2. ASVspoof 2019
```bash
python scripts/download_asvspoof2019.py
```

### 3. LibriSpeech test-clean
```bash
curl -O https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf test-clean.tar.gz -C datasets/librispeech/
```
