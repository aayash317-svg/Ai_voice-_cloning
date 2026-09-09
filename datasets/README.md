# Dataset Management & Installation

The **Voice Integrity Verification Framework** is trained and benchmarked across **38,239 audio clips** combining genuine human speech and multi-generator synthetic voices:

| Dataset | Type | Sample Count | Format | Primary Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **LibriSpeech (test-clean)** | Genuine Human | 2,620 clips | 16 kHz FLAC | Natural clean human speech baseline |
| **ASVspoof 2019 (LA)** | Genuine Human | 2,680 clips | 16 kHz WAV | Telephone & voice verification bonafide |
| **ASVspoof 2019 (LA A07–A19)** | Synthetic Spoofs | 15,399 clips | 16 kHz WAV | Traditional TTS & voice conversion attacks |
| **PhonemeDF (ChatterboxTTS)** | Modern Neural TTS | 17,540 clips | 16 kHz WAV | Modern phoneme-level neural text-to-speech |

---

## Bundled Sample Testing Data (`test_samples/`)

The repository includes pre-packaged testing samples directly in `test_samples/` for immediate out-of-the-box evaluation without needing to download multi-gigabyte archives:

- `test_samples/genuine_sample.wav` — Clean human speech reference
- `test_samples/test_genuine.m4a` — Natural human speech recording
- `test_samples/spoof_sample.wav` — ASVspoof synthetic voice sample
- `test_samples/chatterbox_tts_sample.wav` — PhonemeDF ChatterboxTTS neural synthetic voice

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
