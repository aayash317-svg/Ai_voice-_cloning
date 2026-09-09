# VOICE CLEANING — MASTER DEVELOPMENT & TRAINING INSTRUCTION

## 1. Project Goal
Build a privacy-preserving AI system that detects whether incoming speech is:
1. Genuine human speech
2. AI-generated / synthetic / spoofed speech
The system must produce a confidence/risk score and eventually support near-real-time voice analysis.
The first implementation must be lightweight and suitable for a Mac with limited compute.
Do NOT start with a large deep-learning model.

## 2. Current Development Phase
We are currently building the first working prototype.
* **Dataset**: ASVspoof 2019 LA — reduced/derived dataset currently being used for prototype development.
* **Model**: Random Forest classifier.
* **Feature-based approach**:
  Audio → preprocessing → feature extraction → feature vector → Random Forest → genuine/spoof prediction → probability/confidence → risk score

Do not claim that this is the final production model.
Call it: **"Lightweight baseline prototype."**

## 3. Recommended Project Structure
```text
voice-cloning/
│
├── datasets/
│   └── asvspoof2019/
│       ├── train/
│       ├── dev/
│       └── test/
│
├── backend/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset_loader.py
│   ├── audio_preprocessing.py
│   ├── features.py
│   ├── classifier.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── risk_engine.py
│   ├── privacy.py
│   └── app.py
│
├── models/
│   └── baseline_random_forest.pkl
│
├── features/
│
├── results/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── training_report.txt
│
├── notebooks/
│
├── requirements.txt
└── README.md
```

## 4. Dataset Loader
Implement `dataset_loader.py`.
### Responsibilities:
1. Find audio files.
2. Read the corresponding labels.
3. Map labels into:
   * Genuine → 0
   * Spoof → 1
4. Maintain train/dev/test separation.
5. Never randomly mix train and test files.
6. Log:
   * number of genuine files
   * number of spoof files
   * total files
   * class balance

**Important**:
Do not create data leakage.
The same audio file must never appear in both training and testing.
If speaker IDs or metadata are available, avoid putting the same speaker into both training and evaluation where possible.

## 5. Audio Preprocessing
Create `audio_preprocessing.py`.
For each audio file:
1. Load WAV audio using `soundfile`/`librosa`.
2. Convert to mono.
3. Normalize amplitude where appropriate.
4. Resample to a consistent sample rate if required.
5. Remove unusable/empty audio.
6. Handle very short files safely.
7. Return clean waveform.

Do NOT save unnecessary copies of raw audio.

## 6. Feature Extraction
Create `features.py`.
Extract lightweight acoustic features.

### A. MFCC
Extract approximately:
* 13–20 MFCC coefficients.
Calculate statistics such as:
* mean
* standard deviation
This captures the spectral characteristics of speech.

### B. Spectral Features
Extract:
* Spectral centroid
* Spectral bandwidth
* Spectral rolloff
* Spectral contrast
* Zero-crossing rate
Calculate suitable statistical summaries.

### C. Prosodic Features
Where reliable, extract:
* Fundamental frequency / pitch statistics
* Energy
* RMS energy
* Speech dynamics
Use mean/std/min/max where appropriate.

### D. Temporal Features
Extract useful timing characteristics where possible.
Examples:
* zero-crossing statistics
* energy variation
* frame-level variation

The final output for each audio file must be a fixed-length numerical feature vector.
```text
audio.wav ↓ MFCC + spectral features + prosody + energy ↓ [feature1, feature2, feature3, ... featureN]
```

## 7. Feature Validation
Before training check:
* NaN values
* infinite values
* missing values
* feature dimensions
* class distribution

Use appropriate imputation or replacement only when necessary.
Log the final feature matrix shape. Example:
* `X_train = (N, FEATURES)`
* `y_train = (N,)`

## 8. Baseline Machine Learning Model
Create `classifier.py`.
Use: `RandomForestClassifier`
Recommended starting configuration:
* `n_estimators = 200`
* `random_state = 42`
* `class_weight = "balanced"`

Tune parameters later.
The classifier must output:
1. Genuine probability
2. Spoof probability
3. Predicted class

Example output:
```json
{
  "prediction": "spoof",
  "spoof_probability": 0.94,
  "genuine_probability": 0.06
}
```
Do not hard-code the prediction.

## 9. Training Pipeline
Create `train.py`.
Pipeline:
```text
Dataset ↓ Load audio ↓ Preprocess audio ↓ Extract features ↓ Validate features ↓ Create X_train/y_train ↓ Train Random Forest ↓ Validate on development set ↓ Save trained model
```
Save the trained model to: `models/baseline_random_forest.pkl`
Also save:
* feature configuration
* sample rate
* feature names
* model parameters

This is important so inference uses exactly the same feature pipeline.

## 10. Evaluation
Create `evaluate.py`.
Evaluate the model on a held-out test set.
Calculate:
* **Accuracy**: Useful as a basic metric, but do NOT use it as the only metric.
* **Precision**: Measure how many predicted spoof samples were actually spoof.
* **Recall**: Measure how many actual spoof samples were detected.
* **F1 Score**: Balance precision and recall.
* **Confusion Matrix**: True Genuine, False Genuine, True Spoof, False Spoof.
* **ROC Curve & AUC**: Generate ROC curve and calculate ROC-AUC.
* **EER**: Implement Equal Error Rate. Especially important for speaker/anti-spoofing evaluation.
* **FAR / FRR**: Report False Acceptance Rate & False Rejection Rate.

Save all results under `results/`.

## 11. IMPORTANT — No Fake Accuracy
Never manually increase, modify, or fabricate the model's accuracy.
If the model achieves 70%, report 70%.
If it achieves 82%, report 82%.
If it achieves 95%, report 95%.
Do not optimize only to obtain a high percentage.
The project must prioritize reliable evaluation and no data leakage.
If it is human, show it as human.

## 12. Prediction Pipeline
Create `predict.py`.
Input: A WAV audio file.
Pipeline: `Audio ↓ Preprocessing ↓ Feature extraction ↓ Saved Random Forest model ↓ Prediction`
Output:
```json
{
  "prediction": "genuine/spoof",
  "spoof_probability": 0.xx,
  "genuine_probability": 0.xx
}
```

## 13. Risk Engine
Create `risk_engine.py`.
Convert model probability into a risk score from 0–100.
Example threshold mapping:
* `0–30`: LOW RISK
* `31–60`: MEDIUM RISK
* `61–80`: HIGH RISK
* `81–100`: CRITICAL RISK

*Note*: This is a prototype threshold system. Make risk thresholds configurable.

## 14. Contextual Metadata — Later Phase
Eventually combine:
```text
Voice Spoof Probability + Speaker Consistency + Caller Reputation + Known/Unknown Contact + Transaction Context → Risk Engine → 0–100 Risk Score → Alert
```
Spam reputation is contextual evidence, not proof of synthetic speech.

## 15. Future Speaker Verification Layer
Do not implement the heavy speaker model yet.
Later add VoxCeleb speaker embedding model comparison ("Is this the same person?"). Keep anti-spoofing ("Is this audio synthetic?") separate.

## 16. System Architecture
```text
             ┌──────────────────┐
             │   Voice Input    │
             └────────┬─────────┘
                      ↓
             ┌──────────────────┐
             │ Audio Processing │
             └────────┬─────────┘
                      ↓
          ┌────────────────────────┐
          │   Feature Extraction   │
          │                        │
          │ MFCC                   │
          │ Spectral Features      │
          │ Prosody                │
          │ Energy                 │
          └───────────┬────────────┘
                      ↓
             ┌──────────────────┐
             │ Random Forest    │
             │ Baseline Model   │
             └────────┬─────────┘
                      ↓
             ┌──────────────────┐
             │ Spoof Probability│
             └────────┬─────────┘
                      ↓
             ┌──────────────────┐
             │   Risk Engine    │
             └────────┬─────────┘
                      ↓
         ┌─────────────────────────┐
         │ Risk Score 0–100        │
         └────────────┬────────────┘
                      ↓
             ┌──────────────────┐
             │ Alert / Decision │
             └──────────────────┘
```

## 17. Future Production Architecture
(Audio buffer 3–4 sec overlapping → Anti-Spoof + Speaker models → Risk Engine → Alert/Allow & Hash-chain logging).
Raw audio processed in memory and discarded.

## 18. Privacy Architecture
1. Do not permanently store raw call recordings.
2. Process audio in memory.
3. Discard temporary audio after analysis.
4. Encrypt speaker embeddings (Fernet for prototype).
5. Never log raw audio or biometric data into audit chains.

## 19. Real-Time Pipeline — Later
Target low latency for 3–4 second overlapping windows.

## 20. Model Improvement Roadmap
Phase 1: Random Forest + lightweight features
Phase 2: XGBoost comparison
Phase 3: Larger ASVspoof datasets
Phase 4: IndicSynth + SEA-Spoof
Phase 5: ASVspoof 2021 / ASVspoof 5
Phase 6: In-the-Wild evaluation
Phase 7: Deep-learning anti-spoof model (AASIST, RawNet2, WavLM)

## 21. Required Visualizations
Save in `results/`:
1. Dataset class distribution
2. Training/validation metrics
3. Confusion matrix
4. ROC curve
5. Risk-score distribution
6. Genuine vs spoof probability distribution

## 22. Required Final Demo
Simple CLI demonstration:
* Test 1 (Genuine human speech) → Prediction: Genuine, Risk: LOW
* Test 2 (AI-generated speech) → Prediction: Spoof, Risk: HIGH

## 23. FastAPI — Later
POST `/analyze` and WS `/stream`.

## 24. Development Rules
1. Python 3.11.
2. Use `.venv`.
3. Keep dependencies minimal and baseline lightweight.
4. Prevent data leakage.
5. No fake accuracy.
6. Modular, logged, clean code.

## 25. FIRST TASK — DO ONLY THIS
Do NOT build the entire production system yet.
First complete:
```text
Dataset → Dataset Loader → Audio Preprocessing → Feature Extraction → Random Forest → Evaluation → Confusion Matrix → ROC Curve → EER/FAR/FRR → Save Model
```
Then stop and provide a report containing:
1. Dataset size
2. Genuine samples count
3. Spoof samples count
4. Feature count
5. Training time
6. Test time
7. Accuracy
8. Precision
9. Recall
10. F1
11. ROC-AUC
12. EER
13. FAR
14. FRR
15. Confusion matrix
16. Problems encountered
17. Recommended next step
