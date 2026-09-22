"""
FastAPI & WebSocket Streaming Server for Voice Integrity Verification.
Provides real-time streaming audio analysis, batch verification,
and tamper-evident audit endpoints.
"""

import json
import io
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from backend.config import SAMPLE_RATE, WINDOW_SAMPLES, MODELS_DIR, BASE_DIR, INDIC_TARGET_LANGUAGES
from backend.preprocess import AudioPreprocessor
from backend.features import FeatureExtractor
from backend.classifier import AntiSpoofClassifier
from backend.neural_classifier import SincNetClassifier, NEURAL_MODEL_PATH
from backend.risk_engine import RiskEngine
from backend.privacy import AudioPrivacyBuffer
from backend.audit_chain import AuditChain
from backend.call_analyzer import MultiSpeakerCallAnalyzer
from backend.stream_diarizer import StreamingCallMonitor
from backend.live_call_engine import LiveCallSession

FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="Voice Integrity Verification Framework",
    description="Real-Time AI Impersonation & Cloned Voice Detection Engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    if (FRONTEND_DIR / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

TEST_SAMPLES_DIR = BASE_DIR / "test_samples"
if TEST_SAMPLES_DIR.exists():
    app.mount("/test_samples", StaticFiles(directory=str(TEST_SAMPLES_DIR)), name="test_samples")

@app.get("/")
def serve_dashboard():
    """Serve the interactive web application dashboard."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Voice Integrity Verification API is online. Visit /docs for Swagger UI."}

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Serve favicon or 204 to avoid browser console 404 errors."""
    fav = FRONTEND_DIR / "assets" / "logo.png"
    if fav.exists():
        return FileResponse(str(fav), media_type="image/png")
    return Response(status_code=204)

# Global Framework State
preprocessor = AudioPreprocessor()
feature_extractor = FeatureExtractor()
risk_engine = RiskEngine()
audit_chain = AuditChain()
classifier: Optional[AntiSpoofClassifier] = None
neural_classifier: Optional[SincNetClassifier] = None


_last_loaded_mtime: float = 0.0


def get_or_load_classifier(force_reload: bool = False) -> Optional[AntiSpoofClassifier]:
    """Load latest available Random Forest model checkpoint (.pkl or .joblib), auto-reloading if updated on disk."""
    global classifier, _last_loaded_mtime
    candidates = list(MODELS_DIR.glob("**/*.pkl")) + list(MODELS_DIR.glob("**/*.joblib"))
    if not candidates:
        return None

    # Pick the most recently updated checkpoint file
    latest_file = max(candidates, key=lambda p: p.stat().st_mtime)
    current_mtime = latest_file.stat().st_mtime

    if classifier is not None and not force_reload and current_mtime <= _last_loaded_mtime:
        return classifier

    try:
        classifier = AntiSpoofClassifier.load(latest_file)
        _last_loaded_mtime = current_mtime
        print(f"[+] Loaded latest model checkpoint from: {latest_file} (calibrated_threshold={classifier.calibrated_threshold:.4f})")
        return classifier
    except Exception as e:
        print(f"[!] Could not load {latest_file}: {e}")
        return classifier


def get_or_load_neural_classifier() -> Optional[SincNetClassifier]:
    """Load latest SincNet neural raw-waveform model checkpoint (.pt)."""
    global neural_classifier
    if neural_classifier is not None:
        return neural_classifier

    if NEURAL_MODEL_PATH.exists():
        try:
            model = SincNetClassifier.load(NEURAL_MODEL_PATH)
            if model is not None:
                neural_classifier = model
                print(f"[+] Loaded neural raw-waveform model from: {NEURAL_MODEL_PATH}")
                return neural_classifier
        except Exception as e:
            print(f"[!] Could not load neural model {NEURAL_MODEL_PATH}: {e}")
    return None


def get_ffmpeg_binary() -> Optional[str]:
    """Locate FFmpeg binary across imageio-ffmpeg, system PATH, or standard OS paths."""
    import os
    import shutil

    # 1. Try imageio-ffmpeg (bundled standalone binary on Windows/Linux/Mac)
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass

    # 2. Try system PATH
    which_path = shutil.which("ffmpeg")
    if which_path:
        return which_path

    # 3. Known OS fallback locations
    for candidate in [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
        "C:\\ffmpeg\\bin\\ffmpeg.exe",
    ]:
        if os.path.exists(candidate):
            return candidate

    return None


async def _decode_audio(audio_bytes: bytes, filename: str):
    """
    Decode audio bytes to numpy float32 array + sample rate.
    Tries strategies in order:
      1. soundfile   — in-memory, fastest for WAV/FLAC/OGG/MP3
      2. ffmpeg      — universally handles M4A/AAC/MP4/WebM/WMA via imageio-ffmpeg or PATH
      3. afconvert   — macOS built-in fallback
    """
    import tempfile, os, subprocess

    # --- Strategy 1: soundfile (in-memory, fastest) ---
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
        if hasattr(data, "ndim") and data.ndim > 1:
            data = np.mean(data, axis=1)
        return data.astype(np.float32), sr
    except Exception:
        pass

    suffix = Path(filename).suffix.lower() or ".tmp"

    # Write raw bytes to a temp file for external decoders
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name

    tmp_wav_path = tmp_in_path + "_converted.wav"

    try:
        # --- Strategy 2: ffmpeg (universal for M4A, AAC, WebM, MP4, etc.) ---
        ffmpeg_bin = get_ffmpeg_binary()
        if ffmpeg_bin:
            try:
                result = subprocess.run(
                    [ffmpeg_bin, "-y", "-i", tmp_in_path, "-ar", "16000", "-ac", "1", "-f", "wav", tmp_wav_path],
                    capture_output=True, timeout=30
                )
                if result.returncode == 0 and os.path.exists(tmp_wav_path):
                    data, sr = sf.read(tmp_wav_path)
                    if hasattr(data, "ndim") and data.ndim > 1:
                        data = np.mean(data, axis=1)
                    return data.astype(np.float32), sr
                else:
                    print(f"[!] FFmpeg stderr: {result.stderr.decode('utf-8', errors='ignore')[:300]}")
            except Exception as e:
                print(f"[!] FFmpeg execution error: {e}")

        # --- Strategy 3: macOS afconvert (built-in fallback) ---
        try:
            result = subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEF32@16000", tmp_in_path, tmp_wav_path],
                capture_output=True, timeout=30
            )
            if result.returncode == 0 and os.path.exists(tmp_wav_path):
                data, sr = sf.read(tmp_wav_path)
                if hasattr(data, "ndim") and data.ndim > 1:
                    data = np.mean(data, axis=1)
                return data.astype(np.float32), sr
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{suffix}'. Could not decode audio stream. Please use WAV, FLAC, MP3, OGG, M4A, AAC, or WebM."
        )
    finally:
        for p in [tmp_in_path, tmp_wav_path]:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass





@app.get("/health")
def health_check():
    """System health check and active model status."""
    active_clf = get_or_load_classifier()
    active_neural = get_or_load_neural_classifier()
    return {
        "status": "online",
        "random_forest_loaded": active_clf is not None,
        "calibrated_threshold": round(active_clf.calibrated_threshold, 4) if active_clf else None,
        "neural_sincnet_loaded": active_neural is not None,
        "ensemble_active": active_clf is not None and active_neural is not None,
        "model_type": "Dual-Ensemble (RandomForest + SincNet Raw Audio)",
        "multilingual_active": True,
        "supported_languages": ["english"] + INDIC_TARGET_LANGUAGES,
        "features_dimension": len(active_clf.feature_names) if active_clf else 0,
        "audit_chain_blocks": len(audit_chain.chain)
    }


@app.post("/analyze")
async def analyze_audio_file(
    file: UploadFile = File(...),
    language: str = Query("auto"),
    is_unknown_number: bool = Query(False),
    is_high_value_transaction: bool = Query(False),
    prior_fraud_flag: bool = Query(False)
):
    """
    Analyze uploaded audio file, extract 4-family feature vectors + raw audio,
    compute dual ensemble impersonation risk score, and log security audit event.
    """
    audio_bytes = await file.read()
    try:
        data, sr = await _decode_audio(audio_bytes, file.filename or "audio.tmp")
        audio = preprocessor.load_audio(data, sr=sr)
        audio = preprocessor.normalize_audio(audio)
    except HTTPException:
        raise
    # Extract features with silence protection & VAD speech segmentation
    raw_rms = float(np.sqrt(np.mean(audio**2))) if len(audio) > 0 else 0.0
    raw_peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0

    if raw_rms < 0.005 and raw_peak < 0.025:
        # Near-silent recording / microphone muted - do not evaluate room hiss as fake
        return {
            "filename": file.filename,
            "assessment": {
                "risk_score": 2.0,
                "risk_level": "LOW",
                "alert_triggered": False,
                "signal_breakdown": {
                    "ml_classifier_spoof_prob": 0.02,
                    "acoustic_anomaly": 0.0,
                    "spectral_anomaly": 0.0,
                    "prosody_anomaly": 0.0,
                    "speaker_mismatch": 0.0
                },
                "context_modifier": 0.0,
                "safety_override_triggered": False,
                "explanation": "Audio level is near silent / ambient room noise. No synthetic voice detected. Please speak closer to the microphone."
            },
            "model_breakdown": {
                "random_forest_prob": 0.02,
                "neural_sincnet_prob": None,
                "ensemble_spoof_prob": 0.02
            },
            "features_extracted": {}
        }

    # Extract clean voiced speech segments (removes silent pauses & room hiss)
    voiced_audio = preprocessor.apply_vad(audio, top_db=28.0)
    if len(voiced_audio) >= int(0.5 * SAMPLE_RATE):
        eval_audio = preprocessor.normalize_audio(voiced_audio)
    else:
        eval_audio = preprocessor.normalize_audio(audio)

    features = feature_extractor.extract_all(eval_audio)
    active_clf = get_or_load_classifier()
    active_neural = get_or_load_neural_classifier()

    rf_spoof_prob: Optional[float] = None
    if active_clf:
        vec = feature_extractor.to_vector(features, active_clf.feature_names)
        global_rf = float(active_clf.predict_spoof_risk(vec))

        # For recordings longer than 3.5s, evaluate sliding windows to prevent dilution of synthetic markers
        win_samples = int(3.0 * SAMPLE_RATE)
        if len(eval_audio) >= int(3.5 * SAMPLE_RATE):
            # Dynamic hop: 1.5s for shorter clips, 2.5s for longer files to maintain fast response
            hop_samples = int(2.5 * SAMPLE_RATE) if len(eval_audio) > int(20.0 * SAMPLE_RATE) else int(1.5 * SAMPLE_RATE)
            window_probs = []
            for start in range(0, len(eval_audio) - win_samples + 1, hop_samples):
                chunk = eval_audio[start:start+win_samples]
                voiced_chunk = preprocessor.apply_vad(chunk, top_db=26.0)
                if len(voiced_chunk) < int(0.5 * SAMPLE_RATE):
                    continue
                w_feats = feature_extractor.extract_all(preprocessor.normalize_audio(voiced_chunk))
                w_vec = feature_extractor.to_vector(w_feats, active_clf.feature_names)
                window_probs.append(float(active_clf.predict_spoof_risk(w_vec)))
                if len(window_probs) >= 25:
                    break

            if window_probs:
                p75 = float(np.percentile(window_probs, 75))
                med = float(np.median(window_probs))
                # Robust blend: 60% global utterance, 25% 75th percentile window, 15% median window
                rf_spoof_prob = 0.60 * global_rf + 0.25 * p75 + 0.15 * med
            else:
                rf_spoof_prob = global_rf
        else:
            rf_spoof_prob = global_rf

    neural_spoof_prob: Optional[float] = None
    if active_neural:
        try:
            neural_spoof_prob = float(active_neural.predict_spoof_prob(audio))
        except Exception as e:
            print(f"[!] Neural prediction error: {e}")

    # Dual Ensemble Fusion
    if rf_spoof_prob is not None and neural_spoof_prob is not None:
        spoof_prob = 0.50 * rf_spoof_prob + 0.50 * neural_spoof_prob
    elif rf_spoof_prob is not None:
        spoof_prob = rf_spoof_prob
    elif neural_spoof_prob is not None:
        spoof_prob = neural_spoof_prob
    else:
        spoof_prob = float(np.clip(features.get("spectral_flatness", 0.0) * 1.5, 0.05, 0.95))

    context_meta = {
        "is_unknown_number": is_unknown_number,
        "is_high_value_transaction": is_high_value_transaction,
        "prior_fraud_flag": prior_fraud_flag
    }

    calibrated_th = float(active_clf.calibrated_threshold) if (active_clf and hasattr(active_clf, "calibrated_threshold") and active_clf.calibrated_threshold) else 0.5102

    assessment = risk_engine.compute_risk(
        classifier_spoof_prob=spoof_prob,
        calibrated_threshold=calibrated_th,
        acoustic_anomaly_score=features.get("acoustic_phase_derivative_var", 0.0),
        spectral_anomaly_score=features.get("spectral_hf_energy_ratio", 0.0),
        prosody_anomaly_score=features.get("prosody_jitter", 0.0),
        contrast_anomaly_score=features.get("spectral_contrast_std", 0.0),
        context_metadata=context_meta,
        language_code=language
    )

    # Append non-biometric security event to audit chain
    audit_chain.append_event("BATCH_ANALYSIS", {
        "filename": file.filename,
        "language": language,
        "risk_score": assessment.risk_score,
        "risk_level": assessment.risk_level,
        "alert_triggered": assessment.alert_triggered
    })

    return {
        "filename": file.filename,
        "language": language,
        "assessment": assessment,
        "model_breakdown": {
            "random_forest_prob": round(rf_spoof_prob, 4) if rf_spoof_prob is not None else None,
            "neural_sincnet_prob": round(neural_spoof_prob, 4) if neural_spoof_prob is not None else None,
            "ensemble_spoof_prob": round(spoof_prob, 4)
        },
        "features_extracted": {k: round(v, 4) for k, v in features.items()}
    }


@app.post("/analyze/call")
async def analyze_call_endpoint(
    file: UploadFile = File(...),
    claimed_contact_role: str = Form("father"),
    is_financial_transaction: bool = Form(False),
    language: str = Form("auto")
):
    """
    Multi-Speaker Conversational Call Audio Analysis.
    Performs Quality Gate -> VAD -> Speaker Diarization -> Overlap Exclusion ->
    Independent Per-Speaker Anti-Spoof Evaluation -> Policy Risk Decision.
    """
    audio_bytes = await file.read()
    try:
        data, sr = await _decode_audio(audio_bytes, file.filename or "call_audio.tmp")
        audio = preprocessor.load_audio(data, sr=sr)
        audio = preprocessor.normalize_audio(audio)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio decoding error: {e}")

    active_clf = get_or_load_classifier()
    if not active_clf:
        raise HTTPException(status_code=503, detail="No anti-spoof model checkpoint is currently loaded.")

    call_analyzer = MultiSpeakerCallAnalyzer(
        classifier=active_clf,
        feature_extractor=feature_extractor,
        risk_engine=risk_engine,
        sample_rate=SAMPLE_RATE
    )

    result = call_analyzer.analyze_call(
        audio=audio,
        claimed_contact_role=claimed_contact_role,
        is_financial_transaction=is_financial_transaction,
        language_preset=language
    )
    result["filename"] = file.filename
    result["language_preset"] = language

    # Log multi-speaker forensic event to tamper-evident audit chain
    audit_chain.append_event("CALL_ANALYSIS", {
        "filename": file.filename,
        "call_assessment": result["call_assessment"],
        "overall_call_risk": result.get("overall_call_risk", 0.0),
        "speakers_count": len(result["speakers"]),
        "overlap_percent": result["audio_quality"]["overlap_percent"]
    })

    return result


@app.websocket("/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    Real-time streaming WebSocket endpoint.
    Receives continuous PCM 16-bit / float audio chunks, analyzes sliding windows,
    returns continuous dynamic risk scores, and discards audio buffers immediately.
    """
    await websocket.accept()
    buffer = AudioPrivacyBuffer()
    active_clf = get_or_load_classifier()
    active_neural = get_or_load_neural_classifier()
    language = websocket.query_params.get("language", "auto")

    try:
        while True:
            # Receive raw binary PCM or JSON audio data
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            buffer.append_chunk(chunk)

            # Analyze sliding window (3.0 seconds = 48,000 samples at 16kHz)
            window = buffer.get_window(WINDOW_SAMPLES)
            if window is not None:
                win_rms = float(np.sqrt(np.mean(window**2)))
                win_peak = float(np.max(np.abs(window)))

                # If user is silent (listening / between sentences), do NOT boost room noise
                if win_rms < 0.006 and win_peak < 0.03:
                    await websocket.send_json({
                        "risk_score": 2.5,
                        "risk_level": "LOW",
                        "alert": False,
                        "signal_breakdown": {
                            "ml_classifier_spoof_prob": 0.02,
                            "acoustic_anomaly": 0.0,
                            "spectral_anomaly": 0.0,
                            "prosody_anomaly": 0.0,
                            "speaker_mismatch": 0.0
                        },
                        "safety_override": False
                    })
                    continue

                norm_window = preprocessor.normalize_audio(window)
                features = feature_extractor.extract_all(norm_window)

                rf_prob = None
                if active_clf:
                    vec = feature_extractor.to_vector(features, active_clf.feature_names)
                    rf_prob = float(active_clf.predict_spoof_risk(vec))

                neural_prob = None
                if active_neural:
                    try:
                        neural_prob = float(active_neural.predict_spoof_prob(norm_window))
                    except Exception:
                        pass

                if rf_prob is not None and neural_prob is not None:
                    spoof_prob = 0.50 * rf_prob + 0.50 * neural_prob
                elif rf_prob is not None:
                    spoof_prob = rf_prob
                elif neural_prob is not None:
                    spoof_prob = neural_prob
                else:
                    spoof_prob = float(np.clip(features.get("spectral_flatness", 0.0) * 1.5, 0.05, 0.95))

                calibrated_th = float(active_clf.calibrated_threshold) if (active_clf and hasattr(active_clf, "calibrated_threshold") and active_clf.calibrated_threshold) else 0.5102

                assessment = risk_engine.compute_risk(
                    classifier_spoof_prob=spoof_prob,
                    calibrated_threshold=calibrated_th,
                    acoustic_anomaly_score=features.get("acoustic_phase_derivative_var", 0.0),
                    spectral_anomaly_score=features.get("spectral_hf_energy_ratio", 0.0),
                    prosody_anomaly_score=features.get("prosody_jitter", 0.0),
                    contrast_anomaly_score=features.get("spectral_contrast_std", 0.0),
                    language_code=language
                )

                if assessment.alert_triggered:
                    audit_chain.append_event("STREAM_ALERT", {
                        "risk_score": assessment.risk_score,
                        "risk_level": assessment.risk_level
                    })

                await websocket.send_json({
                    "risk_score": assessment.risk_score,
                    "risk_level": assessment.risk_level,
                    "alert": assessment.alert_triggered,
                    "signal_breakdown": assessment.signal_breakdown,
                    "safety_override": assessment.safety_override_triggered,
                    "language_preset": language
                })

    except WebSocketDisconnect:
        pass
    finally:
        buffer.purge()


@app.websocket("/stream/call")
async def websocket_stream_call_endpoint(websocket: WebSocket):
    """
    Real-time live multi-speaker conversational call monitoring endpoint.
    Processes continuous 16kHz PCM audio stream from live microphone or call audio.
    Identifies who is speaking (Local Caller vs Remote Contact), isolates overlap,
    evaluates anti-spoof characteristics in real time, and sends back instantaneous threat telemetry.
    """
    await websocket.accept()
    active_clf = get_or_load_classifier()
    active_neural = get_or_load_neural_classifier()

    claimed_contact_role = websocket.query_params.get("claimed_contact_role", "remote_contact")
    is_financial_transaction = websocket.query_params.get("is_financial_transaction", "false").lower() in ("true", "1")
    language = websocket.query_params.get("language", "auto")

    monitor = StreamingCallMonitor(
        classifier=active_clf,
        neural_classifier=active_neural,
        feature_extractor=feature_extractor,
        risk_engine=risk_engine,
        sample_rate=SAMPLE_RATE,
        claimed_contact_role=claimed_contact_role,
        is_financial_transaction=is_financial_transaction,
        language_preset=language
    )

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if "bytes" in message and message["bytes"]:
                try:
                    pcm_bytes = message["bytes"]
                    if len(pcm_bytes) % 2 != 0:
                        pcm_bytes = pcm_bytes[:len(pcm_bytes) - 1]
                    if len(pcm_bytes) == 0:
                        continue
                    pcm_data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    telemetry = monitor.process_live_chunk(pcm_data)
                    if telemetry.get("alert") and telemetry.get("speaker_b", {}).get("risk_score", 0.0) >= 70.0:
                        audit_chain.append_event("LIVE_CALL_CLONE_DETECTED", {
                            "role": monitor.claimed_contact_role,
                            "risk_score": telemetry.get("speaker_b", {}).get("risk_score"),
                            "risk_level": telemetry.get("risk_level"),
                            "timestamp": telemetry.get("timestamp")
                        })
                    await websocket.send_json(telemetry)
                except Exception as chunk_err:
                    print(f"[!] Live call chunk processing error: {chunk_err}", flush=True)
                    continue
            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    if "audio_chunk" in payload:
                        raw_bytes = base64.b64decode(payload["audio_chunk"])
                        if len(raw_bytes) % 2 != 0:
                            raw_bytes = raw_bytes[:len(raw_bytes) - 1]
                        if len(raw_bytes) == 0:
                            continue
                        pcm_data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        telemetry = monitor.process_live_chunk(pcm_data)
                        await websocket.send_json(telemetry)
                    elif payload.get("action") == "configure":
                        if "claimed_contact_role" in payload:
                            monitor.claimed_contact_role = payload["claimed_contact_role"]
                        if "is_financial_transaction" in payload:
                            monitor.is_financial_transaction = bool(payload["is_financial_transaction"])
                        if "language" in payload:
                            monitor.language_preset = payload["language"]
                        await websocket.send_json({"status": "configured", "role": monitor.claimed_contact_role, "language": monitor.language_preset})
                except Exception as text_err:
                    print(f"[!] Live call text payload error: {text_err}", flush=True)
                    continue
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[!] WebSocket live call stream error: {e}")


@app.websocket("/ws/live-analyze")
async def websocket_live_analyze_endpoint(websocket: WebSocket):
    """
    Real-Time Live Call Voice Cloning Detection WebSocket Endpoint.
    Conforms to user specification:
    - Receives continuous audio chunks during live calls (binary PCM or JSON base64).
    - Runs rolling in-memory buffer, VAD, Speaker Diarization (Caller vs Father), Overlap Detection,
      anti-spoof feature extraction per speaker, and updates risk score live.
    - Zero retention: Discards raw chunk, stores only hash & security event to audit chain.
    """
    await websocket.accept()
    active_clf = get_or_load_classifier()
    active_neural = get_or_load_neural_classifier()

    call_id = websocket.query_params.get("call_id", "demo-call-001")

    session = LiveCallSession(
        call_id=call_id,
        classifier=active_clf,
        neural_classifier=active_neural,
        feature_extractor=feature_extractor,
        risk_engine=risk_engine,
        sample_rate=SAMPLE_RATE,
    )

    try:
        while True:
            message = await websocket.receive()
            chunk_pcm = None

            if "bytes" in message and message["bytes"]:
                chunk_pcm = np.frombuffer(message["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    if "audio_chunk" in payload:
                        raw_bytes = base64.b64decode(payload["audio_chunk"])
                        chunk_pcm = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    elif payload.get("action") == "configure":
                        await websocket.send_json({"status": "configured", "call_id": session.call_id})
                        continue
                except Exception as ex:
                    print(f"[!] Invalid live analyze message: {ex}")
                    continue

            if chunk_pcm is not None and len(chunk_pcm) > 0:
                response = session.process_chunk(chunk_pcm)
                if response.get("father_ai_prob", 0.0) >= 0.70:
                    audit_chain.append_event("LIVE_CLONE_ALERT", {
                        "call_id": session.call_id,
                        "timestamp": response.get("timestamp"),
                        "father_ai_prob": response.get("father_ai_prob"),
                        "risk_level": response.get("risk_level")
                    })
                await websocket.send_json(response)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[!] WebSocket live analyze error: {e}")


@app.get("/audit/chain")
def get_audit_chain():
    """Retrieve immutable audit chain entries."""
    return {"chain": [vars(b) for b in audit_chain.chain], "length": len(audit_chain.chain)}


@app.get("/audit/verify")
def verify_audit_integrity():
    """Cryptographically verify the integrity of the audit ledger."""
    is_valid, error = audit_chain.verify_integrity()
    return {
        "valid": is_valid,
        "status": "SECURE_AND_INTACT" if is_valid else "TAMPERING_DETECTED",
        "error_detail": error,
        "blocks_verified": len(audit_chain.chain)
    }
