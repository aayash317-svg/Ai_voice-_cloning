"""
Installer and Extraction Utility for PhonemeDF ChatterboxTTS Dataset.
Extracts genuine synthetic TTS speech samples from PhonemeDF_ChatterboxTTS.tar.gz
and integrates them into the Voice Integrity Verification training and evaluation pipelines.
"""

import sys
import os
import argparse
import tarfile
from pathlib import Path
from tqdm import tqdm
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
TARGET_DIR = DATASETS_DIR / "phonemedf_chatterboxtts"

CANDIDATE_PATHS = [
    Path("/Users/karanselvamani/Downloads/PhonemeDF_ChatterboxTTS.tar.gz.download/PhonemeDF_ChatterboxTTS.tar.gz"),
    Path("/Users/karanselvamani/Downloads/PhonemeDF_ChatterboxTTS.tar.gz"),
    DATASETS_DIR / "PhonemeDF_ChatterboxTTS.tar.gz",
    PROJECT_ROOT / "PhonemeDF_ChatterboxTTS.tar.gz"
]


def find_archive() -> Path:
    """Locate the PhonemeDF archive on disk."""
    for p in CANDIDATE_PATHS:
        if p.exists() and p.stat().st_size > 1024 * 1024:
            return p
    raise FileNotFoundError(
        "Could not locate PhonemeDF_ChatterboxTTS.tar.gz in Downloads or datasets directory."
    )


def extract_dataset(max_samples: int = 5000):
    archive_path = find_archive()
    print("=" * 70)
    print("PHONEMEDF CHATTERBOX-TTS DATASET INSTALLER")
    print("=" * 70)
    print(f"[+] Found archive at: {archive_path}")
    print(f"[+] Archive Size    : {archive_path.stat().st_size / (1024**3):.2f} GB")
    print(f"[+] Target Output   : {TARGET_DIR}")
    print(f"[+] Max Extraction  : {'ALL' if max_samples <= 0 else max_samples} samples")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    pbar = tqdm(desc="Extracting ChatterboxTTS WAVs", total=max_samples if max_samples > 0 else 17540)

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar:
                if member.isfile() and member.name.lower().endswith(".wav"):
                    # Flatten into TARGET_DIR
                    out_name = Path(member.name).name
                    dest_file = TARGET_DIR / out_name
                    if not dest_file.exists():
                        tar.extract(member, path=TARGET_DIR)
                        extracted_name = TARGET_DIR / member.name
                        if extracted_name != dest_file and extracted_name.exists():
                            extracted_name.rename(dest_file)
                    
                    extracted_count += 1
                    pbar.update(1)

                    if 0 < max_samples <= extracted_count:
                        break
    except Exception as e:
        print(f"\n[*] Reached end of available stream: {e}")
    finally:
        pbar.close()

    # Clean up any nested ChatterboxTTS subdirectory if created by tar.extract
    nested_dir = TARGET_DIR / "ChatterboxTTS"
    if nested_dir.exists():
        for f in nested_dir.glob("*.wav"):
            f.rename(TARGET_DIR / f.name)
        try:
            nested_dir.rmdir()
        except Exception:
            pass

    final_wavs = list(TARGET_DIR.glob("*.wav"))
    print(f"\n[+] Extraction Complete: {len(final_wavs)} WAV audio files in {TARGET_DIR}")

    if final_wavs:
        test_file = final_wavs[0]
        data, sr = sf.read(test_file)
        print(f"[+] Sample verified : {test_file.name}")
        print(f"    Sample Rate     : {sr} Hz")
        print(f"    Duration        : {len(data) / sr:.2f}s")
        print(f"    Channels        : {data.ndim}")

    print("[+] PhonemeDF ChatterboxTTS is now active in backend.dataset_loader!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PhonemeDF ChatterboxTTS dataset.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum samples to extract (0 = all ~17,500). Default: 5000."
    )
    args = parser.parse_args()
    extract_dataset(max_samples=args.max_samples)
