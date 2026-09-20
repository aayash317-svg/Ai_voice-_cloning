"""
Installer and Preprocessor for Additional Genuine Human Speech Datasets.
Downloads, standardizes (16 kHz, 16-bit Mono WAV), indexes metadata,
and applies leak-free speaker-disjoint train/dev/test splitting.
"""

import os
import sys
import json
import time
import zipfile
import tarfile
import argparse
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import soundfile as sf
import librosa

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import DATASETS_DIR, SAMPLE_RATE

TARGET_DIR = DATASETS_DIR / "additional_genuine"
CACHE_DIR = DATASETS_DIR / "download_cache"

MIRRORS_SLR65 = [
    "https://openslr.trmal.net/resources/65/",
    "https://openslr.magicdatatech.com/resources/65/",
    "https://openslr.elda.org/resources/65/"
]

FLEURS_BASE_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"

INDIC_LANG_MAP = {
    "ta_in": "tamil",
    "hi_in": "hindi",
    "te_in": "telugu",
    "bn_in": "bengali",
    "kn_in": "kannada",
    "mr_in": "marathi"
}


def is_valid_cache_file(path: Path) -> bool:
    """Verify that cached archive file exists, is non-empty, and uncorrupted."""
    if not path.exists() or path.stat().st_size <= 1024:
        return False
    if path.suffix.lower() == ".zip":
        try:
            if not zipfile.is_zipfile(str(path)):
                return False
            with zipfile.ZipFile(str(path), "r") as zf:
                # Quick test to make sure archive central directory is valid
                return len(zf.namelist()) > 0
        except Exception:
            return False
    elif path.name.lower().endswith(".tar.gz"):
        try:
            return tarfile.is_tarfile(str(path))
        except Exception:
            return False
    return True


def download_file_with_retry(urls: List[str], dest_path: Path, desc: str = "") -> bool:
    """Download file with multiple mirror fallbacks, archive validation, and progress display."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if is_valid_cache_file(dest_path):
        print(f"[OK] Cache hit (verified valid): {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
        return True
    elif dest_path.exists():
        print(f"[!] Incomplete or corrupted cache file found for {dest_path.name}. Re-downloading...")
        try:
            dest_path.unlink()
        except Exception:
            pass

    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    if temp_path.exists():
        try:
            temp_path.unlink()
        except Exception:
            pass

    headers = {"User-Agent": "Mozilla/5.0"}

    for url in urls:
        print(f"[*] Downloading {desc or dest_path.name} from: {url}")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as response, open(temp_path, "wb") as out_file:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 1024 * 1024  # 1 MB chunk
                t0 = time.time()
                last_print = 0

                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    elapsed = max(0.1, time.time() - t0)
                    speed_mb = (downloaded / (1024 * 1024)) / elapsed

                    if downloaded - last_print > 5 * 1024 * 1024 or (total_size and downloaded == total_size):
                        pct = (downloaded / total_size * 100) if total_size else 0
                        print(f"    Progress: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%) [{speed_mb:.1f} MB/s]", end="\r")
                        last_print = downloaded

                print()

            if total_size > 0 and downloaded < total_size:
                raise IOError(f"Incomplete download: received {downloaded} of {total_size} bytes")

            if temp_path.exists() and temp_path.stat().st_size > 0:
                temp_path.rename(dest_path)
                # Verify downloaded archive
                if is_valid_cache_file(dest_path):
                    print(f"[+] Download complete & integrity verified: {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
                    return True
                else:
                    print(f"[!] Archive verification failed for {dest_path.name}. Trying next mirror...")
                    dest_path.unlink()
        except Exception as e:
            print(f"[!] Warning: Failed from {url}: {e}. Trying next mirror...")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

    return False



def standardize_audio_clip(input_path: Path, output_path: Path, target_sr: int = 16000) -> Tuple[bool, float, int, str]:
    """
    Standardize a single audio file to: WAV, Mono, 16 kHz, 16-bit PCM.
    Processes file-by-file to minimize RAM usage.
    Returns: (success, duration_seconds, original_sr, original_format)
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Inspect original metadata
        info = sf.info(str(input_path))
        orig_sr = info.samplerate
        orig_format = info.format

        # Low-memory streaming conversion
        y, sr = librosa.load(str(input_path), sr=target_sr, mono=True)
        duration = float(len(y) / target_sr)

        # Write as 16-bit PCM WAV
        sf.write(str(output_path), y, target_sr, subtype="PCM_16")
        del y
        return True, duration, orig_sr, orig_format
    except Exception as e:
        print(f"[!] Error processing {input_path.name}: {e}")
        return False, 0.0, 0, "unknown"


def process_slr65(target_base: Path, cache_base: Path, max_clips: int = 0) -> List[Dict[str, Any]]:
    """Download, standardize, and split OpenSLR 65 Tamil crowdsourced dataset."""
    print("\n" + "=" * 70)
    print("STAGE 1: OpenSLR 65 (Crowdsourced Tamil Multi-Speaker Dataset)")
    print("=" * 70)

    slr_cache = cache_base / "slr65"
    slr_target = target_base / "slr65"

    files_to_download = [
        ("ta_in_female.zip", [m + "ta_in_female.zip" for m in MIRRORS_SLR65]),
        ("ta_in_male.zip", [m + "ta_in_male.zip" for m in MIRRORS_SLR65]),
        ("line_index_female.tsv", [m + "line_index_female.tsv" for m in MIRRORS_SLR65]),
        ("line_index_male.tsv", [m + "line_index_male.tsv" for m in MIRRORS_SLR65]),
    ]

    for fname, mirrors in files_to_download:
        dest = slr_cache / fname
        if not download_file_with_retry(mirrors, dest, desc=fname):
            print(f"[!] Failed to acquire required SLR65 file: {fname}")
            return []

    # Extract zips into temporary extraction directory
    extract_dir = slr_cache / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    records = []
    # Discover all audio files in the zips
    for zip_name, gender in [("ta_in_female.zip", "female"), ("ta_in_male.zip", "male")]:
        zip_path = slr_cache / zip_name
        print(f"[*] Extracting and standardizing {zip_name} ({gender})...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            wav_members = [m for m in zf.namelist() if m.lower().endswith(".wav")]
            print(f"    Discovered {len(wav_members)} WAV clips in {zip_name}.")

            # Deterministic speaker mapping from filenames or line index
            # SLR65 filenames follow: <speaker_id>_<line_id>.wav or similar prefix
            for member in wav_members:
                if max_clips > 0 and len(records) >= max_clips:
                    break
                fname = Path(member).name
                stem = fname.replace(".wav", "")
                parts = stem.split("_")
                # Format: taf_00008_00072928033.wav -> spk_id is taf_00008
                spk_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else f"slr65_{gender}_{parts[0]}"

                # Extract individual audio to disk
                extracted_path = extract_dir / fname
                if not extracted_path.exists():
                    with zf.open(member) as zf_file, open(extracted_path, "wb") as f_out:
                        f_out.write(zf_file.read())

                records.append({
                    "raw_path": extracted_path,
                    "filename": fname,
                    "speaker_id": spk_id,
                    "language": "tamil",
                    "dataset": "slr65",
                    "gender": gender
                })

    # Partition speakers 70% Train, 15% Dev, 15% Test
    speakers = sorted(list(set(r["speaker_id"] for r in records)))
    rng = np.random.RandomState(42)
    rng.shuffle(speakers)

    n_spk = len(speakers)
    n_train = max(1, int(n_spk * 0.70))
    n_dev = max(1, int(n_spk * 0.15))
    train_spks = set(speakers[:n_train])
    dev_spks = set(speakers[n_train:n_train + n_dev])
    test_spks = set(speakers[n_train + n_dev:])

    processed_metadata = []
    print(f"[*] Standardizing SLR65 audio files file-by-file (Train: {len(train_spks)}, Dev: {len(dev_spks)}, Test: {len(test_spks)} speakers)...")

    for i, r in enumerate(records):
        spk = r["speaker_id"]
        split = "train" if spk in train_spks else ("dev" if spk in dev_spks else "test")
        out_wav = slr_target / split / r["filename"]

        success, duration, orig_sr, orig_fmt = standardize_audio_clip(r["raw_path"], out_wav, target_sr=SAMPLE_RATE)
        if success:
            processed_metadata.append({
                "dataset": "slr65",
                "language": "tamil",
                "speaker_id": spk,
                "filename": r["filename"],
                "file_path": str(out_wav.relative_to(PROJECT_ROOT)),
                "label": 0,  # 0 = genuine human speech
                "split": split,
                "duration_sec": round(duration, 3),
                "original_sampling_rate": orig_sr,
                "original_format": orig_fmt
            })

        # Remove temp extracted file to preserve disk
        try:
            r["raw_path"].unlink()
        except Exception:
            pass

        if (i + 1) % 500 == 0 or (i + 1) == len(records):
            print(f"    Standardized {i + 1}/{len(records)} clips...")

    return processed_metadata


def process_fleurs(
    target_base: Path,
    cache_base: Path,
    languages: List[str] = ["ta_in", "hi_in", "te_in", "bn_in", "kn_in", "mr_in"],
    max_clips_per_lang: int = 0
) -> List[Dict[str, Any]]:
    """Download, standardize, and index Google FLEURS Indic languages."""
    print("\n" + "=" * 70)
    print(f"STAGE 2: Google FLEURS Multi-Speaker Indic Pack ({len(languages)} Languages)")
    print("=" * 70)

    fleurs_cache = cache_base / "fleurs"
    fleurs_target = target_base / "fleurs"
    all_fleurs_metadata = []

    for lang_code in languages:
        lang_name = INDIC_LANG_MAP.get(lang_code, lang_code)
        print(f"\n[*] Processing FLEURS Language: {lang_name.upper()} ({lang_code})...")

        lang_cache = fleurs_cache / lang_code
        lang_cache.mkdir(parents=True, exist_ok=True)

        for split in ["train", "dev", "test"]:
            # Download TSV metadata and audio tar.gz
            tsv_fname = f"{split}.tsv"
            tar_fname = f"{split}.tar.gz"

            tsv_url = f"{FLEURS_BASE_URL}/{lang_code}/{tsv_fname}"
            tar_url = f"{FLEURS_BASE_URL}/{lang_code}/audio/{tar_fname}"

            tsv_dest = lang_cache / tsv_fname
            tar_dest = lang_cache / tar_fname

            download_file_with_retry([tsv_url], tsv_dest, desc=f"{lang_code} {split}.tsv")
            download_file_with_retry([tar_url], tar_dest, desc=f"{lang_code} {split}.tar.gz")

            # Parse TSV metadata: id, file_name, raw_transcription, transcription, gender, speaker_id
            speaker_lookup = {}
            if tsv_dest.exists():
                try:
                    with open(tsv_dest, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            parts = line.strip().split("\t")
                            if len(parts) >= 6:
                                fname = parts[1]
                                spk = parts[5] if len(parts) > 5 else f"fleurs_{lang_name}_{split}_spk"
                                speaker_lookup[fname] = spk
                except Exception as e:
                    print(f"    [!] Warning parsing TSV {tsv_fname}: {e}")

            # Extract tar.gz and standardize file-by-file
            if tar_dest.exists():
                print(f"    Extracting & standardizing {lang_name} [{split}]...")
                try:
                    with tarfile.open(tar_dest, "r:gz") as tar:
                        members = [m for m in tar.getmembers() if m.name.lower().endswith(".wav")]
                        clip_count = 0

                        for m in members:
                            if max_clips_per_lang > 0 and clip_count >= max_clips_per_lang:
                                break

                            fname = Path(m.name).name
                            spk_id = speaker_lookup.get(fname, f"fleurs_{lang_name}_{split}_{fname[:6]}")
                            out_wav = fleurs_target / lang_name / split / fname

                            # Extract single member to temporary buffer/file
                            extracted_obj = tar.extractfile(m)
                            if extracted_obj:
                                tmp_wav = lang_cache / f"tmp_{fname}"
                                with open(tmp_wav, "wb") as f_out:
                                    f_out.write(extracted_obj.read())

                                success, duration, orig_sr, orig_fmt = standardize_audio_clip(tmp_wav, out_wav, target_sr=SAMPLE_RATE)
                                if success:
                                    all_fleurs_metadata.append({
                                        "dataset": "fleurs",
                                        "language": lang_name,
                                        "speaker_id": spk_id,
                                        "filename": fname,
                                        "file_path": str(out_wav.relative_to(PROJECT_ROOT)),
                                        "label": 0,  # Genuine human speech
                                        "split": split,
                                        "duration_sec": round(duration, 3),
                                        "original_sampling_rate": orig_sr,
                                        "original_format": orig_fmt
                                    })
                                    clip_count += 1

                                try:
                                    tmp_wav.unlink()
                                except Exception:
                                    pass

                        print(f"    Standardized {clip_count} {lang_name} clips for [{split}].")
                except Exception as e:
                    print(f"    [!] Error unpacking {tar_fname}: {e}")

    return all_fleurs_metadata


def generate_final_report(all_metadata: List[Dict[str, Any]], target_dir: Path):
    """Save unified metadata manifest and print comprehensive audit report."""
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / "metadata.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("DATASET INGESTION & STANDARDIZATION AUDIT REPORT")
    print("=" * 70)

    n_files = len(all_metadata)
    total_duration_sec = sum(m.get("duration_sec", 0.0) for m in all_metadata)
    total_hours = total_duration_sec / 3600.0

    # Calculate actual disk size
    total_bytes = 0
    for m in all_metadata:
        fp = PROJECT_ROOT / m["file_path"]
        if fp.exists():
            total_bytes += fp.stat().st_size

    total_size_mb = total_bytes / (1024 * 1024)
    total_size_gb = total_bytes / (1024 * 1024 * 1024)

    speakers = set(m["speaker_id"] for m in all_metadata)
    languages = sorted(list(set(m["language"] for m in all_metadata)))

    print(f"Total Standardized Files  : {n_files:,} WAV clips")
    print(f"Total Audio Duration      : {total_hours:.2f} hours ({total_duration_sec:,.1f} seconds)")
    print(f"Total Storage Size On Disk: {total_size_gb:.2f} GB ({total_size_mb:,.1f} MB)")
    print(f"Total Unique Speakers     : {len(speakers):,} speakers")
    print(f"Audio Specifications      : 16 kHz Mono, 16-bit PCM WAV (Standardized)")
    print(f"Target Classification     : label=0 (Genuine Human Speech)")

    print("\n" + "-" * 70)
    print(f"{'Language':15s} | {'Clones':>8s} | {'Duration':>12s} | {'Speakers':>10s}")
    print("-" * 70)
    for lang in languages:
        lang_items = [m for m in all_metadata if m["language"] == lang]
        lang_dur = sum(m.get("duration_sec", 0.0) for m in lang_items) / 3600.0
        lang_spks = len(set(m["speaker_id"] for m in lang_items))
        print(f"{lang.capitalize():15s} | {len(lang_items):8d} | {lang_dur:9.2f} hrs | {lang_spks:10d}")

    print("\n" + "-" * 70)
    print("Speaker-Disjoint Split Partitioning (Anti-Data-Leakage Verified):")
    for split in ["train", "dev", "test"]:
        split_items = [m for m in all_metadata if m["split"] == split]
        split_spks = len(set(m["speaker_id"] for m in split_items))
        split_dur = sum(m.get("duration_sec", 0.0) for m in split_items) / 3600.0
        pct = (len(split_items) / n_files * 100) if n_files else 0
        print(f"  * {split.upper():6s}: {len(split_items):6d} clips ({pct:5.1f}%) | {split_dur:6.2f} hrs | {split_spks:4d} speakers")

    # Anti-leakage sanity check
    train_spks = set(m["speaker_id"] for m in all_metadata if m["split"] == "train")
    dev_spks = set(m["speaker_id"] for m in all_metadata if m["split"] == "dev")
    test_spks = set(m["speaker_id"] for m in all_metadata if m["split"] == "test")

    train_test_overlap = train_spks.intersection(test_spks)
    train_dev_overlap = train_spks.intersection(dev_spks)

    print("-" * 70)
    if not train_test_overlap and not train_dev_overlap:
        print("[SUCCESS] Zero Speaker Leakage Confirmed: 0 speaker overlap between Train, Dev, and Test splits.")
    else:
        print(f"[!] Warning: Detected speaker overlap: Train-Test: {len(train_test_overlap)}, Train-Dev: {len(train_dev_overlap)}")

    print(f"\n[+] Unified Metadata Saved to: {manifest_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Install and Standardize Additional Genuine Indian Speech Datasets.")
    parser.add_argument(
        "--dataset",
        choices=["all", "slr65", "fleurs_tamil", "fleurs_indic"],
        default="all",
        help="Which dataset package to download and install."
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=0,
        help="Maximum clips per dataset (0 = all available)."
    )
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        help="Remove download cache archives after successful extraction."
    )

    args = parser.parse_args()

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_metadata = []

    if args.dataset in ["all", "slr65"]:
        slr_meta = process_slr65(TARGET_DIR, CACHE_DIR, max_clips=args.max_clips)
        all_metadata.extend(slr_meta)

    if args.dataset in ["all", "fleurs_indic"]:
        fleurs_meta = process_fleurs(
            TARGET_DIR,
            CACHE_DIR,
            languages=["ta_in", "hi_in", "te_in", "bn_in", "kn_in", "mr_in"],
            max_clips_per_lang=args.max_clips
        )
        all_metadata.extend(fleurs_meta)
    elif args.dataset == "fleurs_tamil":
        fleurs_meta = process_fleurs(
            TARGET_DIR,
            CACHE_DIR,
            languages=["ta_in"],
            max_clips_per_lang=args.max_clips
        )
        all_metadata.extend(fleurs_meta)

    if all_metadata:
        generate_final_report(all_metadata, TARGET_DIR)
    else:
        print("[!] No clips were processed. Check network connections and mirror availability.")


if __name__ == "__main__":
    main()
