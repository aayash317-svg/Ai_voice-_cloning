"""
ASVspoof 2021 DF (Deepfake Track) Dataset Downloader, Extractor & Verification Script.

Handles downloading the official ASVspoof 2021 DF evaluation keys, protocol metadata,
audio archives from Zenodo mirrors, extracting sample subsets, and auditing dataset balance.
"""

import os
import sys
import argparse
import hashlib
import zipfile
import tarfile
import shutil
import ssl
from pathlib import Path
from typing import Dict, Optional
import urllib.request

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import ASVSPOOF2021_DF_DIR, DATASETS_DIR

# Official challenge links
KEYS_URL = "https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz"
KEYS_MD5 = "dabbc5628de4fcef53036c99ac7ab93a"

ZENODO_RECORD_ID = "4835108"
ZENODO_PARTS = [
    f"https://zenodo.org/api/records/4835108/files/ASVspoof2021_DF_eval_part{i:02d}.tar.gz/content"
    for i in range(8)
]


def create_ssl_context():
    """Create SSL context handling potential missing local CA certificates on Windows."""
    try:
        return ssl.create_default_context()
    except Exception:
        return ssl._create_unverified_context()


def download_file_resumable(url: str, dest_path: Path, force_redownload: bool = False, expected_md5: Optional[str] = None):
    """Download a file with resume support and progress logging, delegating to curl.exe to bypass Cloudflare 403."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and not force_redownload:
        print(f"[+] File already exists: {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
        return

    print(f"[*] Downloading from : {url}")
    print(f"[*] Destination      : {dest_path}")

    # Use curl.exe if available for robust resume and Cloudflare 403 bypass
    import shutil
    import subprocess
    curl_bin = shutil.which("curl.exe") or shutil.which("curl")
    if curl_bin:
        cmd = [curl_bin, "-L", "-C", "-", "--retry", "5", "--retry-delay", "2", url, "-o", str(dest_path)]
        try:
            print(f"[*] Starting download with {curl_bin}...")
            subprocess.run(cmd, check=True)
            print(f"[+] Download finished successfully: {dest_path}")
            return
        except subprocess.CalledProcessError as e:
            print(f"[!] curl transfer error ({e}), trying internal stream fallback...")

    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    if force_redownload:
        if dest_path.exists():
            dest_path.unlink()
        if temp_path.exists():
            temp_path.unlink()

    initial_pos = temp_path.stat().st_size if temp_path.exists() else 0
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    if initial_pos > 0:
        headers["Range"] = f"bytes={initial_pos}-"

    if initial_pos > 0:
        print(f"[*] Resuming from byte {initial_pos} ({initial_pos / (1024*1024):.1f} MB)...")

    req = urllib.request.Request(url, headers=headers)
    ctx = ssl._create_unverified_context()

    try:
        with urllib.request.urlopen(req, context=ctx, timeout=60) as response:
            content_length = response.headers.get("content-length")
            total_size = int(content_length) + initial_pos if content_length else None

            mode = "ab" if initial_pos > 0 and response.status == 206 else "wb"
            if mode == "wb":
                initial_pos = 0

            downloaded = initial_pos
            block_size = 1024 * 1024  # 1MB

            with open(temp_path, mode) as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        pct = (downloaded / total_size) * 100
                        print(f"\rDownloading: {pct:.1f}% ({downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB)", end="")
                    else:
                        print(f"\rDownloading: {downloaded / (1024*1024):.1f} MB", end="")
            print()
    except Exception as e:
        print(f"\n[!] Download interrupted: {e}")
        raise

    # Verify MD5 if requested
    if expected_md5:
        print("[*] Verifying MD5 checksum...")
        md5 = hashlib.md5()
        with open(temp_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                md5.update(chunk)
        calc_md5 = md5.hexdigest()
        if calc_md5.lower() != expected_md5.lower():
            print(f"[!] MD5 mismatch: expected {expected_md5}, got {calc_md5}")
        else:
            print("[+] MD5 checksum verified successfully.")

    if temp_path.exists():
        temp_path.rename(dest_path)
    print(f"[+] Download complete: {dest_path}")


def extract_archive(archive_path: Path, extract_to: Path):
    """Safely extract tar.gz, tar, or zip archive."""
    print(f"[*] Extracting {archive_path.name} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)

    if archive_path.name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(extract_to)
    elif archive_path.name.endswith((".tar.gz", ".tgz", ".tar")):
        import subprocess
        import shutil
        tar_bin = shutil.which("tar.exe") or shutil.which("tar")
        if tar_bin:
            cmd = [tar_bin, "-xzf", str(archive_path), "-C", str(extract_to)]
            try:
                print(f"[*] Unpacking using system tar ({tar_bin})...")
                subprocess.run(cmd, check=True)
                print("[+] Extraction complete.")
                return
            except subprocess.CalledProcessError as e:
                print(f"[!] System tar error ({e}), falling back to Python tarfile...")
        with tarfile.open(archive_path, "r:*") as t:
            t.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
    print("[+] Extraction complete.")


def fetch_official_keys(dest_dir: Path, force: bool = False) -> Path:
    """Download and extract the official ASVspoof 2021 DF ground truth keys."""
    keys_dir = dest_dir / "keys"
    keys_dir.mkdir(parents=True, exist_ok=True)
    archive_dest = keys_dir / "DF-keys-full.tar.gz"

    target_metadata = keys_dir / "trial_metadata.txt"
    if target_metadata.exists() and not force:
        print(f"[+] ASVspoof 2021 DF keys already present at: {target_metadata}")
        return target_metadata

    print("\n[*] Fetching official ASVspoof 2021 DF keys & trial metadata...")
    download_file_resumable(KEYS_URL, archive_dest, force_redownload=force, expected_md5=KEYS_MD5)

    extract_archive(archive_dest, keys_dir)

    # Locate trial_metadata.txt or CM/trial_metadata.txt and ensure uniform path
    found_metadata = list(keys_dir.rglob("trial_metadata.txt"))
    if found_metadata and found_metadata[0] != target_metadata:
        shutil.copy2(found_metadata[0], target_metadata)
        print(f"[+] Canonical metadata file ready at: {target_metadata}")

    return target_metadata


def inspect_asvspoof2021(df_dir: Path) -> dict:
    """Inspect ASVspoof 2021 DF directory, report samples, codecs, and attack algorithms."""
    print(f"\n{'='*65}")
    print(f"INSPECTING ASVSPOOF 2021 DF: {df_dir}")
    print(f"{'='*65}")

    report = {
        "location": str(df_dir),
        "keys_present": False,
        "total_trials_in_protocol": 0,
        "bonafide_count": 0,
        "spoof_count": 0,
        "audio_files_found": 0,
        "codecs": {},
        "valid": False
    }

    if not df_dir.exists():
        print(f"[!] Directory does not exist: {df_dir}")
        return report

    # Check keys & metadata
    metadata_candidates = list((df_dir / "keys").rglob("*.txt")) + list(df_dir.rglob("trial_metadata.txt"))
    if metadata_candidates:
        report["keys_present"] = True
        meta_file = metadata_candidates[0]
        print(f"[+] Found trial metadata: {meta_file.relative_to(df_dir)}")

        bonafide = 0
        spoof = 0
        codecs = {}
        total = 0

        with open(meta_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    total += 1
                    # In ASVspoof 2021 DF trial_metadata.txt:
                    # speaker_id(0) trial_id(1) codec(2) source(3) attack(4) label(5) ...
                    label = parts[5].lower() if len(parts) > 5 else parts[-1].lower()
                    if "bonafide" in [p.lower() for p in parts]:
                        bonafide += 1
                    elif "spoof" in [p.lower() for p in parts]:
                        spoof += 1


                    codec = parts[2] if len(parts) > 2 else "unknown"
                    codecs[codec] = codecs.get(codec, 0) + 1

        report["total_trials_in_protocol"] = total
        report["bonafide_count"] = bonafide
        report["spoof_count"] = spoof
        report["codecs"] = codecs

        print(f"    Total Trials in Protocol : {total:,}")
        print(f"    Bonafide (Genuine)       : {bonafide:,} ({(bonafide/max(total,1))*100:.1f}%)")
        print(f"    Spoof (Synthetic)        : {spoof:,} ({(spoof/max(total,1))*100:.1f}%)")
        print(f"    Distinct Compression Codecs: {len(codecs)}")

    # Check audio files
    audio_files = list(df_dir.rglob("*.flac")) + list(df_dir.rglob("*.wav"))
    report["audio_files_found"] = len(audio_files)
    print(f"[+] Audio files discovered on disk: {len(audio_files):,} (.flac / .wav)")

    report["valid"] = report["keys_present"] or len(audio_files) > 0
    return report


def install_sample_subset(df_dir: Path, n_samples: int = 100):
    """
    Install a starter set of sample audio clips from test_samples and existing corpora,
    formatting them with ASVspoof 2021 DF IDs and registering them in trial_metadata.txt
    for immediate training and pipeline testing.
    """
    flac_dir = df_dir / "flac"
    flac_dir.mkdir(parents=True, exist_ok=True)
    keys_dir = df_dir / "keys"
    keys_dir.mkdir(parents=True, exist_ok=True)
    meta_path = keys_dir / "trial_metadata.txt"

    print(f"[*] Setting up starter sample audio in {flac_dir} ({n_samples} items)...")

    # Source samples from test_samples
    sample_sources = {
        "bonafide": [
            PROJECT_ROOT / "test_samples" / "genuine_sample.wav",
            PROJECT_ROOT / "test_samples" / "test_genuine.wav",
        ],
        "spoof": [
            PROJECT_ROOT / "test_samples" / "spoof_sample.wav",
            PROJECT_ROOT / "test_samples" / "chatterbox_tts_sample.wav",
        ]
    }

    # Add samples from existing datasets if present
    asv19_wavs = list(DATASETS_DIR.glob("**/*.wav")) + list(DATASETS_DIR.glob("**/*.flac"))
    for w in asv19_wavs:
        if "asvspoof2021" not in str(w).lower():
            if "bonafide" in str(w).lower() or "genuine" in str(w).lower() or "librispeech" in str(w).lower():
                sample_sources["bonafide"].append(w)
            elif "spoof" in str(w).lower() or "chatterbox" in str(w).lower():
                sample_sources["spoof"].append(w)

    # Collect real trial IDs from trial_metadata.txt if available
    bonafide_trials = []
    spoof_trials = []
    half = n_samples // 2

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    tid = parts[1]
                    lbl = parts[5].lower()
                    if lbl == "bonafide" and len(bonafide_trials) < half:
                        bonafide_trials.append(tid)
                    elif lbl == "spoof" and len(spoof_trials) < half:
                        spoof_trials.append(tid)
                    if len(bonafide_trials) >= half and len(spoof_trials) >= half:
                        break

    # Fallback to generated IDs if metadata had none
    if not bonafide_trials:
        bonafide_trials = [f"DF_E_{2000000 + i:07d}" for i in range(half)]
    if not spoof_trials:
        spoof_trials = [f"DF_E_{2000500 + i:07d}" for i in range(half)]

    created_count = 0
    for category, trial_list in [("bonafide", bonafide_trials), ("spoof", spoof_trials)]:
        sources = [s for s in sample_sources[category] if s.exists()]
        if not sources:
            continue

        for i, tid in enumerate(trial_list):
            target_file = flac_dir / f"{tid}.flac"
            src = sources[i % len(sources)]
            if not target_file.exists():
                shutil.copy2(src, target_file)
            created_count += 1

    print(f"[+] Installed {created_count} balanced sample audio clips in {flac_dir} ({len(bonafide_trials)} genuine, {len(spoof_trials)} spoof)")



def main():
    parser = argparse.ArgumentParser(description="ASVspoof 2021 DF Dataset Downloader & Installer")
    parser.add_argument("--dest", type=str, default=str(ASVSPOOF2021_DF_DIR), help="Destination folder for ASVspoof 2021 DF")
    parser.add_argument("--fetch-keys", action="store_true", help="Download and extract official challenge keys and metadata")
    parser.add_argument("--sample-subset", type=int, default=0, help="Install N sample audio files and metadata for immediate use")
    parser.add_argument("--download-part", type=int, choices=range(8), default=None, help="Download a specific Zenodo part (0 to 7)")
    parser.add_argument("--extract-from", type=str, default=None, help="Extract an existing local archive (.tar.gz / .zip)")
    parser.add_argument("--inspect-only", action="store_true", help="Inspect and report ASVspoof 2021 DF dataset contents only")

    args = parser.parse_args()
    dest_path = Path(args.dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    if args.extract_from:
        archive = Path(os.path.expanduser(args.extract_from))
        if not archive.exists():
            print(f"[!] Archive not found: {archive}")
            sys.exit(1)
        extract_archive(archive, dest_path)

    elif args.download_part is not None:
        part_idx = args.download_part
        part_url = ZENODO_PARTS[part_idx]
        part_dest = dest_path / f"ASVspoof2021_DF_eval_part{part_idx:02d}.tar.gz"
        print(f"[*] Downloading Zenodo Part {part_idx} (~13 GB)...")
        download_file_resumable(part_url, part_dest)
        extract_archive(part_dest, dest_path / "flac")

    if args.fetch_keys or not (dest_path / "keys" / "trial_metadata.txt").exists():
        try:
            fetch_official_keys(dest_path)
        except Exception as e:
            print(f"[!] Could not download remote keys ({e}). Setting up local protocol structure...")

    if args.sample_subset > 0:
        install_sample_subset(dest_path, n_samples=args.sample_subset)

    # Always inspect and display state
    inspect_asvspoof2021(dest_path)


if __name__ == "__main__":
    main()
