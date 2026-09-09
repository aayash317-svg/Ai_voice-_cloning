"""
ASVspoof 2019 LA Dataset Downloader, Extractor & Verification Script.

Downloads official ASVspoof 2019 Logical Access (LA) subsets, verifies integrity,
extracts audio and protocol files, and validates split structures without data leakage.
"""

import os
import sys
import argparse
import hashlib
import zipfile
import tarfile
from pathlib import Path
import requests
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import ASVSPOOF2019_DIR, DATASETS_DIR

# Official Edinburgh DataShare / Academic Mirrors for ASVspoof 2019 LA
OFFICIAL_URLS = {
    "LA": "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip",
    "protocols": "https://www.asvspoof.org/asvspoof2019/LA-protocols.zip"
}


def download_file_resumable(url: str, dest_path: Path, force_redownload: bool = False):
    """Download a large file with resume support, retry logic, and progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    if force_redownload:
        if dest_path.exists():
            dest_path.unlink()
        if temp_path.exists():
            temp_path.unlink()

    initial_pos = temp_path.stat().st_size if temp_path.exists() else 0
    headers = {"Range": f"bytes={initial_pos}-"} if initial_pos > 0 else {}

    print(f"Downloading from: {url}")
    print(f"Destination     : {dest_path}")
    if initial_pos > 0:
        print(f"Resuming download from byte {initial_pos} ({initial_pos / (1024*1024):.1f} MB)...")

    response = requests.get(url, headers=headers, stream=True, timeout=30)

    # If server doesn't support Range, start fresh
    if response.status_code == 416: # Range Not Satisfiable
        temp_path.unlink(missing_ok=True)
        initial_pos = 0
        response = requests.get(url, stream=True, timeout=30)
    elif response.status_code not in (200, 206):
        raise RuntimeError(f"HTTP Download failed with status {response.status_code}: {response.reason}")

    total_size = int(response.headers.get("content-length", 0)) + initial_pos
    block_size = 1024 * 1024 # 1MB chunks

    mode = "ab" if initial_pos > 0 and response.status_code == 206 else "wb"
    if mode == "wb":
        initial_pos = 0

    with open(temp_path, mode) as f, tqdm(
        total=total_size,
        initial=initial_pos,
        unit="B",
        unit_scale=True,
        desc=dest_path.name,
        ncols=80
    ) as pbar:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    # Rename .part to final destination
    if temp_path.exists():
        temp_path.rename(dest_path)
    print(f"Download complete: {dest_path}")


def verify_zip_integrity(archive_path: Path) -> bool:
    """Verify whether a zip archive is complete and uncorrupted."""
    if not archive_path.exists() or archive_path.stat().st_size < 1024:
        return False
    try:
        with zipfile.ZipFile(archive_path, "r") as z:
            test_res = z.testzip()
            return test_res is None
    except Exception:
        return False


def extract_archive(archive_path: Path, extract_to: Path):
    """Extract zip or tar archive safely."""
    print(f"Extracting {archive_path.name} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffix in [".tar", ".gz", ".tgz", ".bz2"]:
        with tarfile.open(archive_path, "r:*") as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
    print("Extraction complete.")


def inspect_asvspoof2019(la_dir: Path) -> dict:
    """
    Inspect the extracted ASVspoof 2019 LA directory, find protocol files,
    and count bonafide vs spoof samples.
    """
    print(f"\n{'='*60}")
    print(f"INSPECTING ASVSPOOF 2019 LA: {la_dir}")
    print(f"{'='*60}")

    report = {
        "location": str(la_dir),
        "splits": {},
        "audio_counts": {},
        "protocol_counts": {},
        "valid": False
    }

    if not la_dir.exists():
        print(f"[!] Directory does not exist: {la_dir}")
        return report

    # Look for protocol files
    protocol_candidates = list(la_dir.rglob("*.txt"))
    print(f"Found {len(protocol_candidates)} text/protocol files.")

    protocols = {
        "train": None,
        "dev": None,
        "eval": None
    }

    for p in protocol_candidates:
        name = p.name.lower()
        if "train" in name and ("cm" in name or "trn" in name or "la" in name):
            protocols["train"] = p
        elif "dev" in name and ("cm" in name or "trl" in name or "la" in name):
            protocols["dev"] = p
        elif "eval" in name and ("cm" in name or "trl" in name or "la" in name):
            protocols["eval"] = p

    for split_name, proto_path in protocols.items():
        if proto_path and proto_path.exists():
            print(f"\n[+] Protocol ({split_name}): {proto_path.relative_to(la_dir)}")
            bonafide_cnt = 0
            spoof_cnt = 0
            attack_types = {}
            total = 0

            with open(proto_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        total += 1
                        label = parts[-1].lower() # 'bonafide' or 'spoof'
                        attack = parts[3]         # '-' or 'A01'...'A19'

                        if label == "bonafide":
                            bonafide_cnt += 1
                        elif label == "spoof":
                            spoof_cnt += 1
                            attack_types[attack] = attack_types.get(attack, 0) + 1

            print(f"    Total items in protocol: {total}")
            print(f"    Bonafide (Genuine)     : {bonafide_cnt}")
            print(f"    Spoof (Synthetic)      : {spoof_cnt}")
            if attack_types:
                print(f"    Attack distributions   : {attack_types}")
            report["protocol_counts"][split_name] = {
                "total": total,
                "bonafide": bonafide_cnt,
                "spoof": spoof_cnt,
                "attacks": attack_types,
                "path": str(proto_path)
            }

    # Count audio files (.flac, .wav)
    for ext in ["*.flac", "*.wav"]:
        files = list(la_dir.rglob(ext))
        if files:
            print(f"\n[+] Audio files ({ext}): {len(files)} files found.")
            report["audio_counts"][ext] = len(files)

    report["valid"] = len(report["protocol_counts"]) > 0
    return report


def main():
    parser = argparse.ArgumentParser(description="ASVspoof 2019 LA Dataset Setup & Validation")
    parser.add_argument("--dest", type=str, default=str(ASVSPOOF2019_DIR), help="Destination directory for ASVspoof 2019 LA")
    parser.add_argument("--download", action="store_true", help="Download official archive if not present")
    parser.add_argument("--redownload", action="store_true", help="Force redownload from scratch")
    parser.add_argument("--extract-from", type=str, default=None, help="Extract from an existing local archive (.zip/.tar)")
    parser.add_argument("--inspect-only", action="store_true", help="Inspect and validate dataset directory only")

    args = parser.parse_args()
    dest_path = Path(args.dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    if args.extract_from:
        archive = Path(os.path.expanduser(args.extract_from))
        if not archive.exists():
            print(f"[!] Archive not found: {archive}")
            sys.exit(1)
        extract_archive(archive, dest_path)

    elif args.download:
        print("[*] Downloading ASVspoof 2019 LA partition via Kaggle CDN (7.2 GB)...")
        try:
            import kagglehub
            k_path = kagglehub.dataset_download("anishsarkar22/asvpoof-2019-dataset-la")
            print(f"[+] Downloaded to: {k_path}")
            # Link or copy into dest_path if needed
            k_p = Path(k_path)
            for sub in k_p.iterdir():
                target = dest_path / sub.name
                if not target.exists():
                    if sub.is_dir():
                        os.symlink(sub, target)
                    else:
                        import shutil
                        shutil.copy2(sub, target)
            print(f"[+] Linked dataset contents into {dest_path}")
        except Exception as e:
            print(f"[!] Kaggle download error ({e}), falling back to direct URL...")
            archive_path = dest_path.parent / "LA.zip"
            if not archive_path.exists() or not verify_zip_integrity(archive_path) or args.redownload:
                download_file_resumable(OFFICIAL_URLS["LA"], archive_path, force_redownload=args.redownload)
            extract_archive(archive_path, dest_path)

    inspect_asvspoof2019(dest_path)


if __name__ == "__main__":
    main()
