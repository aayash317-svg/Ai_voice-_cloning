#!/usr/bin/env bash
# Voice Shield AI - Cloud Native Build Script
# Ensures CPU-only PyTorch is installed to prevent 2.5GB CUDA download and OOM crashes
set -o errexit

echo "=========================================================="
echo "Voice Shield AI: Starting Cloud Native Build"
echo "=========================================================="

echo "[1/3] Upgrading pip..."
python -m pip install --no-cache-dir --upgrade pip

echo "[2/3] Installing lightweight CPU-only PyTorch (prevents OOM)..."
pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "[3/3] Installing application dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "[+] Creating required runtime directories..."
mkdir -p logs results features experiments

echo "=========================================================="
echo "Voice Shield AI: Cloud Build Completed Successfully!"
echo "=========================================================="
