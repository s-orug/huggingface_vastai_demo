#!/bin/bash

# Upgrade pip
echo "[*] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "[*] Installing dependencies..."
pip install flask torch transformers bitsandbytes accelerate
