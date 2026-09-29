#!/usr/bin/env bash
set -euo pipefail

echo "== Paisajes-IA / AWS EC2: preparación CPU =="

if ! command -v apt-get >/dev/null 2>&1; then
  echo "Este script está preparado para Ubuntu/Debian."
  exit 1
fi

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git \
  curl \
  ca-certificates \
  python3 \
  python3-venv \
  python3-pip \
  build-essential \
  unzip \
  jq

python3 --version
uname -m

echo
echo "Preparación básica terminada."
echo "Todavía NO instalamos PyTorch/CUDA ni arrancamos la IA."
echo "Eso se hará cuando la instancia se cambie a GPU."
