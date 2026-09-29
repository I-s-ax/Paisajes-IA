#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== Paisajes-IA / AWS EC2: configuración GPU =="

ARCH="$(uname -m)"
echo "Arquitectura: $ARCH"

if [[ "$ARCH" != "x86_64" ]]; then
  echo "ERROR: se esperaba x86_64 para g4dn."
  exit 1
fi

echo
echo "Comprobando GPU NVIDIA..."

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "No se encontró nvidia-smi."
  echo "Instalando herramientas para el controlador NVIDIA..."
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ubuntu-drivers-common \
    "linux-headers-$(uname -r)"

  echo
  echo "Instalando el controlador NVIDIA recomendado para cómputo..."
  sudo ubuntu-drivers install --gpgpu

  echo
  echo "El controlador fue instalado."
  echo "REINICIA la instancia:"
  echo "  sudo reboot"
  echo
  echo "Después vuelve a ejecutar:"
  echo "  cd ~/Paisajes-IA"
  echo "  bash aws/setup_gpu.sh"
  exit 0
fi

echo
nvidia-smi

echo
echo "Creando entorno Python..."
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

echo
echo "Instalando PyTorch..."
python -m pip install torch torchvision

echo
echo "Instalando dependencias de Paisajes-IA..."
python -m pip install -r requirements.txt

echo
echo "Comprobando CUDA desde PyTorch..."
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
print("CUDA de PyTorch:", torch.version.cuda)

if not torch.cuda.is_available():
    raise SystemExit(
        "PyTorch no detecta CUDA. Revisa nvidia-smi/controlador antes de continuar."
    )

print("GPU:", torch.cuda.get_device_name(0))
print("Memoria GPU:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GiB")
print("GPU lista para Paisajes-IA.")
PY

echo
echo "Descargando modelos mientras la GPU está disponible..."
python src/download_models.py

echo
echo "Configuración GPU terminada."
echo
echo "Para iniciar el servidor:"
echo "  source .venv/bin/activate"
echo "  export PAISAJES_API_KEY='TU_CLAVE'"
echo "  python cloud/http_server.py --device cuda --host 127.0.0.1 --port 8000"
