#!/usr/bin/env bash
set -euo pipefail

echo "== Comprobación GPU AWS =="

echo
echo "Sistema:"
uname -a

echo
echo "NVIDIA:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi no instalado."
fi

echo
echo "Python/PyTorch:"
if [[ -x ".venv/bin/python" ]]; then
  .venv/bin/python - <<'PY'
try:
    import torch
except Exception as exc:
    print("PyTorch no disponible:", exc)
    raise SystemExit(0)

print("PyTorch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
print("CUDA de PyTorch:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
else
  echo "Aún no existe .venv."
fi
