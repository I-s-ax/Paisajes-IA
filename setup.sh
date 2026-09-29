#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "== Paisajes-IA =="
echo "Comprobando Python..."
python --version

echo
echo "Comprobando PyTorch..."
python - <<'PY'
try:
    import torch
except Exception as exc:
    raise SystemExit(
        "PyTorch no está disponible en este Studio. "
        "Usa una imagen de Lightning que incluya PyTorch.\n"
        f"Error: {exc}"
    )

print("PyTorch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Ahora mismo se usará CPU. Puedes cambiar el Studio a GPU después.")
PY

echo
echo "Instalando dependencias..."
python -m pip install -U -r requirements.txt

echo
echo "Listo."
echo "Prueba:"
echo "  python src/analyze.py --help"
