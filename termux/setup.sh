#!/usr/bin/env bash
set -euo pipefail

echo "== Paisajes-IA para Termux =="

pkg install -y python openssh git curl

echo
echo "Solicitando acceso al almacenamiento compartido de Android..."
termux-setup-storage || true

echo
echo "Listo."
echo "Comprueba SSH con el comando que te proporciona Lightning AI."
echo
echo "Después ejecuta:"
echo "  python termux/client.py --help"
