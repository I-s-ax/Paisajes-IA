#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PID_FILE=".runtime/worker.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No hay PID guardado."
  exit 0
fi

PID="$(cat "$PID_FILE" || true)"

if [[ -z "$PID" ]]; then
  rm -f "$PID_FILE"
  echo "PID inválido."
  exit 0
fi

if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "Worker detenido (PID $PID)."
else
  echo "El proceso ya no estaba activo."
fi

rm -f "$PID_FILE"
