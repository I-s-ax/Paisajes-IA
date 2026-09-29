#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p .runtime

PID_FILE=".runtime/worker.pid"
LOG_FILE=".runtime/worker.log"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "El worker ya está ejecutándose (PID $OLD_PID)."
    echo "Log: $LOG_FILE"
    exit 0
  fi
fi

nohup python cloud/worker.py --root /tmp/paisajes-ia >"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

echo "Worker iniciado (PID $PID)."
echo "Log: $LOG_FILE"
echo "Ver estado:"
echo "  tail -f $LOG_FILE"
