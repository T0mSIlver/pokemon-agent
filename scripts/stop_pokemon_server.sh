#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-8765}"
WORKSPACE_DIR="${2:-$ROOT_DIR/.agent-workspace}"
PID_FILE="${3:-$WORKSPACE_DIR/pokemon-agent.pid}"

stop_pid() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    return 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi

  kill "$pid" 2>/dev/null || true
  for _ in {1..20}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
    sleep 0.25
  done

  kill -9 "$pid" 2>/dev/null || true
}

STOPPED_ANY=0

if [[ -f "$PID_FILE" ]]; then
  PID_FROM_FILE="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$PID_FROM_FILE" ]]; then
    stop_pid "$PID_FROM_FILE"
    STOPPED_ANY=1
  fi
  rm -f "$PID_FILE"
fi

while IFS= read -r pid; do
  if [[ -n "$pid" ]]; then
    stop_pid "$pid"
    STOPPED_ANY=1
  fi
done < <(lsof -ti tcp:"$PORT" -sTCP:LISTEN 2>/dev/null || true)

if [[ "$STOPPED_ANY" -eq 1 ]]; then
  echo "Stopped pokemon-agent server on port $PORT"
else
  echo "No pokemon-agent server found on port $PORT"
fi
