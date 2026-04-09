#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROM_PATH="${1:-}"
PORT="${2:-8765}"
WORKSPACE_DIR="${3:-$ROOT_DIR/.agent-workspace}"
PID_FILE="${4:-$WORKSPACE_DIR/pokemon-agent.pid}"
LOG_FILE="${5:-$ROOT_DIR/server.log}"

if [[ -z "$ROM_PATH" ]]; then
  echo "Usage: $0 <ROM_PATH> [PORT] [WORKSPACE_DIR] [PID_FILE] [LOG_FILE]" >&2
  exit 1
fi

mkdir -p "$WORKSPACE_DIR"

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "pokemon-agent server already running on PID $EXISTING_PID"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

PORT_PID="$(lsof -ti tcp:"$PORT" -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
if [[ -n "$PORT_PID" ]]; then
  echo "Port $PORT is already in use by PID $PORT_PID" >&2
  echo "Stop it first with: scripts/stop_pokemon_server.sh $PORT $WORKSPACE_DIR" >&2
  exit 1
fi

nohup uv run pokemon-agent serve \
  --rom "$ROM_PATH" \
  --port "$PORT" \
  --agent-workspace-dir "$WORKSPACE_DIR" \
  > "$LOG_FILE" 2>&1 &

SERVER_PID="$!"
echo "$SERVER_PID" > "$PID_FILE"

echo "Started pokemon-agent server"
echo "PID: $SERVER_PID"
echo "Port: $PORT"
echo "Workspace: $WORKSPACE_DIR"
echo "PID file: $PID_FILE"
echo "Log file: $LOG_FILE"
