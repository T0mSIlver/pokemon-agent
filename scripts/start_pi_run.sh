#!/usr/bin/env bash
# Launch a supervised Pi run against an already-running pokemon-agent server.
#
# Waits until the model endpoint actually answers (local models often need to be
# swapped in), then POSTs /supervisor/start.
#
#   scripts/start_pi_run.sh
#   MODEL=llamacpp/qwen36dense-27b scripts/start_pi_run.sh
#   GOAL="Get to Viridian City and buy Potions" scripts/start_pi_run.sh
set -euo pipefail

PORT="${PORT:-8765}"
SERVER="${SERVER:-http://127.0.0.1:$PORT}"
PROVIDER="${PROVIDER:-llamacpp}"
MODEL="${MODEL:-llamacpp/qwen38-27b}"
THINKING="${THINKING:-max}"
GOAL="${GOAL:-}"
AUTO_CONTINUE="${AUTO_CONTINUE:-true}"
MAX_TURNS="${MAX_TURNS:-null}"
CONTINUE_DELAY="${CONTINUE_DELAY:-1.0}"
MODEL_BASE_URL="${MODEL_BASE_URL:-http://192.168.1.183:8080/v1}"
WAIT_SECONDS="${WAIT_SECONDS:-0}"

if ! curl -sf -m 10 "$SERVER/health" > /dev/null; then
  echo "pokemon-agent server is not answering at $SERVER" >&2
  echo "Start it first: scripts/start_pokemon_server.sh <ROM_PATH>" >&2
  exit 1
fi

probe_model() {
  curl -s -m 300 "$MODEL_BASE_URL/chat/completions" \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer ${MODEL_API_KEY:-dummy}" \
    -d "{\"model\":\"$MODEL\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}"
}

deadline=$(( $(date +%s) + WAIT_SECONDS ))
while true; do
  response="$(probe_model || true)"
  if grep -q '"choices"' <<< "$response"; then
    echo "Model $MODEL is answering."
    break
  fi
  message="$(python3 -c 'import json,sys; print((json.load(sys.stdin).get("error") or {}).get("message",""))' <<< "$response" 2>/dev/null || true)"
  echo "Model $MODEL not ready: ${message:-no response}"
  if (( $(date +%s) >= deadline )); then
    echo "Giving up waiting for $MODEL." >&2
    exit 1
  fi
  sleep 15
done

payload="$(python3 - "$GOAL" "$PROVIDER" "$MODEL" "$THINKING" "$AUTO_CONTINUE" "$MAX_TURNS" "$CONTINUE_DELAY" <<'PY'
import json, sys
goal, provider, model, thinking, auto, max_turns, delay = sys.argv[1:8]
body = {
    "provider": provider,
    "model": model,
    "thinking": thinking,
    "auto_continue": auto.lower() == "true",
    "max_turns": None if max_turns in ("", "null") else int(max_turns),
    "continue_delay_seconds": float(delay),
}
if goal.strip():
    body["goal"] = goal.strip()
print(json.dumps(body))
PY
)"

echo "Starting Pi: $payload"
curl -sS -X POST "$SERVER/supervisor/start" \
  -H 'Content-Type: application/json' \
  -d "$payload" | python3 -m json.tool
