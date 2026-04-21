#!/usr/bin/env bash
set -euo pipefail

path="${1:?usage: bash agent_curl.sh /agent/plan <<'JSON' ... JSON}"
port="${PORT:-8765}"
response_file="$(mktemp)"
trap 'rm -f "$response_file"' EXIT

http_code="$(
curl -sS \
  -o "$response_file" \
  -w '%{http_code}' \
  -X POST \
  -H 'Content-Type: application/json' \
  --data-binary @- \
  "http://localhost:${port}${path}"
)"

cat "$response_file"

if [[ "$http_code" =~ ^[0-9]{3}$ ]] && (( http_code >= 400 )); then
  printf '\n[agent_curl] HTTP %s for %s\n' "$http_code" "$path" >&2
  exit 22
fi
