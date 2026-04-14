#!/usr/bin/env bash
set -euo pipefail

path="${1:?usage: bash scripts/agent_curl.sh /agent/plan <<'JSON' ... JSON}"
port="${PORT:-8000}"

curl -sf \
  -X POST \
  -H 'Content-Type: application/json' \
  --data-binary @- \
  "http://localhost:${port}${path}"
