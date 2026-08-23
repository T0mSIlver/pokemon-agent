#!/usr/bin/env bash
# Open an SSH tunnel from your laptop to the machine running pokemon-agent,
# so the dashboard and the game frames are reachable at http://localhost:<PORT>.
#
# Run this ON YOUR LAPTOP, not on the server.
#
#   scripts/tunnel.sh dev@192.168.1.98            # forward 8765 -> 8765
#   scripts/tunnel.sh dev@192.168.1.98 8765 9000  # remote 8765 -> local 9000
#
# Leave it running; Ctrl-C closes the tunnel.
set -euo pipefail

REMOTE="${1:-}"
REMOTE_PORT="${2:-8765}"
LOCAL_PORT="${3:-$REMOTE_PORT}"

if [[ -z "$REMOTE" ]]; then
  echo "Usage: $0 <user@host> [REMOTE_PORT] [LOCAL_PORT]" >&2
  exit 1
fi

cat <<EOF
Tunnelling ${REMOTE}:${REMOTE_PORT} -> localhost:${LOCAL_PORT}

  Dashboard:  http://localhost:${LOCAL_PORT}/dashboard
  Live frame: http://localhost:${LOCAL_PORT}/artifacts/live_frame_annotated
  Turn frame: http://localhost:${LOCAL_PORT}/artifacts/latest_frame_annotated
  Screenshot: http://localhost:${LOCAL_PORT}/screenshot

Ctrl-C to close.
EOF

exec ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
  "$REMOTE"
