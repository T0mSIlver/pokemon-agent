#!/usr/bin/env bash
# Keep a supervised run going. When a run ends — token budget, idle breaker,
# operator stop — start a fresh Pi session. The emulator keeps its state, so the
# new session picks the game up where the last one left it with a clean context.
# Between the two, the supervisor reports 'critiquing' while it writes the
# retrospective the next session reads; that counts as busy.
#
# 'idle' in the log below is the one status this script cannot cause: it is what a
# freshly constructed supervisor says, so seeing it means the *server* restarted
# and took a session with it. No retrospective was written for that session, and
# none can be - the process that would have written it is gone. The supervisor
# covers that case by reading its ground truth off the run receipts at start
# instead of out of memory.
#
#   scripts/keep_run_alive.sh
#   THINKING=low MODEL=llamacpp/qwen38-27b scripts/keep_run_alive.sh
#
# Runs until you kill it.
set -uo pipefail

PORT="${PORT:-8765}"
SERVER="${SERVER:-http://127.0.0.1:$PORT}"
PROVIDER="${PROVIDER:-llamacpp}"
MODEL="${MODEL:-llamacpp/qwen38-27b}"
THINKING="${THINKING:-off}"
GOAL="${GOAL:-}"
GOAL_STICKY="${GOAL_STICKY:-0}"
POLL_SECONDS="${POLL_SECONDS:-30}"

# Goal semantics, and why they are not "re-pin GOAL on every restart".
#
# They used to be. The operator pinned "Beat Brock for the Boulder Badge", the run
# won it, and every session after that still opened by being told to go win it: the
# goal was a static string that outlived its own completion, and the retrospective
# ended up reporting the badge directly under an order to go and get it.
#
# So GOAL is an opening shove, not standing orders. It is sent on the FIRST start
# this watchdog performs and left out of every restart after it. A start that names
# no goal lets the supervisor walk its precedence chain instead - the NEXT GOAL line
# the last critique wrote, else the objective engine's current objective - and that
# chain is rewritten after every session, so it cannot go stale the way GOAL did.
#
# GOAL_STICKY=1 restores the old behaviour and re-pins GOAL on every restart. That
# is for a goal that genuinely spans the whole run ("log every wild encounter you
# see"), never for one the run can finish.
#
# The pinned goal lives only inside one watchdog process: to change it mid-run,
# restart the watchdog with a new GOAL.
build_payload() {
  GOAL="$1" PROVIDER="$PROVIDER" MODEL="$MODEL" THINKING="$THINKING" python3 -c '
import json, os
body = {
    "provider": os.environ["PROVIDER"],
    "model": os.environ["MODEL"],
    "thinking": os.environ["THINKING"],
    "auto_continue": True,
    "continue_delay_seconds": 1.0,
}
goal = os.environ.get("GOAL", "").strip()
if goal:
    body["goal"] = goal
print(json.dumps(body))
'
}

PAYLOAD="$(build_payload "$GOAL")"
RESTART_PAYLOAD="$(build_payload "")"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

read_field() {
  python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get(sys.argv[1], ""))
except Exception:
    print("")' "$1" 2>/dev/null
}

# A run can report 'running' while doing nothing at all. It has happened: the
# disk filled, every action returned ENOSPC, and the session sat in that state
# for four hours because the supervisor was technically alive. Status alone is
# not liveness.
#
# This watched `emulation.frame_count` for months and could never have fired:
# the 60 Hz realtime loop ticks that counter forever whether or not the agent
# has done anything, so it was dead code guarding the exact failure it was
# written for. `run.presses` only moves when a batch is actually executed.
#
# The threshold has to be generous, because a local 27B model reading maps and
# running sims between batches legitimately goes quiet: this run has ten gaps
# over nine minutes and one of twenty-one. Ten minutes would restart healthy
# sessions all day.
STALL_SECONDS="${STALL_SECONDS:-1800}"
DISK_MIN_MB="${DISK_MIN_MB:-2048}"
last_frame=""
last_frame_at=0

frame_count() {
  curl -sf -m 10 "$SERVER/health" 2>/dev/null | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("run", {}).get("presses", ""))
except Exception:
    print("")' 2>/dev/null
}

free_mb() {
  df -Pm "${DISK_CHECK_PATH:-$PWD}" 2>/dev/null | awk 'NR==2 {print $4}'
}

log "watching $SERVER (model=$MODEL thinking=$THINKING)"

while true; do
  snapshot="$(curl -sf -m 10 "$SERVER/supervisor/state" 2>/dev/null || true)"
  status="$(printf '%s' "$snapshot" | read_field status)"

  # Check liveness and disk before trusting the status, because the failure that
  # cost the most time so far reported 'running' throughout.
  avail="$(free_mb)"
  if [ -n "$avail" ] && [ "$avail" -lt "$DISK_MIN_MB" ]; then
    log "WARNING: only ${avail}MB free - a full disk kills a run with ENOSPC"
  fi

  now="$(date +%s)"
  frame="$(frame_count)"
  if [ -n "$frame" ] && [ "$frame" != "$last_frame" ]; then
    last_frame="$frame"
    last_frame_at="$now"
  elif [ -n "$last_frame" ] && [ "$last_frame_at" -gt 0 ]; then
    stalled=$(( now - last_frame_at ))
    if [ "$stalled" -ge "$STALL_SECONDS" ] && [ "$status" = "running" ]; then
      log "agent has not pressed a button in ${stalled}s while 'running' - restarting the session"
      curl -sS -m 30 -X POST "$SERVER/supervisor/stop" \
        -H 'Content-Type: application/json' -d '{}' >/dev/null 2>&1 || true
      last_frame_at="$now"
      sleep 5
      continue
    fi
  fi

  case "$status" in
    # 'critiquing' is the between-sessions retrospective. It runs at xhigh and
    # can take minutes; starting a session over it would throw the handoff away.
    running|starting|critiquing)
      ;;
    "")
      log "server not answering; waiting"
      ;;
    *)
      reason="$(printf '%s' "$snapshot" | read_field status_reason)"
      log "run is '$status' ($reason) - starting a fresh session"
      # The response carries the goal the new session resolved to. On a restart
      # that names no goal that is the last critique's NEXT GOAL line, so this is
      # the one place a critic that silently stopped running shows up: the goal
      # stops changing between rotations.
      if started="$(curl -sS -m 60 -X POST "$SERVER/supervisor/start" \
        -H 'Content-Type: application/json' -d "$PAYLOAD")"; then
        goal="$(printf '%s' "$started" | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("supervisor", {}).get("goal", ""))
except Exception:
    print("")' 2>/dev/null)"
        log "started${goal:+ - goal: $goal}"
        if [ "$GOAL_STICKY" != "1" ]; then
          # Every later start goes out without a goal, handing the choice to the
          # critic's NEXT GOAL line rather than repeating the opening shove.
          PAYLOAD="$RESTART_PAYLOAD"
        fi
      else
        log "start failed; will retry"
      fi
      ;;
  esac

  sleep "$POLL_SECONDS"
done
