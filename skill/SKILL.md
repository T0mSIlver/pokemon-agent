---
name: pokemon-player
description: Play Pokemon Red through this repo's vision-first harness with the server already running. Refresh `/agent/observe`, read the annotated screenshot and observation files every turn, update `turn_plan.json`, then use navigation or raw actions in short batches.
tags: [pokemon, emulator, vision, gameplay, pathfinding, dashboard]
triggers:
  - play pokemon
  - play pokemon red
  - pokemon harness
  - pokemon dashboard
  - pokemon navigation
  - pokemon vision
---

# Pokemon Player Skill

Use this repo's local server as the sole harness. Pi is the orchestrator. The repo owns screenshots, observation files, objectives, checkpoints, recovery, and dashboard telemetry.

Red is the only first-class target for this skill.

This skill is intended to be loaded by the dashboard's Pi supervisor after the operator has already started the harness server in a terminal.

Do not launch a second harness server from inside Pi.
Do not stop the harness server from inside Pi.
If `http://localhost:8765/health` is unreachable, stop and tell the operator that the server is not running.

## Server Assumption

Assume the operator already started the server, typically from the repo root:

```bash
uv run pokemon-agent serve \
  --rom <ROM_PATH> \
  --port 8765 \
  --agent-workspace-dir "$(pwd)/.agent-workspace"
```

Operator-only optional flags:

- `--no-realtime` disables the background clock.
- `--realtime-fps 30` changes the realtime cadence.

Your job is to verify the server, not to launch it.
Check health first:

```bash
curl -s http://localhost:8765/health | python3 -m json.tool
```

Dashboard:

`http://localhost:8765/dashboard`

The operator starts Pi from the dashboard supervisor. The supervisor will stream tool calls, reasoning, prompts, replies, and turn boundaries into `/dashboard`.
The dashboard also shows the live prompt/output transcript, stderr, auto-continue scheduling, manual save controls, and direct load controls for recovery saves.

If the operator is using the dashboard supervisor, do not start a second Pi session manually.

## Workspace Contract

The server writes these files into the agent workspace:

- `latest_frame.png`
- `latest_frame_annotated.png`
- `latest_observation.json`
- `latest_observation.md`
- `current_objective.json`
- `current_objective.md`
- `turn_plan.json`
- `working_memory.md`
- `checkpoints.jsonl`
- `knowledge_graph.json`
- `recovery_saves.json`
- `run_log.jsonl`

Pi must treat these as the canonical turn artifacts.

API responses are intentionally concise. Do not expect screenshots or full raw observations inline in HTTP responses. Read the workspace files instead.

## Mandatory Turn Loop

Do this every turn:

1. Refresh the workspace bundle.

```bash
curl -s -X POST http://localhost:8765/agent/observe \
  -H "Content-Type: application/json" \
  -d '{"reason": "turn_refresh"}' | python3 -m json.tool
```

2. Read these files in this order:
   - `latest_frame_annotated.png`
   - `latest_frame.png` only if the overlay obscures important detail
   - `latest_observation.md`
   - `current_objective.md`
   - `turn_plan.json`
   - `recovery_saves.json` if the run looks stuck or risky

3. Update `turn_plan.json` before every action batch.

Required shape:

```json
{
  "objective_id": "copy from current_objective.json",
  "summary": "one short sentence describing the next batch",
  "planned_actions": ["walk_up", "press_a"],
  "fallback_actions": ["press_b"],
  "notes": "why this batch should work",
  "updated_at": "ISO-8601 timestamp"
}
```

4. Keep batches short.
   - Overworld movement: usually 1-4 actions before re-observing.
   - Dialog/menu/battle: usually 1-2 actions before re-observing.
   - Never send long blind action chains.

5. Re-run `/agent/observe` after each batch.

If the dashboard supervisor auto-continues turns, still treat each turn as a hard stop after you observe, plan, act in a short batch, and summarize the next step via `turn_plan.json`.

The screenshot read is mandatory.

- Do not plan from ASCII alone.
- Do not treat the ASCII map as a screenshot.
- Use ASCII only as a symbolic collision summary after looking at the annotated frame.

## What To Read Every Turn

From `latest_observation.json` and `latest_observation.md`, pay attention to:

- `screen_text`
- `objective.current`
- `recent_action`
- `state_delta`
- `stuck`
- `recovery.current_recommendation`
- `navigation.snapshot.valid_moves`
- `navigation.snapshot.interaction`
- `movement_guidance`
- `navigation.snapshot.ascii`
- `navigation.location_map.ascii`

From `current_objective.json`, trust:

- the current objective id
- completion predicate
- failure hints
- save recommendation
- route hint

Do not invent a different canonical objective.

ASCII semantics:

- `P` = player
- `G` = goal
- `S` = visible sprite blocker
- `.` = known passable tile
- `#` = known blocked tile
- `?` = unknown tile

The ASCII maps use coordinate labels. They do not encode terrain with distance digits.

## Navigation vs Raw Actions

Use navigation when you know the destination coordinates:

```bash
curl -s -X POST http://localhost:8765/navigation/path \
  -H "Content-Type: application/json" \
  -d '{"x": 10, "y": 5, "mode": "auto"}' | python3 -m json.tool
```

```bash
curl -s -X POST http://localhost:8765/navigation/navigate \
  -H "Content-Type: application/json" \
  -d '{"x": 10, "y": 5, "mode": "auto"}' | python3 -m json.tool
```

Prefer raw actions for:

- dialog
- menus
- battles
- one-tile probing into unexplored terrain
- NPC interaction confirmation

Raw actions:

```bash
curl -s -X POST http://localhost:8765/action \
  -H "Content-Type: application/json" \
  -d '{"actions": ["press_a"]}' | python3 -m json.tool
```

```bash
curl -s -X POST http://localhost:8765/action \
  -H "Content-Type: application/json" \
  -d '{"actions": ["walk_up", "press_a"]}' | python3 -m json.tool
```

```bash
curl -s -X POST http://localhost:8765/action \
  -H "Content-Type: application/json" \
  -d '{"actions": ["a_until_dialog_end"]}' | python3 -m json.tool
```

## Decision Order

Use this order:

1. If `dialog.active` is true, clear dialog first.
2. If `battle.in_battle` is true, handle battle first.
3. If `stuck.level` is `warning` or `danger`, fix that before continuing long exploration.
4. If the destination coordinates are known, prefer navigation.
5. If coordinates are unknown, use the annotated frame, interaction probe, and ASCII maps to explore carefully.

If `movement_guidance` gives a route cue like "best explored north-progress route starts with ...", prefer that over trying to visually parse a large ASCII block on your own.

## Saving And Recovery

Manual save:

```bash
curl -s -X POST http://localhost:8765/save \
  -H "Content-Type: application/json" \
  -d '{"name": "before_brock"}' | python3 -m json.tool
```

Load recovery:

```bash
curl -s -X POST http://localhost:8765/load \
  -H "Content-Type: application/json" \
  -d '{"name": "before_brock"}' | python3 -m json.tool
```

Save when:

- `current_objective.md` recommends it
- entering a risky battle
- entering a dungeon/forest section with poor visibility
- the dashboard shows a clean new checkpoint

Reload when:

- `stuck.level` is `danger`
- repeated short batches produce no movement or no progress
- the top candidate in `recovery_saves.json` is clearly safer than the current state

## Working Memory

`working_memory.md` is Pi-editable scratch space.

Use it for:

- local route notes
- battle reminders
- blockers just discovered
- short hypotheses to test next turn

Do not use it to redefine the canonical objective or recovery policy.

## Dashboard Use

Keep `/dashboard` open during play.

Use it to verify:

- what frame Pi is reading
- what objective the server thinks is current
- what `turn_plan.json` says Pi is trying to do
- every prompt sent to Pi and every streamed reply fragment Pi emits
- whether auto-continue is armed and when the next turn is scheduled
- whether the last action batch changed state
- whether the harness thinks Pi is stuck
- which recovery save is currently recommended

If you need a manual checkpoint, use the dashboard `Save Now` button or:

```bash
curl -s -X POST http://localhost:8765/save \
  -H "Content-Type: application/json" \
  -d '{"name":"manual_YYYYMMDD_HHMMSS"}' | python3 -m json.tool
```

If the dashboard intent panel and Pi's next action disagree, stop and refresh observation before acting.

If the operator wants to recover to a specific save, prefer the dashboard `Load Selected` or
`Load Recommended` controls instead of typing save names manually.

## End Of Session

Before ending a run:

1. Save.
2. Refresh `/agent/observe` once more.
3. Leave `turn_plan.json` and `working_memory.md` in a useful state for resume.
4. Stop Pi from the dashboard if the supervisor is still running.

Server shutdown is operator-owned and outside Pi's responsibilities.
