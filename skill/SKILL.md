---
name: pokemon-player
description: Play Pokemon Red through this repo's strict minimal-turn harness. Inspect the attached frames, refresh `/agent/observe`, read `turn_context.json` and `turn_plan.json`, submit a validated plan with `/agent/plan`, then execute one batch with `/agent/act`.
tags: [pokemon, emulator, vision, gameplay, dashboard]
triggers:
  - play pokemon
  - play pokemon red
  - pokemon harness
  - pokemon dashboard
  - pokemon vision
---

# Pokemon Player Skill

Use this repo's local server as the sole harness. Pi is the orchestrator. Do not launch or stop the harness server from inside Pi.

Red is the only first-class target for this skill.

If `http://localhost:8765/health` is unreachable, stop and tell the operator the server is not running.

## Server Assumption

Assume the operator already started the server from the repo root:

```bash
uv run pokemon-agent serve \
  --rom <ROM_PATH> \
  --port 8765 \
  --agent-workspace-dir "$(pwd)/.agent-workspace"
```

The dashboard is at `http://localhost:8765/dashboard`.

If the operator is already using the dashboard supervisor, do not start a second Pi session manually.

## Canonical Turn Artifacts

Treat these files as the only model-facing workspace contract:

- `latest_frame_annotated.png`
- `latest_frame.png`
- `turn_context.json`
- `turn_plan.json`
- `recovery_saves.json` only when `turn_context.json` says the run is risky or stuck

Ignore `.agent-workspace/debug/`. Those files are dashboard and operator support, not part of the Pi contract.

API responses are intentionally concise. Read the workspace files instead of expecting large inline payloads.

## Mandatory Turn Loop

Do this every turn:

1. Refresh the turn context.

```bash
curl -s -X POST http://localhost:8765/agent/observe | python3 -m json.tool
```

2. Read these files in this order:
   - `latest_frame_annotated.png`
   - `latest_frame.png` only if the overlay hides detail
   - `turn_context.json`
   - `turn_plan.json`
   - `recovery_saves.json` only if the context says recovery matters

   `turn_context.json` includes a `planning` section with:
   - the exact `observation_id` and `objective_id` to copy
   - the allowed branch shape for each mode
   - the valid measurable `expected_outcome` fields

3. Submit one strict plan for the current observation.

```bash
curl -s -X POST http://localhost:8765/agent/plan \
  -H "Content-Type: application/json" \
  -d '{
        "observation_id": "<copy from turn_context.json>",
        "objective_id": "<copy from turn_context.json>",
        "intent": "Probe one tile north.",
        "mode": "overworld",
        "primary_branch": {"kind": "raw_actions", "actions": ["walk_up"]},
        "expected_outcome": {
          "summary": "Move one tile north.",
          "position_delta": {"dx": 0, "dy": -1}
        }
      }' | python3 -m json.tool
```

4. Execute exactly one validated batch.

```bash
curl -s -X POST http://localhost:8765/agent/act | python3 -m json.tool
```

5. Re-run `/agent/observe` before planning again.

Never send long blind action chains. Overworld raw batches are capped at 4 actions. Dialog and battle raw batches are capped at 2 actions. Navigation plans are one target per plan.

## What To Trust In `turn_context.json`

Use these fields as the canonical decision surface:

- `objective.id`, `objective.summary`, `objective.completion_predicate`
- `ui.mode`, `ui.screen_text`
- `position.map_name`, `position.x`, `position.y`, `position.facing`
- `navigation.valid_moves`
- `navigation.interaction`
- `navigation.route_hints`
- `navigation.landmarks`
- `recent_action`
- `recovery`
- `constraints`
- `planning`
- `plan_status`

The screenshot read is mandatory.

- Use the attached annotated frame as primary evidence every turn.
- Use the raw frame only when the overlay hides important detail.
- Do not infer terrain from text alone.

## Planning Rules

- The plan's `observation_id` must match the current `turn_context.json`.
- The plan's `objective_id` must match the current objective in `turn_context.json`.
- `mode=overworld`, `dialog`, and `battle` require a `raw_actions` primary branch.
- `mode=navigation` requires a `navigation` primary branch.
- Every plan must include a measurable `expected_outcome`.
- If `plan_status.state` is `drifted`, `invalid`, or `stale`, stop and re-observe before acting again.

## Decision Order

Use this order:

1. If the UI is in dialog, clear dialog first.
2. If the UI is in battle, resolve the battle first.
3. If `recovery.stuck_level` is `warning` or `danger`, follow `recovery.recommended_actions` verbatim before any further exploration.
4. If a walkable destination is more than one tile away, prefer `mode=navigation` with the exact target coord. Navigation handles pathfinding around walls and sprites — raw `walk_*` batches drift the moment any step hits a collision.
5. Use `raw_actions` only for single-step nudges (`walk_*` x1), dialog (`press_a`/`press_b`), or battle menus.
6. Otherwise use the annotated frame, valid moves, and interaction probe to explore carefully.

When `recent_action.plan_state` is `drifted` or `partial`, read `recent_action.summary` — it shows exactly how many of the requested actions executed and where the block happened. Drop your next batch to a single action and reassess, or switch to `mode=navigation`.

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

Save when the context or dashboard clearly indicates a checkpoint or risky segment. Reload when `recovery.stuck_level` is `danger` or `recovery_saves.json` presents a clearly safer candidate.

## Dashboard Use

Keep `/dashboard` open during play.

Use it to verify:

- what frame Pi is reading
- the current turn context and plan status
- the validated plan in `turn_plan.json`
- tool calls, streamed output, and recent events
- whether auto-continue is armed
- whether the harness thinks Pi is stuck
- which recovery save is currently recommended

If the dashboard and `turn_context.json` disagree, refresh `/agent/observe` before acting.

## End Of Session

Before ending a run:

1. Save.
2. Refresh `/agent/observe` once more.
3. Leave `turn_plan.json` in a useful validated or clearly invalid state.
4. Stop Pi from the dashboard if the supervisor is still running.

Server shutdown is operator-owned and outside Pi's responsibilities.
