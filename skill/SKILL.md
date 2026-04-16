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
curl -s http://localhost:8765/agent/observe | python3 -m json.tool
```

2. Read these files in this order:
   - Use the `read` tool on `latest_frame_annotated.png` every turn before any `/agent/plan` or `/agent/act` call
   - Use the `read` tool on `latest_frame.png` too if the overlay hides detail, text, or sprite placement
   - `turn_context.json`
   - `turn_plan.json`
   - `recovery_saves.json` only if the context says recovery matters

   `turn_context.json` includes a `planning` section with:
   - the exact `planning.observation_id` and `planning.objective_id` to copy
   - the allowed branch shape for each mode
   - the valid measurable `expected_outcome` fields
   - an `expected_outcome_template` example you can copy and adapt

3. Submit one strict plan for the current observation.

```bash
bash agent_curl.sh /agent/plan <<'JSON' | python3 -m json.tool
{
  "observation_id": "<copy from turn_context.json>",
  "objective_id": "<copy from turn_context.json>",
  "intent": "Probe one tile north.",
  "mode": "overworld",
  "primary_branch": {"kind": "raw_actions", "actions": ["walk_up"]},
  "expected_outcome": {
    "summary": "Move one tile north.",
    "position_delta": {"dx": 0, "dy": -1}
  }
}
JSON
```

4. Execute exactly one validated batch.

```bash
bash agent_curl.sh /agent/act <<'JSON' | python3 -m json.tool
{}
JSON
```

5. Re-run `/agent/observe` before planning again.

Do not exceed the limits exposed in `turn_context.json.planning.mode_rules`.

## What To Trust In `turn_context.json`

Use these fields as the canonical decision surface:

- `objective.id`, `objective.summary`, `objective.completion_predicate`
- `ui.mode`, `ui.screen_text`
- `position.map_name`, `position.x`, `position.y`, `position.facing`
- `navigation.valid_moves`
- `navigation.visible_sprites`
- `navigation.ascii_window`
- `navigation.ascii_legend`
- `navigation.interaction`
- `navigation.route_hints`
- `navigation.landmarks`
- `recent_action`
- `recovery`
- `constraints`
- `planning`
- `plan_status`

The screenshot read is mandatory.

- Use the `read` tool on BOTH `latest_frame_annotated.png` AND `latest_frame.png` every turn before planning.
- The annotated frame shows the navigation grid, warp markers (purple `W` boxes), sprite blockers (orange squares), and the interaction target (orange ring).
- The raw frame shows in-game art, NPC outfits, dialog text, signposts, and details the overlay can hide.
- Treat a turn without both reads as a failed turn that must be corrected on the next step.
- Do not skip image inspection just because `turn_context.json` already exists.
- Do not infer terrain from text alone.

## Using Warps (Doorways and Stairs)

Warps in Pokemon Red are counter-intuitive. Standing next to a warp does nothing — you must:

1. Navigate ONTO the warp tile itself (the `W` glyph in `navigation.ascii_window`, or the purple `W` box in the annotated frame).
2. Take ONE more step in the direction of the exit (north for a top doorway, south for a doormat at the bottom of an interior, west/east for side exits).

The map transition only fires on that follow-up step. If you land on a `W` and stop, you will appear stuck. Always plan a navigation target ONE TILE PAST the warp coordinate when leaving a building, or queue a `walk_<direction>` raw action immediately after the navigation reaches the warp tile.

## Planning Rules

- The plan's `observation_id` must match the current `turn_context.json`.
- The plan's `objective_id` must match `turn_context.json`'s `planning.objective_id`.
- Match the branch shape and limits in `turn_context.json.planning.mode_rules`.
- Prefer copying `turn_context.json.planning.branch_templates` and editing them instead of inventing a shape from scratch.
- If `ui.mode` is `dialog`, raw actions may use `a_until_dialog_end` to clear the current dialog.
- Every plan must include a measurable `expected_outcome`.
- Prefer copying `planning.expected_outcome_template` and editing it instead of inventing shape from scratch.
- If `plan_status.state` is `drifted`, `invalid`, or `stale`, stop and re-observe before acting again.
- Use `bash agent_curl.sh` with a heredoc for JSON POST bodies instead of hand-written shell quoting.

Use the current `turn_context.json` as the source of truth for `ui`, `plan_status`, `navigation`, `recent_action`, and `recovery`. Prefer those live fields over stale assumptions from previous turns.

## Saving And Recovery

Manual save:

```bash
bash agent_curl.sh /save <<'JSON' | python3 -m json.tool
{"name": "before_brock"}
JSON
```

Load recovery:

```bash
bash agent_curl.sh /load <<'JSON' | python3 -m json.tool
{"name": "before_brock"}
JSON
```

Save when the context or dashboard clearly indicates a checkpoint or risky segment. Reload when `recovery.stuck_level` is `danger` or `recovery_saves.json` presents a clearly safer candidate. If `recovery.recovery_command` is present in `turn_context.json`, you can run it directly.

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
