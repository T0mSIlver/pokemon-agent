"""Prompt policy for the supervised Pi harness."""

from __future__ import annotations

from pathlib import Path

NAVIGATION_MODE_GUIDANCE = (
    "When navigation.warps or a route_hint names a walkable coord more than one step "
    "away, PREFER mode=\"navigation\" with that target over stacking walk_* actions. "
    "Navigation handles pathfinding around walls and sprites; raw walk_* batches drift "
    "the moment any step hits a collision. Use raw_actions only for same-tile interactions "
    "(press_a, press_b) or single-step nudges."
)

VISUAL_INSPECTION_GUIDANCE = (
    "Before planning each turn, use the read tool on latest_frame_annotated.png. "
    "Use the read tool on latest_frame.png too when the overlay hides detail, text, "
    "or sprite placement. Do not skip the image read just because turn_context.json exists."
)

DRIFT_RECOVERY_GUIDANCE = (
    "If turn_context.recovery.stuck_level is warning/danger or recent_action.plan_state "
    "is drifted, inspect recent_action.summary for the exact block point, drop batch size "
    "to 1 action, and follow any recommended_actions exactly — especially the "
    "'switch to mode=navigation' nudges."
)

CONTINUE_PROMPT = (
    "Continue the loop: inspect the attached frame(s), refresh with GET /agent/observe, "
    "read turn_context.json and turn_plan.json, use turn_context.json planning guidance to "
    "submit one valid plan with /agent/plan, execute one batch with /agent/act, then inspect "
    "the refreshed context before planning again. For JSON POSTs, use "
    "`bash scripts/agent_curl.sh /agent/plan <<'JSON' ... JSON`. "
    + VISUAL_INSPECTION_GUIDANCE
    + " "
    + NAVIGATION_MODE_GUIDANCE
    + " "
    + DRIFT_RECOVERY_GUIDANCE
)


def default_supervisor_prompt(*, server_url: str, workspace_dir: Path, goal: str = "") -> str:
    base = (
        "Play Pokemon Red through the local pokemon-agent harness. "
        "Assume the server is already running. "
        f"Server URL: {server_url}. Workspace: {workspace_dir}. "
        "Use the attached annotated PNG as primary evidence and the raw PNG only when needed. "
        "Refresh with GET /agent/observe, read turn_context.json and turn_plan.json, use the "
        "planning section in turn_context.json to submit one valid plan with /agent/plan, "
        "execute exactly one batch with /agent/act, then re-observe. For JSON POSTs, use "
        "`bash scripts/agent_curl.sh /agent/plan <<'JSON' ... JSON`. "
        + VISUAL_INSPECTION_GUIDANCE
        + " "
        + NAVIGATION_MODE_GUIDANCE
        + " "
        + DRIFT_RECOVERY_GUIDANCE
    )
    goal = goal.strip()
    if goal:
        return f"{base} Mission override: {goal}."
    return base
