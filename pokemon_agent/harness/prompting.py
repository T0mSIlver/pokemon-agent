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
    "Before any planning or action each turn, use the read tool on latest_frame_annotated.png. "
    "Treat that annotated frame read as mandatory primary evidence for the current turn. "
    "Use the read tool on latest_frame.png too when the overlay hides detail, text, "
    "entrances, signposts, or sprite placement. Do not skip the image read just because "
    "turn_context.json exists."
)

PLANNING_ID_GUIDANCE = (
    "Copy observation_id and objective_id from turn_context.json's planning section. "
    "Do not guess them, reuse stale ids, or pull objective_id from an older turn."
)

JSON_POST_GUIDANCE = (
    "For JSON POSTs, use bash scripts/agent_curl.sh with a heredoc instead of hand-built "
    "curl JSON quoting."
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
    "the refreshed context before planning again. "
    + VISUAL_INSPECTION_GUIDANCE
    + " "
    + PLANNING_ID_GUIDANCE
    + " "
    + JSON_POST_GUIDANCE
    + " "
    + NAVIGATION_MODE_GUIDANCE
    + " "
    + DRIFT_RECOVERY_GUIDANCE
)


def continue_supervisor_prompt(*, vision_violation_reason: str = "") -> str:
    reason = vision_violation_reason.strip()
    if not reason:
        return CONTINUE_PROMPT
    if not reason.endswith("."):
        reason += "."
    return (
        "Previous turn violated the frame-inspection policy: "
        + reason
        + " Before any other tool call this turn, use the read tool on "
        "latest_frame_annotated.png. If latest_frame.png is attached and the overlay hides "
        "detail, text, entrances, signposts, or sprite placement, read that too. Do not call "
        "/agent/plan or /agent/act until the frame read succeeds. "
        + CONTINUE_PROMPT
    )


def default_supervisor_prompt(*, server_url: str, workspace_dir: Path, goal: str = "") -> str:
    base = (
        "Play Pokemon Red through the local pokemon-agent harness. "
        "Assume the server is already running. "
        f"Server URL: {server_url}. Workspace: {workspace_dir}. "
        "Use the attached annotated PNG as primary evidence and the raw PNG only when needed. "
        "Refresh with GET /agent/observe, read turn_context.json and turn_plan.json, use the "
        "planning section in turn_context.json to submit one valid plan with /agent/plan, "
        "execute exactly one batch with /agent/act, then re-observe. "
        + VISUAL_INSPECTION_GUIDANCE
        + " "
        + PLANNING_ID_GUIDANCE
        + " "
        + JSON_POST_GUIDANCE
        + " "
        + NAVIGATION_MODE_GUIDANCE
        + " "
        + DRIFT_RECOVERY_GUIDANCE
    )
    goal = goal.strip()
    if goal:
        return f"{base} Mission override: {goal}."
    return base
