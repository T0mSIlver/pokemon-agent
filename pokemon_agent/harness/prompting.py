"""Prompt policy for the supervised Pi harness."""

from __future__ import annotations

from pathlib import Path

VISUAL_INSPECTION_GUIDANCE = (
    "Before any planning or action each turn, use the read tool on BOTH "
    "latest_frame_annotated.png and latest_frame.png. The annotated frame shows the "
    "navigation grid, warp markers (purple W boxes), sprite blockers (orange squares), "
    "and the interaction target (orange ring); the raw frame shows in-game art, NPC "
    "outfits, dialog text, signposts, and details the overlay can hide. Treat both reads "
    "as mandatory primary evidence — turn_context.json never substitutes for the image. "
    "If you skipped a read, do it now before any /agent/plan or /agent/act."
)

WARP_GUIDANCE = (
    "Warps in Pokemon Red are counter-intuitive: standing NEXT TO a warp tile does "
    "nothing. To use a warp, you must (1) navigate ONTO the warp tile itself (the W in "
    "the ASCII window or the purple square in the annotated frame), then (2) take one "
    "more step in the direction of the exit (north for a top doorway, south for a "
    "doormat at the bottom of an interior). The transition only fires on that follow-up "
    "step. If you land on a W and stop, you will appear stuck — keep walking through."
)

PLANNING_ID_GUIDANCE = (
    "Copy observation_id and objective_id from turn_context.json's planning section. "
    "Do not guess them, reuse stale ids, or pull objective_id from an older turn."
)

JSON_POST_GUIDANCE = (
    "For JSON POSTs, use bash agent_curl.sh with a heredoc instead of hand-built curl JSON quoting."
)

DIALOG_GUIDANCE = (
    'When turn_context.json shows ui.mode="dialog", raw_actions may use '
    '"a_until_dialog_end" to clear the active dialog.'
)

TURN_FLOW_GUIDANCE = (
    "After each /agent/act, refresh with GET /agent/observe before submitting another "
    "plan. You may keep making tool calls in the same run; you do not need to stop after "
    "one observe/plan/act cycle."
)

CONTINUE_PROMPT = (
    "Continue playing Pokemon Red through the local pokemon-agent harness. Inspect the "
    "attached frame(s), refresh with GET /agent/observe, read turn_context.json and "
    "turn_plan.json, use turn_context.json planning guidance to submit valid plans with "
    "/agent/plan, and execute them with /agent/act. "
    + VISUAL_INSPECTION_GUIDANCE
    + " "
    + PLANNING_ID_GUIDANCE
    + " "
    + JSON_POST_GUIDANCE
    + " "
    + DIALOG_GUIDANCE
    + " "
    + WARP_GUIDANCE
    + " "
    + TURN_FLOW_GUIDANCE
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
        + " Before any other tool call this turn, use the read tool on BOTH "
        "latest_frame_annotated.png AND latest_frame.png. Do not call /agent/plan or "
        "/agent/act until both frame reads succeed. " + CONTINUE_PROMPT
    )


def default_supervisor_prompt(*, server_url: str, workspace_dir: Path, goal: str = "") -> str:
    base = (
        "Play Pokemon Red through the local pokemon-agent harness. "
        "Assume the server is already running. "
        f"Server URL: {server_url}. Workspace: {workspace_dir}. "
        "Read BOTH latest_frame_annotated.png and latest_frame.png every turn — neither "
        "alone is sufficient. "
        "Refresh with GET /agent/observe, read turn_context.json and turn_plan.json, use the "
        "planning section in turn_context.json to submit valid plans with /agent/plan, and "
        "execute them with /agent/act. "
        + VISUAL_INSPECTION_GUIDANCE
        + " "
        + PLANNING_ID_GUIDANCE
        + " "
        + JSON_POST_GUIDANCE
        + " "
        + DIALOG_GUIDANCE
        + " "
        + WARP_GUIDANCE
        + " "
        + TURN_FLOW_GUIDANCE
    )
    goal = goal.strip()
    if goal:
        return f"{base} Mission override: {goal}."
    return base
