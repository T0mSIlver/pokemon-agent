"""Prompt policy for the supervised Pi harness."""

from __future__ import annotations

from pathlib import Path

CONTINUE_PROMPT = (
    "Continue the loop: inspect the attached frame(s), refresh /agent/observe, read "
    "turn_context.json and turn_plan.json, use turn_context.json planning guidance to submit one "
    "valid plan with /agent/plan, execute one batch with /agent/act, then inspect the refreshed "
    "context before planning again."
)


def default_supervisor_prompt(*, server_url: str, workspace_dir: Path, goal: str = "") -> str:
    base = (
        "Play Pokemon Red through the local pokemon-agent harness. "
        "Assume the server is already running. "
        f"Server URL: {server_url}. Workspace: {workspace_dir}. "
        "Use the attached annotated PNG as primary evidence and the raw PNG only when needed. "
        "Refresh /agent/observe, read turn_context.json and turn_plan.json, use the planning "
        "section in turn_context.json to submit one valid plan with /agent/plan, execute exactly "
        "one batch with /agent/act, then re-observe."
    )
    goal = goal.strip()
    if goal:
        return f"{base} Mission override: {goal}."
    return base
