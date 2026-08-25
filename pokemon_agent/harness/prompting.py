"""User-turn prompts for the Pi harness.

Gameplay policy lives in ``skill/SKILL.md``, which Pi receives as its system
prompt. These builders carry only the per-run facts that policy cannot know:
where the server is, where the workspace is, and what the goal is.
"""

from __future__ import annotations

from pathlib import Path

CONTINUE_PROMPT = "continue"

DEFAULT_GOAL = "Play Pokemon Red and make progress on the current objective."


def continue_supervisor_prompt() -> str:
    return CONTINUE_PROMPT


def default_supervisor_prompt(*, server_url: str, workspace_dir: Path, goal: str = "") -> str:
    return "\n".join(
        [
            f"Server: {server_url}",
            f"Workspace: {workspace_dir}",
            f"Goal: {goal.strip() or DEFAULT_GOAL}",
        ]
    )
