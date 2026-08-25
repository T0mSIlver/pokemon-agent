"""Progress and recovery: trigger auto-saves, and notice a run going nowhere.

Both signals are derived from the observation stream rather than from the
emulator, so :class:`ProgressMonitor` owns the little history they need -- the
last objective seen, how many action turns it has survived, and the recent
position trajectory.
"""

from __future__ import annotations

import json
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

JsonDict = dict[str, Any]


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"


class ProgressMonitor:
    """Auto-save triggers and stuck detection over the observation stream."""

    def __init__(self, *, data_dir: Path, trajectory_limit: int = 60) -> None:
        self.data_dir = data_dir
        self.recent_trajectory: deque[JsonDict] = deque(maxlen=trajectory_limit)
        self.last_objective_id: Optional[str] = None
        self.action_events_since_objective_change = 0

    def maybe_auto_save(
        self,
        *,
        emulator: Any,
        state: JsonDict,
        objective: JsonDict,
        state_delta: JsonDict,
        requested_actions: Optional[list[str]],
        source: str,
    ) -> list[JsonDict]:
        triggers: list[str] = []
        if state_delta.get("fields", {}).get("map"):
            triggers.append("map_transition")
        if objective["current"]["id"] != self.last_objective_id:
            triggers.append("objective_change")
        if source in {"action", "navigation"} and state.get("battle", {}).get("in_battle"):
            triggers.append("battle_entry")

        if not triggers:
            return []

        saves_dir = self.data_dir / "saves"
        saves_dir.mkdir(parents=True, exist_ok=True)
        created: list[JsonDict] = []
        current_map = (state.get("map") or {}).get("map_name", "unknown")
        for trigger in triggers[:2]:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            name = f"auto__{stamp}__{slugify(trigger)}__{slugify(current_map)}"
            path = saves_dir / f"{name}.state"
            if path.exists():
                continue
            emulator.save_state(str(path))
            created.append(
                {
                    "name": name,
                    "path": str(path),
                    "reason": trigger,
                    "source": "auto",
                    "notes": [
                        f"source={source}",
                        f"objective={objective['current']['id']}",
                        f"actions={','.join(requested_actions or []) or 'none'}",
                    ],
                }
            )
        return created

    def detect_stuck(
        self,
        *,
        state: JsonDict,
        objective: JsonDict,
        source: str,
        requested_actions: Optional[list[str]],
    ) -> JsonDict:
        """Detect no-progress loops. Rendered by the operator dashboard only."""
        player = state.get("player") or {}
        signature = {
            "map_name": (state.get("map") or {}).get("map_name"),
            "position": player.get("position") or {},
            "dialog_active": bool(
                state.get("dialog_active") or (state.get("dialog") or {}).get("active")
            ),
            "objective_id": objective["current"]["id"],
            "source": source,
            "actions": requested_actions or [],
        }
        self.recent_trajectory.append(signature)

        recent = list(self.recent_trajectory)[-8:]
        no_movement_loop = False
        dialog_loop = False

        if len(recent) >= 4:
            locations = {
                (item.get("map_name"), json.dumps(item.get("position"), sort_keys=True))
                for item in recent[-4:]
            }
            if len(locations) == 1 and any(item.get("actions") for item in recent[-4:]):
                no_movement_loop = True
            if no_movement_loop and all(item.get("dialog_active") for item in recent[-4:]):
                dialog_loop = True

        if objective["current"]["id"] == self.last_objective_id and source in {
            "action",
            "navigation",
        }:
            self.action_events_since_objective_change += 1
        else:
            self.action_events_since_objective_change = 0

        level = "clear"
        reason = "No stuck pattern detected."
        if dialog_loop:
            level = "warning"
            reason = (
                "Dialog loop detected: repeated actions with the same position and active dialog."
            )
        elif no_movement_loop:
            level = "warning"
            reason = "No-movement loop detected: repeated actions without position or map change."

        if self.action_events_since_objective_change >= 12:
            level = "danger" if level == "warning" else "warning"
            reason = "Current objective has seen many action turns without progress."

        return {
            "level": level,
            "reason": reason,
            "objective_action_count": self.action_events_since_objective_change,
        }

    def note_objective(self, objective_id: Optional[str]) -> None:
        """Remember the objective this observation ended on."""
        self.last_objective_id = objective_id
