"""The deterministic objective ladder and the packs it is loaded from."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from pokemon_agent.state_analysis import selector_matches

JsonDict = dict[str, Any]


@dataclass(slots=True)
class ObjectiveRecord:
    pack_id: str
    id: str
    summary: str
    completion_predicate: str
    failure_hints: list[str]
    save_recommendation: str
    priority: int
    current: bool
    completed: bool
    status: str

    def to_dict(self) -> JsonDict:
        return asdict(self)


@lru_cache(maxsize=1)
def load_red_objective_packs() -> list[JsonDict]:
    data_dir = Path(__file__).parent / "data"
    packs: list[JsonDict] = []
    for path in sorted(data_dir.glob("red_objectives_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            packs.append(payload)
    return sorted(packs, key=lambda pack: int(pack.get("order", 0)))


class ObjectiveEngine:
    """Deterministic Red-first objective progression across chained packs."""

    def __init__(self) -> None:
        self.packs = load_red_objective_packs()
        self.objectives: list[JsonDict] = []
        for pack in self.packs:
            pack_id = str(pack.get("pack_id") or "unknown_pack")
            for item in pack.get("objectives") or []:
                if not isinstance(item, dict):
                    continue
                merged = dict(item)
                merged["pack_id"] = pack_id
                merged.setdefault("selector", {})
                self.objectives.append(merged)
        self.by_id = {item["id"]: item for item in self.objectives}

    def _current_objective_index(self, state: JsonDict) -> int:
        if not self.objectives:
            return 0
        current_index = 0
        for index, item in enumerate(self.objectives):
            if selector_matches(item.get("selector") or {}, state):
                current_index = index
        return current_index

    def evaluate(self, state: JsonDict) -> JsonDict:
        if not self.objectives:
            empty = ObjectiveRecord(
                pack_id="unknown_pack",
                id="no_objectives_loaded",
                summary="Objective data was not loaded.",
                completion_predicate="N/A",
                failure_hints=[],
                save_recommendation="Manual saves only.",
                priority=1,
                current=True,
                completed=False,
                status="current",
            ).to_dict()
            return {
                "game": "red",
                "current": empty,
                "objectives": [empty],
                "progress_percent": 0,
                "current_pack_id": "unknown_pack",
                "packs": [],
                "phase_complete": False,
            }

        current_index = self._current_objective_index(state)
        current_id = self.objectives[current_index]["id"]
        total_steps = max(len(self.objectives) - 1, 1)
        progress_percent = min(100, int((current_index / total_steps) * 100))
        objectives: list[JsonDict] = []
        current_objective: Optional[JsonDict] = None

        for index, item in enumerate(self.objectives):
            completed = index < current_index
            current = item["id"] == current_id
            record = ObjectiveRecord(
                pack_id=item["pack_id"],
                id=item["id"],
                summary=item["summary"],
                completion_predicate=item["completion_predicate"],
                failure_hints=item.get("failure_hints", []),
                save_recommendation=item.get("save_recommendation", ""),
                priority=index + 1,
                current=current,
                completed=completed,
                status="completed" if completed else "current" if current else "pending",
            ).to_dict()
            objectives.append(record)
            if current:
                current_objective = record

        assert current_objective is not None
        return {
            "game": "red",
            "current": current_objective,
            "objectives": objectives,
            "progress_percent": progress_percent,
            "current_pack_id": current_objective["pack_id"],
            "packs": [
                {"pack_id": pack.get("pack_id"), "order": pack.get("order")} for pack in self.packs
            ],
            "phase_complete": current_id == "phase_complete_cut_access",
        }
