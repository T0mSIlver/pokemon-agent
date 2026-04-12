"""Agent workspace, telemetry, and observation runtime for Pokemon Agent."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

from PIL import Image, ImageDraw, ImageFont
from pydantic import ValidationError

from pokemon_agent.harness.context_builder import build_turn_context
from pokemon_agent.harness.contracts import TurnContext, TurnPlan, TurnPlanInput
from pokemon_agent.harness.planning import (
    build_plan_execution_trace,
    default_turn_plan,
    evaluate_plan_outcome,
    invalidate_plan,
    mark_plan_executed,
    parse_stored_turn_plan,
    store_validated_plan,
    validate_turn_plan_submission,
)
from pokemon_agent.memory.red import MAP_NAMES as RED_MAP_NAMES
from pokemon_agent.navigation import LiveNavigationSnapshot, NavigationStore

JsonDict = dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_STUCK_LEVEL_ORDER = {"clear": 0, "warning": 1, "danger": 2}


def _stuck_level_rank(level: str) -> int:
    return _STUCK_LEVEL_ORDER.get(level, 0)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(suffix=path.suffix, dir=path.parent)
    try:
        os.write(fd, data)
        os.close(fd)
        fd = -1
        os.replace(tmp, str(path))
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    _atomic_write_bytes(path, text.encode(encoding))


def _truncate_text_block(value: Any, limit: int = 1600) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 16].rstrip() + "\n...[truncated]..."


@dataclass(slots=True)
class ObjectiveRecord:
    pack_id: str
    id: str
    title: str
    summary: str
    completion_predicate: str
    failure_hints: list[str]
    save_recommendation: str
    route_hint: str
    preferred_landmark_types: list[str]
    priority: int
    current: bool
    completed: bool
    status: str
    progress_percent: int

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class CheckpointRecord:
    id: str
    title: str
    summary: str
    objective_id: str
    map_name: str
    created_at: str
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class RecoveryCandidate:
    name: str
    path: str
    reason: str
    score: int
    modified_at: str
    source: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class LandmarkRecord:
    id: str
    map_id: int
    map_name: str
    kind: str
    title: str
    coord: JsonDict
    discovered_at: str
    last_seen_at: str
    seen_count: int = 1
    source: str = "runtime"
    notes: list[str] = field(default_factory=list)

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


def _coord_to_key(coord: tuple[int, int]) -> str:
    return f"{coord[0]},{coord[1]}"


def _tuple_coord_from_any(value: Any) -> Optional[tuple[int, int]]:
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except Exception:  # noqa: BLE001
            return None
    if isinstance(value, dict) and value.get("x") is not None and value.get("y") is not None:
        try:
            return int(value["x"]), int(value["y"])
        except Exception:  # noqa: BLE001
            return None
    return None


def _coord_payload(coord: Optional[tuple[int, int]]) -> Optional[JsonDict]:
    if coord is None:
        return None
    return {"x": coord[0], "y": coord[1]}


def _manhattan(a: Optional[tuple[int, int]], b: Optional[tuple[int, int]]) -> int:
    if a is None or b is None:
        return 999999
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _stable_id(*parts: Any) -> str:
    joined = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]
    return digest


def _bag_item_counts(state: Optional[JsonDict]) -> dict[str, int]:
    bag = (state or {}).get("bag") or []
    counts: dict[str, int] = {}
    for entry in bag:
        item = str(entry.get("item") or "").strip()
        if not item:
            continue
        counts[item] = int(entry.get("quantity") or 0)
    return counts


def _bag_item_names(state: Optional[JsonDict]) -> set[str]:
    return set(_bag_item_counts(state))


def _badge_count(state: JsonDict) -> int:
    player = state.get("player") or {}
    flags = state.get("flags") or {}
    return int(player.get("badge_count", flags.get("badge_count", 0)) or 0)


def _selector_matches(selector: JsonDict, state: JsonDict) -> bool:
    if not selector:
        return True

    map_name = str((state.get("map") or {}).get("map_name") or "")
    bag_items = _bag_item_names(state)
    battle_active = bool((state.get("battle") or {}).get("in_battle"))
    flags = state.get("flags") or {}

    map_in = selector.get("map_in")
    if map_in and map_name not in set(map_in):
        return False

    map_not_in = selector.get("map_not_in")
    if map_not_in and map_name in set(map_not_in):
        return False

    if selector.get("has_pokedex") is not None:
        if bool(flags.get("has_pokedex")) is not bool(selector.get("has_pokedex")):
            return False

    if selector.get("has_oaks_parcel") is not None:
        if bool(flags.get("has_oaks_parcel")) is not bool(selector.get("has_oaks_parcel")):
            return False

    badge_count = _badge_count(state)
    if selector.get("badge_count_gte") is not None and badge_count < int(
        selector["badge_count_gte"]
    ):
        return False
    if selector.get("badge_count_lte") is not None and badge_count > int(
        selector["badge_count_lte"]
    ):
        return False
    if selector.get("badge_count_lt") is not None and badge_count >= int(
        selector["badge_count_lt"]
    ):
        return False

    if selector.get("battle_active") is not None and battle_active is not bool(
        selector.get("battle_active")
    ):
        return False

    bag_has_any = selector.get("bag_has_any") or []
    if bag_has_any and not any(item in bag_items for item in bag_has_any):
        return False

    bag_has_all = selector.get("bag_has_all") or []
    if bag_has_all and not all(item in bag_items for item in bag_has_all):
        return False

    bag_missing_all = selector.get("bag_missing_all") or []
    if bag_missing_all and any(item in bag_items for item in bag_missing_all):
        return False

    return True


TYPE_EFFECTIVENESS: dict[str, dict[str, float]] = {
    "Normal": {"Rock": 0.5, "Ghost": 0.0},
    "Fire": {"Grass": 2.0, "Bug": 2.0, "Ice": 2.0, "Water": 0.5, "Rock": 0.5, "Fire": 0.5, "Dragon": 0.5},
    "Water": {"Fire": 2.0, "Ground": 2.0, "Rock": 2.0, "Water": 0.5, "Grass": 0.5, "Dragon": 0.5},
    "Grass": {"Water": 2.0, "Ground": 2.0, "Rock": 2.0, "Fire": 0.5, "Grass": 0.5, "Poison": 0.5, "Flying": 0.5, "Bug": 0.5, "Dragon": 0.5},
    "Electric": {"Water": 2.0, "Flying": 2.0, "Electric": 0.5, "Grass": 0.5, "Dragon": 0.5, "Ground": 0.0},
    "Ice": {"Grass": 2.0, "Ground": 2.0, "Flying": 2.0, "Dragon": 2.0, "Water": 0.5, "Ice": 0.5},
    "Fighting": {"Normal": 2.0, "Rock": 2.0, "Ice": 2.0, "Poison": 0.5, "Flying": 0.5, "Psychic": 0.5, "Bug": 0.5, "Ghost": 0.0},
    "Poison": {"Grass": 2.0, "Bug": 2.0, "Poison": 0.5, "Ground": 0.5, "Rock": 0.5, "Ghost": 0.5},
    "Ground": {"Fire": 2.0, "Electric": 2.0, "Poison": 2.0, "Rock": 2.0, "Grass": 0.5, "Bug": 0.5, "Flying": 0.0},
    "Flying": {"Grass": 2.0, "Fighting": 2.0, "Bug": 2.0, "Electric": 0.5, "Rock": 0.5},
    "Psychic": {"Fighting": 2.0, "Poison": 2.0, "Psychic": 0.5},
    "Bug": {"Grass": 2.0, "Poison": 2.0, "Psychic": 2.0, "Fire": 0.5, "Fighting": 0.5, "Flying": 0.5, "Ghost": 0.5},
    "Rock": {"Fire": 2.0, "Ice": 2.0, "Flying": 2.0, "Bug": 2.0, "Fighting": 0.5, "Ground": 0.5},
}


MOVE_METADATA: dict[str, JsonDict] = {
    "Tackle": {"type": "Normal", "power": 35},
    "Scratch": {"type": "Normal", "power": 40},
    "Pound": {"type": "Normal", "power": 40},
    "Quick Attack": {"type": "Normal", "power": 40},
    "Cut": {"type": "Normal", "power": 50},
    "Gust": {"type": "Flying", "power": 40},
    "Wing Attack": {"type": "Flying", "power": 35},
    "Peck": {"type": "Flying", "power": 35},
    "Karate Chop": {"type": "Normal", "power": 50},
    "Low Kick": {"type": "Fighting", "power": 50},
    "Double Kick": {"type": "Fighting", "power": 60},
    "Bite": {"type": "Normal", "power": 60},
    "Vine Whip": {"type": "Grass", "power": 45},
    "Razor Leaf": {"type": "Grass", "power": 55},
    "Absorb": {"type": "Grass", "power": 20},
    "Mega Drain": {"type": "Grass", "power": 40},
    "Ember": {"type": "Fire", "power": 40},
    "Flamethrower": {"type": "Fire", "power": 95},
    "Bubble": {"type": "Water", "power": 20},
    "BubbleBeam": {"type": "Water", "power": 65},
    "Water Gun": {"type": "Water", "power": 40},
    "Surf": {"type": "Water", "power": 95},
    "ThunderShock": {"type": "Electric", "power": 40},
    "Thunderbolt": {"type": "Electric", "power": 95},
    "Shock Wave": {"type": "Electric", "power": 60},
    "Confusion": {"type": "Psychic", "power": 50},
    "Psybeam": {"type": "Psychic", "power": 65},
    "Rock Throw": {"type": "Rock", "power": 50},
    "Seismic Toss": {"type": "Fighting", "power": 50},
    "Dig": {"type": "Ground", "power": 100},
    "Earthquake": {"type": "Ground", "power": 100},
    "Strength": {"type": "Normal", "power": 80},
    "Growl": {"type": "Normal", "power": 0, "status": True},
    "Leer": {"type": "Normal", "power": 0, "status": True},
    "Tail Whip": {"type": "Normal", "power": 0, "status": True},
    "PoisonPowder": {"type": "Poison", "power": 0, "status": True},
    "Sleep Powder": {"type": "Grass", "power": 0, "status": True},
    "String Shot": {"type": "Bug", "power": 0, "status": True},
    "Sand Attack": {"type": "Ground", "power": 0, "status": True},
    "Harden": {"type": "Normal", "power": 0, "status": True},
    "Defense Curl": {"type": "Normal", "power": 0, "status": True},
}


def summarize_party(party: list[JsonDict]) -> str:
    if not party:
        return "No Pokemon in party."
    summary = []
    for mon in party[:3]:
        summary.append(
            f"{mon.get('nickname') or mon.get('species', 'Unknown')} "
            f"Lv{mon.get('level', '?')} "
            f"HP {mon.get('hp', '?')}/{mon.get('max_hp', '?')}"
        )
    remainder = len(party) - len(summary)
    if remainder > 0:
        summary.append(f"+{remainder} more")
    return "; ".join(summary)


def extract_key_state(state: Optional[JsonDict]) -> JsonDict:
    if not state:
        return {}
    player = state.get("player") or {}
    flags = state.get("flags") or {}
    battle = state.get("battle") or {}
    dialog = state.get("dialog") or {}
    map_info = state.get("map") or {}
    party = state.get("party") or []
    return {
        "map_name": map_info.get("map_name"),
        "map_id": map_info.get("map_id"),
        "position": player.get("position") or {},
        "facing": player.get("facing"),
        "badge_count": player.get("badge_count", flags.get("badge_count", 0)),
        "money": player.get("money"),
        "dialog_active": bool(state.get("dialog_active") or dialog.get("active")),
        "battle_active": bool(battle.get("in_battle")),
        "battle_type": battle.get("type"),
        "has_pokedex": bool(flags.get("has_pokedex")),
        "has_oaks_parcel": bool(flags.get("has_oaks_parcel")),
        "party_summary": [
            {
                "name": mon.get("nickname") or mon.get("species"),
                "hp": mon.get("hp"),
                "max_hp": mon.get("max_hp"),
                "status": mon.get("status"),
                "level": mon.get("level"),
            }
            for mon in party
        ],
    }


def _preferred_direction_hint(text: str) -> Optional[str]:
    lowered = text.lower()
    for direction, words in (
        ("north", ("north", "up")),
        ("south", ("south", "down")),
        ("west", ("west", "left")),
        ("east", ("east", "right")),
    ):
        if any(word in lowered for word in words):
            return direction
    return None


def _progress_amount(direction: str, start: tuple[int, int], coord: tuple[int, int]) -> int:
    if direction == "north":
        return start[1] - coord[1]
    if direction == "south":
        return coord[1] - start[1]
    if direction == "west":
        return start[0] - coord[0]
    if direction == "east":
        return coord[0] - start[0]
    return 0


def build_movement_guidance(
    *,
    state: JsonDict,
    snapshot: Optional[LiveNavigationSnapshot],
    navigation_store: Optional[NavigationStore],
    objective: JsonDict,
) -> JsonDict:
    if snapshot is None:
        return {
            "summary": "Navigation guidance unavailable because no live snapshot was captured.",
            "notes": [],
            "preferred_direction": None,
            "candidate_route": None,
        }

    current = snapshot.player_position
    notes: list[str] = [
        (
            "Use the annotated screenshot as primary evidence. "
            "Use ASCII only as a symbolic collision summary."
        ),
        f"Immediate legal moves: {', '.join(snapshot.valid_moves) or 'none'}.",
    ]
    interaction = snapshot.interaction or {}
    target = interaction.get("target_coord") or {}
    if interaction.get("source") == "blocked_tile":
        notes.append(f"Forward movement is blocked at ({target.get('x')}, {target.get('y')}).")

    preferred_direction = _preferred_direction_hint(objective["current"].get("route_hint", ""))
    if preferred_direction is None:
        preferred_direction = {
            "up": "north",
            "down": "south",
            "left": "west",
            "right": "east",
        }.get(snapshot.facing)

    candidate_route = None
    if navigation_store is not None and preferred_direction is not None:
        location_map = navigation_store.get(snapshot.key)
        if location_map is not None:
            distances = location_map.distance_map(
                current,
                extra_blockers=snapshot.sprite_set,
            )
            candidates = [
                coord
                for coord in distances
                if coord != current and _progress_amount(preferred_direction, current, coord) > 0
            ]
            if candidates:
                best = min(
                    candidates,
                    key=lambda coord: (
                        distances[coord],
                        abs(coord[0] - current[0]),
                        abs(coord[1] - current[1]),
                        coord[1],
                        coord[0],
                    ),
                )
                plan = location_map.plan_route(
                    start=current,
                    goal=best,
                    extra_blockers=snapshot.sprite_set,
                    allow_partial=False,
                )
                candidate_route = {
                    "direction": preferred_direction,
                    "target": {"x": best[0], "y": best[1]},
                    "actions": plan.actions,
                    "steps": len(plan.actions),
                }
                if plan.actions:
                    notes.append(
                        f"Best explored {preferred_direction}-progress route starts with: "
                        f"{', '.join(plan.actions[:4])}."
                    )
                notes.append(
                    f"Nearest explored tile that makes {preferred_direction} progress is "
                    f"({best[0]}, {best[1]})."
                )

    if interaction.get("source") == "blocked_tile" and snapshot.valid_moves:
        sidesteps = [move for move in snapshot.valid_moves if move != snapshot.facing]
        if sidesteps:
            notes.append(f"Because forward is blocked, sidestep first: {', '.join(sidesteps)}.")

    summary = notes[-1] if notes else "Navigation guidance unavailable."
    return {
        "summary": summary,
        "notes": notes,
        "preferred_direction": preferred_direction,
        "candidate_route": candidate_route,
    }


def build_state_delta(before: Optional[JsonDict], after: JsonDict) -> JsonDict:
    current = extract_key_state(after)
    previous = extract_key_state(before)
    if not previous:
        return {
            "changed": True,
            "summary": ["Initial observation snapshot captured."],
            "fields": {"initial": current},
            "movement": None,
        }

    fields: JsonDict = {}
    summary: list[str] = []

    before_map = previous.get("map_name")
    after_map = current.get("map_name")
    if before_map != after_map:
        fields["map"] = {"before": before_map, "after": after_map}
        summary.append(f"Map changed from {before_map or 'unknown'} to {after_map or 'unknown'}.")

    before_pos = previous.get("position") or {}
    after_pos = current.get("position") or {}
    movement = None
    if before_pos != after_pos:
        movement = {
            "before": before_pos,
            "after": after_pos,
            "dx": (after_pos.get("x") or 0) - (before_pos.get("x") or 0),
            "dy": (after_pos.get("y") or 0) - (before_pos.get("y") or 0),
        }
        movement["manhattan"] = abs(movement["dx"]) + abs(movement["dy"])
        fields["position"] = movement
        summary.append(
            "Player position changed "
            f"from ({before_pos.get('x')}, {before_pos.get('y')}) "
            f"to ({after_pos.get('x')}, {after_pos.get('y')})."
        )

    for key, label in (
        ("dialog_active", "Dialog"),
        ("battle_active", "Battle"),
        ("has_pokedex", "Pokedex"),
        ("has_oaks_parcel", "Oak's Parcel"),
    ):
        if previous.get(key) != current.get(key):
            fields[key] = {"before": previous.get(key), "after": current.get(key)}
            state = "enabled" if current.get(key) else "disabled"
            summary.append(f"{label} is now {state}.")

    if previous.get("badge_count") != current.get("badge_count"):
        fields["badge_count"] = {
            "before": previous.get("badge_count"),
            "after": current.get("badge_count"),
        }
        summary.append(
            f"Badge count changed from {previous.get('badge_count', 0)} to "
            f"{current.get('badge_count', 0)}."
        )

    before_bag = _bag_item_counts(before)
    after_bag = _bag_item_counts(after)
    bag_changes: list[str] = []
    for item in sorted(set(before_bag) | set(after_bag)):
        previous_qty = before_bag.get(item, 0)
        current_qty = after_bag.get(item, 0)
        if previous_qty == current_qty:
            continue
        if previous_qty == 0 and current_qty > 0:
            bag_changes.append(f"{item} was added to the bag (x{current_qty}).")
        elif current_qty == 0:
            bag_changes.append(f"{item} was removed from the bag.")
        else:
            bag_changes.append(f"{item} quantity changed from {previous_qty} to {current_qty}.")
    if bag_changes:
        fields["bag"] = bag_changes
        summary.extend(bag_changes[:3])

    party_changes: list[str] = []
    before_party = {entry.get("name"): entry for entry in previous.get("party_summary", [])}
    for entry in current.get("party_summary", []):
        name = entry.get("name")
        prior = before_party.get(name)
        if not prior:
            party_changes.append(f"{name} joined the party.")
            continue
        if prior.get("hp") != entry.get("hp"):
            party_changes.append(f"{name} HP changed from {prior.get('hp')} to {entry.get('hp')}.")
        if prior.get("status") != entry.get("status"):
            party_changes.append(
                f"{name} status changed from {prior.get('status')} to {entry.get('status')}."
            )
    if party_changes:
        fields["party"] = party_changes
        summary.extend(party_changes[:3])

    changed = bool(fields)
    if not summary:
        summary.append("No observable structured state change.")
    return {
        "changed": changed,
        "summary": summary,
        "fields": fields,
        "movement": movement,
    }


def classify_action_feedback(
    *,
    source: str,
    requested_actions: Optional[list[str]],
    state_before: Optional[JsonDict],
    state_after: JsonDict,
    state_delta: JsonDict,
    navigation_plan: Optional[JsonDict] = None,
    navigation_execution: Optional[JsonDict] = None,
    plan_execution_trace: Optional[str] = None,
    plan_state: Optional[str] = None,
) -> JsonDict:
    requested_actions = requested_actions or []
    tags: list[str] = []
    notes: list[str] = []

    if state_delta.get("fields", {}).get("map"):
        tags.append("map_transition")
        notes.append("Entered a different map.")
    if state_delta.get("movement"):
        tags.append("movement")
        notes.append("Player position changed.")
    if state_delta.get("fields", {}).get("dialog_active"):
        tags.append("dialog_state_change")
        notes.append("Dialog state changed.")
    if state_delta.get("fields", {}).get("battle_active"):
        if (state_before or {}).get("battle", {}).get("in_battle"):
            tags.append("battle_ended")
        elif state_after.get("battle", {}).get("in_battle"):
            tags.append("battle_started")
        notes.append("Battle state changed.")
    if state_delta.get("fields", {}).get("badge_count"):
        tags.append("milestone")
        notes.append("A badge milestone changed.")
    if not state_delta.get("changed") and requested_actions:
        tags.append("no_progress")
        notes.append("Structured state did not change after the requested actions.")

    if source == "navigation" and navigation_execution:
        if navigation_execution.get("success"):
            tags.append("navigation_success")
        else:
            tags.append("navigation_partial")
        notes.append(navigation_execution.get("status", "Navigation result recorded."))
    elif source == "action" and requested_actions:
        notes.append(f"Executed {len(requested_actions)} raw actions.")
    elif source == "observe":
        notes.append("Fresh observation generated for Pi.")

    if plan_state:
        tags.append(f"plan_{plan_state}")

    if plan_execution_trace:
        summary = plan_execution_trace
        notes = [plan_execution_trace] + [
            note for note in notes if note != "Player position changed."
        ]
    else:
        summary = notes[0] if notes else ""

    if not tags:
        tags.append("observe")

    return {
        "source": source,
        "requested_actions": requested_actions,
        "summary": summary,
        "notes": notes,
        "tags": tags,
        "plan_state": plan_state,
        "plan_execution_trace": plan_execution_trace,
        "navigation_plan": navigation_plan,
        "navigation_execution": navigation_execution,
    }


def classify_ui_mode(state: JsonDict) -> str:
    battle = state.get("battle") or {}
    dialog = state.get("dialog") or {}
    if battle.get("in_battle"):
        return "battle"
    if dialog.get("active") or state.get("dialog_active"):
        return "dialog"
    return "overworld"


def extract_screen_text(image: Image.Image, state: JsonDict) -> JsonDict:
    dialog = state.get("dialog") or {}
    dialog_active = bool(state.get("dialog_active") or dialog.get("active"))
    text = ""
    source = "none"
    note = ""

    crop = image
    if dialog_active:
        crop = image.crop((0, int(image.height * 0.56), image.width, image.height))

    try:
        import pytesseract  # type: ignore[import-not-found]

        scaled = crop.resize((crop.width * 3, crop.height * 3))
        bw = scaled.convert("L").point(lambda px: 255 if px > 120 else 0)
        raw = pytesseract.image_to_string(
            bw,
            config="--psm 6",
        )
        text = re.sub(r"\s+", " ", raw or "").strip()
        if text:
            source = "ocr_dialog" if dialog_active else "ocr_frame"
    except Exception as exc:  # noqa: BLE001
        note = f"OCR unavailable: {type(exc).__name__}"

    if not text and dialog_active:
        waiting = "waiting for input" if dialog.get("waiting_for_input") else "printing"
        text = f"Dialog box visible; OCR text unavailable ({waiting})."
        source = "dialog_state"
    elif not text:
        text = "No readable screen text extracted."
        source = "none"

    return {
        "text": text,
        "source": source,
        "note": note,
        "dialog_active": dialog_active,
        "ui_mode": classify_ui_mode(state),
    }


def render_navigation_overlay(
    image: Image.Image,
    snapshot: Optional[LiveNavigationSnapshot],
    *,
    objective: Optional[JsonDict] = None,
    goal: Optional[tuple[int, int]] = None,
) -> Image.Image:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    if not snapshot:
        draw.rectangle((0, 0, canvas.width, 22), fill=(12, 17, 26, 200))
        draw.text((6, 6), "Navigation overlay unavailable", fill=(255, 255, 255, 255), font=font)
        return Image.alpha_composite(canvas, overlay).convert("RGB")

    tile_width = canvas.width / 10
    tile_height = canvas.height / 9

    for local_y, row in enumerate(snapshot.terrain):
        for local_x, tile in enumerate(row):
            left = int(local_x * tile_width)
            top = int(local_y * tile_height)
            right = int((local_x + 1) * tile_width)
            bottom = int((local_y + 1) * tile_height)
            absolute = snapshot.local_to_absolute(local_x, local_y)
            fill = (24, 123, 73, 64) if tile else (180, 58, 58, 80)
            outline = (110, 230, 174, 180) if tile else (255, 120, 120, 180)
            draw.rectangle((left, top, right, bottom), outline=outline, fill=fill, width=1)
            coord_label = f"{absolute[0]},{absolute[1]}"
            draw.text((left + 2, top + 1), coord_label, fill=(255, 255, 255, 220), font=font)

    for sprite_x, sprite_y in snapshot.sprite_positions:
        local = snapshot.absolute_to_local(sprite_x, sprite_y)
        if local is None:
            continue
        left = int(local[0] * tile_width)
        top = int(local[1] * tile_height)
        right = int((local[0] + 1) * tile_width)
        bottom = int((local[1] + 1) * tile_height)
        draw.rectangle((left + 3, top + 3, right - 3, bottom - 3), fill=(255, 174, 66, 180))

    player_left = int(4 * tile_width)
    player_top = int(4 * tile_height)
    player_right = int(5 * tile_width)
    player_bottom = int(5 * tile_height)
    draw.rectangle(
        (player_left + 1, player_top + 1, player_right - 1, player_bottom - 1),
        outline=(55, 208, 255, 255),
        width=3,
    )
    draw.text((player_left + 4, player_top + 5), "P", fill=(55, 208, 255, 255), font=font)

    if goal is not None:
        goal_local = snapshot.absolute_to_local(goal[0], goal[1])
        if goal_local is not None:
            left = int(goal_local[0] * tile_width)
            top = int(goal_local[1] * tile_height)
            right = int((goal_local[0] + 1) * tile_width)
            bottom = int((goal_local[1] + 1) * tile_height)
            draw.rectangle(
                (left + 1, top + 1, right - 1, bottom - 1), outline=(255, 214, 10, 255), width=3
            )
            draw.text((left + 4, top + 5), "G", fill=(255, 214, 10, 255), font=font)

    interaction = snapshot.interaction or {}
    target_coord = interaction.get("target_coord") or {}
    if target_coord.get("x") is not None and target_coord.get("y") is not None:
        local = snapshot.absolute_to_local(int(target_coord["x"]), int(target_coord["y"]))
        if local is not None:
            left = int(local[0] * tile_width)
            top = int(local[1] * tile_height)
            right = int((local[0] + 1) * tile_width)
            bottom = int((local[1] + 1) * tile_height)
            draw.ellipse(
                (left + 4, top + 4, right - 4, bottom - 4), outline=(255, 125, 0, 255), width=3
            )

    legend_height = 38
    draw.rectangle((0, 0, canvas.width, legend_height), fill=(12, 17, 26, 210))
    title = snapshot.map_name
    pos = snapshot.player_position
    subtitle = (
        f"({pos[0]}, {pos[1]}) facing {snapshot.facing} | "
        f"moves: {', '.join(snapshot.valid_moves) or 'none'}"
    )
    objective_line = objective["title"] if objective else "No objective"
    draw.text((6, 4), title, fill=(255, 255, 255, 255), font=font)
    draw.text((6, 16), subtitle, fill=(165, 180, 196, 255), font=font)
    draw.text((6, 28), f"Objective: {objective_line}", fill=(255, 214, 10, 255), font=font)

    return Image.alpha_composite(canvas, overlay).convert("RGB")


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
                merged.setdefault("preferred_landmark_types", [])
                merged.setdefault("selector", {})
                self.objectives.append(merged)
        self.by_id = {item["id"]: item for item in self.objectives}

    def _current_objective_index(self, state: JsonDict) -> int:
        if not self.objectives:
            return 0
        current_index = 0
        for index, item in enumerate(self.objectives):
            if _selector_matches(item.get("selector") or {}, state):
                current_index = index
        return current_index

    def evaluate(self, state: JsonDict) -> JsonDict:
        if not self.objectives:
            empty = ObjectiveRecord(
                pack_id="unknown_pack",
                id="no_objectives_loaded",
                title="No objectives loaded",
                summary="Objective data was not loaded.",
                completion_predicate="N/A",
                failure_hints=[],
                save_recommendation="Manual saves only.",
                route_hint="Inspect objective data loading.",
                preferred_landmark_types=[],
                priority=1,
                current=True,
                completed=False,
                status="current",
                progress_percent=0,
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
            status = "completed" if completed else "current" if current else "pending"
            progress = (
                100
                if completed
                else progress_percent
                if current
                else int((index / total_steps) * 100)
            )
            record = ObjectiveRecord(
                pack_id=item["pack_id"],
                id=item["id"],
                title=item["title"],
                summary=item["summary"],
                completion_predicate=item["completion_predicate"],
                failure_hints=item.get("failure_hints", []),
                save_recommendation=item.get("save_recommendation", ""),
                route_hint=item.get("route_hint", ""),
                preferred_landmark_types=item.get("preferred_landmark_types", []),
                priority=index + 1,
                current=current,
                completed=completed,
                status=status,
                progress_percent=progress,
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
                {
                    "pack_id": pack.get("pack_id"),
                    "order": pack.get("order"),
                    "title": pack.get("title"),
                }
                for pack in self.packs
            ],
            "phase_complete": current_id == "phase_complete_cut_access",
        }


class AgentRuntime:
    """Owns workspace artifacts, telemetry history, and deterministic assist logic."""

    def __init__(
        self,
        *,
        data_dir: Path,
        workspace_dir: Path,
        objective_engine: Optional[ObjectiveEngine] = None,
        history_limit: int = 400,
    ) -> None:
        self.data_dir = data_dir.expanduser().resolve()
        self.workspace_dir = workspace_dir.expanduser().resolve()
        self.objective_engine = objective_engine or ObjectiveEngine()
        self.history_limit = history_limit
        self.event_history: deque[JsonDict] = deque(maxlen=history_limit)
        self.recent_trajectory: deque[JsonDict] = deque(maxlen=60)
        self.latest_bundle: Optional[JsonDict] = None
        self.live_bundle: Optional[JsonDict] = None
        self.last_state: Optional[JsonDict] = None
        self.last_objective_id: Optional[str] = None
        self.last_turn_plan_hash: Optional[str] = None
        self.checkpoint_ids: set[str] = set()
        self.action_events_since_objective_change = 0
        self.semantic_memory: deque[JsonDict] = deque(maxlen=600)
        self.landmarks_by_id: dict[str, JsonDict] = {}
        self.failure_memory: dict[str, dict[str, JsonDict]] = defaultdict(dict)
        self.dialog_transcript_recent: deque[JsonDict] = deque(maxlen=12)
        self.last_dialog_text = ""
        self.dialog_last_change_at: Optional[str] = None
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir = self.workspace_dir / "debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_workspace_artifacts()
        self._ensure_workspace_files()
        self._load_existing_checkpoint_ids()
        self._load_existing_landmarks()
        self._load_existing_event_memory()

    @property
    def artifacts(self) -> dict[str, Path]:
        return {
            "latest_frame": self.workspace_dir / "latest_frame.png",
            "latest_frame_annotated": self.workspace_dir / "latest_frame_annotated.png",
            "turn_context_json": self.workspace_dir / "turn_context.json",
            "turn_plan_json": self.workspace_dir / "turn_plan.json",
            "recovery_saves_json": self.workspace_dir / "recovery_saves.json",
            "latest_observation_json": self.debug_dir / "latest_observation.json",
            "latest_observation_md": self.debug_dir / "latest_observation.md",
            "current_objective_json": self.debug_dir / "current_objective.json",
            "current_objective_md": self.debug_dir / "current_objective.md",
            "working_memory_md": self.debug_dir / "working_memory.md",
            "checkpoints_jsonl": self.debug_dir / "checkpoints.jsonl",
            "knowledge_graph_json": self.debug_dir / "knowledge_graph.json",
            "landmarks_json": self.debug_dir / "landmarks.json",
            "event_memory_jsonl": self.debug_dir / "event_memory.jsonl",
            "session_brief_md": self.debug_dir / "session_brief.md",
            "run_log_jsonl": self.debug_dir / "run_log.jsonl",
        }

    def _migrate_workspace_artifacts(self) -> None:
        legacy_paths = {
            "latest_observation_json": self.workspace_dir / "latest_observation.json",
            "latest_observation_md": self.workspace_dir / "latest_observation.md",
            "current_objective_json": self.workspace_dir / "current_objective.json",
            "current_objective_md": self.workspace_dir / "current_objective.md",
            "working_memory_md": self.workspace_dir / "working_memory.md",
            "checkpoints_jsonl": self.workspace_dir / "checkpoints.jsonl",
            "knowledge_graph_json": self.workspace_dir / "knowledge_graph.json",
            "landmarks_json": self.workspace_dir / "landmarks.json",
            "event_memory_jsonl": self.workspace_dir / "event_memory.jsonl",
            "session_brief_md": self.workspace_dir / "session_brief.md",
            "run_log_jsonl": self.workspace_dir / "run_log.jsonl",
        }
        for key, legacy_path in legacy_paths.items():
            if not legacy_path.exists():
                continue
            target_path = self.artifacts[key]
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists():
                legacy_path.unlink()
                continue
            legacy_path.replace(target_path)

    def _ensure_workspace_files(self) -> None:
        for path in (
            self.artifacts["checkpoints_jsonl"],
            self.artifacts["event_memory_jsonl"],
            self.artifacts["run_log_jsonl"],
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)

        defaults: dict[str, Any] = {
            "turn_plan_json": default_turn_plan().model_dump(mode="json"),
            "knowledge_graph_json": {
                "updated_at": "",
                "summary": {"nodes": 0, "edges": 0},
                "nodes": [],
                "edges": [],
            },
            "landmarks_json": {
                "updated_at": "",
                "landmarks": [],
            },
            "recovery_saves_json": {
                "updated_at": "",
                "current_recommendation": None,
                "candidates": [],
                "autosave_history": [],
            },
            "turn_context_json": {},
            "latest_observation_json": {},
            "current_objective_json": {},
        }

        for key, payload in defaults.items():
            path = self.artifacts[key]
            if not path.exists():
                path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if not self.artifacts["working_memory_md"].exists():
            self.artifacts["working_memory_md"].write_text(
                "# Working Memory\n\n"
                "- Operator-only notes.\n"
                "- Keep notes factual: location, blockers, routes tried, battle plans.\n"
                "- Canonical model-facing state lives in turn_context.json.\n",
                encoding="utf-8",
            )
        if not self.artifacts["current_objective_md"].exists():
            self.artifacts["current_objective_md"].write_text(
                "Objective state will appear here after the first observation.\n",
                encoding="utf-8",
            )
        if not self.artifacts["latest_observation_md"].exists():
            self.artifacts["latest_observation_md"].write_text(
                "Observation summary will appear here after the first observation.\n",
                encoding="utf-8",
            )
        if not self.artifacts["session_brief_md"].exists():
            self.artifacts["session_brief_md"].write_text(
                "# Session Brief\n\nA deterministic resume brief will appear here.\n",
                encoding="utf-8",
            )

    def _load_existing_checkpoint_ids(self) -> None:
        path = self.artifacts["checkpoints_jsonl"]
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            checkpoint_id = record.get("id")
            if checkpoint_id:
                self.checkpoint_ids.add(checkpoint_id)

    def _load_existing_landmarks(self) -> None:
        payload = self._read_json(self.artifacts["landmarks_json"], {"landmarks": []})
        for landmark in payload.get("landmarks", []):
            if isinstance(landmark, dict) and landmark.get("id"):
                self.landmarks_by_id[str(landmark["id"])] = landmark

    def _load_existing_event_memory(self) -> None:
        path = self.artifacts["event_memory_jsonl"]
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            self.semantic_memory.append(record)
            self._ingest_failure_memory_event(record)
            if record.get("kind") == "dialog_text" and record.get("text"):
                self.last_dialog_text = str(record["text"])
                self.dialog_last_change_at = record.get("timestamp")

    def _write_json(self, path: Path, payload: Any) -> None:
        _atomic_write_text(
            path,
            json.dumps(payload, indent=2, sort_keys=False, default=_json_default),
        )

    def _append_jsonl(self, path: Path, payload: Any) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=_json_default) + "\n")

    def _read_json(self, path: Path, fallback: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return fallback

    def load_turn_plan_model(self) -> TurnPlan:
        return parse_stored_turn_plan(
            self._read_json(self.artifacts["turn_plan_json"], default_turn_plan().model_dump())
        )

    def load_turn_plan(self) -> JsonDict:
        return self._turn_plan_view(self.load_turn_plan_model())

    def load_turn_context(self) -> JsonDict:
        return self._read_json(self.artifacts["turn_context_json"], {})

    def save_turn_plan(self, plan: TurnPlan) -> TurnPlan:
        payload = plan.model_dump(mode="json")
        self._write_json(self.artifacts["turn_plan_json"], payload)
        plan_view = self._turn_plan_view(plan)
        for bundle in (self.latest_bundle, self.live_bundle):
            if not bundle:
                continue
            bundle["turn_plan"] = plan_view
            bundle["plan_status"] = payload.get("status") or {}
        return plan

    def plan_status(self) -> JsonDict:
        return self.load_turn_plan_model().status.model_dump(mode="json")

    def validate_and_store_turn_plan(self, submission: JsonDict) -> TurnPlan:
        try:
            current_context = TurnContext.model_validate(self.load_turn_context())
            payload = TurnPlanInput.model_validate(submission)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc
        validate_turn_plan_submission(payload, current_context)
        plan = store_validated_plan(payload)
        return self.save_turn_plan(plan)

    def mark_turn_plan_executed(
        self,
        *,
        branch: str,
        branch_kind: str,
        requested_actions: list[str],
        executed_actions: int,
        baseline_map_name: Optional[str],
        baseline_position: Optional[JsonDict],
        summary: str,
    ) -> TurnPlan:
        plan = self.load_turn_plan_model()
        plan = mark_plan_executed(
            plan,
            branch=branch,  # type: ignore[arg-type]
            branch_kind=branch_kind,
            requested_actions=requested_actions,
            executed_actions=executed_actions,
            baseline_map_name=baseline_map_name,
            baseline_position=baseline_position,
            summary=summary,
        )
        return self.save_turn_plan(plan)

    def invalidate_turn_plan(self, reason: str, *, state: str = "invalid") -> TurnPlan:
        plan = self.load_turn_plan_model()
        if not plan.observation_id and not plan.intent:
            return plan
        updated = invalidate_plan(plan, reason=reason, state=state)  # type: ignore[arg-type]
        return self.save_turn_plan(updated)

    def _plan_branch_actions_view(self, branch: Any) -> list[str]:
        if not branch:
            return []
        if isinstance(branch, dict):
            kind = branch.get("kind")
            if kind == "raw_actions":
                return list(branch.get("actions") or [])
            if kind == "navigation":
                target = branch.get("target") or {}
                return [
                    "navigate:"
                    f"{target.get('x')},{target.get('y')}:{branch.get('mode', 'auto')}"
                ]
            return []
        kind = getattr(branch, "kind", None)
        if kind == "raw_actions":
            return list(getattr(branch, "actions", []) or [])
        if kind == "navigation":
            target = getattr(branch, "target", None)
            if target is None:
                return []
            return [f"navigate:{target.x},{target.y}:{getattr(branch, 'mode', 'auto')}"]
        return []

    def _turn_plan_view(self, plan_payload: Any) -> JsonDict:
        if isinstance(plan_payload, dict) and "summary" in plan_payload:
            return {
                "objective_id": plan_payload.get("objective_id"),
                "summary": plan_payload.get("summary"),
                "planned_actions": list(plan_payload.get("planned_actions") or []),
                "fallback_actions": list(plan_payload.get("fallback_actions") or []),
                "notes": plan_payload.get("notes"),
                "updated_at": plan_payload.get("updated_at"),
                "status": plan_payload.get("status") or {},
                "mode": plan_payload.get("mode"),
                "observation_id": plan_payload.get("observation_id"),
            }

        plan = (
            plan_payload
            if isinstance(plan_payload, TurnPlan)
            else parse_stored_turn_plan(plan_payload)
        )
        status = plan.status.model_dump(mode="json")
        summary = plan.intent or status.get("reason") or "Awaiting next plan."
        notes = plan.notes or status.get("reason") or ""
        expected_outcome = plan.expected_outcome.summary if plan.expected_outcome else ""
        if expected_outcome:
            notes = f"{notes} Expected: {expected_outcome}".strip()
        return {
            "objective_id": plan.objective_id,
            "summary": summary,
            "planned_actions": self._plan_branch_actions_view(plan.primary_branch),
            "fallback_actions": self._plan_branch_actions_view(plan.fallback_branch),
            "notes": notes,
            "updated_at": plan.updated_at,
            "status": status,
            "mode": plan.mode,
            "observation_id": plan.observation_id,
        }

    def _record_event(self, event_type: str, payload: JsonDict) -> JsonDict:
        event = {
            "type": event_type,
            "timestamp": utc_now(),
            **payload,
        }
        self.event_history.append(event)
        self._append_jsonl(self.artifacts["run_log_jsonl"], event)
        return event

    def record_external_event(self, event_type: str, payload: JsonDict) -> JsonDict:
        """Record an external API event so dashboard history matches websocket traffic."""
        return self._record_event(event_type, payload)

    def _tail_jsonl(self, path: Path, limit: int) -> list[JsonDict]:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        result: list[JsonDict] = []
        for line in lines[-limit:]:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return result

    def history(self, limit: int = 200) -> list[JsonDict]:
        if self.event_history:
            return list(self.event_history)[-limit:]
        return self._tail_jsonl(self.artifacts["run_log_jsonl"], limit)

    def _map_key(
        self,
        *,
        state: Optional[JsonDict] = None,
        snapshot: Optional[LiveNavigationSnapshot] = None,
    ) -> str:
        if state is not None:
            map_info = state.get("map") or {}
            if map_info.get("map_name") is not None:
                return f"{map_info.get('map_id', '?')}:{map_info.get('map_name', 'Unknown')}"
        if snapshot is not None:
            return snapshot.key
        map_info = (state or {}).get("map") or {}
        return f"{map_info.get('map_id', '?')}:{map_info.get('map_name', 'Unknown')}"

    def _persist_landmarks(self) -> None:
        payload = {
            "updated_at": utc_now(),
            "landmarks": sorted(
                self.landmarks_by_id.values(),
                key=lambda entry: (
                    int(entry.get("map_id", 0)),
                    str(entry.get("kind", "")),
                    int((entry.get("coord") or {}).get("y", 0)),
                    int((entry.get("coord") or {}).get("x", 0)),
                    str(entry.get("title", "")),
                ),
            ),
        }
        self._write_json(self.artifacts["landmarks_json"], payload)

    def _landmarks_for_map(self, *, map_id: int, map_name: str) -> list[JsonDict]:
        return [
            landmark
            for landmark in self.landmarks_by_id.values()
            if int(landmark.get("map_id", -1)) == map_id
            and str(landmark.get("map_name", "")) == map_name
        ]

    def _upsert_landmark(
        self,
        *,
        map_id: int,
        map_name: str,
        kind: str,
        coord: tuple[int, int],
        title: str,
        source: str,
        notes: Optional[list[str]] = None,
    ) -> tuple[JsonDict, bool]:
        now = utc_now()
        landmark_id = f"landmark:{_stable_id(map_id, kind, _coord_to_key(coord))}"
        existing = self.landmarks_by_id.get(landmark_id)
        if existing:
            existing["title"] = title
            existing["last_seen_at"] = now
            existing["seen_count"] = int(existing.get("seen_count", 0)) + 1
            existing["source"] = existing.get("source") or source
            merged_notes = list(existing.get("notes") or [])
            for note in notes or []:
                if note and note not in merged_notes:
                    merged_notes.append(note)
            existing["notes"] = merged_notes[:6]
            self.landmarks_by_id[landmark_id] = existing
            return existing, False

        record = LandmarkRecord(
            id=landmark_id,
            map_id=map_id,
            map_name=map_name,
            kind=kind,
            title=title,
            coord={"x": coord[0], "y": coord[1]},
            discovered_at=now,
            last_seen_at=now,
            source=source,
            notes=list(notes or [])[:6],
        ).to_dict()
        self.landmarks_by_id[landmark_id] = record
        return record, True

    def _ingest_failure_memory_event(self, record: JsonDict) -> None:
        if record.get("kind") != "failure":
            return
        map_key = str(record.get("map_key") or "unknown")
        coord = _tuple_coord_from_any(record.get("coord"))
        coord_key = str(record.get("coord_key") or (_coord_to_key(coord) if coord else record.get("type") or "unknown"))
        bucket = self.failure_memory[map_key]
        existing = bucket.get(coord_key)
        count_delta = max(1, int(record.get("count_delta") or 1))
        if existing:
            existing["count"] = int(existing.get("count", 0)) + count_delta
            existing["last_seen_at"] = record.get("timestamp") or utc_now()
            existing["summary"] = record.get("summary") or existing.get("summary")
            existing["reason"] = record.get("reason") or existing.get("reason")
            if record.get("actions"):
                existing["actions"] = record.get("actions")
            bucket[coord_key] = existing
        else:
            bucket[coord_key] = {
                "id": f"failure:{_stable_id(map_key, coord_key, record.get('type'))}",
                "map_key": map_key,
                "map_id": record.get("map_id"),
                "map_name": record.get("map_name"),
                "coord": record.get("coord"),
                "coord_key": coord_key,
                "type": record.get("type"),
                "summary": record.get("summary"),
                "reason": record.get("reason"),
                "actions": record.get("actions") or [],
                "count": count_delta,
                "first_seen_at": record.get("timestamp") or utc_now(),
                "last_seen_at": record.get("timestamp") or utc_now(),
            }

        failure = bucket[coord_key]
        if coord is not None and int(failure.get("count", 0)) >= 2:
            map_id = int(record.get("map_id") or -1)
            map_name = str(record.get("map_name") or "")
            if map_id >= 0 and map_name:
                self._upsert_landmark(
                    map_id=map_id,
                    map_name=map_name,
                    kind="dead_end",
                    coord=coord,
                    title=f"Repeated failure near ({coord[0]}, {coord[1]})",
                    source="failure_memory",
                    notes=[str(record.get("summary") or record.get("reason") or "Repeated failure")],
                )
                self._persist_landmarks()

    def _record_semantic_memory(self, kind: str, summary: str, **payload: Any) -> JsonDict:
        record = {
            "id": payload.pop(
                "id",
                f"memory:{_stable_id(kind, summary, payload.get('map_key'), payload.get('coord_key'), payload.get('objective_id'))}",
            ),
            "timestamp": utc_now(),
            "kind": kind,
            "summary": summary,
            **payload,
        }
        if self.semantic_memory:
            last = self.semantic_memory[-1]
            if (
                last.get("kind") == record["kind"]
                and last.get("summary") == record["summary"]
                and last.get("map_key") == record.get("map_key")
                and kind != "failure"
            ):
                return last
        self.semantic_memory.append(record)
        self._append_jsonl(self.artifacts["event_memory_jsonl"], record)
        self._ingest_failure_memory_event(record)
        return record

    def _discover_landmarks(
        self,
        *,
        snapshot: Optional[LiveNavigationSnapshot],
    ) -> list[JsonDict]:
        if snapshot is None:
            return []

        created: list[JsonDict] = []
        changed = False
        width = int((snapshot.map_dimensions or {}).get("width") or 0)
        height = int((snapshot.map_dimensions or {}).get("height") or 0)

        for warp in snapshot.warps:
            coord = _tuple_coord_from_any(warp)
            if coord is None:
                continue
            edge_exit = (
                coord[0] == 0
                or coord[1] == 0
                or (width > 0 and coord[0] >= width - 1)
                or (height > 0 and coord[1] >= height - 1)
            )
            kind = "exit" if edge_exit else "warp"
            title = (
                f"Map exit at ({coord[0]}, {coord[1]})"
                if edge_exit
                else f"Warp at ({coord[0]}, {coord[1]})"
            )
            record, is_new = self._upsert_landmark(
                map_id=snapshot.map_id,
                map_name=snapshot.map_name,
                kind=kind,
                coord=coord,
                title=title,
                source="navigation",
                notes=[f"target_map_id={warp.get('target_map_id')}"],
            )
            changed = True
            if is_new:
                created.append(record)

        for sign in snapshot.signs:
            coord = _tuple_coord_from_any(sign)
            if coord is None:
                continue
            record, is_new = self._upsert_landmark(
                map_id=snapshot.map_id,
                map_name=snapshot.map_name,
                kind="sign",
                coord=coord,
                title=f"Sign at ({coord[0]}, {coord[1]})",
                source="navigation",
                notes=[f"text_id={sign.get('text_id')}"],
            )
            changed = True
            if is_new:
                created.append(record)

        interaction = snapshot.interaction or {}
        target = _tuple_coord_from_any(interaction.get("target_coord"))
        interaction_kind = str(interaction.get("kind") or "").lower()
        interaction_source = str(interaction.get("source") or "").lower()
        non_destination_sources = {
            "blocked_tile",
            "counter_tile",
            "dialog_lock",
            "unknown_facing",
            "none",
        }
        is_destination = (
            target is not None
            and interaction_kind in {"object", "sign"}
            and interaction_source not in non_destination_sources
            and interaction_source != ""
        )
        if is_destination:
            assert target is not None
            if interaction_source in {"sprite_direct", "sprite_via_talk_over"}:
                kind = "npc_blocker"
                title = f"NPC blocker at ({target[0]}, {target[1]})"
            elif "sign" in interaction_source:
                kind = "sign"
                title = f"Talkable sign at ({target[0]}, {target[1]})"
            else:
                kind = "interactable"
                title = f"Interactable at ({target[0]}, {target[1]})"
            record, is_new = self._upsert_landmark(
                map_id=snapshot.map_id,
                map_name=snapshot.map_name,
                kind=kind,
                coord=target,
                title=title,
                source="interaction_probe",
                notes=[str(interaction.get("reason") or "Visible interaction target")],
            )
            changed = True
            if is_new:
                created.append(record)

        if changed:
            self._persist_landmarks()
        return created

    def _build_navigation_avoidances(
        self,
        *,
        map_key: str,
        current: Optional[tuple[int, int]],
    ) -> list[JsonDict]:
        failures = list(self.failure_memory.get(map_key, {}).values())
        ranked = sorted(
            failures,
            key=lambda entry: (
                -int(entry.get("count", 0)),
                _manhattan(current, _tuple_coord_from_any(entry.get("coord"))),
                str(entry.get("summary") or ""),
            ),
        )
        result: list[JsonDict] = []
        for entry in ranked[:5]:
            coord = _tuple_coord_from_any(entry.get("coord"))
            result.append(
                {
                    "id": entry.get("id"),
                    "kind": entry.get("type"),
                    "coord": _coord_payload(coord),
                    "title": entry.get("summary") or "Repeated failure",
                    "reason": entry.get("reason") or entry.get("summary"),
                    "times_seen": entry.get("count", 0),
                    "last_seen_at": entry.get("last_seen_at"),
                    "actions": (entry.get("actions") or [])[:4],
                }
            )
        return result

    def _build_navigation_assistance(
        self,
        *,
        snapshot: Optional[LiveNavigationSnapshot],
        navigation_store: Optional[NavigationStore],
        objective: JsonDict,
    ) -> JsonDict:
        if snapshot is None or navigation_store is None:
            return {
                "frontiers": [],
                "landmarks": [],
                "distance_ascii": "(navigation guidance unavailable)",
                "route_cards": [],
                "avoidances": [],
            }

        location_map = navigation_store.get(snapshot.key)
        if location_map is None:
            return {
                "frontiers": [],
                "landmarks": [],
                "distance_ascii": "(explored map unavailable)",
                "route_cards": [],
                "avoidances": [],
            }

        current = snapshot.player_position
        preferred_types = set((objective.get("current") or {}).get("preferred_landmark_types") or [])
        preferred_direction = _preferred_direction_hint(
            (objective.get("current") or {}).get("route_hint", "")
        )
        avoidances = self._build_navigation_avoidances(map_key=snapshot.key, current=current)
        blocked_coords = {
            _tuple_coord_from_any(entry.get("coord"))
            for entry in avoidances
            if _tuple_coord_from_any(entry.get("coord")) is not None
        }

        distances = location_map.distance_map(current, extra_blockers=snapshot.sprite_set)
        frontiers: list[JsonDict] = []
        seen_frontier_coords: set[tuple[int, int]] = set()
        for coord, distance in sorted(
            distances.items(),
            key=lambda item: (item[1], item[0][1], item[0][0]),
        ):
            if coord == current:
                continue
            unknown_neighbors = [
                neighbor
                for _, neighbor in (
                    ((0, -1), (coord[0], coord[1] - 1)),
                    ((0, 1), (coord[0], coord[1] + 1)),
                    ((-1, 0), (coord[0] - 1, coord[1])),
                    ((1, 0), (coord[0] + 1, coord[1])),
                )
                if location_map.tiles.get(neighbor) is None
            ]
            if not unknown_neighbors or coord in seen_frontier_coords:
                continue
            route = location_map.plan_route(
                start=current,
                goal=coord,
                extra_blockers=snapshot.sprite_set,
                allow_partial=False,
            )
            progress_bonus = max(0, _progress_amount(preferred_direction or "", current, coord))
            blocked = any(_manhattan(coord, blocked_coord) <= 1 for blocked_coord in blocked_coords)
            novelty_score = max(1, 35 + (len(unknown_neighbors) * 12) + (progress_bonus * 4) - (distance * 2) - (25 if blocked else 0))
            title = f"Probe frontier at ({coord[0]}, {coord[1]})"
            why_now_parts = [f"reveals {len(unknown_neighbors)} unknown edge(s)"]
            if progress_bonus > 0 and preferred_direction:
                why_now_parts.append(f"advances {preferred_direction}")
            if blocked:
                why_now_parts.append("near a recently failed branch")
            frontiers.append(
                {
                    "id": f"frontier:{_stable_id(snapshot.key, _coord_to_key(coord))}",
                    "kind": "frontier",
                    "coord": _coord_payload(coord),
                    "title": title,
                    "novelty_score": novelty_score,
                    "route_steps": len(route.actions),
                    "route_actions": route.actions[:8],
                    "first_action": route.actions[0] if route.actions else None,
                    "why_now": ", ".join(why_now_parts),
                    "blocked_by_recent_failure": blocked,
                }
            )
            seen_frontier_coords.add(coord)

        frontiers = sorted(
            frontiers,
            key=lambda entry: (
                -int(entry.get("novelty_score", 0)),
                int(entry.get("route_steps", 999)),
                str(entry.get("title", "")),
            ),
        )[:8]

        landmarks: list[JsonDict] = []
        route_cards: list[JsonDict] = []
        map_landmarks = self._landmarks_for_map(map_id=snapshot.map_id, map_name=snapshot.map_name)
        for landmark in sorted(
            map_landmarks,
            key=lambda entry: (
                _manhattan(current, _tuple_coord_from_any(entry.get("coord"))),
                str(entry.get("kind", "")),
                str(entry.get("title", "")),
            ),
        ):
            coord = _tuple_coord_from_any(landmark.get("coord"))
            if coord is None:
                continue
            visible = snapshot.absolute_to_local(coord[0], coord[1]) is not None
            distance = _manhattan(current, coord)
            landmark_view = {
                **landmark,
                "coord": _coord_payload(coord),
                "distance": distance,
                "visible": visible,
            }
            landmarks.append(landmark_view)

            route = location_map.plan_route(
                start=current,
                goal=coord,
                extra_blockers=snapshot.sprite_set,
                allow_partial=True,
            )
            blocked = any(_manhattan(coord, blocked_coord) <= 1 for blocked_coord in blocked_coords)
            type_bonus = 24 if str(landmark.get("kind")) in preferred_types else 0
            score = max(
                1,
                55
                + type_bonus
                + (12 if visible else 0)
                - (distance * 2)
                - (18 if blocked else 0)
                - (8 if str(landmark.get("kind")) == "dead_end" else 0),
            )
            route_cards.append(
                {
                    "id": f"route:{landmark.get('id')}",
                    "kind": str(landmark.get("kind") or "landmark"),
                    "coord": _coord_payload(coord),
                    "title": landmark.get("title"),
                    "score": score,
                    "route_steps": len(route.actions),
                    "route_actions": route.actions[:8],
                    "first_action": route.actions[0] if route.actions else None,
                    "why_now": (
                        "Matches the current objective preferences."
                        if type_bonus
                        else "Known landmark with a concrete route."
                    ),
                    "blocked_by_recent_failure": blocked,
                    "target_id": landmark.get("id"),
                }
            )

        for frontier in frontiers:
            route_cards.append(
                {
                    "id": f"route:{frontier['id']}",
                    "kind": frontier["kind"],
                    "coord": frontier["coord"],
                    "title": frontier["title"],
                    "score": frontier["novelty_score"],
                    "route_steps": frontier["route_steps"],
                    "route_actions": frontier["route_actions"],
                    "first_action": frontier["first_action"],
                    "why_now": frontier["why_now"],
                    "blocked_by_recent_failure": frontier["blocked_by_recent_failure"],
                    "target_id": frontier["id"],
                }
            )

        route_cards = sorted(
            route_cards,
            key=lambda entry: (
                -int(entry.get("score", 0)),
                int(entry.get("route_steps", 999)),
                str(entry.get("title", "")),
            ),
        )[:5]

        distance_ascii = location_map.render_distance_ascii(
            start=current,
            extra_blockers=snapshot.sprite_set,
            sprites=snapshot.sprite_positions,
            max_distance=60,
        )

        return {
            "frontiers": frontiers,
            "landmarks": landmarks[:12],
            "distance_ascii": distance_ascii,
            "route_cards": route_cards,
            "avoidances": avoidances,
        }

    def _update_dialog_guidance(
        self,
        *,
        screen_text: JsonDict,
        state: JsonDict,
    ) -> tuple[JsonDict, Optional[JsonDict]]:
        dialog = state.get("dialog") or {}
        dialog_active = bool(state.get("dialog_active") or dialog.get("active"))
        text = str(screen_text.get("text") or "").strip()
        changed_event = None
        if dialog_active and text and not text.startswith("Dialog box visible") and text != self.last_dialog_text:
            self.last_dialog_text = text
            self.dialog_last_change_at = utc_now()
            entry = {"timestamp": self.dialog_last_change_at, "text": text}
            self.dialog_transcript_recent.append(entry)
            changed_event = entry
        elif not dialog_active:
            self.last_dialog_text = ""

        return (
            {
                "transcript_recent": [entry["text"] for entry in list(self.dialog_transcript_recent)[-4:]],
                "should_continue": dialog_active,
                "last_change_at": self.dialog_last_change_at,
                "printing": bool(dialog.get("printing")),
                "waiting_for_input": bool(dialog.get("waiting_for_input")),
            },
            changed_event,
        )

    def _type_multiplier(self, move_type: str, enemy_types: Iterable[str]) -> float:
        multiplier = 1.0
        for enemy_type in enemy_types:
            multiplier *= TYPE_EFFECTIVENESS.get(move_type, {}).get(str(enemy_type), 1.0)
        return multiplier

    def _build_battle_guidance(self, state: JsonDict, dialog_guidance: JsonDict) -> JsonDict:
        battle = state.get("battle") or {}
        if not battle.get("in_battle"):
            return {
                "recommended_mode": "none",
                "recommended_move": None,
                "reason": "No active battle.",
                "safe_short_actions": [],
            }

        if dialog_guidance.get("should_continue"):
            return {
                "recommended_mode": "advance_text",
                "recommended_move": None,
                "reason": "Battle text is still active; clear dialog before selecting another move.",
                "safe_short_actions": ["press_a"],
            }

        party = state.get("party") or []
        active = party[0] if party else {}
        enemy = battle.get("enemy") or {}
        enemy_types = [entry for entry in (enemy.get("types") or []) if entry]
        user_types = [entry for entry in (active.get("types") or []) if entry]
        best_move: Optional[JsonDict] = None
        best_score = -9999.0
        for move in active.get("moves") or []:
            if int(move.get("pp") or 0) <= 0:
                continue
            metadata = MOVE_METADATA.get(str(move.get("name") or ""), {"type": "Normal", "power": 35})
            if metadata.get("status"):
                score = -10.0
            else:
                score = float(metadata.get("power") or 35)
                move_type = str(metadata.get("type") or "Normal")
                if move_type in user_types:
                    score *= 1.2
                score *= self._type_multiplier(move_type, enemy_types)
                score += min(6, int(move.get("pp") or 0))
            if score > best_score:
                best_score = score
                best_move = {
                    "name": move.get("name"),
                    "type": metadata.get("type"),
                    "power": metadata.get("power"),
                    "pp": move.get("pp"),
                    "score": round(score, 2),
                }

        if best_move is None:
            return {
                "recommended_mode": "advance_text",
                "recommended_move": None,
                "reason": "No usable damaging move is visible; keep battle actions extremely short.",
                "safe_short_actions": ["press_a"],
            }

        reason = (
            f"{best_move['name']} scores best against "
            f"{'/'.join(enemy_types) if enemy_types else 'the current enemy'} "
            f"with PP {best_move['pp']}."
        )
        return {
            "recommended_mode": "select_best_move",
            "recommended_move": best_move,
            "reason": reason,
            "safe_short_actions": ["press_a"],
        }

    def _build_hypotheses(
        self,
        *,
        objective: JsonDict,
        navigation_guidance: JsonDict,
        dialog_guidance: JsonDict,
        battle_guidance: JsonDict,
    ) -> list[str]:
        hypotheses: list[str] = []
        current = objective.get("current") or {}
        if dialog_guidance.get("should_continue"):
            hypotheses.append("Clear the active dialog before attempting movement.")
        elif battle_guidance.get("recommended_mode") == "select_best_move":
            move = battle_guidance.get("recommended_move") or {}
            if move.get("name"):
                hypotheses.append(f"In battle, prefer {move['name']} on the next menu advance.")
        route_cards = navigation_guidance.get("route_cards") or []
        if route_cards:
            top = route_cards[0]
            hypotheses.append(f"Best near-term progress is {top.get('title')}.")
        if current.get("route_hint"):
            hypotheses.append(f"Route hint: {current['route_hint']}")
        return hypotheses[:4]

    def _build_memory_snapshot(
        self,
        *,
        objective: JsonDict,
        navigation_guidance: JsonDict,
        dialog_guidance: JsonDict,
        battle_guidance: JsonDict,
        state: JsonDict,
    ) -> JsonDict:
        map_key = self._map_key(state=state)
        failed_attempts = self._build_navigation_avoidances(
            map_key=map_key,
            current=_tuple_coord_from_any((state.get("player") or {}).get("position")),
        )
        recent_facts = [
            {
                "timestamp": entry.get("timestamp"),
                "kind": entry.get("kind"),
                "summary": entry.get("summary"),
            }
            for entry in list(self.semantic_memory)[-8:]
        ]
        hypotheses = self._build_hypotheses(
            objective=objective,
            navigation_guidance=navigation_guidance,
            dialog_guidance=dialog_guidance,
            battle_guidance=battle_guidance,
        )
        return {
            "recent_facts": recent_facts,
            "current_hypotheses": hypotheses,
            "failed_attempts": failed_attempts,
            "session_brief_path": str(self.artifacts["session_brief_md"]),
        }

    def _write_session_brief(
        self,
        *,
        objective: JsonDict,
        navigation_guidance: JsonDict,
        memory_snapshot: JsonDict,
        dialog_guidance: JsonDict,
        battle_guidance: JsonDict,
        state: JsonDict,
    ) -> None:
        current = objective.get("current") or {}
        route_cards = navigation_guidance.get("route_cards") or []
        lines = [
            "# Session Brief",
            "",
            f"- Objective: {current.get('title', 'Unknown objective')}",
            f"- Objective summary: {current.get('summary', 'No summary.')}",
            f"- Map: {(state.get('map') or {}).get('map_name', 'Unknown')}",
            f"- Position: {(state.get('player') or {}).get('position')}",
            "",
            "## Best Routes",
        ]
        for card in route_cards[:5]:
            lines.append(
                f"- {card.get('title')} | score={card.get('score')} | "
                f"actions={', '.join(card.get('route_actions') or []) or 'none'}"
            )
        lines.extend(["", "## Recent Facts"])
        for fact in memory_snapshot.get("recent_facts") or []:
            lines.append(f"- [{fact.get('kind')}] {fact.get('summary')}")
        lines.extend(["", "## Failed Attempts"])
        for attempt in memory_snapshot.get("failed_attempts") or []:
            lines.append(
                f"- {attempt.get('title')} | seen={attempt.get('times_seen')} | "
                f"{attempt.get('reason')}"
            )
        lines.extend(["", "## Dialog"])
        lines.append(
            f"- Continue dialog: {dialog_guidance.get('should_continue')} | "
            f"last_change_at={dialog_guidance.get('last_change_at')}"
        )
        for text in dialog_guidance.get("transcript_recent") or []:
            lines.append(f"- {text}")
        lines.extend(["", "## Battle"])
        lines.append(f"- Mode: {battle_guidance.get('recommended_mode')}")
        lines.append(f"- Reason: {battle_guidance.get('reason')}")
        move = battle_guidance.get("recommended_move") or {}
        if move.get("name"):
            lines.append(f"- Recommended move: {move.get('name')} ({move.get('type')})")
        _atomic_write_text(self.artifacts["session_brief_md"], "\n".join(lines) + "\n")

    def _candidate_checkpoints(
        self, state: JsonDict, objective: JsonDict
    ) -> list[CheckpointRecord]:
        flags = state.get("flags") or {}
        map_name = (state.get("map") or {}).get("map_name") or "Unknown"
        current_objective_id = objective["current"]["id"]
        records: list[CheckpointRecord] = []

        definitions = [
            (
                bool(flags.get("has_oaks_parcel")),
                "oak_parcel_obtained",
                "Oak's Parcel Obtained",
                "Oak's Parcel is now in the bag.",
            ),
            (
                bool(flags.get("has_pokedex")),
                "pokedex_received",
                "Pokedex Received",
                "Oak's parcel was delivered and the Pokedex was received.",
            ),
            (
                map_name == "Viridian Forest",
                "entered_viridian_forest",
                "Entered Viridian Forest",
                "Reached the forest section of the Brock route.",
            ),
            (
                map_name == "Pewter City",
                "arrived_pewter_city",
                "Arrived In Pewter City",
                "Made it through the forest to Pewter City.",
            ),
            (
                map_name == "Pewter Gym",
                "entered_pewter_gym",
                "Entered Pewter Gym",
                "Ready to challenge Brock.",
            ),
            (
                (flags.get("badge_count") or 0) >= 1,
                "boulder_badge_earned",
                "Boulder Badge Earned",
                "Phase 1 objective pack is complete.",
            ),
        ]

        for condition, checkpoint_id, title, summary in definitions:
            if not condition or checkpoint_id in self.checkpoint_ids:
                continue
            records.append(
                CheckpointRecord(
                    id=checkpoint_id,
                    title=title,
                    summary=summary,
                    objective_id=current_objective_id,
                    map_name=map_name,
                    created_at=utc_now(),
                    metadata={"source": "deterministic_progress"},
                )
            )
        return records

    def _persist_checkpoints(self, checkpoints: Iterable[CheckpointRecord]) -> list[JsonDict]:
        emitted: list[JsonDict] = []
        for checkpoint in checkpoints:
            record = checkpoint.to_dict()
            self._append_jsonl(self.artifacts["checkpoints_jsonl"], record)
            self.checkpoint_ids.add(checkpoint.id)
            emitted.append(record)
        return emitted

    def _build_knowledge_graph(
        self,
        *,
        state: JsonDict,
        navigation: Optional[JsonDict],
        objective: JsonDict,
        navigation_store: Optional[NavigationStore],
    ) -> JsonDict:
        existing = self._read_json(
            self.artifacts["knowledge_graph_json"],
            {
                "updated_at": "",
                "summary": {"nodes": 0, "edges": 0},
                "nodes": [],
                "edges": [],
            },
        )
        nodes_by_id = {
            node["id"]: node
            for node in existing.get("nodes", [])
            if isinstance(node, dict) and node.get("id")
        }
        edges_by_key = {
            (edge.get("source"), edge.get("target"), edge.get("type")): edge
            for edge in existing.get("edges", [])
            if isinstance(edge, dict)
        }

        map_info = state.get("map") or {}
        player = state.get("player") or {}
        current_location_id = (
            f"location:{map_info.get('map_id')}:{_slugify(map_info.get('map_name', 'unknown'))}"
        )
        nodes_by_id[current_location_id] = {
            "id": current_location_id,
            "type": "location",
            "label": map_info.get("map_name", "Unknown"),
            "current": True,
            "position": player.get("position"),
        }

        for location in (
            navigation_store.location_maps.values() if navigation_store is not None else []
        ):
            location_id = f"location:{location.map_id}:{_slugify(location.map_name)}"
            node = nodes_by_id.get(location_id, {})
            node.update(
                {
                    "id": location_id,
                    "type": "location",
                    "label": location.map_name,
                    "known_tiles": len(location.tiles),
                    "passable_tiles": location.passable_count(),
                    "blocked_tiles": location.blocked_count(),
                    "current": location_id == current_location_id,
                }
            )
            nodes_by_id[location_id] = node

        current_objective = objective["current"]
        objective_id = f"objective:{current_objective['id']}"
        nodes_by_id[objective_id] = {
            "id": objective_id,
            "type": "objective",
            "label": current_objective["title"],
            "status": current_objective["status"],
            "progress_percent": objective["progress_percent"],
        }
        edges_by_key[(current_location_id, objective_id, "objective_context")] = {
            "source": current_location_id,
            "target": objective_id,
            "type": "objective_context",
        }

        flags = state.get("flags") or {}
        for key, label in (
            ("has_pokedex", "Has Pokedex"),
            ("has_oaks_parcel", "Has Oak's Parcel"),
        ):
            if not flags.get(key):
                continue
            node_id = f"fact:{key}"
            nodes_by_id[node_id] = {
                "id": node_id,
                "type": "fact",
                "label": label,
                "value": True,
            }
            edges_by_key[(current_location_id, node_id, "fact_seen_here")] = {
                "source": current_location_id,
                "target": node_id,
                "type": "fact_seen_here",
            }

        if navigation:
            snapshot = navigation.get("snapshot") or {}
            interaction = snapshot.get("interaction") or {}
            if interaction.get("target_coord"):
                target = interaction["target_coord"]
                node_id = (
                    f"interaction:{interaction.get('kind', 'unknown')}:"
                    f"{target.get('x')}:{target.get('y')}"
                )
                nodes_by_id[node_id] = {
                    "id": node_id,
                    "type": "interaction",
                    "label": interaction.get("kind", "unknown"),
                    "source": interaction.get("source"),
                    "target_coord": target,
                    "reason": interaction.get("reason"),
                }
                edges_by_key[(current_location_id, node_id, "visible_interaction")] = {
                    "source": current_location_id,
                    "target": node_id,
                    "type": "visible_interaction",
                }

            for warp in snapshot.get("warps") or []:
                warp_id = f"warp:{map_info.get('map_id')}:{warp.get('x')}:{warp.get('y')}"
                nodes_by_id[warp_id] = {
                    "id": warp_id,
                    "type": "exit",
                    "label": f"Warp at ({warp.get('x')}, {warp.get('y')})",
                    "target_map_id": warp.get("target_map_id"),
                }
                edges_by_key[(current_location_id, warp_id, "contains_exit")] = {
                    "source": current_location_id,
                    "target": warp_id,
                    "type": "contains_exit",
                }

        graph = {
            "updated_at": utc_now(),
            "summary": {
                "nodes": len(nodes_by_id),
                "edges": len(edges_by_key),
                "current_location": map_info.get("map_name"),
                "current_objective": current_objective["title"],
            },
            "nodes": sorted(nodes_by_id.values(), key=lambda node: node["id"]),
            "edges": sorted(
                edges_by_key.values(),
                key=lambda edge: (edge["source"], edge["type"], edge["target"]),
            ),
        }
        self._write_json(self.artifacts["knowledge_graph_json"], graph)
        return graph

    def _maybe_auto_save(
        self,
        *,
        emulator: Any,
        state: JsonDict,
        objective: JsonDict,
        state_delta: JsonDict,
        checkpoints: list[JsonDict],
        requested_actions: Optional[list[str]],
        source: str,
    ) -> list[JsonDict]:
        triggers: list[str] = []
        if state_delta.get("fields", {}).get("map"):
            triggers.append("map_transition")
        if objective["current"]["id"] != self.last_objective_id:
            triggers.append("objective_change")
        if checkpoints:
            triggers.append("checkpoint")
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
            name = f"auto__{stamp}__{_slugify(trigger)}__{_slugify(current_map)}"
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

    def _build_recovery(self, recent_auto_saves: list[JsonDict]) -> JsonDict:
        current = self._read_json(
            self.artifacts["recovery_saves_json"],
            {
                "updated_at": "",
                "current_recommendation": None,
                "candidates": [],
                "autosave_history": [],
            },
        )
        autosave_history = list(current.get("autosave_history", []))
        autosave_history.extend(recent_auto_saves)
        autosave_history = autosave_history[-50:]

        saves_dir = self.data_dir / "saves"
        candidates: list[RecoveryCandidate] = []
        known_notes = {
            entry.get("name"): entry
            for entry in autosave_history
            if isinstance(entry, dict) and entry.get("name")
        }
        for path in sorted(saves_dir.glob("*.state")):
            meta = known_notes.get(path.stem, {})
            source = meta.get("source", "manual")
            reason = meta.get("reason", "manual_save")
            score = 40
            if reason == "checkpoint":
                score = 95
            elif reason == "objective_change":
                score = 90
            elif reason == "map_transition":
                score = 75
            elif reason == "battle_entry":
                score = 65
            elif source == "manual":
                score = 80
            candidates.append(
                RecoveryCandidate(
                    name=path.stem,
                    path=str(path),
                    reason=reason,
                    score=score,
                    modified_at=datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                    source=source,
                    notes=list(meta.get("notes", [])),
                )
            )

        ranked = sorted(
            candidates,
            key=lambda entry: (entry.score, entry.modified_at, entry.name),
            reverse=True,
        )
        payload = {
            "updated_at": utc_now(),
            "current_recommendation": ranked[0].to_dict() if ranked else None,
            "candidates": [entry.to_dict() for entry in ranked[:12]],
            "autosave_history": autosave_history,
        }
        self._write_json(self.artifacts["recovery_saves_json"], payload)
        return payload

    def _detect_stuck(
        self,
        *,
        state: JsonDict,
        objective: JsonDict,
        source: str,
        requested_actions: Optional[list[str]],
        plan: Optional[TurnPlan] = None,
    ) -> JsonDict:
        player = state.get("player") or {}
        position = player.get("position") or {}
        plan_state = plan.status.state if plan is not None else None
        plan_target: Optional[JsonDict] = None
        if plan is not None and plan.expected_outcome is not None:
            if plan.expected_outcome.position is not None:
                plan_target = {
                    "x": plan.expected_outcome.position.x,
                    "y": plan.expected_outcome.position.y,
                }
            elif (
                plan.primary_branch is not None
                and getattr(plan.primary_branch, "kind", None) == "navigation"
            ):
                target = getattr(plan.primary_branch, "target", None)
                if target is not None:
                    plan_target = {"x": target.x, "y": target.y}
        signature = {
            "map_name": (state.get("map") or {}).get("map_name"),
            "position": position,
            "dialog_active": bool(
                state.get("dialog_active") or (state.get("dialog") or {}).get("active")
            ),
            "objective_id": objective["current"]["id"],
            "source": source,
            "actions": requested_actions or [],
            "plan_state": plan_state,
            "plan_target": plan_target,
        }
        self.recent_trajectory.append(signature)

        recent = list(self.recent_trajectory)[-8:]
        no_movement_loop = False
        dialog_loop = False
        objective_timeout = False
        drift_loop_count = 0
        drift_loop_targets: list[JsonDict] = []

        if len(recent) >= 4:
            locations = {
                (
                    item.get("map_name"),
                    json.dumps(item.get("position"), sort_keys=True),
                )
                for item in recent[-4:]
            }
            if len(locations) == 1 and any(item.get("actions") for item in recent[-4:]):
                no_movement_loop = True
            if no_movement_loop and all(item.get("dialog_active") for item in recent[-4:]):
                dialog_loop = True

        drifts = [item for item in recent if item.get("plan_state") == "drifted"]
        drift_loop_count = len(drifts)
        for item in drifts:
            target = item.get("plan_target")
            if target and target not in drift_loop_targets:
                drift_loop_targets.append(target)

        if objective["current"]["id"] == self.last_objective_id and source in {
            "action",
            "navigation",
        }:
            self.action_events_since_objective_change += 1
        else:
            self.action_events_since_objective_change = 0
        if self.action_events_since_objective_change >= 12:
            objective_timeout = True

        level = "clear"
        reason = "No stuck pattern detected."
        recommended: list[str] = []
        if dialog_loop:
            level = "warning"
            reason = (
                "Dialog loop detected: repeated actions with the same position and active dialog."
            )
            recommended = ["press_a", "press_b", "re-check latest_frame.png if text is ambiguous"]
        elif no_movement_loop:
            level = "warning"
            reason = "No-movement loop detected: repeated actions without position or map change."
            recommended = [
                "inspect valid_moves",
                "use a shorter action batch",
                "consider a recovery save",
            ]

        if drift_loop_count >= 3:
            new_level = "danger" if drift_loop_count >= 5 else "warning"
            if _stuck_level_rank(new_level) > _stuck_level_rank(level):
                level = new_level
            target_note = ""
            if drift_loop_targets:
                first = drift_loop_targets[0]
                target_note = f" target=({first.get('x')}, {first.get('y')})"
                if len(drift_loop_targets) > 1:
                    target_note += f" ({len(drift_loop_targets)} distinct targets)"
            reason = (
                f"Plan drifted {drift_loop_count} of the last {len(recent)} observations"
                f"{target_note}. The last raw_actions batches are not landing as planned."
            )
            if drift_loop_targets:
                first = drift_loop_targets[0]
                recommended = [
                    f"switch to mode=navigation with target={{x:{first.get('x')},y:{first.get('y')}}}",
                    "drop primary_branch to 1 action and re-check after each step",
                    "reload a recovery save if the same target keeps failing",
                ]
            else:
                recommended = [
                    "switch to mode=navigation with a nearby walkable target",
                    "drop primary_branch to 1 action and re-check after each step",
                    "reload a recovery save if the same target keeps failing",
                ]

        if objective_timeout:
            level = "danger" if level == "warning" else "warning"
            reason = "Current objective has seen many action turns without progress."
            recommended = recommended or [
                "review turn_plan.json",
                "reload the top recovery candidate",
            ]

        return {
            "level": level,
            "reason": reason,
            "recommended_actions": recommended,
            "objective_action_count": self.action_events_since_objective_change,
            "drift_count": drift_loop_count,
        }

    def _objective_markdown(self, objective: JsonDict) -> str:
        current = objective["current"]
        lines = [
            "# Current Objective",
            "",
            f"- Title: {current['title']}",
            f"- Summary: {current['summary']}",
            f"- Completion predicate: {current['completion_predicate']}",
            f"- Save recommendation: {current['save_recommendation']}",
            f"- Route hint: {current['route_hint']}",
            f"- Progress: {objective['progress_percent']}%",
            "",
            "## Failure Hints",
        ]
        for hint in current.get("failure_hints", []):
            lines.append(f"- {hint}")
        return "\n".join(lines) + "\n"

    def _observation_markdown(
        self,
        *,
        bundle: JsonDict,
    ) -> str:
        objective = bundle["objective"]["current"]
        state = bundle["state"]
        navigation = bundle.get("navigation") or {}
        snapshot = navigation.get("snapshot") or {}
        turn_plan = bundle["turn_plan"]
        feedback = bundle["recent_action"]
        movement_guidance = bundle["movement_guidance"]
        navigation_guidance = bundle.get("navigation_guidance") or {}
        memory_snapshot = bundle.get("memory") or {}
        dialog_guidance = bundle.get("dialog_guidance") or {}
        battle_guidance = bundle.get("battle_guidance") or {}
        stuck = bundle["stuck"]
        recovery = bundle["recovery"]
        state_delta_summary = (bundle.get("state_delta") or {}).get("summary") or [
            "Live frame sync only. Run /agent/observe for refreshed deltas."
        ]
        lines = [
            "# Vision-First Turn Brief",
            "",
            f"Read first: `{bundle['artifacts']['latest_frame_annotated']}`",
            f"Fallback frame: `{bundle['artifacts']['latest_frame']}`",
            "Mandatory: inspect the annotated frame before choosing actions.",
            "Do not infer terrain from ASCII alone.",
            "",
            f"Objective: {objective['title']}",
            f"Objective summary: {objective['summary']}",
            f"Route hint: {objective['route_hint']}",
            "",
            f"UI mode: {bundle['screen_text']['ui_mode']}",
            f"Map: {(state.get('map') or {}).get('map_name', 'Unknown')}",
            f"Position: {(state.get('player') or {}).get('position')}",
            f"Facing: {(state.get('player') or {}).get('facing')}",
            f"Valid moves: {', '.join(snapshot.get('valid_moves', [])) or 'none'}",
            f"Interaction probe: {(snapshot.get('interaction') or {}).get('reason', 'none')}",
            "ASCII legend: P=player G=goal S=sprite .=passable #=blocked ?=unknown",
            f"Movement guidance: {movement_guidance.get('summary', 'none')}",
            "",
            f"What changed: {' '.join(state_delta_summary[:3])}",
            f"Recent action result: {feedback.get('summary', 'No recent action result.')}",
            f"Planned action batch: {', '.join(turn_plan.get('planned_actions', [])) or 'not set'}",
            f"Fallback: {', '.join(turn_plan.get('fallback_actions', [])) or 'not set'}",
            "",
            f"Screen text: {bundle['screen_text']['text']}",
            "",
            f"Dialog continue: {dialog_guidance.get('should_continue')}",
            f"Battle mode: {battle_guidance.get('recommended_mode')}",
            f"Battle reason: {battle_guidance.get('reason')}",
            "",
            f"Recovery recommendation: "
            f"{(recovery.get('current_recommendation') or {}).get('name', 'none')}",
            f"Stuck signal: {stuck['level']} - {stuck['reason']}",
            "",
            "Top route cards:",
        ]
        for card in navigation_guidance.get("route_cards", [])[:5]:
            lines.append(
                f"- {card.get('title')} | "
                f"actions={', '.join(card.get('route_actions') or []) or 'none'} | "
                f"why={card.get('why_now')}"
            )
        lines.extend(
            [
                "",
                "Nearby landmarks:",
            ]
        )
        for landmark in navigation_guidance.get("landmarks", [])[:5]:
            lines.append(
                f"- {landmark.get('kind')}: {landmark.get('title')} "
                f"(distance {landmark.get('distance')})"
            )
        lines.extend(
            [
                "",
                "Failed attempts to avoid:",
            ]
        )
        for attempt in memory_snapshot.get("failed_attempts", [])[:4]:
            lines.append(
                f"- {attempt.get('title')} | seen={attempt.get('times_seen')} | "
                f"{attempt.get('reason')}"
            )
        lines.extend(
            [
                "",
                "Recent deterministic facts:",
            ]
        )
        for fact in memory_snapshot.get("recent_facts", [])[:5]:
            lines.append(f"- [{fact.get('kind')}] {fact.get('summary')}")
        lines.extend(
            [
                "",
                "Dialog transcript recent:",
            ]
        )
        for text in dialog_guidance.get("transcript_recent", [])[:4]:
            lines.append(f"- {text}")
        lines.extend(
            [
                "",
                "Navigation notes:",
            ]
        )
        for note in movement_guidance.get("notes", []):
            lines.append(f"- {note}")
        return "\n".join(lines) + "\n"

    def _artifact_payload(self) -> JsonDict:
        return {
            "latest_frame": str(self.artifacts["latest_frame"]),
            "latest_frame_annotated": str(self.artifacts["latest_frame_annotated"]),
            "turn_context_json": str(self.artifacts["turn_context_json"]),
            "latest_observation_json": str(self.artifacts["latest_observation_json"]),
            "latest_observation_md": str(self.artifacts["latest_observation_md"]),
            "current_objective_json": str(self.artifacts["current_objective_json"]),
            "current_objective_md": str(self.artifacts["current_objective_md"]),
            "turn_plan_json": str(self.artifacts["turn_plan_json"]),
            "working_memory_md": str(self.artifacts["working_memory_md"]),
            "checkpoints_jsonl": str(self.artifacts["checkpoints_jsonl"]),
            "knowledge_graph_json": str(self.artifacts["knowledge_graph_json"]),
            "landmarks_json": str(self.artifacts["landmarks_json"]),
            "event_memory_jsonl": str(self.artifacts["event_memory_jsonl"]),
            "session_brief_md": str(self.artifacts["session_brief_md"]),
            "recovery_saves_json": str(self.artifacts["recovery_saves_json"]),
            "run_log_jsonl": str(self.artifacts["run_log_jsonl"]),
        }

    def _next_observation_id(
        self,
        *,
        generated_at: str,
        reason: str,
        state: JsonDict,
    ) -> str:
        position = (state.get("player") or {}).get("position") or {}
        return "obs-" + _stable_id(
            generated_at,
            reason,
            (state.get("map") or {}).get("map_id"),
            (state.get("map") or {}).get("map_name"),
            position.get("x"),
            position.get("y"),
            (state.get("metadata") or {}).get("frame_count"),
        )

    def _evaluate_pending_plan(self, bundle: JsonDict) -> TurnPlan:
        plan = self.load_turn_plan_model()
        if plan.status.state != "executed_waiting_observe":
            return plan
        plan = evaluate_plan_outcome(plan, bundle)
        return self.save_turn_plan(plan)

    def _write_turn_context(self, bundle: JsonDict) -> TurnContext:
        plan = self._evaluate_pending_plan(bundle)
        observation_id = str(bundle.get("observation_id") or "")
        if (
            plan.status.state == "validated"
            and plan.observation_id
            and plan.observation_id != observation_id
        ):
            plan = self.save_turn_plan(
                invalidate_plan(
                    plan,
                    reason="A newer observation was generated before the validated plan executed.",
                    state="stale",
                )
            )
        plan_status = plan.status
        context = build_turn_context(
            bundle=bundle,
            plan_status=plan_status,
            map_id_to_name=RED_MAP_NAMES,
        )
        self._write_json(self.artifacts["turn_context_json"], context.model_dump(mode="json"))
        return context

    def _snapshot_from_navigation_payload(
        self,
        navigation: Optional[JsonDict],
    ) -> Optional[LiveNavigationSnapshot]:
        snapshot = None
        if navigation:
            snapshot_payload = navigation.get("snapshot") or {}
            if snapshot_payload:
                try:
                    snapshot = LiveNavigationSnapshot(
                        map_id=int(snapshot_payload["map_id"]),
                        map_name=str(snapshot_payload["map_name"]),
                        player_position=(
                            int(snapshot_payload["player_position"]["x"]),
                            int(snapshot_payload["player_position"]["y"]),
                        ),
                        facing=str(snapshot_payload.get("facing", "unknown")),
                        tileset=str(snapshot_payload.get("tileset", "UNKNOWN")),
                        window_top_left=(
                            int(snapshot_payload["window_top_left"]["x"]),
                            int(snapshot_payload["window_top_left"]["y"]),
                        ),
                        terrain=list(snapshot_payload.get("terrain", [])),
                        sprite_positions=[
                            (int(item["x"]), int(item["y"]))
                            for item in snapshot_payload.get("sprites", [])
                        ],
                        valid_moves=list(snapshot_payload.get("valid_moves", [])),
                        warps=list(snapshot_payload.get("warps", [])),
                        signs=list(snapshot_payload.get("signs", [])),
                        map_dimensions=snapshot_payload.get("map_dimensions"),
                        interaction=snapshot_payload.get("interaction"),
                    )
                except Exception:  # noqa: BLE001
                    snapshot = None
        return snapshot

    def _coerce_screen_image(self, emulator: Any) -> Image.Image:
        screen = emulator.get_screen()
        if not isinstance(screen, Image.Image):
            screen = Image.fromarray(screen)
        return screen

    def _write_frame_artifacts(
        self,
        *,
        screen: Image.Image,
        annotated: Image.Image,
    ) -> None:
        self.artifacts["latest_frame"].parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        screen.save(buf, format="PNG")
        _atomic_write_bytes(self.artifacts["latest_frame"], buf.getvalue())
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        _atomic_write_bytes(self.artifacts["latest_frame_annotated"], buf.getvalue())

    def sync_live_view(
        self,
        *,
        emulator: Any,
        state: JsonDict,
        navigation: Optional[JsonDict],
        navigation_store: Optional[NavigationStore],
    ) -> JsonDict:
        current_objective = self.objective_engine.evaluate(state)
        screen = self._coerce_screen_image(emulator)
        snapshot = self._snapshot_from_navigation_payload(navigation)
        annotated = render_navigation_overlay(
            screen,
            snapshot,
            objective=current_objective["current"],
            goal=None,
        )
        self._write_frame_artifacts(screen=screen, annotated=annotated)

        previous_bundle = self.live_bundle or self.latest_bundle or {}
        previous_screen_text = previous_bundle.get("screen_text") or {}
        dialog_active = bool(
            state.get("dialog_active") or (state.get("dialog") or {}).get("active")
        )
        preserved_text = ""
        preserved_source = "live_sync"
        if (
            isinstance(previous_screen_text.get("text"), str)
            and previous_screen_text.get("text")
            and bool(previous_screen_text.get("dialog_active")) == dialog_active
        ):
            preserved_text = previous_screen_text["text"]
            preserved_source = "live_sync_cached"
        if not preserved_text:
            preserved_text = "Live frame sync active. Run /agent/observe for refreshed OCR."

        movement_guidance = build_movement_guidance(
            state=state,
            snapshot=snapshot,
            navigation_store=navigation_store,
            objective=current_objective,
        )
        navigation_guidance = self._build_navigation_assistance(
            snapshot=snapshot,
            navigation_store=navigation_store,
            objective=current_objective,
        )
        dialog_guidance = (previous_bundle.get("dialog_guidance") or {}).copy()
        dialog_guidance.setdefault("transcript_recent", [])
        dialog_guidance["should_continue"] = dialog_active
        dialog_guidance.setdefault("last_change_at", self.dialog_last_change_at)
        battle_guidance = self._build_battle_guidance(state, dialog_guidance)
        memory_snapshot = self._build_memory_snapshot(
            objective=current_objective,
            navigation_guidance=navigation_guidance,
            dialog_guidance=dialog_guidance,
            battle_guidance=battle_guidance,
            state=state,
        )

        generated_at = utc_now()
        live_bundle = {
            "generated_at": generated_at,
            "observation_id": (self.latest_bundle or {}).get("observation_id"),
            "reason": "realtime_live_sync",
            "source": "live_sync",
            "artifacts": self._artifact_payload(),
            "state": state,
            "navigation": navigation,
            "screen_text": {
                "text": preserved_text,
                "source": preserved_source,
                "ui_mode": classify_ui_mode(state),
                "dialog_active": dialog_active,
                "note": "Live sync updates visuals and core state between observations.",
            },
            "objective": current_objective,
            "turn_plan": self.load_turn_plan(),
            "plan_status": self.plan_status(),
            "recent_action": previous_bundle.get("recent_action") or {},
            "movement_guidance": movement_guidance,
            "navigation_guidance": navigation_guidance,
            "dialog_guidance": dialog_guidance,
            "battle_guidance": battle_guidance,
            "memory": memory_snapshot,
            "state_delta": previous_bundle.get("state_delta")
            or {
                "changed": False,
                "summary": ["Live frame sync only. Run /agent/observe for refreshed deltas."],
                "movement": None,
            },
            "checkpoints": previous_bundle.get("checkpoints")
            or self._tail_jsonl(self.artifacts["checkpoints_jsonl"], 20),
            "knowledge_graph": self._read_json(
                self.artifacts["knowledge_graph_json"],
                previous_bundle.get("knowledge_graph") or {},
            ),
            "recovery": self._read_json(
                self.artifacts["recovery_saves_json"],
                previous_bundle.get("recovery") or {},
            ),
            "stuck": previous_bundle.get("stuck")
            or {
                "level": "clear",
                "reason": "No stuck signal recorded yet.",
                "recommended_actions": [],
                "objective_action_count": 0,
            },
            "workspace_dir": str(self.workspace_dir),
            "turn_context": self.load_turn_context(),
        }

        self._write_session_brief(
            objective=current_objective,
            navigation_guidance=navigation_guidance,
            memory_snapshot=memory_snapshot,
            dialog_guidance=dialog_guidance,
            battle_guidance=battle_guidance,
            state=state,
        )

        self._write_json(self.artifacts["current_objective_json"], current_objective)
        _atomic_write_text(
            self.artifacts["current_objective_md"],
            self._objective_markdown(current_objective),
        )
        self._write_json(
            self.artifacts["latest_observation_json"],
            self._compact_observation_payload(live_bundle),
        )
        _atomic_write_text(
            self.artifacts["latest_observation_md"],
            self._observation_markdown(bundle=live_bundle),
        )

        self.live_bundle = live_bundle
        return {
            "generated_at": live_bundle["generated_at"],
            "source": live_bundle["source"],
            "artifacts": live_bundle["artifacts"],
            "screen_text": live_bundle["screen_text"],
        }

    def _compact_observation_payload(self, bundle: JsonDict) -> JsonDict:
        navigation = bundle.get("navigation") or {}
        snapshot = navigation.get("snapshot") or {}
        location_map = navigation.get("location_map") or {}
        navigation_guidance = bundle.get("navigation_guidance") or {}
        objective = bundle.get("objective") or {}
        recovery = bundle.get("recovery") or {}
        artifacts = bundle.get("artifacts") or {}
        current_objective = objective.get("current") or {}
        recent_action = bundle.get("recent_action") or {}
        movement_guidance = bundle.get("movement_guidance") or {}
        state_delta = bundle.get("state_delta") or {}
        memory_snapshot = bundle.get("memory") or {}
        dialog_guidance = bundle.get("dialog_guidance") or {}
        battle_guidance = bundle.get("battle_guidance") or {}
        candidate_route = movement_guidance.get("candidate_route") or {}
        live_ascii = _truncate_text_block(snapshot.get("ascii"), 900)
        explored_ascii = _truncate_text_block(location_map.get("ascii"), 1400)
        distance_ascii = _truncate_text_block(navigation_guidance.get("distance_ascii"), 1800)
        return {
            "generated_at": bundle.get("generated_at"),
            "observation_id": bundle.get("observation_id"),
            "reason": bundle.get("reason"),
            "source": bundle.get("source"),
            "artifacts": {
                "latest_frame": artifacts.get("latest_frame"),
                "latest_frame_annotated": artifacts.get("latest_frame_annotated"),
                "turn_context_json": artifacts.get("turn_context_json"),
                "latest_observation_md": artifacts.get("latest_observation_md"),
                "current_objective_json": artifacts.get("current_objective_json"),
                "current_objective_md": artifacts.get("current_objective_md"),
                "turn_plan_json": artifacts.get("turn_plan_json"),
                "working_memory_md": artifacts.get("working_memory_md"),
                "landmarks_json": artifacts.get("landmarks_json"),
                "event_memory_jsonl": artifacts.get("event_memory_jsonl"),
                "session_brief_md": artifacts.get("session_brief_md"),
                "recovery_saves_json": artifacts.get("recovery_saves_json"),
            },
            "state": extract_key_state(bundle.get("state")),
            "screen_text": {
                "text": _truncate_text_block((bundle.get("screen_text") or {}).get("text"), 420),
                "source": (bundle.get("screen_text") or {}).get("source"),
                "ui_mode": (bundle.get("screen_text") or {}).get("ui_mode"),
                "dialog_active": (bundle.get("screen_text") or {}).get("dialog_active"),
            },
            "objective": {
                "current": {
                    "id": current_objective.get("id"),
                    "title": current_objective.get("title"),
                    "summary": current_objective.get("summary"),
                    "completion_predicate": current_objective.get("completion_predicate"),
                    "save_recommendation": current_objective.get("save_recommendation"),
                    "route_hint": current_objective.get("route_hint"),
                    "preferred_landmark_types": current_objective.get("preferred_landmark_types"),
                    "failure_hints": (current_objective.get("failure_hints") or [])[:4],
                    "progress_percent": current_objective.get("progress_percent"),
                    "pack_id": current_objective.get("pack_id"),
                },
                "progress_percent": objective.get("progress_percent"),
                "current_pack_id": objective.get("current_pack_id"),
            },
            "turn_plan": {
                "objective_id": (bundle.get("turn_plan") or {}).get("objective_id"),
                "summary": (bundle.get("turn_plan") or {}).get("summary"),
                "planned_actions": ((bundle.get("turn_plan") or {}).get("planned_actions") or [])[
                    :6
                ],
                "fallback_actions": ((bundle.get("turn_plan") or {}).get("fallback_actions") or [])[
                    :6
                ],
                "notes": _truncate_text_block((bundle.get("turn_plan") or {}).get("notes"), 260),
                "updated_at": (bundle.get("turn_plan") or {}).get("updated_at"),
                "status": (bundle.get("turn_plan") or {}).get("status") or {},
                "mode": (bundle.get("turn_plan") or {}).get("mode"),
                "observation_id": (bundle.get("turn_plan") or {}).get("observation_id"),
            },
            "plan_status": bundle.get("plan_status"),
            "recent_action": {
                "source": recent_action.get("source"),
                "summary": recent_action.get("summary"),
                "notes": (recent_action.get("notes") or [])[:4],
                "tags": recent_action.get("tags"),
            },
            "movement_guidance": {
                "summary": movement_guidance.get("summary"),
                "notes": (movement_guidance.get("notes") or [])[:5],
                "preferred_direction": movement_guidance.get("preferred_direction"),
                "candidate_route": (
                    {
                        "direction": candidate_route.get("direction"),
                        "target": candidate_route.get("target"),
                        "steps": candidate_route.get("steps"),
                        "actions": (candidate_route.get("actions") or [])[:6],
                    }
                    if candidate_route
                    else None
                ),
            },
            "state_delta": {
                "changed": state_delta.get("changed"),
                "summary": (state_delta.get("summary") or [])[:4],
                "movement": state_delta.get("movement"),
            },
            "navigation": {
                "valid_moves": snapshot.get("valid_moves", []),
                "interaction": snapshot.get("interaction"),
                "live_ascii": live_ascii,
                "explored_ascii": explored_ascii,
                "distance_ascii": distance_ascii,
                "frontiers": (navigation_guidance.get("frontiers") or [])[:8],
                "landmarks": (navigation_guidance.get("landmarks") or [])[:8],
                "route_cards": (navigation_guidance.get("route_cards") or [])[:5],
                "avoidances": (navigation_guidance.get("avoidances") or [])[:5],
                "ascii_note": (
                    "ASCII is symbolic only and may be truncated. Use the annotated frame first."
                ),
                "window_top_left": snapshot.get("window_top_left"),
                "window_size": snapshot.get("window_size"),
                "bounds": location_map.get("bounds"),
            },
            "memory": memory_snapshot,
            "dialog": dialog_guidance,
            "battle": battle_guidance,
            "checkpoints": [
                {
                    "id": checkpoint.get("id"),
                    "title": checkpoint.get("title"),
                    "created_at": checkpoint.get("created_at"),
                    "objective_id": checkpoint.get("objective_id"),
                    "map_name": checkpoint.get("map_name"),
                }
                for checkpoint in (bundle.get("checkpoints") or [])[-10:]
            ],
            "knowledge_graph_summary": (bundle.get("knowledge_graph") or {}).get("summary"),
            "recovery": {
                "updated_at": recovery.get("updated_at"),
                "current_recommendation": recovery.get("current_recommendation"),
                "candidates": (recovery.get("candidates") or [])[:3],
            },
            "stuck": bundle.get("stuck"),
        }

    def refresh(
        self,
        *,
        emulator: Any,
        state: JsonDict,
        navigation: Optional[JsonDict],
        navigation_store: Optional[NavigationStore],
        reason: str,
        source: str,
        requested_actions: Optional[list[str]] = None,
        navigation_plan: Optional[JsonDict] = None,
        navigation_execution: Optional[JsonDict] = None,
        explicit_save: Optional[JsonDict] = None,
    ) -> JsonDict:
        current_objective = self.objective_engine.evaluate(state)
        screen = self._coerce_screen_image(emulator)

        goal = None
        if navigation_execution and navigation_execution.get("target"):
            goal_data = navigation_execution["target"]
            if goal_data.get("x") is not None and goal_data.get("y") is not None:
                goal = (int(goal_data["x"]), int(goal_data["y"]))

        snapshot = self._snapshot_from_navigation_payload(navigation)

        annotated = render_navigation_overlay(
            screen,
            snapshot,
            objective=current_objective["current"],
            goal=goal,
        )
        screen_text = extract_screen_text(screen, state)
        state_delta = build_state_delta(self.last_state, state)
        evaluated_plan = self._evaluate_pending_plan(
            {
                "state": state,
                "state_delta": state_delta,
                "screen_text": screen_text,
            }
        )
        plan_execution_trace: Optional[str] = None
        plan_state_label: Optional[str] = None
        if evaluated_plan.execution is not None and evaluated_plan.status.state in {
            "matched",
            "partial",
            "drifted",
            "invalid",
            "stale",
        }:
            plan_state_label = evaluated_plan.status.state
            plan_execution_trace = build_plan_execution_trace(
                evaluated_plan,
                {
                    "state": state,
                    "state_delta": state_delta,
                    "screen_text": screen_text,
                },
            )
        action_feedback = classify_action_feedback(
            source=source,
            requested_actions=requested_actions,
            state_before=self.last_state,
            state_after=state,
            state_delta=state_delta,
            navigation_plan=navigation_plan,
            navigation_execution=navigation_execution,
            plan_execution_trace=plan_execution_trace,
            plan_state=plan_state_label,
        )
        movement_guidance = build_movement_guidance(
            state=state,
            snapshot=snapshot,
            navigation_store=navigation_store,
            objective=current_objective,
        )
        landmark_creations = self._discover_landmarks(snapshot=snapshot)
        navigation_guidance = self._build_navigation_assistance(
            snapshot=snapshot,
            navigation_store=navigation_store,
            objective=current_objective,
        )
        dialog_guidance, dialog_change = self._update_dialog_guidance(
            screen_text=screen_text,
            state=state,
        )
        battle_guidance = self._build_battle_guidance(state, dialog_guidance)
        turn_plan = self.load_turn_plan()
        checkpoints = self._persist_checkpoints(
            self._candidate_checkpoints(state, current_objective)
        )
        knowledge_graph = self._build_knowledge_graph(
            state=state,
            navigation=navigation,
            objective=current_objective,
            navigation_store=navigation_store,
        )
        auto_saves = self._maybe_auto_save(
            emulator=emulator,
            state=state,
            objective=current_objective,
            state_delta=state_delta,
            checkpoints=checkpoints,
            requested_actions=requested_actions,
            source=source,
        )
        if explicit_save:
            auto_saves.append(explicit_save)
        recovery = self._build_recovery(auto_saves)
        stuck = self._detect_stuck(
            state=state,
            objective=current_objective,
            source=source,
            requested_actions=requested_actions,
            plan=evaluated_plan,
        )

        map_name = str((state.get("map") or {}).get("map_name") or "Unknown")
        map_id = int((state.get("map") or {}).get("map_id") or -1)
        map_key = self._map_key(state=state, snapshot=snapshot)
        objective_id = str(current_objective["current"]["id"])
        current_position = _tuple_coord_from_any((state.get("player") or {}).get("position"))

        if current_objective["current"]["id"] != self.last_objective_id:
            self._record_semantic_memory(
                "objective_change",
                f"Objective advanced to {current_objective['current']['title']}.",
                map_key=map_key,
                map_id=map_id,
                map_name=map_name,
                objective_id=objective_id,
                pack_id=current_objective.get("current_pack_id"),
            )

        map_field = (state_delta.get("fields") or {}).get("map")
        if map_field:
            self._record_semantic_memory(
                "map_transition",
                f"Entered {map_field.get('after')}.",
                map_key=map_key,
                map_id=map_id,
                map_name=map_name,
                objective_id=objective_id,
                before=map_field.get("before"),
                after=map_field.get("after"),
            )

        for key in ("has_pokedex", "has_oaks_parcel", "badge_count"):
            if key in (state_delta.get("fields") or {}):
                field = state_delta["fields"][key]
                self._record_semantic_memory(
                    "flag_change",
                    f"{key} changed from {field.get('before')} to {field.get('after')}.",
                    map_key=map_key,
                    map_id=map_id,
                    map_name=map_name,
                    objective_id=objective_id,
                    field=key,
                    before=field.get("before"),
                    after=field.get("after"),
                )

        for summary in (state_delta.get("fields") or {}).get("bag", []):
            self._record_semantic_memory(
                "bag_change",
                summary,
                map_key=map_key,
                map_id=map_id,
                map_name=map_name,
                objective_id=objective_id,
            )

        for landmark in landmark_creations:
            coord = _tuple_coord_from_any(landmark.get("coord"))
            self._record_semantic_memory(
                "landmark_discovery",
                f"Learned landmark: {landmark.get('title')}.",
                map_key=map_key,
                map_id=map_id,
                map_name=map_name,
                objective_id=objective_id,
                landmark_id=landmark.get("id"),
                landmark_kind=landmark.get("kind"),
                coord=_coord_payload(coord),
            )

        for checkpoint in checkpoints:
            self._record_semantic_memory(
                "checkpoint",
                checkpoint.get("summary") or checkpoint.get("title") or "Checkpoint reached.",
                map_key=map_key,
                map_id=map_id,
                map_name=map_name,
                objective_id=objective_id,
                checkpoint_id=checkpoint.get("id"),
            )

        if dialog_change is not None:
            self._record_semantic_memory(
                "dialog_text",
                f"Dialog updated: {dialog_change['text']}",
                map_key=map_key,
                map_id=map_id,
                map_name=map_name,
                objective_id=objective_id,
                text=dialog_change["text"],
            )

        for save_event in auto_saves:
            self._record_semantic_memory(
                "save_point",
                f"Save recorded: {save_event.get('name')}.",
                map_key=map_key,
                map_id=map_id,
                map_name=map_name,
                objective_id=objective_id,
                save_name=save_event.get("name"),
                reason=save_event.get("reason"),
            )

        failure_coord = None
        if snapshot is not None:
            failure_coord = _tuple_coord_from_any((snapshot.interaction or {}).get("target_coord"))
        failure_coord = failure_coord or current_position

        if source == "navigation" and navigation_execution and not navigation_execution.get("success"):
            self._record_semantic_memory(
                "failure",
                navigation_execution.get("status") or "Navigation route failed.",
                map_key=map_key,
                map_id=map_id,
                map_name=map_name,
                objective_id=objective_id,
                type="navigation_partial",
                coord=_coord_payload(failure_coord),
                coord_key=_coord_to_key(failure_coord) if failure_coord else None,
                reason=navigation_execution.get("status"),
                actions=requested_actions or [],
            )
        elif requested_actions and "no_progress" in (action_feedback.get("tags") or []):
            self._record_semantic_memory(
                "failure",
                "Repeated actions produced no structured progress.",
                map_key=map_key,
                map_id=map_id,
                map_name=map_name,
                objective_id=objective_id,
                type="no_progress",
                coord=_coord_payload(failure_coord),
                coord_key=_coord_to_key(failure_coord) if failure_coord else None,
                reason=action_feedback.get("summary"),
                actions=requested_actions or [],
            )

        if stuck["level"] != "clear":
            failure_type = "dialog_loop" if "dialog loop" in stuck["reason"].lower() else "stuck"
            self._record_semantic_memory(
                "failure",
                stuck["reason"],
                map_key=map_key,
                map_id=map_id,
                map_name=map_name,
                objective_id=objective_id,
                type=failure_type,
                coord=_coord_payload(failure_coord),
                coord_key=_coord_to_key(failure_coord) if failure_coord else None,
                reason=stuck["reason"],
                actions=requested_actions or [],
            )

        memory_snapshot = self._build_memory_snapshot(
            objective=current_objective,
            navigation_guidance=navigation_guidance,
            dialog_guidance=dialog_guidance,
            battle_guidance=battle_guidance,
            state=state,
        )

        self._write_frame_artifacts(screen=screen, annotated=annotated)
        self._write_session_brief(
            objective=current_objective,
            navigation_guidance=navigation_guidance,
            memory_snapshot=memory_snapshot,
            dialog_guidance=dialog_guidance,
            battle_guidance=battle_guidance,
            state=state,
        )

        generated_at = utc_now()
        observation_id = self._next_observation_id(
            generated_at=generated_at,
            reason=reason,
            state=state,
        )
        bundle = {
            "generated_at": generated_at,
            "observation_id": observation_id,
            "reason": reason,
            "source": source,
            "artifacts": self._artifact_payload(),
            "state": state,
            "navigation": navigation,
            "screen_text": screen_text,
            "objective": current_objective,
            "turn_plan": turn_plan,
            "plan_status": self.plan_status(),
            "recent_action": action_feedback,
            "movement_guidance": movement_guidance,
            "navigation_guidance": navigation_guidance,
            "dialog_guidance": dialog_guidance,
            "battle_guidance": battle_guidance,
            "memory": memory_snapshot,
            "state_delta": state_delta,
            "checkpoints": self._tail_jsonl(self.artifacts["checkpoints_jsonl"], 20),
            "knowledge_graph": knowledge_graph,
            "recovery": recovery,
            "stuck": stuck,
            "workspace_dir": str(self.workspace_dir),
        }

        self._write_json(self.artifacts["current_objective_json"], current_objective)
        _atomic_write_text(
            self.artifacts["current_objective_md"],
            self._objective_markdown(current_objective),
        )
        turn_context = self._write_turn_context(bundle)
        bundle["turn_context"] = turn_context.model_dump(mode="json")
        bundle["plan_status"] = turn_context.plan_status.model_dump(mode="json")
        bundle["turn_plan"] = self.load_turn_plan()
        turn_plan_hash = json.dumps(bundle["turn_plan"], sort_keys=True)
        self._write_json(
            self.artifacts["latest_observation_json"],
            self._compact_observation_payload(bundle),
        )
        _atomic_write_text(
            self.artifacts["latest_observation_md"],
            self._observation_markdown(bundle=bundle),
        )

        emitted_events: list[JsonDict] = []
        emitted_events.append(
            self._record_event(
                "observe",
                {
                    "reason": reason,
                    "source": source,
                    "objective_id": current_objective["current"]["id"],
                    "summary": action_feedback["summary"],
                },
            )
        )
        emitted_events.append(
            self._record_event(
                "ocr",
                {
                    "source": screen_text["source"],
                    "text": screen_text["text"],
                },
            )
        )

        if current_objective["current"]["id"] != self.last_objective_id:
            emitted_events.append(
                self._record_event(
                    "objective",
                    {
                        "objective": current_objective["current"],
                        "progress_percent": current_objective["progress_percent"],
                    },
                )
            )

        for checkpoint in checkpoints:
            emitted_events.append(self._record_event("checkpoint", checkpoint))

        emitted_events.append(
            self._record_event(
                "knowledge",
                {
                    "summary": knowledge_graph["summary"],
                },
            )
        )

        for save_event in auto_saves:
            emitted_events.append(self._record_event("save", save_event))

        emitted_events.append(
            self._record_event(
                "recovery",
                {
                    "current_recommendation": recovery.get("current_recommendation"),
                    "candidate_count": len(recovery.get("candidates", [])),
                },
            )
        )

        if stuck["level"] != "clear":
            emitted_events.append(self._record_event("stuck", stuck))

        if turn_plan_hash != self.last_turn_plan_hash:
            emitted_events.append(
                self._record_event(
                    "turn_plan_updated",
                    {
                        "turn_plan": bundle["turn_plan"],
                    },
                )
            )
            self.last_turn_plan_hash = turn_plan_hash

        self.latest_bundle = bundle
        self.live_bundle = bundle
        self.last_state = state
        self.last_objective_id = current_objective["current"]["id"]
        return {
            "bundle": bundle,
            "events": emitted_events,
        }

    def dashboard_state(self) -> JsonDict:
        bundle = (
            self.live_bundle
            or self.latest_bundle
            or self._read_json(self.artifacts["latest_observation_json"], {})
        )
        if not bundle:
            return {
                "generated_at": utc_now(),
                "visuals": {},
                "agent_intent": {},
                "world_state": {},
                "memory_and_progress": {},
                "timeline": self.history(50),
            }

        state = bundle.get("state") or {}
        navigation = bundle.get("navigation") or {}
        current_objective = (bundle.get("objective") or {}).get("current") or {}
        turn_plan = bundle.get("turn_plan") or {}
        turn_context = bundle.get("turn_context") or self.load_turn_context()
        recovery = bundle.get("recovery") or {}
        knowledge_graph = bundle.get("knowledge_graph") or {}
        memory_snapshot = bundle.get("memory") or {}
        return {
            "observation_id": bundle.get("observation_id") or turn_context.get("observation_id"),
            "generated_at": bundle.get("generated_at"),
            "visuals": {
                "raw_frame_path": (bundle.get("artifacts") or {}).get("latest_frame"),
                "annotated_frame_path": (bundle.get("artifacts") or {}).get(
                    "latest_frame_annotated"
                ),
                "frame_timestamp": bundle.get("generated_at"),
                "ui_mode": (bundle.get("screen_text") or {}).get("ui_mode"),
                "screen_text": bundle.get("screen_text"),
            },
            "agent_intent": {
                "objective": current_objective,
                "turn_context": turn_context,
                "turn_plan": turn_plan,
                "plan_status": bundle.get("plan_status") or turn_context.get("plan_status") or {},
                "recent_action": bundle.get("recent_action"),
                "movement_guidance": bundle.get("movement_guidance"),
                "navigation_guidance": bundle.get("navigation_guidance"),
                "dialog_guidance": bundle.get("dialog_guidance"),
                "battle_guidance": bundle.get("battle_guidance"),
                "state_delta": bundle.get("state_delta"),
            },
            "world_state": {
                "map": state.get("map"),
                "player": state.get("player"),
                "party": state.get("party"),
                "battle": state.get("battle"),
                "dialog": state.get("dialog"),
                "interaction": state.get("interaction")
                or (navigation.get("snapshot") or {}).get("interaction"),
                "valid_moves": (navigation.get("snapshot") or {}).get("valid_moves", []),
                "live_ascii": (navigation.get("snapshot") or {}).get("ascii"),
                "explored_ascii": (navigation.get("location_map") or {}).get("ascii"),
                "navigation": navigation,
            },
            "memory_and_progress": {
                "progress_percent": (bundle.get("objective") or {}).get("progress_percent"),
                "checkpoints": bundle.get("checkpoints"),
                "knowledge_graph_summary": knowledge_graph.get("summary"),
                "memory": memory_snapshot,
                "recovery": recovery,
                "stuck": bundle.get("stuck"),
                "workspace": {
                    "workspace_dir": bundle.get("workspace_dir"),
                    "turn_context_json": (bundle.get("artifacts") or {}).get("turn_context_json"),
                    "turn_plan_json": (bundle.get("artifacts") or {}).get("turn_plan_json"),
                    "recovery_saves_json": (bundle.get("artifacts") or {}).get(
                        "recovery_saves_json"
                    ),
                },
            },
            "timeline": self.history(80),
            "artifacts": bundle.get("artifacts") or {},
        }

    def navigator_payload(self) -> JsonDict:
        bundle = (
            self.live_bundle
            or self.latest_bundle
            or self._read_json(self.artifacts["latest_observation_json"], {})
        )
        navigation_guidance = bundle.get("navigation_guidance") or (bundle.get("navigation") or {})
        route_cards = list(navigation_guidance.get("route_cards") or [])
        best_route = route_cards[0] if route_cards else None
        return {
            "generated_at": bundle.get("generated_at"),
            "reason": bundle.get("reason"),
            "objective": (bundle.get("objective") or {}).get("current"),
            "map": (bundle.get("state") or {}).get("map"),
            "player": (bundle.get("state") or {}).get("player"),
            "best_route": best_route,
            "alternatives": route_cards[1:5],
            "avoidances": (navigation_guidance.get("avoidances") or [])[:5],
            "landmarks": (navigation_guidance.get("landmarks") or [])[:8],
            "memory": bundle.get("memory") or {},
            "dialog": bundle.get("dialog_guidance") or bundle.get("dialog") or {},
            "battle": bundle.get("battle_guidance") or bundle.get("battle") or {},
            "artifacts": {
                "turn_context_json": (bundle.get("artifacts") or {}).get("turn_context_json"),
                "turn_plan_json": (bundle.get("artifacts") or {}).get("turn_plan_json"),
                "recovery_saves_json": (bundle.get("artifacts") or {}).get("recovery_saves_json"),
            },
        }
