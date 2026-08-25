"""Agent workspace, telemetry, and observation runtime for Pokemon Agent."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from PIL import Image, ImageDraw, ImageFont

from pokemon_agent.navigation import LiveNavigationSnapshot

JsonDict = dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _measure_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    if not text:
        return 0, 0
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return draw.textsize(text, font=font)


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = str(text or "").split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if _measure_text(draw, candidate, font=font)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


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
    "Fire": {
        "Grass": 2.0,
        "Bug": 2.0,
        "Ice": 2.0,
        "Water": 0.5,
        "Rock": 0.5,
        "Fire": 0.5,
        "Dragon": 0.5,
    },
    "Water": {"Fire": 2.0, "Ground": 2.0, "Rock": 2.0, "Water": 0.5, "Grass": 0.5, "Dragon": 0.5},
    "Grass": {
        "Water": 2.0,
        "Ground": 2.0,
        "Rock": 2.0,
        "Fire": 0.5,
        "Grass": 0.5,
        "Poison": 0.5,
        "Flying": 0.5,
        "Bug": 0.5,
        "Dragon": 0.5,
    },
    "Electric": {
        "Water": 2.0,
        "Flying": 2.0,
        "Electric": 0.5,
        "Grass": 0.5,
        "Dragon": 0.5,
        "Ground": 0.0,
    },
    "Ice": {"Grass": 2.0, "Ground": 2.0, "Flying": 2.0, "Dragon": 2.0, "Water": 0.5, "Ice": 0.5},
    "Fighting": {
        "Normal": 2.0,
        "Rock": 2.0,
        "Ice": 2.0,
        "Poison": 0.5,
        "Flying": 0.5,
        "Psychic": 0.5,
        "Bug": 0.5,
        "Ghost": 0.0,
    },
    "Poison": {"Grass": 2.0, "Bug": 2.0, "Poison": 0.5, "Ground": 0.5, "Rock": 0.5, "Ghost": 0.5},
    "Ground": {
        "Fire": 2.0,
        "Electric": 2.0,
        "Poison": 2.0,
        "Rock": 2.0,
        "Grass": 0.5,
        "Bug": 0.5,
        "Flying": 0.0,
    },
    "Flying": {"Grass": 2.0, "Fighting": 2.0, "Bug": 2.0, "Electric": 0.5, "Rock": 0.5},
    "Psychic": {"Fighting": 2.0, "Poison": 2.0, "Psychic": 0.5},
    "Bug": {
        "Grass": 2.0,
        "Poison": 2.0,
        "Psychic": 2.0,
        "Fire": 0.5,
        "Fighting": 0.5,
        "Flying": 0.5,
        "Ghost": 0.5,
    },
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


def build_movement_guidance(
    *,
    snapshot: Optional[LiveNavigationSnapshot],
) -> JsonDict:
    """Describe the legal moves visible in the current collision window."""
    if snapshot is None:
        return {
            "summary": "No live collision window was captured for this frame.",
            "notes": [],
            "preferred_direction": None,
        }

    notes: list[str] = [f"Immediate legal moves: {', '.join(snapshot.valid_moves) or 'none'}."]
    interaction = snapshot.interaction or {}
    if interaction.get("source") == "blocked_tile":
        target = interaction.get("target_coord") or {}
        notes.append(f"Forward movement is blocked at ({target.get('x')}, {target.get('y')}).")
        sidesteps = [move for move in snapshot.valid_moves if move != snapshot.facing]
        if sidesteps:
            notes.append(f"Because forward is blocked, sidestep first: {', '.join(sidesteps)}.")

    return {
        "summary": notes[-1],
        "notes": notes,
        "preferred_direction": {
            "up": "north",
            "down": "south",
            "left": "west",
            "right": "east",
        }.get(snapshot.facing),
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
        notes.append("Fresh observation generated.")

    if not tags:
        tags.append("observe")

    return {
        "source": source,
        "requested_actions": requested_actions,
        "summary": notes[0] if notes else "",
        "notes": notes,
        "tags": tags,
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


def classify_ui_state(state: JsonDict) -> JsonDict:
    dialog = state.get("dialog") or {}
    dialog_active = bool(state.get("dialog_active") or dialog.get("active"))
    ui_mode = classify_ui_mode(state)
    text = ""
    source = "vision"
    if dialog_active:
        waiting = "waiting for input" if dialog.get("waiting_for_input") else "printing"
        text = f"Dialog box visible ({waiting})."
        source = "dialog_state"
    return {
        "text": text,
        "source": source,
        "dialog_active": dialog_active,
        "ui_mode": ui_mode,
    }


# Mini-map inset palette. Same language as the main tile grid: unknown ground is
# near-black, passable green, walked dimmed green, wall red, warp purple, player
# cyan. Only the scale changes.
_INSET_MAX_WIDTH = 176
_INSET_MAX_HEIGHT = 320
_INSET_MIN_CELL = 3
_INSET_MAX_CELL = 10
_INSET_UNKNOWN = (9, 12, 19, 255)
_INSET_SEEN = (96, 220, 158, 255)
_INSET_WALKED = (26, 96, 62, 255)
_INSET_WALL = (176, 58, 58, 255)
_INSET_WARP = (213, 80, 255, 255)
_INSET_PLAYER = (55, 208, 255, 255)
_INSET_BORDER = (86, 102, 122, 255)


def _normalise_map_grid(grid: Any) -> Optional[JsonDict]:
    """Coerce an explored-map grid payload into plain sets, or None if unusable."""
    if not isinstance(grid, dict):
        return None
    try:
        width = int(grid.get("width") or 0)
        height = int(grid.get("height") or 0)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0 or width > 512 or height > 512:
        return None

    layers: JsonDict = {"width": width, "height": height}
    for name in ("seen", "walkable", "walked", "warps"):
        tiles: set[tuple[int, int]] = set()
        for item in grid.get(name) or ():
            try:
                tile_x, tile_y = int(item[0]), int(item[1])
            except (TypeError, ValueError, IndexError, KeyError):
                continue
            if 0 <= tile_x < width and 0 <= tile_y < height:
                tiles.add((tile_x, tile_y))
        layers[name] = tiles

    if not (layers["seen"] | layers["walkable"] | layers["walked"] | layers["warps"]):
        return None
    return layers


def _render_map_inset(
    grid: JsonDict,
    *,
    player: Optional[tuple[int, int]],
) -> Image.Image:
    """Draw the whole explored map as a small image with the player marked."""
    width = int(grid["width"])
    height = int(grid["height"])
    cell = min(_INSET_MAX_WIDTH // width, _INSET_MAX_HEIGHT // height, _INSET_MAX_CELL)
    cell = max(cell, _INSET_MIN_CELL)

    map_width = width * cell
    map_height = height * cell
    border = 1
    inset = Image.new("RGBA", (map_width + (border * 2), map_height + (border * 2)), _INSET_BORDER)
    draw = ImageDraw.Draw(inset)
    draw.rectangle(
        (border, border, border + map_width - 1, border + map_height - 1),
        fill=_INSET_UNKNOWN,
    )

    seen = grid["seen"]
    walkable = grid["walkable"]
    walked = grid["walked"]
    warps = grid["warps"]
    for tile in seen | walkable | walked | warps:
        tile_x, tile_y = tile
        if tile in warps:
            fill = _INSET_WARP
        elif tile in walked:
            fill = _INSET_WALKED
        elif tile in walkable:
            fill = _INSET_SEEN
        else:
            fill = _INSET_WALL
        left = border + (tile_x * cell)
        top = border + (tile_y * cell)
        draw.rectangle((left, top, left + cell - 1, top + cell - 1), fill=fill)

    if player is None:
        return inset
    player_x, player_y = int(player[0]), int(player[1])
    if not (0 <= player_x < width and 0 <= player_y < height):
        return inset

    # The player marker is the one pixel that has to survive downscaling and a
    # glance: full-width crosshair, dark halo, solid cyan block, white ring.
    blend = ImageDraw.Draw(inset, "RGBA")
    centre_x = border + (player_x * cell) + (cell // 2)
    centre_y = border + (player_y * cell) + (cell // 2)
    blend.line((border, centre_y, border + map_width - 1, centre_y), fill=(55, 208, 255, 120))
    blend.line((centre_x, border, centre_x, border + map_height - 1), fill=(55, 208, 255, 120))
    half = max(3, min(cell, 6))
    blend.rectangle(
        (centre_x - half - 2, centre_y - half - 2, centre_x + half + 2, centre_y + half + 2),
        fill=(7, 10, 16, 225),
    )
    blend.rectangle(
        (centre_x - half, centre_y - half, centre_x + half, centre_y + half),
        fill=_INSET_PLAYER,
    )
    blend.rectangle(
        (centre_x - half - 2, centre_y - half - 2, centre_x + half + 2, centre_y + half + 2),
        outline=(255, 255, 255, 240),
        width=1,
    )
    return inset


def render_navigation_overlay(
    image: Image.Image,
    snapshot: Optional[LiveNavigationSnapshot],
    *,
    objective: Optional[JsonDict] = None,
    goal: Optional[tuple[int, int]] = None,
    visited: Optional[set[tuple[int, int]]] = None,
    map_grid: Optional[JsonDict] = None,
) -> Image.Image:
    scale = 2
    frame = image.convert("RGBA").resize(
        (image.width * scale, image.height * scale),
        resample=Image.NEAREST,
    )
    font = ImageFont.load_default()
    measure_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
    padding = 8
    line_height = max(_measure_text(measure_draw, "Ag", font=font)[1], 11) + 3

    if not snapshot or not snapshot.width or not snapshot.height:
        canvas_width = frame.width + (padding * 2)
        wrap_width = max(40, canvas_width - (padding * 2))
        header_lines = [
            (line, (255, 255, 255, 255))
            for line in _wrap_text(
                measure_draw,
                "Navigation overlay unavailable.",
                font=font,
                max_width=wrap_width,
            )
        ]
        header_lines.extend(
            (
                line,
                (165, 180, 196, 255),
            )
            for line in _wrap_text(
                measure_draw,
                "No live collision window was captured for this frame.",
                font=font,
                max_width=wrap_width,
            )
        )
        header_height = padding + (len(header_lines) * line_height) + padding
        canvas = Image.new(
            "RGBA",
            (canvas_width, header_height + frame.height + padding),
            (7, 10, 16, 255),
        )
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, canvas.width, header_height), fill=(12, 17, 26, 235))
        canvas.alpha_composite(frame, (padding, header_height))
        draw.rectangle(
            (
                padding - 1,
                header_height - 1,
                padding + frame.width,
                header_height + frame.height,
            ),
            outline=(255, 138, 61, 220),
            width=1,
        )

        text_y = padding
        for line, fill in header_lines:
            draw.text((padding, text_y), line, fill=fill, font=font)
            text_y += line_height
        return canvas.convert("RGB")

    window_min_x = snapshot.window_top_left[0]
    window_max_x = snapshot.window_top_left[0] + snapshot.width - 1
    window_min_y = snapshot.window_top_left[1]
    window_max_y = snapshot.window_top_left[1] + snapshot.height - 1
    visited_locals: set[tuple[int, int]] = set()
    for tile_x, tile_y in visited or ():
        visited_local = snapshot.absolute_to_local(int(tile_x), int(tile_y))
        if visited_local is not None:
            visited_locals.add(visited_local)

    x_labels = [str(window_min_x + local_x) for local_x in range(snapshot.width)]
    y_labels = [str(window_min_y + local_y) for local_y in range(snapshot.height)]
    x_label_height = max(
        (_measure_text(measure_draw, label, font=font)[1] for label in x_labels),
        default=0,
    )
    y_label_width = max(
        (_measure_text(measure_draw, label, font=font)[0] for label in y_labels),
        default=0,
    )

    left_margin = y_label_width + (padding * 2)
    canvas_width = left_margin + frame.width + padding
    wrap_width = max(40, canvas_width - (padding * 2))
    pos = snapshot.player_position
    move_list = ", ".join(snapshot.valid_moves) or "none"
    objective_line = objective["summary"] if objective else "No objective"
    header_blocks = [
        (snapshot.map_name, (255, 255, 255, 255)),
        (
            f"Player ({pos[0]}, {pos[1]}) facing {snapshot.facing} | moves: {move_list}",
            (165, 180, 196, 255),
        ),
        (f"Objective: {objective_line}", (255, 214, 10, 255)),
        (
            (
                "Coords are absolute map tiles. "
                f"Columns show x={window_min_x}..{window_max_x}; "
                f"rows show y={window_min_y}..{window_max_y}. "
                "North is up: walk_up decreases y, walk_down increases y."
            ),
            (110, 230, 174, 255),
        ),
    ]
    if visited_locals:
        header_blocks.append(
            (
                "Dimmed tiles with a grey dot are ground you already walked "
                f"({len(visited_locals)} of {snapshot.width * snapshot.height} in view).",
                (165, 180, 196, 255),
            )
        )
    header_lines: list[tuple[str, tuple[int, int, int, int]]] = []
    for text, fill in header_blocks:
        for line in _wrap_text(measure_draw, text, font=font, max_width=wrap_width):
            header_lines.append((line, fill))

    column_band_height = x_label_height + padding + 2
    header_height = padding + (len(header_lines) * line_height) + padding
    top_margin = header_height + column_band_height

    # Side panel: the whole explored map, drawn to the right of the game window
    # so it covers neither the frame nor the header. Absent grid, absent panel,
    # and the canvas is exactly what it has always been.
    normalised_grid = _normalise_map_grid(map_grid)
    inset: Optional[Image.Image] = None
    panel_width = 0
    panel_title = ""
    panel_caption: list[str] = []
    if normalised_grid is not None:
        inset = _render_map_inset(normalised_grid, player=pos)
        known = (
            normalised_grid["seen"]
            | normalised_grid["walkable"]
            | normalised_grid["walked"]
            | normalised_grid["warps"]
        )
        panel_title = "MINI-MAP: whole map so far"
        caption_width = max(inset.width, 132)
        caption_blocks = [
            (
                f"{normalised_grid['width']}x{normalised_grid['height']} tiles, "
                f"{len(known)} seen, {len(normalised_grid['walked'])} walked."
            ),
            (
                "Cyan box with crosshair is you. Near-black is map you have not seen. "
                "Green passable, dim green walked, red wall, purple warp."
            ),
        ]
        for block in caption_blocks:
            panel_caption.extend(
                _wrap_text(measure_draw, block, font=font, max_width=caption_width)
            )
        panel_content_width = max(
            inset.width,
            _measure_text(measure_draw, panel_title, font=font)[0],
            max(
                (_measure_text(measure_draw, line, font=font)[0] for line in panel_caption),
                default=0,
            ),
        )
        panel_width = panel_content_width + (padding * 2)

    canvas_height = top_margin + frame.height + padding
    if inset is not None:
        canvas_height = max(
            canvas_height,
            top_margin
            + line_height
            + inset.height
            + 4
            + (len(panel_caption) * line_height)
            + padding,
        )
    canvas = Image.new(
        "RGBA",
        (canvas_width + panel_width, canvas_height),
        (7, 10, 16, 255),
    )
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, header_height), fill=(12, 17, 26, 235))
    draw.rectangle((0, header_height, canvas.width, top_margin), fill=(12, 17, 26, 220))
    draw.rectangle((0, top_margin, left_margin, canvas.height), fill=(12, 17, 26, 220))
    if inset is not None:
        panel_left = canvas_width
        draw.rectangle(
            (panel_left, header_height, canvas.width, canvas.height),
            fill=(12, 17, 26, 220),
        )
        panel_y = top_margin
        draw.text(
            (panel_left + padding, panel_y), panel_title, fill=(255, 255, 255, 255), font=font
        )
        panel_y += line_height
        canvas.alpha_composite(inset, (panel_left + padding, panel_y))
        panel_y += inset.height + 4
        for line in panel_caption:
            draw.text((panel_left + padding, panel_y), line, fill=(165, 180, 196, 255), font=font)
            panel_y += line_height
    canvas.alpha_composite(frame, (left_margin, top_margin))
    draw.rectangle(
        (
            left_margin - 1,
            top_margin - 1,
            left_margin + frame.width,
            top_margin + frame.height,
        ),
        outline=(255, 138, 61, 220),
        width=1,
    )

    text_y = padding
    for line, fill in header_lines:
        draw.text((padding, text_y), line, fill=fill, font=font)
        text_y += line_height

    tile_width = frame.width / snapshot.width
    tile_height = frame.height / snapshot.height
    grid_line_width = max(1, scale)

    # Tiles whose centre already carries a letter (P, W, G); the walked dot is
    # skipped there so the glyph stays readable. The dimmed fill still shows.
    glyph_locals: set[tuple[int, int]] = {(4, 4)}
    if goal is not None:
        goal_glyph = snapshot.absolute_to_local(goal[0], goal[1])
        if goal_glyph is not None:
            glyph_locals.add(goal_glyph)
    for warp in snapshot.warps:
        if not isinstance(warp, dict):
            continue
        warp_x = warp.get("x")
        warp_y = warp.get("y")
        if warp_x is None or warp_y is None:
            continue
        warp_glyph = snapshot.absolute_to_local(int(warp_x), int(warp_y))
        if warp_glyph is not None:
            glyph_locals.add(warp_glyph)
    visited_dot_radius = max(2, scale * 2)

    for local_y, row in enumerate(snapshot.terrain):
        for local_x, tile in enumerate(row):
            left = int(left_margin + (local_x * tile_width))
            top = int(top_margin + (local_y * tile_height))
            right = int(left_margin + ((local_x + 1) * tile_width))
            bottom = int(top_margin + ((local_y + 1) * tile_height))
            walked = (local_x, local_y) in visited_locals
            if tile:
                fill = (13, 68, 41, 96) if walked else (24, 123, 73, 72)
                outline = (74, 156, 118, 190) if walked else (110, 230, 174, 190)
            else:
                fill = (108, 35, 35, 110) if walked else (180, 58, 58, 92)
                outline = (206, 96, 96, 200) if walked else (255, 120, 120, 200)
            draw.rectangle((left, top, right, bottom), outline=outline, fill=fill, width=1)
            if walked and (local_x, local_y) not in glyph_locals:
                dot_x = int(left_margin + ((local_x + 0.5) * tile_width))
                dot_y = int(top_margin + ((local_y + 0.5) * tile_height))
                draw.ellipse(
                    (
                        dot_x - visited_dot_radius,
                        dot_y - visited_dot_radius,
                        dot_x + visited_dot_radius,
                        dot_y + visited_dot_radius,
                    ),
                    fill=(203, 213, 225, 235),
                )

    for local_x, label in enumerate(x_labels):
        label_width, label_height = _measure_text(measure_draw, label, font=font)
        x_center = int(left_margin + ((local_x + 0.5) * tile_width))
        draw.text(
            (x_center - (label_width // 2), header_height + 2),
            label,
            fill=(255, 255, 255, 255),
            font=font,
        )

    for local_y, label in enumerate(y_labels):
        label_width, label_height = _measure_text(measure_draw, label, font=font)
        y_center = int(top_margin + ((local_y + 0.5) * tile_height))
        draw.text(
            (
                left_margin - padding - label_width,
                y_center - (label_height // 2),
            ),
            label,
            fill=(255, 255, 255, 255),
            font=font,
        )

    for sprite_x, sprite_y in snapshot.sprite_positions:
        local = snapshot.absolute_to_local(sprite_x, sprite_y)
        if local is None:
            continue
        left = int(left_margin + (local[0] * tile_width))
        top = int(top_margin + (local[1] * tile_height))
        right = int(left_margin + ((local[0] + 1) * tile_width))
        bottom = int(top_margin + ((local[1] + 1) * tile_height))
        inset = max(4, scale * 3)
        draw.rectangle(
            (left + inset, top + inset, right - inset, bottom - inset),
            fill=(255, 174, 66, 190),
        )

    for warp in snapshot.warps:
        wx = warp.get("x") if isinstance(warp, dict) else None
        wy = warp.get("y") if isinstance(warp, dict) else None
        if wx is None or wy is None:
            continue
        warp_local = snapshot.absolute_to_local(int(wx), int(wy))
        if warp_local is None:
            continue
        left = int(left_margin + (warp_local[0] * tile_width))
        top = int(top_margin + (warp_local[1] * tile_height))
        right = int(left_margin + ((warp_local[0] + 1) * tile_width))
        bottom = int(top_margin + ((warp_local[1] + 1) * tile_height))
        draw.rectangle(
            (left + 1, top + 1, right - 1, bottom - 1),
            outline=(213, 80, 255, 255),
            width=grid_line_width + 1,
        )
        warp_label_width, warp_label_height = _measure_text(measure_draw, "W", font=font)
        draw.text(
            (
                left + int((tile_width - warp_label_width) / 2),
                top + int((tile_height - warp_label_height) / 2),
            ),
            "W",
            fill=(213, 80, 255, 255),
            font=font,
        )

    player_left = int(left_margin + (4 * tile_width))
    player_top = int(top_margin + (4 * tile_height))
    player_right = int(left_margin + (5 * tile_width))
    player_bottom = int(top_margin + (5 * tile_height))
    draw.rectangle(
        (player_left + 2, player_top + 2, player_right - 2, player_bottom - 2),
        outline=(55, 208, 255, 255),
        width=grid_line_width + 1,
    )
    player_label_width, player_label_height = _measure_text(measure_draw, "P", font=font)
    draw.text(
        (
            player_left + int((tile_width - player_label_width) / 2),
            player_top + int((tile_height - player_label_height) / 2),
        ),
        "P",
        fill=(55, 208, 255, 255),
        font=font,
    )

    if goal is not None:
        goal_local = snapshot.absolute_to_local(goal[0], goal[1])
        if goal_local is not None:
            left = int(left_margin + (goal_local[0] * tile_width))
            top = int(top_margin + (goal_local[1] * tile_height))
            right = int(left_margin + ((goal_local[0] + 1) * tile_width))
            bottom = int(top_margin + ((goal_local[1] + 1) * tile_height))
            draw.rectangle(
                (left + 2, top + 2, right - 2, bottom - 2),
                outline=(255, 214, 10, 255),
                width=grid_line_width + 1,
            )
            goal_label_width, goal_label_height = _measure_text(measure_draw, "G", font=font)
            draw.text(
                (
                    left + int((tile_width - goal_label_width) / 2),
                    top + int((tile_height - goal_label_height) / 2),
                ),
                "G",
                fill=(255, 214, 10, 255),
                font=font,
            )

    interaction = snapshot.interaction or {}
    target_coord = interaction.get("target_coord") or {}
    if target_coord.get("x") is not None and target_coord.get("y") is not None:
        local = snapshot.absolute_to_local(int(target_coord["x"]), int(target_coord["y"]))
        if local is not None:
            left = int(left_margin + (local[0] * tile_width))
            top = int(top_margin + (local[1] * tile_height))
            right = int(left_margin + ((local[0] + 1) * tile_width))
            bottom = int(top_margin + ((local[1] + 1) * tile_height))
            inset = max(4, scale * 3)
            draw.ellipse(
                (left + inset, top + inset, right - inset, bottom - inset),
                outline=(255, 125, 0, 255),
                width=grid_line_width + 1,
            )

    return canvas.convert("RGB")


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
            if _selector_matches(item.get("selector") or {}, state):
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


class AgentRuntime:
    """Owns workspace artifacts, telemetry history, and deterministic assist logic."""

    def __init__(
        self,
        *,
        data_dir: Path,
        workspace_dir: Path,
        objective_engine: Optional[ObjectiveEngine] = None,
        history_limit: int = 400,
        visited_lookup: Optional[Callable[[int], set[tuple[int, int]]]] = None,
        map_grid_lookup: Optional[Callable[[int], Optional[JsonDict]]] = None,
    ) -> None:
        self.data_dir = data_dir.expanduser().resolve()
        self.workspace_dir = workspace_dir.expanduser().resolve()
        self.objective_engine = objective_engine or ObjectiveEngine()
        self.history_limit = history_limit
        # SEAM: the server sets this to ExploredMaps.visited (map_id -> absolute
        # tiles the player has stood on) so the overlay can shade walked ground.
        # Left as None the overlay renders exactly as before.
        self.visited_lookup = visited_lookup
        # SEAM: the server sets this to ExploredMaps.grid (map_id -> the whole
        # stored map as width/height plus seen/walkable/walked/warp tile sets)
        # so the overlay can inset a mini-map. Left as None the overlay renders
        # exactly as before.
        self.map_grid_lookup = map_grid_lookup
        self.event_history: deque[JsonDict] = deque(maxlen=history_limit)
        self.recent_trajectory: deque[JsonDict] = deque(maxlen=60)
        self.latest_bundle: Optional[JsonDict] = None
        self.live_bundle: Optional[JsonDict] = None
        self.last_state: Optional[JsonDict] = None
        self.last_objective_id: Optional[str] = None
        self.action_events_since_objective_change = 0
        self.dialog_transcript_recent: deque[JsonDict] = deque(maxlen=12)
        self.last_dialog_text = ""
        self.dialog_last_change_at: Optional[str] = None
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir = self.workspace_dir / "debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_workspace_files()

    @property
    def artifacts(self) -> dict[str, Path]:
        return {
            "latest_frame": self.workspace_dir / "latest_frame.png",
            "latest_frame_annotated": self.workspace_dir / "latest_frame_annotated.png",
            "live_frame": self.workspace_dir / "live_frame.png",
            "live_frame_annotated": self.workspace_dir / "live_frame_annotated.png",
            "turn_context_json": self.workspace_dir / "turn_context.json",
            "latest_observation_json": self.debug_dir / "latest_observation.json",
            "current_objective_json": self.debug_dir / "current_objective.json",
            "run_log_jsonl": self.debug_dir / "run_log.jsonl",
        }

    def _ensure_workspace_files(self) -> None:
        run_log = self.artifacts["run_log_jsonl"]
        run_log.parent.mkdir(parents=True, exist_ok=True)
        run_log.touch(exist_ok=True)
        for key in ("turn_context_json", "latest_observation_json", "current_objective_json"):
            path = self.artifacts[key]
            if not path.exists():
                path.write_text("{}\n", encoding="utf-8")

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

    def load_turn_context(self) -> JsonDict:
        return self._read_json(self.artifacts["turn_context_json"], {})

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
        if (
            dialog_active
            and text
            and not text.startswith("Dialog box visible")
            and text != self.last_dialog_text
        ):
            self.last_dialog_text = text
            self.dialog_last_change_at = utc_now()
            entry = {"timestamp": self.dialog_last_change_at, "text": text}
            self.dialog_transcript_recent.append(entry)
            changed_event = entry
        elif not dialog_active:
            self.last_dialog_text = ""

        return (
            {
                "transcript_recent": [
                    entry["text"] for entry in list(self.dialog_transcript_recent)[-4:]
                ],
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
                "reason": (
                    "Battle text is still active; clear dialog before selecting another move."
                ),
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
            metadata = MOVE_METADATA.get(
                str(move.get("name") or ""), {"type": "Normal", "power": 35}
            )
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
                "reason": (
                    "No usable damaging move is visible; keep battle actions extremely short."
                ),
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

    def _maybe_auto_save(
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

    def _detect_stuck(
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

    def _latest_observed_frame_artifacts(self) -> JsonDict:
        artifacts = ((self.latest_bundle or {}).get("artifacts") or {}).copy()
        latest_frame = str(artifacts.get("latest_frame") or self.artifacts["latest_frame"])
        latest_frame_annotated = str(
            artifacts.get("latest_frame_annotated") or self.artifacts["latest_frame_annotated"]
        )
        return {
            "latest_frame": latest_frame,
            "latest_frame_annotated": latest_frame_annotated,
        }

    def _observation_frame_paths(self, observation_id: str) -> dict[str, Path]:
        observation_dir = self.workspace_dir / "observations" / observation_id
        observation_dir.mkdir(parents=True, exist_ok=True)
        return {
            "latest_frame": observation_dir / "latest_frame.png",
            "latest_frame_annotated": observation_dir / "latest_frame_annotated.png",
        }

    def _artifact_payload(
        self,
        *,
        latest_frame: Optional[str] = None,
        latest_frame_annotated: Optional[str] = None,
    ) -> JsonDict:
        observed = self._latest_observed_frame_artifacts()
        return {
            "latest_frame": latest_frame or observed["latest_frame"],
            "latest_frame_annotated": latest_frame_annotated or observed["latest_frame_annotated"],
            "live_frame": str(self.artifacts["live_frame"]),
            "live_frame_annotated": str(self.artifacts["live_frame_annotated"]),
            "turn_context_json": str(self.artifacts["turn_context_json"]),
            "latest_observation_json": str(self.artifacts["latest_observation_json"]),
            "current_objective_json": str(self.artifacts["current_objective_json"]),
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

    def _write_turn_context(self, bundle: JsonDict) -> JsonDict:
        """Write the slim display-only turn context. Not a model contract."""
        current = (bundle.get("objective") or {}).get("current") or {}
        state = bundle.get("state") or {}
        player = state.get("player") or {}
        position = player.get("position") or {}
        screen_text = bundle.get("screen_text") or {}
        context = {
            "observation_id": bundle.get("observation_id"),
            "objective": {
                "id": current.get("id"),
                "summary": current.get("summary"),
                "completion_predicate": current.get("completion_predicate"),
            },
            "position": {
                "map_name": (state.get("map") or {}).get("map_name"),
                "x": position.get("x"),
                "y": position.get("y"),
                "facing": player.get("facing"),
            },
            "ui": {
                "mode": screen_text.get("ui_mode"),
                "screen_text": screen_text.get("text"),
            },
        }
        self._write_json(self.artifacts["turn_context_json"], context)
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

    def _visited_tiles(
        self,
        snapshot: Optional[LiveNavigationSnapshot],
    ) -> Optional[set[tuple[int, int]]]:
        if snapshot is None or self.visited_lookup is None:
            return None
        try:
            return {(int(x), int(y)) for x, y in self.visited_lookup(snapshot.map_id)}
        except Exception:  # noqa: BLE001
            return None

    def _map_grid(
        self,
        snapshot: Optional[LiveNavigationSnapshot],
    ) -> Optional[JsonDict]:
        if snapshot is None or self.map_grid_lookup is None:
            return None
        try:
            grid = self.map_grid_lookup(snapshot.map_id)
        except Exception:  # noqa: BLE001
            return None
        return grid if isinstance(grid, dict) else None

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
        frame_path: Path,
        annotated_path: Path,
    ) -> None:
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        screen.save(buf, format="PNG")
        _atomic_write_bytes(frame_path, buf.getvalue())
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        _atomic_write_bytes(annotated_path, buf.getvalue())

    def sync_live_view(
        self,
        *,
        emulator: Any,
        state: JsonDict,
        navigation: Optional[JsonDict],
    ) -> JsonDict:
        current_objective = self.objective_engine.evaluate(state)
        screen = self._coerce_screen_image(emulator)
        snapshot = self._snapshot_from_navigation_payload(navigation)
        annotated = render_navigation_overlay(
            screen,
            snapshot,
            objective=current_objective["current"],
            goal=None,
            visited=self._visited_tiles(snapshot),
            map_grid=self._map_grid(snapshot),
        )
        self._write_frame_artifacts(
            screen=screen,
            annotated=annotated,
            frame_path=self.artifacts["live_frame"],
            annotated_path=self.artifacts["live_frame_annotated"],
        )

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
            preserved_text = "Live frame sync active. POST /action to advance the game."

        dialog_guidance = (previous_bundle.get("dialog_guidance") or {}).copy()
        dialog_guidance.setdefault("transcript_recent", [])
        dialog_guidance["should_continue"] = dialog_active
        dialog_guidance.setdefault("last_change_at", self.dialog_last_change_at)

        live_bundle = {
            "generated_at": utc_now(),
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
            },
            "objective": current_objective,
            "recent_action": previous_bundle.get("recent_action") or {},
            "movement_guidance": build_movement_guidance(snapshot=snapshot),
            "dialog_guidance": dialog_guidance,
            "battle_guidance": self._build_battle_guidance(state, dialog_guidance),
            "state_delta": previous_bundle.get("state_delta")
            or {
                "changed": False,
                "summary": ["Live frame sync only. POST /action to advance the game."],
                "movement": None,
            },
            "stuck": previous_bundle.get("stuck")
            or {
                "level": "clear",
                "reason": "No stuck signal recorded yet.",
                "objective_action_count": 0,
            },
            "workspace_dir": str(self.workspace_dir),
            "turn_context": self.load_turn_context(),
        }

        self.live_bundle = live_bundle
        return {
            "generated_at": live_bundle["generated_at"],
            "source": live_bundle["source"],
            "artifacts": live_bundle["artifacts"],
            "screen_text": live_bundle["screen_text"],
        }

    def refresh(
        self,
        *,
        emulator: Any,
        state: JsonDict,
        navigation: Optional[JsonDict],
        reason: str,
        source: str,
        requested_actions: Optional[list[str]] = None,
        navigation_plan: Optional[JsonDict] = None,
        navigation_execution: Optional[JsonDict] = None,
        explicit_save: Optional[JsonDict] = None,
    ) -> JsonDict:
        """Produce one observation bundle: frames, objective, deltas, telemetry."""
        current_objective = self.objective_engine.evaluate(state)
        screen = self._coerce_screen_image(emulator)
        snapshot = self._snapshot_from_navigation_payload(navigation)

        goal: Optional[tuple[int, int]] = None
        target = (navigation_execution or {}).get("target") or {}
        if target.get("x") is not None and target.get("y") is not None:
            goal = (int(target["x"]), int(target["y"]))

        generated_at = utc_now()
        observation_id = self._next_observation_id(
            generated_at=generated_at,
            reason=reason,
            state=state,
        )
        observation_frames = self._observation_frame_paths(observation_id)
        annotated = render_navigation_overlay(
            screen,
            snapshot,
            objective=current_objective["current"],
            goal=goal,
            visited=self._visited_tiles(snapshot),
            map_grid=self._map_grid(snapshot),
        )
        self._write_frame_artifacts(
            screen=screen,
            annotated=annotated,
            frame_path=observation_frames["latest_frame"],
            annotated_path=observation_frames["latest_frame_annotated"],
        )
        self._write_frame_artifacts(
            screen=screen,
            annotated=annotated,
            frame_path=self.artifacts["latest_frame"],
            annotated_path=self.artifacts["latest_frame_annotated"],
        )

        screen_text = classify_ui_state(state)
        state_delta = build_state_delta(self.last_state, state)
        action_feedback = classify_action_feedback(
            source=source,
            requested_actions=requested_actions,
            state_before=self.last_state,
            state_after=state,
            state_delta=state_delta,
            navigation_plan=navigation_plan,
            navigation_execution=navigation_execution,
        )
        dialog_guidance, _dialog_change = self._update_dialog_guidance(
            screen_text=screen_text,
            state=state,
        )
        auto_saves = self._maybe_auto_save(
            emulator=emulator,
            state=state,
            objective=current_objective,
            state_delta=state_delta,
            requested_actions=requested_actions,
            source=source,
        )
        if explicit_save:
            auto_saves.append(explicit_save)
        stuck = self._detect_stuck(
            state=state,
            objective=current_objective,
            source=source,
            requested_actions=requested_actions,
        )

        bundle: JsonDict = {
            "generated_at": generated_at,
            "observation_id": observation_id,
            "reason": reason,
            "source": source,
            "artifacts": self._artifact_payload(
                latest_frame=str(observation_frames["latest_frame"]),
                latest_frame_annotated=str(observation_frames["latest_frame_annotated"]),
            ),
            "state": state,
            "navigation": navigation,
            "screen_text": screen_text,
            "objective": current_objective,
            "recent_action": action_feedback,
            "movement_guidance": build_movement_guidance(snapshot=snapshot),
            "dialog_guidance": dialog_guidance,
            "battle_guidance": self._build_battle_guidance(state, dialog_guidance),
            "state_delta": state_delta,
            "stuck": stuck,
            "workspace_dir": str(self.workspace_dir),
        }
        bundle["turn_context"] = self._write_turn_context(bundle)

        self._write_json(self.artifacts["current_objective_json"], current_objective)
        self._write_json(self.artifacts["latest_observation_json"], bundle)

        events: list[JsonDict] = [
            self._record_event(
                "observe",
                {
                    "reason": reason,
                    "source": source,
                    "objective_id": current_objective["current"]["id"],
                    "summary": action_feedback["summary"],
                },
            )
        ]
        if current_objective["current"]["id"] != self.last_objective_id:
            events.append(
                self._record_event(
                    "objective",
                    {
                        "objective": current_objective["current"],
                        "progress_percent": current_objective["progress_percent"],
                    },
                )
            )
        for save_event in auto_saves:
            events.append(self._record_event("save", save_event))
        if stuck["level"] != "clear":
            events.append(self._record_event("stuck", stuck))

        self.latest_bundle = bundle
        self.live_bundle = bundle
        self.last_state = state
        self.last_objective_id = current_objective["current"]["id"]
        return {"bundle": bundle, "events": events}

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
        snapshot = navigation.get("snapshot") or {}
        visual_artifacts = bundle.get("artifacts") or {}
        use_live_frames = (
            bundle.get("source") == "live_sync"
            and visual_artifacts.get("live_frame")
            and visual_artifacts.get("live_frame_annotated")
        )
        return {
            "observation_id": bundle.get("observation_id"),
            "generated_at": bundle.get("generated_at"),
            "visuals": {
                "raw_frame_path": (
                    visual_artifacts.get("live_frame")
                    if use_live_frames
                    else visual_artifacts.get("latest_frame")
                ),
                "annotated_frame_path": (
                    visual_artifacts.get("live_frame_annotated")
                    if use_live_frames
                    else visual_artifacts.get("latest_frame_annotated")
                ),
                "frame_timestamp": bundle.get("generated_at"),
                "ui_mode": (bundle.get("screen_text") or {}).get("ui_mode"),
                "screen_text": bundle.get("screen_text"),
            },
            "agent_intent": {
                "objective": (bundle.get("objective") or {}).get("current") or {},
                "turn_context": bundle.get("turn_context") or self.load_turn_context(),
                "recent_action": bundle.get("recent_action"),
                "movement_guidance": bundle.get("movement_guidance"),
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
                "interaction": state.get("interaction") or snapshot.get("interaction"),
                "valid_moves": snapshot.get("valid_moves", []),
                "live_ascii": snapshot.get("ascii"),
                "navigation": navigation,
            },
            "memory_and_progress": {
                "progress_percent": (bundle.get("objective") or {}).get("progress_percent"),
                "stuck": bundle.get("stuck"),
                "workspace": {
                    "workspace_dir": bundle.get("workspace_dir"),
                    "turn_context_json": visual_artifacts.get("turn_context_json"),
                },
            },
            "timeline": self.history(80),
            "artifacts": visual_artifacts,
        }
