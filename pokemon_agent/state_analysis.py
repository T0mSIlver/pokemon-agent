"""Pure readings of the game-state dict: selectors, deltas, and UI classification.

Nothing here touches the emulator, the filesystem, or a clock. Every function
takes the state dict the memory reader produced and returns plain data, which is
what makes the observation bundle testable without a ROM.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from pokemon_agent.navigation import LiveNavigationSnapshot

JsonDict = dict[str, Any]


def bag_item_counts(state: Optional[JsonDict]) -> dict[str, int]:
    bag = (state or {}).get("bag") or []
    counts: dict[str, int] = {}
    for entry in bag:
        item = str(entry.get("item") or "").strip()
        if not item:
            continue
        counts[item] = int(entry.get("quantity") or 0)
    return counts


def bag_item_names(state: Optional[JsonDict]) -> set[str]:
    return set(bag_item_counts(state))


def badge_count(state: JsonDict) -> int:
    player = state.get("player") or {}
    flags = state.get("flags") or {}
    return int(player.get("badge_count", flags.get("badge_count", 0)) or 0)


def selector_matches(selector: JsonDict, state: JsonDict) -> bool:
    if not selector:
        return True

    map_name = str((state.get("map") or {}).get("map_name") or "")
    bag_items = bag_item_names(state)
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

    badges = badge_count(state)
    if selector.get("badge_count_gte") is not None and badges < int(selector["badge_count_gte"]):
        return False
    if selector.get("badge_count_lte") is not None and badges > int(selector["badge_count_lte"]):
        return False
    if selector.get("badge_count_lt") is not None and badges >= int(selector["badge_count_lt"]):
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

    before_bag = bag_item_counts(before)
    after_bag = bag_item_counts(after)
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


def type_multiplier(move_type: str, enemy_types: Iterable[str]) -> float:
    multiplier = 1.0
    for enemy_type in enemy_types:
        multiplier *= TYPE_EFFECTIVENESS.get(move_type, {}).get(str(enemy_type), 1.0)
    return multiplier


def build_battle_guidance(state: JsonDict, dialog_guidance: JsonDict) -> JsonDict:
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
            "reason": ("Battle text is still active; clear dialog before selecting another move."),
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
            score *= type_multiplier(move_type, enemy_types)
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
            "reason": ("No usable damaging move is visible; keep battle actions extremely short."),
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
