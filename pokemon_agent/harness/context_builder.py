"""Small curated turn context builder for the supervised harness."""

from __future__ import annotations

import json
from typing import Any, Optional

from .contracts import (
    ActionBudget,
    BranchTemplates,
    ExpectedOutcomeTemplate,
    LandmarkHint,
    ModeRule,
    NavigationBranch,
    PlanningGuide,
    PlanStatus,
    RawActionBranch,
    RecentAction,
    RecoveryHint,
    RouteHint,
    TurnContext,
    TurnContextArtifacts,
)
from .planning import CONTEXT_VERSION

ASCII_WINDOW_LIMIT = 1600


def _truncate(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _manhattan(a: Optional[dict], b: Optional[dict]) -> Optional[int]:
    if not a or not b:
        return None
    try:
        return abs(int(a["x"]) - int(b["x"])) + abs(int(a["y"]) - int(b["y"]))
    except (KeyError, TypeError, ValueError):
        return None


def _build_warp_hints(
    *,
    snapshot: dict,
    player_position: dict,
    map_id_to_name: Optional[dict[int, str]] = None,
) -> list[dict]:
    warps = snapshot.get("warps") or []
    result: list[dict] = []
    for warp in warps[:8]:
        try:
            wx = int(warp.get("x"))
            wy = int(warp.get("y"))
        except (TypeError, ValueError):
            continue
        target_map_id = warp.get("target_map_id")
        target_name: Optional[str] = None
        if map_id_to_name and isinstance(target_map_id, int):
            target_name = map_id_to_name.get(target_map_id)
        result.append(
            {
                "coord": {"x": wx, "y": wy},
                "target_map_id": target_map_id,
                "target_map_name": target_name,
                "distance": _manhattan({"x": wx, "y": wy}, player_position),
            }
        )
    result.sort(key=lambda item: (item.get("distance") is None, item.get("distance") or 0))
    return result


def _build_recovery_command(*, save_name: Any, stuck_level: str) -> Optional[str]:
    if stuck_level == "clear" or not save_name:
        return None
    return f"bash agent_curl.sh /load <<'JSON'\n{json.dumps({'name': str(save_name)})}\nJSON"


def _build_expected_outcome_template(
    *,
    destination_npc: Optional[dict],
    current_map_name: str,
    dialog_active: bool,
) -> ExpectedOutcomeTemplate:
    if dialog_active:
        return ExpectedOutcomeTemplate(
            summary="Clear the active dialog box",
            dialog_active=False,
        )
    if destination_npc is not None:
        coord = destination_npc.get("coord") or {}
        interaction_position = None
        if coord.get("x") is not None and coord.get("y") is not None:
            interaction_position = {"x": int(coord.get("x")), "y": int(coord.get("y")) + 1}
        return ExpectedOutcomeTemplate(
            summary=f"Stand next to {destination_npc.get('name') or 'target NPC'}",
            map_name=current_map_name or str(destination_npc.get("map_name") or "") or None,
            position=interaction_position,
        )
    return ExpectedOutcomeTemplate(
        summary="Advance one tile north",
        position_delta={"dx": 0, "dy": -1},
    )


def build_turn_context(
    *,
    bundle: dict,
    plan_status: PlanStatus,
    map_id_to_name: Optional[dict[int, str]] = None,
    goal_override: Optional[str] = None,
) -> TurnContext:
    state = bundle.get("state") or {}
    objective = (bundle.get("objective") or {}).get("current") or {}
    screen_text = bundle.get("screen_text") or {}
    snapshot = ((bundle.get("navigation") or {}).get("snapshot")) or {}
    navigation_guidance = bundle.get("navigation_guidance") or {}
    recent_action = bundle.get("recent_action") or {}
    recovery = bundle.get("recovery") or {}
    current_recommendation = recovery.get("current_recommendation") or {}
    stuck = bundle.get("stuck") or {}
    artifacts = bundle.get("artifacts") or {}
    route_cards = navigation_guidance.get("route_cards") or []
    landmarks = navigation_guidance.get("landmarks") or []
    position = (state.get("player") or {}).get("position") or {}
    current_map_name = str((state.get("map") or {}).get("map_name") or "")
    dialog_active = bool(state.get("dialog_active") or ((state.get("dialog") or {}).get("active")))
    planner_objective_id = str(objective.get("id") or "")
    active_objective_id = (goal_override or "").strip() or planner_objective_id
    warps = _build_warp_hints(
        snapshot=snapshot,
        player_position=position,
        map_id_to_name=map_id_to_name,
    )
    target_npcs = list(objective.get("target_npcs") or [])
    route_hint_raw = objective.get("route_hint")
    destination_npc = next(
        (
            npc
            for npc in target_npcs
            if current_map_name and str(npc.get("map_name") or "") == current_map_name
        ),
        None,
    )
    if destination_npc is not None:
        coord = destination_npc.get("coord") or {}
        route_hint_raw = (
            f"You are inside {current_map_name}. Talk to "
            f"{destination_npc.get('name') or 'the target NPC'} at "
            f"({coord.get('x')}, {coord.get('y')})."
        )
    visible_sprites = [
        {"x": int(sprite.get("x")), "y": int(sprite.get("y"))}
        for sprite in (snapshot.get("sprites") or [])
        if sprite.get("x") is not None and sprite.get("y") is not None
    ][:16]
    ascii_window = _truncate(snapshot.get("ascii"), ASCII_WINDOW_LIMIT) or None
    ascii_legend = dict(snapshot.get("ascii_legend") or {}) or None

    route_hints = [
        RouteHint(
            title=_truncate(card.get("title"), 80),
            why=_truncate(card.get("why_now"), 100),
            actions=list(card.get("route_actions") or [])[:4],
        )
        for card in route_cards[:2]
    ]
    landmark_hints = [
        LandmarkHint(
            kind=str(landmark.get("kind") or "unknown"),
            title=_truncate(landmark.get("title"), 80),
            distance=landmark.get("distance"),
        )
        for landmark in landmarks[:3]
    ]

    return TurnContext(
        version=CONTEXT_VERSION,
        observation_id=str(bundle.get("observation_id") or ""),
        generated_at=str(bundle.get("generated_at") or ""),
        reason=str(bundle.get("reason") or ""),
        source=str(bundle.get("source") or ""),
        objective={
            "id": active_objective_id,
            "title": objective.get("title"),
            "summary": _truncate(objective.get("summary"), 180),
            "completion_predicate": _truncate(objective.get("completion_predicate"), 180),
            "progress_percent": (bundle.get("objective") or {}).get("progress_percent"),
            "route_hint": _truncate(route_hint_raw, 180),
            "target_npcs": target_npcs,
            "mission_override": active_objective_id
            if active_objective_id != planner_objective_id
            else "",
            "planner_objective_id": planner_objective_id,
        },
        ui={
            "mode": screen_text.get("ui_mode"),
            "screen_text": _truncate(screen_text.get("text"), 220),
        },
        position={
            "map_id": (state.get("map") or {}).get("map_id"),
            "map_name": (state.get("map") or {}).get("map_name"),
            "x": position.get("x"),
            "y": position.get("y"),
            "facing": (state.get("player") or {}).get("facing"),
        },
        navigation={
            "coordinate_system": snapshot.get("coordinate_system"),
            "coordinate_note": snapshot.get("coordinate_note"),
            "valid_moves": list(snapshot.get("valid_moves") or [])[:4],
            "visible_sprites": visible_sprites,
            "ascii_window": ascii_window,
            "ascii_legend": ascii_legend,
            "interaction": {
                "kind": (snapshot.get("interaction") or {}).get("kind"),
                "source": (snapshot.get("interaction") or {}).get("source"),
                "reason": _truncate((snapshot.get("interaction") or {}).get("reason"), 140),
                "target_coord": (snapshot.get("interaction") or {}).get("target_coord"),
            },
            "warps": warps,
            "route_hints": [hint.model_dump() for hint in route_hints],
            "landmarks": [hint.model_dump() for hint in landmark_hints],
        },
        recent_action=RecentAction(
            summary=_truncate(recent_action.get("summary"), 160),
            notes=[_truncate(note, 100) for note in (recent_action.get("notes") or [])[:3]],
            tags=list(recent_action.get("tags") or [])[:4],
        ),
        recovery=RecoveryHint(
            recommended_save=current_recommendation.get("name"),
            recovery_command=_build_recovery_command(
                save_name=current_recommendation.get("name"),
                stuck_level=str(stuck.get("level") or "clear"),
            ),
            candidate_count=len(recovery.get("candidates") or []),
            stuck_level=str(stuck.get("level") or "clear"),
            stuck_reason=_truncate(stuck.get("reason"), 160),
            recommended_actions=[
                _truncate(note, 80) for note in (stuck.get("recommended_actions") or [])[:3]
            ],
        ),
        constraints=ActionBudget(),
        planning=PlanningGuide(
            observation_id=str(bundle.get("observation_id") or ""),
            objective_id=active_objective_id,
            mode_rules={
                "overworld": ModeRule(primary_branch_kind="raw_actions", max_actions=4),
                "dialog": ModeRule(primary_branch_kind="raw_actions", max_actions=2),
                "battle": ModeRule(primary_branch_kind="raw_actions", max_actions=2),
                "navigation": ModeRule(
                    primary_branch_kind="navigation",
                    max_targets=1,
                    navigation_mode_options=["auto", "screen", "persistent"],
                ),
            },
            branch_templates=BranchTemplates(
                raw_actions=RawActionBranch(
                    kind="raw_actions",
                    actions=["a_until_dialog_end"] if dialog_active else ["walk_up"],
                ),
                navigation=NavigationBranch(
                    kind="navigation",
                    target={"x": 0, "y": 0},
                    mode="auto",
                ),
            ),
            expected_outcome_template=_build_expected_outcome_template(
                destination_npc=destination_npc,
                current_map_name=current_map_name,
                dialog_active=dialog_active,
            ),
        ),
        plan_status=plan_status,
        artifacts=TurnContextArtifacts(
            latest_frame=str(artifacts.get("latest_frame") or ""),
            latest_frame_annotated=str(artifacts.get("latest_frame_annotated") or ""),
            turn_context_json=str(artifacts.get("turn_context_json") or ""),
            turn_plan_json=str(artifacts.get("turn_plan_json") or ""),
            recovery_saves_json=str(artifacts.get("recovery_saves_json") or "") or None,
        ),
    )
