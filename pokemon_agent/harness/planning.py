"""Validation and lifecycle helpers for strict turn plans."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import ValidationError

from .contracts import (
    BranchSelection,
    Coord,
    PlanExecution,
    PlanState,
    PlanStatus,
    TurnContext,
    TurnPlan,
    TurnPlanInput,
)

PLAN_VERSION = 1
CONTEXT_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_plan_status(
    *,
    state: PlanState = "awaiting_plan",
    observation_id: str = "",
    reason: str = "",
    last_error: Optional[str] = None,
) -> PlanStatus:
    return PlanStatus(
        state=state,
        observation_id=observation_id,
        plan_updated_at=utc_now() if observation_id or reason or last_error else "",
        reason=reason,
        last_error=last_error,
    )


def default_turn_plan() -> TurnPlan:
    return TurnPlan(version=PLAN_VERSION, status=default_plan_status())


def _raw_action_limit(mode: str) -> int:
    if mode == "overworld":
        return 4
    if mode in {"dialog", "battle"}:
        return 2
    return 0


def validate_turn_plan_submission(submission: TurnPlanInput, context: TurnContext) -> None:
    if submission.observation_id != context.observation_id:
        raise ValueError("turn plan observation_id does not match the latest turn_context")
    if submission.objective_id != str(context.objective.get("id") or ""):
        raise ValueError("turn plan objective_id does not match the current objective")

    primary = submission.primary_branch
    if submission.mode == "navigation":
        if primary.kind != "navigation":
            raise ValueError("navigation plans require a navigation primary_branch")
    else:
        if primary.kind != "raw_actions":
            raise ValueError(f"{submission.mode} plans require a raw_actions primary_branch")
        limit = _raw_action_limit(submission.mode)
        if len(primary.actions) > limit:
            raise ValueError(
                f"{submission.mode} primary_branch exceeds the {limit}-action batch limit"
            )

    fallback = submission.fallback_branch
    if fallback is not None and fallback.kind == "raw_actions":
        raw_mode = "overworld" if submission.mode == "navigation" else submission.mode
        limit = _raw_action_limit(raw_mode)
        if len(fallback.actions) > limit:
            raise ValueError(f"fallback raw_actions exceeds the {limit}-action batch limit")


def store_validated_plan(submission: TurnPlanInput) -> TurnPlan:
    now = utc_now()
    return TurnPlan(
        version=PLAN_VERSION,
        observation_id=submission.observation_id,
        objective_id=submission.objective_id,
        intent=submission.intent,
        mode=submission.mode,
        primary_branch=submission.primary_branch,
        fallback_branch=submission.fallback_branch,
        expected_outcome=submission.expected_outcome,
        notes=submission.notes,
        updated_at=now,
        status=PlanStatus(
            state="validated",
            observation_id=submission.observation_id,
            plan_updated_at=now,
            validated_at=now,
            reason="Validated against the latest turn_context.",
        ),
    )


def invalidate_plan(plan: TurnPlan, *, reason: str, state: PlanState = "invalid") -> TurnPlan:
    plan.status = PlanStatus(
        state=state,
        observation_id=plan.observation_id,
        plan_updated_at=plan.updated_at or utc_now(),
        validated_at=plan.status.validated_at,
        executed_at=plan.status.executed_at,
        outcome_checked_at=utc_now(),
        branch_executed=plan.status.branch_executed,
        reason=reason,
        last_error=reason if state in {"invalid", "stale"} else None,
    )
    return plan


def mark_plan_executed(
    plan: TurnPlan,
    *,
    branch: BranchSelection,
    branch_kind: str,
    requested_actions: list[str],
    executed_actions: int,
    baseline_map_name: Optional[str],
    baseline_position: Optional[dict],
    summary: str,
) -> TurnPlan:
    now = utc_now()
    plan.execution = PlanExecution(
        branch=branch,
        branch_kind=branch_kind,  # type: ignore[arg-type]
        requested_actions=requested_actions,
        executed_actions=executed_actions,
        started_at=now,
        completed_at=now,
        baseline_map_name=baseline_map_name,
        baseline_position=Coord.model_validate(baseline_position) if baseline_position else None,
        summary=summary,
    )
    plan.status = PlanStatus(
        state="executed_waiting_observe",
        observation_id=plan.observation_id,
        plan_updated_at=plan.updated_at,
        validated_at=plan.status.validated_at,
        executed_at=now,
        branch_executed=branch,
        reason="Action batch executed. Run /agent/observe before planning again.",
    )
    return plan


def _movement_from_execution_baseline(plan: TurnPlan, bundle: dict) -> dict:
    if plan.execution is not None and plan.execution.baseline_position is not None:
        current_position = (bundle.get("state") or {}).get("player", {}).get("position") or {}
        current_x = current_position.get("x")
        current_y = current_position.get("y")
        if current_x is not None and current_y is not None:
            return {
                "dx": int(current_x) - plan.execution.baseline_position.x,
                "dy": int(current_y) - plan.execution.baseline_position.y,
            }
    return ((bundle.get("state_delta") or {}).get("movement")) or {}


def _match_expected_outcome(plan: TurnPlan, bundle: dict) -> tuple[int, int, list[str]]:
    if plan.expected_outcome is None:
        return 0, 0, []
    current_state = bundle.get("state") or {}
    current_map = str((current_state.get("map") or {}).get("map_name") or "")
    current_position = (current_state.get("player") or {}).get("position") or {}
    dialog_active = bool(
        current_state.get("dialog_active") or ((current_state.get("dialog") or {}).get("active"))
    )
    battle_active = bool(((current_state.get("battle") or {}).get("in_battle")))
    movement = _movement_from_execution_baseline(plan, bundle)

    matched = 0
    total = 0
    notes: list[str] = []

    if plan.expected_outcome.map_name is not None:
        total += 1
        if current_map == plan.expected_outcome.map_name:
            matched += 1
        else:
            notes.append(f"map_name expected {plan.expected_outcome.map_name}, got {current_map}")

    if plan.expected_outcome.position is not None:
        total += 1
        if (
            current_position.get("x") == plan.expected_outcome.position.x
            and current_position.get("y") == plan.expected_outcome.position.y
        ):
            matched += 1
        else:
            notes.append(
                "position expected "
                f"({plan.expected_outcome.position.x}, {plan.expected_outcome.position.y}), "
                f"got ({current_position.get('x')}, {current_position.get('y')})"
            )

    if plan.expected_outcome.position_delta is not None:
        total += 1
        if (
            movement.get("dx") == plan.expected_outcome.position_delta.dx
            and movement.get("dy") == plan.expected_outcome.position_delta.dy
        ):
            matched += 1
        else:
            notes.append(
                "position_delta expected "
                f"({plan.expected_outcome.position_delta.dx}, "
                f"{plan.expected_outcome.position_delta.dy}), "
                f"got ({movement.get('dx')}, {movement.get('dy')})"
            )

    if plan.expected_outcome.dialog_active is not None:
        total += 1
        if dialog_active is plan.expected_outcome.dialog_active:
            matched += 1
        else:
            notes.append(
                f"dialog_active expected {plan.expected_outcome.dialog_active}, got {dialog_active}"
            )

    if plan.expected_outcome.battle_active is not None:
        total += 1
        if battle_active is plan.expected_outcome.battle_active:
            matched += 1
        else:
            notes.append(
                f"battle_active expected {plan.expected_outcome.battle_active}, got {battle_active}"
            )

    return matched, total, notes


def _partial_raw_batch(plan: TurnPlan) -> Optional[tuple[int, int]]:
    """Return (executed, requested) when a raw_actions batch stopped early."""
    if plan.execution is None or plan.execution.branch_kind != "raw_actions":
        return None
    requested = len(plan.execution.requested_actions or [])
    executed = int(plan.execution.executed_actions or 0)
    if 0 < executed < requested:
        return executed, requested
    return None


def evaluate_plan_outcome(plan: TurnPlan, bundle: dict) -> TurnPlan:
    matched, total, notes = _match_expected_outcome(plan, bundle)
    checked_at = utc_now()

    if total == 0:
        return invalidate_plan(
            plan,
            reason=(
                "Plan execution could not be checked because no measurable "
                "expected_outcome was stored."
            ),
        )

    partial_batch = _partial_raw_batch(plan)

    if matched == total:
        state: PlanState = "matched"
        reason = "Expected outcome matched."
    elif matched > 0:
        state = "partial"
        reason = "Expected outcome partially matched."
    elif partial_batch is not None:
        executed, requested = partial_batch
        state = "partial"
        reason = (
            f"Batch blocked after {executed}/{requested} steps "
            f"({plan.execution.requested_actions[0]} likely hit a wall or sprite)."
        )
    else:
        state = "drifted"
        reason = "Expected outcome drifted."

    if notes:
        reason = f"{reason} {'; '.join(notes[:3])}"

    plan.status = PlanStatus(
        state=state,
        observation_id=plan.observation_id,
        plan_updated_at=plan.updated_at,
        validated_at=plan.status.validated_at,
        executed_at=plan.status.executed_at,
        outcome_checked_at=checked_at,
        branch_executed=plan.status.branch_executed,
        reason=reason,
    )
    return plan


def build_plan_execution_trace(plan: TurnPlan, bundle: dict) -> Optional[str]:
    """Return a single-line trace describing what the last plan actually did.

    Combines the requested/executed action counts, observed movement, and the
    post-eval plan state so the agent can see at a glance whether the batch
    landed, partially landed, or drifted — and why.
    """
    if plan.execution is None:
        return None
    exec_ = plan.execution
    status = plan.status
    requested = list(exec_.requested_actions or [])
    executed = int(exec_.executed_actions or 0)
    total = len(requested)

    if exec_.branch_kind == "navigation":
        head = f"navigation branch ran {executed} step(s)"
    else:
        if total and all(action == requested[0] for action in requested):
            desc = f"{requested[0]} x{total}"
        else:
            desc = " ".join(requested) or "raw_actions"
        head = f"{desc} requested; {executed}/{total} executed"

    movement = _movement_from_execution_baseline(plan, bundle) or {}
    dx = int(movement.get("dx") or 0)
    dy = int(movement.get("dy") or 0)
    head += f"; delta=({dx},{dy})"

    state = status.state
    if state in {"matched", "partial", "drifted"}:
        head += f"; outcome={state}"

    if state != "matched" and status.reason:
        reason = status.reason
        for prefix in (
            "Expected outcome drifted.",
            "Expected outcome partially matched.",
            "Expected outcome matched.",
        ):
            if reason.startswith(prefix):
                reason = reason[len(prefix) :].strip()
                break
        if reason:
            head += f" ({reason})"
    return head


def parse_stored_turn_plan(payload: object) -> TurnPlan:
    if isinstance(payload, dict) and "summary" in payload and "planned_actions" in payload:
        now = str(payload.get("updated_at") or "")
        primary_branch = None
        if payload.get("planned_actions"):
            primary_branch = {
                "kind": "raw_actions",
                "actions": list(payload.get("planned_actions") or []),
            }
        fallback_branch = None
        if payload.get("fallback_actions"):
            fallback_branch = {
                "kind": "raw_actions",
                "actions": list(payload.get("fallback_actions") or []),
            }
        try:
            return TurnPlan.model_validate(
                {
                    "version": PLAN_VERSION,
                    "observation_id": payload.get("observation_id") or "",
                    "objective_id": payload.get("objective_id") or "",
                    "intent": payload.get("summary") or "",
                    "mode": payload.get("mode") or "overworld",
                    "primary_branch": primary_branch,
                    "fallback_branch": fallback_branch,
                    "notes": payload.get("notes") or "",
                    "updated_at": now,
                    "status": {
                        "state": ((payload.get("status") or {}).get("state")) or "awaiting_plan",
                        "observation_id": payload.get("observation_id") or "",
                        "plan_updated_at": now,
                        "reason": ((payload.get("status") or {}).get("reason")) or "",
                    },
                }
            )
        except ValidationError:
            return default_turn_plan()
    try:
        return TurnPlan.model_validate(payload or {})
    except ValidationError:
        return default_turn_plan()
