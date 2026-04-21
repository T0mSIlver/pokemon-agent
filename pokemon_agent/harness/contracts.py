"""Pydantic contracts for the minimal-turn supervised harness."""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

PlanMode = Literal["overworld", "dialog", "battle", "navigation"]
BranchKind = Literal["raw_actions", "navigation"]
PlanState = Literal[
    "awaiting_plan",
    "validated",
    "executed_waiting_observe",
    "matched",
    "partial",
    "drifted",
    "invalid",
    "stale",
]
BranchSelection = Literal["primary", "fallback"]
ExpectedOutcomeCheckField = Literal[
    "map_name",
    "position",
    "position_delta",
    "dialog_active",
    "battle_active",
]

EXPECTED_OUTCOME_CHECK_FIELDS: tuple[ExpectedOutcomeCheckField, ...] = (
    "map_name",
    "position",
    "position_delta",
    "dialog_active",
    "battle_active",
)


class Coord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int
    y: int


class CoordDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dx: int
    dy: int


class ExpectedOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=220)
    map_name: Optional[str] = None
    position: Optional[Coord] = None
    position_delta: Optional[CoordDelta] = None
    dialog_active: Optional[bool] = None
    battle_active: Optional[bool] = None

    @model_validator(mode="after")
    def ensure_structured_check(self) -> "ExpectedOutcome":
        checks = (
            self.map_name,
            self.position,
            self.position_delta,
            self.dialog_active,
            self.battle_active,
        )
        if not any(value is not None for value in checks):
            raise ValueError(
                "expected_outcome requires at least one measurable check in addition to summary"
            )
        return self


class RawActionBranch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["raw_actions"]
    actions: list[str] = Field(min_length=1, max_length=4)


class NavigationBranch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["navigation"]
    target: Coord
    mode: Literal["auto", "screen", "persistent"] = "auto"


PlanBranch = Annotated[RawActionBranch | NavigationBranch, Field(discriminator="kind")]


class ModeRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_branch_kind: BranchKind
    max_actions: Optional[int] = None
    max_targets: Optional[int] = None
    navigation_mode_options: list[Literal["auto", "screen", "persistent"]] = Field(
        default_factory=list
    )


class BranchTemplates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_actions: RawActionBranch
    navigation: NavigationBranch


class ExpectedOutcomeTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    map_name: Optional[str] = None
    position: Optional[Coord] = None
    position_delta: Optional[CoordDelta] = None
    dialog_active: Optional[bool] = None
    battle_active: Optional[bool] = None


class PlanningGuide(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str
    objective_id: str
    mode_rules: dict[PlanMode, ModeRule]
    branch_templates: BranchTemplates
    expected_outcome_requires_check: bool = True
    expected_outcome_check_fields: list[ExpectedOutcomeCheckField] = Field(
        default_factory=lambda: list(EXPECTED_OUTCOME_CHECK_FIELDS)
    )
    expected_outcome_template: Optional[ExpectedOutcomeTemplate] = None


class PlanExecution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    branch: BranchSelection
    branch_kind: Literal["raw_actions", "navigation"]
    requested_actions: list[str] = Field(default_factory=list)
    executed_actions: int = 0
    started_at: str
    completed_at: Optional[str] = None
    baseline_map_name: Optional[str] = None
    baseline_position: Optional[Coord] = None
    summary: str = ""


class PlanStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: PlanState = "awaiting_plan"
    observation_id: str = ""
    plan_updated_at: str = ""
    validated_at: Optional[str] = None
    executed_at: Optional[str] = None
    outcome_checked_at: Optional[str] = None
    branch_executed: Optional[BranchSelection] = None
    reason: str = ""
    last_error: Optional[str] = None


class TurnPlanInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str = Field(min_length=1)
    objective_id: str = Field(min_length=1)
    intent: str = Field(min_length=1, max_length=220)
    mode: PlanMode
    primary_branch: PlanBranch
    fallback_branch: Optional[PlanBranch] = None
    expected_outcome: ExpectedOutcome
    notes: str = Field(default="", max_length=220)


class TurnPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    observation_id: str = ""
    objective_id: str = ""
    intent: str = ""
    mode: Optional[PlanMode] = None
    primary_branch: Optional[PlanBranch] = None
    fallback_branch: Optional[PlanBranch] = None
    expected_outcome: Optional[ExpectedOutcome] = None
    notes: str = ""
    updated_at: str = ""
    status: PlanStatus = Field(default_factory=PlanStatus)
    execution: Optional[PlanExecution] = None


class RouteHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    why: str
    actions: list[str] = Field(default_factory=list)


class LandmarkHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    title: str
    distance: Optional[int] = None


class InteractionHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Optional[str] = None
    source: Optional[str] = None
    reason: str = ""
    target_coord: Optional[Coord] = None


class RecentAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = ""
    notes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class RecoveryHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommended_save: Optional[str] = None
    recovery_command: Optional[str] = None
    candidate_count: int = 0
    stuck_level: str = "clear"
    stuck_reason: str = ""
    recommended_actions: list[str] = Field(default_factory=list)


class ActionBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inspect_frames_first: bool = True
    overworld_raw_action_max: int = 4
    dialog_raw_action_max: int = 2
    battle_raw_action_max: int = 2
    navigation_target_max: int = 1


class TurnContextArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latest_frame: str
    latest_frame_annotated: str
    turn_context_json: str
    turn_plan_json: str
    recovery_saves_json: Optional[str] = None


class TurnContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 2
    observation_id: str
    generated_at: str
    reason: str
    source: str
    objective: dict
    ui: dict
    position: dict
    navigation: dict
    recent_action: RecentAction = Field(default_factory=RecentAction)
    recovery: RecoveryHint = Field(default_factory=RecoveryHint)
    constraints: ActionBudget = Field(default_factory=ActionBudget)
    planning: PlanningGuide
    plan_status: PlanStatus = Field(default_factory=PlanStatus)
    artifacts: TurnContextArtifacts
