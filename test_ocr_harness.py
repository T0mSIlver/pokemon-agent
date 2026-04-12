from pokemon_agent.harness.context_builder import build_turn_context
from pokemon_agent.harness.contracts import PlanStatus
from pokemon_agent.server import _compact_screen_text


def test_compact_screen_text_keeps_ocr_note():
    compact = _compact_screen_text(
        {
            "text": "Dialog box visible; OCR text unavailable (waiting for input).",
            "source": "dialog_state",
            "ui_mode": "battle",
            "dialog_active": True,
            "note": "OCR unavailable: ModuleNotFoundError",
        }
    )

    assert compact == {
        "text": "Dialog box visible; OCR text unavailable (waiting for input).",
        "source": "dialog_state",
        "ui_mode": "battle",
        "dialog_active": True,
        "note": "OCR unavailable: ModuleNotFoundError",
    }


def test_turn_context_includes_ocr_note():
    context = build_turn_context(
        bundle={
            "generated_at": "2026-04-12T15:21:47.866522+00:00",
            "observation_id": "obs-ocr",
            "reason": "observe",
            "source": "observe",
            "objective": {
                "current": {
                    "id": "return_oaks_parcel",
                    "title": "Return Oak's Parcel",
                    "summary": "Bring the parcel back to Oak's Lab.",
                    "completion_predicate": "The flags report that the player has the Pokedex.",
                    "route_hint": "Go south to Pallet Town.",
                },
                "progress_percent": 15,
            },
            "state": {
                "map": {"map_id": 12, "map_name": "Route 1"},
                "player": {"position": {"x": 9, "y": 28}, "facing": "left"},
            },
            "screen_text": {
                "text": "Dialog box visible; OCR text unavailable (waiting for input).",
                "source": "dialog_state",
                "ui_mode": "battle",
                "dialog_active": True,
                "note": "OCR unavailable: ModuleNotFoundError",
            },
            "navigation": {"snapshot": {}},
            "navigation_guidance": {},
            "recent_action": {},
            "recovery": {},
            "stuck": {},
            "artifacts": {
                "latest_frame": "/tmp/latest_frame.png",
                "latest_frame_annotated": "/tmp/latest_frame_annotated.png",
                "turn_context_json": "/tmp/turn_context.json",
                "turn_plan_json": "/tmp/turn_plan.json",
            },
        },
        plan_status=PlanStatus(),
    )

    assert context.ui["screen_text_note"] == "OCR unavailable: ModuleNotFoundError"
