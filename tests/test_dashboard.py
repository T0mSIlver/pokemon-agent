"""The dashboard is static files plus one mounting helper, so this is what can
be checked without a browser: that every id the script reaches for exists in the
page, that the embedded ladder still matches the ladder on disk, and that the
mount puts the assets where ``index.html`` asks for them.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from pokemon_agent.dashboard import ASSETS_ROUTE, INDEX_ROUTES, mount_dashboard

STATIC = Path(__file__).resolve().parents[1] / "pokemon_agent" / "dashboard" / "static"
DATA = Path(__file__).resolve().parents[1] / "pokemon_agent" / "data"

INDEX_HTML = (STATIC / "index.html").read_text(encoding="utf-8")
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")
STYLE_CSS = (STATIC / "style.css").read_text(encoding="utf-8")


class _IdCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name == "id" and value:
                self.ids.append(value)


def _element_ids() -> list[str]:
    parser = _IdCollector()
    parser.feed(INDEX_HTML)
    return parser.ids


def _referenced_ids() -> set[str]:
    return set(re.findall(r"\$\('([A-Za-z0-9_-]+)'\)", APP_JS))


def test_every_id_the_script_reaches_for_exists_in_the_page():
    missing = sorted(_referenced_ids() - set(_element_ids()))
    assert not missing, f"app.js calls $() for ids that index.html does not define: {missing}"


def test_page_ids_are_unique():
    ids = _element_ids()
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    assert not duplicates, f"duplicate ids in index.html: {duplicates}"


def test_the_script_reaches_for_a_realistic_number_of_ids():
    # A guard against a refactor quietly dropping half the panel wiring.
    assert len(_referenced_ids()) >= 91


@pytest.mark.parametrize(
    "element_id",
    [
        "hudCampaign",
        "campaignRungChip",
        "campaignPressesChip",
        "campaignSourceChip",
        "campaignHeadline",
        "campaignStats",
        "campaignRail",
        "campaignRailFill",
        "campaignRailReadout",
        "campaignChart",
        "campaignChartCaption",
        "campaignBenchmark",
        "campaignLadderDetails",
        "campaignLadderRows",
        "healthWindowChip",
        "healthStrip",
    ],
)
def test_campaign_panel_anchors_are_present(element_id):
    assert element_id in _element_ids()


@pytest.mark.parametrize(
    "element_id",
    [
        # The panels the operator watches around the clock. Losing any of these
        # to a layout change is the failure mode this file exists to catch.
        "annotatedFrame",
        "rawFrame",
        "piStream",
        "piStreamList",
        "piSteerInput",
        "piSteerButton",
        "manualSaveButton",
        "loadSaveButton",
        "loadRecommendedButton",
        "critiqueText",
        "objectiveTitle",
    ],
)
def test_pre_existing_panels_survive(element_id):
    assert element_id in _element_ids()


def _embedded_ladder() -> list[list[str]]:
    match = re.search(r"const RED_LADDER_RAW = \[\n(.*?)\n    \];", APP_JS, re.DOTALL)
    assert match, "RED_LADDER_RAW is missing from app.js"
    body = match.group(1).strip().rstrip(",")
    return json.loads(f"[{body}]")


def test_embedded_ladder_matches_the_ladder_on_disk():
    # The browser has no endpoint for red_milestones.json and the page has no
    # build step, so the ladder is duplicated into app.js. This is the seam.
    on_disk = json.loads((DATA / "red_milestones.json").read_text(encoding="utf-8"))["ladder"]
    expected = [[rung["id"], rung["label"], rung["kind"]] for rung in on_disk]
    assert _embedded_ladder() == expected


def test_ladder_has_sixty_three_rungs():
    assert len(_embedded_ladder()) == 63


def test_first_gym_rung_is_on_the_ladder():
    ids = [rung[0] for rung in _embedded_ladder()]
    assert "EVENT_BEAT_BROCK" in ids


def test_published_reference_points_are_quoted_exactly():
    assert "const REF_POKEAGENT_BEST = 1608;" in APP_JS
    assert "const REF_POKEAGENT_EFFICIENT = 649;" in APP_JS
    assert "const REF_HUMAN_SPEEDRUN_SECONDS = 18 * 60;" in APP_JS


def test_progress_endpoint_is_consumed_but_never_required():
    # The page must render its ladder with /progress absent.
    assert "api('/progress')" in APP_JS
    assert "progressState.available = false;" in APP_JS


def test_no_endpoints_were_invented():
    # Both quoting styles: api('/x') and api(`/x?${...}`).
    called = {path.split("?")[0] for path in re.findall(r"api\([`'](/[^`'\n]*)[`']\)", APP_JS)}
    known = {
        "/dashboard/state",
        "/dashboard/history",
        "/saves",
        "/progress",
        "/supervisor/stream",
        "/supervisor/start",
        "/supervisor/continue",
        "/supervisor/stop",
        "/supervisor/steer",
        "/save",
        "/load",
    }
    assert called, "no endpoint calls found — the regex has drifted"
    assert called <= known, f"app.js calls unknown endpoints: {sorted(called - known)}"


@pytest.mark.parametrize(
    "selector",
    [
        # Classes the script creates at runtime; without a rule they render raw.
        ".hud-rung",
        ".hud-health-pill",
        ".hud-health-glyph",
        ".hud-health-meter-fill",
        ".hud-health-note",
        ".hud-benchmark-row",
        ".hud-benchmark-bar-fill",
        ".hud-benchmark-group",
        ".hud-ladder-kind",
    ],
)
def test_runtime_classes_are_styled(selector):
    assert selector in STYLE_CSS


@pytest.mark.parametrize("token", ["--sev-good", "--sev-warn", "--sev-crit", "--sev-idle"])
def test_severity_tokens_are_defined_apart_from_the_accent(token):
    assert f"{token}:" in STYLE_CSS


def test_severity_reads_as_form_not_only_colour():
    # A stripe and a glyph, so the state survives a colour-blind operator.
    assert 'data-sev="crit"' in STYLE_CSS
    assert "border-left-color: var(--sev-crit)" in STYLE_CSS
    assert "HEALTH_GLYPH" in APP_JS


def test_wide_content_scrolls_inside_its_own_container():
    assert ".hud-rail-scroll {" in STYLE_CSS
    assert "overflow-x: auto;" in STYLE_CSS
    assert ".hud-ladder-scroll {" in STYLE_CSS


def test_digits_line_up():
    assert STYLE_CSS.count("font-variant-numeric: tabular-nums") >= 6


def test_page_asks_for_assets_under_the_mounted_prefix():
    for asset in ("app.js", "style.css"):
        assert f"{ASSETS_ROUTE}/{asset}" in INDEX_HTML


def test_nothing_is_truncated_by_a_character_cap_in_the_critic_panel():
    # The retrospective is a scroll container, not a slice.
    assert "hud-scrollbox" in INDEX_HTML
    assert 'id="critiqueText"' in INDEX_HTML


def _client():
    fastapi = pytest.importorskip("fastapi")
    starlette_testclient = pytest.importorskip("starlette.testclient")
    app = fastapi.FastAPI()
    assert mount_dashboard(app) is True
    return starlette_testclient.TestClient(app)


def test_mount_serves_the_shell_and_its_assets():
    client = _client()
    for route in INDEX_ROUTES:
        response = client.get(route)
        assert response.status_code == 200, route
        assert "hud-shell" in response.text
    for asset in ("app.js", "style.css"):
        response = client.get(f"{ASSETS_ROUTE}/{asset}")
        assert response.status_code == 200, asset


def test_mount_is_idempotent():
    fastapi = pytest.importorskip("fastapi")
    app = fastapi.FastAPI()
    assert mount_dashboard(app) is True
    before = len(app.router.routes)
    assert mount_dashboard(app) is True
    assert len(app.router.routes) == before
