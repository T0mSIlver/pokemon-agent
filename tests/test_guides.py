"""The walkthrough library: what the agent can see, open, and what we record."""

import json

import pytest

from pokemon_agent import guides
from pokemon_agent.guides import GuideLog, Section


@pytest.fixture(autouse=True)
def _fresh_corpus():
    """Every test parses the real corpus from disk, not a leftover cache."""
    guides.reload()
    yield
    guides.reload()


# ----------------------------------------------------------------------
# The corpus
# ----------------------------------------------------------------------


def test_corpus_has_at_least_two_real_routes():
    names = guides.guides()
    assert "speedrun_glitchless" in names
    assert "standard_playthrough" in names


def test_every_section_parses_with_the_fields_the_index_needs():
    sections = guides.index()
    assert sections, "the corpus parsed to nothing"
    for section in sections:
        assert isinstance(section, Section)
        assert section.guide and section.slug and section.title
        assert section.summary, f"{section.ref} has no summary to show in the index"
        assert section.words > 0


def test_guide_slug_pairs_are_unique():
    refs = [section.ref for section in guides.index()]
    assert len(refs) == len(set(refs))


def test_slugs_are_addressable_identifiers():
    for section in guides.index():
        assert section.slug == section.slug.lower()
        assert " " not in section.slug


def test_sections_are_long_enough_to_act_on_and_short_enough_to_open():
    """150-400 words: worth opening on its own, cheap enough to open on a hunch."""
    for section in guides.index():
        assert 120 <= section.words <= 450, f"{section.ref} is {section.words} words"


def test_each_guide_cites_its_source():
    for path in sorted(guides.GUIDES_DIR.glob("*.md")):
        header = path.read_text(encoding="utf-8").split("\n---", 1)[0]
        assert "source" in header, f"{path.name} does not cite a source"


# ----------------------------------------------------------------------
# The index
# ----------------------------------------------------------------------


def test_outline_fits_in_a_prompt():
    text = guides.outline()
    assert len(text) < guides.OUTLINE_BUDGET_CHARS, f"outline is {len(text)} chars"


def test_outline_names_every_section_and_its_guide():
    text = guides.outline()
    for name in guides.guides():
        assert name in text
    for section in guides.index():
        assert section.slug in text
        assert section.summary in text


def test_outline_does_not_leak_the_bodies():
    """The index is a listing. If it carried the prose there would be nothing to open."""
    text = guides.outline()
    body = guides.read("speedrun_glitchless", "mt-moon")
    assert body is not None
    assert body[:120] not in text


# ----------------------------------------------------------------------
# read
# ----------------------------------------------------------------------


def test_read_returns_the_body_of_a_real_section():
    body = guides.read("speedrun_glitchless", "mt-moon")
    assert body is not None
    assert "Moon Stone" in body


def test_read_strips_the_slug_and_summary_markers():
    body = guides.read("speedrun_glitchless", "mt-moon")
    assert "<!--" not in body
    assert "slug:" not in body


def test_read_returns_none_for_a_bogus_address():
    assert guides.read("speedrun_glitchless", "no-such-section") is None
    assert guides.read("no_such_guide", "mt-moon") is None


def test_every_indexed_section_is_readable():
    for section in guides.index():
        assert guides.read(section.guide, section.slug), f"{section.ref} reads empty"


# ----------------------------------------------------------------------
# search
# ----------------------------------------------------------------------


def test_search_ranks_mt_moon_first():
    hits = guides.search("mt moon")
    assert hits
    assert hits[0].slug == "mt-moon"


def test_search_finds_a_leader_by_name():
    assert guides.search("misty")[0].slug in {"misty", "gym-leaders"}


def test_search_respects_the_limit_and_handles_an_empty_query():
    assert len(guides.search("gym", limit=2)) <= 2
    assert guides.search("") == ()
    assert guides.search("   ") == ()


def test_search_returns_nothing_for_a_word_the_corpus_never_uses():
    assert guides.search("zzzzqqqx") == ()


def test_search_is_stable():
    assert guides.search("rocket") == guides.search("rocket")


# ----------------------------------------------------------------------
# GuideLog
# ----------------------------------------------------------------------


def test_guide_log_round_trips_through_disk(tmp_path):
    path = tmp_path / "guide_reads.jsonl"
    log = GuideLog(path)
    log.record_read("speedrun_glitchless", "mt-moon", at_map="MT MOON 1F", presses=1200)
    log.record_read("battles", "gym-leaders", at_map="PEWTER GYM", presses=1450)

    reloaded = GuideLog(path)
    assert [(r["guide"], r["slug"]) for r in reloaded.reads()] == [
        ("speedrun_glitchless", "mt-moon"),
        ("battles", "gym-leaders"),
    ]
    assert reloaded.reads()[0]["at_map"] == "MT MOON 1F"
    assert reloaded.reads()[0]["presses"] == 1200


def test_guide_log_writes_one_json_object_per_line(tmp_path):
    path = tmp_path / "guide_reads.jsonl"
    log = GuideLog(path)
    log.record_read("battles", "mechanics", at_map=None, presses=None)
    log.record_read("battles", "mechanics", at_map=None, presses=None)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["slug"] == "mechanics" for line in lines)


def test_guide_log_starts_empty_for_a_missing_file(tmp_path):
    log = GuideLog(tmp_path / "nothing-here.jsonl")
    assert log.reads() == ()
    assert log.summary()["total_reads"] == 0
    assert log.summary()["sections"] == []


def test_summary_counts_repeat_reads(tmp_path):
    log = GuideLog(tmp_path / "guide_reads.jsonl")
    log.record_read("speedrun_glitchless", "mt-moon", at_map="ROUTE 4", presses=900)
    log.record_read("speedrun_glitchless", "mt-moon", at_map="MT MOON 1F", presses=1100)
    log.record_read("speedrun_glitchless", "mt-moon", at_map="MT MOON B2F", presses=1500)
    log.record_read("battles", "gym-leaders", at_map="MT MOON B2F", presses=1600)

    summary = log.summary()
    assert summary["total_reads"] == 4
    assert summary["unique_sections"] == 2
    assert summary["repeat_reads"] == 2
    assert summary["guides"] == {"speedrun_glitchless": 3, "battles": 1}

    # Most-read section first, so "what did it lean on" is the top line.
    top = summary["sections"][0]
    assert top["ref"] == "speedrun_glitchless/mt-moon"
    assert top["reads"] == 3
    assert top["maps"] == ["ROUTE 4", "MT MOON 1F", "MT MOON B2F"]


def test_summary_places_each_read_at_a_point_in_the_run(tmp_path):
    """The press counts are the whole reason the log exists: they date the read."""
    log = GuideLog(tmp_path / "guide_reads.jsonl")
    log.record_read("speedrun_glitchless", "misty", at_map="ROUTE 4", presses=900)
    log.record_read("speedrun_glitchless", "misty", at_map="CERULEAN GYM", presses=2400)

    summary = log.summary()
    entry = summary["sections"][0]
    assert entry["first_presses"] == 900
    assert entry["last_presses"] == 2400
    assert entry["first_seq"] == 1
    assert entry["last_seq"] == 2
    assert summary["first_read"]["presses"] == 900
    assert summary["last_read"]["presses"] == 2400


def test_summary_tolerates_missing_context(tmp_path):
    log = GuideLog(tmp_path / "guide_reads.jsonl")
    log.record_read("battles", "elite-four", at_map=None, presses=None)

    entry = log.summary()["sections"][0]
    assert entry["reads"] == 1
    assert entry["maps"] == []
    assert entry["first_presses"] is None


def test_guide_log_skips_corrupt_lines_without_losing_the_rest(tmp_path):
    path = tmp_path / "guide_reads.jsonl"
    path.write_text(
        '{"guide":"battles","slug":"mechanics"}\nnot json at all\n[1,2,3]\n',
        encoding="utf-8",
    )
    log = GuideLog(path)
    assert len(log.reads()) == 1
    assert log.reads()[0]["slug"] == "mechanics"


def test_recorded_sections_are_real_addresses(tmp_path):
    """A read we log should be a read the agent could actually have performed."""
    log = GuideLog(tmp_path / "guide_reads.jsonl")
    for section in guides.index()[:5]:
        log.record_read(section.guide, section.slug, at_map="PALLET TOWN", presses=1)
    for entry in log.reads():
        assert guides.read(entry["guide"], entry["slug"]) is not None
