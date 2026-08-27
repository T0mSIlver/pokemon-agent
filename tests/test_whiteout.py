"""What a whiteout costs, and whether anything says so.

Measured on run 20260825T224823Z-983b, 33 hours and 85,843 presses: 19
whiteouts, at least 10,801 presses spent walking back from them (12.6% of the
run, more than the whole Cerulean Gym line item), and $15,249 halved away. The
model's own transcripts name the faint every single time — "I whited out and
woke up at the Poke Center" — so it does not need telling that it fainted. What
it was never told is the bill: it checked its money after three of the nineteen,
and read one clean halving, 2,490 to 1,245, as "1245 money (I spent some)". It
had spent nothing.
"""

from pokemon_agent.bench.metrics import whiteout_events
from pokemon_agent.state_analysis import WhiteoutWatch, party_is_down


def state(map_name="Mt Moon 1F", x=5, y=8, hp=73, money=3961, party=1):
    return {
        "map": {"map_name": map_name},
        "player": {"position": {"x": x, "y": y}, "money": money},
        "party": [{"species": "Charmeleon", "hp": hp, "max_hp": 73} for _ in range(party)],
    }


class Flagged:
    """A receipt as the counters read it: only the flag matters."""

    def __init__(self, whiteout):
        self.whiteout = whiteout


def flags(pattern):
    return [Flagged(char == "W") for char in pattern]


# -- the flag itself ---------------------------------------------------------


def test_a_party_with_one_member_standing_has_not_whited_out():
    assert party_is_down(state(hp=1)) is False


def test_an_empty_party_is_not_a_whiteout():
    # The frame between releasing and depositing reads as party-of-zero, and
    # `all()` over nothing is True. Guarding on emptiness is what stops that
    # scoring as a loss.
    assert party_is_down(state(party=0)) is False


# -- counting events rather than frames --------------------------------------


def test_one_faint_mashed_through_over_several_batches_counts_once():
    # The measured failure: 40 receipts carried the flag across 19 actual
    # whiteouts, because the whole party reads as down on every batch the faint
    # takes to resolve. `scope live`, the critic handoff, `bench report` and
    # `scope diff` all published the inflated number, and the inflation factor
    # varies with how long the model mashed A, so runs were not comparable with
    # each other either.
    assert whiteout_events(flags("..WWWW..")) == 1


def test_two_faints_with_play_between_them_count_twice():
    assert whiteout_events(flags("..WW...W.")) == 2


def test_a_run_that_never_lost_scores_zero():
    assert whiteout_events(flags("........")) == 0


def test_a_run_that_opens_mid_faint_still_counts_it():
    # A session that resumes on the faint frame has no falling edge before it.
    assert whiteout_events(flags("WW...")) == 1


# -- the note the model reads ------------------------------------------------


def test_the_landing_frame_names_where_the_party_went_down():
    watch = WhiteoutWatch()
    assert watch.observe(state(hp=0)) is None
    note = watch.observe(state(map_name="Route 4", x=11, y=6, hp=73, money=1980))
    assert "Mt Moon 1F (5,8)" in note
    assert "Route 4 (11,6)" in note


def test_the_note_prices_the_halving():
    # The half the model never had. It reads money only through a separate
    # `poke state` call, which one 120-step session made five times.
    watch = WhiteoutWatch()
    watch.observe(state(hp=0, money=3961))
    note = watch.observe(state(map_name="Route 4", x=11, y=6, hp=73, money=1980))
    assert "$1,981 of your $3,961" in note


def test_the_note_fires_once_and_not_on_the_walk_that_follows():
    # It belongs to the batch that landed. Repeating it would make it wallpaper,
    # which is what `here_before` became at 2,339 sendings and zero uses.
    watch = WhiteoutWatch()
    watch.observe(state(hp=0))
    assert watch.observe(state(map_name="Route 4", x=11, y=6, hp=73, money=1980))
    assert watch.observe(state(map_name="Route 4", x=12, y=6, hp=73, money=1980)) is None


def test_the_faint_resolving_in_place_is_not_yet_a_teleport():
    # Between one and seven batches of every measured whiteout were spent
    # standing on the tile the party went down on, pressing A through the text.
    # Firing there would name the wrong destination.
    watch = WhiteoutWatch()
    watch.observe(state(hp=0))
    assert watch.observe(state(hp=0)) is None
    assert watch.observe(state(hp=0)) is None


def test_an_ordinary_map_change_says_nothing():
    watch = WhiteoutWatch()
    watch.observe(state(hp=73))
    assert watch.observe(state(map_name="Route 4", x=11, y=6, hp=73)) is None


def test_a_forgotten_faint_never_lands():
    # A save reload rewinds the money with everything else, so the note would
    # price a loss that no longer exists. Ten of the nineteen measured whiteouts
    # were undone that way within four receipts.
    watch = WhiteoutWatch()
    watch.observe(state(hp=0))
    watch.forget()
    assert watch.observe(state(map_name="Cerulean Pokecenter", x=3, y=3, hp=78)) is None


def test_money_the_reader_could_not_see_leaves_the_rest_of_the_note_standing():
    # Where the player went is worth saying even when the wallet is unreadable;
    # a note that is all-or-nothing would go silent on the frames that need it.
    watch = WhiteoutWatch()
    watch.observe(state(hp=0, money=None))
    note = watch.observe(state(map_name="Route 4", x=11, y=6, hp=73, money=None))
    assert "Route 4 (11,6)" in note
    assert "$" not in note
