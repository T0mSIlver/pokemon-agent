#!/usr/bin/env python3
"""One command for everything the agent does to the game.

The agent used to drive the game with hand-written curl. Measured over a live
run, 260 of 261 failed tool calls had an odd number of single quotes: the model
dropped the closing quote on ``-d '{"actions": [...]}'``. The identical command
succeeded 414 times, so it was a sampling flake on one character that silently
discarded roughly 40% of the agent's actions.

Nothing here needs quoting and nothing here is hand-built JSON, so there is no
string literal for a sampler to truncate.

Standalone by design: stdlib only, no package imports. It is copied into the
agent's workspace as ``poke`` and run from there.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

DEFAULT_PORT = 8765
TIMEOUT_SECONDS = 120.0

EXIT_OK = 0
EXIT_HTTP_ERROR = 1
EXIT_BAD_USAGE = 2
EXIT_NO_SERVER = 3

DIRECTIONS = ("up", "down", "left", "right")
BUTTONS = ("a", "b", "start", "select")

#: Every action the server's parser accepts by name.
ACTIONS = (
    *(f"walk_{direction}" for direction in DIRECTIONS),
    *(f"press_{button}" for button in BUTTONS),
    "hold_a_30",
    "wait_60",
    "a_until_dialog_end",
)

#: Short forms. `up up a` is less to type and less to get wrong than
#: `walk_up walk_up press_a`, and long batches are where typing hurts.
ALIASES = {
    **{direction: f"walk_{direction}" for direction in DIRECTIONS},
    **{button: f"press_{button}" for button in BUTTONS},
    "wait": "wait_60",
    "adialog": "a_until_dialog_end",
}

#: A batch longer than this is a loop, not a plan. These mirror the server's
#: limits exactly; the server is the one that enforces them, and refusing here
#: too only means the model learns from a fast local error instead of a 400.
#: The server holds its single emulator lock for the whole batch, so an
#: uncapped `wait_1000000000` would take the game away for hundreds of days.
MAX_REPEAT = 40
MAX_ACTIONS_PER_BATCH = 40
MAX_FRAMES_PER_ACTION = 600
MAX_FRAMES_PER_BATCH = 3600

#: A plan that is only ever walked on paper costs no emulator time and takes no
#: lock, so the batch caps above do not apply to it — only a ceiling that stops
#: a runaway loop asking the server to simulate forever. Four hundred is the
#: same number `agent_api.WALK_MAX_ACTIONS` uses for a chunked walk, and the
#: longest plan any session ever offered to `poke sim` was 159 actions.
MAX_PLAN_ACTIONS = 400

#: What each action costs in emulator frames, for the batch budget. Walks and
#: presses are the fixed press-plus-wait cadence. `a_until_dialog_end` is counted
#: at the server's own worst case, ten presses of 30 frames: budget it any
#: cheaper here and the CLI waves through batches the server then refuses.
FRAMES_PER_INPUT = 20
FRAMES_DIALOG_WORST_CASE = 300

FRAME_FILES = {
    "raw": "latest_frame.png",
    "annotated": "latest_frame_annotated.png",
}


class ActionError(Exception):
    """A bad action name, caught before anything is sent."""


class ServerError(Exception):
    """The server answered with a status the agent needs to read."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class Unreachable(Exception):
    """Nothing is listening."""


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def resolve_action(token: str) -> Optional[str]:
    """Canonical action name for one token, or None if it is not an action."""

    name = token.strip().lower()
    if name in ALIASES:
        return ALIASES[name]
    if name in ACTIONS:
        return name
    parts = name.split("_")
    if len(parts) == 2 and parts[0] == "wait" and parts[1].isdigit() and int(parts[1]) > 0:
        return name
    if (
        len(parts) == 3
        and parts[0] == "hold"
        and parts[1] in BUTTONS + DIRECTIONS
        and parts[2].isdigit()
        and int(parts[2]) > 0
    ):
        return name
    return None


def frames_for(action: str) -> int:
    """Emulator frames one canonical action costs, for the batch budget."""

    if action == "a_until_dialog_end":
        return FRAMES_DIALOG_WORST_CASE
    parts = action.split("_")
    if parts[0] == "wait":
        return int(parts[1])
    if parts[0] == "hold":
        return int(parts[2])
    return FRAMES_PER_INPUT


def action_help() -> str:
    return (
        "actions: up down left right a b start select wait adialog\n"
        "  long form: " + " ".join(ACTIONS) + "\n"
        "  also wait_N and hold_<button>_N for any frame count\n"
        "  repeat with a colon: up:4 sends four walk_up"
    )


def expand_actions(tokens: list[str], *, batch: bool = True) -> list[str]:
    """Tokens as typed into the action list the server expects.

    Handles aliases and the ``name:count`` repeat form. Raises on anything it
    does not recognise, so a typo costs nothing but the error.

    ``batch=False`` keeps the vocabulary and the per-action frame ceiling but
    drops the per-batch caps, which exist only because the server holds its one
    emulator lock for the whole batch it is executing. A plan that is only
    simulated is never executed and never takes that lock, so the caps do not
    apply to it. Applying them anyway cost one session 505 of its 549 calls:
    the model grew a candidate route through Mt. Moon one segment at a time and
    re-simulated it, and every attempt past forty actions came back refused,
    41 to 159 actions, 183,599 characters of command and refusal for no answer
    at all.
    """

    max_repeat = MAX_REPEAT if batch else MAX_PLAN_ACTIONS
    actions: list[str] = []
    for token in tokens:
        name, separator, count_text = token.partition(":")
        count = 1
        if separator:
            if not count_text.isdigit() or int(count_text) < 1:
                raise ActionError(
                    f"bad repeat count in {token!r} - write up:4 to send four walk_up"
                )
            count = int(count_text)
            if count > max_repeat:
                raise ActionError(f"{token!r} repeats more than {max_repeat} times")
        action = resolve_action(name)
        if action is None:
            raise ActionError(f"unknown action {name!r}\n{action_help()}")
        frames = frames_for(action)
        if frames > MAX_FRAMES_PER_ACTION:
            raise ActionError(
                f"{token!r} asks for {frames} frames; the limit is "
                f"{MAX_FRAMES_PER_ACTION} ({MAX_FRAMES_PER_ACTION // 60} seconds)"
            )
        actions.extend([action] * count)
    if not actions:
        raise ActionError(f"no actions given\n{action_help()}")
    if not batch:
        if len(actions) > MAX_PLAN_ACTIONS:
            raise ActionError(
                f"that plan is {len(actions)} actions; even on paper the limit is "
                f"{MAX_PLAN_ACTIONS}. A plan that long is a loop, not a route."
            )
        return actions
    if len(actions) > MAX_ACTIONS_PER_BATCH:
        # "Send fewer" named no number, and a model that had just grown its plan
        # read it as noise and grew the plan again: 505 refusals in a row, every
        # one longer than the last. Name the exact edit instead.
        raise ActionError(
            f"that batch is {len(actions)} actions; the limit is "
            f"{MAX_ACTIONS_PER_BATCH}. Drop the last {len(actions) - MAX_ACTIONS_PER_BATCH}, "
            f"or send the first {MAX_ACTIONS_PER_BATCH} and re-plan from where they land."
        )
    total = sum(frames_for(action) for action in actions)
    if total > MAX_FRAMES_PER_BATCH:
        raise ActionError(
            f"that batch runs {total} frames; the limit is {MAX_FRAMES_PER_BATCH} "
            f"({MAX_FRAMES_PER_BATCH // 60} seconds)"
        )
    return actions


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def server_port(args: argparse.Namespace) -> str:
    return str(getattr(args, "port", None) or os.environ.get("PORT") or DEFAULT_PORT)


def base_url(args: argparse.Namespace) -> str:
    if getattr(args, "url", None):
        return str(args.url).rstrip("/")
    return f"http://localhost:{server_port(args)}"


def server_label(args: argparse.Namespace) -> str:
    """How to name the server when it does not answer. The port is what the
    operator can act on; a full URL is only interesting if one was passed."""

    if getattr(args, "url", None):
        return f"at {base_url(args)}"
    return f"on port {server_port(args)}"


def _detail_from(error: urllib.error.HTTPError) -> str:
    """The server writes its 400s and 429s for the agent to read. Keep the words."""

    try:
        body = error.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - an unreadable body must not mask the status
        body = ""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return body.strip() or str(error.reason) or f"HTTP {error.code}"
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail
        if detail is not None:
            return json.dumps(detail)
    return body.strip()


def fetch(url: str, path: str, *, method: str = "GET", payload: Optional[dict] = None) -> bytes:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        raise ServerError(error.code, _detail_from(error)) from None
    except urllib.error.URLError as error:
        raise Unreachable(str(getattr(error, "reason", error))) from None
    except OSError as error:
        raise Unreachable(str(error)) from None


def fetch_json(url: str, path: str, *, method: str = "GET", payload: Optional[dict] = None):
    raw = fetch(url, path, method=method, payload=payload)
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except (json.JSONDecodeError, ValueError):
        raise ServerError(0, f"server sent something that is not JSON: {raw[:200]!r}") from None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def compact(payload) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def ago(timestamp: Optional[float]) -> str:
    if not timestamp:
        return "?"
    seconds = max(0, int(time.time() - timestamp))
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def state_lines(state: dict) -> list[str]:
    player = state.get("player") or {}
    position = player.get("position") or {}
    map_name = (state.get("map") or {}).get("map_name") or "?"
    x, y = position.get("x"), position.get("y")
    where = f"{map_name} "
    where += f"({x},{y})" if x is not None and y is not None else "(position unread)"
    # Same reason the action payload refuses it: an encounter interrupts the step
    # that started it, so in a battle the facing byte is still the direction from
    # before that step. Two of four measured battle frames read a direction the
    # overworld disagreed with the moment the fight ended.
    if (state.get("battle") or {}).get("in_battle"):
        lines = [f"{where} facing unread in a battle"]
    else:
        lines = [f"{where} facing {player.get('facing') or 'unread'}"]

    party = state.get("party") or []
    if party:
        lines.append("party:")
        for mon in party:
            types = "/".join(mon.get("types") or []) or "?"
            status = mon.get("status") or "OK"
            tail = "" if status == "OK" else f" {status}"
            lines.append(
                f"  {mon.get('species')} L{mon.get('level')} "
                f"{mon.get('hp')}/{mon.get('max_hp')} {types}{tail}"
            )
            # Only the empty ones, and only for the lead. The payload has carried
            # PP per move all along and nothing printed it, so `poke fight` was
            # refused 12 times for a move that had run dry — and `poke calc`
            # still ranks a 0 PP move as the best one available, because its own
            # payload does not carry PP either. Naming the dry moves once costs
            # a line only on the turn it would have cost a wasted call.
            if mon is party[0]:
                empty = [
                    move.get("name")
                    for move in mon.get("moves") or []
                    if isinstance(move, dict) and move.get("pp") == 0
                ]
                if empty:
                    lines.append(f"  no PP: {', '.join(name for name in empty if name)}")
    else:
        lines.append("party: empty")

    badges = player.get("badges") or (state.get("flags") or {}).get("badges") or []
    lines.append(
        f"badges: {', '.join(badges) if badges else 'none'}   money: {player.get('money')}"
    )

    bag = state.get("bag") or []
    items = ", ".join(f"{entry.get('item')} x{entry.get('quantity')}" for entry in bag)
    lines.append(f"bag: {items or 'empty'}")

    battle = state.get("battle") or {}
    if battle.get("in_battle"):
        enemy = battle.get("enemy") or {}
        if enemy.get("species"):
            lines.append(
                f"battle: {enemy.get('species')} L{enemy.get('level')} "
                f"{enemy.get('hp')}/{enemy.get('max_hp')}"
            )
        else:
            lines.append("battle: yes")
    if state.get("dialog_active") or (state.get("dialog") or {}).get("active"):
        lines.append("dialog: open")
    return lines


def map_lines(payload: dict) -> list[str]:
    coverage = payload.get("coverage") or {}
    lines = [
        f"{payload.get('map_name')} (map {payload.get('map_id')}) "
        f"{payload.get('width')}x{payload.get('height')}",
        f"seen {coverage.get('seen')}/{coverage.get('total')} "
        f"({coverage.get('percent')}%)   walked {coverage.get('walked')}",
    ]
    player = payload.get("player") or {}
    if player:
        lines.append(f"you: ({player.get('x')},{player.get('y')})")
    warps = payload.get("warps") or []
    if warps:
        # Name where each one goes. A list of bare coordinates cannot tell the
        # ladder deeper into a cave from the one door out of it.
        lines.append("warps:")
        for w in warps:
            target = w.get("to")
            lines.append(f"  ({w.get('x')},{w.get('y')})" + (f" -> {target}" if target else ""))
    # Above "nearest unexplored", because in the room this was written for the
    # unexplored tile is the wrong answer and the nurse is the right one: a
    # Cerulean Center read 100% seen and still pointed at (5,6) and the door.
    services = payload.get("services") or []
    if services:
        lines.append(
            "who is here: "
            + ", ".join(
                f"{entry.get('service')} at ({(entry.get('at') or [None, None])[0]},"
                f"{(entry.get('at') or [None, None])[1]})"
                for entry in services
            )
        )
    nearest = payload.get("unexplored_nearest")
    if nearest:
        lines.append(f"nearest unexplored: ({nearest.get('x')},{nearest.get('y')})")
    if payload.get("image_path"):
        lines.append(f"png: {payload['image_path']}")
    return lines


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def action_lines(payload: dict) -> str:
    """One line for what just happened, instead of the whole JSON object.

    `act` was the only acting verb that printed its payload raw: 4,251 of them
    over the run, 960,537 bytes, and tool text is 98% of the model's prompt, so
    each one is paid when it arrives and again on every turn after. The same
    facts as prose come to roughly a quarter of that.

    Nothing is dropped from the API -- `--json` still prints the object and the
    Python client still reads every field. This is only what the shell prints.
    """
    # Never `(None,None) facing None`. A missing field here means the server could
    # not read it off this frame, and printing None as though it were a reading is
    # the worst of the three things this line can do: a battle payload used to
    # render `Mt Moon 1F (None,None) facing None`, which cost a whole `poke state`
    # call to undo. Say the field is unread and the next call is about the game
    # again rather than about the answer.
    x, y = payload.get("x"), payload.get("y")
    where = f"{payload.get('map') or 'map unread'} "
    where += f"({x},{y})" if x is not None and y is not None else "(position unread)"
    facing = payload.get("facing")
    parts = [f"{where} facing {facing}" if facing else where]

    moved = payload.get("moved")
    if moved is not None:
        parts.append(f"moved {moved}")
    if payload.get("blocked_after") is not None:
        parts.append(f"blocked after {payload['blocked_after']}")
    if payload.get("hp"):
        parts.append(f"hp {payload['hp']}")

    lines = ["  ".join(parts)]

    # Ahead of everything else, because it is the reason the line above says a
    # different map than the last one did. Without it a whiteout renders as a
    # walk: the model reads a new town and full HP and has to infer the rest.
    if payload.get("whiteout"):
        lines.append(str(payload["whiteout"]))

    # Before anything read off the frame, because it says the frame is not one the
    # game has come to rest on: mid-transition the map name and the coordinates
    # can belong to two different maps.
    if payload.get("settled") is False:
        lines.append(
            "the game was still moving when this was read - it is mid-transition, "
            "so the map and the coordinates may not belong together. wait and look again"
        )

    run = payload.get("run") or {}
    if run:
        lines.append("run " + " ".join(f"{d}:{n}" for d, n in run.items()))
    # The two are mutually exclusive by construction: the server sends `no_walk`
    # exactly on the frames where it drops `run`, and it is the reason it dropped
    # them. Printing nothing there is what let a battle or a dialog look like a
    # dead end.
    if payload.get("no_walk"):
        lines.append(str(payload["no_walk"]))
    # Which field is missing from the first line, and why. Without it the line
    # just stops after the coordinates and reads as though facing did not matter.
    if payload.get("facing_unread"):
        lines.append(str(payload["facing_unread"]))
    exits = payload.get("exits") or {}
    if exits:
        rendered = []
        for target, where_to in exits.items():
            rendered.append(
                f"{target} {tuple(where_to)}"
                if isinstance(where_to, list)
                else f"{target} {where_to}"
            )
        lines.append("exits " + " | ".join(rendered))

    # Only when they are true, because a line that is always there is read once.
    flags = []
    if payload.get("faces"):
        flags.append(f"facing a {payload['faces']} - press a")
    if payload.get("on_warp"):
        warp = payload.get("warp") or {}
        step = warp.get("step")
        # A warp whose target map the reader could not name is still a warp worth
        # knowing you are standing on; "on a warp to None" made it read as one
        # that leads nowhere.
        destination = f" to {warp['to']}" if warp.get("to") else " (destination unread)"
        flags.append(f"on a warp{destination}" + (f", step {step}" if step else ""))
    if payload.get("here_before"):
        flags.append(f"stood here {payload['here_before']} times before")
    # Not in a battle, where the flag is only saying that battle text is on
    # screen: the BATTLE block below already says that, and "dialog open" next to
    # it reads as an NPC talking mid-fight.
    if payload.get("dialog") and not payload.get("battle"):
        flags.append("dialog open")
    if flags:
        lines.append("  ".join(flags))

    if payload.get("battle"):
        enemy = payload.get("enemy")
        you = payload.get("you")
        head = "BATTLE" + (f" {you}" if you else "")
        lines.append(f"{head} vs {enemy}" if enemy else head)
        # One move per line. They used to be four names on one comma-separated
        # line, which is what a name list wants; each of them is now a priced
        # row -- type, PP, damage, kill count -- and four of those run past a
        # terminal width joined by commas.
        for move in payload.get("your_moves") or []:
            lines.append(f"  {move}")
        # Labelled, because under a list of moves "Scratch up to 4" reads as a
        # fifth move of yours rather than as the hit you are about to take.
        if payload.get("incoming"):
            lines.append(f"  incoming: {payload['incoming']}")
        for key in ("no_damage", "locked_in"):
            if payload.get(key):
                lines.append(f"  {payload[key]}")
        # Under the moves, because it is the other thing this turn could be
        # spent on and it is only ever sent on a wild encounter.
        if payload.get("catch"):
            lines.append(f"  catch: {payload['catch']}")
        # Beside `catch`, because it is the third thing the turn could be spent
        # on and the only one of the three the run that this was written for
        # never once spent it on. Labelled for the same reason `incoming` is:
        # under a list of moves, "Potion x7 +20 -> 25/95" reads as a fifth move.
        if payload.get("items"):
            lines.append(f"  items: {payload['items']}")
        menu, highlighted = payload.get("menu"), payload.get("highlighted")
        if menu == "other":
            # `other` is the reader saying neither battle menu is up, so there is
            # no cursor to name. It printed "menu other on None", which reads as a
            # menu with a missing entry rather than as battle text on screen.
            lines.append("no battle menu up - this frame is text or an animation")
        elif menu:
            lines.append(f"menu {menu} on {highlighted}" if highlighted else f"menu {menu}")
    # Above `screen_text` and below the fight, because both of them describe the
    # frame and these describe the button about to be pressed.
    if payload.get("learn"):
        lines.append(str(payload["learn"]))
    if payload.get("ahead"):
        lines.append(str(payload["ahead"]))
    # Beside `ahead` and for the same reason: both answer "what could this party
    # do that it is not doing", one from the gym's side and one from the bag's.
    if payload.get("tm"):
        lines.append(str(payload["tm"]))
    # Twelve maps in the game carry this and none of the others do, so it never
    # competes with anything: on a mart floor it is the only line that says what
    # standing there is for.
    if payload.get("shop"):
        lines.append(f"for sale  {payload['shop']}")
    # Same bargain as `shop`: thirteen maps in the game carry it and none of the
    # others do, so it never competes with anything, and on those thirteen it is
    # the only line that says why you walked in.
    if payload.get("heal"):
        lines.append(str(payload["heal"]))
    # And the counters that are neither: a room where something is traded rather
    # than sold. One map carries this today. Before it, the frame inside the
    # Cerulean Bike Shop was three lines about walking, none of which mentioned
    # the clerk, and the run spent 839 presses there without the Bicycle.
    if payload.get("counter"):
        lines.append(str(payload["counter"]))
    # Labelled, and labelled on every line of it, because this is the one thing
    # in the block the harness is not the author of: it is decoded off the tile
    # map, so it is the game's words and a nickname's are the player's. Every
    # other line here is the harness talking - "no battle menu up", "the game
    # was still moving when this was read" - and an unlabelled multi-line
    # dialog under them reads as more of the same. It is the same misreading
    # `incoming:` is labelled against two dozen lines further up, with the
    # difference that a nickname can be written to produce it on purpose.
    if payload.get("screen_text"):
        lines += [f"screen  {line}" for line in str(payload["screen_text"]).splitlines()]
    return "\n".join(lines)


def cmd_act(args: argparse.Namespace, url: str) -> int:
    actions = expand_actions(args.actions)
    payload = fetch_json(url, "/action", method="POST", payload={"actions": actions})
    print(compact(payload) if args.json else action_lines(payload))
    return EXIT_OK


def _print_battle_result(payload: dict, as_json: bool) -> int:
    """The same prose `poke act` prints, plus the move the server confirmed.

    These two commands printed the raw payload, alone among the commands here.
    That was tolerable when a battle answer was four move names; the answer now
    carries a priced table, and one run spent 790 battle commands on it.
    """
    if as_json:
        print(compact(payload))
        return EXIT_OK
    used = payload.get("used")
    if used:
        print(f"used {used}" + ("  (cursor needed a retry)" if payload.get("retried") else ""))
    elif "fled" in payload:
        print("fled" if payload.get("fled") else "could not get away")
    print(action_lines(payload))
    return EXIT_OK


def cmd_fight(args: argparse.Namespace, url: str) -> int:
    """Attack by name. The server does the menu work; this only carries the name."""
    payload = {"move": " ".join(args.move)}
    answer = fetch_json(url, "/battle/fight", method="POST", payload=payload)
    return _print_battle_result(answer, args.json)


def cmd_run(args: argparse.Namespace, url: str) -> int:
    return _print_battle_result(fetch_json(url, "/battle/run", method="POST"), args.json)


def cmd_catch(args: argparse.Namespace, url: str) -> int:
    """Throw a ball at the wild Pokemon. No ball named means the weakest carried."""
    payload = {"ball": " ".join(args.ball)} if args.ball else {}
    answer = fetch_json(url, "/battle/catch", method="POST", payload=payload)
    if args.json:
        print(compact(answer))
        return EXIT_OK
    threw, species = answer.get("threw"), answer.get("species")
    if answer.get("caught"):
        print(f"caught {species} with a {threw}")
    else:
        print(f"threw a {threw}, {species} broke free  ({answer.get('balls_left')} left)")
    print(action_lines(answer))
    return EXIT_OK


def cmd_item(args: argparse.Namespace, url: str) -> int:
    """Use an item mid-battle. No item named means the weakest healing one carried."""
    payload: dict = {}
    if args.item:
        payload["item"] = " ".join(args.item)
    if args.on is not None:
        payload["on"] = args.on
    answer = fetch_json(url, "/battle/item", method="POST", payload=payload)
    if args.json:
        print(compact(answer))
        return EXIT_OK
    used, on = answer.get("used"), answer.get("on")
    restored = answer.get("restored")
    head = f"used a {used}" + (f" on {on}" if on else "")
    if restored is not None:
        head += f", +{restored} HP"
    print(head + f"  ({answer.get('left')} left)")
    print(action_lines(answer))
    return EXIT_OK


def split_buy_tokens(tokens: list[str]) -> tuple[str, int]:
    """``poke buy poke ball 10`` as the item and the count.

    The item is words and the count is a number, so a trailing number is the
    count and everything before it is the name. Two positionals would have
    argparse swallow the number into the name — and no item in the game ends in
    a digit except the TMs, which are one token and never the last of several.
    """
    if len(tokens) > 1 and tokens[-1].isdigit():
        return " ".join(tokens[:-1]), int(tokens[-1])
    return " ".join(tokens), 1


def cmd_buy(args: argparse.Namespace, url: str) -> int:
    """Buy from the counter on this map. Walks to the till first if you are not at it."""
    item, count = split_buy_tokens(list(args.item))
    payload = {"item": item, "count": count}
    answer = fetch_json(url, "/mart/buy", method="POST", payload=payload)
    if args.json:
        print(compact(answer))
        return EXIT_OK
    print(
        f"bought {answer.get('count')} x {answer.get('bought')} for ${answer.get('spent')}"
        f"  (${answer.get('money')} left, holding {answer.get('have')})"
    )
    print(action_lines(answer))
    return EXIT_OK


def cmd_heal(args: argparse.Namespace, url: str) -> int:
    """Heal at the nurse on this map. Walks to her counter first."""
    answer = fetch_json(url, "/pokecenter/heal", method="POST", payload={})
    if args.json:
        print(compact(answer))
        return EXIT_OK
    was = answer.get("was") or []
    healed = ", ".join(answer.get("healed") or []) or "the party"
    print(
        f"healed {healed} in {answer.get('presses')} presses"
        + (f"  (was {'; '.join(was)})" if was else "")
    )
    print(action_lines(answer))
    return EXIT_OK


def cmd_state(args: argparse.Namespace, url: str) -> int:
    state = fetch_json(url, "/state")
    print(compact(state) if args.json else "\n".join(state_lines(state)))
    return EXIT_OK


def cmd_map(args: argparse.Namespace, url: str) -> int:
    path = "/map" if args.map_id is None else f"/map?map_id={args.map_id}"
    payload = fetch_json(url, path)
    print(compact(payload) if args.json else "\n".join(map_lines(payload)))
    return EXIT_OK


def cmd_route(args: argparse.Namespace, url: str) -> int:
    """Which maps lie between here and somewhere else.

    Hops, never button presses. Route 4 is one map whose halves are separated
    by Mt. Moon, so "you are on Route 4" does not say which side you are on.
    """
    target = urllib.parse.quote(" ".join(args.to))
    payload = fetch_json(url, f"/route?to={target}")
    if args.json:
        print(compact(payload))
        return EXIT_OK
    hops = payload.get("hops")
    if hops is None:
        print(f"no route from {payload.get('from')} to {payload.get('to')}")
        return EXIT_OK
    if not hops:
        print(f"already on {payload.get('to')}")
        return EXIT_OK
    print(f"{payload.get('from')} to {payload.get('to')}, {len(hops)} hops:")
    for hop in hops:
        where = f" at {tuple(hop['at'])}" if hop.get("at") else ""
        edge = f" ({hop['edge']})" if hop.get("edge") else ""
        print(f"  {hop['kind']:<11}{edge:<9} -> {hop['to']}{where}")
    return EXIT_OK


def cmd_goto(args: argparse.Namespace, url: str) -> int:
    """Walk toward a map or a tile, re-planning on each map as you go."""
    target = " ".join(args.target)
    if "," in target and all(part.strip().lstrip("-").isdigit() for part in target.split(",", 1)):
        x, y = (int(part) for part in target.split(",", 1))
        payload = {"x": x, "y": y}
    else:
        payload = {"target": target}
    answer = fetch_json(url, "/goto", method="POST", payload=payload)
    if args.json:
        print(compact(answer))
        return EXIT_OK

    print(action_lines(answer))
    walked, arrived = answer.get("walked"), answer.get("arrived")
    if walked:
        print(f"walked {walked}" + (" and arrived" if arrived else ", did not arrive"))
    if not arrived and answer.get("stopped_because"):
        print(answer["stopped_because"])

    # When it could not get there, say what it *can* get to. A refusal that names
    # the one tile out of reach reads as a map problem; the same refusal with the
    # reachable exits beside it reads as a decision.
    onward = answer.get("onward") or {}
    for exit_ in (onward.get("exits") or [])[:4]:
        where = tuple(exit_["at"]) if isinstance(exit_.get("at"), list) else exit_.get("at")
        print(f"  reachable: {exit_.get('to')} at {where}, {exit_.get('steps')} steps")
    if onward.get("kind") == "unexplored" and onward.get("unseen_at"):
        print(f"  unseen ground at {tuple(onward['unseen_at'])} - walk there and look")
    return EXIT_OK


def cmd_calc(args: argparse.Namespace, url: str) -> int:
    """What each of your moves would do to what you are fighting."""
    payload = fetch_json(url, "/calc")
    if args.json:
        print(compact(payload))
        return EXIT_OK
    enemy = payload.get("enemy") or {}
    types = "/".join(enemy.get("types") or [])
    print(f"vs {enemy.get('species')} L{enemy.get('level')} {enemy.get('hp')} HP ({types})")
    for move in payload.get("moves") or []:
        damage = move.get("damage") or [0, 0]
        effect = move.get("effectiveness")
        marker = "" if effect in (1, 1.0, None) else f"  x{effect:g}"
        pp = move.get("pp")
        kos = move.get("turns_to_ko")
        # PP first, because it decides whether the rest of the line is an option
        # at all. The payload has carried it since the last time this was looked
        # at and this print dropped it, so the table went on presenting a dry
        # move as the best move: 54 of 106 auto-saved battle entries from one run
        # had at least one, and `poke fight` refused 12 of them.
        pp_text = "  --PP" if pp is None else f"  {pp:>2}PP"
        ko_text = "  out of PP" if pp == 0 else (f"  KO in {kos}" if kos else "  cannot KO")
        print(f"  {move['move']:<16}{pp_text} {damage[0]:>3}-{damage[1]:<3}{ko_text}{marker}")
    threat = payload.get("threat")
    if threat is not None:
        move = payload.get("threat_move")
        print(f"  worst incoming: {threat}" + (f" ({move})" if move else ""))
    return EXIT_OK


def cmd_frontier(args: argparse.Namespace, url: str) -> int:
    """Tiles you can reach on this map that you have never stood on.

    The payload says how far to trust the list — how many tiles the live window
    confirmed, how many are believed off the remembered map, and which of the
    two the answer rests on — and this printed none of it. So 34 of the 47
    frontier calls across every session asked for ``--json`` instead, and 35
    piped that to ``head``: the real payload is 927 bytes and the ``head -c
    400`` they reached for stops inside the tile array, before ``count``,
    ``confirmed_count``, ``believed_count`` and ``basis``. The one question the
    JSON was opened for was the one the truncation cut off.
    """
    payload = fetch_json(url, "/frontier")
    if args.json:
        print(compact(payload))
        return EXIT_OK
    tiles = payload.get("tiles") or []
    print(f"{payload.get('map')} from {tuple(payload.get('from') or ())}: {len(tiles)} unseen")
    confirmed = payload.get("confirmed_count")
    believed = payload.get("believed_count")
    if confirmed is not None or believed is not None:
        print(f"  {confirmed or 0} confirmed, {believed or 0} believed ({payload.get('basis')})")
    for tile in tiles[: args.limit]:
        print(f"  {tuple(tile)}")
    if len(tiles) > args.limit:
        print(f"  ... and {len(tiles) - args.limit} more")
    return EXIT_OK


def cmd_sim(args: argparse.Namespace, url: str) -> int:
    """Try a plan without spending it. Nothing here touches the game.

    ``batch=False``: a simulated plan is never executed, so the caps that exist
    to bound how long the server holds the emulator do not bound this. The
    server simulates from the live tile and cannot resume from a hypothetical
    one, so a long plan cannot be split and re-sent either — refusing it here
    left the model with nothing it could do but ask again.
    """
    actions = expand_actions(args.actions, batch=False)
    payload = fetch_json(url, "/sim", method="POST", payload={"actions": actions})
    if args.json:
        print(compact(payload))
        return EXIT_OK
    # Lead with the tile the plan starts on, because the model cannot choose it
    # and kept forgetting it. `sim` walks from wherever the player actually is;
    # its answer used to name only the endpoint, so a chain of sims printed a
    # column of hypothetical tiles and nothing else, and the model wrote "From
    # (14,8) check left" about a tile it had never stood on. Naming the origin
    # costs one clause of the line it is already reading.
    start = payload.get("start")
    where = ""
    if isinstance(start, (list, tuple)) and len(start) == 2:
        where = f"from {payload.get('map') or 'here'} ({start[0]},{start[1]}): "
    blocked = payload.get("blocked_at")
    if blocked is None:
        print(f"{where}clean: ends at {tuple(payload['end'])} facing {payload.get('facing')}")
    else:
        print(
            f"{where}blocked at step {blocked} ({actions[blocked]}) by "
            f"{payload.get('blocked_by')}, stops at {tuple(payload['end'])}"
        )
    if payload.get("warp_at") is not None:
        print(f"steps onto a warp at step {payload['warp_at']}")
    return EXIT_OK


def cmd_guide(args: argparse.Namespace, url: str) -> int:
    """Read a walkthrough section. Nothing is ever pushed at you; you choose."""
    if args.search:
        query = urllib.parse.quote(" ".join(args.search))
        payload = fetch_json(url, f"/guide?q={query}")
        for hit in payload.get("results") or []:
            print(f"  {hit['ref']:<38} {hit.get('summary', '')}")
        return EXIT_OK
    if args.ref:
        payload = fetch_json(url, f"/guide?ref={urllib.parse.quote(args.ref)}")
        print(payload.get("body", ""))
        return EXIT_OK
    print(fetch_json(url, "/guide").get("outline", ""))
    return EXIT_OK


def cmd_progress(args: argparse.Namespace, url: str) -> int:
    """How far through the game you are, what it cost, and what is open next.

    The `open now` block is the point of the verb. The 58 milestones form a
    graph, and this is the handful whose preconditions the game already
    satisfies: everything else is behind a road, a door or a badge you do not
    have yet. Nothing here says which one to take.
    """
    payload = fetch_json(url, "/progress")
    if args.json:
        print(compact(payload))
        return EXIT_OK
    count, total = payload.get("count", 0), payload.get("total", 0)
    print(f"{count}/{total} milestones, {payload.get('presses', 0)} presses")
    if payload.get("furthest_label"):
        print(f"furthest: {payload['furthest_label']}")
    for label in payload.get("latest") or []:
        print(f"  reached {label}")
    # Rungs this run reached and the game no longer holds, because a save was
    # reloaded past them. Named rather than silently dropped: a run that went
    # backwards is the one fact a model rereading its own progress most needs,
    # and the alternative -- leaving them in the count -- is how a session was
    # told it had a bicycle three rungs after a reload took it back.
    lost = payload.get("lost") or []
    if lost:
        names = ", ".join(str(item.get("label") or item.get("milestone_id")) for item in lost)
        print(f"reached earlier in this run, not held now: {names}")
    open_now = payload.get("frontier") or []
    if open_now:
        print(f"open now ({len(open_now)}), pick one:")
        for entry in open_now:
            gives = ", ".join(entry.get("gives") or [])
            print(f"  {entry.get('label')}" + (f" -> {gives}" if gives else ""))
    return EXIT_OK


def workspace_dir(url: str) -> Path:
    """Where the frames are written.

    The CLI runs from the workspace, so that is the first guess; the server is
    asked only when the files are not there.
    """

    here = Path(os.environ.get("POKE_WORKSPACE") or Path.cwd())
    if any((here / name).exists() for name in FRAME_FILES.values()):
        return here
    try:
        reported = (fetch_json(url, "/") or {}).get("agent_workspace_dir")
    except (ServerError, Unreachable):
        return here
    return Path(reported) if reported else here


def cmd_frame(args: argparse.Namespace, url: str) -> int:
    if args.refresh:
        target = Path(args.refresh).expanduser()
        data = fetch(url, "/screenshot")
        target.write_bytes(data)
        print(f"{target.resolve()} ({len(data)} bytes)")
        return EXIT_OK
    directory = workspace_dir(url)
    for label, name in FRAME_FILES.items():
        path = directory / name
        when = ago(path.stat().st_mtime) if path.exists() else "missing"
        print(f"{label + ':':11}{path} ({when})")
    return EXIT_OK


def cmd_save(args: argparse.Namespace, url: str) -> int:
    payload = fetch_json(url, "/save", method="POST", payload={"name": args.name})
    print(f"saved {args.name} -> {(payload.get('save') or {}).get('path')}")
    return EXIT_OK


def cmd_load(args: argparse.Namespace, url: str) -> int:
    """Load a save. The server refuses one that would cost you milestones.

    ``--force`` is how you say you meant it: recovering a branch you really have
    lost needs a load that goes backwards, and nothing else does.
    """
    payload = fetch_json(
        url,
        "/load",
        method="POST",
        payload={"name": args.name, "force": bool(getattr(args, "force", False))},
    )
    print(f"loaded {args.name} -> {(payload.get('save') or {}).get('path')}")
    return EXIT_OK


def cmd_saves(args: argparse.Namespace, url: str) -> int:
    """The saves you can go back to, newest first.

    ``named=true``, because asking for the plain list asks for the newest forty
    of everything and the harness writes an ``auto__`` checkpoint on every
    battle and every map change. With the 465 saves this run has made, all
    forty rows were autosaves and not one of the 165 named saves appeared —
    the command could not answer the only question anyone asks it. The
    autosaves are still there and ``poke load`` still takes their names; the
    count is reported so nobody thinks they are gone.
    """

    payload = fetch_json(url, "/saves?named=true") or {}
    saves = payload.get("saves") or []
    if not saves:
        print("no named saves (auto__ checkpoints are not listed)")
        return EXIT_OK
    for save in saves:
        print(f"{save.get('name')} ({ago(save.get('modified'))})")
    total = payload.get("count")
    if isinstance(total, int) and total > len(saves):
        print(f"... and {total - len(saves)} more; auto__ checkpoints are not listed")
    return EXIT_OK


def cmd_health(args: argparse.Namespace, url: str) -> int:
    payload = fetch_json(url, "/health")
    supervisor = (payload.get("pi_supervisor") or {}).get("status") or "none"
    print(
        f"{payload.get('status')} "
        f"emulator={'ready' if payload.get('emulator_ready') else 'down'} "
        f"workspace={'ready' if payload.get('agent_workspace_ready') else 'down'} "
        f"supervisor={supervisor}"
    )
    return EXIT_OK


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    # SUPPRESS, not None: a subparser copies its whole namespace over the
    # parent's, so a plain default here would erase `poke --port N state`.
    common.add_argument(
        "--port",
        default=argparse.SUPPRESS,
        help=f"server port (default $PORT, else {DEFAULT_PORT})",
    )
    common.add_argument(
        "--url",
        default=argparse.SUPPRESS,
        help="full server base URL, overrides --port",
    )

    parser = argparse.ArgumentParser(
        prog="poke",
        description="Play the game: press buttons, read state, manage saves.",
        parents=[common],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    act = subparsers.add_parser(
        "act",
        parents=[common],
        help="send actions, e.g. poke act up up a",
        description=action_help(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    act.add_argument("actions", nargs="+", metavar="ACTION")
    act.add_argument("--json", action="store_true", help="the whole payload instead of a summary")
    act.set_defaults(func=cmd_act)

    fight = subparsers.add_parser(
        "fight",
        parents=[common],
        help="attack with a named move, e.g. poke fight ember",
        description=(
            "Attack with a move, by name. Case does not matter and a unique prefix "
            "is enough, so `poke fight ember` and `poke fight emb` are the same call.\n"
            "The move menu remembers where its cursor was last turn and wraps at both "
            "ends, so the server reads the cursor and steps it onto the move you named "
            "rather than guessing. It refuses if you are not in a battle, if nothing "
            "is called that, or if the move is out of PP."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    fight.add_argument("move", nargs="+", metavar="MOVE")
    fight.add_argument("--json", action="store_true", help="the whole payload instead of a summary")
    fight.set_defaults(func=cmd_fight)

    run = subparsers.add_parser("run", parents=[common], help="flee the current battle")
    run.add_argument("--json", action="store_true", help="the whole payload instead of a summary")
    run.set_defaults(func=cmd_run)

    catch = subparsers.add_parser(
        "catch",
        parents=[common],
        help="throw a ball at the wild Pokemon, e.g. poke catch",
        description=(
            "Throw a ball. With no ball named it throws the weakest one you carry, "
            "so a Poke Ball goes before a Great Ball and the Master Ball is never "
            "spent by accident; name one to override.\n"
            "The odds are already on every wild battle frame, under `catch`: what the "
            "throw does now and what it does once the Pokemon is worn down. It refuses "
            "in a trainer battle, where the ball bounces off and the turn is wasted."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    catch.add_argument("ball", nargs="*", metavar="BALL")
    catch.add_argument("--json", action="store_true", help="the whole payload instead of a summary")
    catch.set_defaults(func=cmd_catch)

    item = subparsers.add_parser(
        "item",
        parents=[common],
        help="use a healing item in the battle you are in, e.g. poke item potion",
        description=(
            "Use an item on one of your Pokemon without leaving the battle. With no "
            "item named it uses the weakest healing item you carry, so a Potion goes "
            "before a Hyper Potion and a Full Restore is not spent on a scratch.\n"
            "What each one would restore is already on any battle frame whose Pokemon "
            "is hurt, under `items`: the item, what it puts back, and the HP it leaves "
            "you on. `--on` picks a party slot; the default is the Pokemon on the field."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    item.add_argument("item", nargs="*", metavar="ITEM")
    item.add_argument(
        "--on",
        type=int,
        default=None,
        metavar="SLOT",
        help="party slot to use it on, 1 for the lead (default: the Pokemon on the field)",
    )
    item.add_argument("--json", action="store_true", help="the whole payload instead of a summary")
    item.set_defaults(func=cmd_item)

    buy = subparsers.add_parser(
        "buy",
        parents=[common],
        help="buy from the mart you are standing in, e.g. poke buy poke ball 10",
        description=(
            "Buy from the counter on this map. A unique prefix is enough, so "
            "`poke buy poke` and `poke buy 'poke ball'` are the same call.\n"
            "You do not have to be at the till: it walks to the counter, talks to the "
            "clerk, picks the quantity and confirms, then backs out to the overworld. "
            "It refuses if the map is not a mart, if this counter does not stock the "
            "item, or if the money will not cover it — and says what it will cover."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    buy.add_argument("item", nargs="+", metavar="ITEM [COUNT]")
    buy.add_argument("--json", action="store_true", help="the whole payload instead of a summary")
    buy.set_defaults(func=cmd_buy)

    heal = subparsers.add_parser(
        "heal",
        parents=[common],
        help="heal at the nurse on this map, e.g. poke heal",
        description=(
            "Heal the whole party at the nurse on this map: HP, PP and status.\n"
            "You do not have to be at her counter: it walks there, talks to her, "
            "answers YES and reads the conversation out, then checks the party came "
            "back full. Her counter is a talk-over tile, so the walk ends two tiles "
            "out — which is the part that cost one run 4,152 presses to find.\n"
            "It refuses on a map with no nurse, and on a party that is already full."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    heal.add_argument("--json", action="store_true", help="the whole payload instead of a summary")
    heal.set_defaults(func=cmd_heal)

    state = subparsers.add_parser("state", parents=[common], help="party, bag, badges, position")
    state.add_argument("--json", action="store_true", help="raw payload")
    state.set_defaults(func=cmd_state)

    map_command = subparsers.add_parser(
        "map", parents=[common], help="whole-map summary and PNG path"
    )
    map_command.add_argument("--map-id", type=int, help="another map you have visited")
    map_command.add_argument("--json", action="store_true", help="raw payload")
    map_command.set_defaults(func=cmd_map)

    frame = subparsers.add_parser("frame", parents=[common], help="paths of the workspace frames")
    frame.add_argument(
        "--refresh",
        nargs="?",
        const="fresh_frame.png",
        metavar="PATH",
        help="fetch a fresh screenshot to PATH instead",
    )
    frame.set_defaults(func=cmd_frame)

    save = subparsers.add_parser("save", parents=[common], help="save state under a name")
    save.add_argument("name")
    save.set_defaults(func=cmd_save)

    load = subparsers.add_parser("load", parents=[common], help="load a named save")
    load.add_argument("name")
    load.add_argument(
        "--force",
        action="store_true",
        help="load even if that save is behind the game you have now",
    )
    load.set_defaults(func=cmd_load)

    saves = subparsers.add_parser("saves", parents=[common], help="list saves")
    saves.set_defaults(func=cmd_saves)

    health = subparsers.add_parser("health", parents=[common], help="is the server answering")
    health.set_defaults(func=cmd_health)

    route = subparsers.add_parser(
        "route", parents=[common], help="which maps lie between here and somewhere"
    )
    route.add_argument("to", nargs="+", help="destination map name")
    route.add_argument("--json", action="store_true")
    route.set_defaults(func=cmd_route)

    goto = subparsers.add_parser("goto", parents=[common], help="walk to a map or a tile")
    goto.add_argument("target", nargs="+", help="map name, or x,y on this map")
    goto.add_argument("--json", action="store_true", help="the whole payload")
    goto.set_defaults(func=cmd_goto)

    calc = subparsers.add_parser(
        "calc", parents=[common], help="damage each of your moves would do right now"
    )
    calc.add_argument("--json", action="store_true")
    calc.set_defaults(func=cmd_calc)

    frontier = subparsers.add_parser(
        "frontier", parents=[common], help="reachable tiles on this map you have not seen"
    )
    frontier.add_argument("--limit", type=int, default=12)
    frontier.add_argument("--json", action="store_true")
    frontier.set_defaults(func=cmd_frontier)

    sim = subparsers.add_parser(
        "sim", parents=[common], help="try a plan without sending it to the game"
    )
    sim.add_argument("actions", nargs="+")
    sim.add_argument("--json", action="store_true")
    sim.set_defaults(func=cmd_sim)

    guide = subparsers.add_parser(
        "guide", parents=[common], help="walkthrough sections; no argument lists them"
    )
    guide.add_argument("ref", nargs="?", help="a section reference, guide/slug")
    guide.add_argument("-s", "--search", nargs="+", help="find sections by keyword")
    guide.set_defaults(func=cmd_guide)

    progress = subparsers.add_parser(
        "progress", parents=[common], help="milestones reached and presses spent"
    )
    progress.add_argument("--json", action="store_true")
    progress.set_defaults(func=cmd_progress)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    url = base_url(args)
    try:
        return args.func(args, url)
    except ActionError as error:
        print(str(error), file=sys.stderr)
        return EXIT_BAD_USAGE
    except ServerError as error:
        print(f"error {error.status}: {error.detail}", file=sys.stderr)
        return EXIT_HTTP_ERROR
    except Unreachable as error:
        print(f"server not answering {server_label(args)}: {error}", file=sys.stderr)
        return EXIT_NO_SERVER


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        # `poke saves | head` closes the pipe early. Python would print a
        # teardown traceback into the agent's context for it; nothing is wrong.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(EXIT_OK)
