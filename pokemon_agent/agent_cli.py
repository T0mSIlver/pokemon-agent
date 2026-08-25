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


def expand_actions(tokens: list[str]) -> list[str]:
    """Tokens as typed into the action list the server expects.

    Handles aliases and the ``name:count`` repeat form. Raises on anything it
    does not recognise, so a typo costs nothing but the error.
    """

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
            if count > MAX_REPEAT:
                raise ActionError(f"{token!r} repeats more than {MAX_REPEAT} times")
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
    if len(actions) > MAX_ACTIONS_PER_BATCH:
        raise ActionError(
            f"that batch is {len(actions)} actions; the limit is "
            f"{MAX_ACTIONS_PER_BATCH}. Send fewer and look at the frame after."
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
    lines = [f"{map_name} ({position.get('x')},{position.get('y')}) facing {player.get('facing')}"]

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
        lines.append("warps: " + " ".join(f"({w.get('x')},{w.get('y')})" for w in warps))
    nearest = payload.get("unexplored_nearest")
    if nearest:
        lines.append(f"nearest unexplored: ({nearest.get('x')},{nearest.get('y')})")
    if payload.get("image_path"):
        lines.append(f"png: {payload['image_path']}")
    return lines


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_act(args: argparse.Namespace, url: str) -> int:
    actions = expand_actions(args.actions)
    print(compact(fetch_json(url, "/action", method="POST", payload={"actions": actions})))
    return EXIT_OK


def cmd_fight(args: argparse.Namespace, url: str) -> int:
    """Attack by name. The server does the menu work; this only carries the name."""
    payload = {"move": " ".join(args.move)}
    print(compact(fetch_json(url, "/battle/fight", method="POST", payload=payload)))
    return EXIT_OK


def cmd_run(args: argparse.Namespace, url: str) -> int:
    print(compact(fetch_json(url, "/battle/run", method="POST")))
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
    print(compact(fetch_json(url, "/goto", method="POST", payload=payload)))
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
        kos = move.get("turns_to_ko")
        ko_text = f"  KO in {kos}" if kos else "  cannot KO"
        print(f"  {move['move']:<16} {damage[0]:>3}-{damage[1]:<3}{ko_text}{marker}")
    threat = payload.get("threat")
    if threat is not None:
        print(f"  worst incoming: {threat}")
    return EXIT_OK


def cmd_frontier(args: argparse.Namespace, url: str) -> int:
    """Tiles you can reach on this map that you have never stood on."""
    payload = fetch_json(url, "/frontier")
    if args.json:
        print(compact(payload))
        return EXIT_OK
    tiles = payload.get("tiles") or []
    print(f"{payload.get('map')} from {tuple(payload.get('from') or ())}: {len(tiles)} unseen")
    for tile in tiles[: args.limit]:
        print(f"  {tuple(tile)}")
    if len(tiles) > args.limit:
        print(f"  ... and {len(tiles) - args.limit} more")
    return EXIT_OK


def cmd_sim(args: argparse.Namespace, url: str) -> int:
    """Try a plan without spending it. Nothing here touches the game."""
    actions = expand_actions(args.actions)
    payload = fetch_json(url, "/sim", method="POST", payload={"actions": actions})
    if args.json:
        print(compact(payload))
        return EXIT_OK
    blocked = payload.get("blocked_at")
    if blocked is None:
        print(f"clean: ends at {tuple(payload['end'])} facing {payload.get('facing')}")
    else:
        print(
            f"blocked at step {blocked} ({actions[blocked]}) by "
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
    """How far through the game you are, and what it has cost so far."""
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
    payload = fetch_json(url, "/load", method="POST", payload={"name": args.name})
    print(f"loaded {args.name} -> {(payload.get('save') or {}).get('path')}")
    return EXIT_OK


def cmd_saves(args: argparse.Namespace, url: str) -> int:
    saves = (fetch_json(url, "/saves") or {}).get("saves") or []
    if not saves:
        print("no saves")
        return EXIT_OK
    for save in saves:
        print(f"{save.get('name')} ({ago(save.get('modified'))})")
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
    fight.set_defaults(func=cmd_fight)

    run = subparsers.add_parser("run", parents=[common], help="flee the current battle")
    run.set_defaults(func=cmd_run)

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
