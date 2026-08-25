"""``python -m pokemon_agent.scope <command>`` — one question, one screen.

    live                  what is happening right now
    tools                 which verbs the model reaches for, and how they go
    waste                 where the presses went
    loops                 command sequences that repeat
    context               what is filling the 140k window
    session [id]          a digest of one session
    timeline              milestones, cumulative presses, wall clock
    diff <run_a> <run_b>  two runs side by side
    where                 which workspace and run store were discovered

``--json`` on any of them prints the same figures as data; ``--full`` lifts the
row caps. Everything is read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from pokemon_agent.scope import commands
from pokemon_agent.scope.analysis import DEFAULT_WINDOW
from pokemon_agent.scope.commands import Context, ScopeError
from pokemon_agent.scope.discover import discover

COMMANDS = ("live", "tools", "waste", "loops", "context", "session", "timeline", "diff", "where")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pokemon_agent.scope",
        description="Read a live Pokemon run the way a supervising agent has to.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("command", choices=COMMANDS, help="what to answer")
    parser.add_argument(
        "args",
        nargs="*",
        help="session id for 'session'; two run ids for 'diff'",
    )
    parser.add_argument("--workspace", default=None, help="agent workspace (default: discovered)")
    parser.add_argument("--data-dir", default=None, help="run store parent (default: discovered)")
    parser.add_argument("--session", default=None, help="session id or filename fragment")
    parser.add_argument("--run", default=None, help="run id (default: runs/CURRENT)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--full", action="store_true", help="lift the row caps")
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"context window size for the occupancy figure (default: {DEFAULT_WINDOW})",
    )
    return parser


def run(args: argparse.Namespace) -> tuple[list[str], dict]:
    paths = discover(args.workspace, args.data_dir)
    session_id = args.session
    if args.command == "session" and args.args:
        session_id = args.args[0]
    ctx = Context(
        paths=paths,
        session_id=session_id,
        run_id=args.run,
        full=args.full,
        window=args.window,
    )
    if args.command == "diff":
        if len(args.args) != 2:
            raise ScopeError("diff needs exactly two run ids: scope diff <run_a> <run_b>")
        return commands.command_diff(ctx, args.args[0], args.args[1])
    handler = {
        "live": commands.command_live,
        "tools": commands.command_tools,
        "waste": commands.command_waste,
        "loops": commands.command_loops,
        "context": commands.command_context,
        "session": commands.command_session,
        "timeline": commands.command_timeline,
        "where": commands.command_where,
    }[args.command]
    return handler(ctx)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        lines, payload = run(args)
    except ScopeError as exc:
        sys.stderr.write(f"scope: {exc}\n")
        return 2
    if args.json:
        sys.stdout.write(json.dumps(payload, indent=2, default=str) + "\n")
    else:
        sys.stdout.write("\n".join(lines).rstrip() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
