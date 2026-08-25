"""``python -m pokemon_agent.bench`` — read the scoreboard off disk.

python -m pokemon_agent.bench                     # every run in the store
python -m pokemon_agent.bench <run_id>            # one run, scored
python -m pokemon_agent.bench --compare a b c     # side by side
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from pokemon_agent.bench.metrics import compute
from pokemon_agent.bench.registry import DEFAULT_DATA_DIR, RunRegistry
from pokemon_agent.bench.report import format_comparison, format_run, format_run_list

#: Where the store lives when nothing on the command line says otherwise.
DATA_DIR_ENV = "POKE_BENCH_DIR"


def default_data_dir() -> Path:
    return Path(os.environ.get(DATA_DIR_ENV) or DEFAULT_DATA_DIR).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pokemon_agent.bench",
        description="Button presses to each milestone, per run.",
    )
    parser.add_argument("run_id", nargs="?", help="Run to print. Omit to list every run.")
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="RUN_ID",
        help="Two or more runs, side by side per milestone.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=f"Store location (default: ${DATA_DIR_ENV} or {DEFAULT_DATA_DIR}).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else default_data_dir()
    registry = RunRegistry(data_dir)

    try:
        if args.compare:
            pairs = [(run_id, compute(registry.load(run_id))) for run_id in args.compare]
            sys.stdout.write(format_comparison(pairs))
            return 0
        if args.run_id:
            sys.stdout.write(format_run(compute(registry.load(args.run_id))))
            return 0
    except FileNotFoundError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    summaries = registry.list_runs()
    if not summaries:
        sys.stderr.write(f"No runs under {registry.runs_dir}.\n")
        return 1
    sys.stdout.write(format_run_list(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
