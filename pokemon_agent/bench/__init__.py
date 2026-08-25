"""The measurement layer: what a run cost, in buttons.

Every harness change so far has been argued from a transcript and an impression.
This package replaces that with a number. A run writes one receipt per agent
action batch through :class:`RunRegistry`; :func:`compute` folds those receipts
into :class:`RunMetrics`, whose headline is ``presses_to`` — the cumulative
button presses at the first attainment of each milestone, the same currency
published results are quoted in.

::

    registry = RunRegistry(Path("~/.pokemon-agent"))
    run_id = registry.start_run(harness_sha=..., config_hash=..., model=...,
                                start_checkpoint=None, goal="reach Pewter")
    registry.append(run_id, {"presses": 6, "map": "Route 3", "pos": [12, 8], ...})
    registry.finish(run_id, "context exhausted")
    print(format_run(compute(registry.load(run_id))))
"""

from pokemon_agent.bench.metrics import (
    REFERENCE_POINTS,
    Attainment,
    LadderEntry,
    MapRevisit,
    RunMetrics,
    compute,
    load_ladder,
)
from pokemon_agent.bench.registry import (
    DEFAULT_DATA_DIR,
    META_FILENAME,
    RECEIPTS_FILENAME,
    RUNS_DIRNAME,
    SCHEMA_VERSION,
    Receipt,
    RunMeta,
    RunRecord,
    RunRegistry,
    RunSummary,
)
from pokemon_agent.bench.report import format_comparison, format_run, format_run_list

__all__ = [
    "Attainment",
    "DEFAULT_DATA_DIR",
    "LadderEntry",
    "MapRevisit",
    "META_FILENAME",
    "RECEIPTS_FILENAME",
    "REFERENCE_POINTS",
    "RUNS_DIRNAME",
    "Receipt",
    "RunMeta",
    "RunMetrics",
    "RunRecord",
    "RunRegistry",
    "RunSummary",
    "SCHEMA_VERSION",
    "compute",
    "format_comparison",
    "format_run",
    "format_run_list",
    "load_ladder",
]
