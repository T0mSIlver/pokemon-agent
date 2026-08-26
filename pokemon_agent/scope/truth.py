"""Ground truth, for checking what the model said against what is actually there.

Every other module in ``scope`` counts what happened. This one is the only place
that knows what *should* have happened: the 223 maps of Pokemon Red, their
sizes, their warps, the edges between them, and the verbs the harness ships.

All of it is already in the tree — ``pokemon_agent.gamedata`` reads the tables
generated from the pret/pokered decompilation, ``pokemon_agent.world`` walks the
map graph, ``pokemon_agent.agent_cli`` defines the verbs. Nothing is duplicated
here; this module only makes those sources cheap to ask, cached for the life of
the process, and safe to ask from a report that must never crash because a data
file is missing. Every accessor degrades to "I don't know", which the callers
render as *unchecked* rather than as *false* — an absent table must never turn
into an accusation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

Coord = tuple[int, int]


@dataclass(frozen=True)
class Warp:
    """One tile that moves the player to another map."""

    at: Coord
    to_map: str


@dataclass(frozen=True)
class MapTruth:
    """What the game data says about one map."""

    name: str
    width: int = 0
    height: int = 0
    warps: tuple[Warp, ...] = ()
    #: ``{"east": "Cerulean City"}`` — walking off that edge changes map.
    connections: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.connections is None:
            object.__setattr__(self, "connections", {})

    @property
    def exits(self) -> tuple[str, ...]:
        """Every other map reachable in one hop, warps and edges together."""

        out = {warp.to_map for warp in self.warps if warp.to_map}
        out.update(name for name in self.connections.values() if name)
        return tuple(sorted(out))

    def contains(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def warp_at(self, x: int, y: int) -> Optional[Warp]:
        for warp in self.warps:
            if warp.at == (x, y):
                return warp
        return None


@lru_cache(maxsize=1)
def _world_payload() -> dict[str, Any]:
    """``{map name: raw record}``, or ``{}`` when the tables are not installed."""

    try:
        from pokemon_agent import gamedata
    except Exception:  # noqa: BLE001 — a missing table must not stop a report
        return {}
    try:
        payload = gamedata.world()
    except Exception:  # noqa: BLE001
        return {}
    return payload if isinstance(payload, dict) else {}


def _connection_name(value: Any) -> str:
    """A connection is a bare map name in the current tables, a dict in older ones."""

    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("to_map", "map", "name"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    return ""


@lru_cache(maxsize=1)
def maps() -> dict[str, MapTruth]:
    """Every map the game has, by the same name the receipts use."""

    out: dict[str, MapTruth] = {}
    for name, record in _world_payload().items():
        if not isinstance(record, dict):
            continue
        size = record.get("size")
        width, height = (0, 0)
        if isinstance(size, (list, tuple)) and len(size) == 2:
            try:
                width, height = int(size[0]), int(size[1])
            except (TypeError, ValueError):
                width, height = 0, 0
        warps: list[Warp] = []
        for item in record.get("warps") or ():
            if not isinstance(item, dict):
                continue
            try:
                at = (int(item["x"]), int(item["y"]))
            except (KeyError, TypeError, ValueError):
                continue
            warps.append(Warp(at=at, to_map=str(item.get("to_map") or "")))
        connections = {
            str(edge): _connection_name(value)
            for edge, value in (record.get("connections") or {}).items()
        }
        out[str(name)] = MapTruth(
            name=str(name),
            width=width,
            height=height,
            warps=tuple(warps),
            connections={edge: name for edge, name in connections.items() if name},
        )
    return out


def map_truth(name: Optional[str]) -> Optional[MapTruth]:
    """What is known about ``name``, or ``None`` if the game has no such map."""

    if not name:
        return None
    return maps().get(name)


def known_maps() -> int:
    return len(maps())


@lru_cache(maxsize=1)
def _router() -> Any:
    try:
        from pokemon_agent.world import World

        return World.load()
    except Exception:  # noqa: BLE001 — the router is a bonus, never a requirement
        return None


@lru_cache(maxsize=4096)
def hops(from_map: str, to_map: str) -> Optional[tuple[str, ...]]:
    """Map names along the graph route, or ``None`` when there is no route.

    This is the same answer ``poke route`` gives the model, which is the point:
    a report about whether the model went the right way should be judging it
    against the advice it was actually offered.
    """

    router = _router()
    if router is None or not from_map or not to_map:
        return None
    try:
        route = router.route(from_map, to_map)
    except Exception:  # noqa: BLE001
        return None
    if route is None:
        return None
    names: list[str] = [from_map]
    for hop in route:
        target = getattr(hop, "to_map", None)
        if isinstance(target, str) and target:
            names.append(target)
    return tuple(names)


def hop_distance(from_map: str, to_map: str) -> Optional[int]:
    path = hops(from_map, to_map)
    return None if path is None else max(0, len(path) - 1)


@lru_cache(maxsize=1)
def harness_verbs() -> tuple[str, ...]:
    """Every ``poke`` subcommand the CLI defines, asked of the CLI itself.

    Hard-coding the list would answer "what did it never try?" with a list that
    silently goes stale the next time a verb ships, which is the exact failure
    the question exists to catch.
    """

    try:
        from pokemon_agent import agent_cli

        parser = agent_cli.build_parser()
    except Exception:  # noqa: BLE001
        return ()
    out: set[str] = set()
    subparsers = getattr(parser, "_subparsers", None)
    for action in getattr(subparsers, "_group_actions", ()) or ():
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict):
            out.update(str(name) for name in choices)
    return tuple(sorted(out))
