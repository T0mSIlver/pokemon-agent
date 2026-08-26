"""Persistent per-map memory of what the agent has actually seen and walked.

The agent only ever sees a 10x9 window — 5.5% of Viridian Forest — and forgets
it between turns, so it re-treads ground it has already covered. This module
accumulates every window it is shown into a whole-map record that survives
server restarts and the fresh supervisor sessions the watchdog starts.

It is a PULL layer: `GET /map` summarises it and points at a rendered PNG.
Nothing here is ever pushed into the per-action response.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

from PIL import Image, ImageDraw, ImageFont

try:  # The canonical map table is data, not a hard dependency of the store.
    from pokemon_agent import gamedata as _gamedata
except Exception:  # noqa: BLE001 — a missing data file must not break the store
    _gamedata = None  # type: ignore[assignment]

Coord = Tuple[int, int]

STORE_VERSION = 1

# Tile paints lifted from `render_navigation_overlay`, so the whole-map picture
# and the annotated frame speak one colour language.
COLOR_UNKNOWN = (7, 10, 16)
COLOR_SEEN = (24, 123, 73)
COLOR_WALKED = (13, 68, 41)
COLOR_WALL = (180, 58, 58)
COLOR_WARP = (213, 80, 255)
COLOR_PLAYER = (55, 208, 255)
COLOR_AXIS = (90, 105, 125)
COLOR_LABEL = (165, 180, 196)
COLOR_TITLE = (255, 255, 255)
COLOR_BORDER = (255, 138, 61)

LEGEND = {
    "cyan": "you are here",
    "purple": "warp (step onto it, then walk in the exit direction)",
    "dark green": "you have walked here",
    "green": "seen and passable, never walked",
    "red": "wall or blocked tile",
    "black": "never seen",
}

#: What the long edge of a rendered map aims for, in pixels. The frames the
#: agent already reads are 160x144 upscaled, so this sits in the same league.
TARGET_LONG_EDGE_PX = 336

#: Outside this range a tile is either unreadable or a waste of pixels.
MIN_TILE_PX = 4
MAX_TILE_PX = 16

_TITLE_BAND_PX = 14
_TICK_BAND_PX = 13
_LEFT_MARGIN_PX = 24
_EDGE_PAD_PX = 5

#: A tick every 5 tiles and a number every 10: dense enough to place a
#: coordinate, sparse enough that the labels never collide at 7 pixels a tile.
_TICK_EVERY = 5
_LABEL_EVERY = 10


#: Octant boundaries at 22.5 degrees: tan(22.5 deg) = 0.414, so 1 / 0.414.


#: Tilesets the game only ever draws single rooms with. The largest such map in
#: Red is a 20x18 gym, so anything bigger claiming one of these is not that map.
ROOM_TILESETS = frozenset(
    {
        "REDS_HOUSE_1",
        "REDS_HOUSE_2",
        "MART",
        "DOJO",
        "POKECENTER",
        "GYM",
        "HOUSE",
        "MUSEUM",
        "LAB",
        "CLUB",
    }
)
ROOM_MAX_TILES = 20

#: The smallest outdoor map in the game (Pallet Town, Cinnabar Island, Route 7).
OVERWORLD_MIN_TILES = (20, 18)


def _log(message: str) -> None:
    print(f"[explored-map] {message}")


class CanonicalMap(NamedTuple):
    """A map's real geometry, as the pokered decompilation records it."""

    name: str
    width: int
    height: int
    warps: FrozenSet[Coord]


_CANONICAL: Optional[Dict[int, CanonicalMap]] = None


def canonical_maps() -> Dict[int, CanonicalMap]:
    """Real map geometry by map id, from ``data/game/world.json``.

    Empty — never an exception — when the data file is missing, so the store
    degrades to its own self-consistency checks rather than refusing to run.
    """
    global _CANONICAL
    if _CANONICAL is not None:
        return _CANONICAL
    table: Dict[int, CanonicalMap] = {}
    try:
        world = _gamedata.world() if _gamedata is not None else {}
        for name, entry in world.items():
            size = entry.get("size") or []
            if len(size) != 2:
                continue
            table[int(entry["map_id"])] = CanonicalMap(
                name=str(name),
                width=int(size[0]),
                height=int(size[1]),
                warps=frozenset(
                    (int(warp["x"]), int(warp["y"]))
                    for warp in entry.get("warps") or []
                    if isinstance(warp, dict) and "x" in warp and "y" in warp
                ),
            )
    except Exception as exc:  # noqa: BLE001 — unreadable data is not a crash
        _log(f"canonical map sizes unavailable: {exc}")
        table = {}
    _CANONICAL = table
    return _CANONICAL


def _label_warps(map_name: str, coords: Sequence[Coord]) -> List[Dict[str, object]]:
    """Attach each warp's destination, because eight bare coordinates say nothing.

    On Mt Moon B1F this used to print

        warps: (5,5) (13,27) (17,11) (21,17) (23,3) (25,9) (25,15) (27,3)

    Seven of those are ladders further into the mountain and one, (27,3), is the
    only way out, to Route 4, which connects east to Cerulean City -- the goal.
    Nothing in that line distinguished them. The run spent 8,387 presses, 56% of
    itself, inside Mt. Moon, stood on B1F thirty separate times, and never got
    closer to (27,3) than (25,8).

    The destination comes from the generated map table, so a warp the player has
    seen but never taken is still named. Unknown stays unlabelled rather than
    guessed.
    """
    try:
        from pokemon_agent import gamedata

        record = gamedata.world().get(map_name) or {}
        known = {
            (w.get("x"), w.get("y")): w.get("to_map")
            for w in (record.get("warps") or [])
            if w.get("to_map")
        }
    except Exception:
        known = {}

    labelled: List[Dict[str, object]] = []
    for x, y in coords:
        entry: Dict[str, object] = {"x": x, "y": y}
        target = known.get((x, y))
        if target:
            entry["to"] = target
        labelled.append(entry)
    return labelled


def _inside(coord: Coord, width: int, height: int) -> bool:
    return 0 <= coord[0] < width and 0 <= coord[1] < height


def declared_size(snapshot: dict) -> Optional[Coord]:
    """The map size a snapshot claims, or None if it does not claim one."""
    dimensions = snapshot.get("map_dimensions")
    if not isinstance(dimensions, dict):
        return None
    try:
        width = int(dimensions.get("width") or 0)
        height = int(dimensions.get("height") or 0)
    except (TypeError, ValueError):
        return None
    return (width, height) if width > 0 and height > 0 else None


def incoherence(snapshot: dict, map_id: int, known_size: Optional[Coord] = None) -> Optional[str]:
    """Why this snapshot cannot be describing `map_id`, or None if it can.

    A snapshot read while the game is still loading a map carries the previous
    map's geometry under the new map's id. It contradicts either itself or the
    game's own map table, and merging it corrupts the record permanently — so
    it is rejected outright rather than folded in.

    `known_size` is what the store already believes the map measures, used to
    place coordinates from a snapshot that declares no dimensions of its own.
    """
    declared = declared_size(snapshot)
    size = declared if declared is not None else known_size
    dimensions = snapshot.get("map_dimensions")
    if declared is not None and isinstance(dimensions, dict):
        width_blocks = dimensions.get("width_blocks")
        height_blocks = dimensions.get("height_blocks")
        # A map is laid out in 2x2-tile blocks; the two units cannot disagree.
        if width_blocks is not None and int(width_blocks) * 2 != declared[0]:
            return f"{declared[0]} tiles wide but {width_blocks} blocks"
        if height_blocks is not None and int(height_blocks) * 2 != declared[1]:
            return f"{declared[1]} tiles high but {height_blocks} blocks"

    canonical = canonical_maps().get(map_id)
    if declared is not None and canonical is not None:
        if declared != (canonical.width, canonical.height):
            return (
                f"declared {declared[0]}x{declared[1]} but {canonical.name} is "
                f"{canonical.width}x{canonical.height}"
            )

    if size is not None:
        player = _as_coord(snapshot.get("player_position"))
        if player is not None and not _inside(player, *size):
            return f"player {player} outside {size[0]}x{size[1]}"
        for warp in snapshot.get("warps") or []:
            coord = _as_coord(warp)
            if coord is not None and not _inside(coord, *size):
                return f"warp {coord} outside {size[0]}x{size[1]}"

        tileset = str(snapshot.get("tileset") or "")
        if tileset in ROOM_TILESETS and max(size) > ROOM_MAX_TILES:
            return f"{size[0]}x{size[1]} is too big for the {tileset} tileset"
        if tileset == "OVERWORLD" and (
            size[0] < OVERWORLD_MIN_TILES[0] or size[1] < OVERWORLD_MIN_TILES[1]
        ):
            return f"{size[0]}x{size[1]} is too small for the OVERWORLD tileset"
    return None


def _as_coord(value: object) -> Optional[Coord]:
    """Read an {x, y} mapping — or an {x, y}-bearing wrapper — as a coordinate."""
    if not isinstance(value, dict):
        return None
    source = value
    nested = value.get("coord")
    if isinstance(nested, dict):
        source = nested
    x = source.get("x")
    y = source.get("y")
    if x is None or y is None:
        return None
    try:
        return int(x), int(y)
    except (TypeError, ValueError):
        return None


def _pack_rows(points: Set[Coord], width: int, height: int) -> List[str]:
    """Pack a tile set into one hex bitmask per row; an all-zero row packs to ""."""
    if width <= 0 or height <= 0:
        return []
    by_row: Dict[int, Set[int]] = {}
    for x, y in points:
        if 0 <= x < width and 0 <= y < height:
            by_row.setdefault(y, set()).add(x)
    nibbles = (width + 3) // 4
    rows: List[str] = []
    for y in range(height):
        xs = by_row.get(y)
        if not xs:
            rows.append("")
            continue
        value = 0
        for x in xs:
            value |= 1 << (width - 1 - x)
        rows.append(f"{value:0{nibbles}x}")
    return rows


def _unpack_rows(rows: Iterable[object], width: int) -> Set[Coord]:
    points: Set[Coord] = set()
    if width <= 0:
        return points
    for y, row in enumerate(rows):
        if not isinstance(row, str) or not row:
            continue
        value = int(row, 16)
        for x in range(width):
            if (value >> (width - 1 - x)) & 1:
                points.add((x, y))
    return points


def _auto_tile_px(width: int, height: int) -> int:
    """Pixels per tile that put the map's long edge near the target size."""
    longest = max(1, width, height)
    return max(MIN_TILE_PX, min(MAX_TILE_PX, round(TARGET_LONG_EDGE_PX / longest)))


def _fit(draw: "ImageDraw.ImageDraw", text: str, font, max_width: int) -> str:
    """Trim `text` until it fits, so a long map name cannot run off the canvas."""
    while text and draw.textlength(text, font=font) > max_width:
        text = text[:-1]
    return text


class _MapRecord:
    """Everything known about one map id."""

    __slots__ = (
        "map_id",
        "map_name",
        "width",
        "height",
        "seen",
        "walkable",
        "visits",
        "warps",
        "player",
    )

    def __init__(self, map_id: int, map_name: str = "") -> None:
        self.map_id = map_id
        self.map_name = map_name
        self.width = 0
        self.height = 0
        self.seen: Set[Coord] = set()
        self.walkable: Set[Coord] = set()
        # Tile -> how many separate arrivals. A tile stood on 24 times is the
        # single loudest signal that the agent is going in circles.
        self.visits: Dict[Coord, int] = {}
        self.warps: Set[Coord] = set()
        self.player: Optional[Coord] = None

    @property
    def walked(self):
        """The tiles arrived at, as a set-like view over the visit counts."""
        return self.visits.keys()

    def grow_to(self, x: int, y: int) -> None:
        self.width = max(self.width, x + 1)
        self.height = max(self.height, y + 1)

    def resize(self, width: int, height: int) -> int:
        """Adopt a new real size, dropping everything that falls outside it.

        Size is metadata the game knows exactly, not something to accumulate: a
        tile beyond the new bounds was learned from another map's geometry and
        cannot be on this one. Returns how many seen tiles that dropped.
        """
        self.width = width
        self.height = height
        before = len(self.seen)
        self.seen = {coord for coord in self.seen if _inside(coord, width, height)}
        self.walkable = {coord for coord in self.walkable if _inside(coord, width, height)}
        self.visits = {
            coord: count for coord, count in self.visits.items() if _inside(coord, width, height)
        }
        self.warps = {coord for coord in self.warps if _inside(coord, width, height)}
        if self.player is not None and not _inside(self.player, width, height):
            self.player = None
        return before - len(self.seen)

    def note(self, coord: Coord, *, passable: bool) -> None:
        # `seen` and `walkable` are both monotone unions: a tile seen passable
        # once stays passable, so a blocker that was only there for one frame
        # cannot leave a permanent phantom wall behind.
        self.seen.add(coord)
        if passable:
            self.walkable.add(coord)
        self.grow_to(*coord)

    def to_json(self) -> dict:
        return {
            "map_name": self.map_name,
            "width": self.width,
            "height": self.height,
            "player": {"x": self.player[0], "y": self.player[1]} if self.player else None,
            "seen": _pack_rows(self.seen, self.width, self.height),
            "walkable": _pack_rows(self.walkable, self.width, self.height),
            "walked": _pack_rows(set(self.visits), self.width, self.height),
            # Only the revisited tiles need a number; `walked` already implies 1.
            "visit_counts": sorted(
                [x, y, count] for (x, y), count in self.visits.items() if count > 1
            ),
            "warps": sorted([x, y] for x, y in self.warps),
        }

    @classmethod
    def from_json(cls, map_id: int, payload: dict) -> "_MapRecord":
        if not isinstance(payload, dict):
            raise TypeError(f"map {map_id} is not an object")
        record = cls(map_id, str(payload.get("map_name") or ""))
        record.width = int(payload.get("width") or 0)
        record.height = int(payload.get("height") or 0)
        record.seen = _unpack_rows(payload.get("seen") or [], record.width)
        record.walkable = _unpack_rows(payload.get("walkable") or [], record.width)
        # A store written before visit counts existed carries only the walked
        # set. Migrate it — every known tile counts as one arrival — rather than
        # dropping a run's worth of accumulated map memory on the floor.
        record.visits = {
            coord: 1 for coord in _unpack_rows(payload.get("walked") or [], record.width)
        }
        for entry in payload.get("visit_counts") or []:
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                continue
            record.visits[(int(entry[0]), int(entry[1]))] = max(1, int(entry[2]))
        record.warps = {
            (int(pair[0]), int(pair[1]))
            for pair in payload.get("warps") or []
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        }
        record.player = _as_coord(payload.get("player"))
        return record


class ExploredMaps:
    """Accumulates navigation snapshots into whole-map records, persisted as JSON."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._maps: Dict[int, _MapRecord] = {}
        self.current_map_id: Optional[int] = None
        self.dirty = False
        # Bumped only when a snapshot actually taught us something, so callers
        # can skip re-rendering the map image sixty times a second.
        self.revision = 0
        #: Snapshots dropped as incoherent, and what the last one contradicted.
        self.rejected = 0
        self._last_rejection: Optional[Tuple[int, str]] = None
        #: What loading the store had to repair, by map id. Empty means clean.
        self.repairs: Dict[int, List[str]] = {}
        self._fingerprint: Optional[tuple] = None
        self._load()

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def record(self, snapshot: dict) -> bool:
        """Fold one `LiveNavigationSnapshot.to_dict()` into the stored map.

        Returns False for a snapshot that was ignored, either because it names
        no map or because it contradicts the geometry of the map it names.
        """
        if not isinstance(snapshot, dict):
            return False
        raw_map_id = snapshot.get("map_id")
        if raw_map_id is None:
            return False
        try:
            map_id = int(raw_map_id)
        except (TypeError, ValueError):
            return False

        known = self._maps.get(map_id)
        known_size = (known.width, known.height) if known and known.width else None
        reason = incoherence(snapshot, map_id, known_size)
        if reason is not None:
            self.rejected += 1
            # A transition can produce the same bad frame many times a second;
            # say it once per run of identical rejections.
            if self._last_rejection != (map_id, reason):
                self._last_rejection = (map_id, reason)
                _log(f"ignoring incoherent snapshot for map {map_id}: {reason}")
            return False
        self._last_rejection = None

        record = known
        if record is None:
            record = _MapRecord(map_id)
            self._maps[map_id] = record
        previous_map_id = self.current_map_id
        self.current_map_id = map_id

        name = snapshot.get("map_name")
        if name:
            record.map_name = str(name)

        size = declared_size(snapshot)
        if size is not None and (record.width, record.height) != size:
            # The record covers the WHOLE map, allocated from its real size — it
            # is not a cache of the windows that happened to be on screen, and
            # not the largest size ever misread. The latest coherent reading of
            # the game's own map table wins, phantoms and all.
            dropped = record.resize(*size)
            if dropped:
                _log(f"map {map_id} resized to {size[0]}x{size[1]}, dropping {dropped} tiles")

        # A sprite standing on a tile makes that tile read as blocked, so skip
        # those tiles rather than learning a wandering NPC as a wall.
        sprites = {
            coord
            for coord in (_as_coord(item) for item in snapshot.get("sprites") or [])
            if coord is not None
        }

        top_left = _as_coord(snapshot.get("window_top_left")) or (0, 0)
        for local_y, row in enumerate(snapshot.get("terrain") or []):
            if not isinstance(row, (list, tuple)):
                continue
            for local_x, tile in enumerate(row):
                coord = (top_left[0] + local_x, top_left[1] + local_y)
                if coord[0] < 0 or coord[1] < 0 or coord in sprites:
                    continue
                if record.width and coord[0] >= record.width:
                    continue
                if record.height and coord[1] >= record.height:
                    continue
                record.note(coord, passable=bool(tile))

        player = _as_coord(snapshot.get("player_position"))
        if player is not None:
            # Standing on a tile is the strongest available proof it is passable.
            record.note(player, passable=True)
            # Only an *arrival* counts. The live loop records the same standing
            # position many times a second; that is not 60 visits a second.
            if previous_map_id != map_id or record.player != player:
                record.visits[player] = record.visits.get(player, 0) + 1
            record.player = player

        if "warps" in snapshot:
            # The game reports a map's whole warp table every frame, so this is
            # a replacement, not a union: warps unioned in from another map's
            # table would otherwise stay on the record for good.
            warps: Set[Coord] = set()
            for warp in snapshot.get("warps") or []:
                coord = _as_coord(warp)
                if coord is None or coord[0] < 0 or coord[1] < 0:
                    continue
                if record.width and not _inside(coord, record.width, record.height):
                    continue
                warps.add(coord)
                if not record.width:
                    record.grow_to(*coord)
            record.warps = warps

        self.dirty = True
        fingerprint = (
            map_id,
            record.width,
            record.height,
            len(record.seen),
            len(record.walkable),
            len(record.visits),
            len(record.warps),
            record.player,
        )
        if fingerprint != self._fingerprint:
            self._fingerprint = fingerprint
            self.revision += 1
        return True

    def repair(self) -> Dict[int, List[str]]:
        """Drop geometry the game's own map table says is impossible.

        A store written before dimensions were replaceable holds maps that
        inherited another map's size and warps from a single frame read mid
        transition. Every map the decompilation knows is measured against it;
        for the rest only self-consistency is checkable.
        """
        report: Dict[int, List[str]] = {}
        for map_id, record in self._maps.items():
            fixes: List[str] = []
            canonical = canonical_maps().get(map_id)
            if canonical is not None:
                size = (canonical.width, canonical.height)
                if (record.width, record.height) != size:
                    was = f"{record.width}x{record.height}"
                    dropped = record.resize(*size)
                    fixes.append(f"{was} -> {size[0]}x{size[1]}, dropping {dropped} phantom tiles")
                phantom = record.warps - canonical.warps
                if phantom:
                    record.warps -= phantom
                    fixes.append(f"dropped {len(phantom)} warps the map header does not have")
            elif record.width and record.height:
                outside = {
                    coord
                    for coord in record.warps
                    if not _inside(coord, record.width, record.height)
                }
                if outside:
                    record.warps -= outside
                    fixes.append(f"dropped {len(outside)} warps outside the map")
            if fixes:
                report[map_id] = fixes
        if report:
            self.dirty = True
            self.revision += 1
        return report

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def map_ids(self) -> List[int]:
        return sorted(self._maps)

    def knows(self, map_id: int) -> bool:
        return map_id in self._maps

    def visited(self, map_id: int) -> Set[Coord]:
        """Tiles the player has actually stood on."""
        record = self._maps.get(map_id)
        return set(record.visits) if record else set()

    def visit_count(self, map_id: int, x: int, y: int) -> int:
        """How many separate times the player has arrived on this tile."""
        record = self._maps.get(map_id)
        return record.visits.get((int(x), int(y)), 0) if record else 0

    def player_position(self, map_id: int) -> Optional[Coord]:
        record = self._maps.get(map_id)
        return record.player if record else None

    def grid(self, map_id: int) -> Optional[dict]:
        """Raw tile sets for one map, for callers that draw their own picture.

        The sets are copies: a caller folding them into an overlay cannot
        corrupt the store. `None` means the map has never been recorded.
        """
        record = self._maps.get(map_id)
        if record is None:
            return None
        return {
            "width": record.width,
            "height": record.height,
            "seen": set(record.seen),
            "walkable": set(record.walkable),
            "walked": set(record.visits),
            "warps": set(record.warps),
        }

    def coverage(self, map_id: int) -> dict:
        record = self._maps.get(map_id)
        if record is None:
            return {"seen": 0, "walkable_seen": 0, "walked": 0, "total": 0, "percent": 0.0}
        total = record.width * record.height
        seen = len(record.seen)
        return {
            "seen": seen,
            "walkable_seen": len(record.walkable),
            "walked": len(record.walked),
            "total": total,
            "percent": round(100.0 * seen / total, 1) if total else 0.0,
        }

    def unexplored_nearest(self, map_id: int, player: Optional[Coord] = None) -> Optional[dict]:
        """Closest passable-but-never-walked tile, by straight-line distance."""
        record = self._maps.get(map_id)
        if record is None:
            return None
        origin = player if player is not None else record.player
        if origin is None:
            return None
        candidates = record.walkable - record.walked
        if not candidates:
            return None
        best = min(
            candidates,
            key=lambda c: ((c[0] - origin[0]) ** 2 + (c[1] - origin[1]) ** 2, c[1], c[0]),
        )
        distance = math.hypot(best[0] - origin[0], best[1] - origin[1])
        return {"x": best[0], "y": best[1], "distance": round(distance, 2)}

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _tile_color(
        record: _MapRecord, coord: Coord, player: Optional[Coord]
    ) -> Tuple[int, int, int]:
        if player is not None and coord == player:
            return COLOR_PLAYER
        if coord in record.warps:
            return COLOR_WARP
        if coord in record.visits:
            return COLOR_WALKED
        if coord in record.walkable:
            return COLOR_SEEN
        if coord in record.seen:
            return COLOR_WALL
        return COLOR_UNKNOWN

    def summary(self, map_id: int, *, player: Optional[Coord] = None) -> dict:
        """The cheap half of `GET /map`: shape, counts and warps — never a grid.

        Raises `KeyError` for a map id that has never been visited.
        """
        record = self._maps.get(map_id)
        if record is None:
            raise KeyError(map_id)
        origin = player if player is not None else record.player
        payload: Dict[str, object] = {
            "map_id": map_id,
            "map_name": record.map_name,
            "width": record.width,
            "height": record.height,
            "coverage": self.coverage(map_id),
            "warps": _label_warps(record.map_name, sorted(record.warps)),
            "unexplored_nearest": self.unexplored_nearest(map_id, origin),
            "legend": dict(LEGEND),
        }
        if origin is not None:
            payload["player"] = {"x": origin[0], "y": origin[1]}
        return payload

    def render_image(
        self,
        map_id: int,
        *,
        player: Optional[Coord] = None,
        tile_px: Optional[int] = None,
    ) -> Image.Image:
        """Draw the whole map as one flat colour block per tile.

        The palette is the annotated frame's: green is ground you can walk, the
        dimmed green is ground you already walked, red is blocked, purple is a
        warp and cyan is you. Raises `KeyError` for an unvisited map id.
        """
        record = self._maps.get(map_id)
        if record is None:
            raise KeyError(map_id)

        width = max(1, record.width)
        height = max(1, record.height)
        tile = _auto_tile_px(width, height) if tile_px is None else int(tile_px)
        tile = max(MIN_TILE_PX, min(MAX_TILE_PX, tile))
        origin = player if player is not None else record.player

        # One pixel per tile, then a nearest-neighbour blow-up: far cheaper than
        # thousands of rectangles, and every block stays hard-edged.
        tiles = Image.new("RGB", (width, height))
        tiles.putdata(
            [self._tile_color(record, (x, y), origin) for y in range(height) for x in range(width)]
        )
        tiles = tiles.resize((width * tile, height * tile), resample=Image.NEAREST)

        left = _LEFT_MARGIN_PX
        top = _TITLE_BAND_PX + _TICK_BAND_PX
        canvas = Image.new(
            "RGB",
            (left + tiles.width + _EDGE_PAD_PX, top + tiles.height + _EDGE_PAD_PX),
            COLOR_UNKNOWN,
        )
        canvas.paste(tiles, (left, top))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        title = f"{record.map_name or f'map {map_id}'}  {width}x{height}"
        if origin is not None:
            title += f"  you ({origin[0]}, {origin[1]})"
        draw.text(
            (2, 2),
            _fit(draw, title, font, canvas.width - 4),
            fill=COLOR_TITLE,
            font=font,
        )
        draw.rectangle(
            (left - 1, top - 1, left + tiles.width, top + tiles.height),
            outline=COLOR_BORDER,
        )
        self._draw_axes(draw, font, width=width, height=height, tile=tile, left=left, top=top)
        if origin is not None:
            self._draw_player(draw, origin, tile=tile, left=left, top=top)
        return canvas

    @staticmethod
    def _draw_axes(
        draw: "ImageDraw.ImageDraw",
        font,
        *,
        width: int,
        height: int,
        tile: int,
        left: int,
        top: int,
    ) -> None:
        for x in range(0, width, _TICK_EVERY):
            px = left + int((x + 0.5) * tile)
            draw.line((px, top - 3, px, top - 1), fill=COLOR_AXIS)
            if x % _LABEL_EVERY:
                continue
            label = str(x)
            draw.text(
                (px - int(draw.textlength(label, font=font) / 2), _TITLE_BAND_PX),
                label,
                fill=COLOR_LABEL,
                font=font,
            )
        for y in range(0, height, _TICK_EVERY):
            py = top + int((y + 0.5) * tile)
            draw.line((left - 3, py, left - 1, py), fill=COLOR_AXIS)
            if y % _LABEL_EVERY:
                continue
            label = str(y)
            draw.text(
                (left - 5 - int(draw.textlength(label, font=font)), py - 5),
                label,
                fill=COLOR_LABEL,
                font=font,
            )

    @staticmethod
    def _draw_player(
        draw: "ImageDraw.ImageDraw",
        origin: Coord,
        *,
        tile: int,
        left: int,
        top: int,
    ) -> None:
        px = left + (origin[0] * tile)
        py = top + (origin[1] * tile)
        # The tile is already cyan, but one 7-pixel block in a 300-pixel map is
        # easy to miss — so ring it at two tiles' reach.
        reach = tile * 2
        draw.rectangle(
            (px - reach, py - reach, px + tile - 1 + reach, py + tile - 1 + reach),
            outline=COLOR_PLAYER,
            width=2,
        )

    def write_image(self, map_id: int, path: Path, **kwargs) -> Optional[Path]:
        """Render `map_id` to `path` atomically, so a reader never sees half a PNG.

        Returns `None` — not an error — for a map that has never been visited.
        """
        try:
            image = self.render_image(map_id, **kwargs)
        except KeyError:
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        )
        temp_path = Path(handle.name)
        try:
            with handle:
                image.save(handle, format="PNG")
            os.replace(temp_path, path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        return path

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            maps = payload["maps"]
            if not isinstance(maps, dict):
                raise TypeError("maps is not an object")
            loaded = {
                int(key): _MapRecord.from_json(int(key), value) for key, value in maps.items()
            }
            current = payload.get("current_map_id")
            current_id = int(current) if current is not None else None
        except Exception as exc:  # noqa: BLE001 — a bad file must never break a request
            _log(f"ignoring unreadable store {self.path}: {exc}")
            self._maps = {}
            self.current_map_id = None
            return
        self._maps = loaded
        self.current_map_id = current_id
        self.repairs = self.repair()
        for map_id, fixes in sorted(self.repairs.items()):
            name = self._maps[map_id].map_name or f"map {map_id}"
            _log(f"repaired {name} ({map_id}): {'; '.join(fixes)}")

    def save(self) -> None:
        """Write the store atomically. Never raises into the request path."""
        payload = {
            "version": STORE_VERSION,
            "current_map_id": self.current_map_id,
            "maps": {str(map_id): record.to_json() for map_id, record in self._maps.items()},
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=str(self.path.parent),
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            )
            temp_path = Path(handle.name)
            try:
                with handle:
                    json.dump(payload, handle, separators=(",", ":"))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, self.path)
            except Exception:
                temp_path.unlink(missing_ok=True)
                raise
        except Exception as exc:  # noqa: BLE001 — persistence is best-effort
            _log(f"could not save {self.path}: {exc}")
            return
        self.dirty = False
