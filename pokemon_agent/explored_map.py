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
from typing import Dict, Iterable, List, Optional, Set, Tuple

from PIL import Image, ImageDraw, ImageFont

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


def _log(message: str) -> None:
    print(f"[explored-map] {message}")


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
        self._fingerprint: Optional[tuple] = None
        self._load()

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def record(self, snapshot: dict) -> None:
        """Fold one `LiveNavigationSnapshot.to_dict()` into the stored map."""
        if not isinstance(snapshot, dict):
            return
        raw_map_id = snapshot.get("map_id")
        if raw_map_id is None:
            return
        try:
            map_id = int(raw_map_id)
        except (TypeError, ValueError):
            return

        record = self._maps.get(map_id)
        if record is None:
            record = _MapRecord(map_id)
            self._maps[map_id] = record
        previous_map_id = self.current_map_id
        self.current_map_id = map_id

        name = snapshot.get("map_name")
        if name:
            record.map_name = str(name)

        dimensions = snapshot.get("map_dimensions")
        if isinstance(dimensions, dict):
            width = dimensions.get("width")
            height = dimensions.get("height")
            if width and height:
                # The record covers the WHOLE map, allocated from its real size —
                # it is not a cache of the windows that happened to be on screen.
                record.width = max(record.width, int(width))
                record.height = max(record.height, int(height))

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

        for warp in snapshot.get("warps") or []:
            coord = _as_coord(warp)
            if coord is None:
                continue
            record.warps.add(coord)
            record.grow_to(*coord)

        self.dirty = True
        fingerprint = (
            map_id,
            len(record.seen),
            len(record.walkable),
            len(record.visits),
            len(record.warps),
            record.player,
        )
        if fingerprint != self._fingerprint:
            self._fingerprint = fingerprint
            self.revision += 1

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
            "warps": [{"x": x, "y": y} for x, y in sorted(record.warps)],
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
