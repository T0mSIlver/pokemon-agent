"""Live navigation snapshot models for Pokemon Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Coord = Tuple[int, int]
MAP_COORDINATE_SYSTEM = "map_tile_absolute"
MAP_COORDINATE_NOTE = (
    "All x/y values are absolute map tile coordinates. "
    "In the annotated frame, columns are x and rows are y."
)

TILE_PAIR_BLOCKERS: dict[str, set[frozenset[int]]] = {
    "CAVERN": {
        frozenset((288, 261)),
        frozenset((321, 261)),
        frozenset((298, 261)),
        frozenset((261, 289)),
        frozenset((276, 261)),
    },
    "FOREST": {
        frozenset((304, 302)),
        frozenset((338, 302)),
        frozenset((341, 302)),
        frozenset((342, 302)),
        frozenset((288, 302)),
        frozenset((350, 302)),
        frozenset((351, 302)),
        frozenset((276, 302)),
        frozenset((328, 302)),
    },
}


def location_key(map_id: int, map_name: str) -> str:
    """Build a stable key for per-location navigation data."""
    return f"{map_id}:{map_name}"


def _coord_dict(coord: Optional[Coord]) -> Optional[Dict[str, int]]:
    if coord is None:
        return None
    return {"x": coord[0], "y": coord[1]}


def _ascii_header(min_x: int, max_x: int) -> str:
    return "     " + "".join(str(x % 10) for x in range(min_x, max_x + 1))


def tile_pair_allows(
    tileset: Optional[str],
    tile_a: Optional[int],
    tile_b: Optional[int],
) -> bool:
    """Return whether movement between two adjacent tiles is allowed."""
    if tileset is None or tile_a is None or tile_b is None:
        return True
    blocked_pairs = TILE_PAIR_BLOCKERS.get(tileset)
    if not blocked_pairs:
        return True
    return frozenset((tile_a, tile_b)) not in blocked_pairs


@dataclass(slots=True)
class LiveNavigationSnapshot:
    """Live navigation state derived from the current emulator frame."""

    map_id: int
    map_name: str
    player_position: Coord
    facing: str
    tileset: str
    window_top_left: Coord
    terrain: List[List[int]]
    sprite_positions: List[Coord] = field(default_factory=list)
    valid_moves: List[str] = field(default_factory=list)
    warps: List[Dict[str, int]] = field(default_factory=list)
    signs: List[Dict[str, int]] = field(default_factory=list)
    map_dimensions: Optional[Dict[str, int]] = None
    tile_ids: Dict[Coord, int] = field(default_factory=dict)
    interaction: Optional[Dict[str, object]] = None

    @property
    def key(self) -> str:
        return location_key(self.map_id, self.map_name)

    @property
    def width(self) -> int:
        return len(self.terrain[0]) if self.terrain else 0

    @property
    def height(self) -> int:
        return len(self.terrain)

    @property
    def sprite_set(self) -> set[Coord]:
        return set(self.sprite_positions)

    def absolute_to_local(self, x: int, y: int) -> Optional[Coord]:
        local_x = x - self.window_top_left[0]
        local_y = y - self.window_top_left[1]
        if 0 <= local_x < self.width and 0 <= local_y < self.height:
            return local_x, local_y
        return None

    def local_to_absolute(self, local_x: int, local_y: int) -> Coord:
        return (
            self.window_top_left[0] + local_x,
            self.window_top_left[1] + local_y,
        )

    def render_window_ascii(self, goal: Optional[Coord] = None) -> str:
        """Render the current 9x10 live collision window as ASCII."""
        if not self.terrain:
            return "(no live collision data)"

        goal_local = None
        if goal is not None:
            goal_local = self.absolute_to_local(goal[0], goal[1])

        warp_set: set[Coord] = set()
        for warp in self.warps:
            wx = warp.get("x") if isinstance(warp, dict) else None
            wy = warp.get("y") if isinstance(warp, dict) else None
            if wx is None or wy is None:
                continue
            warp_set.add((int(wx), int(wy)))

        min_x = self.window_top_left[0]
        max_x = self.window_top_left[0] + self.width - 1
        lines = [_ascii_header(min_x, max_x)]
        for local_y, row in enumerate(self.terrain):
            chars: List[str] = []
            for local_x, tile in enumerate(row):
                absolute = self.local_to_absolute(local_x, local_y)
                if (local_x, local_y) == (4, 4):
                    chars.append("P")
                elif goal_local == (local_x, local_y):
                    chars.append("G")
                elif absolute in warp_set:
                    chars.append("W")
                elif absolute in self.sprite_set:
                    chars.append("S")
                elif tile:
                    chars.append(".")
                else:
                    chars.append("#")
            absolute_y = self.window_top_left[1] + local_y
            lines.append(f"{absolute_y:>4} " + "".join(chars))
        return "\n".join(lines)

    def to_dict(self, goal: Optional[Coord] = None) -> Dict[str, object]:
        return {
            "location_key": self.key,
            "map_id": self.map_id,
            "map_name": self.map_name,
            "coordinate_system": MAP_COORDINATE_SYSTEM,
            "coordinate_note": MAP_COORDINATE_NOTE,
            "player_position": _coord_dict(self.player_position),
            "facing": self.facing,
            "tileset": self.tileset,
            "window_top_left": _coord_dict(self.window_top_left),
            "window_size": {"width": self.width, "height": self.height},
            "terrain": self.terrain,
            "sprites": [_coord_dict(coord) for coord in self.sprite_positions],
            "valid_moves": self.valid_moves,
            "warps": self.warps,
            "signs": self.signs,
            "map_dimensions": self.map_dimensions,
            "interaction": self.interaction,
            "ascii": self.render_window_ascii(goal=goal),
            "ascii_legend": {
                "P": "player",
                "G": "goal",
                "W": "warp tile (step ONTO it, then walk in the exit direction)",
                "S": "visible sprite blocker",
                ".": "passable tile",
                "#": "blocked tile",
            },
        }
