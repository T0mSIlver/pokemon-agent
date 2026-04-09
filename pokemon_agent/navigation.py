"""Navigation models and explored-map routing for Pokemon Agent."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from pokemon_agent.pathfinding import DIRECTIONS, directions_to_actions

Coord = Tuple[int, int]

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


def _coord_to_key(coord: Coord) -> str:
    return f"{coord[0]},{coord[1]}"


def _key_to_coord(value: str) -> Coord:
    x_str, y_str = value.split(",", 1)
    return int(x_str), int(y_str)


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


def _neighbors(coord: Coord) -> Iterable[Tuple[str, Coord]]:
    x, y = coord
    for direction, (dx, dy) in DIRECTIONS.items():
        yield direction, (x + dx, y + dy)


@dataclass(slots=True)
class NavigationPath:
    """A planned route expressed as directions and executable actions."""

    mode: str
    target: Coord
    reached: bool
    status: str
    directions: List[str] = field(default_factory=list)
    final_position: Optional[Coord] = None
    partial: bool = False
    visible_target: bool = False

    @property
    def actions(self) -> List[str]:
        return directions_to_actions(self.directions)

    def to_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "status": self.status,
            "reached": self.reached,
            "partial": self.partial,
            "visible_target": self.visible_target,
            "target": _coord_dict(self.target),
            "final_position": _coord_dict(self.final_position),
            "path": self.directions,
            "actions": self.actions,
            "steps": len(self.directions),
        }


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
            "player_position": _coord_dict(self.player_position),
            "facing": self.facing,
            "tileset": self.tileset,
            "window_top_left": _coord_dict(self.window_top_left),
            "window_size": {"width": self.width, "height": self.height},
            "terrain": self.terrain,
            "sprites": [_coord_dict(coord) for coord in self.sprite_positions],
            "valid_moves": self.valid_moves,
            "warps": self.warps,
            "map_dimensions": self.map_dimensions,
            "interaction": self.interaction,
            "ascii": self.render_window_ascii(goal=goal),
            "ascii_legend": {
                "P": "player",
                "G": "goal",
                "S": "visible sprite blocker",
                ".": "passable tile",
                "#": "blocked tile",
            },
        }


@dataclass(slots=True)
class LocationNavigationMap:
    """A persistent explored map for a single in-game location."""

    map_id: int
    map_name: str
    tileset: Optional[str] = None
    tiles: Dict[Coord, bool] = field(default_factory=dict)
    tile_ids: Dict[Coord, int] = field(default_factory=dict)
    updates: int = 0

    @property
    def key(self) -> str:
        return location_key(self.map_id, self.map_name)

    def update_from_snapshot(self, snapshot: LiveNavigationSnapshot) -> None:
        """Merge a live collision window into the explored map."""
        self.tileset = snapshot.tileset
        for local_y, row in enumerate(snapshot.terrain):
            for local_x, tile in enumerate(row):
                absolute = snapshot.local_to_absolute(local_x, local_y)
                self.tiles[absolute] = bool(tile)
                tile_id = snapshot.tile_ids.get(absolute)
                if tile_id is not None:
                    self.tile_ids[absolute] = tile_id
        # The player must always be traversable from the planner's perspective,
        # even if the raw collision window marks the standing tile oddly.
        self.tiles[snapshot.player_position] = True
        player_tile_id = snapshot.tile_ids.get(snapshot.player_position)
        if player_tile_id is not None:
            self.tile_ids[snapshot.player_position] = player_tile_id
        self.updates += 1

    def bounds(
        self,
        extra: Optional[Iterable[Coord]] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        coords = set(self.tiles)
        if extra is not None:
            coords.update(extra)
        if not coords:
            return None
        xs = [coord[0] for coord in coords]
        ys = [coord[1] for coord in coords]
        return min(xs), max(xs), min(ys), max(ys)

    def passable_count(self) -> int:
        return sum(1 for value in self.tiles.values() if value)

    def blocked_count(self) -> int:
        return sum(1 for value in self.tiles.values() if not value)

    def _traverse(
        self,
        start: Coord,
        extra_blockers: Optional[Iterable[Coord]] = None,
    ) -> Tuple[Dict[Coord, Tuple[Coord, str]], Dict[Coord, int]]:
        blockers = set(extra_blockers or [])
        if self.tiles.get(start) is not True or start in blockers:
            return {}, {}

        queue = deque([start])
        parents: Dict[Coord, Tuple[Coord, str]] = {}
        distances: Dict[Coord, int] = {start: 0}

        while queue:
            current = queue.popleft()
            for direction, neighbor in _neighbors(current):
                if neighbor in distances or neighbor in blockers:
                    continue
                if self.tiles.get(neighbor) is not True:
                    continue
                if not tile_pair_allows(
                    self.tileset,
                    self.tile_ids.get(current),
                    self.tile_ids.get(neighbor),
                ):
                    continue
                parents[neighbor] = (current, direction)
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

        return parents, distances

    def distance_map(
        self,
        start: Coord,
        extra_blockers: Optional[Iterable[Coord]] = None,
        max_distance: Optional[int] = None,
    ) -> Dict[Coord, int]:
        """Return BFS distance to explored passable tiles."""
        _, distances = self._traverse(start, extra_blockers=extra_blockers)
        if max_distance is None:
            return distances
        return {
            coord: distance for coord, distance in distances.items() if distance <= max_distance
        }

    def _reconstruct_path(
        self,
        parents: Dict[Coord, Tuple[Coord, str]],
        current: Coord,
    ) -> List[str]:
        directions: List[str] = []
        while current in parents:
            current, direction = parents[current]
            directions.append(direction)
        directions.reverse()
        return directions

    def plan_route(
        self,
        start: Coord,
        goal: Coord,
        extra_blockers: Optional[Iterable[Coord]] = None,
        allow_partial: bool = True,
    ) -> NavigationPath:
        """Plan a route through the explored persistent map."""
        parents, distances = self._traverse(start, extra_blockers=extra_blockers)
        if not distances:
            return NavigationPath(
                mode="persistent",
                target=goal,
                reached=False,
                status="Current position is not yet part of the explored passable map.",
                final_position=start,
            )

        blockers = set(extra_blockers or [])
        candidate_goals: List[Coord] = []
        interaction_only = False

        if self.tiles.get(goal) is True and goal not in blockers:
            candidate_goals.append(goal)
        else:
            interaction_only = self.tiles.get(goal) is False or goal in blockers
            for _, neighbor in _neighbors(goal):
                if self.tiles.get(neighbor) is True and neighbor not in blockers:
                    candidate_goals.append(neighbor)

        reachable_candidates = [coord for coord in candidate_goals if coord in distances]
        if reachable_candidates:
            final = min(reachable_candidates, key=lambda coord: distances[coord])
            directions = self._reconstruct_path(parents, final)
            reached = final == goal
            if reached:
                status = "Found a route to the explored target tile."
            elif interaction_only:
                status = (
                    "Target tile is blocked or occupied; "
                    "routed to the closest explored adjacent tile."
                )
            else:
                status = (
                    "Target tile is not yet traversable; "
                    "routed to the closest explored adjacent tile."
                )
            return NavigationPath(
                mode="persistent",
                target=goal,
                reached=reached,
                status=status,
                directions=directions,
                final_position=final,
                partial=not reached,
            )

        if allow_partial:
            best = min(
                distances,
                key=lambda coord: (
                    abs(coord[0] - goal[0]) + abs(coord[1] - goal[1]),
                    distances[coord],
                    coord[1],
                    coord[0],
                ),
            )
            if best != start:
                return NavigationPath(
                    mode="persistent",
                    target=goal,
                    reached=False,
                    status=(
                        "Target is outside the explored map; routed to the closest explored tile."
                    ),
                    directions=self._reconstruct_path(parents, best),
                    final_position=best,
                    partial=True,
                )

        return NavigationPath(
            mode="persistent",
            target=goal,
            reached=False,
            status="No explored route is available yet for the requested target.",
            final_position=start,
        )

    def render_ascii(
        self,
        player: Optional[Coord] = None,
        goal: Optional[Coord] = None,
        sprites: Optional[Iterable[Coord]] = None,
        distances: Optional[Dict[Coord, int]] = None,
    ) -> str:
        """Render the explored map as ASCII."""
        sprite_set = set(sprites or [])
        extra: set[Coord] = set(sprite_set)
        if player is not None:
            extra.add(player)
        if goal is not None:
            extra.add(goal)
        if distances:
            extra.update(distances)

        bounds = self.bounds(extra=extra)
        if bounds is None:
            return "(no explored map data)"

        min_x, max_x, min_y, max_y = bounds
        lines = [_ascii_header(min_x, max_x)]

        for y in range(min_y, max_y + 1):
            chars: List[str] = []
            for x in range(min_x, max_x + 1):
                coord = (x, y)
                if coord == player:
                    chars.append("P")
                elif coord == goal:
                    chars.append("G")
                elif coord in sprite_set:
                    chars.append("S")
                else:
                    tile = self.tiles.get(coord)
                    if tile is True:
                        chars.append(".")
                    elif tile is False:
                        chars.append("#")
                    else:
                        chars.append("?")
            lines.append(f"{y:>4} " + "".join(chars))

        return "\n".join(lines)

    def to_dict(
        self,
        player: Optional[Coord] = None,
        goal: Optional[Coord] = None,
        sprites: Optional[Iterable[Coord]] = None,
        distances: Optional[Dict[Coord, int]] = None,
    ) -> Dict[str, object]:
        bounds = self.bounds(extra=[coord for coord in [player, goal] if coord is not None])
        bounds_dict = None
        if bounds is not None:
            min_x, max_x, min_y, max_y = bounds
            bounds_dict = {
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
            }

        return {
            "location_key": self.key,
            "map_id": self.map_id,
            "map_name": self.map_name,
            "tileset": self.tileset,
            "updates": self.updates,
            "known_tiles": len(self.tiles),
            "known_tile_ids": len(self.tile_ids),
            "passable_tiles": self.passable_count(),
            "blocked_tiles": self.blocked_count(),
            "bounds": bounds_dict,
            "ascii": self.render_ascii(
                player=player,
                goal=goal,
                sprites=sprites,
                distances=distances,
            ),
            "ascii_legend": {
                "P": "player",
                "G": "goal",
                "S": "visible sprite blocker",
                ".": "explored passable tile",
                "#": "known blocked tile",
                "?": "unexplored or unknown tile",
            },
        }

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "map_id": self.map_id,
            "map_name": self.map_name,
            "tileset": self.tileset,
            "updates": self.updates,
            "tiles": {_coord_to_key(coord): value for coord, value in sorted(self.tiles.items())},
            "tile_ids": {
                _coord_to_key(coord): value for coord, value in sorted(self.tile_ids.items())
            },
        }

    @classmethod
    def from_json_dict(cls, data: Dict[str, object]) -> "LocationNavigationMap":
        tiles = {
            _key_to_coord(key): bool(value) for key, value in dict(data.get("tiles", {})).items()
        }
        return cls(
            map_id=int(data["map_id"]),
            map_name=str(data["map_name"]),
            tileset=(str(data["tileset"]) if data.get("tileset") is not None else None),
            tiles=tiles,
            tile_ids={
                _key_to_coord(key): int(value)
                for key, value in dict(data.get("tile_ids", {})).items()
            },
            updates=int(data.get("updates", 0)),
        )


class NavigationStore:
    """Persistent storage for explored per-location navigation maps."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path.expanduser().resolve() if path is not None else None
        self.location_maps: Dict[str, LocationNavigationMap] = {}
        if self.path is not None and self.path.exists():
            self.load()

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.location_maps = {
            key: LocationNavigationMap.from_json_dict(value)
            for key, value in dict(raw.get("locations", {})).items()
        }

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "locations": {
                key: location_map.to_json_dict()
                for key, location_map in sorted(self.location_maps.items())
            }
        }
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.path)

    def get(self, key: str) -> Optional[LocationNavigationMap]:
        return self.location_maps.get(key)

    def get_or_create(self, map_id: int, map_name: str) -> LocationNavigationMap:
        key = location_key(map_id, map_name)
        location_map = self.location_maps.get(key)
        if location_map is None:
            location_map = LocationNavigationMap(map_id=map_id, map_name=map_name)
            self.location_maps[key] = location_map
        return location_map

    def update(self, snapshot: LiveNavigationSnapshot) -> LocationNavigationMap:
        location_map = self.get_or_create(snapshot.map_id, snapshot.map_name)
        location_map.update_from_snapshot(snapshot)
        self.save()
        return location_map
