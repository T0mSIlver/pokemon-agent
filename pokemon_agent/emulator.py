"""Unified emulator wrapper supporting PyBoy (GB/GBC) and PyGBA (GBA).

Provides a common interface for ROM loading, button input, frame advance,
screen capture, memory access, and save states across emulator backends.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from pokemon_agent.navigation import (
    MAP_COORDINATE_NOTE,
    MAP_COORDINATE_SYSTEM,
    MOVE_DIRECTIONS,
    WARP_CARPET_TILES,
    LiveNavigationSnapshot,
    facing_edge_of_map,
    ledge_hop_allows,
    ledge_landing,
    tile_pair_allows,
    warp_uses_front_tile_rule,
)

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment,misc]


INTERACTION_RAY_LOCAL: dict[str, list[tuple[int, int]]] = {
    "up": [(4, 3), (4, 2)],
    "down": [(4, 5), (4, 6)],
    "left": [(3, 4), (2, 4)],
    "right": [(5, 4), (6, 4)],
}

#: The player sits at local (4, 4) of the 9x10 window; these are its neighbours.
NEIGHBOUR_LOCAL: dict[str, tuple[int, int]] = {
    "up": (4, 3),
    "down": (4, 5),
    "left": (3, 4),
    "right": (5, 4),
}

# Red/Blue WRAM. Every one of these was confirmed against the ROM by stepping a
# save state frame by frame, not taken on faith from the decomp.
ADDR_PLAYER_SPRITE_Y_STEP = 0xC103  # wSpritePlayerStateData1 y delta, per frame
ADDR_PLAYER_SPRITE_X_STEP = 0xC105  # ... and its x delta
ADDR_WALK_COUNTER = 0xCFC5  # wWalkCounter: 8 -> 0 across one tile of walking
ADDR_STATUS_FLAGS_5 = 0xD730  # wStatusFlags5
ADDR_MOVEMENT_FLAGS = 0xD736  # wMovementFlags
ADDR_CUR_MAP = 0xD35E  # wCurMap
ADDR_PLAYER_Y = 0xD361  # wYCoord
ADDR_PLAYER_X = 0xD362  # wXCoord

#: wStatusFlags5 bit 7 — the engine is feeding the player simulated joypad
#: states (a ledge hop, a forced walk, a cutscene), so input is not ours yet.
SCRIPTED_MOVEMENT_BIT = 0x80

#: wMovementFlags bit 6 (mid ledge-hop or fishing) and bit 7 (spin tiles).
MOVEMENT_BUSY_BITS = 0xC0

#: wMovementFlags bit 2 — the engine only arms a warp once the player has
#: *walked onto* its tile, so a save state dropped onto one starts disarmed and
#: the collision warp check is skipped entirely.
STANDING_ON_WARP_BIT = 0x04


def _coord_dict(coord: Optional[tuple[int, int]]) -> Optional[dict[str, int]]:
    if coord is None:
        return None
    return {"x": coord[0], "y": coord[1]}


def _read_dialog_active(reader: Any) -> bool:
    try:
        dialog = reader.read_dialog()
    except Exception:
        return False
    return bool(dialog.get("active"))


def _build_interaction_probe(
    snapshot: LiveNavigationSnapshot,
    tilemap: List[List[int]],
    signs: List[Dict[str, int]],
    talk_over_tiles: set[int],
    dialog_active: bool,
) -> Dict[str, object]:
    checks = INTERACTION_RAY_LOCAL.get(snapshot.facing)
    can_move_forward: Optional[bool]
    if snapshot.facing in INTERACTION_RAY_LOCAL:
        can_move_forward = snapshot.facing in snapshot.valid_moves
    else:
        can_move_forward = None

    def probe(local_coord: tuple[int, int], distance: int) -> Dict[str, object]:
        local_x, local_y = local_coord
        absolute = snapshot.local_to_absolute(local_x, local_y)
        sign = sign_lookup.get(absolute)
        tile_id: Optional[int] = None
        tile_y = local_y * 2 + 1
        tile_x = local_x * 2
        if 0 <= tile_y < len(tilemap) and 0 <= tile_x < len(tilemap[tile_y]):
            tile_id = int(tilemap[tile_y][tile_x])
        return {
            "coord": _coord_dict(absolute),
            "distance": distance,
            "tile_id": tile_id,
            "passable": bool(snapshot.terrain[local_y][local_x]),
            "talk_over": tile_id in talk_over_tiles if tile_id is not None else False,
            "has_sprite": absolute in sprite_set,
            "has_sign": sign is not None,
            "sign_text_id": sign.get("text_id") if sign is not None else None,
        }

    sign_lookup = {(int(sign["x"]), int(sign["y"])): sign for sign in signs}
    sprite_set = snapshot.sprite_set

    if checks is None:
        return {
            "kind": "unknown",
            "source": "unknown_facing",
            "reason": "The player's facing direction could not be resolved.",
            "coordinate_system": MAP_COORDINATE_SYSTEM,
            "coordinate_note": MAP_COORDINATE_NOTE,
            "dialog_active": dialog_active,
            "can_move_forward": can_move_forward,
            "target_coord": None,
            "distance": None,
            "sign_text_id": None,
            "front_tile": None,
            "second_tile": None,
        }

    front = probe(checks[0], 1)
    second = probe(checks[1], 2)

    kind = "none"
    source = "none"
    reason = "No likely sign or object was detected in front of the player."
    target_coord: Optional[tuple[int, int]] = None
    distance: Optional[int] = None
    sign_text_id: Optional[int] = None

    if front["has_sprite"]:
        kind = "object"
        source = "sprite_direct"
        reason = "A visible sprite occupies the tile directly in front of the player."
        target_coord = (
            int(front["coord"]["x"]),
            int(front["coord"]["y"]),
        )
        distance = 1
    elif front["has_sign"]:
        kind = "sign"
        source = "sign_direct"
        reason = "The tile directly in front of the player matches a map sign/background event."
        target_coord = (
            int(front["coord"]["x"]),
            int(front["coord"]["y"]),
        )
        distance = 1
        sign_text_id = int(front["sign_text_id"])
    elif front["talk_over"] and second["has_sprite"]:
        kind = "object"
        source = "sprite_over_counter"
        reason = (
            "A talk-over tile is directly in front of the player and a visible sprite "
            "is one tile behind it."
        )
        target_coord = (
            int(second["coord"]["x"]),
            int(second["coord"]["y"]),
        )
        distance = 2
    elif front["talk_over"] and second["has_sign"]:
        kind = "sign"
        source = "sign_over_counter"
        reason = (
            "A talk-over tile is directly in front of the player and the next tile "
            "matches a map sign/background event."
        )
        target_coord = (
            int(second["coord"]["x"]),
            int(second["coord"]["y"]),
        )
        distance = 2
        sign_text_id = int(second["sign_text_id"])
    elif front["talk_over"]:
        kind = "background"
        source = "counter_tile"
        reason = (
            "The tile directly in front of the player is a talk-over counter tile, "
            "but no sprite or sign was detected beyond it."
        )
        target_coord = (
            int(front["coord"]["x"]),
            int(front["coord"]["y"]),
        )
        distance = 1
    elif can_move_forward is False and not front["passable"]:
        kind = "background"
        source = "blocked_tile"
        reason = (
            "Forward movement is blocked by a non-passable background tile with no "
            "visible sprite on it."
        )
        target_coord = (
            int(front["coord"]["x"]),
            int(front["coord"]["y"]),
        )
        distance = 1
    elif dialog_active:
        kind = "unknown"
        source = "dialog_lock"
        reason = (
            "Dialog is active, but no sprite or sign was detected in the immediate interaction ray."
        )

    return {
        "kind": kind,
        "source": source,
        "reason": reason,
        "coordinate_system": MAP_COORDINATE_SYSTEM,
        "coordinate_note": MAP_COORDINATE_NOTE,
        "dialog_active": dialog_active,
        "can_move_forward": can_move_forward,
        "target_coord": _coord_dict(target_coord),
        "distance": distance,
        "sign_text_id": sign_text_id,
        "front_tile": front,
        "second_tile": second,
    }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Emulator(ABC):
    """Abstract emulator interface.

    Subclasses wrap a concrete emulator library (PyBoy, PyGBA, etc.) and
    expose a uniform API for the agent layer.
    """

    BUTTONS: List[str] = ["a", "b", "start", "select", "up", "down", "left", "right"]

    def __init__(self) -> None:
        self.frame_count: int = 0
        self.rom_path: Optional[str] = None

    # -- lifecycle ----------------------------------------------------------

    @abstractmethod
    def load(self, rom_path: str) -> None:
        """Load a ROM file and initialise the emulator."""

    @abstractmethod
    def close(self) -> None:
        """Shut down the emulator and release resources."""

    # -- input --------------------------------------------------------------

    @abstractmethod
    def press(self, button: str, frames: int = 1) -> None:
        """Press *button* and hold it for *frames* frames.

        Parameters
        ----------
        button : str
            One of ``BUTTONS``.
        frames : int
            How many frames to hold the button before releasing.
        """

    @abstractmethod
    def release_all(self) -> None:
        """Release every button."""

    # -- timing -------------------------------------------------------------

    @abstractmethod
    def tick(self, frames: int = 1) -> None:
        """Advance the emulation by *frames* frames."""

    def settle(self, *, max_frames: int = 600, quiet_frames: int = 30) -> bool:
        """Tick until the player stops moving and the game returns control.

        Returns True if it settled within max_frames, False if it gave up.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not know when the game has settled."
        )

    # -- video --------------------------------------------------------------

    @abstractmethod
    def get_screen(self) -> "Image.Image":
        """Return the current screen as a PIL Image."""

    # -- memory -------------------------------------------------------------

    @abstractmethod
    def read_u8(self, addr: int) -> int:
        """Read an unsigned 8-bit value from *addr*."""

    @abstractmethod
    def read_u16(self, addr: int) -> int:
        """Read an unsigned 16-bit little-endian value from *addr*."""

    @abstractmethod
    def read_u32(self, addr: int) -> int:
        """Read an unsigned 32-bit little-endian value from *addr*."""

    @abstractmethod
    def read_range(self, addr: int, size: int) -> bytes:
        """Read *size* bytes starting at *addr*."""

    # -- save / load --------------------------------------------------------

    @abstractmethod
    def save_state(self, path: str) -> None:
        """Persist an emulator save-state to *path*."""

    @abstractmethod
    def load_state(self, path: str) -> None:
        """Restore an emulator save-state from *path*."""

    # -- info ---------------------------------------------------------------

    def get_info(self) -> Dict:
        """Return runtime metadata about the emulator."""
        return {
            "backend": self.__class__.__name__,
            "rom_path": self.rom_path,
            "frame_count": self.frame_count,
        }

    # -- navigation ---------------------------------------------------------

    def get_valid_moves(self, reader: Any) -> List[str]:
        """Return currently valid movement directions."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide navigation primitives."
        )

    def get_navigation_snapshot(self, reader: Any) -> LiveNavigationSnapshot:
        """Return a live navigation snapshot for the current frame."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide navigation primitives."
        )


# ---------------------------------------------------------------------------
# PyBoy backend (Game Boy / Game Boy Color)
# ---------------------------------------------------------------------------


class PyBoyEmulator(Emulator):
    """Wraps the *PyBoy* library for .gb / .gbc ROMs.

    Runs headless (``window='null'``) so no display server is required.
    """

    def __init__(self) -> None:
        super().__init__()
        self._pyboy: Optional[object] = None

    # -- lifecycle ----------------------------------------------------------

    def load(self, rom_path: str) -> None:
        """Load a Game Boy ROM via PyBoy."""
        try:
            from pyboy import PyBoy  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PyBoy is required for .gb/.gbc ROMs.  Install it with:  pip install pyboy"
            ) from exc

        rom_path = str(Path(rom_path).expanduser().resolve())
        if not os.path.isfile(rom_path):
            raise FileNotFoundError(f"ROM not found: {rom_path}")

        self._pyboy = PyBoy(rom_path, window="null")
        self.rom_path = rom_path
        self.frame_count = 0

    def close(self) -> None:
        """Stop PyBoy."""
        if self._pyboy is not None:
            self._pyboy.stop(save=False)  # type: ignore[union-attr]
            self._pyboy = None

    # -- input --------------------------------------------------------------

    def press(self, button: str, frames: int = 1) -> None:
        """Press a button and hold it for *frames* frames, then release.

        Uses button_press/button_release (not button()) to ensure the
        button stays held for the full duration. PyBoy's button() auto-
        releases after ``delay`` ticks which can cause issues with Gen 1
        walk registration that needs multi-frame holds.
        """
        button = button.lower()
        if button not in self.BUTTONS:
            raise ValueError(f"Unknown button '{button}'. Valid: {self.BUTTONS}")
        pb = self._pyboy
        pb.button_press(button)  # type: ignore[union-attr]
        self.tick(frames)
        pb.button_release(button)  # type: ignore[union-attr]

    def release_all(self) -> None:
        """Release all buttons."""
        pb = self._pyboy
        for btn in self.BUTTONS:
            try:
                pb.button_release(btn)  # type: ignore[union-attr]
            except Exception:
                pass

    # -- timing -------------------------------------------------------------

    def tick(self, frames: int = 1) -> None:
        """Advance emulation by *frames* frames."""
        pb = self._pyboy
        for _ in range(frames):
            pb.tick()  # type: ignore[union-attr]
            self.frame_count += 1

    def _movement_busy(self) -> bool:
        return bool(
            self.read_u8(ADDR_WALK_COUNTER)
            or self.read_u8(ADDR_PLAYER_SPRITE_Y_STEP)
            or self.read_u8(ADDR_PLAYER_SPRITE_X_STEP)
            or self.read_u8(ADDR_MOVEMENT_FLAGS) & MOVEMENT_BUSY_BITS
            or self.read_u8(ADDR_STATUS_FLAGS_5) & SCRIPTED_MOVEMENT_BIT
        )

    def _settle_key(self) -> tuple[int, int, int]:
        return (
            self.read_u8(ADDR_CUR_MAP),
            self.read_u8(ADDR_PLAYER_X),
            self.read_u8(ADDR_PLAYER_Y),
        )

    def settle(self, *, max_frames: int = 600, quiet_frames: int = 30) -> bool:
        """Tick until the player stops moving and the game returns control.

        Returns True if it settled within max_frames, False if it gave up.
        """
        quiet = 0
        settled_key = self._settle_key()
        for _ in range(max_frames):
            self.tick(1)
            key = self._settle_key()
            if key != settled_key or self._movement_busy():
                # A ledge hop passes through an intermediate tile and a warp
                # passes through an incoherent map; restarting the window is
                # what keeps either from being observed as a resting place.
                settled_key = key
                quiet = 0
                continue
            quiet += 1
            if quiet >= quiet_frames:
                return True
        return False

    def press_and_settle(
        self,
        button: str,
        frames: int = 8,
        *,
        max_frames: int = 600,
        quiet_frames: int = 30,
    ) -> bool:
        """Hold *button* for *frames*, then wait for the result to be observable."""
        self.press(button, frames)
        return self.settle(max_frames=max_frames, quiet_frames=quiet_frames)

    # -- video --------------------------------------------------------------

    def get_screen(self) -> "Image.Image":
        """Return current screen as a PIL Image (160×144)."""
        return self._pyboy.screen.image  # type: ignore[union-attr]

    # -- memory -------------------------------------------------------------

    def read_u8(self, addr: int) -> int:
        return self._pyboy.memory[addr] & 0xFF  # type: ignore[index]

    def read_u16(self, addr: int) -> int:
        lo = self._pyboy.memory[addr] & 0xFF  # type: ignore[index]
        hi = self._pyboy.memory[addr + 1] & 0xFF  # type: ignore[index]
        return (hi << 8) | lo

    def read_u32(self, addr: int) -> int:
        b = bytes(self._pyboy.memory[addr : addr + 4])  # type: ignore[index]
        return int.from_bytes(b, "little")

    def read_range(self, addr: int, size: int) -> bytes:
        return bytes(self._pyboy.memory[addr : addr + size])  # type: ignore[index]

    # -- save / load --------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Save emulator state to a file."""
        path = str(Path(path).expanduser().resolve())
        with open(path, "wb") as f:
            self._pyboy.save_state(f)  # type: ignore[union-attr]

    def load_state(self, path: str) -> None:
        """Load emulator state from a file."""
        path = str(Path(path).expanduser().resolve())
        with open(path, "rb") as f:
            self._pyboy.load_state(f)  # type: ignore[union-attr]

    # -- info ---------------------------------------------------------------

    def get_info(self) -> Dict:
        info = super().get_info()
        info["platform"] = "GB/GBC"
        return info

    # -- navigation ---------------------------------------------------------

    def _require_pyboy(self) -> object:
        if self._pyboy is None:
            raise RuntimeError("PyBoy emulator is not loaded")
        return self._pyboy

    def _matrix_to_rows(self, matrix: Any) -> List[List[int]]:
        data = matrix.tolist() if hasattr(matrix, "tolist") else list(matrix)
        rows: List[List[int]] = []
        for row in data:
            if hasattr(row, "tolist"):
                row = row.tolist()
            rows.append([int(value) for value in row])
        return rows

    def _downsample_collision(self, matrix: Any) -> List[List[int]]:
        rows = self._matrix_to_rows(matrix)
        height = len(rows)
        width = len(rows[0]) if rows else 0

        if height == 9 and width == 10:
            return [[1 if value else 0 for value in row] for row in rows]
        if height != 18 or width != 20:
            raise ValueError(
                f"Unexpected collision map shape: {height}x{width} (expected 18x20 or 9x10)"
            )

        downsampled: List[List[int]] = []
        for y in range(0, 18, 2):
            out_row: List[int] = []
            for x in range(0, 20, 2):
                block = (
                    rows[y][x],
                    rows[y][x + 1],
                    rows[y + 1][x],
                    rows[y + 1][x + 1],
                )
                out_row.append(1 if any(block) else 0)
            downsampled.append(out_row)
        return downsampled

    def _get_visible_sprites(self) -> set[tuple[int, int]]:
        pb = self._require_pyboy()
        sprites_by_y: dict[int, list[tuple[int, int]]] = {}

        for index in range(40):
            sprite = pb.get_sprite(index)  # type: ignore[attr-defined]
            if not getattr(sprite, "on_screen", False):
                continue
            grid_x = int(sprite.x / 160 * 10)
            grid_y = int(sprite.y / 144 * 9)
            if not (0 <= grid_x < 10 and 0 <= grid_y < 9):
                continue
            sprites_by_y.setdefault(int(sprite.y), []).append((grid_x, grid_y))

        bottom_tiles: set[tuple[int, int]] = set()
        y_levels = sorted(sprites_by_y)
        for top_y, bottom_y in zip(y_levels, y_levels[1:]):
            if bottom_y - top_y != 8:
                continue
            top_xs = {x for x, _ in sprites_by_y[top_y]}
            for grid_x, grid_y in sprites_by_y[bottom_y]:
                if grid_x in top_xs:
                    bottom_tiles.add((grid_x, grid_y))
        return bottom_tiles

    def _tile_id_at_local(
        self,
        tilemap: List[List[int]],
        local_coord: tuple[int, int],
    ) -> Optional[int]:
        local_x, local_y = local_coord
        tile_y = local_y * 2 + 1
        tile_x = local_x * 2
        if 0 <= tile_y < len(tilemap) and 0 <= tile_x < len(tilemap[tile_y]):
            return int(tilemap[tile_y][tile_x])
        return None

    def _tile_ids_for_window(
        self,
        origin: tuple[int, int],
        tilemap: List[List[int]],
    ) -> Dict[tuple[int, int], int]:
        window_top_left = (origin[0] - 4, origin[1] - 4)
        tile_ids: Dict[tuple[int, int], int] = {}
        for local_y in range(9):
            for local_x in range(10):
                tile_id = self._tile_id_at_local(tilemap, (local_x, local_y))
                if tile_id is None:
                    continue
                tile_ids[
                    (
                        window_top_left[0] + local_x,
                        window_top_left[1] + local_y,
                    )
                ] = tile_id
        return tile_ids

    def _movement_components(self, reader: Any) -> Dict[str, Any]:
        pb = self._require_pyboy()
        map_info = reader.read_map_info()
        coords = reader.read_coordinates()
        sprites_local = self._get_visible_sprites()
        sprites_local.discard((4, 4))
        facing = (
            reader.read_facing()
            if hasattr(reader, "read_facing")
            else reader.read_player().get("facing", "unknown")
        )
        map_dimensions = None
        if hasattr(reader, "read_map_dimensions"):
            try:
                map_dimensions = reader.read_map_dimensions()
            except Exception:
                map_dimensions = None

        return {
            "map_info": map_info,
            "coords": coords,
            "facing": facing,
            "tileset": reader.read_tileset(),
            "warps": reader.read_warps(),
            "signs": reader.read_signs(),
            "talk_over_tiles": set(reader.read_talk_over_tiles()),
            "dialog_active": _read_dialog_active(reader),
            "map_dimensions": map_dimensions,
            "standing_on_warp": bool(self.read_u8(ADDR_MOVEMENT_FLAGS) & STANDING_ON_WARP_BIT),
            "terrain": self._downsample_collision(pb.game_wrapper.game_area_collision()),  # type: ignore[attr-defined]
            "tilemap": self._matrix_to_rows(pb.game_wrapper._get_screen_background_tilemap()),  # type: ignore[attr-defined]
            "sprites_local": sprites_local,
        }

    def _compute_ledge_hops(
        self,
        tilemap: List[List[int]],
        tileset: str,
        player_coords: tuple[int, int],
    ) -> Dict[str, tuple[int, int]]:
        """Directions that jump a ledge, mapped to where the jump lands.

        Ledges read as blocked in the collision map, so they have to be
        recovered from the tile pair before the collision result is trusted.
        """
        standing_tile = self._tile_id_at_local(tilemap, (4, 4))
        hops: Dict[str, tuple[int, int]] = {}
        for direction, local in NEIGHBOUR_LOCAL.items():
            ledge_tile = self._tile_id_at_local(tilemap, local)
            if ledge_hop_allows(tileset, direction, standing_tile, ledge_tile):
                hops[direction] = ledge_landing(player_coords, direction)
        return hops

    def _compute_valid_moves(
        self,
        terrain: List[List[int]],
        tilemap: List[List[int]],
        tileset: str,
        sprites_local: set[tuple[int, int]],
        player_coords: tuple[int, int],
        warps: List[Dict[str, int]],
        ledge_hops: Optional[Dict[str, tuple[int, int]]] = None,
    ) -> List[str]:
        moves: List[str] = []
        current_tile = self._tile_id_at_local(tilemap, (4, 4))
        if ledge_hops is None:
            ledge_hops = self._compute_ledge_hops(tilemap, tileset, player_coords)

        for direction in MOVE_DIRECTIONS:
            local_x, local_y = NEIGHBOUR_LOCAL[direction]
            if direction in ledge_hops:
                moves.append(direction)
                continue
            if not terrain[local_y][local_x]:
                continue
            if (local_x, local_y) in sprites_local:
                continue
            neighbor_tile = self._tile_id_at_local(tilemap, (local_x, local_y))
            if not tile_pair_allows(tileset, current_tile, neighbor_tile):
                continue
            moves.append(direction)

        return moves

    def _compute_warp_exits(
        self,
        terrain: List[List[int]],
        tilemap: List[List[int]],
        tileset: str,
        map_id: int,
        map_dimensions: Optional[Dict[str, int]],
        player_coords: tuple[int, int],
        warps: List[Dict[str, int]],
        armed: bool,
    ) -> tuple[List[str], bool, Optional[str]]:
        """Directions that fire the warp under the player, whether it is armed
        right now, and a note when neither can be answered.

        These are never walkable steps: pokered only reaches its warp check
        after the move has already collided, so every candidate here is a
        direction ordinary collision rejects.
        """
        warp_coords = {(int(warp["x"]), int(warp["y"])) for warp in warps}
        if player_coords not in warp_coords:
            return [], False, None

        front_tile_rule = warp_uses_front_tile_rule(tileset, map_id)
        exits: List[str] = []
        for direction in MOVE_DIRECTIONS:
            local_x, local_y = NEIGHBOUR_LOCAL[direction]
            if terrain[local_y][local_x]:
                continue
            if front_tile_rule:
                front_tile = self._tile_id_at_local(tilemap, (local_x, local_y))
                if front_tile in WARP_CARPET_TILES[direction]:
                    exits.append(direction)
                continue
            at_edge = facing_edge_of_map(player_coords, direction, map_dimensions)
            if at_edge is None:
                unknown = (
                    "Standing on a warp whose exit direction is the edge of the map, "
                    "but the map dimensions could not be read."
                )
                return [], False, unknown
            if at_edge:
                exits.append(direction)

        if not exits:
            unmatched = (
                "Standing on a warp, but no adjacent tile matches the rule that fires "
                "it. It may fire on stepping onto a neighbouring tile instead."
            )
            return [], False, unmatched
        if not armed:
            disarmed = (
                "The exit direction is known, but the warp is not armed: the engine "
                "only arms it once the player walks onto the tile. Step off and back "
                "on, then press the exit direction."
            )
            return exits, False, disarmed
        return exits, True, None

    def get_valid_moves(self, reader: Any) -> List[str]:
        components = self._movement_components(reader)
        return self._compute_valid_moves(
            terrain=components["terrain"],
            tilemap=components["tilemap"],
            tileset=components["tileset"],
            sprites_local=components["sprites_local"],
            player_coords=components["coords"],
            warps=components["warps"],
        )

    def get_navigation_snapshot(self, reader: Any) -> LiveNavigationSnapshot:
        components = self._movement_components(reader)
        map_info = components["map_info"]
        coords = components["coords"]
        ledge_hops = self._compute_ledge_hops(components["tilemap"], components["tileset"], coords)
        valid_moves = self._compute_valid_moves(
            terrain=components["terrain"],
            tilemap=components["tilemap"],
            tileset=components["tileset"],
            sprites_local=components["sprites_local"],
            player_coords=coords,
            warps=components["warps"],
            ledge_hops=ledge_hops,
        )
        warp_exits, warp_armed, warp_exit_note = self._compute_warp_exits(
            terrain=components["terrain"],
            tilemap=components["tilemap"],
            tileset=components["tileset"],
            map_id=map_info["map_id"],
            map_dimensions=components["map_dimensions"],
            player_coords=coords,
            warps=components["warps"],
            armed=components["standing_on_warp"],
        )
        snapshot = LiveNavigationSnapshot(
            map_id=map_info["map_id"],
            map_name=map_info["map_name"],
            player_position=coords,
            facing=components["facing"],
            tileset=components["tileset"],
            window_top_left=(coords[0] - 4, coords[1] - 4),
            terrain=components["terrain"],
            sprite_positions=[
                (coords[0] - 4 + local_x, coords[1] - 4 + local_y)
                for local_x, local_y in sorted(components["sprites_local"])
            ],
            valid_moves=valid_moves,
            warps=components["warps"],
            signs=components["signs"],
            map_dimensions=components["map_dimensions"],
            tile_ids=self._tile_ids_for_window(coords, components["tilemap"]),
            ledge_hops=ledge_hops,
            warp_exit_directions=warp_exits,
            warp_exit_armed=warp_armed,
            warp_exit_note=warp_exit_note,
        )
        snapshot.interaction = _build_interaction_probe(
            snapshot,
            tilemap=components["tilemap"],
            signs=components["signs"],
            talk_over_tiles=components["talk_over_tiles"],
            dialog_active=components["dialog_active"],
        )
        return snapshot


# ---------------------------------------------------------------------------
# PyGBA backend (Game Boy Advance)
# ---------------------------------------------------------------------------


class PyGBAEmulator(Emulator):
    """Wraps the *PyGBA / mgba-py* library for .gba ROMs.

    This is a Phase-2 backend.  The interface mirrors :class:`PyBoyEmulator`
    so agent code is backend-agnostic.
    """

    _BUTTON_MAP = {
        "a": "press_a",
        "b": "press_b",
        "start": "press_start",
        "select": "press_select",
        "up": "press_up",
        "down": "press_down",
        "left": "press_left",
        "right": "press_right",
    }

    def __init__(self) -> None:
        super().__init__()
        self._gba: Optional[object] = None

    # -- lifecycle ----------------------------------------------------------

    def load(self, rom_path: str) -> None:
        """Load a GBA ROM via PyGBA / mgba."""
        try:
            from pygba import PyGBA  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PyGBA (mgba-py) is required for .gba ROMs.  Install it with:  pip install pygba"
            ) from exc

        rom_path = str(Path(rom_path).expanduser().resolve())
        if not os.path.isfile(rom_path):
            raise FileNotFoundError(f"ROM not found: {rom_path}")

        self._gba = PyGBA.load(rom_path)  # type: ignore[attr-defined]
        self.rom_path = rom_path
        self.frame_count = 0

    def close(self) -> None:
        """Release PyGBA resources."""
        self._gba = None

    # -- input --------------------------------------------------------------

    def press(self, button: str, frames: int = 1) -> None:
        button = button.lower()
        method = self._BUTTON_MAP.get(button)
        if method is None:
            raise ValueError(f"Unknown button '{button}'. Valid: {self.BUTTONS}")
        getattr(self._gba, method)()  # type: ignore[union-attr]
        self.tick(frames)

    def release_all(self) -> None:
        # PyGBA buttons auto-release after wait(); no-op here.
        pass

    # -- timing -------------------------------------------------------------

    def tick(self, frames: int = 1) -> None:
        self._gba.wait(frames)  # type: ignore[union-attr]
        self.frame_count += frames

    # -- video --------------------------------------------------------------

    def get_screen(self) -> "Image.Image":
        return self._gba.screen.to_pil()  # type: ignore[union-attr]

    # -- memory -------------------------------------------------------------

    def read_u8(self, addr: int) -> int:
        return self._gba.read_u8(addr)  # type: ignore[union-attr]

    def read_u16(self, addr: int) -> int:
        return self._gba.read_u16(addr)  # type: ignore[union-attr]

    def read_u32(self, addr: int) -> int:
        return self._gba.read_u32(addr)  # type: ignore[union-attr]

    def read_range(self, addr: int, size: int) -> bytes:
        return bytes(self._gba.read_u8(addr + i) for i in range(size))  # type: ignore[union-attr]

    # -- save / load --------------------------------------------------------

    def save_state(self, path: str) -> None:
        self._gba.save_state(path)  # type: ignore[union-attr]

    def load_state(self, path: str) -> None:
        self._gba.load_state(path)  # type: ignore[union-attr]

    # -- info ---------------------------------------------------------------

    def get_info(self) -> Dict:
        info = super().get_info()
        info["platform"] = "GBA"
        return info


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_EXT_MAP = {
    ".gb": PyBoyEmulator,
    ".gbc": PyBoyEmulator,
    ".gba": PyGBAEmulator,
}


def create_emulator(rom_path: str) -> Emulator:
    """Create the appropriate emulator for *rom_path* based on file extension.

    Parameters
    ----------
    rom_path : str
        Path to a Game Boy (.gb/.gbc) or Game Boy Advance (.gba) ROM.

    Returns
    -------
    Emulator
        A loaded, ready-to-use emulator instance.

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    """
    ext = Path(rom_path).suffix.lower()
    cls = _EXT_MAP.get(ext)
    if cls is None:
        raise ValueError(f"Unsupported ROM extension '{ext}'. Supported: {', '.join(_EXT_MAP)}")
    emu = cls()
    emu.load(rom_path)
    return emu
