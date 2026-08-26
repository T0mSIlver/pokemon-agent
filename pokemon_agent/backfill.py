"""Decode every map the save library has been on, so routing works from the first press.

A route crosses maps the player is not standing on, and a map's terrain is only
readable while it is loaded. So the router learns a floor when the agent walks
onto it, and until then answers about that floor from a static table that has
never seen a tile.

That cold start is expensive at exactly the wrong moment. Measured: a scripted
player standing on Route 4 asked for a route to Cerulean, and got the one-hop
"walk off the east edge" answer because the only floor it had decoded was the
one under its feet. The real route runs back through three Mt Moon floors it had
never been on with the decoder running, and it could not know that.

Every save state is a frame where some map was loaded. Replaying them is a cheap
way to learn the whole explored world at once: one PyBoy, one load per save,
skip the ones whose map is already known.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from pokemon_agent.explored_map import ExploredMaps


def backfill(
    rom: Path,
    saves_dir: Path,
    store: ExploredMaps,
    *,
    limit: Optional[int] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Decode one save per distinct map and fold the result into `store`.

    Newest saves first, because a newer one is likelier to be a map the run has
    actually reached. Returns what it learned, for a caller that wants to say so.
    """
    from pokemon_agent.emulator import PyBoyEmulator
    from pokemon_agent.memory.red import RedBlueMemoryReader

    say = on_progress or (lambda _message: None)
    states = sorted(saves_dir.glob("*.state"), key=lambda p: p.stat().st_mtime, reverse=True)
    if limit is not None:
        states = states[:limit]

    emulator = PyBoyEmulator()
    emulator.load(str(rom))
    reader = RedBlueMemoryReader(emulator)

    began = time.time()
    learned: dict[int, str] = {}
    skipped = failed = 0
    try:
        for state in states:
            try:
                emulator.load_state(str(state))
                emulator.tick(1)
                snapshot = emulator.get_navigation_snapshot(reader)
            except Exception:
                failed += 1
                continue
            terrain = snapshot.map_terrain
            if not terrain or not terrain.get("walkable"):
                skipped += 1
                continue
            map_id = int(snapshot.map_id)
            if store.terrain(map_id) is not None:
                skipped += 1
                continue
            store.record({**snapshot.to_dict(), "map_terrain": terrain})
            if store.terrain(map_id) is not None:
                learned[map_id] = snapshot.map_name
                say(f"  {snapshot.map_name}: {len(terrain['walkable'])} walkable")
    finally:
        emulator.close()

    if learned:
        store.save()
    return {
        "learned": learned,
        "states_read": len(states),
        "skipped": skipped,
        "failed": failed,
        "seconds": round(time.time() - began, 1),
    }
