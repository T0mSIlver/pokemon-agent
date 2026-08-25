"""Game-state orchestrator.

:func:`build_game_state` calls every reader method and assembles the
results into a single JSON-serialisable dictionary.

:func:`pokemon_agent.agent_cli.state_lines` is the summary the agent actually
reads; this module only assembles the dict.
"""

from __future__ import annotations

import datetime
import traceback
from typing import Any, Dict, Optional

from pokemon_agent.memory.reader import GameMemoryReader


def build_game_state(
    reader: GameMemoryReader,
    frame_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Read all game data and assemble a complete state snapshot.

    Parameters
    ----------
    reader : GameMemoryReader
        An initialised memory reader bound to a running emulator.
    frame_count : int, optional
        Current emulator frame count (injected into metadata).

    Returns
    -------
    dict
        A JSON-serialisable game-state dictionary.  Sections that fail
        to read are ``None`` with an ``"_error"`` key.
    """
    state: Dict[str, Any] = {
        "metadata": {
            "game": reader.game_name,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "frame_count": frame_count,
        },
    }

    sections = {
        "player": reader.read_player,
        "party": reader.read_party,
        "bag": reader.read_bag,
        "battle": reader.read_battle,
        "dialog": reader.read_dialog,
        "map": reader.read_map_info,
        "flags": reader.read_flags,
    }

    for key, fn in sections.items():
        try:
            state[key] = fn()
        except NotImplementedError as exc:
            state[key] = None
            state[f"{key}_error"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            state[key] = None
            state[f"{key}_error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    return state
