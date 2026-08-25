"""Direction vectors and action naming for Pokemon Agent movement."""

from __future__ import annotations

from typing import Dict, List, Tuple

# Direction vectors: name -> (dx, dy)
DIRECTIONS: Dict[str, Tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


def directions_to_actions(directions: List[str]) -> List[str]:
    """Convert direction strings to ``walk_<dir>`` action strings.

    Parameters
    ----------
    directions:
        List of direction names, e.g. ``['up', 'up', 'right']``.

    Returns
    -------
    List of action strings: ``['walk_up', 'walk_up', 'walk_right']``.
    """
    return [f"walk_{d}" for d in directions]
