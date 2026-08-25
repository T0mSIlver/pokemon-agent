"""Save-state names, and the one place that turns one into a path.

A save name arrives from the network. Appended straight to the saves directory
it is not a name at all: ``../escaped`` writes outside the directory, and an
absolute name leaves the data directory entirely. So a name is validated once,
against a conservative pattern, and every caller — startup auto-load, manual
save, manual load, listing — resolves it through :func:`resolve_save_path`,
which re-checks the *resolved* parent rather than trusting the string.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

#: Letters, digits and the three separators the harness itself writes into save
#: names (``map-transition``, ``objective_change``, ``slot.3``). No path
#: separators, no leading dot, no ``~``, no NUL.
SAVE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

#: The extension every save file carries. Part of the name check: a caller
#: cannot smuggle one in and end up with ``foo.state.state`` or ``foo.py``.
SAVE_SUFFIX = ".state"

#: Long enough for the harness's own ``auto__<stamp>__<trigger>__<map>`` names.
MAX_SAVE_NAME_LENGTH = 128


class SaveNameError(ValueError):
    """A save name that is not a plain file name inside the saves directory."""


def validate_save_name(name: object) -> str:
    """Return *name* unchanged if it is a legal save name, else raise.

    Rejects anything that is not a single path component: separators, ``.``,
    ``..``, absolute paths, drive letters, leading dots and NUL bytes.
    """
    if not isinstance(name, str):
        raise SaveNameError("save name must be a string")
    candidate = name.strip()
    if not candidate:
        raise SaveNameError("save name must not be empty")
    if candidate in (".", ".."):
        raise SaveNameError(f"{candidate!r} is not a save name")
    if len(candidate) > MAX_SAVE_NAME_LENGTH:
        raise SaveNameError(
            f"save name is {len(candidate)} characters; the limit is {MAX_SAVE_NAME_LENGTH}"
        )
    if "/" in candidate or "\\" in candidate or "\x00" in candidate:
        raise SaveNameError(
            f"{name!r} contains a path separator. A save name is a plain file name, "
            "not a path: it always lands directly in the saves directory."
        )
    if Path(candidate).name != candidate or Path(candidate).is_absolute():
        raise SaveNameError(f"{name!r} is not a plain file name")
    if candidate.lower().endswith(SAVE_SUFFIX):
        raise SaveNameError(
            f"{name!r} already ends in {SAVE_SUFFIX}; name the save without its extension"
        )
    if not SAVE_NAME_RE.match(candidate):
        raise SaveNameError(
            f"{name!r} is not a valid save name. Use letters, digits, '.', '-' and '_', "
            "starting with a letter or digit."
        )
    return candidate


def resolve_save_path(saves_dir: Path, name: object) -> Path:
    """The file ``name`` refers to inside *saves_dir*, or raise.

    Validation of the string is not enough on its own — a symlinked saves
    directory or a resolved parent that moves would still escape — so the
    candidate is resolved and its parent compared against the resolved saves
    directory before it is handed back.
    """
    checked = validate_save_name(name)
    base = Path(saves_dir).expanduser().resolve()
    candidate = (base / f"{checked}{SAVE_SUFFIX}").resolve()
    if candidate.parent != base:
        raise SaveNameError(
            f"{name!r} does not resolve inside the saves directory. Refusing to touch {candidate}."
        )
    return candidate


def list_save_files(saves_dir: Path) -> Iterable[Path]:
    """Every readable save file in *saves_dir* whose name is a legal save name.

    A file whose name would be rejected on the way in is not offered on the way
    out either: nothing should be listed that ``POST /load`` would refuse.
    """
    base = Path(saves_dir).expanduser().resolve()
    if not base.is_dir():
        return []
    found = []
    for path in sorted(base.glob(f"*{SAVE_SUFFIX}")):
        if not path.is_file():
            continue
        try:
            validate_save_name(path.stem)
        except SaveNameError:
            continue
        found.append(path)
    return found


__all__ = [
    "MAX_SAVE_NAME_LENGTH",
    "SAVE_NAME_RE",
    "SAVE_SUFFIX",
    "SaveNameError",
    "list_save_files",
    "resolve_save_path",
    "validate_save_name",
]
