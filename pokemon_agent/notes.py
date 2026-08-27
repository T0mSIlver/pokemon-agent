"""The harness's half of ``NOTES.md``.

``NOTES.md`` is the only memory that survives a session, and for a long time all
of it was hand-written. That is fine for the half a model is the only possible
author of - what it tried, what failed, the geography it worked out - and it is
a liability for the half the harness can simply read. One live file said the
party knew Leer when the party knew Cut, and its own next-steps list said "get
HM01 Cut if not already have it". Every other checkable line in it was wrong
too: the map, the position, the level, the HP, the money, the milestone count.

So the file is split by verifiability. The harness owns a delimited block at the
top and rewrites it from the same state read the rest of the harness uses; the
model owns everything below, and nothing here ever touches that. Losing the
model's notes would be a worse bug than the one this fixes, so every path
through :func:`splice_state_block` either replaces exactly the harness block or
prepends a new one - none of them delete text they did not write.

The block carries no timestamp on purpose. It would be the one line that changed
on every rewrite, and "rewriting twice in a row changes nothing" is the property
that makes an accidental double refresh a no-op.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

NOTES_FILENAME = "NOTES.md"

#: HTML comments, so the delimiters are invisible in any rendered view of the
#: file and cannot be mistaken for prose the model is meant to answer.
BLOCK_BEGIN = "<!-- harness-state:begin -->"
BLOCK_END = "<!-- harness-state:end -->"

BLOCK_TITLE = "## Current state - written by the harness, read from the game"

#: Said inside the block because that is the only place a model editing the
#: block is guaranteed to be looking.
BLOCK_PREAMBLE = (
    "This block is not yours. It is overwritten from RAM at the start and end of "
    "every session, so an edit here is discarded. Do not copy these facts further "
    "down the file either: a stale copy of them is the exact bug this block "
    "exists to end. Below the closing delimiter is yours - what you tried, what "
    "failed and why, geography you worked out, plans, open questions."
)

#: What a field says when the read that fills it did not happen. Every
#: harness-owned field gets a line even when it is unknown: a missing line reads
#: as an invitation to write one.
UNREAD = "unread"

#: Seeded once, into a notes file that does not exist or has nothing in it. It
#: belongs to the model from the moment it is written; refreshes never touch it.
SEED_BODY = "## Your notes\n"


def _position(player: Mapping[str, Any]) -> str:
    position = player.get("position")
    if not isinstance(position, Mapping):
        return UNREAD
    x, y = position.get("x"), position.get("y")
    if x is None or y is None:
        return UNREAD
    return f"({x},{y})"


def _move_pp(move: Mapping[str, Any]) -> str:
    """``Ember 25/25``, falling back to the bare count when the table is silent.

    The denominator is the greater of the move table's PP and what the mon is
    carrying, because PP Ups raise the real ceiling above the table and a
    denominator below the numerator would read as corruption. It therefore
    under-reports the ceiling on a PP-Upped move and never over-reports it,
    which is the same direction ``capabilities._max_pp`` errs in.
    """
    name = str(move.get("name") or "?")
    pp = move.get("pp")
    if pp is None:
        return name
    try:
        from pokemon_agent import gamedata

        record = gamedata.move(name) or {}
        full = int(record.get("pp") or 0)
    except Exception:  # noqa: BLE001 - a missing table costs a denominator, not a line
        full = 0
    if full <= 0:
        return f"{name} {pp}"
    return f"{name} {pp}/{max(full, int(pp))}"


def _party_lines(party: Sequence[Any]) -> list[str]:
    if not party:
        return ["- Party: empty"]
    rows = ["- Party:"]
    for mon in party:
        if not isinstance(mon, Mapping):
            continue
        types = "/".join(str(one) for one in mon.get("types") or ()) or "?"
        status = str(mon.get("status") or "OK")
        tail = "" if status.upper() in ("OK", "NONE", "") else f" {status}"
        moves = [_move_pp(one) for one in mon.get("moves") or () if isinstance(one, Mapping)]
        head = (
            f"  - {mon.get('species') or '?'} L{mon.get('level') or '?'} "
            f"{mon.get('hp')}/{mon.get('max_hp')} HP {types}{tail}"
        )
        rows.append(head + (f" - {', '.join(moves)}" if moves else " - no moves read"))
    return rows


def _milestone_line(progress: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(progress, Mapping):
        return f"- Milestones: {UNREAD}"
    count = progress.get("count")
    total = progress.get("total")
    if total in (None, 0):
        total = _milestone_total()
    furthest = str(progress.get("furthest") or "").strip()
    if count is None:
        return f"- Milestones: {UNREAD}"
    scored = f"{count}/{total}" if total else str(count)
    return f"- Milestones: {scored}" + (f", furthest: {furthest}" if furthest else "")


def _milestone_total() -> Optional[int]:
    try:
        from pokemon_agent.milestones import MILESTONES

        return len(MILESTONES)
    except Exception:  # noqa: BLE001 - a denominator, not a blocker
        return None


def state_block_lines(
    state: Optional[Mapping[str, Any]] = None,
    progress: Optional[Mapping[str, Any]] = None,
) -> list[str]:
    """The harness-owned facts, one line per field, every field always present.

    *state* is the ``/state`` dict :func:`pokemon_agent.state.builder.build_game_state`
    assembles; *progress* is ``{"count", "total", "furthest"}``, where ``furthest``
    is the ladder *label*. Nothing here reads memory: both arguments come from
    readers that already exist.
    """
    state = state if isinstance(state, Mapping) else {}
    player = state.get("player") if isinstance(state.get("player"), Mapping) else {}
    map_info = state.get("map") if isinstance(state.get("map"), Mapping) else {}

    map_name = str(map_info.get("map_name") or UNREAD)
    facing = str(player.get("facing") or UNREAD)
    where = f"- Where: {map_name} {_position(player)}"
    battle = state.get("battle") if isinstance(state.get("battle"), Mapping) else {}
    # Same reason ``agent_cli.state_lines`` refuses it: an encounter interrupts
    # the step that started it, so mid-battle the facing byte still holds the
    # direction from before that step.
    where += " (in a battle, facing unread)" if battle.get("in_battle") else f" facing {facing}"

    money = player.get("money")
    badges = [str(one) for one in (player.get("badges") or ())]
    if not badges:
        flags = state.get("flags") if isinstance(state.get("flags"), Mapping) else {}
        badges = [str(one) for one in (flags.get("badges") or ())]

    bag: Iterable[Any] = state.get("bag") or ()
    items = ", ".join(
        f"{entry.get('item')} x{entry.get('quantity')}"
        for entry in bag
        if isinstance(entry, Mapping)
    )

    party = state.get("party")
    lines = [where]
    lines.append(f"- Money: ${money}" if money is not None else f"- Money: {UNREAD}")
    lines.append(f"- Badges: {', '.join(badges) if badges else 'none'}")
    lines.extend(_party_lines(party if isinstance(party, Sequence) else ()))
    lines.append(f"- Bag: {items or 'empty'}")
    lines.append(_milestone_line(progress))
    return lines


def state_block(
    state: Optional[Mapping[str, Any]] = None,
    progress: Optional[Mapping[str, Any]] = None,
) -> str:
    """The whole delimited block, delimiters included, ending in one newline."""
    body = "\n".join(state_block_lines(state, progress))
    return f"{BLOCK_BEGIN}\n{BLOCK_TITLE}\n\n{BLOCK_PREAMBLE}\n\n{body}\n{BLOCK_END}\n"


def splice_state_block(existing: Optional[str], block: str) -> str:
    """*existing* with the harness block replaced, or gained, and nothing lost.

    Five shapes reach this function and each of them has to keep the model's
    text:

    * no file, or a file with nothing but whitespace in it - the block plus a
      seeded heading for the model to write under;
    * a well-formed file - the region from the opening delimiter to the first
      closing delimiter after it is replaced, and the rest is copied verbatim;
    * a file whose delimiters are gone, or which starts with something else -
      the block is prepended and every byte of the old file is kept;
    * an opening delimiter with no closing one - there is no way to tell where
      the old block stopped, so nothing is deleted; a fresh block goes on top
      and the orphan survives as text;
    * a second delimiter pair further down, from a copy-paste - only the pair at
      the top is the harness's, so only that one is rewritten.

    The last two settle into the well-formed shape on the next call and are
    byte-stable from then on.
    """
    body = existing or ""
    if not body.strip():
        return f"{block}\n{SEED_BODY}"

    if body.startswith(BLOCK_BEGIN):
        end = body.find(BLOCK_END, len(BLOCK_BEGIN))
        if end != -1:
            rest = body[end + len(BLOCK_END) :].lstrip("\n")
            return f"{block}\n{rest}" if rest else block

    # Anything else: keep it all, put the block above it. Blank lines at the
    # front are dropped so a second pass sees the same bytes a first one wrote.
    kept = body.lstrip("\n")
    spliced = f"{block}\n{kept}"
    return spliced if spliced.endswith("\n") else spliced + "\n"


def strip_state_block(text: str) -> str:
    """*text* without any harness block - what the model actually wrote.

    The critic heads this file "CLAIMS, not facts, and unverified", which is
    true of the model's half and the opposite of the harness's. Handing the
    critic the block would file measurements under claims and spend tokens
    telling it what it already measured.

    *Any* block, not just the first one. :func:`splice_state_block` deliberately
    rewrites only the pair at the top and leaves a copied one further down
    alone, so a model that copy-pastes the block - which is exactly what the
    preamble inside it tells the model not to do - keeps a second one in its own
    half of the file. That copy wears the harness's delimiters, the harness's
    title and the sentence "written by the harness, read from the game", and
    stripping only the first pair sends it to the critic as prose the model
    wrote. It is then a stale measurement in the harness's voice sitting inside
    the section headed "CLAIMS". Nothing wearing these delimiters is the model's
    prose: either the harness wrote it or it is a copy of something the harness
    wrote, and neither belongs under that heading.

    An opening delimiter with no closing one is left alone, for the same reason
    :func:`splice_state_block` leaves it: there is no way to tell where the
    block it opened was meant to stop, and guessing would delete model text.
    """
    body = text or ""
    while True:
        start = body.find(BLOCK_BEGIN)
        if start == -1:
            return body
        end = body.find(BLOCK_END, start + len(BLOCK_BEGIN))
        if end == -1:
            return body
        body = body[:start] + body[end + len(BLOCK_END) :].lstrip("\n")


def notes_path(workspace_dir: Path) -> Path:
    return Path(workspace_dir) / NOTES_FILENAME


def refresh_notes(
    workspace_dir: Path,
    state: Optional[Mapping[str, Any]] = None,
    progress: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Rewrite the harness block in the workspace's ``NOTES.md``.

    Raises on a write it could not do. Callers inside a run catch it and name
    the exception type - a refresh that failed must not take a session down,
    and a refresh that failed silently is the bug that cost this project an
    hour when a ``/route`` fallback swallowed a ``TypeError``.
    """
    path = notes_path(workspace_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A missing file is the first refresh a workspace ever gets. A
    # ``UnicodeDecodeError`` is deliberately not caught: undecodable bytes are
    # not the model's prose and cannot be spliced, so the read fails, the caller
    # names the type, and the file is left byte-for-byte as it was.
    try:
        existing = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing = ""
    spliced = splice_state_block(existing, state_block(state, progress))
    _atomic_write(path, spliced)
    return path


def _atomic_write(path: Path, text: str) -> None:
    """Write via a sibling temp file and rename, so a crash cannot truncate NOTES."""
    tmp = path.with_name(path.name + ".harness-tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
