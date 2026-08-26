"""A retrievable walkthrough library, and a record of what the agent opened.

The agent is given walkthroughs, but nothing here is ever pushed into its
context. It sees `outline()` — a few dozen lines naming every section and what
each one covers — and decides for itself which to open with `read()`. That is
the whole point: choosing *which* route to follow, and *when*, is the agent's
call, not the harness's.

Prose is not checkable, and a wrong walkthrough is worse than no walkthrough
because the agent believes it: `standard_playthrough` said "Exit west onto
Route 4" where the exit is east, and the run spent thousands of presses on it
before anyone read the map. So every section that moves between maps carries a
`<!-- hops: A -east-> B -warp-> C -->` line naming the chain in `world.json`'s
own map names, `read()` renders it above the prose, and a test walks every triple
against the decoded map data. A direction written into that line is checked; a
direction written only into a sentence is not, which is why the sentence is now
the second place a route is stated rather than the first.

`GuideLog` is the other half. Every `read()` the agent performs can be recorded
against the map it was standing on and the press count at the time, so "did it
exercise agency?" becomes a question with an answer: which sections it opened,
how often, and at what point in the run.

Stdlib only. `search()` is plain keyword scoring — no embeddings, no model
calls, no network.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

#: Where the corpus lives. One `.md` file per guide, each a sequence of
#: `##` sections carrying a stable slug.
GUIDES_DIR = Path(__file__).resolve().parent.parent / "data" / "guides"

#: `outline()` is meant to sit inside a prompt. Above this it stops being a
#: cheap index and starts being the thing it was supposed to replace.
OUTLINE_BUDGET_CHARS = 2500

_SECTION_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$", re.MULTILINE)
_META_RE = re.compile(r"<!--\s*(?P<key>[a-z_]+)\s*:\s*(?P<value>.*?)\s*-->", re.DOTALL)
_WORD_RE = re.compile(r"[a-z0-9]+")

#: `A -east-> B`. The edge is a compass direction for a map connection, or the
#: literal `warp` for a door, ladder or cave mouth.
_HOP_RE = re.compile(r"\s*-(?P<edge>north|south|east|west|warp)->\s*")

#: What a hop's edge may say. Anything else is a typo the test will not decode.
HOP_EDGES = ("north", "south", "east", "west", "warp")

#: Field weights for `search`. A hit in the title says far more about what a
#: section is *about* than the same word buried in its body.
_WEIGHT_TITLE = 8
_WEIGHT_SLUG = 6
_WEIGHT_SUMMARY = 4
_WEIGHT_BODY = 1

#: One body mention is a signal; forty is a section that simply uses the word.
_MAX_BODY_HITS = 3


def _log(message: str) -> None:
    print(f"[guides] {message}")


@dataclass(frozen=True)
class Section:
    """One addressable chunk of one guide."""

    guide: str
    slug: str
    title: str
    summary: str
    words: int
    #: `(from_map, edge, to_map)` triples, in the order the section walks them.
    #: Empty for a section that does not move, like anything in `battles`.
    hops: Tuple[Tuple[str, str, str], ...] = ()

    @property
    def ref(self) -> str:
        """`guide/slug` — the stable address of this section."""
        return f"{self.guide}/{self.slug}"

    @property
    def route_line(self) -> str:
        """The hop chain as one line, or `""` for a section that stays put.

        A section can cover two stretches that do not join up — Celadon to Fuchsia
        and Celadon to Saffron are one section and two roads — so the rendering
        starts a new run whenever a hop does not begin where the last one ended.
        """
        if not self.hops:
            return ""
        parts: List[str] = []
        previous = None
        for from_map, edge, to_map in self.hops:
            if from_map != previous:
                parts.append(("; " if parts else "") + from_map)
            parts.append(f"-{edge}-> {to_map}")
            previous = to_map
        return "Route: " + " ".join(parts)


def parse_hops(text: str) -> Tuple[Tuple[str, str, str], ...]:
    """`A -east-> B -warp-> C` into `(("A","east","B"), ("B","warp","C"))`.

    Several chains separated by `;` flatten into one tuple; the gap between them
    shows up as a hop that does not start where the last one ended.

    A malformed chain yields nothing rather than raising: a guide that will not
    parse must still be readable, and the test is where a bad chain is caught.
    """
    hops: List[Tuple[str, str, str]] = []
    for chain in text.split(";"):
        pieces = _HOP_RE.split(chain.strip())
        if len(pieces) < 3 or len(pieces) % 2 == 0:
            return ()
        names = [piece.strip() for piece in pieces[0::2]]
        edges = [piece.strip() for piece in pieces[1::2]]
        if not all(names) or any(edge not in HOP_EDGES for edge in edges):
            return ()
        hops.extend(zip(names, edges, names[1:]))
    return tuple(hops)


@dataclass(frozen=True)
class _Entry:
    """A section plus the body text `index()` deliberately does not expose."""

    section: Section
    body: str
    tokens: Tuple[str, ...]


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _split_front_matter(text: str) -> Tuple[Dict[str, str], str]:
    """Peel a `---`-delimited `key: value` header off the top of a file.

    Deliberately not YAML: the header only ever carries flat strings, and a
    parser dependency for that would be a poor trade.
    """
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    header = text[3:end]
    body = text[end + 4 :]
    meta: Dict[str, str] = {}
    for line in header.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, body.lstrip("\n")


def _parse_guide(path: Path) -> List[_Entry]:
    """Read one guide file into its sections. A malformed file yields nothing."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:  # noqa: BLE001 — a missing guide must not break retrieval
        _log(f"could not read {path}: {exc}")
        return []

    meta, body = _split_front_matter(text)
    guide = meta.get("guide") or path.stem

    matches = list(_SECTION_RE.finditer(body))
    entries: List[_Entry] = []
    for position, match in enumerate(matches):
        start = match.end()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(body)
        chunk = body[start:end]

        section_meta = {m.group("key"): m.group("value") for m in _META_RE.finditer(chunk)}
        slug = section_meta.get("slug", "").strip()
        if not slug:
            _log(f"{path.name}: section {match.group('title')!r} has no slug, skipping")
            continue

        section_body = _META_RE.sub("", chunk).strip()
        title = match.group("title").strip()
        summary = section_meta.get("summary", "").strip() or title
        entries.append(
            _Entry(
                section=Section(
                    guide=guide,
                    slug=slug,
                    title=title,
                    summary=summary,
                    words=len(section_body.split()),
                    hops=parse_hops(section_meta.get("hops", "")),
                ),
                body=section_body,
                tokens=tuple(_tokenize(section_body)),
            )
        )
    return entries


_CACHE: Optional[Tuple[_Entry, ...]] = None
_CACHE_DIR: Optional[Path] = None


def _entries() -> Tuple[_Entry, ...]:
    """Every parsed section, loaded once and reused."""
    global _CACHE, _CACHE_DIR
    if _CACHE is not None and _CACHE_DIR == GUIDES_DIR:
        return _CACHE
    loaded: List[_Entry] = []
    if GUIDES_DIR.is_dir():
        for path in sorted(GUIDES_DIR.glob("*.md")):
            loaded.extend(_parse_guide(path))
    else:
        _log(f"no guide corpus at {GUIDES_DIR}")
    _CACHE = tuple(loaded)
    _CACHE_DIR = GUIDES_DIR
    return _CACHE


def reload() -> None:
    """Drop the parsed corpus, so the next call re-reads from disk."""
    global _CACHE, _CACHE_DIR
    _CACHE = None
    _CACHE_DIR = None


# ----------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------


def index() -> Tuple[Section, ...]:
    """Every section in the library, in file then document order."""
    return tuple(entry.section for entry in _entries())


def guides() -> Tuple[str, ...]:
    """The guide names, in the order they appear in the index."""
    seen: List[str] = []
    for section in index():
        if section.guide not in seen:
            seen.append(section.guide)
    return tuple(seen)


def outline() -> str:
    """The compact listing the agent is shown, in place of the corpus itself.

    Small enough to live in a prompt: one line per section, `slug: summary`,
    grouped under the guide it belongs to.
    """
    sections = index()
    if not sections:
        return "No guides available."

    lines = ["Pokemon Red guide library. read(guide, slug) returns one section."]
    for guide in guides():
        owned = [s for s in sections if s.guide == guide]
        lines.append("")
        lines.append(f"[{guide}] {len(owned)} sections")
        for section in owned:
            lines.append(f"{section.slug}: {section.summary}")
    return "\n".join(lines)


def read(guide: str, slug: str) -> Optional[str]:
    """The body of one section, or `None` for an address that does not exist.

    The checked hop chain leads, because it is the only part of the section a
    test has walked against the decoded maps. The prose that follows says what
    to do on the way; the first line says where the way goes.
    """
    for entry in _entries():
        if entry.section.guide == guide and entry.section.slug == slug:
            route = entry.section.route_line
            return f"{route}\n\n{entry.body}" if route else entry.body
    return None


def find(ref: str) -> Tuple[Section, ...]:
    """Every section addressed by *ref*, as `guide/slug` or as a bare slug.

    A bare slug is the address the agent reaches for, and six of the thirty slugs
    live in two guides at once, so this returns a tuple: one match is the answer,
    several is a question worth asking back, none is a miss worth a suggestion.
    That distinction is why a `read` used to fail outright and cost a second call.
    """
    guide, _, slug = str(ref).strip().partition("/")
    if slug:
        return tuple(
            section for section in index() if section.guide == guide and section.slug == slug
        )
    return tuple(section for section in index() if section.slug == guide)


def search(query: str, limit: int = 5) -> Tuple[Section, ...]:
    """Rank sections against `query` by plain keyword overlap.

    Title, slug and summary hits count for more than body hits, and body hits
    saturate — a section that mentions a term three times is about it; one that
    mentions it forty times is just long.
    """
    terms = _tokenize(query)
    if not terms:
        return ()

    scored: List[Tuple[int, str, str, Section]] = []
    for entry in _entries():
        section = entry.section
        title_tokens = _tokenize(section.title)
        slug_tokens = _tokenize(section.slug)
        summary_tokens = _tokenize(section.summary)

        score = 0
        for term in terms:
            score += _WEIGHT_TITLE * title_tokens.count(term)
            score += _WEIGHT_SLUG * slug_tokens.count(term)
            score += _WEIGHT_SUMMARY * summary_tokens.count(term)
            score += _WEIGHT_BODY * min(entry.tokens.count(term), _MAX_BODY_HITS)
        if score:
            # Guide and slug break ties, so the ranking is stable run to run.
            scored.append((-score, section.guide, section.slug, section))

    scored.sort(key=lambda item: item[:3])
    return tuple(item[3] for item in scored[: max(0, limit)])


# ----------------------------------------------------------------------
# Read telemetry
# ----------------------------------------------------------------------


class GuideLog:
    """Append-only JSONL record of which sections the agent chose to open.

    Persisted with the same atomic write as `pokemon_agent.explored_map`: the
    whole file is rendered to a temporary sibling and `os.replace`d over the
    target, so a reader never sees a torn line. Persistence is best-effort and
    never raises into the caller.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._reads: List[dict] = []
        self._load()

    # -- ingest --------------------------------------------------------

    def record_read(
        self,
        guide: str,
        slug: str,
        *,
        at_map: Optional[str],
        presses: Optional[int],
    ) -> None:
        """Note that the agent opened `guide/slug`, and where it was at the time.

        `at_map` and `presses` are what make the record answerable later: they
        place the read at a point in the run, so its effect on the play that
        followed can be looked for.
        """
        entry = {
            "seq": len(self._reads) + 1,
            "ts": time.time(),
            "guide": str(guide),
            "slug": str(slug),
            "at_map": str(at_map) if at_map is not None else None,
            "presses": int(presses) if presses is not None else None,
        }
        self._reads.append(entry)
        self.save()

    # -- queries -------------------------------------------------------

    def reads(self) -> Tuple[dict, ...]:
        """Every recorded read, oldest first."""
        return tuple(dict(entry) for entry in self._reads)

    def summary(self) -> dict:
        """Which sections were opened, how often, and at which point in the run."""
        per_section: Dict[Tuple[str, str], dict] = {}
        per_guide: Dict[str, int] = {}
        for entry in self._reads:
            key = (entry["guide"], entry["slug"])
            record = per_section.get(key)
            if record is None:
                record = {
                    "guide": entry["guide"],
                    "slug": entry["slug"],
                    "ref": f"{entry['guide']}/{entry['slug']}",
                    "reads": 0,
                    "first_seq": entry["seq"],
                    "last_seq": entry["seq"],
                    "first_presses": entry["presses"],
                    "last_presses": entry["presses"],
                    "maps": [],
                }
                per_section[key] = record
            record["reads"] += 1
            record["last_seq"] = entry["seq"]
            if entry["presses"] is not None:
                if record["first_presses"] is None:
                    record["first_presses"] = entry["presses"]
                record["last_presses"] = entry["presses"]
            if entry["at_map"] and entry["at_map"] not in record["maps"]:
                record["maps"].append(entry["at_map"])
            per_guide[entry["guide"]] = per_guide.get(entry["guide"], 0) + 1

        sections = sorted(
            per_section.values(),
            key=lambda record: (-record["reads"], record["guide"], record["slug"]),
        )
        return {
            "total_reads": len(self._reads),
            "unique_sections": len(per_section),
            "guides": per_guide,
            "sections": sections,
            "repeat_reads": sum(r["reads"] - 1 for r in sections),
            "first_read": dict(self._reads[0]) if self._reads else None,
            "last_read": dict(self._reads[-1]) if self._reads else None,
        }

    # -- persistence ---------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        loaded: List[dict] = []
        try:
            lines: Iterable[str] = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:  # noqa: BLE001 — a bad file must never break a read
            _log(f"ignoring unreadable log {self.path}: {exc}")
            return
        for number, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    raise TypeError("line is not an object")
            except Exception as exc:  # noqa: BLE001 — skip the line, keep the file
                _log(f"{self.path}:{number}: ignoring bad line: {exc}")
                continue
            entry.setdefault("seq", len(loaded) + 1)
            entry.setdefault("ts", None)
            entry.setdefault("at_map", None)
            entry.setdefault("presses", None)
            loaded.append(entry)
        self._reads = loaded

    def save(self) -> None:
        """Rewrite the log atomically. Never raises into the caller."""
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
                    for entry in self._reads:
                        handle.write(json.dumps(entry, separators=(",", ":")))
                        handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, self.path)
            except Exception:
                temp_path.unlink(missing_ok=True)
                raise
        except Exception as exc:  # noqa: BLE001 — persistence is best-effort
            _log(f"could not save {self.path}: {exc}")


__all__ = [
    "GUIDES_DIR",
    "OUTLINE_BUDGET_CHARS",
    "GuideLog",
    "Section",
    "guides",
    "index",
    "outline",
    "read",
    "reload",
    "search",
]
