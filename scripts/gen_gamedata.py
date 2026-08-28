#!/usr/bin/env python3
"""Generate the static Pokemon Red game database from the pret/pokered decomp.

Run:  .venv/bin/python scripts/gen_gamedata.py

Everything in pokemon_agent/data/game/*.json is produced by this script and
nothing else -- edit the parsers here, never the JSON. Re-running with the same
upstream commit rewrites byte-identical files, so it is safe to run any time.

Source of truth
---------------
https://github.com/pret/pokered at the commit recorded in every output file's
"generated_from" field. By default the generator resolves master; pass --sha to
pin an older commit. Files are fetched over HTTPS (raw.githubusercontent.com)
and cached on disk, so a second run costs no network.

Map names
---------
Every map key in every output file is a name that already exists in
MAP_NAMES in pokemon_agent/memory/red.py. pokered's own labels (`PalletTown`)
and constants (`PALLET_TOWN`) are NOT those names, so everything goes through
MapNames below, which fails loudly on anything it cannot resolve. A silent
mismatch here poisons every downstream tool.

Joins with the live memory reader
---------------------------------
Species, move, item and type names are likewise taken from red.py's tables
(SPECIES_NAMES, MOVE_NAMES, ITEM_NAMES, TYPE_NAMES) by resolving pokered's
constants to the ids those tables are keyed by. That is what lets the agent
read "Pidgey" out of RAM and look it up in species.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pokemon_agent.memory.red import (  # noqa: E402
    INTERNAL_SPECIES_TO_DEX,
    ITEM_NAMES,
    MAP_NAMES,
    MOVE_NAMES,
    SPECIES_NAMES,
    TYPE_NAMES,
)

RAW = "https://raw.githubusercontent.com/pret/pokered/{sha}/{path}"
API_DIR = "https://api.github.com/repos/pret/pokered/contents/{path}?ref={sha}"
API_COMMIT = "https://api.github.com/repos/pret/pokered/commits/master"

DEFAULT_OUT = REPO_ROOT / "pokemon_agent" / "data" / "game"
DEFAULT_CACHE = Path(tempfile.gettempdir()) / "pokered-cache"

# A pokered block is 2x2 tiles. map_constants.asm gives map dimensions in
# BLOCKS; warps, objects and the player's own wYCoord/wXCoord are all in TILES,
# so world.json reports tiles (and keeps the raw block figure alongside it).
TILES_PER_BLOCK = 2

# data/types/type_matchups.asm stores the multiplier x10.
EFFECT_MULTIPLIERS = {"SUPER_EFFECTIVE": 2.0, "NOT_VERY_EFFECTIVE": 0.5, "NO_EFFECT": 0.0}


class GenError(RuntimeError):
    """Anything that means the generated data would be wrong. Never swallowed."""


# ===================================================================
# Fetching
# ===================================================================


class Fetcher:
    def __init__(self, sha: str, cache_dir: Path):
        self.sha = sha
        self.cache_dir = cache_dir / sha
        self.hits = 0
        self.downloads = 0

    def _get(self, url: str, tries: int = 4) -> bytes:
        last: Optional[Exception] = None
        for attempt in range(tries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "pokemon-agent-gamedata"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return resp.read()
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    raise GenError(f"404 fetching {url}") from exc
                last = exc
            except Exception as exc:  # network flake
                last = exc
            time.sleep(1.5 * (attempt + 1))
        raise GenError(f"failed to fetch {url}: {last}")

    def text(self, path: str) -> str:
        cached = self.cache_dir / path
        if cached.is_file():
            self.hits += 1
            return cached.read_text(encoding="utf-8")
        body = self._get(RAW.format(sha=self.sha, path=path))
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(body)
        self.downloads += 1
        return body.decode("utf-8")

    def many(self, paths: Sequence[str]) -> Dict[str, str]:
        with ThreadPoolExecutor(max_workers=8) as pool:
            return dict(zip(paths, pool.map(self.text, paths)))

    def listdir(self, path: str) -> List[str]:
        cached = self.cache_dir / "_listing" / (path.replace("/", "_") + ".json")
        if cached.is_file():
            self.hits += 1
            entries = json.loads(cached.read_text(encoding="utf-8"))
        else:
            body = self._get(API_DIR.format(path=path, sha=self.sha))
            entries = [
                {"name": e["name"], "type": e["type"]} for e in json.loads(body.decode("utf-8"))
            ]
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(entries), encoding="utf-8")
            self.downloads += 1
        return sorted(e["name"] for e in entries if e["type"] == "file")


def resolve_master_sha() -> str:
    req = urllib.request.Request(API_COMMIT, headers={"User-Agent": "pokemon-agent-gamedata"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))["sha"]


# ===================================================================
# asm helpers
# ===================================================================


def strip_comment(line: str) -> str:
    return line.split(";", 1)[0].rstrip()


def _branch_taken(condition: str) -> bool:
    """We build the Red database, so `IF DEF(_RED)` is live and `_BLUE` is not."""
    if "_BLUE" in condition:
        return False
    return True


def joined_lines(text: str) -> List[str]:
    """Logical lines: backslash continuations glued, Blue-only branches dropped."""
    out: List[str] = []
    pending = ""
    # stack of (this branch is live, some branch of this IF was already taken)
    conditionals: List[List[bool]] = []
    for raw in text.splitlines():
        line = strip_comment(raw)
        if line.rstrip().endswith("\\"):
            pending += line.rstrip()[:-1] + " "
            continue
        line = (pending + line).strip()
        pending = ""

        keyword = line.split("(", 1)[0].split()[0].upper() if line.strip() else ""
        if keyword == "IF":
            taken = _branch_taken(line)
            conditionals.append([taken, taken])
            continue
        if keyword == "ELIF" and conditionals:
            taken = not conditionals[-1][1] and _branch_taken(line)
            conditionals[-1] = [taken, conditionals[-1][1] or taken]
            continue
        if keyword == "ELSE" and conditionals:
            conditionals[-1] = [not conditionals[-1][1], True]
            continue
        if keyword == "ENDC" and conditionals:
            conditionals.pop()
            continue
        if all(live for live, _ in conditionals):
            out.append(line)
    if pending:
        out.append(pending.strip())
    return out


def directive(line: str) -> Tuple[str, List[str]]:
    """Split "\tmove POUND, X, 40" into ("move", ["POUND", "X", "40"])."""
    line = line.strip()
    if not line:
        return "", []
    head, _, rest = line.partition(" ")
    if "\t" in head:
        head, _, rest2 = line.partition("\t")
        rest = rest2
    args = [a.strip() for a in rest.split(",")] if rest.strip() else []
    return head.strip(), [a for a in args if a]


def numeric(token: str) -> int:
    token = token.strip()
    if token.startswith("$"):
        return int(token[1:], 16)
    if token.startswith("-"):
        return -numeric(token[1:])
    return int(token)


def positional_consts(text: str, macro: str = "const") -> Dict[int, str]:
    """value -> constant name, honouring const_def/const_next/const_skip.

    const_skip is why this cannot be a plain list: the species table skips the
    MissingNo slots, and counting lines instead of values shifts every id after
    them -- the same class of bug that has bitten MAP_NAMES twice.
    """
    out: Dict[int, str] = {}
    value = 0
    in_macro = False
    for line in joined_lines(text):
        head, args = directive(line)
        if head.upper() == "MACRO":
            in_macro = True
            continue
        if head.upper() == "ENDM":
            in_macro = False
            continue
        if in_macro:
            continue
        if head == "const_def":
            value = numeric(args[0]) if args else 0
        elif head == "const_next":
            value = numeric(args[0])
        elif head == "const_skip":
            value += numeric(args[0]) if args else 1
        elif head == macro and args:
            out[value] = args[0]
            value += 1
    return out


# ===================================================================
# Map name normalisation -- the load-bearing part
# ===================================================================


class MapNames:
    """pokered map constant / label  ->  the exact name in red.py's MAP_NAMES.

    Ids are positional in constants/map_constants.asm, which is where red.py's
    table came from, so the two are zipped and any length mismatch is fatal.
    """

    def __init__(self, fetcher: Fetcher):
        text = fetcher.text("constants/map_constants.asm")
        self.order: List[Tuple[str, int, int]] = []
        for line in joined_lines(text):
            head, args = directive(line)
            if head == "map_const" and len(args) == 3:
                self.order.append((args[0], numeric(args[1]), numeric(args[2])))
        if len(self.order) != len(MAP_NAMES):
            raise GenError(
                f"map_constants.asm has {len(self.order)} maps but red.py MAP_NAMES has "
                f"{len(MAP_NAMES)}. The table has drifted -- fix red.py before regenerating."
            )
        self.const_to_id = {c: i for i, (c, _, _) in enumerate(self.order)}
        self.const_to_name = {c: MAP_NAMES[i] for i, (c, _, _) in enumerate(self.order)}
        self.blocks = {c: (w, h) for c, w, h in self.order}
        self.label_to_const: Dict[str, str] = {}

    def learn_label(self, label: str, const: str) -> None:
        known = self.label_to_const.get(label)
        if known and known != const:
            raise GenError(f"map label {label!r} maps to both {known} and {const}")
        if const not in self.const_to_name:
            raise GenError(f"map constant {const!r} (label {label!r}) is not in map_constants.asm")
        self.label_to_const[label] = const

    def const(self, const: str) -> str:
        try:
            return self.const_to_name[const]
        except KeyError:
            raise GenError(
                f"map constant {const!r} does not resolve to a MAP_NAMES entry"
            ) from None

    def label(self, label: str) -> str:
        try:
            return self.const_to_name[self.label_to_const[label]]
        except KeyError:
            raise GenError(
                f"map label {label!r} does not resolve to a map constant "
                "(no data/maps/headers file declares it)"
            ) from None


# ===================================================================
# Name tables shared by several outputs
# ===================================================================


class NameTables:
    """pokered constants -> the display names red.py reads out of RAM."""

    def __init__(self, fetcher: Fetcher):
        self.warnings: List[str] = []

        # -- species: internal index order defines the constants, dex order the names
        internal = positional_consts(fetcher.text("constants/pokemon_constants.asm"))
        dex_consts = positional_consts(fetcher.text("constants/pokedex_constants.asm"))
        self.dex_of_const = {c.removeprefix("DEX_"): dex for dex, c in dex_consts.items()}
        # pokemon_constants.asm starts at NO_MON = $00, so the position in the
        # list is the internal species index the party struct stores.
        self.internal_to_const = internal
        self.species_by_internal: Dict[int, str] = {}
        for idx, const in self.internal_to_const.items():
            dex = self.dex_of_const.get(const)
            if dex is None:  # MISSINGNO / unused slots
                continue
            self.species_by_internal[idx] = const
            known = INTERNAL_SPECIES_TO_DEX.get(idx)
            if known != dex:
                self.warnings.append(
                    f"red.py INTERNAL_SPECIES_TO_DEX[{idx}] is {known}, "
                    f"pokered says {dex} ({const})"
                )

        # -- moves: position in data/moves/moves.asm is the move id
        self.move_id_of_const: Dict[str, int] = {}
        for line in joined_lines(fetcher.text("data/moves/moves.asm")):
            head, args = directive(line)
            if head == "move" and len(args) == 6 and not line.startswith("MACRO"):
                self.move_id_of_const[args[0]] = len(self.move_id_of_const) + 1

        # -- items: position in constants/item_constants.asm is the item id,
        #    with TMs/HMs added by their own macros further down the file.
        self.item_id_of_const: Dict[str, int] = {}
        self.tm_hm_of_move: Dict[str, str] = {}
        item_id = 0
        hm_n = tm_n = 0
        in_macro = False
        for line in joined_lines(fetcher.text("constants/item_constants.asm")):
            head, args = directive(line)
            if head.upper() == "MACRO":
                in_macro = True
                continue
            if head.upper() == "ENDM":
                in_macro = False
                continue
            if in_macro or not args:
                continue
            if head == "const":
                self.item_id_of_const[args[0]] = item_id
                item_id += 1
            elif head == "const_next":
                item_id = numeric(args[0])  # the ids jump to $C4 for the HMs
            elif head == "add_hm":
                # add_hm CUT defines the item HM_CUT and makes CUT teachable as HM01.
                hm_n += 1
                label = f"HM{hm_n:02d}"
                self.item_id_of_const[label] = item_id
                self.item_id_of_const[f"HM_{args[0]}"] = item_id
                self.tm_hm_of_move[args[0]] = label
                item_id += 1
            elif head == "add_tm":
                tm_n += 1
                label = f"TM{tm_n:02d}"
                self.item_id_of_const[label] = item_id
                self.item_id_of_const[f"TM_{args[0]}"] = item_id
                self.tm_hm_of_move[args[0]] = label
                item_id += 1

        # -- types
        self.type_value_of_const: Dict[str, int] = {}
        value = 0
        for line in joined_lines(fetcher.text("constants/type_constants.asm")):
            head, args = directive(line)
            if head == "const" and args:
                self.type_value_of_const[args[0]] = value
                value += 1
            elif head == "const_next" and args:
                value = numeric(args[0])

    # -- lookups, all fatal on a miss ------------------------------------
    def species(self, const: str) -> str:
        dex = self.dex_of_const.get(const)
        if dex is None or dex not in SPECIES_NAMES:
            raise GenError(f"species constant {const!r} does not resolve to a Pokedex entry")
        return SPECIES_NAMES[dex]

    def dex(self, const: str) -> int:
        dex = self.dex_of_const.get(const)
        if dex is None:
            raise GenError(f"species constant {const!r} has no Pokedex number")
        return dex

    def move(self, const: str) -> str:
        move_id = self.move_id_of_const.get(const)
        if move_id is None or move_id not in MOVE_NAMES:
            raise GenError(f"move constant {const!r} does not resolve to a move id")
        return MOVE_NAMES[move_id]

    def item(self, const: str) -> str:
        item_id = self.item_id_of_const.get(const)
        if item_id is None or item_id not in ITEM_NAMES:
            raise GenError(f"item constant {const!r} does not resolve to an item id")
        return ITEM_NAMES[item_id]

    def item_by_id(self, item_id: int) -> str:
        """The display name for a raw item id, for tables indexed by id."""
        if item_id not in ITEM_NAMES:
            raise GenError(f"item id {item_id} is not in red.py's ITEM_NAMES")
        return ITEM_NAMES[item_id]

    def type_(self, const: str) -> str:
        value = self.type_value_of_const.get(const)
        if value is None or value not in TYPE_NAMES:
            raise GenError(f"type constant {const!r} does not resolve to a type id")
        return TYPE_NAMES[value]


# ===================================================================
# world.json
# ===================================================================

DIRECTIONS = ("north", "south", "west", "east")


def parse_headers(fetcher: Fetcher) -> Dict[str, Dict[str, Any]]:
    """Keyed by map label (the file stem), because one header lies about its
    constant: UndergroundPathRoute7Copy declares UNDERGROUND_PATH_ROUTE_7. The
    objects file's def_warps_to is the map's real identity."""
    names = fetcher.listdir("data/maps/headers")
    texts = fetcher.many([f"data/maps/headers/{n}" for n in names])
    raw: Dict[str, Dict[str, Any]] = {}
    for path, text in texts.items():
        label = Path(path).stem
        declared = None
        tileset = ""
        connections: List[Tuple[str, str]] = []
        for line in joined_lines(text):
            head, args = directive(line)
            if head == "map_header":
                if len(args) < 2:
                    raise GenError(f"{path}: unparseable map_header {line!r}")
                if args[0] != label:
                    raise GenError(f"{path}: map_header label {args[0]!r} != file name {label!r}")
                declared = args[1]
                tileset = args[2] if len(args) > 2 else ""
            elif head == "connection":
                if args[0] not in DIRECTIONS:
                    raise GenError(f"{path}: unknown connection direction {args[0]!r}")
                connections.append((args[0], args[2]))
        if declared is None:
            raise GenError(f"{path}: no map_header line")
        raw[label] = {"declared_const": declared, "tileset": tileset, "connections": connections}
    return raw


def parse_objects(fetcher: Fetcher, maps: MapNames) -> Dict[str, Dict[str, Any]]:
    names = fetcher.listdir("data/maps/objects")
    texts = fetcher.many([f"data/maps/objects/{n}" for n in names])
    out: Dict[str, Dict[str, Any]] = {}
    for path, text in texts.items():
        label = Path(path).stem
        const = None
        warps: List[Dict[str, Any]] = []
        objects: List[Dict[str, Any]] = []
        for line in joined_lines(text):
            head, args = directive(line)
            if head == "def_warps_to":
                const = args[0]
            elif head == "warp_event":
                # warp_event x, y, DEST_MAP, dest warp id (1-based in source)
                warps.append(
                    {
                        "x": numeric(args[0]),
                        "y": numeric(args[1]),
                        "dest_const": args[2],
                        "to_warp": numeric(args[3]) - 1,
                    }
                )
            elif head == "object_event":
                # 6 args: a person. 7: a ground item. 8: a trainer or a static
                # wild encounter (Snorlax, the birds, Mewtwo).
                entry = {"x": numeric(args[0]), "y": numeric(args[1]), "sprite": args[2]}
                # The text id is what joins an object to the script it runs, and
                # it is the only way to tell which of a mart's three NPCs is the
                # one standing behind the till. See build_shops.
                if len(args) >= 6:
                    entry["text"] = args[5]
                if len(args) == 7:
                    entry["item"] = args[6]
                elif len(args) == 8:
                    entry["who"] = args[6]
                    entry["n"] = numeric(args[7])
                objects.append(entry)
        if const is None:
            raise GenError(f"{path}: no def_warps_to line")
        maps.learn_label(label, const)
        out[label] = {"const": const, "warps": warps, "objects": objects}
    return out


def resolve_last_map(
    const: str,
    warp_index: int,
    warps_by_const: Dict[str, List[Dict[str, Any]]],
    outdoor: set,
) -> Optional[str]:
    """LAST_MAP means "wherever the player came from" -- a door, resolved at runtime.

    Three narrowing passes, most trustworthy first:
      1. only one map warps in here at all;
      2. the map whose warp at this exact index leads back here;
      3. of those, the one you are outdoors on (a house door leads to the town,
         not to the upstairs the other staircase came from).
    A door that is still ambiguous after that -- Victory Road 2F, reachable from
    both the floor above and the floor below -- is left unresolved rather than
    guessed at.
    """
    candidates = [
        src for src, warps in warps_by_const.items() if any(w["dest_const"] == const for w in warps)
    ]
    if len(candidates) == 1:
        return candidates[0]
    exact = [
        c
        for c in candidates
        if warp_index < len(warps_by_const[c])
        and warps_by_const[c][warp_index]["dest_const"] == const
    ]
    if len(exact) == 1:
        return exact[0]
    outside = [c for c in (exact or candidates) if c in outdoor]
    if len(outside) == 1:
        return outside[0]
    return None


def build_world(
    headers: Dict[str, Dict[str, Any]],
    objects: Dict[str, Dict[str, Any]],
    maps: MapNames,
    report: List[str],
) -> Dict[str, Any]:
    warps_by_const = {data["const"]: data["warps"] for data in objects.values()}
    # "Outdoors" = the towns and routes you can walk between: an OVERWORLD
    # tileset, or a map with connections (Indigo Plateau uses PLATEAU).
    outdoor = {
        objects[label]["const"]
        for label, header in headers.items()
        if label in objects and (header["tileset"] == "OVERWORLD" or header["connections"])
    }
    unresolved = 0
    out: Dict[str, Any] = {}
    for label, header in headers.items():
        obj = objects.get(label)
        if obj is None:
            raise GenError(f"{label} has a header but no data/maps/objects file")
        const = obj["const"]
        if header["declared_const"] != const:
            report.append(
                f"{label}: header declares {header['declared_const']}, "
                f"objects say {const} (using the objects file)"
            )
        width_blocks, height_blocks = maps.blocks[const]
        warps: List[Dict[str, Any]] = []
        for index, warp in enumerate(obj["warps"]):
            dest_const = warp["dest_const"]
            if dest_const == "LAST_MAP":
                dest_const = resolve_last_map(const, warp["to_warp"], warps_by_const, outdoor)
            to_map = maps.const(dest_const) if dest_const else None
            if to_map is None:
                unresolved += 1
            elif dest_const in warps_by_const:
                count = len(warps_by_const[dest_const])
                if not 0 <= warp["to_warp"] < count:
                    report.append(
                        f"{maps.const(const)} warp {index} points at {to_map} warp "
                        f"{warp['to_warp']}, which only has {count} warps"
                    )
            warps.append(
                {
                    "x": warp["x"],
                    "y": warp["y"],
                    "to_map": to_map,
                    "to_warp": warp["to_warp"],
                }
            )
        out[maps.const(const)] = {
            "map_id": maps.const_to_id[const],
            # tiles, not blocks -- see TILES_PER_BLOCK
            "size": [width_blocks * TILES_PER_BLOCK, height_blocks * TILES_PER_BLOCK],
            "size_blocks": [width_blocks, height_blocks],
            "connections": {d: maps.const(c) for d, c in header["connections"]},
            "warps": warps,
        }
    if unresolved:
        report.append(f"{unresolved} LAST_MAP warps had no unambiguous source map (to_map: null)")
    return out


# ===================================================================
# trainers.json
# ===================================================================


def parse_trainer_parties(fetcher: Fetcher, names: NameTables) -> Dict[str, List[List[Dict]]]:
    text = fetcher.text("data/trainers/parties.asm")
    lines = joined_lines(text)
    order: List[str] = []
    parties: Dict[str, List[List[Dict[str, Any]]]] = {}
    current: Optional[str] = None
    in_pointer_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("TrainerDataPointers"):
            in_pointer_table = True
            continue
        head, args = directive(stripped)
        if in_pointer_table:
            if head == "dw":
                order.append(args[0])
                continue
            if head.startswith("assert_table_length"):
                in_pointer_table = False
            continue
        if stripped.endswith(":") and not stripped.startswith("db"):
            current = stripped[:-1].strip(":")
            parties.setdefault(current, [])
            continue
        if head == "db" and current:
            team: List[Dict[str, Any]] = []
            if args[0] == "$FF":
                rest = args[1:]
                while rest and rest[0] != "0":
                    team.append({"species": names.species(rest[1]), "level": numeric(rest[0])})
                    rest = rest[2:]
            else:
                level = numeric(args[0])
                for token in args[1:]:
                    if token == "0":
                        break
                    team.append({"species": names.species(token), "level": level})
            if team:
                parties[current].append(team)
    return {label: parties[label] for label in order}


def parse_trainer_classes(fetcher: Fetcher) -> Tuple[List[str], List[str]]:
    by_id = positional_consts(fetcher.text("constants/trainer_constants.asm"), "trainer_const")
    consts = [c for _, c in sorted(by_id.items()) if c != "NOBODY"]
    display: List[str] = []
    for line in joined_lines(fetcher.text("data/trainers/names.asm")):
        head, args = directive(line)
        if head == "li" and args:
            display.append(args[0].strip().strip('"'))
    if len(consts) != len(display):
        raise GenError(f"{len(consts)} trainer constants but {len(display)} trainer names")
    return consts, [title_case_trainer(n) for n in display]


def title_case_trainer(name: str) -> str:
    """ "BUG CATCHER" -> "Bug Catcher", "LT.SURGE" -> "Lt.Surge"."""

    def cap(word: str) -> str:
        return word[:1].upper() + word[1:].lower() if word else word

    parts = [".".join(cap(p) for p in word.split(".")) for word in name.split(" ")]
    return " ".join(parts)


def build_trainers(
    objects: Dict[str, Dict[str, Any]],
    maps: MapNames,
    parties: Dict[str, List[List[Dict]]],
    classes: Tuple[List[str], List[str]],
    report: List[str],
) -> Dict[str, Any]:
    class_consts, class_names = classes
    labels = list(parties)
    if len(labels) != len(class_consts):
        raise GenError(f"{len(labels)} party lists but {len(class_consts)} trainer classes")
    party_of_const = dict(zip(class_consts, labels))
    name_of_const = dict(zip(class_consts, class_names))

    out: Dict[str, List[Dict[str, Any]]] = {}
    static_encounters = 0
    for data in objects.values():
        const = data["const"]
        placed: List[Dict[str, Any]] = []
        for obj in data["objects"]:
            who = obj.get("who")
            if not who:
                continue
            if not who.startswith("OPP_"):
                static_encounters += 1  # a fixed wild mon, not a trainer
                continue
            class_const = who[len("OPP_") :]
            if class_const not in party_of_const:
                raise GenError(f"{const}: unknown trainer class {who}")
            teams = parties[party_of_const[class_const]]
            index = obj["n"]
            if not 1 <= index <= len(teams):
                raise GenError(
                    f"{const}: {who} party {index} out of range (class has {len(teams)})"
                )
            placed.append(
                {
                    "trainer_class": name_of_const[class_const],
                    "index": index,
                    "x": obj["x"],
                    "y": obj["y"],
                    "team": teams[index - 1],
                }
            )
        if placed:
            out[maps.const(const)] = placed
    report.append(f"{static_encounters} static wild encounters skipped (not trainers)")
    return out


# ===================================================================
# encounters.json
# ===================================================================

# data/wild/probabilities.asm: the chance of each of the ten slots, /256.
SLOT_CHANCES = [51, 51, 39, 25, 25, 25, 13, 13, 11, 3]


def parse_slot_chances(fetcher: Fetcher) -> List[float]:
    weights: List[int] = []
    for line in joined_lines(fetcher.text("data/wild/probabilities.asm")):
        head, args = directive(line)
        if head == "wild_chance" and args:
            weights.append(numeric(args[0]))
    if sum(weights) != 256:
        raise GenError(f"wild slot chances sum to {sum(weights)}, not 256")
    if weights != SLOT_CHANCES:
        raise GenError(f"wild slot chances changed upstream: {weights}")
    return [round(w / 256, 4) for w in weights]


def build_encounters(fetcher: Fetcher, names: NameTables, chances: List[float]) -> Dict[str, Any]:
    # WildDataPointers is indexed by map id, and several maps share one table
    # (Routes 19 and 20 both point at SeaRoutesWildMons), so the pointer table
    # -- not the file name -- decides which map gets which encounters.
    pointers: List[str] = []
    in_table = False
    for line in joined_lines(fetcher.text("data/wild/grass_water.asm")):
        stripped = line.strip()
        if stripped.startswith("WildDataPointers"):
            in_table = True
            continue
        head, args = directive(stripped)
        if in_table and head == "dw":
            pointers.append(args[0])
        elif in_table and head.startswith("assert_table_length"):
            in_table = False
    if len(pointers) != len(MAP_NAMES):
        raise GenError(f"WildDataPointers has {len(pointers)} entries, expected {len(MAP_NAMES)}")

    files = fetcher.listdir("data/wild/maps")
    texts = fetcher.many([f"data/wild/maps/{n}" for n in files])
    by_label: Dict[str, Any] = {}
    for path, text in sorted(texts.items()):
        label = Path(path).stem
        section: Optional[str] = None
        blocks: Dict[str, Dict[str, Any]] = {"grass": None, "water": None}
        for line in joined_lines(text):
            head, args = directive(line)
            if head == "def_grass_wildmons":
                section = "grass"
                blocks[section] = {"rate": numeric(args[0]), "slots": []}
            elif head == "def_water_wildmons":
                section = "water"
                blocks[section] = {"rate": numeric(args[0]), "slots": []}
            elif head in ("end_grass_wildmons", "end_water_wildmons"):
                section = None
            elif head == "db" and section and len(args) == 2:
                slots = blocks[section]["slots"]
                slots.append(
                    {
                        "species": names.species(args[1]),
                        "level": numeric(args[0]),
                        "chance": chances[len(slots)],
                    }
                )
        for key, block in list(blocks.items()):
            if block is not None and not block["slots"]:
                blocks[key] = None  # rate 0, nothing lives here
            elif block is not None and len(block["slots"]) != len(chances):
                raise GenError(f"{path}: {key} has {len(block['slots'])} slots, expected 10")
        by_label[f"{label}WildMons"] = blocks

    out: Dict[str, Any] = {}
    for map_id, pointer in enumerate(pointers):
        if pointer == "NothingWildMons":
            continue
        if pointer not in by_label:
            raise GenError(f"{MAP_NAMES[map_id]} points at {pointer}, which has no wild data file")
        out[MAP_NAMES[map_id]] = by_label[pointer]
    return out


# ===================================================================
# items.json
# ===================================================================


def build_items(
    fetcher: Fetcher, objects: Dict[str, Dict[str, Any]], maps: MapNames, names: NameTables
) -> Dict[str, Any]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for data in objects.values():
        for obj in data["objects"]:
            # An item id of 0 is "no item" -- a plain person written with the
            # seven-argument form (Blue's sister, the Town Map on the desk).
            if "item" not in obj or obj["item"] == "0":
                continue
            out.setdefault(maps.const(data["const"]), []).append(
                {
                    "x": obj["x"],
                    "y": obj["y"],
                    "item": names.item(obj["item"]),
                    "hidden": False,
                }
            )

    # Hidden items and hidden Game Corner coins live in the hidden-event table,
    # grouped by map. hidden_event's args are x, y, handler, argument.
    text = fetcher.text("data/events/hidden_events.asm")
    current: Optional[str] = None
    in_macro = False
    for line in joined_lines(text):
        head, args = directive(line)
        if head.upper() == "MACRO":
            in_macro = True
            continue
        if head.upper() == "ENDM":
            in_macro = False
            continue
        if in_macro:
            continue
        if head == "hidden_events_for":
            current = args[0]
        elif head == "hidden_event" and current and len(args) == 4:
            handler = args[2]
            if handler == "HiddenItems":
                entry = {
                    "x": numeric(args[0]),
                    "y": numeric(args[1]),
                    "item": names.item(args[3]),
                    "hidden": True,
                }
            elif handler == "HiddenCoins":
                # argument is COIN + <amount>
                amount = numeric(args[3].split("+", 1)[1])
                entry = {
                    "x": numeric(args[0]),
                    "y": numeric(args[1]),
                    "item": names.item("COIN"),
                    "hidden": True,
                    "amount": amount,
                }
            else:
                continue
            out.setdefault(maps.const(current), []).append(entry)
    return {
        name: sorted(entries, key=lambda e: (e["hidden"], e["y"], e["x"]))
        for name, entries in out.items()
    }


# ===================================================================
# shops.json
# ===================================================================

CLERK_SUFFIX = re.compile(r"(Clerk\d*|Cashier\d*)$")


def _clerk_key(name: str) -> str:
    """A clerk's name with the spelling taken out, so two spellings of it join.

    A mart script label and the object's text id name the same person
    differently: ``CeladonMart2FClerk1Text`` against
    ``TEXT_CELADONMART2F_CLERK1``. Strip the decoration off each and they are
    the same string, which is what lets a counter carry the tile its clerk
    stands on. Checked on all fourteen referenced marts; every one joins, and
    :func:`build_shops` raises rather than emit a counter that does not.
    """
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def parse_item_prices(fetcher: Fetcher, names: NameTables) -> Dict[str, int]:
    """Every item's shop price, by display name.

    Two tables, because the game keeps them apart: ItemPrices is one BCD entry
    per item id starting at MASTER_BALL = $01, and the TMs are nybbles of
    thousands in a second table indexed by TM number. A mart that sells TMs
    (Celadon 2F) needs both, and a price of 0 means "not for sale" rather than
    "free", so those are dropped instead of recorded.
    """
    prices: Dict[str, int] = {}
    item_id = 0
    for line in joined_lines(fetcher.text("data/items/prices.asm")):
        head, args = directive(line)
        if head != "bcd3" or not args:
            continue
        item_id += 1  # the first bcd3 entry is item id $01
        value = numeric(args[0])
        if value:
            prices[names.item_by_id(item_id)] = value
    tm = 0
    for line in joined_lines(fetcher.text("data/items/tm_prices.asm")):
        head, args = directive(line)
        if head != "nybble" or not args:
            continue
        tm += 1
        value = numeric(args[0]) * 1000
        if value:
            prices[f"TM{tm:02d}"] = value
    for check, expect in (("Poke Ball", 200), ("Potion", 300), ("TM01", 3000)):
        if prices.get(check) != expect:
            raise GenError(f"{check} priced {prices.get(check)}, expected {expect}")
    return prices


def build_shops(
    fetcher: Fetcher,
    maps: MapNames,
    names: NameTables,
    objects: Dict[str, Dict[str, Any]],
    report: List[str],
) -> Dict:
    text = fetcher.text("data/items/marts.asm")
    prices = parse_item_prices(fetcher, names)
    out: Dict[str, Dict[str, Any]] = {}
    label: Optional[str] = None
    skipped: List[str] = []
    for line in joined_lines(text):
        stripped = line.strip()
        if stripped.endswith(("::", ":")) and "script_mart" not in stripped:
            label = stripped.rstrip(":")
            continue
        head, args = directive(stripped)
        if head != "script_mart" or label is None:
            continue
        base = CLERK_SUFFIX.sub("", label.removesuffix("Text"))
        if base not in maps.label_to_const:
            skipped.append(label)  # unreferenced marts with no map of their own
            label = None
            continue
        map_name = maps.label(base)
        items = [names.item(a) for a in args]
        # The tile the clerk stands on. Without it "buy a Potion" starts with
        # "find the till in a room with three identical-looking NPCs in it",
        # which is a walk the agent has to guess its way through.
        wanted = _clerk_key(label.removesuffix("Text"))
        at = [
            [obj["x"], obj["y"]]
            for obj in objects.get(base, {}).get("objects", ())
            if _clerk_key(str(obj.get("text") or "").removeprefix("TEXT_")) == wanted
        ]
        if len(at) != 1:
            raise GenError(f"{label}: matched {len(at)} objects on {map_name}, expected 1")
        entry = out.setdefault(map_name, {"items": [], "prices": {}, "counters": []})
        entry["counters"].append({"label": label, "at": at[0], "items": items})
        for item in items:
            if item not in entry["items"]:
                entry["items"].append(item)
            if item in prices:
                entry["prices"][item] = prices[item]
        label = None
    if skipped:
        report.append(f"marts with no map of their own, skipped: {', '.join(sorted(skipped))}")
    return out


# ===================================================================
# services.json
# ===================================================================
#
# shops.json answers "what is for sale here". This answers the wider question
# it is a special case of: which person in this room does something for you,
# and where do they stand. `poke map` in a Poke Center printed the room size,
# the two warps and the nearest unexplored tile in a room that was already 100%
# seen -- it never mentioned the nurse, and one 34-hour run spent 4,152 presses
# across 116 Center visits hunting for her.
#
# The join is over what a script *does*, not over which room it is in or which
# sprite is drawn. Every map script carries a text-pointer table --
# `dw_const CeruleanPokecenterNurseText, TEXT_CERULEANPOKECENTER_NURSE` -- and
# the object_event that runs that text carries the same TEXT_ constant, so a
# body classified as a service names an exact tile. Classifying by sprite would
# have worked here by luck (all thirteen healers are SPRITE_NURSE) and would
# have been wrong the moment a service was drawn as anything else.

#: A text body that is nothing but one of these macros is that service.
SERVICE_MACROS = {"script_pokecenter_nurse": "heal"}

#: A body that reaches one of these calls provides that service. Silph Co 9F's
#: nurse is hand-written rather than the macro -- she calls `predef HealParty`
#: inside a conditional -- and she heals exactly like the other twelve.
SERVICE_CALLS = {"predef HealParty": "heal"}

#: How a script hands an item over, and where the item's name is written. Both
#: forms are (the instruction that does it, the pattern that names the item on
#: the nearest line above it). This is what makes "somebody in this room hands
#: you something" answerable at all: the Warden's HM04, Mr Fuji's Poke Flute, the
#: Silph president's Master Ball and Oak's aide's HM05 each gate main-line
#: progress and none of them was in this file.
GIFT_FORMS = (
    # The general case: `lb bc, HM_STRENGTH, 1` sits directly above the
    # `call GiveItem` it feeds, at all forty-one of its sites.
    ("call GiveItem", re.compile(r"lb bc, (\w+), \d+$")),
    # Oak's three aides never call it. They load the reward into a hardware
    # register and hand off to a shared routine that lives in another file, so
    # following calls within the file cannot reach the give. HM05 Flash -- the
    # move Rock Tunnel needs -- is one of the three.
    ("ldh [hOaksAideRewardItem], a", re.compile(r"ld a, (\w+)$")),
)

# `call GivePokemon` is deliberately NOT in that table. Three of its six sites --
# the Fighting Dojo's two prize balls and the Cinnabar Lab fossil revival -- name
# no species at all: they load `wCurPartySpecies`, decided at run time by which
# ball the player walked up to or which fossil is in the bag. Emitting the three
# that do use the literal form (Eevee, Lapras, the Magikarp salesman's Magikarp)
# and silently dropping the other three would be a table that is right where it
# is easy and absent where it is not, which is the failure mode this file exists
# to avoid. A gift Pokemon is also not an item, so it has nowhere to go in the
# schema below without a second one.

#: A text body is not the whole script it runs. Erika's text reaches TM21 through
#: `call z, CeladonGymReceiveTM21` and Mom's heal sits behind
#: `call RedsHouse1FMomHealScript`; both targets are labels at column 0, which the
#: body split below treats as the *end* of the calling body. Reading only a body's
#: own lines therefore missed all eight gym leaders' TMs, the Celadon roof girl's
#: drink trade and Mom's free heal. Calls are followed into other bodies of the
#: same file -- one source file is as far as a label can be resolved here, and
#: everything a service does is written next to it.
LOCAL_CALL = re.compile(r"^(?:call|jp|jr)\s+(?:[a-z]{1,2},\s*)?([A-Z]\w*)$")

#: One hop finds every service in the game -- measured: depth 1 and depth 6 both
#: yield the same fifty-seven entries, and depth 0 yields forty-five. The bound is
#: a guard rather than a threshold, so a future script that chains further is
#: still reached without any depth walking a whole map script in.
MAX_CALL_DEPTH = 3

#: What the whole game holds, checked rather than assumed: eleven Poke Center
#: nurses, the Indigo Plateau lobby nurse, the Silph Co 9F nurse and Mom in Red's
#: House; forty-one people who hand an item over, forty-three items between them
#: (the Celadon roof girl gives one of three). If a regeneration finds a different
#: number the classifier has drifted and the agent would be pointed at the wrong
#: tile, so it fails instead.
EXPECTED_SERVICES = {"gift": 43, "heal": 14}

# Three things this file finds and deliberately does not emit. Every entry here
# is a person you walk to and press A at, and none of these is one:
#
# * Pokemon Tower 5F's purified zone. `PokemonTower5FDefaultScript` runs
#   `predef HealParty` when the player stands on one of the four tiles in
#   `PokemonTower5FPurifiedZoneCoords`. It is a floor trigger -- no object_event,
#   no TEXT_ constant, nobody to talk to -- so an entry for it would point
#   `poke heal` at a tile where an A press does nothing, which is worse than no
#   entry. It is a real free heal and it is the one thing here worth revisiting
#   if the schema ever grows a "stand here" kind.
# * Viridian Mart's Oak's Parcel. `ViridianMartOaksParcelScript` hangs off the
#   map's SCRIPT_ pointer table and fires on entry, not off any text. The parcel
#   arrives during a forced cutscene; there is nothing to walk to.
# * Oak's Lab's `predef HealParty` is the automatic heal after the rival battle,
#   inside a cutscene body no text pointer reaches. Correctly not a service.
#
# The Game Corner's three prize vendors (`script_prize_vendor`) and the Celadon
# roof's three vending machines (`script_vending_machine`) are the same one-macro
# shape as a nurse and would classify in one line each -- but every one of the six
# is a `bg_event`, a piece of scenery, not an `object_event`. parse_objects only
# reads object_events, and the "counter is a talk-over tile, stand two out" rule
# every reader of this file applies is wrong for scenery you face directly. They
# need a bg_event join and a stand-tile rule of their own.
#
# Still unclassified for want of a reader each, none of them on the main line:
# the eight `predef DoInGameTradeDialogue` in-game trades, the Name Rater, the
# Day Care and the Safari Zone gate.


def reachable_body(bodies: Dict[str, List[str]], start: str) -> List[str]:
    """Every instruction a text body can run, following calls within the file.

    See LOCAL_CALL: the body split closes a body at the next column-0 label, so a
    script that hands its work to a helper label looks empty from the body alone.
    """
    seen: set = set()
    queue: List[Tuple[str, int]] = [(start, 0)]
    flat: List[str] = []
    while queue:
        label, depth = queue.pop(0)
        if label in seen or label not in bodies or depth > MAX_CALL_DEPTH:
            continue
        seen.add(label)
        for entry in bodies[label]:
            if not entry:
                continue
            flat.append(entry)
            call = LOCAL_CALL.match(entry)
            if call and call.group(1) in bodies:
                queue.append((call.group(1), depth + 1))
    return flat


def parse_map_scripts(fetcher: Fetcher) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    """Per map label, the services each of its text bodies provides.

    ``{"CeruleanPokecenter": {"TEXT_CERULEANPOKECENTER_NURSE": [{"service": "heal"}]}}``.
    Bodies that are plain dialogue -- ``text_far`` -- and bodies whose asm does
    nothing we have a name for are left out entirely.

    A list per constant because one person can hand over more than one thing: the
    Celadon Mart roof girl gives TM49, TM48 or TM13 depending on which drink she
    is offered, and naming only the first would be a wrong answer about the other
    two.
    """
    names = [n for n in fetcher.listdir("scripts") if n.endswith(".asm")]
    texts = fetcher.many([f"scripts/{n}" for n in names])
    pointer = re.compile(r"dw_const\s+(\w+),\s*(TEXT_\w+)")
    out: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
    for path, text in texts.items():
        label = Path(path).stem
        lines = joined_lines(text)
        # Pairs, not a dict: one body can back several text ids. The Game Corner's
        # three prize vendors and the Celadon roof's three vending machines each
        # share one body, and keying by label would keep only the last of them.
        consts = [(m.group(1), m.group(2)) for line in lines for m in [pointer.search(line)] if m]
        if not consts:
            continue
        # Bodies, in source order: a label at column 0 opens one and the next
        # such label closes it. Local labels (`.beat_giovanni`) are body, not a
        # new body, which is what keeps Silph Co 9F's branches together.
        bodies: Dict[str, List[str]] = {}
        current: Optional[str] = None
        for line in lines:
            if not line.startswith((" ", "\t")) and line.rstrip(":") != line:
                current = line.rstrip(":")
                bodies[current] = []
            elif current is not None:
                bodies[current].append(line.strip())
        found: Dict[str, List[Dict[str, str]]] = {}
        for body_label, const in consts:
            body = reachable_body(bodies, body_label)
            if not body:
                continue
            service = SERVICE_MACROS.get(body[0].split()[0])
            if service is None:
                joined = "\n".join(body)
                service = next(
                    (name for call, name in SERVICE_CALLS.items() if call in joined), None
                )
            if service:
                found[const] = [{"service": service}]
                continue
            # One entry per item the body can hand over, in source order: the
            # Celadon roof girl gives TM49, TM48 or TM13 depending on the drink.
            items = gift_items(body)
            if items:
                found[const] = [{"service": "gift", "item": item} for item in items]
        if found:
            out[label] = found
    return out


def gift_items(body: List[str]) -> List[str]:
    """The item constants a body can hand over, in source order. See GIFT_FORMS.

    Empty when the body gives nothing, which is almost every body in the game.
    """
    items: List[str] = []
    for index, entry in enumerate(body):
        for marker, names_it in GIFT_FORMS:
            if entry != marker:
                continue
            const = next(
                (
                    match.group(1)
                    for above in reversed(body[:index])
                    for match in [names_it.match(above)]
                    if match
                ),
                None,
            )
            if const is None:
                raise GenError(f"{marker!r} with nothing above it naming the item: {body}")
            items.append(const)
    return items


def build_services(
    fetcher: Fetcher,
    maps: MapNames,
    objects: Dict[str, Dict[str, Any]],
    names: NameTables,
    report: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Every map's service NPCs, with the tile each one stands on.

    The tile is where the *person* is, not where you stand to talk to them: a
    Center's counter is a talk-over tile, so the nurse at (3,1) is reached from
    (3,3). Which tiles those are depends on live collision, so the server works
    them out the same way it does for a mart clerk rather than freezing a guess
    here.
    """
    scripts = parse_map_scripts(fetcher)
    out: Dict[str, List[Dict[str, Any]]] = {}
    counts: Dict[str, int] = {}
    for label, by_const in sorted(scripts.items()):
        if label not in maps.label_to_const:
            continue  # a script for a map with no header of its own
        entries = []
        for const, services in sorted(by_const.items()):
            at = [
                [obj["x"], obj["y"]]
                for obj in objects.get(label, {}).get("objects", ())
                if str(obj.get("text") or "") == const
            ]
            if len(at) != 1:
                raise GenError(
                    f"{const} on {maps.label(label)}: matched {len(at)} objects, expected 1"
                )
            for service in services:
                entry = {"service": service["service"], "text": const, "at": at[0]}
                if "item" in service:
                    entry["item"] = names.item(service["item"])
                entries.append(entry)
                counts[entry["service"]] = counts.get(entry["service"], 0) + 1
        out[maps.label(label)] = entries
    if counts != EXPECTED_SERVICES:
        raise GenError(f"services found {counts}, expected {EXPECTED_SERVICES}")
    report.append(f"services: {', '.join(f'{n} {kind}' for kind, n in sorted(counts.items()))}")
    return out


# ===================================================================
# species.json
# ===================================================================

EVO_METHODS = {"EVOLVE_LEVEL": "level", "EVOLVE_ITEM": "item", "EVOLVE_TRADE": "trade"}


def parse_evos_moves(fetcher: Fetcher, names: NameTables) -> Dict[str, Dict[str, Any]]:
    text = fetcher.text("data/pokemon/evos_moves.asm")
    lines = joined_lines(text)
    order: List[str] = []
    in_table = False
    blocks: Dict[str, List[List[str]]] = {}
    current: Optional[str] = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("EvosMovesPointerTable"):
            in_table = True
            continue
        head, args = directive(stripped)
        if in_table:
            if head == "dw":
                order.append(args[0])
            elif head.startswith("assert_table_length"):
                in_table = False
            continue
        if stripped.endswith(":"):
            current = stripped.rstrip(":")
            blocks.setdefault(current, [])
            continue
        if head == "db" and current is not None:
            blocks[current].append(args)

    out: Dict[str, Dict[str, Any]] = {}
    for position, label in enumerate(order, start=1):
        const = names.species_by_internal.get(position)
        if const is None:  # MissingNo slot
            continue
        if const.replace("_", "").lower() != label.removesuffix("EvosMoves").lower():
            raise GenError(f"evos_moves entry {position} is {label!r} but species is {const!r}")
        evolutions: List[Dict[str, Any]] = []
        learnset: List[Dict[str, Any]] = []
        done_evos = False
        for args in blocks[label]:
            if args == ["0"]:
                done_evos = True
                continue
            if not done_evos:
                method = EVO_METHODS.get(args[0])
                if method is None:
                    raise GenError(f"{label}: unknown evolution method {args[0]!r}")
                if method == "level":
                    param: Any = numeric(args[1])
                    target = args[2]
                elif method == "item":
                    param = names.item(args[1])
                    target = args[3]
                else:  # trade: min level, species
                    param = numeric(args[1])
                    target = args[2]
                evolutions.append({"to": names.species(target), "method": method, "param": param})
            else:
                learnset.append({"level": numeric(args[0]), "move": names.move(args[1])})
        out[const] = {"evolutions": evolutions, "learnset": learnset}
    return out


def build_species(fetcher: Fetcher, names: NameTables) -> Dict[str, Any]:
    files = fetcher.listdir("data/pokemon/base_stats")
    texts = fetcher.many([f"data/pokemon/base_stats/{n}" for n in files])
    evos = parse_evos_moves(fetcher, names)

    out: Dict[str, Any] = {}
    for path, text in sorted(texts.items()):
        rows: List[List[str]] = []
        tmhm: List[str] = []
        for line in joined_lines(text):
            head, args = directive(line)
            if head == "db":
                rows.append(args)
            elif head == "tmhm":
                tmhm = args
        if len(rows) < 7:
            raise GenError(f"{path}: only {len(rows)} db rows, layout changed upstream")
        dex_const, stats, types, catch, base_exp, starting_moves, growth = rows[:7]
        const = dex_const[0].removeprefix("DEX_")
        dex = names.dex(const)
        type_names = [names.type_(t) for t in types]
        if len(type_names) == 2 and type_names[0] == type_names[1]:
            type_names = type_names[:1]  # Gen 1 stores a mono-type twice

        learnset = [{"level": 1, "move": names.move(m)} for m in starting_moves if m != "NO_MOVE"]
        learnset += evos[const]["learnset"]

        out[names.species(const)] = {
            "dex": dex,
            "types": type_names,
            "base": {
                "hp": numeric(stats[0]),
                "atk": numeric(stats[1]),
                "def": numeric(stats[2]),
                "spd": numeric(stats[3]),
                "spc": numeric(stats[4]),
            },
            "catch_rate": numeric(catch[0]),
            "base_exp": numeric(base_exp[0]),
            "growth": growth[0].removeprefix("GROWTH_").lower(),
            "learnset": learnset,
            "tm_hm": sorted(tm_hm_labels(tmhm, names)),
            "evolutions": evos[const]["evolutions"],
        }
    if len(out) != 151:
        raise GenError(f"{len(out)} species parsed, expected 151")
    return out


def tm_hm_labels(moves: Iterable[str], names: NameTables) -> List[str]:
    labels = []
    for move in moves:
        if move == "UNUSED":
            continue  # the 56th bit of the tm/hm bitfield is padding
        label = names.tm_hm_of_move.get(move)
        if label is None:
            raise GenError(f"{move!r} is in a tmhm list but is not a TM or HM")
        labels.append(label)
    return labels


# ===================================================================
# moves.json
# ===================================================================


def build_moves(fetcher: Fetcher, names: NameTables) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    move_id = 0
    for line in joined_lines(fetcher.text("data/moves/moves.asm")):
        head, args = directive(line)
        if head != "move" or len(args) != 6:
            continue
        move_id += 1
        const, effect, power, type_const, accuracy, pp = args
        # Gen 1 stores accuracy as a byte out of 255; pokered writes the
        # percentage and multiplies by $ff/100 at assembly time, so the source
        # number IS the percentage and the ROM byte is pct * 255 // 100.
        percent = numeric(accuracy)
        out[names.move(const)] = {
            "id": move_id,
            "type": names.type_(type_const),
            "power": numeric(power),
            "accuracy": percent,
            "accuracy_byte": percent * 255 // 100,
            "pp": numeric(pp),
            "effect": effect,
        }
    if not out:
        raise GenError("no moves parsed")
    return out


# ===================================================================
# tms.json
# ===================================================================


def build_tms(names: NameTables) -> Dict[str, str]:
    """``{"TM28": "Dig", ..., "HM01": "Cut"}`` -- which move each TM teaches.

    ``NameTables`` already builds this while numbering ``add_tm``/``add_hm`` in
    item_constants.asm; it was thrown away after being used to label each
    species' ``tm_hm`` list. Without it the bag and the moveset cannot be joined
    at all: "TM28 x1" and "Charmeleon can learn TM28" are both in the payload
    and neither of them says Dig.
    """
    out: Dict[str, str] = {}
    for const, label in names.tm_hm_of_move.items():
        if label in out:
            raise GenError(f"{label} teaches two moves: {out[label]!r} and {const!r}")
        out[label] = names.move(const)
    tms = [label for label in out if label.startswith("TM")]
    hms = [label for label in out if label.startswith("HM")]
    if len(tms) != 50 or len(hms) != 5:
        raise GenError(f"{len(tms)} TMs and {len(hms)} HMs parsed, expected 50 and 5")
    return dict(sorted(out.items()))


# ===================================================================
# types.json
# ===================================================================


def build_types(fetcher: Fetcher, names: NameTables, report: List[str]) -> Dict[str, Any]:
    # Cross-check red.py's TYPE_NAMES against data/types/names.asm ordering.
    ordered = [
        (const, value)
        for const, value in sorted(names.type_value_of_const.items(), key=lambda kv: kv[1])
    ]
    disagreements = []
    for const, value in ordered:
        expected = TYPE_NAMES.get(value)
        pretty = const.removesuffix("_TYPE").title()
        if expected is None or expected.lower() != pretty.lower():
            disagreements.append(
                f"{const}=${value:02X}: red.py says {expected!r}, pokered {pretty!r}"
            )
    if disagreements:
        report.append("TYPE_NAMES disagreement: " + "; ".join(disagreements))
    else:
        report.append("TYPE_NAMES agrees with pokered's type constants (Bird at $06 included)")

    chart: Dict[str, Dict[str, float]] = {}
    for line in joined_lines(fetcher.text("data/types/type_matchups.asm")):
        head, args = directive(line)
        if head != "db" or len(args) != 3:
            continue
        if args[0] == "-1":
            continue
        attacker, defender, effect = args
        if effect not in EFFECT_MULTIPLIERS:
            raise GenError(f"unknown type effect {effect!r}")
        chart.setdefault(names.type_(attacker), {})[names.type_(defender)] = EFFECT_MULTIPLIERS[
            effect
        ]
    return {
        "types": [TYPE_NAMES[value] for _, value in ordered],
        # Gen 1 is physical for type ids below $14 and special at or above it.
        "physical_types": [TYPE_NAMES[v] for _, v in ordered if v < 0x14],
        "special_types": [TYPE_NAMES[v] for _, v in ordered if v >= 0x14],
        "chart": chart,
    }


# ===================================================================
# main
# ===================================================================


def write_json(path: Path, payload: Dict[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path.stat().st_size


def with_sha(sha: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Every file records its upstream commit under "generated_from"."""
    return {"generated_from": sha, **payload}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", help="pokered commit to generate from (default: master)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args(argv)

    sha = args.sha or resolve_master_sha()
    fetcher = Fetcher(sha, args.cache_dir)
    report: List[str] = []

    maps = MapNames(fetcher)
    names = NameTables(fetcher)
    report.extend(names.warnings)

    headers = parse_headers(fetcher)
    objects = parse_objects(fetcher, maps)

    world = build_world(headers, objects, maps, report)
    trainers = build_trainers(
        objects, maps, parse_trainer_parties(fetcher, names), parse_trainer_classes(fetcher), report
    )
    encounters = build_encounters(fetcher, names, parse_slot_chances(fetcher))
    items = build_items(fetcher, objects, maps, names)
    shops = build_shops(fetcher, maps, names, objects, report)
    services = build_services(fetcher, maps, objects, names, report)
    species = build_species(fetcher, names)
    moves = build_moves(fetcher, names)
    tms = build_tms(names)
    types = build_types(fetcher, names, report)

    outputs = [
        ("world.json", with_sha(sha, {"maps": world}), f"{len(world)} maps"),
        ("trainers.json", with_sha(sha, trainers), f"{len(trainers)} maps with trainers"),
        ("encounters.json", with_sha(sha, encounters), f"{len(encounters)} maps"),
        ("items.json", with_sha(sha, items), f"{len(items)} maps with items"),
        ("shops.json", with_sha(sha, shops), f"{len(shops)} marts"),
        ("services.json", with_sha(sha, services), f"{len(services)} maps with a service"),
        ("species.json", with_sha(sha, species), f"{len(species)} species"),
        ("moves.json", with_sha(sha, moves), f"{len(moves)} moves"),
        ("tms.json", with_sha(sha, tms), f"{len(tms)} TMs and HMs"),
        ("types.json", with_sha(sha, types), f"{len(types['types'])} types"),
    ]

    print(f"pokered @ {sha}")
    print(f"cache {fetcher.cache_dir}  ({fetcher.downloads} downloaded, {fetcher.hits} cached)")
    print()
    for filename, payload, summary in outputs:
        size = write_json(args.out / filename, payload)
        print(f"  {filename:<16} {summary:<28} {size / 1024:7.1f} KiB")

    warps = sum(len(m["warps"]) for m in world.values())
    trainer_count = sum(len(v) for v in trainers.values())
    item_count = sum(len(v) for v in items.values())
    print()
    print(f"  {warps} warps, {trainer_count} placed trainers, {item_count} items on the ground")
    for note in report:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
