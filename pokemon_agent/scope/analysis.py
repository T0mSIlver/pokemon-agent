"""The measurements. Every function here returns numbers; none of them print.

Each analysis answers exactly one of the questions a supervisor asks about a
run in progress, and each is built to be summarised into a screenful. Where a
choice had to be made between a precise figure that needs a paragraph to
interpret and a coarse one that does not, the coarse one won.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from pokemon_agent.bench.registry import Receipt, RunRecord
from pokemon_agent.scope.runs import ContextOracle
from pokemon_agent.scope.transcript import (
    ADVISORY_VERBS,
    Call,
    ImageRef,
    Session,
    median,
    poke_verbs,
)

#: Longest repeated sequence ``loops`` will look for. Anything longer is either
#: the whole session or a coincidence.
MAX_NGRAM = 8

#: Context window the model is being run in, for the occupancy percentage.
DEFAULT_WINDOW = 140_000

#: Pixels per token. The rule of thumb published for image inputs is
#: ``width * height / 750``; it is an estimate and is labelled as one wherever
#: it is printed. The *measured* figure next to it does not depend on it.
PIXELS_PER_TOKEN = 750

#: When an image's dimensions cannot be read, fall back to its encoded size. A
#: Game Boy frame PNG runs a few hundred bytes per token at these dimensions.
BYTES_PER_TOKEN = 750


# -- tools --------------------------------------------------------------------


@dataclass(frozen=True)
class ToolStat:
    """One verb or program, and what it did across the session."""

    label: str
    kind: str
    calls: int
    failures: int
    median_result_bytes: int
    total_result_bytes: int
    #: Median tiles moved, for the verbs that move the player. ``None`` for the
    #: advisory verbs, which move nothing and are judged on being called at all.
    median_moved: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind,
            "calls": self.calls,
            "failures": self.failures,
            "median_result_bytes": self.median_result_bytes,
            "total_result_bytes": self.total_result_bytes,
            "median_moved": self.median_moved,
        }


@dataclass(frozen=True)
class ToolReport:
    session_id: str
    steps: int
    total_calls: int
    stats: tuple[ToolStat, ...]
    advisory: tuple[tuple[str, int, int], ...]
    bash_calls: int
    bash_poke_calls: int
    bash_other_calls: int
    other_programs: tuple[tuple[str, int], ...]
    result_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "steps": self.steps,
            "total_calls": self.total_calls,
            "tools": [stat.to_dict() for stat in self.stats],
            "advisory": [
                {"verb": verb, "calls": calls, "failures": failures}
                for verb, calls, failures in self.advisory
            ],
            "bash_calls": self.bash_calls,
            "bash_poke_calls": self.bash_poke_calls,
            "bash_other_calls": self.bash_other_calls,
            "other_programs": [
                {"program": name, "calls": calls} for name, calls in self.other_programs
            ],
            "result_bytes": self.result_bytes,
        }


def _moved_of(call: Call) -> Optional[int]:
    payload = call.result_json
    if payload is None:
        return None
    value = payload.get("moved")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return None


def tool_report(session: Session) -> ToolReport:
    """Every verb the model reached for, how often, and how well it went.

    A bash line can invoke several ``poke`` verbs at once, so a verb is counted
    once per invocation rather than once per bash call — which is why the verb
    counts and the bash call count do not have to add up.
    """

    grouped: dict[tuple[str, str], list[Call]] = {}
    bash_calls = bash_poke = bash_other = 0
    other_programs: dict[str, int] = {}
    result_bytes = 0

    for call in session.calls:
        result_bytes += call.result_bytes
        if call.tool == "bash":
            bash_calls += 1
            verbs = poke_verbs(call.command)
            if verbs:
                bash_poke += 1
                for verb in verbs:
                    grouped.setdefault((f"poke {verb}", "poke"), []).append(call)
            else:
                bash_other += 1
                other_programs[call.label] = other_programs.get(call.label, 0) + 1
                grouped.setdefault((call.label, "bash"), []).append(call)
        else:
            grouped.setdefault((call.label, "tool"), []).append(call)

    stats: list[ToolStat] = []
    for (label, kind), calls in grouped.items():
        moved = [value for value in (_moved_of(call) for call in calls) if value is not None]
        sizes = [call.result_bytes for call in calls]
        stats.append(
            ToolStat(
                label=label,
                kind=kind,
                calls=len(calls),
                failures=sum(1 for call in calls if call.is_error),
                median_result_bytes=int(median(sizes) or 0),
                total_result_bytes=sum(sizes),
                median_moved=median(moved) if moved else None,
            )
        )
    stats.sort(key=lambda stat: (-stat.calls, stat.label))

    advisory: list[tuple[str, int, int]] = []
    for verb in ADVISORY_VERBS:
        calls = grouped.get((f"poke {verb}", "poke"), [])
        advisory.append((verb, len(calls), sum(1 for call in calls if call.is_error)))

    return ToolReport(
        session_id=session.short_id,
        steps=len(session.steps),
        total_calls=len(session.calls),
        stats=tuple(stats),
        advisory=tuple(advisory),
        bash_calls=bash_calls,
        bash_poke_calls=bash_poke,
        bash_other_calls=bash_other,
        other_programs=tuple(sorted(other_programs.items(), key=lambda item: (-item[1], item[0]))),
        result_bytes=result_bytes,
    )


# -- loops --------------------------------------------------------------------


@dataclass(frozen=True)
class Loop:
    """A sequence of commands the model ran more than once."""

    tokens: tuple[str, ...]
    count: int
    first_step: int
    last_step: int
    where: str = ""

    @property
    def length(self) -> int:
        return len(self.tokens)

    @property
    def covered(self) -> int:
        """Calls spent inside this loop — the size of the problem."""

        return self.length * self.count

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": list(self.tokens),
            "count": self.count,
            "length": self.length,
            "calls_covered": self.covered,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "where": self.where,
        }


def _canonical_rotation(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """The lexically smallest rotation, so ``up,down`` and ``down,up`` are one loop."""

    return min(tuple(tokens[shift:] + tokens[:shift]) for shift in range(len(tokens)))


def _is_primitive(tokens: tuple[str, ...]) -> bool:
    """False when the sequence is just a shorter cycle written out twice.

    ``a b a b`` and ``a b`` cover the same calls and mean the same thing; the
    short one is the loop and the long one is an artefact of the window size.
    """

    length = len(tokens)
    for period in range(1, length // 2 + 1):
        if length % period == 0 and all(
            tokens[index] == tokens[index % period] for index in range(length)
        ):
            return False
    return True


def _non_overlapping(positions: Sequence[int], length: int) -> list[int]:
    kept: list[int] = []
    reach = -1
    for position in positions:
        if position > reach:
            kept.append(position)
            reach = position + length - 1
    return kept


def _is_cyclic_subsequence(short: tuple[str, ...], cycle: tuple[str, ...]) -> bool:
    """Whether ``short`` is a run of ``cycle`` repeating, wrap-around included.

    Without the wrap, the ``left, up`` half of an ``up, down, left`` cycle looks
    like a loop of its own and gets its own row saying the same thing.
    """

    span = len(short)
    if span > len(cycle):
        return False
    doubled = cycle + cycle[: span - 1]
    return any(doubled[start : start + span] == short for start in range(len(cycle)))


#: A single command has to repeat at least this many times back to back before
#: it counts as a loop rather than as a command the model happens to use a lot.
MIN_SOLO_RUN = 3


def _longest_run(tokens: Sequence[str], token: str) -> tuple[int, int]:
    """``(length, start)`` of the longest back-to-back run of one token."""

    best = (0, 0)
    index = 0
    while index < len(tokens):
        if tokens[index] != token:
            index += 1
            continue
        start = index
        while index < len(tokens) and tokens[index] == token:
            index += 1
        if index - start > best[0]:
            best = (index - start, start)
    return best


def find_loops(
    session: Session,
    *,
    limit: int = 8,
    min_count: int = 2,
    max_ngram: int = MAX_NGRAM,
) -> list[Loop]:
    """Repeated command sequences, longest and most frequent first.

    Occurrences are counted without overlap, so ``a a a a`` is two runs of
    ``a a`` and not three. Sub-sequences of a longer loop that is at least as
    frequent are dropped, because reporting ``act up`` separately from
    ``act up / act down`` says the same thing twice. Only primitive cycles are
    considered, so a loop is reported at its true period rather than at some
    multiple of it.
    """

    tokens = [call.signature or call.label for call in session.calls]
    where = _step_locations(session)
    candidates: list[Loop] = []

    # A single command hammered over and over is the commonest way to be stuck,
    # but a command merely used often is not a loop — so for one-token cycles
    # only an unbroken run counts.
    for token in set(tokens):
        run, start = _longest_run(tokens, token)
        if run >= MIN_SOLO_RUN:
            candidates.append(
                Loop(
                    tokens=(token,),
                    count=run,
                    first_step=session.calls[start].step,
                    last_step=session.calls[start + run - 1].step,
                    where=where.get(session.calls[start].step, ""),
                )
            )

    for length in range(2, min(max_ngram, len(tokens) // 2) + 1):
        positions: dict[tuple[str, ...], list[int]] = {}
        for start in range(len(tokens) - length + 1):
            positions.setdefault(tuple(tokens[start : start + length]), []).append(start)
        for gram, found in positions.items():
            if len(found) < min_count or not _is_primitive(gram):
                continue
            kept = _non_overlapping(found, length)
            if len(kept) < min_count:
                continue
            first_call, last_call = kept[0], kept[-1] + length - 1
            candidates.append(
                Loop(
                    tokens=gram,
                    count=len(kept),
                    first_step=session.calls[first_call].step,
                    last_step=session.calls[last_call].step,
                    where=where.get(session.calls[first_call].step, ""),
                )
            )

    # One entry per cycle: keep the best-scoring rotation of each.
    by_cycle: dict[tuple[str, ...], Loop] = {}
    for loop in candidates:
        key = _canonical_rotation(loop.tokens)
        best = by_cycle.get(key)
        if best is None or (loop.covered, loop.length) > (best.covered, best.length):
            by_cycle[key] = loop

    ranked = sorted(by_cycle.values(), key=lambda loop: (-loop.covered, -loop.length))
    maximal: list[Loop] = []
    for loop in ranked:
        if any(_redundant(loop, other) for other in maximal):
            continue
        maximal.append(loop)
        if len(maximal) >= limit:
            break
    return maximal


def _redundant(loop: Loop, accepted: Loop) -> bool:
    """Whether ``loop`` is the same episode as one already reported.

    Two ways it can be. It is literally a stretch of the accepted cycle — then
    it adds a row and no information. Or it uses no command the accepted loop
    does not, over the same stretch of the session — the same oscillation seen
    through a wider window, which otherwise fills the table with eight rows of
    ``act up`` and ``act left`` in eight arrangements.
    """

    if (
        accepted.length > loop.length
        and accepted.count >= loop.count
        and _is_cyclic_subsequence(loop.tokens, accepted.tokens)
    ):
        return True
    overlaps = loop.first_step <= accepted.last_step and accepted.first_step <= loop.last_step
    return overlaps and set(loop.tokens) <= set(accepted.tokens)


def tail_repeat(session: Session, *, max_ngram: int = MAX_NGRAM) -> Optional[Loop]:
    """The cycle the session is stuck in *right now*, if it is stuck in one.

    Looks only at the end of the transcript, so it stays meaningful on a file
    that is still being appended to: this is the alarm, not the post-mortem.
    """

    tokens = [call.signature or call.label for call in session.calls]
    best: Optional[Loop] = None
    for length in range(1, min(max_ngram, len(tokens) // 2) + 1):
        gram = tuple(tokens[-length:])
        repeats = 1
        cursor = len(tokens) - 2 * length
        while cursor >= 0 and tuple(tokens[cursor : cursor + length]) == gram:
            repeats += 1
            cursor -= length
        if repeats >= 2:
            candidate = Loop(
                tokens=gram,
                count=repeats,
                first_step=session.calls[cursor + length].step,
                last_step=session.calls[-1].step,
            )
            if best is None or candidate.covered > best.covered:
                best = candidate
    return best


def _step_locations(session: Session) -> dict[int, str]:
    """Step index to the map the game was on, wherever a result reveals it."""

    locations: dict[int, str] = {}
    current = ""
    for call in session.calls:
        payload = call.result_json
        if payload is not None and isinstance(payload.get("map"), str):
            current = payload["map"]
        if current:
            locations[call.step] = current
    return locations


# -- waste --------------------------------------------------------------------

#: Order matters: the first bucket a batch qualifies for is the one it lands in.
WASTE_BUCKETS: tuple[str, ...] = ("productive", "battle", "dialog", "blocked", "revisit")


@dataclass
class WasteSplit:
    """Presses and batches by bucket, for one map or for the whole run."""

    name: str = ""
    presses: dict[str, int] = field(default_factory=lambda: dict.fromkeys(WASTE_BUCKETS, 0))
    batches: dict[str, int] = field(default_factory=lambda: dict.fromkeys(WASTE_BUCKETS, 0))

    @property
    def total_presses(self) -> int:
        return sum(self.presses.values())

    @property
    def total_batches(self) -> int:
        return sum(self.batches.values())

    def share(self, bucket: str) -> float:
        total = self.total_presses
        return (self.presses.get(bucket, 0) / total) if total else 0.0

    def add(self, bucket: str, presses: int) -> None:
        self.presses[bucket] = self.presses.get(bucket, 0) + presses
        self.batches[bucket] = self.batches.get(bucket, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "presses": dict(self.presses),
            "batches": dict(self.batches),
            "total_presses": self.total_presses,
            "shares": {bucket: round(self.share(bucket), 4) for bucket in WASTE_BUCKETS},
        }


@dataclass(frozen=True)
class WasteReport:
    run_id: str
    overall: WasteSplit
    by_map: tuple[WasteSplit, ...]
    unclassified_presses: int
    milestones: int
    context_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "overall": self.overall.to_dict(),
            "by_map": [split.to_dict() for split in self.by_map],
            "unclassified_presses": self.unclassified_presses,
            "milestones": self.milestones,
            "context_samples": self.context_samples,
        }


def classify_receipt(
    receipt: Receipt, seen: set[tuple[str, int, int]], oracle: Optional[ContextOracle]
) -> str:
    """Which bucket a batch of presses belongs in.

    The whole batch goes in one bucket. A batch is the unit the model chose and
    the unit the receipt records, and splitting it would mean inventing a
    per-press story the data does not contain.
    """

    if receipt.milestones_new:
        return "productive"
    context = oracle.at(receipt.t) if oracle is not None else None
    if context is not None and context.battle:
        return "battle"
    if context is not None and context.dialog:
        return "dialog"
    if receipt.pos is not None:
        tile = (receipt.map_name or "?", receipt.pos[0], receipt.pos[1])
        if tile not in seen:
            return "productive"
    if receipt.moved == 0:
        return "blocked"
    return "revisit"


def waste_report(record: RunRecord, oracle: Optional[ContextOracle] = None) -> WasteReport:
    """Where the presses went, overall and per map."""

    seen: set[tuple[str, int, int]] = set()
    overall = WasteSplit(name="all")
    by_map: dict[str, WasteSplit] = {}
    unclassified = 0
    milestones = 0

    for receipt in record.receipts:
        if receipt.presses <= 0:
            if receipt.pos is not None:
                seen.add((receipt.map_name or "?", receipt.pos[0], receipt.pos[1]))
            continue
        bucket = classify_receipt(receipt, seen, oracle)
        milestones += len(receipt.milestones_new)
        if receipt.pos is None:
            unclassified += receipt.presses
        overall.add(bucket, receipt.presses)
        map_name = receipt.map_name or "?"
        by_map.setdefault(map_name, WasteSplit(name=map_name)).add(bucket, receipt.presses)
        if receipt.pos is not None:
            seen.add((map_name, receipt.pos[0], receipt.pos[1]))

    ordered = tuple(sorted(by_map.values(), key=lambda split: (-split.total_presses, split.name)))
    return WasteReport(
        run_id=record.run_id,
        overall=overall,
        by_map=ordered,
        unclassified_presses=unclassified,
        milestones=milestones,
        context_samples=len(oracle) if oracle is not None else 0,
    )


# -- context ------------------------------------------------------------------


def image_tokens(image: ImageRef) -> int:
    """Estimated prompt cost of one image. See :data:`PIXELS_PER_TOKEN`."""

    pixels = image.pixels
    if pixels:
        return max(1, math.ceil(pixels / PIXELS_PER_TOKEN))
    return max(1, math.ceil(image.nbytes / BYTES_PER_TOKEN))


@dataclass(frozen=True)
class ContextReport:
    session_id: str
    model: str
    steps: int
    window: int
    peak_prompt: int
    peak_step: int
    final_prompt: int
    first_prompt: int
    median_growth: float
    mean_growth: float
    output_tokens: int
    #: Estimated image tokens resident at the peak, and the text that is not.
    image_tokens_at_peak: int
    text_tokens_at_peak: int
    image_count: int
    image_bytes: int
    image_tokens_total: int
    tool_result_images: int
    prompt_images: int
    common_size: str
    est_tokens_per_image: int
    measured_with_image: Optional[float]
    measured_without_image: Optional[float]
    curve: tuple[int, ...]

    @property
    def peak_share(self) -> float:
        return (self.peak_prompt / self.window) if self.window else 0.0

    @property
    def image_share(self) -> float:
        return (self.image_tokens_at_peak / self.peak_prompt) if self.peak_prompt else 0.0

    @property
    def verdict(self) -> str:
        """The one-line answer to "are the frames eating the window?".

        Two independent readings have to agree before it says yes: the
        estimated resident image tokens, and the observed prompt growth after a
        step that read a frame compared with one that did not.
        """

        if not self.image_count:
            return "no images in this session at all"
        share = self.image_share
        grew = (
            self.measured_with_image is not None
            and self.measured_without_image is not None
            and self.measured_with_image > self.measured_without_image
        )
        if share >= 0.20 and grew:
            return "images are a leading cost - cutting a frame per turn would show"
        if share >= 0.20 or grew:
            return "images are a real but secondary cost - text is the bigger half"
        return "images are not what is filling the window; tool text is"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "model": self.model,
            "steps": self.steps,
            "window": self.window,
            "peak_prompt_tokens": self.peak_prompt,
            "peak_step": self.peak_step,
            "peak_share": round(self.peak_share, 4),
            "final_prompt_tokens": self.final_prompt,
            "first_prompt_tokens": self.first_prompt,
            "median_growth_per_step": self.median_growth,
            "mean_growth_per_step": self.mean_growth,
            "output_tokens": self.output_tokens,
            "image_tokens_at_peak_est": self.image_tokens_at_peak,
            "text_tokens_at_peak_est": self.text_tokens_at_peak,
            "verdict": self.verdict,
            "images": {
                "count": self.image_count,
                "bytes": self.image_bytes,
                "est_tokens_total": self.image_tokens_total,
                "from_tool_results": self.tool_result_images,
                "from_prompts": self.prompt_images,
                "common_size": self.common_size,
                "est_tokens_each": self.est_tokens_per_image,
                "measured_growth_after_image": self.measured_with_image,
                "measured_growth_without_image": self.measured_without_image,
            },
            "curve": list(self.curve),
        }


def context_report(session: Session, *, window: int = DEFAULT_WINDOW) -> ContextReport:
    """What is filling the context window, and how fast.

    Totals are the provider's own numbers — ``input + cacheRead`` is what the
    model had to read at that step — so the headline is measured, not guessed.
    Only the text/image *split* is estimated, because the provider reports one
    number for the whole prompt.
    """

    prompts = [step.usage.prompt for step in session.steps if step.usage.prompt > 0]
    deltas = [later - earlier for earlier, later in zip(prompts, prompts[1:])]

    images = [image for step in session.steps for image in step.images]
    sizes: dict[str, int] = {}
    for image in images:
        if image.width and image.height:
            key = f"{image.width}x{image.height}"
            sizes[key] = sizes.get(key, 0) + 1
    common_size = max(sizes.items(), key=lambda item: item[1])[0] if sizes else "unknown"

    peak_step = 0
    peak_prompt = 0
    for index, step in enumerate(session.steps):
        if step.usage.prompt > peak_prompt:
            peak_prompt, peak_step = step.usage.prompt, index

    resident = 0
    for step in session.steps[: peak_step + 1]:
        resident += sum(image_tokens(image) for image in step.images)
    resident = min(resident, peak_prompt)

    with_image: list[int] = []
    without_image: list[int] = []
    for index in range(len(session.steps) - 1):
        here, nxt = session.steps[index], session.steps[index + 1]
        if here.usage.prompt <= 0 or nxt.usage.prompt <= 0:
            continue
        delta = nxt.usage.prompt - here.usage.prompt
        (with_image if here.images else without_image).append(delta)

    per_image = [image_tokens(image) for image in images]

    return ContextReport(
        session_id=session.short_id,
        model=session.model or "?",
        steps=len(session.steps),
        window=window,
        peak_prompt=peak_prompt,
        peak_step=peak_step,
        final_prompt=prompts[-1] if prompts else 0,
        first_prompt=prompts[0] if prompts else 0,
        median_growth=round(median(deltas) or 0.0, 1),
        mean_growth=round(sum(deltas) / len(deltas), 1) if deltas else 0.0,
        output_tokens=sum(step.usage.output for step in session.steps),
        image_tokens_at_peak=resident,
        text_tokens_at_peak=max(0, peak_prompt - resident),
        image_count=len(images),
        image_bytes=sum(image.nbytes for image in images),
        image_tokens_total=sum(per_image),
        tool_result_images=sum(1 for image in images if image.origin == "toolResult"),
        prompt_images=sum(1 for image in images if image.origin == "user"),
        common_size=common_size,
        est_tokens_per_image=int(median(per_image) or 0),
        measured_with_image=round(median(with_image), 1) if with_image else None,
        measured_without_image=round(median(without_image), 1) if without_image else None,
        curve=tuple(prompts),
    )


# -- timeline -----------------------------------------------------------------


@dataclass(frozen=True)
class TimelineRow:
    label: str
    milestone_id: str
    presses: int
    delta_presses: int
    seconds: Optional[float]
    delta_seconds: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "milestone": self.milestone_id,
            "label": self.label,
            "presses": self.presses,
            "delta_presses": self.delta_presses,
            "seconds": self.seconds,
            "delta_seconds": self.delta_seconds,
        }


def timeline_rows(record: RunRecord) -> list[TimelineRow]:
    """Each milestone in the order it was actually reached, priced in presses."""

    from pokemon_agent.scope.runs import run_metrics

    metrics = run_metrics(record)
    ordered = sorted(metrics.attainments, key=lambda item: item.seq)
    rows: list[TimelineRow] = []
    previous_presses = 0
    previous_seconds: Optional[float] = 0.0
    for item in ordered:
        rows.append(
            TimelineRow(
                label=item.label,
                milestone_id=item.milestone_id,
                presses=item.presses,
                delta_presses=item.presses - previous_presses,
                seconds=item.seconds,
                delta_seconds=(
                    round(item.seconds - previous_seconds, 1)
                    if item.seconds is not None and previous_seconds is not None
                    else None
                ),
            )
        )
        previous_presses = item.presses
        previous_seconds = item.seconds
    return rows


# -- session digest -----------------------------------------------------------


#: A silence longer than this is the harness restarting the turn, not the model
#: thinking. Measured live, consecutive calls sit 1.3-2.2 s apart at the median
#: and under 6 s at the 95th percentile, so half a minute is unambiguous.
STALL_SECONDS = 30.0


@dataclass(frozen=True)
class Phase:
    """A stretch of the session spent in one place."""

    where: str
    first_step: int
    last_step: int
    calls: int
    seconds: float
    presses: int
    milestones: tuple[str, ...]
    top_commands: tuple[tuple[str, int], ...]
    blocked_calls: int
    #: Tiles first stood on during this phase — the only output of walking.
    new_tiles: int = 0
    #: What the phase started with: a map change, or a stall in the harness.
    boundary: str = "map"

    @property
    def presses_per_new_tile(self) -> Optional[float]:
        return round(self.presses / self.new_tiles, 1) if self.new_tiles else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "where": self.where,
            "boundary": self.boundary,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "calls": self.calls,
            "seconds": round(self.seconds, 1),
            "presses": self.presses,
            "new_tiles": self.new_tiles,
            "presses_per_new_tile": self.presses_per_new_tile,
            "milestones": list(self.milestones),
            "top_commands": [{"command": name, "calls": n} for name, n in self.top_commands],
            "blocked_calls": self.blocked_calls,
        }


def _call_time(call: Call) -> Optional[float]:
    return call.ended_at if call.ended_at is not None else call.started_at


def phases(session: Session, receipts: Sequence[Receipt], *, min_calls: int = 3) -> list[Phase]:
    """Split the session where the game changed map or the harness stalled.

    Those are the two boundaries the transcript reports honestly and cheaply,
    and both are ones a reader cares about: "spent 190 calls on Route 3" starts
    an investigation, and a thirty-second silence is the turn being relaunched.
    A phase that stays on one map for a long time is not split further —
    ``new_tiles`` says whether the time was spent or wasted without needing an
    arbitrary cut.
    """

    locations = _step_locations(session)
    groups: list[list[Call]] = []
    boundaries: list[str] = []
    current_where: Optional[str] = None
    previous_time: Optional[float] = None

    for call in session.calls:
        where = locations.get(call.step, current_where or "?")
        moment = _call_time(call)
        stalled = (
            previous_time is not None
            and moment is not None
            and moment - previous_time >= STALL_SECONDS
        )
        previous_time = moment if moment is not None else previous_time
        if groups and where == current_where and not stalled:
            groups[-1].append(call)
            continue
        if groups and len(groups[-1]) < min_calls and not stalled:
            groups[-1].append(call)
            continue
        groups.append([call])
        boundaries.append("stall" if stalled else "map")
        current_where = where

    seen: set[tuple[str, int, int]] = set()
    out: list[Phase] = []
    for group, boundary in zip(groups, boundaries):
        times = [value for value in (_call_time(call) for call in group) if value is not None]
        start = min(times) if times else None
        end = max(times) if times else None
        window = [
            receipt
            for receipt in receipts
            if receipt.t and start is not None and end is not None and start <= receipt.t <= end
        ]
        counts: dict[str, int] = {}
        for call in group:
            counts[call.label] = counts.get(call.label, 0) + 1
        milestones: list[str] = []
        new_tiles = 0
        for receipt in window:
            milestones.extend(receipt.milestones_new)
            if receipt.pos is not None:
                tile = (receipt.map_name or "?", receipt.pos[0], receipt.pos[1])
                if tile not in seen:
                    seen.add(tile)
                    new_tiles += 1
        out.append(
            Phase(
                where=locations.get(group[0].step, "?"),
                first_step=group[0].step,
                last_step=group[-1].step,
                calls=len(group),
                seconds=(end - start) if start is not None and end is not None else 0.0,
                presses=sum(receipt.presses for receipt in window),
                milestones=tuple(dict.fromkeys(milestones)),
                top_commands=tuple(sorted(counts.items(), key=lambda item: -item[1])[:3]),
                blocked_calls=sum(1 for receipt in window if receipt.blocked),
                new_tiles=new_tiles,
                boundary=boundary,
            )
        )
    return out


# -- diff ---------------------------------------------------------------------


@dataclass(frozen=True)
class DiffRow:
    metric: str
    left: Any
    right: Any
    #: Whether more of this metric is better, worse, or neither.
    direction: str = "neutral"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "a": self.left,
            "b": self.right,
            "direction": self.direction,
        }


def _rate_per_minute(presses: int, seconds: float) -> float:
    return round(presses / (seconds / 60.0), 1) if seconds > 0 else 0.0


def diff_rows(left: RunRecord, right: RunRecord) -> list[DiffRow]:
    """The same figures for two runs, in the order an A/B test is read."""

    from pokemon_agent.scope.runs import ladder_progress, run_metrics

    metrics = [run_metrics(left), run_metrics(right)]
    ladders = [ladder_progress(left), ladder_progress(right)]
    rows = [
        DiffRow("model", metrics[0].model or "?", metrics[1].model or "?"),
        DiffRow("status", metrics[0].status, metrics[1].status),
        DiffRow("harness", metrics[0].harness_sha[:8], metrics[1].harness_sha[:8]),
        DiffRow("receipts", metrics[0].receipts, metrics[1].receipts),
        DiffRow("presses", metrics[0].total_presses, metrics[1].total_presses, "lower"),
        DiffRow("ladder", str(ladders[0]), str(ladders[1]), "higher"),
        DiffRow(
            "milestones earned",
            metrics[0].milestones_reached,
            metrics[1].milestones_reached,
            "higher",
        ),
        DiffRow(
            "presses/milestone",
            metrics[0].presses_per_milestone,
            metrics[1].presses_per_milestone,
            "lower",
        ),
        DiffRow("blocked rate", metrics[0].blocked_rate, metrics[1].blocked_rate, "lower"),
        DiffRow("revisit ratio", metrics[0].revisit_ratio, metrics[1].revisit_ratio, "lower"),
        DiffRow("tool error rate", metrics[0].tool_error_rate, metrics[1].tool_error_rate, "lower"),
        DiffRow("unique tiles", metrics[0].unique_positions, metrics[1].unique_positions, "higher"),
        DiffRow("whiteouts", metrics[0].whiteouts, metrics[1].whiteouts, "lower"),
        DiffRow("reloads", metrics[0].reloads, metrics[1].reloads, "lower"),
        DiffRow(
            "wall clock (min)",
            round(metrics[0].wall_clock_seconds / 60.0, 1),
            round(metrics[1].wall_clock_seconds / 60.0, 1),
            "lower",
        ),
        DiffRow(
            "presses/min",
            _rate_per_minute(metrics[0].total_presses, metrics[0].wall_clock_seconds),
            _rate_per_minute(metrics[1].total_presses, metrics[1].wall_clock_seconds),
            "higher",
        ),
    ]
    return rows


def diff_milestones(
    left: RunRecord, right: RunRecord
) -> list[tuple[str, Optional[int], Optional[int]]]:
    """Presses-to for every milestone either run reached, in ladder order."""

    from pokemon_agent.scope.runs import run_metrics

    metrics_left, metrics_right = run_metrics(left), run_metrics(right)
    labels: dict[str, str] = {}
    for metrics in (metrics_left, metrics_right):
        for item in metrics.attainments:
            labels[item.milestone_id] = item.label
    order: list[str] = []
    for metrics in (metrics_left, metrics_right):
        for item in metrics.attainments:
            if item.milestone_id not in order:
                order.append(item.milestone_id)
    return [
        (
            labels.get(milestone_id, milestone_id),
            metrics_left.presses_for(milestone_id),
            metrics_right.presses_for(milestone_id),
        )
        for milestone_id in order
    ]
