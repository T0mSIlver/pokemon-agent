# Relationship to upstream

This repo is a fork of [NousResearch/pokemon-agent](https://github.com/NousResearch/pokemon-agent).
It has diverged substantially and **we do not track upstream `main`.**

```bash
git remote add upstream https://github.com/NousResearch/pokemon-agent.git
git fetch upstream
git log --oneline main..upstream/main     # what they've done since we split
```

## Decision: do not merge (2026-07-12)

At the time of review we were 23 commits ahead, upstream 8 ahead, off a common base of
`c2e222e` (2026-03-09). A real `git merge upstream/main` conflicts across 7 files —
including all three dashboard files, which **both sides rewrote from scratch**.

The merge was rejected because upstream's post-split work either duplicates ours with a
weaker design, or collides head-on with a deliberate architectural choice.

| Upstream addition | Verdict | Why |
| --- | --- | --- |
| `collision.py` — "ground-truth collision map" | **Reject** | We already read collision from RAM, and more authoritatively. PyBoy's `game_area_collision()` dereferences the engine's own `wTilesetCollisionPtr` (0xD530); upstream hand-transcribes a static tileset→walkable table, which risks transcription error and misses `wGrassTile`. We also model tile-pair blockers (ledges, `navigation.py:TILE_PAIR_BLOCKERS`) and NPC sprite blocking (`emulator.py:_get_visible_sprites`), which upstream has no answer for. Their tile IDs are raw `wTileMap` values; ours carry PyBoy's +0x100 offset — not drop-in compatible either. |
| `overlay.py` | **Reject the file** | `agent_runtime.py:render_navigation_overlay` already does everything it does and more (sprite + warp markers, objective header band). One idea worth stealing: upstream's stable `A1..J9` cell names give the model a fixed spatial vocabulary, where our absolute map coords change every step. |
| `autopilot.py` + Hermes brain | **Reject** | Competes directly with `pi_supervisor.py` + the typed `harness/` plan contract. Upstream drives a `hermes chat` subprocess with free-form prose and expects the model to curl back into the server; session ID is recovered by regex-scraping stdout. Adopting it would mean two incompatible brains. |
| Dashboard redesign ("Field Log") | **Reject** | Both sides rewrote `app.js` / `style.css` / `index.html` independently. Ours carries the Pi chat transcript, tool-call telemetry and fullscreen viewports we depend on. |
| Gen-1 species mapping fix (`960d113`) | **Already had it** | We fixed this independently. Our `INTERNAL_SPECIES_TO_DEX` is byte-identical to theirs across all 151 species. |

## What we did take

- **Mono-type dedupe** (`memory/red.py:_decode_types`). Gen 1 writes a mono-type's single
  type into *both* type bytes, so a Squirtle was reporting `["Water", "Water"]`. Ported by
  hand; covered by `tests/test_species_types.py`.

## The one real gap — closed

**`sessions.py` — run-scoped game sessions.** This was the single thing upstream had that we
genuinely lacked: saves went to one flat global directory, the workspace was a shared folder,
and `PiSupervisor.session_id` was never persisted, so a restart kept the emulator state but
orphaned the agent's brain.

Built in `pokemon_agent/sessions.py` and wired up as the `/games` endpoints — **adapted, not
copied**. Differences from upstream's version:

- Keyed on the Pi session id, not `hermes_session_id`.
- The session directory doubles as the `AgentRuntime` `data_dir`, so the runtime's *auto*
  saves follow the run too. Upstream only scoped server-written saves.
- The run owns the agent workspace (memory, frames, Pi transcript), not just saves.
- Dropped upstream's `_latest_badges()` — it returns `1` whenever any badge milestone exists,
  which is simply wrong. Badge count should come from the RAM reader.
- Dropped upstream's free-text `DEFAULT_OBJECTIVES` in favour of a reference to our objective
  packs (`objective_pack`), so it doesn't fork the `ObjectiveEngine`.
- `adopt_session()` fails closed when a transcript is missing, rather than leaving a dangling
  id that would make Pi silently resume whichever session happened to be newest.
