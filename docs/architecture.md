# Architecture

Four processes, one emulator.

```
llama.cpp (192.168.1.183)          the model
      ^
      | OpenAI API via Bifrost
      |
  pi --mode rpc  <-------------->  pi_supervisor.py     one long-lived session
      |                                  |
      | bash: ./poke ...                 | JSON-RPC over stdio
      v                                  v
  agent_cli.py  --- HTTP --->  server.py (FastAPI)
                                         |
                                         v
                                   emulator.py (PyBoy)
                                         |
                                         v
                                   memory/red.py        RAM -> JSON
```

Only `server.py` touches the emulator. Everything else goes through HTTP.

## The four processes

**The game server** (`server.py`) owns the PyBoy instance and every byte of
game state. It exposes `/action`, `/state`, `/save`, `/load`, `/map`, and the
frame artifacts. It is the single writer. Nothing else may call PyBoy.

**The supervisor** (`pi_supervisor.py`) runs one `pi --mode rpc` child process
for the whole session and speaks JSON-RPC to it over stdio. It passes
`skill/SKILL.md` as the system prompt, attaches the two frames to the first
message, and sends `continue` after that. It also owns the goal, the critic
handoff, and the operator steer box.

**The agent** is the model inside that `pi` process. It sees frames and runs
`./poke` in bash. It has no other way to reach the game.

**The runtime** (`agent_runtime.py`) sits inside the server process. It draws
the annotated frame, writes the workspace artifacts, and tracks objectives.

## The turn loop

1. The supervisor sends `continue`.
2. The model runs `./poke act up up a`.
3. `agent_cli.py` POSTs to `/action`.
4. The server presses the buttons, ticks the emulator, and reads RAM.
5. The runtime redraws `latest_frame.png` and `latest_frame_annotated.png`.
6. `/action` returns a compact summary. The model reads the frames itself.

Step 6 is the part that took several rewrites to get right. See "Observation
design" below.

## Why the model drives a CLI instead of curl

The model used to build `curl` calls by hand. Across one 261-failure sample,
260 failures were an unbalanced quote in a JSON body, which bash rejects before
anything reaches the server. Roughly 40% of intended actions were lost that way.

`./poke act right:6 a` has nothing to misquote. Tool-call failures went to zero
and stayed there. Every capability added since goes through the same CLI for
the same reason.

## Observation design

The rule the observation payload follows: **report what happened, do not
advise.**

Three fields changed behaviour when added:

| Field | What it says | Measured effect |
|---|---|---|
| `faces` | what the player is facing | `press_a` went from 0 to 138 uses |
| `blocked_after` | which step of the batch hit a wall | blocked batches fell from 53% to 7% |
| `moved` | tiles actually moved | batch size fell from 17.2 to 8.9 |

Three did not:

| Signal | Outcome |
|---|---|
| `here_before` visit counter | reached 49 on one tile with no change in behaviour |
| `GET /map` | called 4 times in 299 tool calls |
| Prompt guidance to heal and to save | never acted on |

The pattern holds across every change so far. A fact that arrives inside the
loop the model is already running changes what it does. A fact it has to notice
and remember does not. Anything new should either refuse an action, report a
consequence, or not exist.

## The capability modules

Six modules the server exposes over HTTP. None of them are reachable from the
agent directly: `agent_cli.py` is stdlib-only and staged into the workspace as a
standalone script, so everything it can do is an endpoint.

| Module | Holds |
|---|---|
| `milestones.py` | 507 named event flags, 55 of them ordered into a ladder alongside three RAM bits |
| `bench/` | run registry and the presses-to-milestone scoreboard |
| `gamedata/` | the whole game as JSON, generated from pokered |
| `world.py` | routing over the map graph, plan simulation, `frontier()` |
| `guides/` | three walkthrough routes as a retrievable corpus |
| `interventions.py` | detectors deciding when to stop the player and think |
| `slots.py` | borrowing the model's KV slot for a thinking session |

`gamedata/` and `guides/` are generated, not hand-written. `scripts/gen_gamedata.py`
and `scripts/gen_milestones.py` rebuild them from pokered and are idempotent;
re-running produces byte-identical files. Regenerate rather than editing the JSON.

## What is measured

The headline metric is **button presses to each milestone**, chosen because
PokeAgent and Continual Harness both report it, so runs here are comparable to
published ones. For scale: PokeAgent's best entry reached the first gym in 1,608
actions, the most efficient in 649, and a human speedrunner takes about 18
minutes.

Two rules make the number honest:

- **Presses never reset on a save-state reload.** `bench/metrics.py` has no
  branch on `reloaded` at all. A gym won on the fourth attempt costs what all
  four attempts cost.
- **Nothing judges progress except the game's own RAM.** A milestone is a bit in
  `wEventFlags`, a badge bit, or an item in the bag. No model is asked whether
  the run is going well.

## Interventions

The player model runs non-thinking so it stays fast. A thinking session can be
swapped in by saving the player's KV slot to disk, running the thinker in the
freed slot, and restoring. The box runs `--parallel 1`, so there is exactly one
slot and this is the only way.

`interventions.py` decides when. Six detectors read receipts and fire on their
own: a commit gate before something irreversible, sustained low HP, the same
command failing twice, no milestone in 800 presses, a revisit ratio over 2.5,
and first arrival on one of eighteen known-hard maps.

The harness fires these. The model is never asked to notice it is stuck and call
for help, because that is the advisory pattern that has failed every time.

`slots.py` does the borrowing, and the dangerous half is the giving back:
between the erase and the restore the player's whole context exists only as a
file. `borrowed_slot` refuses to hand the slot over unless the save is confirmed
non-empty, and raises `SlotLost` with the filename rather than failing quietly.

## Coordinates

North is up. `walk_up` decreases `y`. The annotated frame's row numbers are
absolute map tiles, and the header repeats the rule because the model got it
wrong otherwise.

Map names come from `MAP_NAMES` in `memory/red.py`, regenerated wholesale from
[pret/pokered](https://github.com/pret/pokered). The hand-maintained table it
replaced had 169 of 248 entries wrong, which is why the Pewter Poké Center used
to report itself as Mt. Moon 1F. Anything that names a map must use those exact
strings, apostrophes included: the map is `Red's House 1F`, not
`Reds House 1F`.

## Emulator timing

Loading a save state and reading immediately returns stale data. Mid-transition
saves relocate the player for about two seconds after the load. Always tick the
emulator and let it settle before trusting a read. This has caused two separate
multi-hour debugging sessions.

## Persistence

| Path | Holds |
|---|---|
| `<data_dir>/saves/` | emulator save states |
| `<data_dir>/explored_maps.json` | per-map fog of war, survives restarts |
| `<workspace>/debug/run_log.jsonl` | every turn, appended |
| `<workspace>/debug/critic_last.jsonl` | last retrospective, raw |
| `<workspace>/NOTES.md` | the model's own notes below a `harness-state` block the harness rewrites from the game at session start and end |

The workspace is the model's home directory. Nothing in it is generated on the
model's behalf except the two frames, the staged `poke` binary, and the
`harness-state` block at the top of `NOTES.md`.

## Prefix caching

`skill/SKILL.md` is the system prompt and must stay byte-identical across
sessions. llama.cpp reuses the KV cache only for an exact prefix match, so a
single changed character anywhere in the system prompt costs a full re-prefill.
Goals, retrospectives, and workspace paths go in the first *user* message
instead, which is why `_resolve_goal` runs before `_initial_message` and never
after.

## Sessions and compaction

A session ends when the token budget is reached, currently 110,000. The
watchdog (`scripts/keep_run_alive.sh`) starts a new one. Between the two, the
critic reads the finished session at high reasoning effort and writes a
retrospective ending in a `NEXT GOAL:` line, which becomes the next session's
goal unless an operator goal overrides it.

Goal precedence, resolved once at `start()`:

1. operator goal passed to `/supervisor/start`
2. the critic's `NEXT GOAL:` line
3. the objective engine
4. a hard-coded fallback

The goal is fixed for a session's lifetime. To redirect a running session, use
`POST /supervisor/steer`, which delivers a message at the next turn boundary.

## Remote access

The server binds locally. To watch a run from another machine, use
`scripts/tunnel.sh`. See [remote-access.md](remote-access.md).
