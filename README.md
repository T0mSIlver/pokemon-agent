# pokemon-agent

A local language model plays Pokémon Red, unattended, around the clock.

A headless Game Boy emulator runs the real game. A FastAPI server owns it and
reads its RAM. A long-lived `pi` session drives a model against that server
through a small CLI. An operator dashboard lets you watch and steer.

This is a fork of [NousResearch/pokemon-agent](https://github.com/NousResearch/pokemon-agent),
which ships the game server and expects you to bring your own agent. This fork
adds the agent side and a scoreboard.

The current run uses `qwen38-27b` on a single 3090, in non-thinking mode.

## What is different from upstream

The harness used to be much larger. It enforced a typed per-turn plan contract,
ran a deterministic route planner, and kept a semantic memory store, all so a
weak local model could produce something valid. A model trained for agentic
work does not need any of that, and it cost more in prompt tokens and failure
modes than it bought. It is gone.

What replaced it:

- `poke`, a CLI staged into the model's workspace. Bare-word actions, no JSON,
  nothing to misquote.
- An annotated frame with a tile grid, warp markers, sprite blockers, and a
  per-map fog of war that survives restarts.
- A retrospective written between sessions at high reasoning effort, which
  hands the next session a goal.
- A milestone oracle built from the 507 named event flags in the game's own
  RAM, and a run registry that scores a run in button presses.

We do not track upstream `main`. See [docs/upstream.md](docs/upstream.md) for
what upstream has, what we rejected, and why.

## Install

Not published to PyPI. `pip install pokemon-agent` fetches upstream, not this.

```bash
git clone https://github.com/T0mSIlver/pokemon-agent.git
cd pokemon-agent
uv venv
uv pip install -e ".[pyboy,dashboard,dev]"
```

Bring your own ROM. None are included and none can be committed.

```bash
mkdir -p roms && cp /path/to/pokemon_red.gb roms/
```

`roms/`, `*.gb`, and save states are gitignored.

## Run

Start the server:

```bash
uv run pokemon-agent serve \
  --rom roms/pokemon_red.gb \
  --port 8765 \
  --agent-workspace-dir "$(pwd)/.agent-workspace"
```

Then open `http://localhost:8765/dashboard` and start a run from the Pi
Supervisor panel, or from the shell:

```bash
MODEL=llamacpp/qwen38-27b GOAL="Reach Viridian City" scripts/start_pi_run.sh
```

`scripts/keep_run_alive.sh` restarts a run when its token budget runs out, so a
playthrough continues across sessions without supervision.

To watch from another machine, `scripts/tunnel.sh dev@<host>` and open the
dashboard locally. Full URL list in [docs/remote-access.md](docs/remote-access.md).

## How the model plays

Everything the model does goes through one command in its workspace:

```bash
./poke act up up a          # walk two tiles north, then press A
./poke act right:6 a        # repeat form
./poke fight 2              # use the second move in battle
./poke state                # compact JSON state
./poke map                  # the map it is standing in
```

It reads `latest_frame.png` and `latest_frame_annotated.png` with vision after
every action. `skill/SKILL.md` is the system prompt and documents the rest.

Actions available to `poke act`:

| Action | Effect |
|---|---|
| `up` `down` `left` `right` | walk one tile (long form `walk_up`, and so on) |
| `a` `b` `start` `select` | press a button |
| `adialog` | press A until the dialog closes, up to 10 times |
| `wait_60` | idle 60 frames |
| `hold_a_30` | hold A for 30 frames |

Append `:N` to any action to repeat it.

## Scoring a run

The headline number is button presses to each milestone, which is what
[PokeAgent](https://arxiv.org/html/2603.15563) and
[Continual Harness](https://arxiv.org/html/2605.09998v1) both report, so runs
here are comparable to published ones.

```bash
uv run python -m pokemon_agent.bench                    # list runs
uv run python -m pokemon_agent.bench <run_id>           # one run
uv run python -m pokemon_agent.bench --compare a b c    # side by side
```

The ladder has 58 rungs, from the starter through the Hall of Fame, each backed
by a bit in the game's own RAM that the game never clears again. No model judges progress. Presses never reset
on a save-state reload, so a gym won on the fourth attempt reads as exactly
that.

For reference: PokeAgent's fastest entry reached the first gym in 1,608
actions, the most efficient in 649, and a human speedrunner takes about 18
minutes.

## Development

```bash
uv run pytest          # full suite, no ROM needed
uv run ruff check .
uv run ruff format .
```

`pytest` collects only `tests/`. The scripts in `scripts/manual/` boot a real
emulator against a ROM and are meant to be run by hand.

## Documentation

- [docs/architecture.md](docs/architecture.md) covers the process layout, the
  turn loop, the observation design rule, and the coordinate and timing traps
  that have cost this project the most time.
- [docs/remote-access.md](docs/remote-access.md) covers tunnelling and frame
  URLs.
- [docs/upstream.md](docs/upstream.md) records why we diverged.

## HTTP API

| Endpoint | Method | Purpose |
|---|---|---|
| `/state` | GET | full game state as JSON |
| `/action` | POST | press buttons, refresh frames, return a compact summary |
| `/map` | GET | the current map's collision and warps |
| `/battle/fight` | POST | select a move by index |
| `/battle/run` | POST | flee |
| `/screenshot` | GET | current frame as PNG |
| `/save` `/load` `/saves` | POST/GET | emulator save states |
| `/artifacts/{name}` | GET | workspace artifacts, including both frames |
| `/supervisor/start` `/continue` `/stop` | POST | control the model session |
| `/supervisor/steer` | POST | inject a message into a running session |
| `/supervisor/state` | GET | supervisor snapshot |
| `/health` | GET | health check |
| `/ws` | WebSocket | live event stream |
| `/dashboard` | GET | operator console |

`/action` is rate limited to 60 calls a minute. A runaway loop once sent 5,550.

## Library use

```python
from pokemon_agent.emulator import create_emulator
from pokemon_agent.memory.red import PokemonRedReader
from pokemon_agent.state.builder import build_game_state

emu = create_emulator("roms/pokemon_red.gb")
reader = PokemonRedReader(emu)
state = build_game_state(reader)

print(state["player"]["badges_list"])
print([p["species"] for p in state["party"]])
```

## Supported games

Pokémon Red and Blue work. Yellow shares the same RAM layout and should work,
though nothing here is tested against it. FireRed has a partial reader in
`memory/firered.py` that cannot decrypt party data yet. Gold, Silver, and
Emerald are not implemented.

## Help wanted

- A Gold/Silver/Crystal memory reader.
- FireRed party decryption.
- Run-scoped sessions, so two playthroughs can run side by side. See
  [docs/upstream.md](docs/upstream.md).

## License

MIT. See [LICENSE](LICENSE).

## Built on

- [PyBoy](https://github.com/Baekalfen/PyBoy) for Game Boy emulation.
- [pret/pokered](https://github.com/pret/pokered) for every RAM address, the
  map name table, and the 507 event constants the milestone ladder is built
  from.
- [The Making of Gemini Plays Pokémon](https://blog.jcz.dev/the-making-of-gemini-plays-pokemon),
  which is the best account anywhere of what actually goes wrong when a model
  plays this game, and where several ideas here came from.
