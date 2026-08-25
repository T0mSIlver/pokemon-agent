# What this fork changes

A fork of [NousResearch/pokemon-agent](https://github.com/NousResearch/pokemon-agent). Upstream ships a game server and expects you to bring your own agent. This fork adds the agent side: a supervisor that runs a Pi model against the server, an annotated frame the model can actually navigate from, and an operator dashboard to watch it.

The harness used to be much bigger. It enforced a typed per-turn plan contract, ran a deterministic route planner, and kept a semantic memory store, all so that a weak local model could produce something valid. Strong agentic models do not need any of it, and it cost more in prompt tokens and failure modes than it bought. It is gone. What is left:

- **Pi supervisor** (`pokemon_agent/pi_supervisor.py`) drives one long-lived `pi --mode rpc` process. `skill/SKILL.md` is passed as the system prompt, the two frames are attached to the first message, and every turn after that is the word `continue`. The model chooses its own actions.
- **Annotated frames** (`pokemon_agent/agent_runtime.py`) draw a tile grid over the 160x144 screen: walkable vs blocked, warp tiles, NPC/object blockers, the tile the player is facing, and a header with map name, position, facing, valid moves, and current objective. This is the model's main input.
- **Collision and warp data from RAM** (`pokemon_agent/navigation.py`, `pokemon_agent/emulator.py`) reads the engine's own collision pointer rather than a hand-transcribed tileset table, and models ledge tile-pairs and sprite blocking. It feeds the overlay; there is no route-planner endpoint.
- **Objective tracking** (`pokemon_agent/data/red_objectives*.json`) gives the overlay header a current objective and the dashboard a progress readout.
- **Operator dashboard** (`pokemon_agent/dashboard/`) shows the frames, the Pi chat transcript with prompts, replies, thinking, tool calls and stderr, plus save controls and live telemetry.
- **Emulator fixes** — Gen 1 movement timing, overlay coordinate correctness, mono-type dedupe in the Red memory reader.
- **Ops scripts** — `scripts/start_pokemon_server.sh`, `scripts/stop_pokemon_server.sh`, `scripts/start_pi_run.sh`, `scripts/tunnel.sh`.

The model's own memory is a `NOTES.md` file in its workspace that it writes and reads itself. Nothing on the harness side summarises the run for it.

OCR has been removed; the agent reads the screen with vision.

We do **not** track upstream `main` — it has diverged in incompatible directions. See
[`docs/upstream.md`](docs/upstream.md) for what upstream has, what we rejected and why, and
the one gap still worth closing.

---

# 🎮 pokemon-agent

**AI-powered Pokémon gameplay agent with headless emulation, REST API, and a live operator dashboard.**

Let any AI agent — [Hermes Agent](https://github.com/NousResearch/hermes-agent), Claude Code, Codex, or your own — play Pokémon games autonomously via a clean HTTP API. Runs headlessly on any server or terminal. No display, no GUI, no emulator window needed.

```
┌──────────────────────┐
│   Your AI Agent      │  Any LLM-powered agent
│   (Hermes, Claude,   │  makes the decisions
│    Codex, custom)    │
└─────────┬────────────┘
          │ HTTP API
┌─────────▼────────────┐
│   pokemon-agent      │  This package:
│   ┌────────────────┐ │  - Headless emulator
│   │ Game Server    │ │  - Memory reader
│   │ (FastAPI)      │ │  - Game state parser
│   ├────────────────┤ │  - REST + WebSocket API
│   │ Emulator       │ │  - Optional dashboard
│   │ (PyBoy/PyGBA)  │ │
│   └────────────────┘ │
└──────────────────────┘
```

## Features

- **🔌 Headless emulation** — No display server, X11, or GUI needed. Pure in-process emulation.
- **🌐 REST API** — `GET /state`, `POST /action`, `GET /screenshot` — control the game over HTTP.
- **📡 WebSocket** — Real-time event streaming for live monitoring.
- **🧠 Structured game state** — RAM is parsed into clean JSON: party, bag, badges, map, battle, dialog.
- **🖼️ Annotated frames** — Tile grid, walkability, warp markers, sprite blockers, and an objective header drawn over the raw screen.
- **🎨 Live dashboard** — Operator console with both frames, objective state, and Pi supervisor telemetry.
- **💬 Pi chat transcript** — Prompts, assistant replies, thinking, tool calls, and stderr in one place.
- **🎮 Multi-game** — Supports Game Boy (Pokémon Red/Blue) via PyBoy, GBA (FireRed) via PyGBA.
- **🤖 Agent-agnostic** — Works with any AI agent, RL framework, or custom script.

## Quick Start

### Installation

This fork is not published to PyPI — install it from source. (`pip install pokemon-agent`
would fetch upstream NousResearch, not this fork.)

```bash
git clone https://github.com/T0mSIlver/pokemon-agent.git
cd pokemon-agent

uv venv
uv pip install -e ".[pyboy,dashboard,dev]"
```

Drop `dev` if you only want to run the stack, and `dashboard` if you don't want the web GUI.

### The ROM

> **You must provide your own ROM.** No game ROMs are included, and none can be committed.

Put it wherever you like and pass `--rom`. The manual diagnostic scripts in `scripts/manual/`
assume `roms/pokemon_red.gb` relative to the repo root, so that's the convenient default:

```bash
mkdir -p roms && cp /path/to/pokemon_red.gb roms/
```

`roms/`, `*.gb`, and save states are all gitignored.

### Development

```bash
uv run pytest          # full suite — no ROM needed
uv run ruff check .    # lint
uv run ruff format .   # format
```

`pytest` collects only `tests/`. The scripts in `scripts/manual/` are interactive
diagnostics that boot a real emulator against a ROM — run them by hand, e.g.
`uv run python scripts/manual/check_movement.py`.

### Start the Server Manually

```bash
uv run pokemon-agent serve \
  --rom path/to/pokemon_red.gb \
  --port 8765 \
  --agent-workspace-dir "$(pwd)/.agent-workspace"
```

```
╔══════════════════════════════════════╗
║       🎮 Pokémon Agent Server       ║
╚══════════════════════════════════════╝
  Game:       Pokemon Red
  ROM:        pokemon_red.gb
  API:        http://localhost:8765
  Dashboard:  http://localhost:8765/dashboard
  WebSocket:  ws://localhost:8765/ws
```

Start the server yourself in a terminal first. Once it is running, open the dashboard and launch Pi from there.

### Start Pi From The Server Dashboard

1. Open [http://localhost:8765/dashboard](http://localhost:8765/dashboard).
2. Confirm the server is healthy and the latest frame is visible.
3. Use the Pi Supervisor panel to choose the goal and model settings.
4. Click `Start Pi`.

The dashboard will then show:

- annotated and raw frames
- current objective
- Pi chat transcript with explicit message roles
- streamed assistant output and thinking
- tool calls, stderr, and recent events
- a manual `Save Now` button

Or start a run from the shell — this waits for the model endpoint to answer first,
which matters for local models that get swapped in on demand:

```bash
MODEL=llamacpp/qwen38-27b GOAL="Reach Viridian City" scripts/start_pi_run.sh
```

### Watching a Run Remotely

Run the server on the box with the ROM and reach it over an SSH tunnel:

```bash
# on your laptop
scripts/tunnel.sh dev@192.168.1.98
# then open http://localhost:8765/dashboard
```

Frames are also fetchable directly — `/artifacts/live_frame_annotated`,
`/artifacts/latest_frame_annotated`, `/screenshot`. See
[`docs/remote-access.md`](docs/remote-access.md) for the full URL list and
troubleshooting.

### Play from Any Agent

The whole model-facing API is `POST /action` to act and the frames to see. Everything else is for operators.

```bash
# Get game state
curl http://localhost:8765/state | python -m json.tool

# Take a screenshot
curl http://localhost:8765/screenshot -o screen.png

# Send actions — the response is a compact state summary
curl -X POST http://localhost:8765/action \
  -H "Content-Type: application/json" \
  -d '{"actions": ["walk_up", "walk_up", "press_a"]}'

# Save/load state
curl -X POST http://localhost:8765/save -d '{"name": "before_brock"}'
curl -X POST http://localhost:8765/load -d '{"name": "before_brock"}'
```

`POST /action` also refreshes two PNGs in the agent workspace:

- `latest_frame_annotated.png` — the grid overlay with warps, blockers, and the objective header
- `latest_frame.png` — the raw 160x144 screen

An agent reads those files to see the game. `turn_context.json` is written alongside them for
the dashboard and for debugging; it is not part of any agent contract.

`skill/SKILL.md` is the system prompt the supervisor gives Pi. It documents the overlay, the
warp rule, and the action set, and it is the right starting point for writing your own agent.

### Game State (JSON)

```json
{
  "player": {
    "name": "ASH",
    "money": 3000,
    "badges": 1,
    "badges_list": ["Boulder"],
    "position": {"map_id": 1, "map_name": "PALLET TOWN", "x": 7, "y": 5},
    "facing": "down",
    "play_time": {"hours": 1, "minutes": 23, "seconds": 45}
  },
  "party": [
    {
      "nickname": "SQUIRTLE",
      "species": "Squirtle",
      "level": 12,
      "hp": 33,
      "max_hp": 33,
      "moves": ["Tackle", "Tail Whip", "Bubble"],
      "status": null,
      "types": ["Water"]
    }
  ],
  "bag": [{"item": "Potion", "quantity": 3}],
  "battle": null,
  "dialog": {"active": false, "text": null},
  "flags": {"has_pokedex": true, "badges_earned": ["Boulder"]},
  "metadata": {"game": "Pokemon Red", "frame_count": 12345}
}
```

## Actions Reference

| Action | Description |
|--------|-------------|
| `press_a` | Press A button (8 frames press + 12 wait) |
| `press_b` | Press B button |
| `press_start` | Press Start button |
| `press_select` | Press Select button |
| `walk_up` | Walk one tile up |
| `walk_down` | Walk one tile down |
| `walk_left` | Walk one tile left |
| `walk_right` | Walk one tile right |
| `hold_a_30` | Hold A for 30 frames |
| `wait_60` | Wait 60 frames (~1 second) |
| `a_until_dialog_end` | Press A repeatedly until dialog closes (max 10 presses) |

## Dashboard

Install with the dashboard extra to get the full operator console:

```bash
pip install pokemon-agent[dashboard]
```

Then open `http://localhost:8765/dashboard` in your browser.

The dashboard shows:
- **Annotated and raw frames** — The same images the agent is looking at
- **Pi supervisor controls** — Start, continue, stop, and auto-continue configuration
- **Chat transcript** — Explicit `user`, `assistant`, `assistant thinking`, and `system` message roles
- **Tool and stderr streams** — Live visibility into what Pi is calling and what fails
- **Objective state** — Where the run is in the Red objective chain
- **Save controls** — Create manual saves and load named saves from the UI

## Supported Games

| Game | Emulator | Status | Install |
|------|----------|--------|---------|
| Pokémon Red/Blue | PyBoy | ✅ Supported | `pip install pyboy` |
| Pokémon Yellow | PyBoy | ✅ Supported | `pip install pyboy` |
| Pokémon Gold/Silver | PyBoy | 🔜 Planned | `pip install pyboy` |
| Pokémon FireRed/LeafGreen | PyGBA | 🔜 Phase 2 | `pip install pygba` |
| Pokémon Ruby/Sapphire/Emerald | PyGBA | 🔜 Phase 2 | `pip install pygba` |

## Use with Hermes Agent

[Hermes Agent](https://github.com/NousResearch/hermes-agent) ships its own `pokemon-player` skill that installs upstream pokemon-agent, starts the server, and plays. It is a separate agent from the Pi supervisor here.

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server info |
| `/state` | GET | Full game state JSON |
| `/screenshot` | GET | Current frame (PNG) |
| `/screenshot/base64` | GET | Current frame (base64 JSON) |
| `/action` | POST | Execute game actions, refresh the frames, return compact state |
| `/save` | POST | Save emulator state |
| `/load` | POST | Load emulator state |
| `/saves` | GET | List saved states |
| `/artifacts/{artifact}` | GET | Serve workspace artifacts such as the live and latest frames |
| `/dashboard/state` | GET | Aggregated dashboard state |
| `/dashboard/history` | GET | Structured recent event history |
| `/supervisor/state` | GET | Pi supervisor snapshot |
| `/supervisor/start` | POST | Launch Pi from the server with an optional goal override |
| `/supervisor/continue` | POST | Continue one Pi turn |
| `/supervisor/stop` | POST | Stop the supervised Pi session |
| `/health` | GET | Health check |
| `/ws` | WebSocket | Live event stream |
| `/dashboard` | GET | Web dashboard (if installed) |

## Python API

You can also use `pokemon-agent` as a library:

```python
from pokemon_agent.emulator import create_emulator
from pokemon_agent.memory.red import PokemonRedReader
from pokemon_agent.state.builder import build_game_state

# Load ROM headlessly
emu = create_emulator("pokemon_red.gb")

# Create memory reader
reader = PokemonRedReader(emu)

# Get structured game state
state = build_game_state(reader)
print(f"Player: {state['player']['name']}")
print(f"Badges: {state['player']['badges']}")
print(f"Party: {[p['species'] for p in state['party']]}")

# Send inputs
emu.press("a", frames=10)
emu.tick(20)

# Get screenshot
image = emu.get_screen()  # PIL Image
image.save("screenshot.png")
```

## Architecture

```
pokemon_agent/
├── cli.py               # CLI entry point (pokemon-agent command)
├── server.py            # FastAPI game server (REST + WebSocket)
├── emulator.py          # PyBoy/PyGBA wrapper (headless)
├── agent_runtime.py     # Workspace artifacts, frame overlay, objective engine
├── pi_supervisor.py     # Long-lived Pi process, frame attachments, continue loop
├── navigation.py        # Collision / warp / sprite snapshot from RAM
├── pathfinding.py       # Direction constants and action mapping
├── harness/             # Prompt builders
├── data/                # Pokémon Red objective chains
├── memory/
│   ├── reader.py        # Abstract game memory reader
│   ├── red.py           # Pokémon Red/Blue RAM parser
│   └── firered.py       # FireRed RAM parser (Phase 2)
├── state/
│   └── builder.py       # Structured state builder
└── dashboard/           # Optional [dashboard] extra
    ├── mount.py         # FastAPI static mount
    └── static/          # Dashboard page, styles, WebSocket client
```

## Contributing

Contributions welcome! Areas where help is needed:

- **Pokémon Gold/Silver/Crystal** memory reader (`memory/gold.py`)
- **Pokémon FireRed** full memory reader with decryption (`memory/firered.py`)
- **Pokémon Emerald** memory reader (`memory/emerald.py`)
- **Run-scoped sessions** so two playthroughs can run side by side (see `docs/upstream.md`)
- **Dashboard** enhancements (progress tracking, key moments, replay)
- **Tests** for memory readers and state builders

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- [PyBoy](https://github.com/Baekalfen/PyBoy) — Game Boy emulator in Python
- [PyGBA](https://github.com/dvruette/pygba) — GBA emulator wrapper
- [pret/pokered](https://github.com/pret/pokered) — Pokémon Red decompilation (memory addresses)
- [pret/pokefirered](https://github.com/pret/pokefirered) — FireRed decompilation
- [gpt-play-pokemon-firered](https://github.com/Clad3815/gpt-play-pokemon-firered) — Architecture inspiration
- [Hermes Agent](https://github.com/NousResearch/hermes-agent) — AI agent platform by Nous Research
