# What this fork changes

This is a fork of [NousResearch/pokemon-agent](https://github.com/NousResearch/pokemon-agent) that extends the stock harness with a stricter, more observable agent runtime and a richer operator UI. Highlights relative to upstream:

- **Vision-first Pi supervisor** (`pokemon_agent/pi_supervisor.py`) — supervises a Claude Code–style Pi subprocess, enforces vision policy on every turn, tracks tool calls / thinking / stderr, and handles auto-continue for text-only replies.
- **Strict turn-plan harness with typed contracts** (`pokemon_agent/harness/`) — `contracts.py`, `context_builder.py`, `planning.py`, and `prompting.py` define a typed per-turn plan/context pipeline so weaker local models get deterministic inputs and validated outputs.
- **Deterministic navigation + route guidance** (`pokemon_agent/navigation.py`, `navigation_maps.json`, `pokemon_agent/data/red_objectives*.json`) — frontiers, landmarks, distance maps, route cards, avoidances, and NPC-aware objective guidance for Pokémon Red.
- **Durable semantic memory** — append-only event memory, session brief, and per-map failed-attempt tracking in `pokemon_agent/agent_runtime.py` and `pokemon_agent/memory/red.py`, so loops and resumes keep context.
- **Expanded dashboard** (`pokemon_agent/dashboard/`) — Pi chat transcript (prompts, assistant replies, thinking, tool calls, stderr, auto-continue), fullscreen frame viewports, improved tool-call payload rendering, and live Pi telemetry.
- **Server + emulator upgrades** (`pokemon_agent/server.py`, `pokemon_agent/emulator.py`) — larger REST surface for the supervisor workflow, overlay coordinate fixes, and movement/timing corrections.
- **Test coverage** — `test_agent_runtime.py`, `test_navigation.py`, `test_pi_supervisor.py` cover the new runtime, navigation, and supervisor paths.
- **Ops scripts** — `scripts/start_pokemon_server.sh`, `scripts/stop_pokemon_server.sh`, `scripts/agent_curl.sh` for running the stack and poking it from the shell.

OCR has been removed; the agent relies on Claude's vision capabilities instead.

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
- **🎨 Live dashboard** — Operator console with annotated frames, objectives, recovery, and Pi supervisor telemetry.
- **🧭 Deterministic route guidance** — Frontiers, landmarks, distance maps, route cards, and avoidances narrow choices for weaker local models.
- **🗂️ Durable semantic memory** — Append-only event memory, a session brief, and per-map failed-attempt tracking survive loops and resumes.
- **💬 Pi chat transcript** — See prompts, assistant replies, thinking, tool calls, stderr, and auto-continue scheduling in one place.
- **🎮 Multi-game** — Supports Game Boy (Pokémon Red/Blue) via PyBoy, GBA (FireRed) via PyGBA.
- **🤖 Agent-agnostic** — Works with any AI agent, RL framework, or custom script.

## Quick Start

### Installation

```bash
# Core (emulator + API server)
pip install pokemon-agent pyboy

# With dashboard (optional web GUI)
pip install pokemon-agent[dashboard] pyboy
```

> **Note:** You must provide your own ROM file. This package does not include any game ROMs.

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

The server is meant to be started by you in a terminal first. After it is running, open the dashboard and launch Pi from there.

### Start Pi From The Server Dashboard

1. Open [http://localhost:8765/dashboard](http://localhost:8765/dashboard).
2. Confirm the server is healthy and the latest frame is visible.
3. Use the Pi Supervisor panel to choose the goal/model settings.
4. Click `Start Pi`.

The dashboard will then show:

- annotated and raw frames
- current objective and turn plan
- Pi chat transcript with explicit message roles
- streamed assistant output and thinking
- tool calls, stderr, and recent events
- stuck/recovery signals
- a manual `Save Now` button

### Play from Any Agent

The Pi supervisor flow above is the preferred path. The lower-level HTTP API is still available for custom agents and scripts:

```bash
# Get game state
curl http://localhost:8765/state | python -m json.tool

# Take a screenshot
curl http://localhost:8765/screenshot -o screen.png

# Send actions
curl -X POST http://localhost:8765/action \
  -H "Content-Type: application/json" \
  -d '{"actions": ["walk_up", "walk_up", "press_a"]}'

# Save/load state
curl -X POST http://localhost:8765/save -d '{"name": "before_brock"}'
curl -X POST http://localhost:8765/load -d '{"name": "before_brock"}'
```

For vision-first agents, prefer:

```bash
curl -s -X POST http://localhost:8765/agent/observe | python3 -m json.tool
```

Then read the curated workspace files in `.agent-workspace/`:

- `latest_frame_annotated.png`
- `latest_frame.png`
- `turn_context.json`
- `turn_plan.json`
- `recovery_saves.json` when the harness says the run is risky or stuck

Ignore `.agent-workspace/debug/`. It is operator-facing debug output, not part of the model contract.

`turn_context.json` includes a compact `planning` section with the exact `observation_id`,
`objective_id`, allowed branch shapes, and valid measurable `expected_outcome` fields.

Submit plans and actions through the strict loop:

```bash
# Validate and persist one plan for the latest observation
curl -s -X POST http://localhost:8765/agent/plan \
  -H "Content-Type: application/json" \
  -d '{
        "observation_id": "copy from turn_context.json",
        "objective_id": "copy from turn_context.json",
        "intent": "Probe one tile north.",
        "mode": "overworld",
        "primary_branch": {"kind": "raw_actions", "actions": ["walk_up"]},
        "expected_outcome": {
          "summary": "Move one tile north.",
          "position_delta": {"dx": 0, "dy": -1}
        }
      }' | python3 -m json.tool

# Execute exactly one validated batch, then re-observe
curl -s -X POST http://localhost:8765/agent/act | python3 -m json.tool
```

If you launch Pi from the dashboard supervisor, the current `latest_frame_annotated.png` and
`latest_frame.png` are also attached to each turn as image inputs, so Pi can visually inspect
buildings, doors, NPCs, and other scene details instead of relying on text alone.

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
| `press_a` | Press A button (10 frames press + 20 wait) |
| `press_b` | Press B button |
| `press_start` | Press Start button |
| `press_select` | Press Select button |
| `walk_up` | Walk one tile up (16 frames + 8 wait) |
| `walk_down` | Walk one tile down |
| `walk_left` | Walk one tile left |
| `walk_right` | Walk one tile right |
| `hold_a_30` | Hold A for 30 frames |
| `wait_60` | Wait 60 frames (~1 second) |
| `a_until_dialog_end` | Press A repeatedly until dialog closes |

## Dashboard

Install with the dashboard extra to get the full operator console:

```bash
pip install pokemon-agent[dashboard]
```

Then open `http://localhost:8765/dashboard` in your browser.

The dashboard shows:
- **Annotated and raw frames** — The same images Pi is expected to inspect
- **Pi supervisor controls** — Start, continue, stop, and auto-continue configuration
- **Chat transcript** — Explicit `user`, `assistant`, `assistant thinking`, and `system` message roles
- **Tool and stderr streams** — Live visibility into what Pi is calling and what fails
- **Objective / plan / recovery state** — What the harness thinks Pi is trying to do and whether it is stuck
- **Save controls** — Create manual saves and load named or recommended recovery saves from the UI

## Supported Games

| Game | Emulator | Status | Install |
|------|----------|--------|---------|
| Pokémon Red/Blue | PyBoy | ✅ Supported | `pip install pyboy` |
| Pokémon Yellow | PyBoy | ✅ Supported | `pip install pyboy` |
| Pokémon Gold/Silver | PyBoy | 🔜 Planned | `pip install pyboy` |
| Pokémon FireRed/LeafGreen | PyGBA | 🔜 Phase 2 | `pip install pygba` |
| Pokémon Ruby/Sapphire/Emerald | PyGBA | 🔜 Phase 2 | `pip install pygba` |

## Use with Hermes Agent

[Hermes Agent](https://github.com/NousResearch/hermes-agent) has a built-in `pokemon-player` skill:

```
You: "Play Pokémon Red"
Hermes: *installs pokemon-agent, starts server, begins playing*
```

The skill teaches Hermes battle strategy, exploration patterns, team management, and how to use its persistent memory for tracking objectives across sessions.

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server info |
| `/state` | GET | Full game state JSON |
| `/screenshot` | GET | Current frame (PNG) |
| `/screenshot/base64` | GET | Current frame (base64 JSON) |
| `/agent/observe` | POST | Refresh the curated turn context and frame artifacts |
| `/agent/plan` | POST | Validate and persist one strict turn plan |
| `/agent/act` | POST | Execute the validated primary or fallback plan branch |
| `/agent/navigator` | GET | Return the best deterministic route card and alternatives |
| `/action` | POST | Execute game actions |
| `/save` | POST | Save emulator state |
| `/load` | POST | Load emulator state |
| `/saves` | GET | List saved states |
| `/minimap` | GET | ASCII minimap |
| `/navigation/map` | GET | Current live and explored navigation maps |
| `/navigation/path` | POST | Plan a route without executing it |
| `/navigation/navigate` | POST | Plan and execute a route |
| `/artifacts/{artifact}` | GET | Serve curated workspace artifacts such as frames and turn context |
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
├── __init__.py          # Package version
├── cli.py               # CLI entry point (pokemon-agent command)
├── server.py            # FastAPI game server (REST + WebSocket)
├── emulator.py          # PyBoy/PyGBA wrapper (headless)
├── pathfinding.py       # A* grid navigation
├── memory/
│   ├── reader.py        # Abstract game memory reader
│   ├── red.py           # Pokémon Red/Blue RAM parser
│   └── firered.py       # FireRed RAM parser (Phase 2)
├── state/
│   └── builder.py       # Structured state builder
└── dashboard/           # Optional [dashboard] extra
    ├── mount.py         # FastAPI static mount
    ├── history.py       # JSONL event logger
    └── static/
        ├── index.html   # Dashboard page
        ├── style.css    # Dark cyberpunk theme
        └── app.js       # WebSocket client
```

## Contributing

Contributions welcome! Areas where help is needed:

- **Pokémon Gold/Silver/Crystal** memory reader (`memory/gold.py`)
- **Pokémon FireRed** full memory reader with decryption (`memory/firered.py`)
- **Pokémon Emerald** memory reader (`memory/emerald.py`)
- **Battle AI** improvements and type matchup optimization
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
