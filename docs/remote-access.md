# Remote access over SSH

The server binds `0.0.0.0:8765`, but the intended way to reach it from another
machine is an SSH tunnel — nothing has to be exposed on the LAN and no extra
service is involved.

## 1. Open the tunnel (on your laptop)

```bash
ssh -N -L 8765:127.0.0.1:8765 dev@192.168.1.98
```

or, from a checkout of this repo:

```bash
scripts/tunnel.sh dev@192.168.1.98        # forward 8765 -> 8765
scripts/tunnel.sh dev@192.168.1.98 8765 9000   # remote 8765 -> local 9000
```

Leave it running. Ctrl-C closes it.

Nothing needs configuring on the server side: `sshd` runs on port 22 with the
default `AllowTcpForwarding yes`, so local forwarding works with the keys already
in `~/.ssh/authorized_keys`.

## 2. Look at the game

Everything below is served by the same port, so it all comes through the tunnel:

| URL | What it is |
|-----|------------|
| `http://localhost:8765/dashboard` | Full operator console — frames, objective, Pi transcript |
| `http://localhost:8765/artifacts/live_frame_annotated` | Live frame with the tile grid / objective overlay (PNG) |
| `http://localhost:8765/artifacts/live_frame` | Live frame, raw 160x144 (PNG) |
| `http://localhost:8765/artifacts/latest_frame_annotated` | Frame from the last `POST /action`, annotated — the image Pi is looking at |
| `http://localhost:8765/artifacts/latest_frame` | Frame from the last action, raw |
| `http://localhost:8765/screenshot` | Current frame straight from the emulator (PNG) |
| `http://localhost:8765/artifacts/turn_context_json` | Turn context snapshot, for the dashboard and debugging |
| `ws://localhost:8765/ws` | Live event stream |

The dashboard is the thing to open normally. Note the artifact URLs have **no
file extension** — the key is the artifact name, not the filename.

### Just the frames, no browser

Poll the live frame into a local file and let your image viewer follow it:

```bash
while true; do
  curl -s http://localhost:8765/artifacts/live_frame_annotated -o /tmp/poke.png
  sleep 1
done
```

or grab a single frame:

```bash
curl -s http://localhost:8765/screenshot -o shot.png && open shot.png
```

## Forwarding more than one port

If you also want the llama.cpp/router UI on the model host, add another `-L`:

```bash
ssh -N -L 8765:127.0.0.1:8765 -L 8080:192.168.1.183:8080 dev@192.168.1.98
```

## Troubleshooting

- **Connection refused through the tunnel** — the tunnel is up but the server is
  not. Check on the host: `curl -s localhost:8765/health`.
- **`bind: Address already in use`** — a previous tunnel is still open, or
  something local owns 8765. Pick a different local port (third argument to
  `scripts/tunnel.sh`).
- **404 on an artifact** — that artifact has not been produced yet. `latest_frame*`
  only exists after the first `POST /action`; `live_frame*` only when the
  realtime clock is running (it is, unless the server was started with
  `--no-realtime`).
