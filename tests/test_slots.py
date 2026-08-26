"""Slot borrowing, against a fake llama.cpp that can be made to misbehave."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from pokemon_agent.slots import (
    SlotClient,
    SlotError,
    SlotLost,
    borrowed_slot,
)


class FakeServer:
    """A stand-in for llama-server's /slots endpoints."""

    def __init__(self):
        self.processing = False
        self.saved = {}
        self.calls = []
        self.save_tokens = 61608
        self.restore_tokens = None  # None means "however many were saved"
        self.fail_restore_times = 0
        self.empty_body = False
        self.last_post_body = None

    def slots(self):
        return [
            {
                "id": 0,
                "n_ctx": 140032,
                "is_processing": self.processing,
                "id_task": 0,
                "n_prompt_tokens": 61608,
                "n_prompt_tokens_cache": 0,
                "next_token": [{"n_decoded": 0}],
            }
        ]

    def save(self, filename):
        self.saved[filename] = self.save_tokens
        return {
            "id_slot": 0,
            "filename": filename,
            "n_saved": self.save_tokens,
            "n_written": self.save_tokens * 96,
            "timings": {"save_ms": 812.0},
        }

    def restore(self, filename):
        if self.fail_restore_times > 0:
            self.fail_restore_times -= 1
            raise RuntimeError("restore blew up")
        n = self.restore_tokens
        if n is None:
            n = self.saved.get(filename, 0)
        return {
            "id_slot": 0,
            "filename": filename,
            "n_restored": n,
            "n_read": n * 96,
            "timings": {"restore_ms": 640.0},
        }


@pytest.fixture
def fake():
    state = FakeServer()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, payload, code=200):
            body = b"" if state.empty_body else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            state.calls.append(("GET", self.path))
            if self.path.startswith("/slots"):
                self._send(state.slots())
            else:
                self._send({"error": "nope"}, 404)

        def do_POST(self):
            state.calls.append(("POST", self.path))
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}") if length else {}
            state.last_post_body = payload
            try:
                if "action=save" in self.path:
                    self._send(state.save(payload["filename"]))
                elif "action=restore" in self.path:
                    self._send(state.restore(payload["filename"]))
                elif "action=erase" in self.path:
                    self._send({"id_slot": 0, "n_erased": 61608})
                else:
                    self._send({"error": "unknown action"}, 400)
            except RuntimeError as exc:
                self._send({"error": str(exc)}, 500)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()


@pytest.fixture
def client(fake):
    return SlotClient(base_url=fake.url, model="qwen38-27b", slot_id=0, timeout=5)


class TestState:
    def test_it_reads_the_slot(self, client):
        state = client.state()
        assert state.id == 0
        assert state.n_ctx == 140032
        assert state.is_processing is False

    def test_the_model_goes_on_the_query_string(self, client, fake):
        client.state()
        assert any("model=qwen38-27b" in path for _, path in fake.calls)

    def test_an_empty_body_is_an_error_not_an_answer(self, client, fake):
        fake.empty_body = True
        with pytest.raises(SlotError, match="empty body"):
            client.state()

    def test_wait_idle_gives_up_rather_than_hanging(self, client, fake):
        fake.processing = True
        assert client.wait_idle(timeout=0.5, poll=0.1) is False

    def test_wait_idle_returns_once_it_settles(self, client):
        assert client.wait_idle(timeout=2, poll=0.1) is True


class TestSaveRestore:
    def test_save_reports_what_it_wrote(self, client):
        result = client.save("player.bin")
        assert result.n_saved == 61608
        assert result.megabytes == pytest.approx(61608 * 96 / 1_048_576)
        assert result.save_ms == 812.0

    def test_restore_returns_the_tokens_back(self, client):
        client.save("player.bin")
        assert client.restore("player.bin").n_restored == 61608

    def test_erase_reports_what_it_dropped(self, client):
        assert client.erase() == 61608


class TestBorrowedSlot:
    def test_the_happy_path_saves_erases_and_restores(self, client, fake):
        with borrowed_slot(client, "player.bin") as saved:
            assert saved.n_saved == 61608
        actions = [p for m, p in fake.calls if m == "POST"]
        assert any("action=save" in p for p in actions)
        assert any("action=erase" in p for p in actions)
        assert any("action=restore" in p for p in actions)

    def test_it_restores_even_when_the_body_raises(self, client, fake):
        with pytest.raises(ValueError):
            with borrowed_slot(client, "player.bin"):
                raise ValueError("subagent exploded")
        assert any("action=restore" in p for m, p in fake.calls if m == "POST")

    def test_an_empty_save_never_gives_the_slot_away(self, client, fake):
        fake.save_tokens = 0
        with pytest.raises(SlotError, match="not proceeding"):
            with borrowed_slot(client, "player.bin", allow_unsaved=False):
                pytest.fail("body must not run")
        assert not any("action=erase" in p for m, p in fake.calls if m == "POST")

    def test_a_failed_save_still_runs_the_body_and_strands_nothing(self, client, fake):
        # The save is an optimisation, not a precondition: without it the player
        # re-prefills, which is a minute of wall clock, not a lost run. Refusing
        # to intervene when the save fails would mean never intervening at all on
        # a server whose KV cache cannot be serialised.
        fake.save_tokens = 0
        ran = False
        with borrowed_slot(client, "player.bin") as saved:
            ran = True
            assert saved is None
        assert ran
        posts = [p for m, p in fake.calls if m == "POST"]
        assert not any("action=erase" in p for p in posts)
        assert not any("action=restore" in p for p in posts)

    def test_the_model_goes_in_the_post_body_not_just_the_query(self, client, fake):
        # llama-server in router mode reads ?model= on GET but not on POST.
        client.save("player.bin")
        body = fake.last_post_body
        assert body["model"] == "qwen38-27b"
        assert body["filename"] == "player.bin"

    def test_a_busy_slot_is_never_taken(self, client, fake):
        fake.processing = True
        with pytest.raises(SlotError, match="still busy"):
            with borrowed_slot(client, "player.bin", wait=0.4):
                pytest.fail("body must not run")
        assert not any("action=save" in p for m, p in fake.calls if m == "POST")

    def test_it_retries_a_failing_restore(self, client, fake):
        fake.fail_restore_times = 2
        with borrowed_slot(client, "player.bin", restore_attempts=3, backoff=0.01):
            pass
        restores = [p for m, p in fake.calls if m == "POST" and "action=restore" in p]
        assert len(restores) == 3

    def test_a_lost_context_is_raised_loudly_with_its_filename(self, client, fake):
        fake.fail_restore_times = 99
        with pytest.raises(SlotLost) as caught:
            with borrowed_slot(client, "player.bin", restore_attempts=2, backoff=0.01):
                pass
        assert caught.value.filename == "player.bin"
        assert "not loaded" in str(caught.value)

    def test_a_restore_that_returns_nothing_counts_as_lost(self, client, fake):
        fake.restore_tokens = 0
        with pytest.raises(SlotLost):
            with borrowed_slot(client, "player.bin", restore_attempts=2, backoff=0.01):
                pass
