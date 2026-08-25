"""Borrow the model's single KV slot, then give it back.

The box runs llama-server in router mode with ``--parallel 1``, so there is
exactly one slot and the player's whole context lives in it. To run a thinking
session we have to take that slot away and hand it back afterwards:

    save slot 0 to disk  ->  run the subagent in the freed slot  ->  restore

Without this the player pays a full re-prefill of its entire context every time
we interrupt it. With it we pay two KV round-trips instead.

The dangerous half is the giving back. Between the erase and the restore the
player's context exists only as a file, so :func:`borrowed_slot` refuses to
proceed unless the save is confirmed, and it shouts rather than shrugging if
the restore fails. Losing that file loses the run's memory.

Server side this needs ``--slots`` and ``--slot-save-path``, both of which the
qwen38-27b preset already sets. Requests go to the router, which needs a
``model`` query parameter to know which instance is meant.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

DEFAULT_BASE_URL = "http://127.0.0.1:8090"
DEFAULT_SLOT_ID = 0
DEFAULT_TIMEOUT = 120.0

#: A 140k-token context is gigabytes of quantised KV. Saving it is not instant,
#: and the restore has to be given room to finish or we strand the player.
SAVE_TIMEOUT = 600.0


class SlotError(RuntimeError):
    """The slot API refused, or answered with something unusable."""


class SlotLost(SlotError):
    """A restore failed after the slot had already been given away.

    This is the bad one. The player's context is on disk under the filename in
    :attr:`filename` and is not in the model. Nothing recovers automatically.
    """

    def __init__(self, message: str, filename: str) -> None:
        super().__init__(message)
        self.filename = filename


@dataclass(frozen=True)
class SlotState:
    id: int
    n_ctx: int
    is_processing: bool
    id_task: int
    n_prompt_tokens: int
    n_prompt_tokens_cache: int

    @classmethod
    def from_dict(cls, payload: dict) -> "SlotState":
        return cls(
            id=int(payload.get("id", 0)),
            n_ctx=int(payload.get("n_ctx", 0)),
            is_processing=bool(payload.get("is_processing", False)),
            id_task=int(payload.get("id_task", -1)),
            n_prompt_tokens=int(payload.get("n_prompt_tokens", 0)),
            n_prompt_tokens_cache=int(payload.get("n_prompt_tokens_cache", 0)),
        )


@dataclass(frozen=True)
class SaveResult:
    filename: str
    n_saved: int
    n_written: int
    save_ms: float

    @property
    def megabytes(self) -> float:
        return self.n_written / 1_048_576


@dataclass(frozen=True)
class RestoreResult:
    filename: str
    n_restored: int
    n_read: int
    restore_ms: float


class SlotClient:
    """Thin client over llama.cpp's ``/slots`` endpoints, via the router."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "",
        slot_id: int = DEFAULT_SLOT_ID,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.slot_id = slot_id
        self.timeout = timeout

    def _url(self, path: str, **params: str) -> str:
        if self.model:
            params = {"model": self.model, **params}
        query = urllib.parse.urlencode(params)
        return f"{self.base_url}{path}" + (f"?{query}" if query else "")

    def _request(
        self,
        url: str,
        *,
        method: str = "GET",
        payload: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        body = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"} if body else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:400]
            raise SlotError(f"{method} {url} -> {error.code}: {detail}") from error
        except OSError as error:
            raise SlotError(f"{method} {url} unreachable: {error}") from error
        if not raw.strip():
            # The router answers empty while it is swapping models. That is not
            # an answer we can act on, and treating it as one has already led to
            # a wrong conclusion once.
            raise SlotError(f"{method} {url} returned an empty body")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as error:
            raise SlotError(f"{method} {url} returned non-JSON: {raw[:200]!r}") from error

    def state(self) -> SlotState:
        payload = self._request(self._url("/slots"))
        if not isinstance(payload, list) or not payload:
            raise SlotError(f"/slots returned no slots: {payload!r}")
        for entry in payload:
            if int(entry.get("id", -1)) == self.slot_id:
                return SlotState.from_dict(entry)
        raise SlotError(f"no slot with id {self.slot_id}")

    def wait_idle(self, timeout: float = 300.0, poll: float = 2.0) -> bool:
        """Block until the slot stops generating. False if it never does."""

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if not self.state().is_processing:
                    return True
            except SlotError:
                pass  # a swapping router is briefly unreachable; keep waiting
            time.sleep(poll)
        return False

    def save(self, filename: str) -> SaveResult:
        payload = self._request(
            self._url(f"/slots/{self.slot_id}", action="save"),
            method="POST",
            payload={"filename": filename},
            timeout=SAVE_TIMEOUT,
        )
        return SaveResult(
            filename=payload.get("filename", filename),
            n_saved=int(payload.get("n_saved", 0)),
            n_written=int(payload.get("n_written", 0)),
            save_ms=float((payload.get("timings") or {}).get("save_ms", 0.0)),
        )

    def restore(self, filename: str) -> RestoreResult:
        payload = self._request(
            self._url(f"/slots/{self.slot_id}", action="restore"),
            method="POST",
            payload={"filename": filename},
            timeout=SAVE_TIMEOUT,
        )
        return RestoreResult(
            filename=payload.get("filename", filename),
            n_restored=int(payload.get("n_restored", 0)),
            n_read=int(payload.get("n_read", 0)),
            restore_ms=float((payload.get("timings") or {}).get("restore_ms", 0.0)),
        )

    def erase(self) -> int:
        payload = self._request(
            self._url(f"/slots/{self.slot_id}", action="erase"),
            method="POST",
        )
        return int(payload.get("n_erased", 0))


@contextmanager
def borrowed_slot(
    client: SlotClient,
    filename: str,
    *,
    wait: float = 300.0,
    restore_attempts: int = 3,
    backoff: float = 2.0,
) -> Iterator[SaveResult]:
    """Take the slot, run the body, put the context back.

    Refuses to hand the slot over unless the save is confirmed non-empty, so a
    failed save costs an intervention rather than the run. Restores on the way
    out whether the body succeeded or raised, and retries before giving up.
    """

    if not client.wait_idle(timeout=wait):
        raise SlotError(f"slot {client.slot_id} still busy after {wait:.0f}s")

    saved = client.save(filename)
    if saved.n_saved <= 0:
        raise SlotError(
            f"save of slot {client.slot_id} wrote {saved.n_saved} tokens; not proceeding"
        )

    client.erase()
    try:
        yield saved
    finally:
        last: Optional[Exception] = None
        for attempt in range(restore_attempts):
            try:
                restored = client.restore(filename)
                if restored.n_restored > 0:
                    break
                last = SlotError(f"restore returned {restored.n_restored} tokens")
            except SlotError as error:
                last = error
            time.sleep(backoff * (attempt + 1))
        else:
            raise SlotLost(
                f"could not restore slot {client.slot_id} from {filename!r} after "
                f"{restore_attempts} attempts ({last}). The context is on disk under "
                f"the slot-save path and is not loaded.",
                filename,
            )


__all__ = [
    "SlotClient",
    "SlotState",
    "SaveResult",
    "RestoreResult",
    "SlotError",
    "SlotLost",
    "borrowed_slot",
    "DEFAULT_BASE_URL",
    "DEFAULT_SLOT_ID",
]
