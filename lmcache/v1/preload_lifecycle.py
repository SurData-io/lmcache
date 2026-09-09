# SPDX-License-Identifier: Apache-2.0
"""Versioned preload lifecycle protocol and drain coordinator.

SurData (or any L2 connector) implements PreloadResourceProvider. LMCache owns
the request FIFO and the drain loop. Connector never stores waiters.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Deque, Dict, Optional, Protocol

logger = logging.getLogger("lmcache.preload_lifecycle")

PRELOAD_LIFECYCLE_ABI = 1


@dataclass
class PreloadWork:
    lookup_id: str
    generation: int
    keys: list
    n_keys: int
    backend_name: str = ""
    valid: bool = True
    granted_keys: int = 0
    reservation: Any = None
    io_started: bool = False
    enqueued_mono: float = field(default_factory=time.monotonic)
    grant_event: Optional[asyncio.Event] = None


class PreloadResourceProvider(Protocol):
    PRELOAD_LIFECYCLE_ABI: int

    def try_reserve_prefix(
        self, n_keys: int, lookup_id: str = "", generation: int = 0
    ) -> tuple[int, Any]:
        """Return (granted_key_count, reservation). granted==0 means DRAM full."""

    def add_slot_freed_listener(self, cb: Callable[[], None]) -> None: ...

    def drop_reservation(self, reservation: Any) -> None: ...


class PreloadLifecycleCoordinator:
    """Upstream FIFO + event-driven drain. One pending drain task at a time."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.provider: Optional[PreloadResourceProvider] = None
        self._q: Deque[PreloadWork] = deque()
        self._gens: Dict[str, int] = {}
        self._by_id: Dict[tuple[str, int], PreloadWork] = {}
        self._drain_scheduled = False
        self._closed = False
        self.enqueued = 0
        self.invalidated = 0
        self.stale_dropped = 0
        self.dram_full_pending = 0
        self.drain_wakes = 0
        self.h2d_complete = 0
        self.slot_freed = 0
        self.stall_alarms = 0
        self._empty_drains = 0
        self._last_resident_progress = time.monotonic()
        self.stall_timeout_s = 600.0

    def attach_provider(self, provider: Optional[PreloadResourceProvider]) -> None:
        if provider is None:
            self.provider = None
            return
        if provider is self.provider:
            return
        abi = int(getattr(provider, "PRELOAD_LIFECYCLE_ABI", 0) or 0)
        if abi < PRELOAD_LIFECYCLE_ABI:
            raise RuntimeError(
                f"preload lifecycle ABI {abi} < required {PRELOAD_LIFECYCLE_ABI}"
            )
        self.provider = provider
        provider.add_slot_freed_listener(self.on_slot_freed)

    def next_generation(self, lookup_id: str) -> int:
        gen = self._gens.get(lookup_id, 0) + 1
        self._gens[lookup_id] = gen
        return gen

    def on_enqueued(self, work: PreloadWork) -> None:
        self.enqueued += 1
        self._q.append(work)
        self._by_id[(work.lookup_id, work.generation)] = work
        self.request_drain()

    def on_invalidated(self, lookup_id: str) -> None:
        self.invalidated += 1
        found = False
        for (lid, gen), work in list(self._by_id.items()):
            if lid != lookup_id:
                continue
            found = True
            work.valid = False
            if (
                work.reservation is not None
                and self.provider is not None
                and not work.io_started
            ):
                self.provider.drop_reservation(work.reservation)
                work.reservation = None
            if work.grant_event is not None and not work.grant_event.is_set():
                work.grant_event.set()
        if found:
            self._request_drain_threadsafe()

    def _request_drain_threadsafe(self) -> None:
        try:
            if self.loop.is_closed():
                return
            self.loop.call_soon_threadsafe(self.request_drain)
        except Exception:
            logger.debug("preload drain kick failed", exc_info=True)

    def on_h2d_complete(self, lookup_id: str = "") -> None:
        del lookup_id
        self.h2d_complete += 1
        self._last_resident_progress = time.monotonic()
        self._empty_drains = 0

    def on_slot_freed(self) -> None:
        self.slot_freed += 1
        self._last_resident_progress = time.monotonic()
        self._empty_drains = 0
        self._request_drain_threadsafe()

    def request_drain(self) -> None:
        if self._closed or self._drain_scheduled:
            return
        self._drain_scheduled = True
        self.drain_wakes += 1
        self.loop.create_task(self._drain())

    async def _drain(self) -> None:
        self._drain_scheduled = False
        progressed = False
        while self._q:
            work = self._q[0]
            if not work.valid:
                self._q.popleft()
                self._by_id.pop((work.lookup_id, work.generation), None)
                self.stale_dropped += 1
                if work.grant_event is not None:
                    work.grant_event.set()
                progressed = True
                continue
            if self.provider is None:
                work.granted_keys = work.n_keys
                self._q.popleft()
                if work.grant_event is not None:
                    work.grant_event.set()
                progressed = True
                continue
            granted, reservation = self.provider.try_reserve_prefix(
                work.n_keys,
                lookup_id=work.lookup_id,
                generation=work.generation,
            )
            if granted < 1:
                self.dram_full_pending += 1
                now = time.monotonic()
                if now - self._last_resident_progress >= self.stall_timeout_s:
                    self.stall_alarms += 1
                    self._last_resident_progress = now
                    logger.warning(
                        "preload drain stall lookup_id=%s queue=%s pending=%s",
                        work.lookup_id,
                        len(self._q),
                        self.dram_full_pending,
                    )
                break
            work.granted_keys = granted
            work.reservation = reservation
            self._q.popleft()
            if work.grant_event is not None:
                work.grant_event.set()
            progressed = True
        if not progressed:
            self._empty_drains += 1

    async def run_when_reserved(
        self,
        work: PreloadWork,
        get_coro_factory: Callable[[PreloadWork], Awaitable[Any]],
    ) -> Any:
        if self._closed or not work.valid:
            return []
        if work.grant_event is None:
            work.grant_event = asyncio.Event()
        self.on_enqueued(work)
        await work.grant_event.wait()
        if not work.valid or work.granted_keys < 1:
            if work.reservation is not None and self.provider is not None:
                self.provider.drop_reservation(work.reservation)
                work.reservation = None
            return []
        try:
            work.io_started = True
            return await get_coro_factory(work)
        except asyncio.CancelledError:
            self.on_invalidated(work.lookup_id)
            raise
        finally:
            self._by_id.pop((work.lookup_id, work.generation), None)

    def stats(self) -> dict:
        return {
            "preload_lifecycle_abi": PRELOAD_LIFECYCLE_ABI,
            "preload_fifo_depth": len(self._q),
            "preload_enqueued": self.enqueued,
            "preload_invalidated": self.invalidated,
            "preload_stale_dropped": self.stale_dropped,
            "preload_dram_full_pending": self.dram_full_pending,
            "preload_drain_wakes": self.drain_wakes,
            "preload_h2d_complete": self.h2d_complete,
            "preload_slot_freed": self.slot_freed,
            "preload_stall_alarms": self.stall_alarms,
            "preload_provider": 0 if self.provider is None else 1,
        }

    def close(self) -> None:
        self._closed = True
        logger.info(
            "PRELOAD_LIFECYCLE_STATS %s",
            " ".join(f"{k}={v}" for k, v in sorted(self.stats().items())),
        )
        while self._q:
            work = self._q.popleft()
            work.valid = False
            if work.grant_event is not None:
                work.grant_event.set()


def find_provider(storage_backends) -> Optional[PreloadResourceProvider]:
    for backend in storage_backends.values():
        conn = getattr(backend, "connection", None)
        for obj in (conn, backend):
            if obj is None:
                continue
            provider = getattr(obj, "preload_resource_provider", None)
            if provider is None:
                provider = obj if hasattr(obj, "try_reserve_prefix") else None
            if provider is not None and hasattr(provider, "try_reserve_prefix"):
                return provider
    return None
