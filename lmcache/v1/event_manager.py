# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import Enum, auto
from typing import Any
import asyncio
import threading

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class EventType(Enum):
    LOADING = auto()


class EventStatus(Enum):
    ONGOING = auto()
    DONE = auto()
    NOT_FOUND = auto()


def release_prefetch_result(memory_objs: Any) -> None:
    """Unpin and drop refs for nested [(key, MemoryObj), ...] prefetch results."""
    if not memory_objs:
        return
    try:
        memory_objs_flat = [mm for m in memory_objs for mm in m]
    except TypeError:
        logger.error("prefetch result is not a nested sequence: %s", type(memory_objs))
        return
    for item in memory_objs_flat:
        try:
            _key, memory_obj = item
        except (TypeError, ValueError):
            continue
        if memory_obj is None:
            continue
        try:
            if getattr(memory_obj, "is_pinned", False):
                memory_obj.unpin()
            get_ref_count = getattr(memory_obj, "get_ref_count", None)
            if get_ref_count is None or get_ref_count() > 0:
                memory_obj.ref_count_down()
        except Exception as e:
            logger.error("Error releasing prefetch memory object: %s", e)


class EventManager:
    """
    A thread-safe event manager to manage asynchronous events.
    Each event is identified by its type and a unique id.
    Events are organized by status for efficient counting.
    """

    def __init__(self) -> None:
        # Guard by lock
        # Structure: events[event_type][event_status][event_id] = future
        self.events: dict[EventType, dict[EventStatus, dict[str, asyncio.Future]]] = {
            et: {es: {} for es in EventStatus} for et in EventType
        }
        self.lock = threading.Lock()
        # Abort flags live under the same lock as event maps. Keyed by
        # (event_type, event_id); entries exist only while LOADING is ONGOING
        # after abort, and are dropped on claim/pop.
        self._aborted: set[tuple[EventType, str]] = set()

    def add_event(
        self,
        event_type: EventType,
        event_id: str,
        future: asyncio.Future,
    ) -> None:
        """
        Add an event with the given type and id.
        """
        with self.lock:
            status_dict = self.events.get(event_type, None)
            assert status_dict is not None, (
                f"Invalid event type {event_type} in EventManager."
            )
            status_dict[EventStatus.ONGOING][event_id] = future

    def pop_event(
        self,
        event_type: EventType,
        event_id: str,
    ) -> asyncio.Future:
        """
        Pop and return the event with the given type and id.
        """
        with self.lock:
            status_dict = self.events.get(event_type, None)
            assert status_dict is not None, (
                f"Invalid event type {event_type} in EventManager."
            )
            done_events = status_dict[EventStatus.DONE]
            assert event_id in done_events, (
                f"Event {event_id} of type {event_type} is not done or not found."
            )
            self._aborted.discard((event_type, event_id))
            return done_events.pop(event_id)

    def mark_lookup_aborted(
        self,
        event_type: EventType,
        event_id: str,
    ) -> bool:
        """Record abort for an in-flight lookup, or signal DONE for sync cleanup.

        Returns True only when the event is already DONE so the caller can
        claim and release immediately. ONGOING records the abort flag and
        returns False (prefetch callback will release). NOT_FOUND returns
        False without recording a flag (zero-hit / already reclaimed).
        """
        with self.lock:
            status_dict = self.events.get(event_type, None)
            assert status_dict is not None, (
                f"Invalid event type {event_type} in EventManager."
            )
            key = (event_type, event_id)
            if event_id in status_dict[EventStatus.DONE]:
                self._aborted.add(key)
                return True
            if event_id in status_dict[EventStatus.ONGOING]:
                self._aborted.add(key)
                return False
            return False

    def claim_done_event(
        self,
        event_type: EventType,
        event_id: str,
        only_if_aborted: bool = False,
    ) -> asyncio.Future | None:
        """Atomically pop a DONE event. Returns None if missing or still ONGOING.

        If ``only_if_aborted`` is True, pop only when this id was marked
        aborted. An abort flag on an ONGOING event is left in place.
        """
        with self.lock:
            key = (event_type, event_id)
            if only_if_aborted and key not in self._aborted:
                return None
            status_dict = self.events.get(event_type, None)
            if status_dict is None:
                return None
            future = status_dict[EventStatus.DONE].pop(event_id, None)
            if future is not None:
                self._aborted.discard(key)
            return future

    def update_event_status(
        self,
        event_type: EventType,
        event_id: str,
        status: EventStatus,
    ) -> None:
        """
        Update the status of the event with the given type and id.
        """
        with self.lock:
            status_dict = self.events.get(event_type, None)
            assert status_dict is not None, (
                f"Invalid event type {event_type} in EventManager."
            )
            # Find the event in any status dict
            event = None
            for s in EventStatus:
                if event_id in status_dict[s]:
                    event = status_dict[s].pop(event_id)
                    break

            if event is None:
                raise KeyError(f"Event {event_id} of type {event_type} not found.")

            # Move to new status dict
            status_dict[status][event_id] = event

    def get_event_status(
        self,
        event_type: EventType,
        event_id: str,
    ) -> EventStatus:
        """
        Get the status of the event with the given type and id.
        """
        with self.lock:
            status_dict = self.events.get(event_type, None)
            assert status_dict is not None, (
                f"Invalid event type {event_type} in EventManager."
            )
            for status in EventStatus:
                if event_id in status_dict[status]:
                    return status
            return EventStatus.NOT_FOUND

    def get_events_count_by_status(
        self,
        event_type: EventType,
        status: EventStatus,
    ) -> int:
        """
        Get the count of events for the given event type and status.
        This is a lightweight O(1) operation using dict length.
        """
        with self.lock:
            status_dict = self.events.get(event_type, None)
            if status_dict is None:
                return 0
            return len(status_dict[status])

    def get_event_future(
        self,
        event_type: EventType,
        event_id: str,
    ) -> asyncio.Future:
        """
        Pop and return the event with the given type and id.
        """
        with self.lock:
            status_dict = self.events.get(event_type, None)
            assert status_dict is not None, (
                f"Invalid event type {event_type} in EventManager."
            )
            done_events = status_dict[EventStatus.DONE]
            assert event_id in done_events, (
                f"Event {event_id} of type {event_type} is not done or not found."
            )
            return done_events[event_id]

    def try_get_done_future(
        self,
        event_type: EventType,
        event_id: str,
    ) -> asyncio.Future | None:
        """Return the DONE future if it exists and is complete. Never wait."""
        with self.lock:
            status_dict = self.events.get(event_type, None)
            if status_dict is None:
                return None
            future = status_dict[EventStatus.DONE].get(event_id)
            if future is None or not future.done():
                return None
            return future
