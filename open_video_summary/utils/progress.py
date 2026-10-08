"""Best-effort progress observers and a shared, optional waiting monitor."""

import atexit
import time
from _thread import start_new_thread
from contextlib import contextmanager
from threading import Condition, Event


def notify(observer, event):
    if observer is not None:
        try:
            observer(event)
        except Exception:
            # Observation must not change research decisions or service retries.
            pass


class _Heartbeat:
    def __init__(self, observer, event_factory, interval):
        self.observer, self.event_factory, self.interval = observer, event_factory, interval
        self.next_tick = time.monotonic() + interval
        self.cancelled = Event()

    def emit(self):
        if self.cancelled.is_set():
            return
        try:
            event = self.event_factory()
            if not self.cancelled.is_set():
                notify(self.observer, event)
        except Exception:
            # Building an observation is as optional as delivering it.
            pass


class _HeartbeatMonitor:
    """Reuse one daemon for all scopes without waiting for thread startup.

    Thread.start() waits indefinitely for its Python bootstrap. A bootstrap
    MemoryError can therefore prevent the provider callback from ever running.
    The low-level starter returns without that handshake. If the daemon never
    starts, or later fails, only optional waiting events are lost.
    """

    def __init__(self, *, start_thread=start_new_thread):
        self._start_thread = start_thread
        self._condition = Condition()
        self._entries = set()
        self._started = self._closed = False

    def register(self, observer, event_factory, interval):
        entry = _Heartbeat(observer, event_factory, interval)
        with self._condition:
            if self._closed:
                return None
            self._entries.add(entry)
            if not self._started:
                self._started = True
                try:
                    self._start_thread(self._run, ())
                except Exception:
                    self._close_locked()
                    return None
            self._condition.notify()
        return entry

    def unregister(self, entry):
        # Do not join the monitor or wait for a slow observer. A callback that
        # was already executing may finish after this scope has closed.
        entry.cancelled.set()
        with self._condition:
            self._entries.discard(entry)
            self._condition.notify()

    def _close_locked(self):
        self._closed = True
        for entry in self._entries:
            entry.cancelled.set()
        self._entries.clear()
        self._condition.notify_all()

    def close(self):
        with self._condition:
            self._close_locked()

    def _run(self):
        try:
            while True:
                with self._condition:
                    if self._closed:
                        return
                    if not self._entries:
                        # Remain available while idle; closing the last scope
                        # cannot race a new scope into losing its monitor.
                        self._condition.wait()
                        continue
                    now = time.monotonic()
                    due = [entry for entry in self._entries if entry.next_tick <= now]
                    if not due:
                        self._condition.wait(min(entry.next_tick for entry in self._entries) - now)
                        continue
                    for entry in due:
                        entry.next_tick = now + entry.interval
                # User callbacks never hold the registration lock.
                for entry in due:
                    entry.emit()
                # An idle process-wide monitor must not retain the completed
                # analysis through the previous tick's callback closures.
                del entry, due
        except BaseException:
            # A dead monitor must never become a dependency of research work.
            self.close()


_MONITOR = _HeartbeatMonitor()
atexit.register(_MONITOR.close)


@contextmanager
def heartbeat(observer, event_factory, interval):
    monitor = _MONITOR
    entry = None
    if observer is not None:
        try:
            entry = monitor.register(observer, event_factory, interval)
        except Exception:
            pass
    try:
        yield
    finally:
        if entry is not None:
            try:
                monitor.unregister(entry)
            except Exception:
                pass
