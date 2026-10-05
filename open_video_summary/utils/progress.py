"""Best-effort progress observers and a scoped waiting heartbeat."""

from contextlib import contextmanager
from threading import Event, Thread


def notify(observer, event):
    if observer is not None:
        try:
            observer(event)
        except Exception:
            # Observation must not change research decisions or service retries.
            pass


@contextmanager
def heartbeat(observer, event_factory, interval):
    if observer is None:
        yield
        return
    stop = Event()

    def waiting():
        while not stop.wait(interval):
            notify(observer, event_factory())

    thread = Thread(target=waiting, name="ovs-information-progress", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
