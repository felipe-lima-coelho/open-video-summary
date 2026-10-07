"""Finite retry delays shared by hosted generation and evaluation adapters."""

import math
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from open_video_summary.errors import RequestCancelledError


def check_cancelled(cancel_event):
    signal = getattr(cancel_event, "raise_if_set", None)
    if callable(signal):
        signal()
        return
    if cancel_event is not None and cancel_event.is_set():
        raise RequestCancelledError("The analysis was interrupted; no new request was sent.")


def wait_for_retry(delay, sleep, cancel_event=None):
    check_cancelled(cancel_event)
    if cancel_event is not None and sleep is time.sleep:
        cancel_event.wait(delay)
    else:
        sleep(delay)
    check_cancelled(cancel_event)


def retry_after(headers, cap=None, *, now=None):
    """Parse the server's minimum delay without shortening it to a local cap.

    ``cap`` is retained for source compatibility but deliberately does not clip
    an advertised delay. The operation deadline decides whether waiting fits.
    """
    value = next((value for key, value in (headers or {}).items()
                  if str(key).lower() == "retry-after"), None)
    if value is None:
        milliseconds = next((value for key, value in (headers or {}).items()
                             if str(key).lower() == "retry-after-ms"), None)
        try:
            seconds = float(milliseconds) / 1000
            return max(0.0, seconds) if math.isfinite(seconds) else None
        except (TypeError, ValueError):
            return None
    if value is None:
        return None
    try:
        seconds = float(value)
        if math.isfinite(seconds):
            return max(0.0, seconds)
    except (TypeError, ValueError):
        pass
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return max(0.0, (date.astimezone(timezone.utc)
                         - (now or datetime.now(timezone.utc))).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def retry_delay(attempt, base, *, backoff_cap, retry_after_seconds=None,
                jitter=random.uniform):
    backoff = min(backoff_cap, base * (2 ** (attempt - 1)))
    # Add at most 25% jitter without shortening an advertised server delay.
    delay = min(backoff_cap, backoff + jitter(0.0, backoff * 0.25))
    return max(delay, retry_after_seconds or 0.0)
