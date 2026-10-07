"""Process-local admission and failure coordination for physical API requests."""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import deque
from dataclasses import dataclass
from threading import Lock

from open_video_summary.errors import (
    AuthenticationError, ConfigurationError, ProviderConfigurationError,
    RateLimitError, RequestDeadlineError, ServiceUnavailableError,
)
from open_video_summary.utils.retry import check_cancelled, wait_for_retry


@dataclass(frozen=True)
class RequestLimits:
    requests: float | None = None
    tokens: int | None = None
    window_seconds: float = 60.0
    learn_headers: bool = False
    recovery_seconds: float = 10.0
    server_error_threshold: int = 3
    server_error_cooldown: float = 5.0

    def __post_init__(self):
        for name in ("requests", "tokens", "window_seconds", "recovery_seconds",
                     "server_error_threshold", "server_error_cooldown"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value <= 0):
                raise ConfigurationError(f"Request limit {name} must be positive and finite.")
        if type(self.server_error_threshold) is not int:
            raise ConfigurationError("Server error threshold must be an integer.")
        if self.requests is not None and self.requests < 1:
            raise ConfigurationError("A request window must permit at least one request.")


@dataclass(frozen=True)
class Admission:
    sequence: int
    estimated_input_tokens: int
    reserved_tokens: int
    wait_seconds: float
    wait_reasons: tuple[str, ...]


def estimate_input_tokens(payload: bytes) -> int:
    """Conservatively reserve UTF-8 bytes plus envelope allowance, not exact tokens.

    The complete serialized request includes state, questions, schema and model
    settings. This intentionally overestimates common text tokenization; service
    usage remains the source for actual token counts.
    """
    return len(payload) + 64


def _headers(values):
    return {str(key).lower(): str(value) for key, value in (values or {}).items()}


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except (TypeError, ValueError):
        return None


def _reset_seconds(value):
    number = _number(value)
    if number is not None:
        return number
    if not isinstance(value, str):
        return None
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h|d)", value.strip())
    if not parts or "".join(a + b for a, b in parts) != value.strip():
        return None
    factors = {"ms": .001, "s": 1, "m": 60, "h": 3600, "d": 86400}
    result = sum(float(number) * factors[unit] for number, unit in parts)
    return result if math.isfinite(result) else None


class RequestController:
    """A shared sliding window, smooth pacing, cooldown and terminal stop.

    Reservations are retained for the entire window, including unsuccessful
    requests and retries. Headers may tighten local limits, but do not prove
    account-wide availability when other processes consume the same allowance.
    """

    def __init__(self, limits: RequestLimits, *, clock=time.monotonic):
        self.limits, self._clock = limits, clock
        self._lock = Lock()
        self._reservations = deque()
        self._pending = {}
        self._sequence = 0
        self._pause_until = self._ramp_start = self._next_send = 0.0
        self._pause_reason = None
        self._failure_times = deque()
        self._terminal = None
        self._requests, self._tokens = limits.requests, limits.tokens
        self._header_requests = self._header_tokens = None
        self._request_reset = self._token_reset = 0.0
        self._bootstrap_done = not limits.learn_headers
        self._virtual_offset = 0.0

    def _now(self):
        return self._clock() + self._virtual_offset

    def wait(self, delay, sleep=time.sleep, cancel_event=None, deadline=None):
        """Wait interruptibly; custom sleep hooks support deterministic test clocks."""
        check_cancelled(cancel_event)
        now = self._now()
        if deadline is not None and (now >= deadline or now + delay >= deadline):
            raise RequestDeadlineError("The request operation deadline cannot accommodate the required wait.")
        before = self._clock()
        if sleep is time.sleep:
            end = before + delay
            while True:
                check_cancelled(cancel_event)
                with self._lock:
                    self._raise_terminal()
                remaining = end - self._clock()
                if remaining <= 0:
                    break
                wait_for_retry(min(.1, remaining), sleep, cancel_event)
        else:
            wait_for_retry(delay, sleep, cancel_event)
        # Existing offline adapters inject recording/no-op sleep hooks. Treat
        # their missing elapsed time as virtual time, without changing real waits.
        if sleep is not time.sleep and self._clock is time.monotonic:
            with self._lock:
                self._virtual_offset += max(0.0, delay - (self._clock() - before))

    def deadline(self, seconds):
        return self._now() + seconds

    def remaining(self, deadline):
        remaining = deadline - self._now()
        if remaining <= 0:
            raise RequestDeadlineError("The request operation deadline was reached.")
        return remaining

    def _raise_terminal(self):
        if self._terminal is not None:
            kind, message = self._terminal
            raise kind(message)

    def admit(self, estimated_input_tokens, output_tokens=0, *, deadline,
              sleep=time.sleep, cancel_event=None):
        reserved = estimated_input_tokens + output_tokens
        waited, reasons = 0.0, []
        while True:
            check_cancelled(cancel_event)
            with self._lock:
                self._raise_terminal()
                now = self._now()
                if now >= deadline:
                    raise RequestDeadlineError("The request operation deadline was reached before sending.")
                window = self.limits.window_seconds
                while self._reservations and self._reservations[0][0] <= now - window:
                    self._reservations.popleft()
                if now >= self._request_reset:
                    self._header_requests = None
                if now >= self._token_reset:
                    self._header_tokens = None
                if self._tokens is not None and reserved > self._tokens:
                    raise ConfigurationError("Estimated request tokens exceed the configured or learned token window; adjust the output budget or token limit.")
                waits = []
                if now < self._pause_until:
                    waits.append((self._pause_until - now, self._pause_reason or "shared_cooldown"))
                if now < self._next_send:
                    waits.append((self._next_send - now, "request_pacing"))
                if self._requests is not None and len(self._reservations) + 1 > self._requests:
                    waits.append((self._reservations[0][0] + window - now, "request_window"))
                token_sum = sum(item[2] for item in self._reservations)
                if self._tokens is not None and token_sum + reserved > self._tokens:
                    excess = token_sum + reserved - self._tokens
                    for sent_at, _, tokens in self._reservations:
                        excess -= tokens
                        if excess <= 0:
                            waits.append((sent_at + window - now, "token_window"))
                            break
                if self._header_requests is not None and self._header_requests < 1:
                    waits.append((self._request_reset - now, "header_request_remaining"))
                if self._header_tokens is not None and self._header_tokens < reserved:
                    waits.append((self._token_reset - now, "header_token_remaining"))
                if not waits:
                    self._sequence += 1
                    sequence = self._sequence
                    self._reservations.append((now, sequence, reserved))
                    self._pending[sequence] = reserved
                    if self._header_requests is not None:
                        self._header_requests -= 1
                    if self._header_tokens is not None:
                        self._header_tokens -= reserved
                    spacing = window / self._requests if self._requests else 0.0
                    if not self._bootstrap_done and self._requests is None:
                        # A local 4/s discovery pace leaves slow in-flight calls
                        # isolated. It is not an assumed account RPM allowance.
                        spacing = max(spacing, .25)
                    if self._ramp_start and now < self._ramp_start + self.limits.recovery_seconds:
                        fraction = max(0.0, (now - self._ramp_start) / self.limits.recovery_seconds)
                        spacing = max(spacing, .25 * (1.0 - fraction))
                    self._next_send = now + spacing
                    return Admission(sequence, estimated_input_tokens, reserved, waited, tuple(reasons))
                delay, reason = max(waits)
                for _, waiting_reason in waits:
                    if waiting_reason not in reasons:
                        reasons.append(waiting_reason)
                delay = max(.001, delay)
            self.wait(delay, sleep, cancel_event, deadline)
            waited += delay

    def complete(self, admission, *, headers=None, error=None, cooldown=0.0):
        """Publish service feedback before another retry or worker can send."""
        with self._lock:
            now = self._now()
            if now >= self._request_reset:
                self._header_requests = None
            if now >= self._token_reset:
                self._header_tokens = None
            sequence = admission.sequence if admission else -1
            self._pending.pop(sequence, None)
            values = _headers(headers)
            if self.limits.learn_headers:
                for resource in ("requests", "tokens", "project-tokens"):
                    observed = _number(values.get(f"x-ratelimit-limit-{resource}"))
                    attr = "_requests" if resource == "requests" else "_tokens"
                    current = getattr(self, attr)
                    if observed is not None and observed >= 1:
                        conservative = max(1, math.floor(observed * .9))
                        setattr(self, attr, min(current, conservative) if current else conservative)
                    remaining = _number(values.get(f"x-ratelimit-remaining-{resource}"))
                    if remaining is not None:
                        reset = _reset_seconds(values.get(f"x-ratelimit-reset-{resource}"))
                        reset_at = now + (reset if reset is not None else self.limits.window_seconds)
                        # Headers can arrive out of order. Reserve all currently
                        # in-flight work again rather than assuming which sends
                        # the provider already included in this response.
                        later = sum(self._pending.values())
                        if resource == "requests":
                            later = len(self._pending)
                        allowance = max(0, remaining - later)
                        rem_attr = "_header_requests" if resource == "requests" else "_header_tokens"
                        reset_attr = "_request_reset" if resource == "requests" else "_token_reset"
                        prior = getattr(self, rem_attr)
                        setattr(self, rem_attr, min(prior, allowance) if prior is not None else allowance)
                        setattr(self, reset_attr, max(getattr(self, reset_attr), reset_at))
                self._bootstrap_done = self._requests is not None
            if isinstance(error, (AuthenticationError, ProviderConfigurationError)):
                self._terminal = type(error), str(error)
            elif isinstance(error, RateLimitError):
                self._pause_until = max(self._pause_until, now + cooldown)
                self._pause_reason = "rate_limit_cooldown"
                self._ramp_start = self._pause_until
                self._next_send = max(self._next_send, self._pause_until)
            elif isinstance(error, ServiceUnavailableError):
                while self._failure_times and self._failure_times[0] < now - 30:
                    self._failure_times.popleft()
                self._failure_times.append(now)
                if len(self._failure_times) >= self.limits.server_error_threshold:
                    self._pause_until = max(self._pause_until, now + max(cooldown, self.limits.server_error_cooldown))
                    self._pause_reason = "service_error_cooldown"
                    self._ramp_start = self._pause_until
                    self._next_send = max(self._next_send, self._pause_until)

    def snapshot(self):
        with self._lock:
            return {"request_limit": self._requests, "token_limit": self._tokens,
                    "window_seconds": self.limits.window_seconds,
                    "header_learning": self.limits.learn_headers,
                    "terminal_error": self._terminal[0].__name__ if self._terminal else None,
                    "scope": "process-local explicitly shared request scope"}


class RequestScope:
    """Explicit scope shared by related adapters and all their worker forks."""

    def __init__(self):
        self._controllers, self._lock = {}, Lock()

    def controller(self, config, limits):
        credential = hashlib.sha256((config.api_key or "").encode()).digest()
        key = (config.provider, config.base_url.rstrip("/"), credential,
               getattr(config, "organization", None), getattr(config, "project", None),
               getattr(config, "limit_group", None) or config.model)
        with self._lock:
            if key not in self._controllers:
                self._controllers[key] = RequestController(limits)
            controller = self._controllers[key]
            if controller.limits.window_seconds != limits.window_seconds:
                raise ConfigurationError("Adapters sharing a limit group must use the same request window.")
            # A second role in the same group can only tighten known ceilings.
            with controller._lock:
                for attr, value in (("_requests", limits.requests), ("_tokens", limits.tokens)):
                    current = getattr(controller, attr)
                    if value is not None:
                        setattr(controller, attr, min(current, value) if current else value)
            return controller
