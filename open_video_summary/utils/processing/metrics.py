"""Per-evaluation visual timings without retaining frames or descriptors."""

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class VisualProfile:
    stage_seconds: Counter = field(default_factory=Counter)
    counters: Counter = field(default_factory=Counter)
    total_seconds: float = 0.0
    status: str = "failed"

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "total_seconds": self.total_seconds,
            "stage_seconds": dict(self.stage_seconds),
            "counters": dict(self.counters),
        }


_profile: ContextVar[VisualProfile | None] = ContextVar("visual_profile", default=None)


@dataclass
class _Stage:
    profile: VisualProfile
    child_seconds: float = 0.0


_active_stage: ContextVar[_Stage | None] = ContextVar("visual_stage", default=None)


@contextmanager
def collect_visual_profile(profile: VisualProfile):
    """Collect exclusive substage times in the current evaluation only."""
    token = _profile.set(profile)
    started = perf_counter()
    try:
        yield
        profile.status = "completed"
    finally:
        profile.total_seconds = perf_counter() - started
        _profile.reset(token)


@contextmanager
def visual_stage(name: str):
    profile = _profile.get()
    if profile is None:
        yield
        return
    parent = _active_stage.get()
    stage = _Stage(profile)
    token = _active_stage.set(stage)
    started = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - started
        profile.stage_seconds[name] += elapsed - stage.child_seconds
        if parent is not None and parent.profile is profile:
            parent.child_seconds += elapsed
        _active_stage.reset(token)


def visual_count(name: str, amount: int = 1) -> None:
    profile = _profile.get()
    if profile is not None:
        profile.counters[name] += amount


def visual_maximum(name: str, value: int) -> None:
    profile = _profile.get()
    if profile is not None:
        profile.counters[name] = max(profile.counters[name], value)
