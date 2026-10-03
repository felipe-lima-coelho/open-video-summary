"""Provider-independent requests and results used by the research pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ServiceMetadata:
    """Requested, sent and service-reported settings without research content."""

    provider: str
    requested_model: str
    returned_model: str | None = None
    requested_reasoning_effort: str | None = None
    sent_reasoning_effort: str | None = None
    reported_reasoning_effort: str | None = None
    requested_language: str | None = None
    reported_language: str | None = None
    duration_seconds: float | None = None
    audio_duration_seconds: float | None = None
    adapter_version: str = "1"
    sdk_version: str | None = None
    attempts: int = 1
    status: str = "completed"
    temperature_sent: float | None = None


@dataclass(frozen=True)
class OutputSpec:
    """The domain's expected output, independent of vendor response formats."""

    kind: Literal["text", "topics", "topic", "string_list", "answers", "pattern"] = (
        "text"
    )
    topic_ids: tuple[str, ...] = ()
    max_items: int | None = None
    pattern: str | None = None
    answer_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    output: OutputSpec = field(default_factory=OutputSpec)
    temperature: float | None = 0.2
    images: tuple[bytes, ...] = ()


@dataclass(frozen=True)
class GenerationResult:
    text: str
    value: Any
    metadata: ServiceMetadata


class ResponseInterpreter(Protocol):
    def interpret(self, text: str, spec: OutputSpec) -> Any: ...


class LanguageModel(Protocol):
    records: list[ServiceMetadata]

    def preflight(self) -> None: ...

    def generate(self, request: GenerationRequest) -> GenerationResult: ...


@dataclass(frozen=True)
class TimedWord:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class TranscriptSegment:
    """Optional native utterance boundaries retained by the local adapter."""

    text: str
    start: float
    end: float
    words: tuple[TimedWord, ...] = ()


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    words: tuple[TimedWord, ...]
    language: str | None = None
    segments: tuple[TranscriptSegment, ...] = ()
    metadata: ServiceMetadata | None = None


class SpeechToText(Protocol):
    records: list[ServiceMetadata]

    def preflight(self) -> None: ...

    def transcribe(
        self, media_path: str | Path, language: str | None = None
    ) -> TranscriptionResult: ...
