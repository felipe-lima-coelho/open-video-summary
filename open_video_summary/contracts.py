"""Provider-independent requests and results used by the research pipeline."""

from dataclasses import dataclass, field
from collections.abc import Mapping
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
    sent_language: str | None = None
    reported_language: str | None = None
    duration_seconds: float | None = None
    audio_duration_seconds: float | None = None
    adapter_version: str = "1"
    sdk_version: str | None = None
    attempts: int = 1
    status: str = "completed"
    temperature_sent: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_sent: bool = False
    estimated_input_tokens: int | None = None
    reserved_tokens: int | None = None
    wait_seconds: float = 0.0
    wait_reasons: tuple[str, ...] = ()
    retry_reason: str | None = None
    max_output_tokens: int | None = None
    operation_timeout_seconds: float | None = None
    request_limit: float | None = None
    token_limit: int | None = None
    rate_window_seconds: float | None = None


@dataclass(frozen=True)
class ProviderProgress:
    event: str
    provider: str
    attempt: int
    max_attempts: int
    elapsed_seconds: float = 0.0
    delay_seconds: float | None = None
    error_type: str | None = None


@dataclass(frozen=True)
class OutputSpec:
    """The domain's expected output, independent of vendor response formats."""

    kind: Literal[
        "text", "topics", "topic", "string_list", "answers", "pattern",
        "information_units", "information_qa", "information_foci",
    ] = (
        "text"
    )
    topic_ids: tuple[str, ...] = ()
    max_items: int | None = None
    pattern: str | None = None
    answer_ids: tuple[str, ...] = ()
    segment_ids: tuple[str, ...] = ()


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
class NoulResult:
    id: str
    probability: float
    confidence: float | None = None


@dataclass(frozen=True)
class ChoiceResult:
    id: str
    selected: str
    probabilities: tuple[tuple[str, float], ...] = ()
    confidence: float | None = None


@dataclass(frozen=True)
class EvaluationMetadata:
    requested_model: str
    returned_model: str | None
    duration_seconds: float
    attempts: int
    status: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    sdk_version: str | None = None
    provider: str = "unknown"
    adapter_version: str = "1"
    request_sent: bool = False
    estimated_input_tokens: int | None = None
    reserved_tokens: int | None = None
    wait_seconds: float = 0.0
    wait_reasons: tuple[str, ...] = ()
    retry_reason: str | None = None
    operation_timeout_seconds: float | None = None
    request_limit: float | None = None
    token_limit: int | None = None
    rate_window_seconds: float | None = None
    base_estimated_input_tokens: int | None = None
    estimation_multiplier: float = 1.0
    estimation_multiplier_after: float = 1.0
    estimation_adjusted: bool = False
    accounted_tokens: int | None = None
    token_adjustment: int = 0
    usage_status: str = "unknown"
    token_request_overhead: int = 0
    token_question_overhead: int = 0
    question_count: int | None = None


@dataclass(frozen=True)
class EvaluationResult:
    noul: tuple[NoulResult, ...]
    choice: tuple[ChoiceResult, ...]
    metadata: EvaluationMetadata


class Evaluator(Protocol):
    """Typed probability decisions, separate from generative language models."""

    records: list[EvaluationMetadata]

    def preflight(self) -> None: ...

    def evaluate(
        self,
        context: str,
        noul: dict[str, str] | None = None,
        choice: dict[str, tuple[str, tuple[str, ...] | Mapping[str, str]]] | None = None,
    ) -> EvaluationResult: ...


@dataclass(frozen=True)
class TimedWord:
    text: str
    start: float
    end: float
    # The source separator before the next timed word, including whitespace.
    separator_after: str = " "


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
