"""Stdlib HTTP adapter for TypeSafe's non-generative evaluator API."""

from __future__ import annotations

import json
import math
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from collections.abc import Mapping
from typing import Callable
from urllib.parse import urlsplit

from open_video_summary.errors import (
    AuthenticationError,
    ConfigurationError,
    InvalidResponseError,
    ProviderError,
    RateLimitError,
    ServiceTimeoutError,
    ServiceUnavailableError,
)

DEFAULT_TYPESAFE_MODEL = "jev-1.13.0"
_ENDPOINT_PATH = "/v1/systemone"
_MAX_RETRY_AFTER_SECONDS = 30.0
_PROBABILITY_SUM_TOLERANCE = 0.02


@dataclass(frozen=True)
class TypeSafeConfig:
    """Configuration for TypeSafe's HTTP API.

    The default model is pinned to a version. A caller can explicitly select a
    different model by setting ``model``. Credentials stay out of repr output
    and are required only when ``preflight`` or ``evaluate`` is called.
    """

    api_key: str | None = field(default=None, repr=False)
    model: str = DEFAULT_TYPESAFE_MODEL
    base_url: str = "https://api.typesafe.ai"
    timeout_seconds: float = 30.0
    max_attempts: int = 2
    retry_backoff_seconds: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ConfigurationError("TypeSafe model must be a nonempty string.")
        if not isinstance(self.base_url, str):
            raise ConfigurationError("TypeSafe base URL must be an HTTPS endpoint.")
        try:
            endpoint = urlsplit(self.base_url)
        except ValueError:
            raise ConfigurationError(
                "TypeSafe base URL must be a valid HTTPS endpoint."
            ) from None
        local_hosts = {"localhost", "127.0.0.1", "::1"}
        if (
            endpoint.scheme not in {"https", "http"}
            or not endpoint.hostname
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.query
            or endpoint.fragment
            or (
                endpoint.scheme != "https"
                and endpoint.hostname.lower() not in local_hosts
            )
        ):
            raise ConfigurationError(
                "TypeSafe base URL must use HTTPS and must not contain credentials."
            )
        try:
            endpoint.port
        except ValueError:
            raise ConfigurationError("TypeSafe base URL has an invalid port.") from None
        if isinstance(self.max_attempts, bool) or not isinstance(
            self.max_attempts, int
        ):
            raise ConfigurationError(
                "TypeSafe max_attempts must be an integer from 1 to 5."
            )
        if not 1 <= self.max_attempts <= 5:
            raise ConfigurationError(
                "TypeSafe max_attempts must be an integer from 1 to 5."
            )
        if not _is_finite_number(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ConfigurationError(
                "TypeSafe timeout_seconds must be a positive finite number."
            )
        if (
            not _is_finite_number(self.retry_backoff_seconds)
            or self.retry_backoff_seconds < 0
        ):
            raise ConfigurationError(
                "TypeSafe retry_backoff_seconds must be a nonnegative finite number."
            )
        if self.api_key is not None and not isinstance(self.api_key, str):
            raise ConfigurationError(
                "TypeSafe api_key must be a string when configured."
            )


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
    provider: str = field(default="typesafe", init=False)
    adapter_version: str = field(default="1", init=False)


@dataclass(frozen=True)
class EvaluationResult:
    noul: tuple[NoulResult, ...]
    choice: tuple[ChoiceResult, ...]
    metadata: EvaluationMetadata


@dataclass(frozen=True)
class TransportResponse:
    """Response value used by injected transports.

    An offline transport is a callable with signature
    ``transport(url, *, headers, body, timeout) -> TransportResponse``. The
    request body is UTF-8 JSON bytes and response ``body`` is bytes. This small
    boundary lets callers test the real serializer and response validator
    without a network request or additional dependency.
    """

    status_code: int
    headers: Mapping[str, str]
    body: bytes


Transport = Callable[..., TransportResponse]


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _stdlib_transport(
    url: str,
    *,
    headers: Mapping[str, str],
    body: bytes,
    timeout: float,
) -> TransportResponse:
    request = urllib.request.Request(
        url, data=body, headers=dict(headers), method="POST"
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        # HTTPError is also the response object for non-2xx statuses. Read its
        # body here without surfacing it in errors or metadata.
        return TransportResponse(
            status_code=error.code,
            headers=dict(error.headers.items()) if error.headers else {},
            body=error.read(),
        )
    with response:
        return TransportResponse(
            status_code=response.status,
            headers=dict(response.headers.items()),
            body=response.read(),
        )


class TypeSafeEvaluator:
    """Evaluate boolean and choice questions with TypeSafe System One."""

    def __init__(
        self,
        config: TypeSafeConfig,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.transport = transport or _stdlib_transport
        self.sleep = sleep
        self.records: list[EvaluationMetadata] = []

    def preflight(self) -> None:
        """Validate local settings and credentials without contacting the API."""
        # TypeSafeConfig validates endpoint and retry settings at construction.
        if self.config.api_key is None or not self.config.api_key.strip():
            raise AuthenticationError("TypeSafe API key is not configured.")
        if self.config.api_key != self.config.api_key.strip():
            raise ConfigurationError("TypeSafe API key has surrounding whitespace.")
        if any(
            ord(character) < 32 or ord(character) == 127
            for character in self.config.api_key
        ):
            raise ConfigurationError(
                "TypeSafe API key contains invalid control characters."
            )

    def evaluate(
        self,
        context: str,
        noul: dict[str, str] | None = None,
        choice: dict[str, tuple[str, tuple[str, ...]]] | None = None,
    ) -> EvaluationResult:
        self.preflight()
        question_specs, noul_ids, choice_options = self._build_questions(
            context, noul, choice
        )
        request_body = json.dumps(
            {
                "state": context,
                "model": self.config.model,
                "questions": question_specs,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        url = self.config.base_url.rstrip("/") + _ENDPOINT_PATH
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        started = time.monotonic()

        for attempt in range(1, self.config.max_attempts + 1):
            request_started = time.monotonic()
            try:
                response = self.transport(
                    url,
                    headers=headers.copy(),
                    body=request_body,
                    timeout=self.config.timeout_seconds,
                )
            except (TimeoutError, socket.timeout) as exc:
                error = ServiceTimeoutError("TypeSafe request timed out.")
                should_retry = True
                retry_after = None
                metadata = self._failed_metadata(
                    attempt, time.monotonic() - request_started, error
                )
                self.records.append(metadata)
                if attempt < self.config.max_attempts and should_retry:
                    self._wait_before_retry(attempt, retry_after)
                    continue
                raise error from None
            except urllib.error.URLError as exc:
                if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                    error = ServiceTimeoutError("TypeSafe request timed out.")
                else:
                    error = ServiceUnavailableError("TypeSafe could not be reached.")
                metadata = self._failed_metadata(
                    attempt, time.monotonic() - request_started, error
                )
                self.records.append(metadata)
                if attempt < self.config.max_attempts:
                    self._wait_before_retry(attempt, None)
                    continue
                raise error from None
            except OSError as exc:
                error = ServiceUnavailableError("TypeSafe could not be reached.")
                metadata = self._failed_metadata(
                    attempt, time.monotonic() - request_started, error
                )
                self.records.append(metadata)
                if attempt < self.config.max_attempts:
                    self._wait_before_retry(attempt, None)
                    continue
                raise error from None
            except Exception as exc:
                # Injected transports may wrap low-level failures differently.
                # Do not retain or expose their exception text.
                error = ServiceUnavailableError("TypeSafe request failed.")
                metadata = self._failed_metadata(
                    attempt, time.monotonic() - request_started, error
                )
                self.records.append(metadata)
                if attempt < self.config.max_attempts:
                    self._wait_before_retry(attempt, None)
                    continue
                raise error from None

            if not isinstance(response, TransportResponse):
                error = InvalidResponseError(
                    "TypeSafe transport returned an invalid response."
                )
                self.records.append(
                    self._failed_metadata(
                        attempt, time.monotonic() - request_started, error
                    )
                )
                raise error from None
            if (
                not isinstance(response.status_code, int)
                or isinstance(response.status_code, bool)
                or not 100 <= response.status_code <= 599
                or not isinstance(response.headers, Mapping)
                or any(
                    not isinstance(key, str) or not isinstance(value, str)
                    for key, value in response.headers.items()
                )
                or not isinstance(response.body, bytes)
            ):
                error = InvalidResponseError(
                    "TypeSafe returned an invalid HTTP response."
                )
                self.records.append(
                    self._failed_metadata(
                        attempt, time.monotonic() - request_started, error
                    )
                )
                raise error from None
            if 200 <= response.status_code < 300:
                try:
                    answers, returned_model, input_tokens, output_tokens = (
                        self._parse_response(response, noul_ids, choice_options)
                    )
                except InvalidResponseError as error:
                    self.records.append(
                        self._failed_metadata(
                            attempt, time.monotonic() - request_started, error
                        )
                    )
                    raise error from None
                metadata = EvaluationMetadata(
                    requested_model=self.config.model,
                    returned_model=returned_model,
                    duration_seconds=max(0.0, time.monotonic() - started),
                    attempts=attempt,
                    status="success",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                self.records.append(metadata)
                return EvaluationResult(
                    noul=tuple(answers[key] for key in noul_ids),
                    choice=tuple(answers[key] for key in choice_options),
                    metadata=metadata,
                )

            error = self._http_error(response.status_code)
            self.records.append(
                self._failed_metadata(
                    attempt, time.monotonic() - request_started, error
                )
            )
            retryable = isinstance(
                error,
                (RateLimitError, ServiceUnavailableError, ServiceTimeoutError),
            )
            if attempt < self.config.max_attempts and retryable:
                self._wait_before_retry(attempt, _retry_after(response.headers))
                continue
            raise error from None

        # The loop always returns or raises, but keep static type checkers clear.
        raise ServiceUnavailableError("TypeSafe request failed.")

    def _build_questions(
        self,
        context: str,
        noul: dict[str, str] | None,
        choice: dict[str, tuple[str, tuple[str, ...]]] | None,
    ) -> tuple[dict[str, dict], tuple[str, ...], dict[str, tuple[str, ...]]]:
        if not isinstance(context, str) or not context.strip():
            raise ValueError("TypeSafe evaluation context must be a nonempty string.")
        noul = {} if noul is None else noul
        choice = {} if choice is None else choice
        if not isinstance(noul, dict) or not isinstance(choice, dict):
            raise ValueError("TypeSafe questions must be dictionaries.")
        if not noul and not choice:
            raise ValueError("At least one TypeSafe question is required.")
        if set(noul).intersection(choice):
            raise ValueError("TypeSafe question identifiers must be unique.")

        questions: dict[str, dict] = {}
        noul_ids = tuple(noul)
        choice_options: dict[str, tuple[str, ...]] = {}
        for identifier, instructions in noul.items():
            _validate_identifier(identifier)
            if not isinstance(instructions, str) or not instructions.strip():
                raise ValueError(
                    "TypeSafe question instructions must be nonempty strings."
                )
            questions[identifier] = {"type": "noul", "instructions": instructions}

        for identifier, specification in choice.items():
            _validate_identifier(identifier)
            if not isinstance(specification, tuple) or len(specification) != 2:
                raise ValueError(
                    "TypeSafe choice questions require instructions and options."
                )
            instructions, options = specification
            if not isinstance(instructions, str) or not instructions.strip():
                raise ValueError(
                    "TypeSafe question instructions must be nonempty strings."
                )
            if (
                not isinstance(options, tuple)
                or not 1 <= len(options) <= 255
                or any(
                    not isinstance(option, str) or not option.strip()
                    for option in options
                )
                or len(set(options)) != len(options)
            ):
                raise ValueError(
                    "TypeSafe choice options must be 1 to 255 unique strings."
                )
            questions[identifier] = {
                "type": "choice",
                "instructions": instructions,
                "criteria": {option: option for option in options},
            }
            choice_options[identifier] = options
        return questions, noul_ids, choice_options

    def _parse_response(
        self,
        response: TransportResponse,
        noul_ids: tuple[str, ...],
        choice_options: Mapping[str, tuple[str, ...]],
    ) -> tuple[
        dict[str, NoulResult | ChoiceResult], str | None, int | None, int | None
    ]:
        if not isinstance(response.body, bytes):
            raise InvalidResponseError("TypeSafe returned a nonbinary response body.")
        try:
            document = json.loads(
                response.body.decode("utf-8"), object_pairs_hook=_unique_object_pairs
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise InvalidResponseError("TypeSafe returned malformed JSON.") from None
        if not isinstance(document, dict):
            raise InvalidResponseError(
                "TypeSafe returned an invalid response envelope."
            )
        answers = document.get("answers")
        expected_ids = set(noul_ids).union(choice_options)
        if not isinstance(answers, dict) or set(answers) != expected_ids:
            raise InvalidResponseError(
                "TypeSafe returned missing or unexpected question IDs."
            )

        normalized: dict[str, NoulResult | ChoiceResult] = {}
        noul_set = set(noul_ids)
        for identifier, answer in answers.items():
            if not isinstance(answer, dict):
                raise InvalidResponseError("TypeSafe returned an invalid answer.")
            if identifier in noul_set:
                value = answer.get("noul")
                if answer.get("type") != "noul" or not _valid_probability(value):
                    raise InvalidResponseError(
                        "TypeSafe returned an invalid boolean score."
                    )
                normalized[identifier] = NoulResult(identifier, float(value))
                continue

            options = choice_options[identifier]
            selected = answer.get("choice")
            probabilities = answer.get("probabilities")
            confidence = answer.get("confidence")
            if answer.get("type") != "choice" or selected not in options:
                raise InvalidResponseError(
                    "TypeSafe selected an unknown choice option."
                )
            if not isinstance(probabilities, dict) or set(probabilities) != set(
                options
            ):
                raise InvalidResponseError(
                    "TypeSafe returned invalid choice probabilities."
                )
            if any(not _valid_probability(probabilities[key]) for key in options):
                raise InvalidResponseError(
                    "TypeSafe returned invalid choice probabilities."
                )
            total = sum(float(probabilities[key]) for key in options)
            if not math.isclose(
                total, 1.0, rel_tol=0.0, abs_tol=_PROBABILITY_SUM_TOLERANCE
            ):
                raise InvalidResponseError(
                    "TypeSafe choice probabilities do not sum to one."
                )
            if confidence is not None and not _valid_probability(confidence):
                raise InvalidResponseError(
                    "TypeSafe returned invalid choice confidence."
                )
            normalized[identifier] = ChoiceResult(
                identifier,
                selected,
                tuple((key, float(probabilities[key])) for key in options),
                None if confidence is None else float(confidence),
            )

        returned_model = document.get("model")
        if returned_model is not None and (
            not isinstance(returned_model, str) or not returned_model.strip()
        ):
            raise InvalidResponseError("TypeSafe returned an invalid model identifier.")
        input_tokens, output_tokens = _read_usage(document.get("usage"))
        return normalized, returned_model, input_tokens, output_tokens

    def _http_error(self, status_code: int) -> ProviderError:
        if status_code in {401, 403}:
            return AuthenticationError("TypeSafe authentication failed.")
        if status_code == 429:
            return RateLimitError("TypeSafe request limit was reached.")
        if status_code == 408:
            return ServiceTimeoutError("TypeSafe request timed out.")
        if status_code == 529 or 500 <= status_code <= 599:
            return ServiceUnavailableError("TypeSafe is temporarily unavailable.")
        if 400 <= status_code <= 499:
            return ConfigurationError(
                f"TypeSafe rejected the request (HTTP {status_code})."
            )
        return InvalidResponseError("TypeSafe returned an unexpected HTTP status.")

    def _failed_metadata(
        self, attempt: int, duration: float, error: ProviderError
    ) -> EvaluationMetadata:
        if isinstance(error, AuthenticationError):
            status = "authentication_error"
        elif isinstance(error, RateLimitError):
            status = "rate_limited"
        elif isinstance(error, ServiceTimeoutError):
            status = "timeout"
        elif isinstance(error, ServiceUnavailableError):
            status = "service_unavailable"
        elif isinstance(error, ConfigurationError):
            status = "request_rejected"
        else:
            status = "invalid_response"
        return EvaluationMetadata(
            requested_model=self.config.model,
            returned_model=None,
            duration_seconds=max(0.0, duration),
            attempts=attempt,
            status=status,
        )

    def _wait_before_retry(self, attempt: int, retry_after: float | None) -> None:
        fallback = min(
            _MAX_RETRY_AFTER_SECONDS,
            self.config.retry_backoff_seconds * (2 ** (attempt - 1)),
        )
        delay = (
            fallback
            if retry_after is None
            else min(_MAX_RETRY_AFTER_SECONDS, max(0.0, retry_after))
        )
        self.sleep(delay)


def _validate_identifier(value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("TypeSafe question identifiers must be nonempty strings.")


def _valid_probability(value: object) -> bool:
    return _is_finite_number(value) and 0.0 <= value <= 1.0


def _unique_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field.")
        result[key] = value
    return result


def _read_usage(value: object) -> tuple[int | None, int | None]:
    if value is None:
        return None, None
    if not isinstance(value, dict):
        raise InvalidResponseError("TypeSafe returned invalid usage metadata.")
    values: list[int | None] = []
    for field_name in ("input_tokens", "output_tokens"):
        count = value.get(field_name)
        if count is None:
            values.append(None)
        elif isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            values.append(count)
        else:
            raise InvalidResponseError("TypeSafe returned invalid usage metadata.")
    return values[0], values[1]


def _retry_after(headers: Mapping[str, str]) -> float | None:
    value = next(
        (v for key, v in headers.items() if key.lower() == "retry-after"), None
    )
    if value is None:
        return None
    try:
        seconds = float(value)
        if math.isfinite(seconds):
            return min(_MAX_RETRY_AFTER_SECONDS, max(0.0, seconds))
    except (TypeError, ValueError):
        pass
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return min(
            _MAX_RETRY_AFTER_SECONDS,
            max(0.0, (date.astimezone(timezone.utc) - now).total_seconds()),
        )
    except (TypeError, ValueError, OverflowError):
        return None
