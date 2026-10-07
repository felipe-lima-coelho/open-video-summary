"""Stdlib HTTP adapter for TypeSafe's non-generative evaluator API."""

from __future__ import annotations

import json
import math
import random
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from collections.abc import Mapping
from typing import Callable
from urllib.parse import urlsplit

from open_video_summary.contracts import (
    ChoiceResult,
    EvaluationMetadata,
    EvaluationResult,
    NoulResult,
    ProviderProgress,
)
from open_video_summary.errors import (
    AuthenticationError,
    ConfigurationError,
    InvalidResponseError,
    ProviderError,
    ProviderConfigurationError,
    RateLimitError,
    RequestCancelledError,
    RequestDeadlineError,
    RunStoppedError,
    ServiceTimeoutError,
    ServiceUnavailableError,
)
from open_video_summary.utils.progress import notify
from open_video_summary.utils.retry import check_cancelled, retry_after, retry_delay
from open_video_summary.utils.request_control import RequestLimits, RequestScope, estimate_input_tokens

DEFAULT_TYPESAFE_MODEL = "jev-1.13.0"
_ENDPOINT_PATH = "/v1/systemone"
_MAX_BACKOFF_SECONDS = 30.0
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
    operation_timeout_seconds: float = 90.0
    request_limit: float = 60.0
    token_limit: int = 80000
    rate_window_seconds: float = 1.0
    limit_group: str | None = None
    provider: str = field(default="typesafe", init=False)

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
        if not _is_finite_number(self.operation_timeout_seconds) or self.operation_timeout_seconds <= 0:
            raise ConfigurationError("TypeSafe operation_timeout_seconds must be positive and finite.")
        RequestLimits(requests=self.request_limit, tokens=self.token_limit,
                      window_seconds=self.rate_window_seconds)


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
        *, progress=None, jitter=random.uniform, request_scope=None,
    ) -> None:
        self.config = config
        self.transport = transport or _stdlib_transport
        self.sleep = sleep
        self.progress, self.jitter = progress, jitter
        self.cancel_event = None
        self.can_fork = transport is None
        self.records: list[EvaluationMetadata] = []
        self.set_request_scope(request_scope or RequestScope())

    def set_request_scope(self, scope):
        self.request_scope = scope
        self.controller = scope.controller(self.config, RequestLimits(
            requests=self.config.request_limit, tokens=self.config.token_limit,
            window_seconds=self.config.rate_window_seconds))

    def fork(self):
        """Keep worker metadata and transport ownership separate."""
        if not self.can_fork:
            raise ConfigurationError("An injected evaluator transport cannot be shared by workers.")
        return type(self)(self.config, sleep=self.sleep, jitter=self.jitter,
                          request_scope=self.request_scope)

    def _record(self, metadata):
        self.records.append(metadata)
        success = metadata.status == "success"
        notify(self.progress, ProviderProgress(
            "attempt_completed" if success else "attempt_failed",
            self.config.provider, metadata.attempts, self.config.max_attempts,
            metadata.duration_seconds, error_type=None if success else metadata.status))

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
        choice: dict[str, tuple[str, tuple[str, ...] | Mapping[str, str]]] | None = None,
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
        deadline = self.controller.deadline(self.config.operation_timeout_seconds)
        estimated = estimate_input_tokens(request_body)
        self._estimated_input = estimated
        previous_error = None
        pending_retry_wait = 0.0
        for attempt in range(1, self.config.max_attempts + 1):
            check_cancelled(self.cancel_event)
            request_started = time.monotonic()
            admission, sent, response_headers = None, False, None
            self._retry_wait, pending_retry_wait = pending_retry_wait, 0.0
            notify(self.progress, ProviderProgress("attempt_started", self.config.provider,
                attempt, self.config.max_attempts))
            try:
                admission = self.controller.admit(estimated, deadline=deadline,
                    sleep=self.sleep, cancel_event=self.cancel_event)
                timeout = min(self.config.timeout_seconds, self.controller.remaining(deadline))
                check_cancelled(self.cancel_event)
                sent = True
                response = self.transport(url, headers=headers.copy(), body=request_body, timeout=timeout)
                if not isinstance(response, TransportResponse):
                    raise InvalidResponseError("TypeSafe transport returned an invalid response.")
                if (type(response.status_code) is not int or not 100 <= response.status_code <= 599
                        or not isinstance(response.headers, Mapping)
                        or any(not isinstance(key, str) or not isinstance(value, str)
                               for key, value in response.headers.items())
                        or not isinstance(response.body, bytes)):
                    raise InvalidResponseError("TypeSafe returned an invalid HTTP response.")
                response_headers = response.headers
                self.controller.remaining(deadline)
                if not 200 <= response.status_code < 300:
                    raise self._http_error(response.status_code, response.body)
                answers, returned_model, input_tokens, output_tokens = self._parse_response(
                    response, noul_ids, choice_options)
                self.controller.complete(admission, headers=response_headers)
                metadata = EvaluationMetadata(requested_model=self.config.model,
                    returned_model=returned_model, duration_seconds=max(0.0, time.monotonic() - started),
                    attempts=attempt, status="success", input_tokens=input_tokens, output_tokens=output_tokens,
                    provider=self.config.provider,
                    **self._attempt_fields(admission, sent, previous_error))
                self._record(metadata)
                return EvaluationResult(tuple(answers[key] for key in noul_ids),
                    tuple(answers[key] for key in choice_options), metadata)
            except Exception as exc:
                if isinstance(exc, ProviderError):
                    error = exc
                elif isinstance(exc, (TimeoutError, socket.timeout)) or (
                        isinstance(exc, urllib.error.URLError)
                        and isinstance(exc.reason, (TimeoutError, socket.timeout))):
                    error = ServiceTimeoutError("TypeSafe request timed out.")
                else:
                    error = ServiceUnavailableError("TypeSafe could not be reached.")
                delay = retry_delay(attempt, self.config.retry_backoff_seconds,
                    backoff_cap=_MAX_BACKOFF_SECONDS,
                    retry_after_seconds=retry_after(response_headers), jitter=self.jitter)
                self.controller.complete(admission, headers=response_headers, error=error, cooldown=delay)
                self._record(replace(self._failed_metadata(attempt,
                    time.monotonic() - request_started, error),
                    **self._attempt_fields(admission, sent, type(error).__name__)))
                # Invalid structured responses were never retried by this adapter.
                retryable = isinstance(error, (RateLimitError, ServiceUnavailableError, ServiceTimeoutError))
                if not retryable or attempt == self.config.max_attempts:
                    raise error from None
                notify(self.progress, ProviderProgress("retry_scheduled", self.config.provider,
                    attempt, self.config.max_attempts, delay_seconds=delay, error_type=type(error).__name__))
                previous_error = type(error).__name__
                if not isinstance(error, RateLimitError):
                    self.controller.wait(delay, self.sleep, self.cancel_event, deadline)
                    pending_retry_wait = delay
        raise ServiceUnavailableError("TypeSafe request failed.")

    def _attempt_fields(self, admission, sent, retry_reason):
        limits = self.controller.snapshot()
        return dict(request_sent=sent,
            estimated_input_tokens=admission.estimated_input_tokens if admission else self._estimated_input,
            reserved_tokens=admission.reserved_tokens if admission else self._estimated_input,
            wait_seconds=(admission.wait_seconds if admission else 0.0) + self._retry_wait,
            wait_reasons=(admission.wait_reasons if admission else ())
                        + (("individual_retry_backoff",) if self._retry_wait else ()), retry_reason=retry_reason,
            operation_timeout_seconds=self.config.operation_timeout_seconds,
            request_limit=limits["request_limit"], token_limit=limits["token_limit"],
            rate_window_seconds=limits["window_seconds"])

    def _build_questions(
        self,
        context: str,
        noul: dict[str, str] | None,
        choice: dict[str, tuple[str, tuple[str, ...] | Mapping[str, str]]] | None,
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
            if isinstance(options, Mapping):
                criteria = dict(options)
                options = tuple(criteria)
                if any(
                    not isinstance(description, str) or not description.strip()
                    for description in criteria.values()
                ):
                    raise ValueError("TypeSafe choice descriptions must be nonempty strings.")
            else:
                criteria = None
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
            if criteria is None:
                criteria = {option: option for option in options}
            questions[identifier] = {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
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

    def _http_error(self, status_code: int, body: bytes = b"") -> ProviderError:
        if status_code in {401, 403}:
            return AuthenticationError("TypeSafe authentication failed.")
        if status_code == 429:
            try:
                detail = json.loads(body)
                detail = detail.get("error", detail) if isinstance(detail, dict) else None
                code = detail.get("code") if isinstance(detail, dict) else None
            except (ValueError, TypeError):
                code = None
            if isinstance(code, str) and code in {"insufficient_quota", "billing_hard_limit_reached", "quota_exceeded",
                        "billing_limit_exceeded", "usage_limit_reached"}:
                return ProviderConfigurationError("TypeSafe quota or billing limit was reached.")
            return RateLimitError("TypeSafe request limit was reached.")
        if status_code == 408:
            return ServiceTimeoutError("TypeSafe request timed out.")
        if status_code == 404:
            return ProviderConfigurationError("TypeSafe endpoint or model was not found (HTTP 404).")
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
        elif isinstance(error, RequestDeadlineError):
            status = "deadline_exceeded"
        elif isinstance(error, RequestCancelledError):
            status = "cancelled"
        elif isinstance(error, RunStoppedError):
            status = "run_stopped"
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
            provider=self.config.provider,
        )

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
    return retry_after(headers)
