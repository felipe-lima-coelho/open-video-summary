"""LLM adapters translate vendor APIs and validate results inside the ACL."""

import base64
import ast
import json
import math
import re
import time
from abc import ABC, abstractmethod
from ast import literal_eval
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from open_video_summary.contracts import (
    GenerationRequest,
    GenerationResult,
    OutputSpec,
    ServiceMetadata,
)
from open_video_summary.errors import (
    AuthenticationError,
    ConfigurationError,
    InvalidResponseError,
    ProviderError,
    RateLimitError,
    ServiceTimeoutError,
    ServiceUnavailableError,
)
from open_video_summary.utils.providers import LLMConfig


def _value(obj, key: str, default=None):
    return (
        obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
    )


def _sdk_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _external_error(exc: Exception, provider: str) -> ProviderError:
    """Do not include service messages, request bodies or SDK exceptions."""
    if isinstance(exc, ProviderError):
        return exc
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    status = status or getattr(response, "status_code", None)
    name = type(exc).__name__
    body = getattr(exc, "body", None)
    detail = _value(body, "error", body)
    code = _value(detail, "code")
    if status in {401, 403} or name in {"AuthenticationError", "PermissionDeniedError"}:
        return AuthenticationError(f"{provider} authentication or permission failed.")
    if status == 429 or name == "RateLimitError":
        if code in {"insufficient_quota", "billing_hard_limit_reached"}:
            return ConfigurationError(f"{provider} quota or billing limit was reached.")
        return RateLimitError(f"{provider} request limit was reached.")
    if isinstance(exc, TimeoutError) or "Timeout" in name or status == 408:
        return ServiceTimeoutError(f"{provider} request timed out.")
    if status is not None and status >= 500:
        return ServiceUnavailableError(
            f"{provider} is temporarily unavailable (HTTP {status})."
        )
    if "Connection" in name or isinstance(exc, ConnectionError):
        return ServiceUnavailableError(f"{provider} could not be reached.")
    if status in {400, 404, 409, 422}:
        parameter = _value(detail, "param") or _value(body, "param")
        suffix = (
            f" (parameter: {parameter})"
            if isinstance(parameter, str)
            and re.fullmatch(r"[\w.\[\]]{1,80}", parameter)
            else ""
        )
        return ConfigurationError(
            f"{provider} rejected the model or request settings (HTTP {status}){suffix}."
        )
    return ServiceUnavailableError(f"{provider} request failed ({name}).")


class DomainResponseInterpreter:
    """Normalize JSON envelopes and reject invalid topic structures/identifiers."""

    def interpret(self, text: str, spec: OutputSpec) -> Any:
        if not isinstance(text, str) or not text.strip():
            raise InvalidResponseError("The language model returned empty text.")
        text = text.strip()
        if spec.kind == "text":
            return text
        if spec.kind == "pattern":
            if not spec.pattern:
                raise ConfigurationError(
                    "A pattern output requires a regular expression."
                )
            try:
                result = re.search(spec.pattern, text, flags=re.DOTALL)
            except re.error:
                raise ConfigurationError(
                    "The requested output pattern is invalid."
                ) from None
            if result is None:
                raise InvalidResponseError(
                    "The language model response did not match the expected pattern."
                )
            return result.group(0)

        if text.startswith("```") and text.endswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()

        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise InvalidResponseError(
                        "The language model returned duplicate JSON identifiers."
                    )
                result[key] = value
            return result

        try:
            value = json.loads(text, object_pairs_hook=unique_keys)
        except (TypeError, ValueError):
            # Existing local notebooks also accept Python literal dictionaries.
            try:
                expression = ast.parse(text, mode="eval")
                for node in ast.walk(expression):
                    if not isinstance(node, ast.Dict):
                        continue
                    keys = set()
                    for key_node in node.keys:
                        if key_node is None:
                            raise ValueError("Dictionary expansion is unsupported")
                        key = literal_eval(key_node)
                        if key in keys:
                            raise InvalidResponseError(
                                "The language model returned duplicate literal identifiers."
                            )
                        keys.add(key)
                value = literal_eval(expression)
            except (ValueError, SyntaxError, TypeError, RecursionError):
                raise InvalidResponseError(
                    "The language model returned malformed structured text."
                ) from None

        if (
            spec.kind == "topics"
            and isinstance(value, dict)
            and isinstance(value.get("topics"), list)
        ):
            pairs = value["topics"]
            if any(
                not isinstance(item, dict) or set(item) != {"id", "label"}
                for item in pairs
            ):
                raise InvalidResponseError(
                    "The language model returned an invalid topic envelope."
                )
            identifiers = [item["id"] for item in pairs]
            if any(not isinstance(key, str) for key in identifiers) or len(
                set(identifiers)
            ) != len(identifiers):
                raise InvalidResponseError(
                    "The language model returned duplicate or invalid topic identifiers."
                )
            value = {item["id"]: item["label"] for item in pairs}
        elif (
            spec.kind == "topic"
            and isinstance(value, dict)
            and set(value) == {"id", "label"}
        ):
            if not isinstance(value["id"], str):
                raise InvalidResponseError(
                    "The language model returned an invalid topic identifier."
                )
            value = {value["id"]: value["label"]}
        elif (
            spec.kind == "string_list"
            and isinstance(value, dict)
            and set(value) == {"items"}
        ):
            value = value["items"]
        elif (
            spec.kind == "answers"
            and isinstance(value, dict)
            and set(value) == {"answers"}
        ):
            pairs = value["answers"]
            if not isinstance(pairs, list) or any(
                not isinstance(item, dict)
                or set(item) != {"id", "answer"}
                or not isinstance(item["id"], str)
                for item in pairs
            ):
                raise InvalidResponseError(
                    "The language model returned invalid question answers."
                )
            if len({item["id"] for item in pairs}) != len(pairs):
                raise InvalidResponseError(
                    "The language model returned duplicate answer identifiers."
                )
            value = {item["id"]: item["answer"] for item in pairs}

        if spec.kind in {"topics", "topic"}:
            if (
                not isinstance(value, dict)
                or not value
                or any(
                    not isinstance(key, str)
                    or not key.strip()
                    or not isinstance(label, str)
                    or not label.strip()
                    for key, label in value.items()
                )
            ):
                raise InvalidResponseError(
                    "Expected a nonempty mapping of topic identifiers to labels."
                )
            if spec.kind == "topic" and (
                len(value) != 1 or next(iter(value)) not in spec.topic_ids
            ):
                raise InvalidResponseError(
                    "The language model selected an unknown or multiple topic identifiers."
                )
        elif spec.kind == "string_list":
            if (
                not isinstance(value, list)
                or not value
                or any(not isinstance(item, str) or not item.strip() for item in value)
            ):
                raise InvalidResponseError("Expected a nonempty list of strings.")
        elif spec.kind == "answers":
            if (
                not isinstance(value, dict)
                or set(value) != set(spec.answer_ids)
                or any(
                    answer is not None and not isinstance(answer, bool)
                    for answer in value.values()
                )
            ):
                raise InvalidResponseError(
                    "Expected one boolean or null answer for every question identifier."
                )
        else:
            raise ConfigurationError(f"Unknown output kind '{spec.kind}'.")
        if spec.max_items is not None and len(value) > spec.max_items:
            raise InvalidResponseError(
                "The language model returned more items than requested."
            )
        return value


class LLMAdapter(ABC):
    """One finite budget covers external failures and invalid domain responses."""

    def __init__(self, config: LLMConfig, *, client=None, sleep=time.sleep):
        self.config = config
        self.model = config.model
        self.max_attempts = config.max_attempts
        self.attempts_interval = config.retry_backoff_seconds
        self.records: list[ServiceMetadata] = []
        self._client = client
        self._injected_client = client is not None
        self._sleep = sleep
        self.interpreter = DomainResponseInterpreter()
        self._validate_config()

    def _validate_config(self):
        if not self.model.strip() or self.config.max_attempts < 1:
            raise ConfigurationError(
                "A model and a positive attempt budget are required."
            )
        if (
            self.config.timeout_seconds <= 0
            or not math.isfinite(self.config.timeout_seconds)
            or self.config.retry_backoff_seconds < 0
            or not math.isfinite(self.config.retry_backoff_seconds)
        ):
            raise ConfigurationError(
                "Provider timeouts must be positive and retry delays nonnegative."
            )

    @abstractmethod
    def preflight(self) -> None: ...

    @abstractmethod
    def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[str, str | None, str | None]: ...

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if not request.prompt.strip():
            raise ConfigurationError("The generation prompt must not be empty.")
        for attempt in range(1, self.max_attempts + 1):
            started = time.monotonic()
            self._request_sent = False
            self._reported_model = None
            self._reported_effort = None
            metadata = ServiceMetadata(
                provider=self.config.provider,
                requested_model=self.model,
                requested_reasoning_effort=self.config.reasoning_effort,
                sdk_version=_sdk_version(self.config.provider),
                attempts=attempt,
            )
            try:
                text, model, effort = self._generate_once(request)
                metadata = replace(
                    metadata,
                    returned_model=model,
                    reported_reasoning_effort=effort,
                    sent_reasoning_effort=(
                        self.config.reasoning_effort if self._request_sent else None
                    ),
                    temperature_sent=(
                        request.temperature
                        if self._request_sent and self.config.provider == "ollama"
                        else None
                    ),
                )
                value = self.interpreter.interpret(text, request.output)
                metadata = replace(
                    metadata, duration_seconds=time.monotonic() - started
                )
                self.records.append(metadata)
                return GenerationResult(
                    text=text.strip(), value=value, metadata=metadata
                )
            except Exception as exc:
                error = _external_error(exc, self.config.provider)
                self.records.append(
                    replace(
                        metadata,
                        duration_seconds=time.monotonic() - started,
                        status=type(error).__name__,
                        returned_model=self._reported_model,
                        reported_reasoning_effort=self._reported_effort,
                        sent_reasoning_effort=(
                            self.config.reasoning_effort if self._request_sent else None
                        ),
                        temperature_sent=(
                            request.temperature
                            if self._request_sent and self.config.provider == "ollama"
                            else None
                        ),
                    )
                )
                if not error.retryable or attempt == self.max_attempts:
                    raise error from None
                self._sleep(min(self.attempts_interval * (2 ** (attempt - 1)), 10.0))
        raise InvalidResponseError("The language model exhausted its attempt budget.")

    def generate_pattern(self, prompt: str, pattern: str, **kwargs) -> str:
        """Compatibility entry point for existing notebooks and callers."""
        options = dict(kwargs.pop("options", {}) or {})
        images = tuple(kwargs.pop("images", ()) or ())
        temperature = options.pop("temperature", None)
        response_format = kwargs.pop("format", options.pop("format", None))
        if options or kwargs or response_format not in {None, "", "json"}:
            raise ConfigurationError(
                "Unsupported legacy generation options; use GenerationRequest."
            )
        return self.generate(
            GenerationRequest(
                prompt=prompt,
                output=OutputSpec(kind="pattern", pattern=pattern),
                temperature=temperature,
                images=images,
            )
        ).value

    def close(self) -> None:
        if self._client is not None and not self._injected_client:
            closer = getattr(self._client, "close", None)
            if closer is None:
                closer = getattr(getattr(self._client, "_client", None), "close", None)
            if closer is not None:
                closer()
            self._client = None


class OllamaAdapter(LLMAdapter):
    def __init__(
        self,
        model: str = "gemma2",
        max_attempts: int = 3,
        attempts_interval: float = 3,
        *,
        config: LLMConfig | None = None,
        client=None,
        sleep=time.sleep,
    ) -> None:
        super().__init__(
            config
            or LLMConfig(
                model=model,
                max_attempts=max_attempts,
                retry_backoff_seconds=attempts_interval,
            ),
            client=client,
            sleep=sleep,
        )

    def _validate_config(self):
        super()._validate_config()
        if self.config.reasoning_effort is not None:
            raise ConfigurationError(
                "OVS_LLM_REASONING_EFFORT is unsupported by the Ollama adapter."
            )

    def _get_client(self):
        if self._client is None:
            try:
                import ollama
            except ImportError:
                raise ConfigurationError(
                    "Install the ollama package to use the Ollama provider."
                ) from None
            self._client = ollama.Client(
                host=self.config.base_url, timeout=self.config.timeout_seconds
            )
        return self._client

    def preflight(self) -> None:
        self._validate_config()
        try:
            available = self._get_client().list()
            names = {
                _value(item, "name") or _value(item, "model")
                for item in _value(available, "models", [])
            }
        except Exception as exc:
            raise _external_error(exc, "ollama") from None
        if self.model not in names and f"{self.model}:latest" not in names:
            raise ConfigurationError(
                f"Ollama model '{self.model}' is missing; install it on the selected server."
            )

    def _generate_once(self, request: GenerationRequest):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "prompt": request.prompt,
            "stream": False,
        }
        if request.output.kind != "text":
            kwargs["format"] = "json"
        if request.temperature is not None:
            kwargs["options"] = {"temperature": request.temperature}
        if request.images:
            kwargs["images"] = list(request.images)
        client = self._get_client()
        self._request_sent = True
        response = client.generate(**kwargs)
        self._reported_model = _value(response, "model")
        return _value(response, "response"), self._reported_model, None


class OpenAIAdapter(LLMAdapter):
    """Responses API details and envelopes are contained in this adapter."""

    _known_efforts = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
    # Official model pages document these exact aliases. Date snapshots inherit
    # their alias capabilities; other/future variants are checked by the API.
    _model_efforts = {
        "gpt-6-luna": {"none", "low", "medium", "high", "xhigh", "max"},
        "gpt-6-astra": {"low", "medium", "high", "xhigh", "max"},
        "gpt-6.1-sol": {"low", "medium", "high", "xhigh", "max"},
        "gpt-5": {"minimal", "low", "medium", "high"},
    }
    _nonreasoning_aliases = (
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
    )
    _nonreasoning_model_ids = {
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-instruct",
    }

    @staticmethod
    def _known_alias(model: str, alias: str) -> bool:
        return (
            re.fullmatch(re.escape(alias) + r"(?:-\d{4}-\d{2}-\d{2})?", model)
            is not None
        )

    def __init__(
        self, *, config: LLMConfig | None = None, client=None, sleep=time.sleep
    ):
        super().__init__(
            config
            or LLMConfig(
                provider="openai",
                model="gpt-6-luna",
                base_url="https://api.openai.com/v1",
            ),
            client=client,
            sleep=sleep,
        )

    def _validate_config(self):
        super()._validate_config()
        effort, model = self.config.reasoning_effort, self.model
        if effort is not None and effort not in self._known_efforts:
            raise ConfigurationError(f"Unknown reasoning effort '{effort}'.")
        if effort is not None and (
            model in self._nonreasoning_model_ids
            or any(
                self._known_alias(model, alias)
                for alias in self._nonreasoning_aliases
            )
        ):
            raise ConfigurationError(
                f"Model '{model}' does not support reasoning effort."
            )
        for alias, supported in self._model_efforts.items():
            if (
                self._known_alias(model, alias)
                and effort is not None
                and effort not in supported
            ):
                raise ConfigurationError(
                    f"Model '{model}' does not support {effort} reasoning effort."
                )
        if (
            any(
                self._known_alias(model, alias)
                for alias in ("o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini")
            )
            and effort is not None
            and effort not in {"low", "medium", "high"}
        ):
            raise ConfigurationError(
                f"Model '{model}' requires low, medium or high reasoning effort."
            )

    def _get_client(self):
        if self._client is None:
            if not self.config.api_key:
                raise ConfigurationError(
                    "OPENAI_API_KEY is required for the OpenAI provider."
                )
            try:
                from openai import OpenAI
            except ImportError:
                raise ConfigurationError(
                    "Install the pinned OpenAI SDK with Responses support."
                ) from None
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
                max_retries=0,
            )
        return self._client

    def preflight(self) -> None:
        self._validate_config()
        self._get_client()

    @staticmethod
    def _format(spec: OutputSpec):
        item = {
            "type": "object",
            "properties": {"id": {"type": "string"}, "label": {"type": "string"}},
            "required": ["id", "label"],
            "additionalProperties": False,
        }
        if spec.kind == "topic":
            item["properties"]["id"]["enum"] = list(spec.topic_ids)
            schema = item
        elif spec.kind == "topics":
            schema = {
                "type": "object",
                "properties": {"topics": {"type": "array", "items": item}},
                "required": ["topics"],
                "additionalProperties": False,
            }
        elif spec.kind == "string_list":
            schema = {
                "type": "object",
                "properties": {"items": {"type": "array", "items": {"type": "string"}}},
                "required": ["items"],
                "additionalProperties": False,
            }
        elif spec.kind == "answers":
            answer = {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "enum": list(spec.answer_ids)},
                    "answer": {"type": ["boolean", "null"]},
                },
                "required": ["id", "answer"],
                "additionalProperties": False,
            }
            schema = {
                "type": "object",
                "properties": {"answers": {"type": "array", "items": answer}},
                "required": ["answers"],
                "additionalProperties": False,
            }
        else:
            return None
        return {
            "type": "json_schema",
            "name": f"ovs_{spec.kind}",
            "strict": True,
            "schema": schema,
        }

    def _generate_once(self, request: GenerationRequest):
        if request.images:
            content: list[dict] = [{"type": "input_text", "text": request.prompt}]
            content.extend(
                {
                    "type": "input_image",
                    "image_url": "data:image/jpeg;base64,"
                    + base64.b64encode(image).decode("ascii"),
                }
                for image in request.images
            )
            input_data = [{"role": "user", "content": content}]
        else:
            input_data = request.prompt
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_data,
            "store": False,
        }
        if self.config.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.config.reasoning_effort}
        response_format = self._format(request.output)
        if response_format:
            kwargs["text"] = {"format": response_format}
        # Temperature is deliberately omitted: known reasoning models reject it,
        # and unknown/future models must not inherit Ollama sampling parameters.
        client = self._get_client()
        self._request_sent = True
        response = client.responses.create(**kwargs)
        self._reported_model = _value(response, "model")
        self._reported_effort = _value(_value(response, "reasoning"), "effort")
        if _value(response, "status") != "completed":
            raise InvalidResponseError(
                "OpenAI returned an incomplete or failed response."
            )
        parts = []
        for item in _value(response, "output", []) or []:
            if _value(item, "type") != "message":
                continue
            for part in _value(item, "content", []) or []:
                if _value(part, "type") == "refusal":
                    raise InvalidResponseError("OpenAI refused the requested response.")
                if _value(part, "type") == "output_text":
                    parts.append(_value(part, "text", ""))
        text = "".join(parts)
        return text, self._reported_model, self._reported_effort
