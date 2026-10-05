"""Typed configuration with CLI > process > root .env > configured defaults."""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping
from urllib.parse import urlsplit

from open_video_summary.errors import ConfigurationError
from open_video_summary.utils.config import PROJECT_DIR

VisualScope = Literal["segment", "video"]


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "ollama"
    model: str = "gemma2"
    reasoning_effort: str | None = None
    base_url: str = "http://localhost:11434"
    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: float = 120.0
    max_attempts: int = 3
    retry_backoff_seconds: float = 1.0


@dataclass(frozen=True)
class STTConfig:
    provider: str = "whisper_local"
    model: str = "base"
    language: str | None = "pt"
    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: float = 120.0
    max_attempts: int = 3
    retry_backoff_seconds: float = 1.0


@dataclass(frozen=True)
class EvaluatorConfig:
    """Independent, non-generative evaluator settings without exposed keys."""

    provider: str = "typesafe"
    model: str = "jev-1.13.0"
    base_url: str = "https://api.typesafe.ai"
    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: float = 30.0
    max_attempts: int = 2
    retry_backoff_seconds: float = 0.5


@dataclass(frozen=True)
class ProviderConfig:
    llm: LLMConfig
    stt: STTConfig


def _optional(value) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _positive(value, name: str, *, integer: bool = False):
    try:
        result = int(value) if integer else float(value)
    except (TypeError, ValueError):
        raise ConfigurationError(f"{name} must be a positive number.") from None
    if result <= 0 or not math.isfinite(result):
        raise ConfigurationError(f"{name} must be a positive number.")
    return result


def load_environment_values(
    *,
    environ: Mapping[str, str] | None = None,
    env_file: str | Path | None = None,
) -> dict[str, str | None]:
    """Read root settings without interpolation or changing process values."""
    from dotenv import dotenv_values

    path = Path(env_file) if env_file is not None else PROJECT_DIR / ".env"
    values = dict(dotenv_values(path, interpolate=False)) if path.is_file() else {}
    values.update(os.environ if environ is None else environ)
    return values


def _setting(supplied, values, argument: str, variable: str, default=None):
    explicit = supplied.get(argument)
    value = explicit if explicit is not None else values.get(variable)
    return _optional(value) if _optional(value) is not None else default


def load_thread_count(
    cli_value: int | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    env_file: str | Path | None = None,
) -> int:
    """Resolve the CLI CPU thread budget without loading provider settings."""
    if cli_value is not None:
        value, name = cli_value, "--threads"
    else:
        values = load_environment_values(environ=environ, env_file=env_file)
        value = _optional(values.get("OVS_THREADS")) or "2"
        name = "OVS_THREADS"

    try:
        return _positive(value, name, integer=True)
    except ConfigurationError:
        raise ConfigurationError(f"{name} must be a positive integer.") from None


def load_visual_scope(
    cli_value: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    env_file: str | Path | None = None,
) -> VisualScope:
    """Resolve visual extraction scope independently of provider credentials."""
    if cli_value is not None:
        value, name = _optional(cli_value) or "segment", "--visual-scope"
    else:
        values = load_environment_values(environ=environ, env_file=env_file)
        value = _optional(values.get("OVS_VISUAL_SCOPE")) or "segment"
        name = "OVS_VISUAL_SCOPE"
    value = value.lower()
    if value not in {"segment", "video"}:
        raise ConfigurationError(f"{name} must be 'segment' or 'video'.")
    return value


def load_provider_config(
    overrides: Mapping[str, object] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    env_file: str | Path | None = None,
    default_llm_models: Mapping[str, str] | None = None,
) -> ProviderConfig:
    """Read the root .env without modifying the process or exposing its values.

    Explicit None arguments mean "not supplied". An empty optional process
    value masks the .env value and is treated as absent. Credentials are never
    included in dataclass representations.
    """
    from open_video_summary.adapters.factory import LLM_PROVIDERS, STT_PROVIDERS

    values = load_environment_values(environ=environ, env_file=env_file)
    supplied = overrides or {}

    def get(argument: str, variable: str, default=None):
        return _setting(supplied, values, argument, variable, default)

    llm_provider = get("llm_provider", "OVS_LLM_PROVIDER", "ollama").lower()
    stt_provider = get("stt_provider", "OVS_STT_PROVIDER", "whisper_local").lower()
    if llm_provider not in LLM_PROVIDERS:
        raise ConfigurationError(f"Unknown LLM provider '{llm_provider}'.")
    if stt_provider not in STT_PROVIDERS:
        raise ConfigurationError(f"Unknown STT provider '{stt_provider}'.")
    llm_definition = LLM_PROVIDERS[llm_provider]
    stt_definition = STT_PROVIDERS[stt_provider]
    attempts = _positive(
        get("max_attempts", "OVS_MAX_ATTEMPTS", "3"), "OVS_MAX_ATTEMPTS", integer=True
    )
    llm = LLMConfig(
        provider=llm_provider,
        model=get(
            "llm_model",
            "OVS_LLM_MODEL",
            (default_llm_models or {}).get(llm_provider, llm_definition.default_model),
        ),
        reasoning_effort=get("reasoning_effort", "OVS_LLM_REASONING_EFFORT"),
        base_url=get("llm_base_url", "OVS_LLM_BASE_URL", llm_definition.base_url),
        api_key=_optional(values.get(llm_definition.key_variable)),
        timeout_seconds=_positive(
            get("llm_timeout", "OVS_LLM_TIMEOUT_SECONDS", "120"),
            "OVS_LLM_TIMEOUT_SECONDS",
        ),
        max_attempts=attempts,
    )
    language = get("language", "OVS_STT_LANGUAGE", "pt")
    stt = STTConfig(
        provider=stt_provider,
        model=get("stt_model", "OVS_STT_MODEL", stt_definition.default_model),
        language=None if language.lower() == "auto" else language,
        api_key=_optional(values.get(stt_definition.key_variable)),
        timeout_seconds=_positive(
            get("stt_timeout", "OVS_STT_TIMEOUT_SECONDS", "120"),
            "OVS_STT_TIMEOUT_SECONDS",
        ),
        max_attempts=attempts,
    )
    endpoint = urlsplit(llm.base_url)
    if endpoint.scheme not in {"http", "https"} or not endpoint.netloc:
        raise ConfigurationError("OVS_LLM_BASE_URL must be an HTTP(S) endpoint.")
    if endpoint.username is not None or endpoint.password is not None:
        raise ConfigurationError("OVS_LLM_BASE_URL must not contain credentials.")
    return ProviderConfig(llm, stt)


def load_evaluator_config(
    overrides: Mapping[str, object] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    env_file: str | Path | None = None,
) -> EvaluatorConfig:
    """Resolve only the selected evaluator; unrelated commands need not load it.

    A registered provider supplies its adapter, model and endpoint defaults.
    Every evaluator uses the role's OVS_EVALUATOR_API_KEY from process/root .env,
    independent of LLM/STT credentials. CLI settings do not carry secrets, and
    former TypeSafe/Jev settings are not used as fallbacks.
    """
    from open_video_summary.adapters.factory import EVALUATOR_PROVIDERS

    values = load_environment_values(environ=environ, env_file=env_file)
    supplied = overrides or {}

    def get(argument: str, variable: str, default=None):
        return _setting(supplied, values, argument, variable, default)

    provider = get("evaluator_provider", "OVS_EVALUATOR_PROVIDER", "typesafe").lower()
    if provider not in EVALUATOR_PROVIDERS:
        raise ConfigurationError(f"Unknown evaluator provider '{provider}'.")
    definition = EVALUATOR_PROVIDERS[provider]
    config = EvaluatorConfig(
        provider=provider,
        model=get("evaluator_model", "OVS_EVALUATOR_MODEL", definition.default_model),
        base_url=get(
            "evaluator_base_url", "OVS_EVALUATOR_BASE_URL", definition.base_url
        ),
        # This role has one neutral credential variable for every adapter.
        api_key=_optional(values.get("OVS_EVALUATOR_API_KEY")),
        timeout_seconds=_positive(
            get("evaluator_timeout", "OVS_EVALUATOR_TIMEOUT_SECONDS", "30"),
            "OVS_EVALUATOR_TIMEOUT_SECONDS",
        ),
        max_attempts=_positive(
            get("evaluator_max_attempts", "OVS_EVALUATOR_MAX_ATTEMPTS", "2"),
            "OVS_EVALUATOR_MAX_ATTEMPTS",
            integer=True,
        ),
    )
    try:
        endpoint = urlsplit(config.base_url)
        _ = endpoint.port
    except ValueError:
        raise ConfigurationError(
            "OVS_EVALUATOR_BASE_URL must be a valid HTTP(S) endpoint."
        ) from None
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
        raise ConfigurationError("OVS_EVALUATOR_BASE_URL must be an HTTP(S) endpoint.")
    if (
        endpoint.username is not None
        or endpoint.password is not None
        or endpoint.query
        or endpoint.fragment
    ):
        raise ConfigurationError(
            "OVS_EVALUATOR_BASE_URL must not contain credentials, query or fragment."
        )
    return config
