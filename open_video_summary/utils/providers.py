"""Typed configuration with CLI > process > root .env > provider defaults."""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

from open_video_summary.errors import ConfigurationError
from open_video_summary.utils.config import PROJECT_DIR


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


def load_provider_config(
    overrides: Mapping[str, object] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    env_file: str | Path | None = None,
) -> ProviderConfig:
    """Read the root .env without modifying the process or exposing its values.

    Explicit None arguments mean "not supplied". An empty optional process
    value masks the .env value and is treated as absent. Credentials are never
    included in dataclass representations.
    """
    from dotenv import dotenv_values
    from open_video_summary.adapters.factory import LLM_PROVIDERS, STT_PROVIDERS

    path = Path(env_file) if env_file is not None else PROJECT_DIR / ".env"
    values = dict(dotenv_values(path, interpolate=False)) if path.is_file() else {}
    values.update(os.environ if environ is None else environ)
    supplied = overrides or {}

    def get(argument: str, variable: str, default=None):
        explicit = supplied.get(argument)
        value = explicit if explicit is not None else values.get(variable)
        return _optional(value) if _optional(value) is not None else default

    def optional_get(argument: str, variable: str, default=None):
        explicit = supplied.get(argument)
        if explicit is not None:
            return _optional(explicit)
        return _optional(values[variable]) if variable in values else default

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
        model=get("llm_model", "OVS_LLM_MODEL", llm_definition.default_model),
        reasoning_effort=get("reasoning_effort", "OVS_LLM_REASONING_EFFORT"),
        base_url=get("llm_base_url", "OVS_LLM_BASE_URL", llm_definition.base_url),
        api_key=_optional(values.get(llm_definition.key_variable)),
        timeout_seconds=_positive(
            get("llm_timeout", "OVS_LLM_TIMEOUT_SECONDS", "120"),
            "OVS_LLM_TIMEOUT_SECONDS",
        ),
        max_attempts=attempts,
    )
    stt = STTConfig(
        provider=stt_provider,
        model=get("stt_model", "OVS_STT_MODEL", stt_definition.default_model),
        language=optional_get("language", "OVS_STT_LANGUAGE", "pt"),
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
