"""Registries select independent generation, transcription and evaluation roles."""

from dataclasses import dataclass
from typing import Callable
from collections.abc import Mapping

from open_video_summary.contracts import Evaluator, LanguageModel, SpeechToText
from open_video_summary.errors import ConfigurationError
from open_video_summary.utils.providers import (
    EvaluatorConfig,
    LLMConfig,
    ProviderConfig,
    STTConfig,
)


@dataclass(frozen=True)
class ProviderDefinition:
    constructor: Callable
    default_model: str
    base_url: str = ""
    key_variable: str = ""


def _ollama(config: LLMConfig) -> LanguageModel:
    from open_video_summary.adapters.llm import OllamaAdapter

    return OllamaAdapter(config=config)


def _openai(config: LLMConfig) -> LanguageModel:
    from open_video_summary.adapters.llm import OpenAIAdapter

    return OpenAIAdapter(config=config)


def _whisper(config: STTConfig) -> SpeechToText:
    from open_video_summary.adapters.stt import LocalWhisperSTT

    return LocalWhisperSTT(config=config)


def _elevenlabs(config: STTConfig) -> SpeechToText:
    from open_video_summary.adapters.stt import ElevenLabsSTT

    return ElevenLabsSTT(config=config)


def _typesafe(config: EvaluatorConfig) -> Evaluator:
    from open_video_summary.adapters.typesafe import TypeSafeConfig, TypeSafeEvaluator

    return TypeSafeEvaluator(
        TypeSafeConfig(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
            timeout_seconds=config.timeout_seconds,
            max_attempts=config.max_attempts,
            retry_backoff_seconds=config.retry_backoff_seconds,
            operation_timeout_seconds=config.operation_timeout_seconds,
            request_limit=config.request_limit,
            token_limit=config.token_limit,
            rate_window_seconds=config.rate_window_seconds,
            limit_group=config.limit_group,
            token_request_overhead=config.token_request_overhead,
            token_question_overhead=config.token_question_overhead,
        )
    )


LLM_PROVIDERS: dict[str, ProviderDefinition] = {
    "ollama": ProviderDefinition(_ollama, "gemma2", "http://localhost:11434"),
    "openai": ProviderDefinition(
        _openai, "gpt-6-luna", "https://api.openai.com/v1", "OPENAI_API_KEY"
    ),
}
STT_PROVIDERS: dict[str, ProviderDefinition] = {
    "whisper_local": ProviderDefinition(_whisper, "base"),
    "elevenlabs": ProviderDefinition(
        _elevenlabs, "scribe_v2", key_variable="ELEVENLABS_API_KEY"
    ),
}
EVALUATOR_PROVIDERS: dict[str, ProviderDefinition] = {
    "typesafe": ProviderDefinition(
        _typesafe, "jev-1.13.0", "https://api.typesafe.ai", "OVS_EVALUATOR_API_KEY"
    ),
}


def create_llm(config: LLMConfig) -> LanguageModel:
    try:
        definition = LLM_PROVIDERS[config.provider]
    except KeyError:
        raise ConfigurationError(f"Unknown LLM provider '{config.provider}'.") from None
    return definition.constructor(config)


def create_stt(config: STTConfig) -> SpeechToText:
    try:
        definition = STT_PROVIDERS[config.provider]
    except KeyError:
        raise ConfigurationError(f"Unknown STT provider '{config.provider}'.") from None
    return definition.constructor(config)


def create_evaluator(config: EvaluatorConfig) -> Evaluator:
    try:
        definition = EVALUATOR_PROVIDERS[config.provider]
    except KeyError:
        raise ConfigurationError(
            f"Unknown evaluator provider '{config.provider}'."
        ) from None
    return definition.constructor(config)


def create_providers(config: ProviderConfig) -> tuple[LanguageModel, SpeechToText]:
    """Construct only the selected pair; constructors must not load models."""
    return create_llm(config.llm), create_stt(config.stt)


def create_configured_llm(
    *, default_models: Mapping[str, str] | None = None
) -> LanguageModel:
    """Preserve a caller's local model default while honoring explicit settings."""
    from open_video_summary.utils.providers import load_provider_config

    return create_llm(load_provider_config(default_llm_models=default_models).llm)


def create_configured_evaluator(
    overrides: Mapping[str, object] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    env_file=None,
) -> Evaluator:
    """Construct only the evaluator selected by CLI/environment configuration."""
    from open_video_summary.utils.providers import load_evaluator_config

    return create_evaluator(
        load_evaluator_config(overrides, environ=environ, env_file=env_file)
    )
