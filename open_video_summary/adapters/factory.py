"""One registry selects the independent LLM and transcription adapters."""

from dataclasses import dataclass
from typing import Callable
from collections.abc import Mapping

from open_video_summary.contracts import LanguageModel, SpeechToText
from open_video_summary.errors import ConfigurationError
from open_video_summary.utils.providers import LLMConfig, ProviderConfig, STTConfig


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


def create_providers(config: ProviderConfig) -> tuple[LanguageModel, SpeechToText]:
    """Construct only the selected pair; constructors must not load models."""
    return create_llm(config.llm), create_stt(config.stt)


def create_configured_llm(
    *, default_models: Mapping[str, str] | None = None
) -> LanguageModel:
    """Preserve a caller's local model default while honoring explicit settings."""
    from open_video_summary.utils.providers import load_provider_config

    return create_llm(load_provider_config(default_llm_models=default_models).llm)
