import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from open_video_summary.adapters.factory import (
    LLM_PROVIDERS,
    STT_PROVIDERS,
    ProviderDefinition,
    create_providers,
)
from open_video_summary.errors import ConfigurationError
from open_video_summary.utils.providers import load_provider_config


class ProviderConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.env = Path(self.directory.name) / ".env"

    def load(self, overrides=None, environ=None):
        return load_provider_config(
            overrides, environ={} if environ is None else environ, env_file=self.env
        )

    def test_defaults_and_provider_endpoints(self):
        config = self.load()
        self.assertEqual(("ollama", "gemma2"), (config.llm.provider, config.llm.model))
        self.assertEqual("http://localhost:11434", config.llm.base_url)
        self.assertEqual("whisper_local", config.stt.provider)
        external = self.load(environ={"OVS_LLM_PROVIDER": "openai"})
        self.assertEqual("https://api.openai.com/v1", external.llm.base_url)
        self.assertEqual("gpt-6-luna", external.llm.model)

    def test_cli_process_file_and_default_precedence(self):
        self.env.write_text(
            "OVS_LLM_PROVIDER=openai\nOVS_LLM_MODEL=file-model\n"
            "OVS_STT_PROVIDER=elevenlabs\nOVS_STT_MODEL=scribe_v2\n",
            encoding="utf-8",
        )
        process = {"OVS_LLM_MODEL": "process-model"}
        config = self.load({"llm_model": "cli-model", "stt_model": None}, process)
        self.assertEqual("cli-model", config.llm.model)
        self.assertEqual("scribe_v2", config.stt.model)
        self.assertEqual("process-model", process["OVS_LLM_MODEL"])
        self.assertEqual("process-model", self.load(environ=process).llm.model)
        self.assertEqual("file-model", self.load().llm.model)

    def test_optional_empty_values_and_explicit_none_effort(self):
        self.env.write_text("OVS_LLM_REASONING_EFFORT=high\n", encoding="utf-8")
        absent = self.load(environ={"OVS_LLM_REASONING_EFFORT": " "})
        self.assertIsNone(absent.llm.reasoning_effort)
        explicit = self.load(environ={"OVS_LLM_REASONING_EFFORT": "none"})
        self.assertEqual("none", explicit.llm.reasoning_effort)
        config = self.load(
            environ={"OPENAI_API_KEY": " ", "OVS_STT_LANGUAGE": ""}
        )
        self.assertIsNone(config.llm.api_key)
        self.assertEqual("pt", config.stt.language)
        self.assertIsNone(self.load(environ={"OVS_STT_LANGUAGE": "auto"}).stt.language)

    def test_custom_endpoint_and_secret_not_in_repr(self):
        config = self.load(
            environ={
                "OVS_LLM_PROVIDER": "openai",
                "OVS_LLM_BASE_URL": "https://gateway.example/v1",
                "OPENAI_API_KEY": "test-private-key",
            }
        )
        self.assertEqual("https://gateway.example/v1", config.llm.base_url)
        self.assertNotIn("test-private-key", repr(config))

    def test_invalid_configuration_is_actionable(self):
        for environment in (
            {"OVS_LLM_PROVIDER": "missing"},
            {"OVS_STT_PROVIDER": "missing"},
            {"OVS_MAX_ATTEMPTS": "0"},
            {"OVS_MAX_ATTEMPTS": "3.5"},
            {"OVS_LLM_TIMEOUT_SECONDS": "NaN"},
            {"OVS_STT_TIMEOUT_SECONDS": "-2"},
            {"OVS_LLM_BASE_URL": "file:///bad"},
            {"OVS_LLM_BASE_URL": "https://key:secret@example.org"},
        ):
            with self.subTest(environment=environment):
                with self.assertRaises(ConfigurationError):
                    self.load(environ=environment)

    def test_factory_selects_independent_pairs(self):
        llm_marker, stt_marker = object(), object()
        with patch.dict(
            LLM_PROVIDERS,
            {"openai": ProviderDefinition(lambda config: llm_marker, "future", "https://api.test")},
        ), patch.dict(
            STT_PROVIDERS,
            {"whisper_local": ProviderDefinition(lambda config: stt_marker, "base")},
        ):
            config = self.load(environ={"OVS_LLM_PROVIDER": "openai"})
            self.assertEqual((llm_marker, stt_marker), create_providers(config))

    def test_registering_provider_requires_no_algorithm_change(self):
        with patch.dict(
            LLM_PROVIDERS,
            {"new": ProviderDefinition(lambda config: config, "next", "https://new.test")},
        ):
            config = self.load(environ={"OVS_LLM_PROVIDER": "new"})
            self.assertEqual("next", config.llm.model)
            self.assertEqual("https://new.test", config.llm.base_url)


if __name__ == "__main__":
    unittest.main()
