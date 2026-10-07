import tempfile
import os
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from open_video_summary.adapters.factory import (
    EVALUATOR_PROVIDERS,
    LLM_PROVIDERS,
    STT_PROVIDERS,
    ProviderDefinition,
    create_configured_evaluator,
    create_evaluator,
    create_providers,
)
from open_video_summary.errors import ConfigurationError
from open_video_summary.utils.providers import (
    EvaluatorConfig,
    load_evaluator_config,
    load_provider_config,
)


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
        self.assertEqual(360, external.llm.operation_timeout_seconds)
        self.assertEqual(16384, external.llm.max_output_tokens)
        self.assertIsNone(external.llm.request_limit)
        self.assertIsNone(external.llm.token_limit)
        self.assertTrue(external.llm.learn_rate_limits)

    def test_request_controls_respect_cli_process_file_and_validate_limits(self):
        self.env.write_text("OVS_LLM_PROVIDER=openai\nOVS_LLM_REQUEST_LIMIT=100\n"
            "OVS_LLM_TOKEN_LIMIT=100000\nOVS_LLM_OPERATION_TIMEOUT_SECONDS=400\n"
            "OVS_LLM_MAX_OUTPUT_TOKENS=8000\nOVS_LLM_PROJECT=fixture-project\n"
            "OVS_LLM_LIMIT_GROUP=fixture-family\n", encoding="utf-8")
        config = self.load({"llm_request_limit": 50}, {"OVS_LLM_REQUEST_LIMIT": "80"})
        self.assertEqual(50, config.llm.request_limit)
        self.assertEqual(100000, config.llm.token_limit)
        self.assertEqual(400, config.llm.operation_timeout_seconds)
        self.assertEqual(8000, config.llm.max_output_tokens)
        self.assertEqual("fixture-project", config.llm.project)
        self.assertEqual("fixture-family", config.llm.limit_group)
        for name, value in (("OVS_LLM_REQUEST_LIMIT", "0"), ("OVS_LLM_TOKEN_LIMIT", "NaN"),
                            ("OVS_LLM_OPERATION_TIMEOUT_SECONDS", "-1"),
                            ("OVS_LLM_LEARN_RATE_LIMITS", "maybe")):
            with self.subTest(name=name):
                with self.assertRaises(ConfigurationError):
                    self.load(environ={name: value})

    def test_evaluator_window_defaults_and_overrides_are_independent(self):
        defaults = load_evaluator_config(environ={}, env_file=self.env)
        self.assertEqual((60, 80000, 1, 90), (defaults.request_limit, defaults.token_limit,
            defaults.rate_window_seconds, defaults.operation_timeout_seconds))
        self.env.write_text("OVS_EVALUATOR_REQUEST_LIMIT=55\nOVS_EVALUATOR_TOKEN_LIMIT=75000\n"
            "OVS_EVALUATOR_OPERATION_TIMEOUT_SECONDS=80\n", encoding="utf-8")
        selected = load_evaluator_config({"evaluator_request_limit": 40},
            environ={"OVS_EVALUATOR_REQUEST_LIMIT": "50"}, env_file=self.env)
        self.assertEqual(40, selected.request_limit)
        self.assertEqual(75000, selected.token_limit)
        self.assertEqual(80, selected.operation_timeout_seconds)

    def test_evaluator_token_overhead_precedence_and_bounds(self):
        defaults = load_evaluator_config(environ={}, env_file=self.env)
        self.assertEqual((256, 128), (defaults.token_request_overhead, defaults.token_question_overhead))
        self.env.write_text("OVS_EVALUATOR_TOKEN_REQUEST_OVERHEAD=300\n"
            "OVS_EVALUATOR_TOKEN_QUESTION_OVERHEAD=150\n", encoding="utf-8")
        selected = load_evaluator_config({"evaluator_token_request_overhead": 0},
            environ={"OVS_EVALUATOR_TOKEN_REQUEST_OVERHEAD": "400",
                     "OVS_EVALUATOR_TOKEN_QUESTION_OVERHEAD": "200"}, env_file=self.env)
        self.assertEqual((0, 200), (selected.token_request_overhead, selected.token_question_overhead))
        evaluator = create_evaluator(selected)
        self.assertEqual((0, 200), (evaluator.config.token_request_overhead, evaluator.config.token_question_overhead))
        for name in ("OVS_EVALUATOR_TOKEN_REQUEST_OVERHEAD", "OVS_EVALUATOR_TOKEN_QUESTION_OVERHEAD"):
            for value in ("-1", "100001", "0.5", "NaN"):
                with self.subTest(name=name, value=value), self.assertRaises(ConfigurationError):
                    load_evaluator_config(environ={name: value}, env_file=self.env)

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
        config = self.load(environ={"OPENAI_API_KEY": " ", "OVS_STT_LANGUAGE": ""})
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
        with (
            patch.dict(
                LLM_PROVIDERS,
                {
                    "openai": ProviderDefinition(
                        lambda config: llm_marker, "future", "https://api.test"
                    )
                },
            ),
            patch.dict(
                STT_PROVIDERS,
                {
                    "whisper_local": ProviderDefinition(
                        lambda config: stt_marker, "base"
                    )
                },
            ),
        ):
            config = self.load(environ={"OVS_LLM_PROVIDER": "openai"})
            self.assertEqual((llm_marker, stt_marker), create_providers(config))

    def test_registering_provider_requires_no_algorithm_change(self):
        with patch.dict(
            LLM_PROVIDERS,
            {
                "new": ProviderDefinition(
                    lambda config: config, "next", "https://new.test"
                )
            },
        ):
            config = self.load(environ={"OVS_LLM_PROVIDER": "new"})
            self.assertEqual("next", config.llm.model)
            self.assertEqual("https://new.test", config.llm.base_url)

    def test_root_dotenv_is_used_from_another_working_directory(self):
        self.env.write_text("OVS_LLM_MODEL=root-model\n", encoding="utf-8")
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as elsewhere:
            try:
                os.chdir(elsewhere)
                with patch(
                    "open_video_summary.utils.providers.PROJECT_DIR", self.env.parent
                ):
                    config = load_provider_config(environ={})
                self.assertEqual("root-model", config.llm.model)
            finally:
                os.chdir(original)

    def test_filter_model_default_never_overrides_explicit_environment(self):
        defaults = {"ollama": "ministral-3"}
        config = load_provider_config(
            environ={}, env_file=self.env, default_llm_models=defaults
        )
        self.assertEqual("ministral-3", config.llm.model)
        explicit = load_provider_config(
            environ={"OVS_LLM_MODEL": "configured"},
            env_file=self.env,
            default_llm_models=defaults,
        )
        self.assertEqual("configured", explicit.llm.model)


class EvaluatorConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.env = Path(self.directory.name) / ".env"

    def load(self, overrides=None, environ=None):
        return load_evaluator_config(
            overrides, environ={} if environ is None else environ, env_file=self.env
        )

    def test_default_role_selects_typesafe_with_pinned_jev_and_optional_key(self):
        config = self.load()
        self.assertEqual("typesafe", config.provider)
        self.assertEqual("jev-1.13.0", config.model)
        self.assertEqual("https://api.typesafe.ai", config.base_url)
        self.assertIsNone(config.api_key)
        self.assertEqual(30.0, config.timeout_seconds)
        self.assertEqual(2, config.max_attempts)
        evaluator = create_evaluator(config)
        self.assertEqual("typesafe", evaluator.config.provider)
        self.assertEqual(config.model, evaluator.config.model)
        self.assertEqual(config.base_url, evaluator.config.base_url)
        self.assertEqual(0.5, evaluator.config.retry_backoff_seconds)

    def test_cli_process_file_precedence_covers_each_role_setting(self):
        self.env.write_text(
            "OVS_EVALUATOR_PROVIDER=typesafe\nOVS_EVALUATOR_MODEL=file-model\n"
            "OVS_EVALUATOR_API_KEY=file-key\nOVS_EVALUATOR_BASE_URL=https://file.example\n"
            "OVS_EVALUATOR_TIMEOUT_SECONDS=19\nOVS_EVALUATOR_MAX_ATTEMPTS=4\n",
            encoding="utf-8",
        )
        process = {
            "OVS_EVALUATOR_PROVIDER": "fixture",
            "OVS_EVALUATOR_MODEL": "process-model",
            "OVS_EVALUATOR_API_KEY": "process-key",
            "OVS_EVALUATOR_BASE_URL": "https://process.example",
            "OVS_EVALUATOR_TIMEOUT_SECONDS": "12",
            "OVS_EVALUATOR_MAX_ATTEMPTS": "3",
        }
        original = dict(process)
        overrides = {
            "evaluator_provider": "typesafe",
            "evaluator_model": "cli-model",
            "evaluator_base_url": "https://cli.example",
            "evaluator_timeout": 7.5,
            "evaluator_max_attempts": 1,
        }
        with patch.dict(
            EVALUATOR_PROVIDERS,
            {
                "fixture": ProviderDefinition(
                    lambda config: config,
                    "fixture-model",
                    "https://fixture.example",
                    "OVS_EVALUATOR_API_KEY",
                )
            },
        ):
            file = self.load()
            environment = self.load(environ=process)
            explicit = self.load(overrides, process)
        self.assertEqual("typesafe", file.provider)
        self.assertEqual("fixture", environment.provider)
        self.assertEqual("typesafe", explicit.provider)
        for setting, expected in {
            "model": ("file-model", "process-model", "cli-model"),
            "api_key": ("file-key", "process-key", "process-key"),
            "base_url": (
                "https://file.example",
                "https://process.example",
                "https://cli.example",
            ),
            "timeout_seconds": (19.0, 12.0, 7.5),
            "max_attempts": (4, 3, 1),
        }.items():
            self.assertEqual(
                expected,
                tuple(
                    getattr(config, setting) for config in (file, environment, explicit)
                ),
            )
        self.assertEqual(original, process)
        self.assertNotIn("process-key", repr(explicit))

    def test_blank_key_masks_file_and_no_cross_service_or_former_name_fallback(self):
        self.env.write_text("OVS_EVALUATOR_API_KEY=file-private\n", encoding="utf-8")
        environment = {
            "OVS_EVALUATOR_API_KEY": " ",
            "OPENAI_API_KEY": "generator-private",
            "ELEVENLABS_API_KEY": "transcription-private",
            "TYPESAFE_API_KEY": "former-private",
            "OVS_JEV_MODEL": "former-model",
            "OVS_JEV_BASE_URL": "https://former.example",
            "OVS_JEV_TIMEOUT_SECONDS": "99",
            "OVS_JEV_MAX_ATTEMPTS": "5",
        }
        config = self.load(environ=environment)
        self.assertIsNone(config.api_key)
        self.assertEqual("jev-1.13.0", config.model)
        self.assertEqual("https://api.typesafe.ai", config.base_url)
        self.assertEqual(30.0, config.timeout_seconds)
        self.assertEqual(2, config.max_attempts)
        self.assertEqual(
            "process-private",
            self.load(environ={"OVS_EVALUATOR_API_KEY": "process-private"}).api_key,
        )

    def test_registry_defaults_constructor_use_only_generic_evaluator_credential(self):
        with patch.dict(
            EVALUATOR_PROVIDERS,
            {
                "fixture": ProviderDefinition(
                    lambda config: config,
                    "registered-model",
                    "https://registered.example",
                    "FIXTURE_EVALUATOR_KEY",
                )
            },
        ):
            configured = create_configured_evaluator(
                {"evaluator_provider": "fixture"},
                environ={
                    "FIXTURE_EVALUATOR_KEY": "unselected-private",
                    "OVS_EVALUATOR_API_KEY": "evaluation-private",
                    "OPENAI_API_KEY": "generator-private",
                },
                env_file=self.env,
            )
        self.assertEqual("fixture", configured.provider)
        self.assertEqual("registered-model", configured.model)
        self.assertEqual("https://registered.example", configured.base_url)
        self.assertEqual("evaluation-private", configured.api_key)
        self.assertNotIn("evaluation-private", repr(configured))

    def test_unknown_provider_and_invalid_role_settings_fail_locally(self):
        for environment in (
            {"OVS_EVALUATOR_PROVIDER": "jev"},
            {"OVS_EVALUATOR_PROVIDER": "missing"},
            {"OVS_EVALUATOR_TIMEOUT_SECONDS": "NaN"},
            {"OVS_EVALUATOR_TIMEOUT_SECONDS": "0"},
            {"OVS_EVALUATOR_MAX_ATTEMPTS": "3.5"},
            {"OVS_EVALUATOR_MAX_ATTEMPTS": "0"},
            {"OVS_EVALUATOR_BASE_URL": "file:///bad"},
            {"OVS_EVALUATOR_BASE_URL": "https://key:private@example.org"},
            {"OVS_EVALUATOR_BASE_URL": "https://example.org:bad"},
            {"OVS_EVALUATOR_BASE_URL": "https://example.org/?key=private"},
        ):
            with (
                self.subTest(environment=environment),
                self.assertRaises(ConfigurationError),
            ):
                self.load(environ=environment)
        with self.assertRaisesRegex(ConfigurationError, "Unknown evaluator provider"):
            create_evaluator(EvaluatorConfig(provider="missing"))
        for environment in (
            {"OVS_EVALUATOR_MAX_ATTEMPTS": "6"},
            {"OVS_EVALUATOR_BASE_URL": "http://remote.example"},
        ):
            with (
                self.subTest(environment=environment),
                self.assertRaises(ConfigurationError),
            ):
                create_configured_evaluator(environ=environment, env_file=self.env)

    def test_factory_routes_only_evaluator_credential_and_overrides_to_http(self):
        from tests.test_typesafe_adapter import FakeTransport, response

        evaluator = create_configured_evaluator(
            {
                "evaluator_model": "jev-1.12.0",
                "evaluator_base_url": "https://gateway.example/tenant",
                "evaluator_timeout": 4.5,
                "evaluator_max_attempts": 1,
            },
            environ={
                "OVS_EVALUATOR_API_KEY": "evaluation-private",
                "OPENAI_API_KEY": "generator-private",
            },
            env_file=self.env,
        )
        transport = FakeTransport(
            response({"supported": {"type": "noul", "noul": 0.9}}, model="jev-1.12.0")
        )
        evaluator.transport = transport
        result = evaluator.evaluate(
            "Synthetic context.", noul={"supported": "Supported?"}
        )
        sent = transport.calls[0]
        self.assertEqual("https://gateway.example/tenant/v1/systemone", sent["url"])
        self.assertEqual("Bearer evaluation-private", sent["headers"]["Authorization"])
        self.assertEqual(4.5, sent["timeout"])
        self.assertEqual("jev-1.12.0", json.loads(sent["body"])["model"])
        self.assertEqual("typesafe", result.metadata.provider)
        self.assertEqual("jev-1.12.0", result.metadata.requested_model)
        self.assertEqual("jev-1.12.0", result.metadata.returned_model)
        self.assertNotIn("evaluation-private", repr(result))
        self.assertNotIn("generator-private", repr(sent))

    def test_generation_and_transcription_do_not_resolve_unrelated_evaluator(self):
        config = load_provider_config(
            environ={
                "OVS_EVALUATOR_PROVIDER": "missing",
                "OVS_EVALUATOR_TIMEOUT_SECONDS": "invalid",
            },
            env_file=self.env,
        )
        self.assertEqual("ollama", config.llm.provider)
        self.assertEqual("whisper_local", config.stt.provider)

    def test_generator_and_evaluator_credentials_are_independent(self):
        environment = {
            "OVS_LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "generator-private",
            "OVS_EVALUATOR_API_KEY": "evaluation-private",
        }
        generation = load_provider_config(environ=environment, env_file=self.env)
        evaluation = self.load(environ=environment)
        self.assertEqual("generator-private", generation.llm.api_key)
        self.assertEqual("evaluation-private", evaluation.api_key)
        without_evaluator_key = self.load(
            environ={"OPENAI_API_KEY": "generator-private"}
        )
        self.assertIsNone(without_evaluator_key.api_key)


if __name__ == "__main__":
    unittest.main()
