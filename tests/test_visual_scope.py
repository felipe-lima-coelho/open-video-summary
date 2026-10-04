"""Visual scope configuration stays independent of provider initialization."""

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from open_video_summary import __main__ as cli
from open_video_summary.core.selection_criteria.quality import QualityPick
from open_video_summary.errors import ConfigurationError
from open_video_summary.utils.providers import load_visual_scope


class VisualScopeConfigurationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.env_file = self.root / ".env"

    def load(self, value=None, environ=None):
        return load_visual_scope(
            value, environ={} if environ is None else environ, env_file=self.env_file
        )

    def test_cli_process_file_and_default_precedence(self):
        self.assertEqual("segment", self.load())
        self.env_file.write_text("OVS_VISUAL_SCOPE=video\n", encoding="utf-8")
        self.assertEqual("video", self.load())
        self.assertEqual("segment", self.load(environ={"OVS_VISUAL_SCOPE": "segment"}))
        self.assertEqual("video", self.load("video", {"OVS_VISUAL_SCOPE": "invalid"}))
        self.assertEqual("video", self.load(environ={"OVS_VISUAL_SCOPE": " VIDEO "}))

    def test_blank_values_are_absent_and_process_masks_file(self):
        self.env_file.write_text("OVS_VISUAL_SCOPE=\n", encoding="utf-8")
        self.assertEqual("segment", self.load())
        self.env_file.write_text("OVS_VISUAL_SCOPE=video\n", encoding="utf-8")
        self.assertEqual("segment", self.load(environ={"OVS_VISUAL_SCOPE": " "}))
        self.assertEqual("segment", self.load(" "))

    def test_invalid_values_fail_without_reading_provider_configuration(self):
        for value in ("full_video", "interval", "0", "segment,video"):
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                self.load(environ={"OVS_VISUAL_SCOPE": value})
        self.env_file.write_text("OPENAI_API_KEY=fixture\nOVS_LLM_PROVIDER=invalid\nOVS_VISUAL_SCOPE=video\n")
        with patch("open_video_summary.utils.providers.load_provider_config", side_effect=AssertionError):
            self.assertEqual("video", self.load())

    def test_resolver_preserves_environment_and_uses_root_from_foreign_cwd(self):
        environment = {"OVS_VISUAL_SCOPE": "segment"}
        self.assertEqual("segment", self.load(environ=environment))
        self.assertEqual({"OVS_VISUAL_SCOPE": "segment"}, environment)
        self.env_file.write_text("OVS_VISUAL_SCOPE=video\n")
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as elsewhere:
            os.chdir(elsewhere)
            try:
                with patch("open_video_summary.utils.providers.PROJECT_DIR", self.root):
                    self.assertEqual("video", load_visual_scope(environ={}))
            finally:
                os.chdir(original)

    def test_cli_override_and_parser_default(self):
        self.assertIsNone(cli.build_parser().parse_args(["summarize"]).visual_scope)
        with (
            patch.dict(os.environ, {"OVS_VISUAL_SCOPE": "invalid"}),
            patch.object(cli, "_cpu_settings"),
            patch.object(cli, "_summarize") as summarize,
        ):
            cli.main(["summarize", "--visual-scope", "video", "--no-render"])
        self.assertEqual("video", summarize.call_args.args[0].visual_scope)

    def test_help_and_other_commands_do_not_resolve_visual_scope(self):
        with (
            patch.object(cli, "load_visual_scope", side_effect=AssertionError),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            cli.main(["summarize", "--help"])
        with (
            patch.dict(os.environ, {"OVS_VISUAL_SCOPE": "invalid"}),
            patch.object(cli, "_cpu_settings"),
            patch.object(cli, "_doctor") as doctor,
        ):
            cli.main(["doctor"])
        doctor.assert_called_once()

    def test_library_scope_defaults_and_validation(self):
        self.assertEqual("segment", QualityPick("fixture").visual_scope)
        self.assertEqual("video", QualityPick("fixture", visual_scope="video").visual_scope)
        with self.assertRaisesRegex(ValueError, "visual_scope"):
            QualityPick("fixture", visual_scope="invalid")


if __name__ == "__main__":
    unittest.main()
