import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from open_video_summary import __main__ as cli
from open_video_summary.entities.video import Video, VideoSegment
from open_video_summary.errors import ConfigurationError
from open_video_summary.utils.providers import load_thread_count


class ThreadConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.env_file = self.root / ".env"

    def load(self, cli_value=None, environ=None):
        return load_thread_count(
            cli_value,
            environ={} if environ is None else environ,
            env_file=self.env_file,
        )

    def test_cli_process_dotenv_and_default_precedence(self):
        self.env_file.write_text("OVS_THREADS=6\n", encoding="utf-8")
        self.assertEqual(4, self.load(environ={"OVS_THREADS": "4"}))
        self.assertEqual(6, self.load(environ={}))
        self.assertEqual(8, self.load(8, {"OVS_THREADS": "invalid"}))

    def test_absent_and_blank_values_use_default_two(self):
        self.assertEqual(2, self.load(environ={}))
        self.env_file.write_text("OVS_THREADS=\n", encoding="utf-8")
        self.assertEqual(2, self.load(environ={}))
        self.env_file.write_text("OVS_THREADS=6\n", encoding="utf-8")
        # As with provider settings, a blank process value masks the file value.
        self.assertEqual(2, self.load(environ={"OVS_THREADS": " "}))

    def test_invalid_values_must_be_positive_integers(self):
        for value in ("abc", "1.5", "0", "-1", "NaN"):
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                self.load(environ={"OVS_THREADS": value})

    def test_resolving_does_not_change_process_environment(self):
        process = {"OVS_THREADS": "5"}
        self.assertEqual(5, self.load(environ=process))
        self.assertEqual({"OVS_THREADS": "5"}, process)

    def test_root_dotenv_is_used_from_another_working_directory(self):
        self.env_file.write_text("OVS_THREADS=7\n", encoding="utf-8")
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as elsewhere:
            os.chdir(elsewhere)
            try:
                with patch("open_video_summary.utils.providers.PROJECT_DIR", self.root):
                    self.assertEqual(7, load_thread_count(environ={}))
            finally:
                os.chdir(original)

    def test_help_does_not_resolve_environment_settings(self):
        with (
            patch(
                "open_video_summary.__main__.load_thread_count",
                side_effect=AssertionError("settings were resolved for --help"),
            ),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            cli.main(["--help"])

    def test_explicit_cli_value_overrides_invalid_process_setting(self):
        with (
            patch.dict(os.environ, {"OVS_THREADS": "invalid"}),
            patch.object(cli, "_cpu_settings") as cpu_settings,
            patch.object(cli, "_summarize") as summarize,
        ):
            cli.main(["--threads", "3", "summarize"])

        cpu_settings.assert_called_once_with(3)
        self.assertEqual(3, summarize.call_args.args[0].threads)

    def test_resolved_value_reaches_cpu_and_visual_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.mp4"
            source_path.write_bytes(b"fixture")
            output_path = root / "summary.mp4"
            segment = VideoSegment("text", 0, 1)
            source = Video("source", str(source_path), segments=[segment])
            summary = Video("summary", str(output_path), segments=[segment])
            criterion = SimpleNamespace(
                name="QualityPick", visual_threads=1, last_profile={"status": "ok"}
            )
            summarizer = SimpleNamespace(
                summarize=Mock(return_value=summary), selection_criteria=[criterion]
            )
            set_torch_threads = Mock()

            with (
                patch.dict(os.environ, {"OVS_THREADS": "4"}),
                patch.object(cli.ModelPaths, "SUBJECTIVITY_CLASSIFIER", str(root)),
                patch.dict(
                    "sys.modules",
                    {
                        "torch": SimpleNamespace(set_num_threads=set_torch_threads),
                        "open_video_summary.core.summarizers": SimpleNamespace(
                            HSMVideoSumm=summarizer
                        ),
                    },
                ),
                patch(
                    "open_video_summary.parsers.video.VideoLoader.load_videos_from_json",
                    return_value=[source],
                ),
                patch(
                    "open_video_summary.parsers.video.VideoDumper.dump_videos_to_json"
                ),
                patch("cv2.setNumThreads") as set_cv_threads,
            ):
                cli.main(["summarize", "--output", str(output_path), "--no-render"])
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "TF_NUM_INTRAOP_THREADS",
                    "TF_NUM_INTEROP_THREADS",
                ):
                    self.assertEqual("4", os.environ[name])

            set_cv_threads.assert_called_once_with(4)
            set_torch_threads.assert_called_once_with(4)
            self.assertEqual(4, criterion.visual_threads)
            summarizer.summarize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
