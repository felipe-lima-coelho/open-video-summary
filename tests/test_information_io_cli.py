"""Report persistence, opt-in configuration and actual CLI integration offline."""

import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from open_video_summary import __main__ as cli
from open_video_summary.core.summarizers.base import Summarizer
from open_video_summary.core.summarizers.information_analysis import InformationAnalyzer
from open_video_summary.core.summarizers.information_config import (
    InformationAnalysisConfig,
    configured_information_analyzer,
)
from open_video_summary.core.summarizers.information_contracts import capture_snapshot
from open_video_summary.core.summarizers.information_io import (
    save_information_report,
    validate_information_destination,
)
from open_video_summary.errors import ConfigurationError
from open_video_summary.parsers.video import VideoDumper
from open_video_summary.utils.config import PROJECT_DIR
from tests.test_information_analysis import (
    ScriptedGenerator,
    SyntheticEvaluator,
    candidate,
    videos,
)


class InformationIOTests(unittest.TestCase):
    def setUp(self):
        (PROJECT_DIR / "outputs").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            dir=PROJECT_DIR / "outputs", prefix="information-test-"
        )
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        text = "O prazo é 30 dias, segundo a palestrante."
        source = videos([text])
        self.report = InformationAnalyzer(
            ScriptedGenerator(
                {
                    ("v0:s0", "direct"): [
                        candidate("v0:s0", text, attribution="palestrante")
                    ]
                }
            ),
            SyntheticEvaluator(),
            InformationAnalysisConfig(qa_enabled=False),
        ).analyze(capture_snapshot(source))

    def test_atomic_utf8_json_csv_portable_paths_and_no_overwrites(self):
        path, csv_path = self.root / "report.json", self.root / "table.csv"
        save_information_report(self.report, path, csv_path=csv_path)
        text = path.read_text(encoding="utf-8")
        self.assertIn("palestrante", text)
        exported = json.loads(text)
        self.assertEqual(
            "data/raw/absent.mp4", exported["snapshot"]["source"][0]["path"]
        )
        self.assertIn("O prazo é", csv_path.read_text(encoding="utf-8"))
        self.assertFalse(any(self.root.glob("*.tmp")))
        before = path.read_bytes()
        with self.assertRaises(ConfigurationError):
            save_information_report(self.report, path)
        self.assertEqual(before, path.read_bytes())

    def test_aliases_to_dataset_audit_video_and_handler_are_protected(self):
        path = self.root / "dataset.json"
        alias = self.root / "nested" / ".." / "dataset.json"
        for artifact in (
            "dataset.json",
            "summary_audit.json",
            "summary_handler.json",
            "summary.mp4",
            "source.mp4",
        ):
            protected = self.root / artifact
            with self.subTest(artifact=artifact), self.assertRaises(ConfigurationError):
                validate_information_destination(protected, reserved=[protected])
        with self.assertRaises(ConfigurationError):
            validate_information_destination(alias, reserved=[path])
        with self.assertRaises(ConfigurationError):
            validate_information_destination("data/processed/should_not_write.json")
        with self.assertRaises(ConfigurationError):
            validate_information_destination(self.root / "wrong.mp4")
        path.write_text("input", encoding="utf-8")
        hardlink = self.root / "hardlink.json"
        os.link(path, hardlink)
        with self.assertRaises(ConfigurationError):
            validate_information_destination(hardlink, reserved=[path])
        self.assertEqual("input", path.read_text(encoding="utf-8"))

    def test_collision_fallback_records_failure_without_modifying_input(self):
        source = videos(["Bom dia."])
        dataset = self.root / "dataset.json"
        VideoDumper.dump_videos_to_json(source, str(dataset))
        before = dataset.read_bytes()
        summarizer = Summarizer([])
        analyzer = InformationAnalyzer(ScriptedGenerator(), SyntheticEvaluator())
        with self.assertWarns(RuntimeWarning):
            summarizer.summarize(
                source,
                save_output=False,
                information_analyzer=analyzer,
                information_output_path=str(dataset),
                information_input_path=str(dataset),
            )
        self.assertEqual(before, dataset.read_bytes())
        self.assertEqual("failed", summarizer.last_information_report.status)
        self.assertEqual("ConfigurationError", summarizer.last_information_error)
        self.assertNotEqual(dataset, summarizer.last_information_path)
        if summarizer.last_information_path:
            self.addCleanup(summarizer.last_information_path.unlink)
            self.assertEqual(
                "failed",
                json.loads(
                    summarizer.last_information_path.read_text(encoding="utf-8")
                )["status"],
            )

    def test_library_default_report_destination_is_unique_per_call(self):
        summarizer = Summarizer([])
        analyzer = InformationAnalyzer(ScriptedGenerator(), SyntheticEvaluator())
        paths = []
        for _ in range(2):
            summarizer.summarize(
                videos(["Olá."]),
                video_output_path=str(self.root / f"run{len(paths)}.mp4"),
                handler_output_path=str(self.root / f"handler{len(paths)}.json"),
                information_analyzer=analyzer,
            )
            paths.append(summarizer.last_information_path)
            self.addCleanup(summarizer.last_information_path.unlink)
        self.assertNotEqual(paths[0], paths[1])
        self.assertTrue(all(path.is_file() for path in paths))


class InformationConfigurationAndCLITests(unittest.TestCase):
    def setUp(self):
        (PROJECT_DIR / "outputs").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            dir=PROJECT_DIR / "outputs", prefix="information-cli-"
        )
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env_file = self.root / ".env"

    def test_config_precedence_optional_keys_bounds_and_no_secret_serialization(self):
        self.env_file.write_text(
            "TYPESAFE_API_KEY=file-secret\nOVS_INFORMATION_QA=false\nOVS_INFORMATION_MAX_CALLS=4\n",
            encoding="utf-8",
        )
        analyzer = configured_information_analyzer(
            {"information_max_calls": 9},
            environ={
                "TYPESAFE_API_KEY": "process-secret",
                "OVS_INFORMATION_MAX_CALLS": "7",
            },
            env_file=self.env_file,
        )
        self.assertEqual(9, analyzer.config.max_calls)
        self.assertFalse(analyzer.config.qa_enabled)
        self.assertEqual("process-secret", analyzer.evaluator.config.api_key)
        self.assertNotIn("process-secret", repr(analyzer.evaluator.config))
        missing = configured_information_analyzer(
            environ={"TYPESAFE_API_KEY": ""}, env_file=self.env_file
        )
        report = missing.analyze(capture_snapshot(videos(["Texto."])))
        self.assertEqual("failed", report.status)
        self.assertNotIn("secret", json.dumps(report.to_dict()))
        for values in (
            {"information_max_calls": 0},
            {"information_rounds": -1},
            {"information_equivalence": float("nan")},
            {"information_context_chars": 1},
        ):
            with self.subTest(values=values), self.assertRaises(ConfigurationError):
                configured_information_analyzer(
                    values, environ={}, env_file=self.env_file
                )

    def test_text_only_cli_runs_without_source_video_or_classifier_or_ffmpeg(self):
        dataset, path, csv_path = (
            self.root / "dataset.json",
            self.root / "report.json",
            self.root / "table.csv",
        )
        VideoDumper.dump_videos_to_json(videos(["Bom dia."]), str(dataset))
        analyzer = InformationAnalyzer(ScriptedGenerator(), SyntheticEvaluator())
        with (
            patch(
                "open_video_summary.core.summarizers.information_config.configured_information_analyzer",
                return_value=analyzer,
            ),
            patch.object(cli, "_cpu_settings"),
            patch("shutil.which", return_value=None),
            patch.object(
                cli.ModelPaths, "SUBJECTIVITY_CLASSIFIER", str(self.root / "absent")
            ),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            cli.main(
                [
                    "analyze-information",
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(path),
                    "--information-csv",
                    str(csv_path),
                ]
            )
        self.assertEqual(
            "completed", json.loads(path.read_text(encoding="utf-8"))["status"]
        )
        self.assertIn("Information analysis: completed", output.getvalue())
        self.assertTrue(csv_path.exists())

    def test_cli_missing_key_saves_failed_report_and_exits_nonzero_without_network(
        self,
    ):
        dataset, path = self.root / "dataset.json", self.root / "failed.json"
        VideoDumper.dump_videos_to_json(videos(["Regra."]), str(dataset))
        analyzer = configured_information_analyzer(
            environ={}, env_file=self.root / "missing.env"
        )
        with (
            patch(
                "open_video_summary.core.summarizers.information_config.configured_information_analyzer",
                return_value=analyzer,
            ),
            patch.object(cli, "_cpu_settings"),
            patch(
                "urllib.request.urlopen",
                side_effect=AssertionError("Unexpected live request"),
            ),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as exit,
        ):
            cli.main(
                [
                    "analyze-information",
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(path),
                ]
            )
        self.assertEqual(1, exit.exception.code)
        exported = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual("failed", exported["status"])
        self.assertFalse(exported["counts"]["valid_zero"])

    def test_primary_summarize_cli_calls_inventory_before_introduction_and_preserves_audit(
        self,
    ):
        source_path = self.root / "source.mp4"
        source_path.write_bytes(b"fixture")
        source = videos(["Uma informação."])
        source[0].path = str(source_path)
        source[0].segments[0].video_path = str(source_path)
        events = []

        class Introduction:
            name = "Introduction"

            def evaluate(self, handler):
                events.append("Introduction")
                self.assert_empty = len(handler.output) == 0
                handler.add_output_segment(handler.source[0].segments[0], self.name)
                return handler

        intro = Introduction()
        quality = SimpleNamespace(
            name="QualityPick",
            visual_threads=1,
            last_profile={"status": "fixture"},
            evaluate=lambda handler: handler,
        )
        summarizer = Summarizer([intro, quality])
        actual = InformationAnalyzer(ScriptedGenerator(), SyntheticEvaluator())

        def observe(snapshot):
            events.append("inventory")
            return actual.analyze(snapshot)

        analyzer = SimpleNamespace(analyze=Mock(side_effect=observe))
        dataset, info, summary = (
            self.root / "dataset.json",
            self.root / "inventory.json",
            self.root / "summary.mp4",
        )
        VideoDumper.dump_videos_to_json(source, str(dataset))
        with (
            patch.dict(
                "sys.modules",
                {
                    "open_video_summary.core.summarizers": SimpleNamespace(
                        HSMVideoSumm=summarizer
                    ),
                    "torch": SimpleNamespace(set_num_threads=Mock()),
                },
            ),
            patch(
                "open_video_summary.core.summarizers.information_config.configured_information_analyzer",
                return_value=analyzer,
            ),
            patch.object(cli.ModelPaths, "SUBJECTIVITY_CLASSIFIER", str(self.root)),
            patch("cv2.setNumThreads"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            cli.main(
                [
                    "summarize",
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(summary),
                    "--no-render",
                    "--information-report",
                    str(info),
                ]
            )
        self.assertEqual(["inventory", "Introduction"], events)
        self.assertTrue(intro.assert_empty)
        analyzer.analyze.assert_called_once()
        exported = json.loads(info.read_text(encoding="utf-8"))
        self.assertEqual("before_introduction", exported["snapshot"]["stage_id"])
        self.assertEqual(
            ["Introduction", "QualityPick"],
            exported["snapshot"]["selection_stage_order"],
        )
        audit = json.loads(
            summary.with_name("summary_audit.json").read_text(encoding="utf-8")
        )
        self.assertNotIn("information", audit)
        self.assertNotIn("InformationAnalyzer", audit["criteria"])

    def test_disabled_summarize_and_other_commands_do_not_configure_information(self):
        with (
            patch(
                "open_video_summary.core.summarizers.information_config.configured_information_analyzer",
                side_effect=AssertionError("Information config unexpectedly loaded"),
            ),
            patch.object(cli, "_summarize") as summarize,
            patch.object(cli, "_doctor") as doctor,
            patch.object(cli, "_cpu_settings"),
        ):
            cli.main(["summarize"])
            cli.main(["doctor"])
        self.assertFalse(summarize.call_args.args[0].analyze_information)
        doctor.assert_called_once()

    def test_import_information_module_in_fresh_process_has_no_hsm_classifier_import(
        self,
    ):
        code = "import sys; import open_video_summary.core.summarizers.information_analysis; assert 'open_video_summary.classifiers.text' not in sys.modules; assert 'tensorflow' not in sys.modules"
        result = subprocess.run(
            [str(PROJECT_DIR / ".venv/Scripts/python.exe"), "-c", code],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
        )
        self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
