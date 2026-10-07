"""Report persistence, opt-in configuration and actual CLI integration offline."""

import contextlib
import csv
import io
import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from open_video_summary import __main__ as cli
from open_video_summary.adapters.factory import (
    EVALUATOR_PROVIDERS,
    LLM_PROVIDERS,
    ProviderDefinition,
)
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
from tests import test_information_analysis as analysis_fixtures


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
        self.assertFalse(exported["counts"]["occurrences_provisional"])
        self.assertFalse(exported["counts"]["counts_provisional"])
        row = next(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8"))))
        self.assertEqual("False", row["counts_provisional"])
        self.assertEqual(
            "source_anchor_group", exported["occurrences"][0]["alignment_state"]
        )
        self.assertFalse(any(self.root.glob("*.tmp")))
        before = path.read_bytes()
        with self.assertRaises(ConfigurationError):
            save_information_report(self.report, path)
        self.assertEqual(before, path.read_bytes())

    def test_unknown_canonical_qualifiers_export_as_null_with_proposal_audit(self):
        text = "O Google decidiu não atualizar mais os aplicativos dos celulares da Huawei."
        raw = candidate("v0:s0", text, negated=True)
        evaluator, _ = analysis_fixtures.InformationEvaluationRegressionTests().evaluator({"attribution_annotation": 0.57})
        report = InformationAnalyzer(
            ScriptedGenerator({("v0:s0", "direct"): [raw]}), evaluator,
            InformationAnalysisConfig(qa_enabled=False),
        ).analyze(capture_snapshot(videos([text])))
        path, csv_path = self.root / "unknown.json", self.root / "unknown.csv"
        save_information_report(report, path, csv_path=csv_path)
        exported = json.loads(path.read_text(encoding="utf-8"))
        unit, record = exported["units"][0], exported["candidates"][0]
        self.assertEqual(6, exported["schema_version"])
        self.assertIsNone(unit["qualifiers"])
        self.assertEqual("unknown", unit["qualifier_state"])
        self.assertEqual(record["id"], unit["representative_candidate_id"])
        self.assertEqual("uncertain", record["annotation_state"])
        self.assertEqual(raw["qualifiers"], record["candidate"]["proposed_qualifiers"])
        row = next(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8"))))
        self.assertEqual("null", row["qualifiers"])
        self.assertEqual("unknown", row["qualifier_state"])
        self.assertEqual(record["id"], row["representative_candidate_id"])

    def test_verified_canonical_qualifiers_export_with_their_representative(self):
        path, csv_path = self.root / "verified.json", self.root / "verified.csv"
        save_information_report(self.report, path, csv_path=csv_path)
        exported = json.loads(path.read_text(encoding="utf-8"))
        unit = exported["units"][0]
        self.assertEqual("verified", unit["qualifier_state"])
        self.assertEqual("palestrante", unit["qualifiers"]["attribution"])
        row = next(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8"))))
        self.assertEqual("verified", row["qualifier_state"])
        self.assertEqual(unit["qualifiers"], json.loads(row["qualifiers"]))

    def test_overlapping_occurrence_groups_export_provisional_state_and_cli_label(self):
        text = "Backup is daily. I repeat: backup is daily."
        raw = [
            candidate(
                "v0:s0",
                text,
                text="Backup is daily.",
                quote=text[start:end],
                offset=start,
            )
            for start, end in ((0, 26), (17, 43))
        ]
        report = InformationAnalyzer(
            ScriptedGenerator({("v0:s0", "direct"): raw}),
            SyntheticEvaluator(),
            InformationAnalysisConfig(qa_enabled=False),
        ).analyze(capture_snapshot(videos([text])))
        path, csv_path = self.root / "pending.json", self.root / "pending.csv"
        save_information_report(report, path, csv_path=csv_path)
        exported = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual("partial", exported["status"])
        self.assertEqual(1, exported["counts"]["unique_units"])
        self.assertEqual(2, exported["counts"]["occurrences"])
        self.assertTrue(exported["counts"]["occurrences_provisional"])
        self.assertTrue(exported["counts"]["counts_provisional"])
        self.assertTrue(exported["counts"]["by_segment"][0]["occurrences_provisional"])
        self.assertTrue(exported["counts"]["by_video"][0]["occurrences_provisional"])
        self.assertEqual(
            ["unresolved_overlap", "unresolved_overlap"],
            [item["alignment_state"] for item in exported["occurrences"]],
        )
        rows = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8"))))
        self.assertEqual(2, len(rows))
        self.assertTrue(all(row["status"] == "partial" for row in rows))
        self.assertTrue(
            all(row["alignment_state"] == "unresolved_overlap" for row in rows)
        )
        with contextlib.redirect_stdout(io.StringIO()) as output:
            cli._print_information_report(report, path)
        self.assertIn("1 provisional unit", output.getvalue())
        self.assertIn("2 provisional occurrence evidence groups", output.getvalue())
        self.assertIn("alignment pending", output.getvalue())

    def test_unexamined_pair_count_state_is_consistent_in_json_csv_and_cli(self):
        text = "Backup is performed daily."
        raw = [
            candidate("v0:s0", text, text=claim, quote=text)
            for claim in (text, "The backup runs every day.")
        ]
        report = InformationAnalyzer(
            ScriptedGenerator({("v0:s0", "direct"): raw}),
            SyntheticEvaluator(),
            InformationAnalysisConfig(qa_enabled=False, max_pair_comparisons=0),
        ).analyze(capture_snapshot(videos([text])))
        path, csv_path = self.root / "unexamined.json", self.root / "unexamined.csv"
        save_information_report(report, path, csv_path=csv_path)
        exported = json.loads(path.read_text(encoding="utf-8"))
        rows = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8"))))

        self.assertEqual("partial", exported["status"])
        self.assertTrue(exported["counts"]["counts_provisional"])
        self.assertFalse(exported["counts"]["occurrences_provisional"])
        self.assertFalse(exported["counts"]["by_segment"][0]["occurrences_provisional"])
        self.assertEqual(
            ["source_anchor_group", "source_anchor_group"],
            [item["alignment_state"] for item in exported["occurrences"]],
        )
        self.assertEqual(2, len(rows))
        self.assertTrue(all(row["counts_provisional"] == "True" for row in rows))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            cli._print_information_report(report, path)
        self.assertIn("2 provisional units", output.getvalue())
        self.assertIn("2 provisional occurrence evidence groups", output.getvalue())
        self.assertNotIn("alignment pending", output.getvalue())

    def test_noncompleted_report_status_forces_aggregate_counts_provisional(self):
        partial = replace(self.report, status="partial")
        self.assertTrue(partial.counts.counts_provisional)
        self.assertTrue(partial.to_dict()["counts"]["counts_provisional"])

    def test_distinct_units_with_unknown_subtype_remain_identified_in_json_csv_and_cli(self):
        signals = {"left_entails_right": .46, "right_entails_left": .08, "incompatible": .08,
                   "correction_left": .03, "correction_right": .04, "same_complete_meaning": .03}
        class SubtypeEvaluator(SyntheticEvaluator):
            def evaluate(self, context, noul=None, choice=None):
                result = super().evaluate(context, noul=noul, choice=choice)
                if "same_complete_meaning" in (noul or {}):
                    return replace(result, noul=tuple(replace(item, probability=signals[item.id]) for item in result.noul))
                return result
        claims = ("O italiano acha a medida absurda.", "O italiano usa um smartphone chinês.")
        text = " ".join(claims)
        report = InformationAnalyzer(
            ScriptedGenerator({("v0:s0", "direct"): [candidate("v0:s0", text, quote=claim) for claim in claims]}),
            SubtypeEvaluator(relation_probability=.89), InformationAnalysisConfig(qa_enabled=False),
        ).analyze(capture_snapshot(videos([text])))
        path, csv_path = self.root / "subtype.json", self.root / "subtype.csv"
        save_information_report(report, path, csv_path=csv_path)
        exported = json.loads(path.read_text(encoding="utf-8"))
        rows = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8"))))
        self.assertEqual("uncertain", exported["relations"][0]["relation"])
        self.assertEqual("distinct", exported["relations"][0]["equivalence_state"])
        self.assertFalse(exported["counts"]["counts_provisional"])
        self.assertTrue(all(row["counts_provisional"] == "False" for row in rows))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            cli._print_information_report(report, path)
        self.assertIn("2 identified units", output.getvalue())
        self.assertIn("Descriptive relation subtypes pending: 1", output.getvalue())

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
        settings_root = patch(
            "open_video_summary.utils.providers.PROJECT_DIR", self.root
        )
        settings_root.start()
        self.addCleanup(settings_root.stop)
        process_settings = patch.dict(os.environ, {}, clear=True)
        process_settings.start()
        self.addCleanup(process_settings.stop)

    def test_config_precedence_optional_keys_bounds_and_no_secret_serialization(self):
        self.env_file.write_text(
            "OVS_EVALUATOR_API_KEY=file-secret\nOVS_INFORMATION_QA=false\nOVS_INFORMATION_MAX_CALLS=4\n",
            encoding="utf-8",
        )
        analyzer = configured_information_analyzer(
            {"information_max_calls": 9},
            environ={
                "OVS_EVALUATOR_API_KEY": "process-secret",
                "OVS_INFORMATION_MAX_CALLS": "7",
            },
            env_file=self.env_file,
        )
        self.assertEqual(9, analyzer.config.max_calls)
        self.assertFalse(analyzer.config.qa_enabled)
        self.assertEqual("process-secret", analyzer.evaluator.config.api_key)
        self.assertNotIn("process-secret", repr(analyzer.evaluator.config))
        missing = configured_information_analyzer(
            environ={"OVS_EVALUATOR_API_KEY": ""}, env_file=self.env_file
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

    def test_registered_evaluator_is_selected_without_algorithm_changes_or_key_leakage(
        self,
    ):
        text = "O backup é diário."
        generator = ScriptedGenerator({("v0:s0", "direct"): [candidate("v0:s0", text)]})
        selected = []

        def construct(config):
            evaluator = SyntheticEvaluator()
            evaluator.config = config
            selected.append(evaluator)
            return evaluator

        with (
            patch.dict(
                LLM_PROVIDERS,
                {
                    "ollama": ProviderDefinition(
                        lambda config: generator, "fixture", "http://localhost:11434"
                    )
                },
            ),
            patch.dict(
                EVALUATOR_PROVIDERS,
                {
                    "fixture": ProviderDefinition(
                        construct,
                        "fixture-evaluator",
                        "https://evaluator.example",
                        "OVS_EVALUATOR_API_KEY",
                    )
                },
            ),
        ):
            analyzer = configured_information_analyzer(
                {"evaluator_provider": "fixture", "information_qa": False},
                environ={
                    "OVS_EVALUATOR_API_KEY": "evaluator-private",
                    "OPENAI_API_KEY": "generator-private",
                },
                env_file=self.env_file,
            )
            report = analyzer.analyze(capture_snapshot(videos([text])))
        self.assertIs(selected[0], analyzer.evaluator)
        self.assertEqual("evaluator-private", analyzer.evaluator.config.api_key)
        self.assertEqual("fixture-evaluator", analyzer.evaluator.config.model)
        self.assertEqual("completed", report.status)
        exported = report.to_dict()
        self.assertEqual("fixture", exported["metadata"]["evaluator_provider"])
        self.assertEqual("fixture", exported["metadata"]["evaluator"]["provider"])
        self.assertTrue(
            all(
                call.provider == "fixture"
                for call in report.calls
                if not call.operation.startswith("extract_")
            )
        )
        self.assertNotIn("evaluator-private", json.dumps(exported))
        self.assertNotIn("generator-private", json.dumps(exported))

    def test_cli_evaluator_options_reach_selected_factory_with_precedence(self):
        dataset, path = self.root / "input.json", self.root / "report.json"
        VideoDumper.dump_videos_to_json(videos(["Bom dia."]), str(dataset))
        self.env_file.write_text(
            "OVS_EVALUATOR_MODEL=file-model\nOVS_EVALUATOR_API_KEY=file-key\n",
            encoding="utf-8",
        )
        constructed = []

        def construct(config):
            evaluator = SyntheticEvaluator()
            evaluator.config = config
            constructed.append(evaluator)
            return evaluator

        with (
            patch.dict(
                LLM_PROVIDERS,
                {
                    "ollama": ProviderDefinition(
                        lambda config: ScriptedGenerator(),
                        "fixture",
                        "http://localhost:11434",
                    )
                },
            ),
            patch.dict(
                EVALUATOR_PROVIDERS,
                {
                    "fixture": ProviderDefinition(
                        construct,
                        "fixture-evaluator",
                        "https://default.example",
                        "OVS_EVALUATOR_API_KEY",
                    )
                },
            ),
            patch.dict(
                os.environ,
                {
                    "OVS_EVALUATOR_PROVIDER": "unknown",
                    "OVS_EVALUATOR_MODEL": "process-model",
                    "OVS_EVALUATOR_API_KEY": "process-key",
                },
            ),
            patch.object(cli, "_cpu_settings"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            cli.main(
                [
                    "analyze-information",
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(path),
                    "--evaluator-provider",
                    "fixture",
                    "--evaluator-model",
                    "cli-model",
                    "--evaluator-base-url",
                    "https://cli.example",
                    "--evaluator-timeout",
                    "7.5",
                    "--evaluator-max-attempts",
                    "1",
                ]
            )
        config = constructed[0].config
        self.assertEqual("fixture", config.provider)
        self.assertEqual("cli-model", config.model)
        self.assertEqual("process-key", config.api_key)
        self.assertEqual("https://cli.example", config.base_url)
        self.assertEqual(7.5, config.timeout_seconds)
        self.assertEqual(1, config.max_attempts)
        self.assertNotIn("process-key", path.read_text(encoding="utf-8"))

    def test_unknown_evaluator_cli_fails_configuration_before_generation_or_network(
        self,
    ):
        dataset = self.root / "input.json"
        VideoDumper.dump_videos_to_json(videos(["Regra."]), str(dataset))
        with (
            patch.object(cli, "_cpu_settings"),
            patch(
                "open_video_summary.adapters.factory.create_llm",
                side_effect=AssertionError(
                    "Generator constructed before evaluator configuration validation"
                ),
            ),
            patch(
                "urllib.request.urlopen",
                side_effect=AssertionError("Unexpected live request"),
            ),
            contextlib.redirect_stderr(io.StringIO()) as error,
            self.assertRaises(SystemExit) as exit,
        ):
            cli.main(
                [
                    "analyze-information",
                    "--dataset",
                    str(dataset),
                    "--evaluator-provider",
                    "missing",
                ]
            )
        self.assertEqual(1, exit.exception.code)
        self.assertIn("Unknown evaluator provider 'missing'", error.getvalue())
        self.assertEqual([dataset], list(self.root.glob("*.json")))

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
            patch.dict(
                os.environ,
                {
                    "OVS_EVALUATOR_PROVIDER": "unsupported",
                    "OVS_EVALUATOR_TIMEOUT_SECONDS": "invalid",
                },
            ),
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
        code = "import sys; import open_video_summary.core.summarizers.information_analysis; assert 'open_video_summary.classifiers.text' not in sys.modules; assert 'tensorflow' not in sys.modules; assert 'open_video_summary.adapters.typesafe' not in sys.modules"
        result = subprocess.run(
            [str(PROJECT_DIR / ".venv/Scripts/python.exe"), "-c", code],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
        )
        self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
