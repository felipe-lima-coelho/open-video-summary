"""Real core v2 to offline harness integration with scripted synthetic decisions."""

import copy
import json
import socket
import unittest
from unittest.mock import patch

from test_information_analysis import ScriptedGenerator, candidate
from test_information_inventory import InventoryEvaluator
from open_video_summary.core.summarizers.information_analysis import InformationAnalyzer
from open_video_summary.core.summarizers.information_config import (
    InformationAnalysisConfig,
)
from open_video_summary.core.summarizers.information_contracts import capture_snapshot
from open_video_summary.entities.video import Video, VideoSegment
from open_video_summary.evaluation import evaluate_information_report
from open_video_summary.utils.paths import project_path


class InformationReferenceCoreIntegrationTests(unittest.TestCase):
    def test_real_schema7_core_report_matches_frozen_separate_synthetic_reference(self):
        reference = json.loads(
            project_path(
                "data/evaluation/information/synthetic_contrast_cases_v1.json"
            ).read_text(encoding="utf-8")
        )
        source_video = reference["snapshot"]["source"][0]
        segments = source_video["segments"]
        source = [
            Video(
                source_video["name"],
                source_video["path"],
                topics=source_video["topics"],
                segments=[
                    VideoSegment(
                        row["content"],
                        row["start"],
                        row["end"],
                        row["order"],
                        row["video_topic"],
                        row["global_topic"],
                        video_path=row["video_path"],
                    )
                    for row in segments
                ],
            )
        ]
        by_id = {row["id"]: row for row in segments}
        items = {}
        for unit in reference["units"]:
            occurrence = unit["occurrences"][0]
            assertion = occurrence["assertion_evidence"][0]
            contexts = [
                dict(anchor, role="context")
                for anchor in occurrence["context_evidence"]
            ]
            qualifiers = unit["qualifiers"]
            value = candidate(
                occurrence["segment_id"],
                by_id[occurrence["segment_id"]]["content"],
                text=unit["text"],
                quote=assertion["quote"],
                offset=assertion["start_char"],
                contexts=contexts,
                attribution=qualifiers["attribution"],
                negated=qualifiers["negated"],
                conditions=qualifiers["conditions"],
                quantities=qualifiers["quantities"],
                modality=qualifiers["modality"],
                unit_type=unit["unit_type"],
            )
            items.setdefault((occurrence["segment_id"], "direct"), []).append(value)
        original = copy.deepcopy(reference)
        with patch.object(
            socket,
            "socket",
            side_effect=AssertionError("Network disabled for integration fixture."),
        ):
            report = InformationAnalyzer(
                ScriptedGenerator(items),
                InventoryEvaluator(),
                InformationAnalysisConfig(
                    experimental_mode="direct",
                    concurrency=1,
                    qa_enabled=False,
                    max_calls=256,
                    max_coverage_rounds=0,
                    max_literal_repairs=0,
                    max_relation_adjudications=0,
                    direct_window_chars=4000,
                    pair_batch_size=1,
                ),
            ).analyze(capture_snapshot(source))
            result = evaluate_information_report(report, reference)
        self.assertEqual(original, reference)
        self.assertEqual(
            (7, "contextual-propositions-v2", "completed"),
            (report.schema_version, report.protocol_version, report.status),
        )
        self.assertEqual(5, report.counts.unique_units)
        serialized = report.to_dict()
        for unit in serialized["units"]:
            self.assertEqual("verified", unit["qualifier_state"])
            self.assertIsInstance(unit["qualifiers"], dict)
        self.assertEqual(5, result["metrics"]["unit_coverage"]["covered"])
        self.assertEqual(5, result["metrics"]["occurrence_coverage"]["covered"])
        self.assertEqual(
            0,
            result["metrics"]["traceability"]["report_occurrences_with_literal_errors"],
        )
        self.assertTrue(result["provenance"]["matched_protocol"])
        self.assertTrue(result["provenance"]["diagnostic_only"])
        self.assertEqual("direct", result["experiment"]["mode"])


if __name__ == "__main__":
    unittest.main()
