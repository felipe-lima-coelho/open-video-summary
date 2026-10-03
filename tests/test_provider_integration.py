"""Offline segmentation on synthetic multilingual service responses."""

import copy
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from open_video_summary import __main__ as cli
from open_video_summary.adapters.llm import OllamaAdapter, OpenAIAdapter
from open_video_summary.adapters.stt import _elevenlabs_result, _local_result
from open_video_summary.contracts import (
    GenerationResult,
    ServiceMetadata,
    TimedWord,
    TranscriptionResult,
)
from open_video_summary.core.segmenter.video_segmenter import (
    ClusteredVideoSegmenter,
    WordVideoSegmenter,
)
from open_video_summary.entities.video import Video, VideoSegment
from open_video_summary.core.selection_criteria.filtering import (
    VideoQuestionBasedFiltering,
)
from open_video_summary.handlers.summary import SummarySegmentHandler
from open_video_summary.errors import (
    AuthenticationError,
    ConfigurationError,
    NoSpeechError,
)
from open_video_summary.parsers.video import VideoDumper
from open_video_summary.utils.config import PROJECT_DIR
from open_video_summary.utils.paths import portable_path
from open_video_summary.utils.providers import (
    LLMConfig,
    ProviderConfig,
    STTConfig,
    load_provider_config,
)


WORDS = [
    {"text": "Primeiro", "start": 0.1, "end": 3.0},
    {"text": "assunto.", "start": 3.5, "end": 6.0},
    {"text": "Segundo", "start": 8.0, "end": 10.0},
    {"text": "assunto.", "start": 10.5, "end": 14.0},
]
TEXT = "Primeiro assunto. Segundo assunto."
LOCAL_TRANSCRIPT = {
    "text": TEXT,
    "language": "pt",
    "segments": [
        {"text": "Primeiro assunto.", "start": 0.0, "end": 6.1, "words": WORDS[:2]},
        {"text": "Segundo assunto.", "start": 7.9, "end": 14.2, "words": WORDS[2:]},
    ],
}
SCRIBE_TRANSCRIPT = {
    "text": TEXT,
    "language_code": "por",
    "words": [dict(word, type="word") for word in WORDS],
}


def openai_response(value):
    return {
        "status": "completed",
        "model": "gpt-6-luna-snapshot",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": value}]}
        ],
    }


def llm_adapter(provider, values):
    if provider == "ollama":
        client = SimpleNamespace(
            generate=Mock(
                side_effect=[{"response": value, "model": "gemma2"} for value in values]
            ),
            list=Mock(return_value={"models": [{"name": "gemma2"}]}),
        )
        return OllamaAdapter(client=client, sleep=Mock()), client.generate
    create = Mock(side_effect=[openai_response(value) for value in values])
    config = LLMConfig(
        provider="openai",
        model="gpt-6-luna",
        reasoning_effort="high",
        base_url="https://api.openai.com/v1",
    )
    return (
        OpenAIAdapter(
            config=config,
            client=SimpleNamespace(responses=SimpleNamespace(create=create)),
            sleep=Mock(),
        ),
        create,
    )


def speech_adapter(provider, fixture=None):
    if provider == "whisper_local":
        result = _local_result(copy.deepcopy(fixture or LOCAL_TRANSCRIPT))
    else:
        result = _elevenlabs_result(copy.deepcopy(fixture or SCRIBE_TRANSCRIPT))
    metadata = ServiceMetadata(
        provider=provider,
        requested_model="fixture-model",
        reported_language=result.language,
    )
    return SimpleNamespace(
        records=[metadata],
        preflight=Mock(),
        transcribe=Mock(return_value=replace(result, metadata=metadata)),
        close=Mock(),
    )


class SegmentationIntegrationTests(unittest.TestCase):
    def setUp(self):
        directory = PROJECT_DIR / "outputs"
        directory.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=directory)
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)

    def test_word_segmentation_for_all_four_provider_pairs(self):
        for llm_provider in ("ollama", "openai"):
            for stt_provider in ("whisper_local", "elevenlabs"):
                with self.subTest(llm=llm_provider, stt=stt_provider):
                    language, _ = llm_adapter(
                        llm_provider,
                        [
                            '{"0":"Tema A","1":"Tema B"}',
                            '{"0":"Tema A"}',
                            '{"1":"Tema B"}',
                            '["Global A","Global B"]',
                            '{"0":"Global A"}',
                            '{"1":"Global B"}',
                        ],
                    )
                    speech = speech_adapter(stt_provider)
                    segmenter = WordVideoSegmenter(
                        min_segment_length=5,
                        max_segment_length=120,
                        max_subtopics=3,
                        llm_adapter=language,
                        stt_adapter=speech,
                    )
                    video = Video(name="synthetic", path="data/raw/synthetic.mp4")
                    result = segmenter.create_videos_segments([video])[0]
                    self.assertEqual(["Tema A", "Tema B"], result.topics)
                    self.assertEqual(
                        ["Primeiro assunto.", "Segundo assunto."],
                        [s.content for s in result.segments],
                    )
                    self.assertEqual(
                        [(0.1, 6.0), (8.0, 14.0)],
                        [(s.start, s.end) for s in result.segments],
                    )
                    self.assertEqual(
                        ["Global A", "Global B"],
                        [s.global_topic for s in result.segments],
                    )
                    self.assertEqual([0, 1], [s.order for s in result.segments])
                    self.assertTrue(
                        all(s.video_path == video.path for s in result.segments)
                    )
                    speech.transcribe.assert_called_once_with(
                        PROJECT_DIR / video.path, language=None
                    )
                    output = self.output / f"{llm_provider}_{stt_provider}.json"
                    VideoDumper.dump_videos_to_json([result], output.as_posix())
                    data = json.loads(output.read_text(encoding="utf-8"))[0]
                    self.assertEqual("data/raw/synthetic.mp4", data["path"])
                    self.assertEqual(data["path"], data["segments"][0]["video_path"])

    def test_clustered_segmentation_preserves_native_local_chunks(self):
        for stt_provider in ("whisper_local", "elevenlabs"):
            with self.subTest(stt=stt_provider):
                language, _ = llm_adapter(
                    "openai",
                    [
                        TEXT,
                        "Primeiro assunto.",
                        "Segundo assunto.",
                        '{"0":"Tema A","1":"Tema B"}',
                        '{"0":"Tema A"}',
                        '{"1":"Tema B"}',
                    ],
                )
                segmenter = ClusteredVideoSegmenter(
                    min_segment_length=5,
                    max_segment_length=120,
                    max_subtopics=3,
                    llm_adapter=language,
                    stt_adapter=speech_adapter(stt_provider),
                )
                video = segmenter.create_video_segments(
                    Video("synthetic", "data/raw/synthetic.mp4")
                )
                expected = (
                    [(0.0, 6.1), (7.9, 14.2)]
                    if stt_provider == "whisper_local"
                    else [(0.1, 6.0), (8.0, 14.0)]
                )
                self.assertEqual(expected, [(s.start, s.end) for s in video.segments])
                self.assertEqual(
                    ["Tema A", "Tema B"], [s.video_topic for s in video.segments]
                )
                self.assertEqual([0, 1], [s.order for s in video.segments])
                self.assertTrue(all(s.video_path == video.path for s in video.segments))

    def test_existing_topics_are_reused_without_regeneration(self):
        language, create = llm_adapter(
            "ollama", ['{"0":"Existing"}', '{"0":"Existing"}']
        )
        segmenter = WordVideoSegmenter(
            min_segment_length=5,
            max_subtopics=3,
            llm_adapter=language,
            stt_adapter=speech_adapter("whisper_local"),
        )
        video = segmenter.create_video_segments(
            Video("existing", "data/raw/synthetic.mp4", topics=["Existing"])
        )
        self.assertEqual(2, create.call_count)
        self.assertEqual(["Existing"], video.topics)
        self.assertEqual(1, len(video.segments))
        self.assertEqual((0.1, 14.0), (video.segments[0].start, video.segments[0].end))

    def test_quote_punctuation_is_preserved_and_ends_sentences(self):
        language, _ = llm_adapter("ollama", [])
        segmenter = WordVideoSegmenter(
            min_segment_length=0.1,
            llm_adapter=language,
            stt_adapter=speech_adapter("elevenlabs"),
        )
        words = (TimedWord("“Primeiro!”", 0, 1), TimedWord("Segundo.", 2, 3))
        result = segmenter.get_segments_from_words(words)
        self.assertEqual(["“Primeiro!”", "Segundo."], [s.content for s in result])
        clustered = ClusteredVideoSegmenter(
            llm_adapter=language, stt_adapter=speech_adapter("elevenlabs")
        )
        chunks = clustered.transcript_segments(
            TranscriptionResult("“Primeiro!” Segundo.", words)
        )
        self.assertEqual(
            [(0, 1), (2, 3)], [(chunk.start, chunk.end) for chunk in chunks]
        )

    def test_unspaced_and_mixed_text_survives_word_and_clustered_flows(self):
        samples = [
            ("「你好世界！」", "第二句话。", ["你好", "世界", "第二", "句话"], ""),
            ("Hello 世界。", "第二phrase!", ["Hello", "世界", "第二", "phrase"], " "),
        ]
        for first, second, tokens, separator in samples:
            full_text = first + separator + second
            raw_words = [dict(word, text=token) for word, token in zip(WORDS, tokens)]
            for stt_provider in ("whisper_local", "elevenlabs"):
                if stt_provider == "whisper_local":
                    fixture = {
                        "text": full_text, "language": "zh",
                        "segments": [
                            {"text": first, "start": 0, "end": 6.1, "words": raw_words[:2]},
                            {"text": second, "start": 7.9, "end": 14.2, "words": raw_words[2:]},
                        ],
                    }
                else:
                    fixture = {
                        "text": full_text, "language_code": "zho",
                        "words": [dict(word, type="word") for word in raw_words],
                    }
                for segmenter_type in (WordVideoSegmenter, ClusteredVideoSegmenter):
                    with self.subTest(text=full_text, stt=stt_provider, segmenter=segmenter_type.__name__):
                        responses = ['{"0":"Tema A","1":"Tema B"}', '{"0":"Tema A"}', '{"1":"Tema B"}']
                        if segmenter_type is ClusteredVideoSegmenter:
                            responses = [full_text, first, second] + responses
                        language, generate = llm_adapter("openai", responses)
                        speech = speech_adapter(stt_provider, fixture)
                        segmenter = segmenter_type(
                            min_segment_length=5, max_segment_length=120,
                            max_subtopics=3, llm_adapter=language, stt_adapter=speech,
                        )
                        result = segmenter.create_video_segments(Video("synthetic", "data/raw/synthetic.mp4"))
                        self.assertEqual(full_text, speech.transcribe.return_value.text)
                        self.assertEqual([first, second], [segment.content for segment in result.segments])
                        expected = [(0.1, 6), (8, 14)]
                        if segmenter_type is ClusteredVideoSegmenter and stt_provider == "whisper_local":
                            expected = [(0, 6.1), (7.9, 14.2)]
                        self.assertEqual(expected, [(segment.start, segment.end) for segment in result.segments])
                        self.assertEqual(["Tema A", "Tema B"], [segment.video_topic for segment in result.segments])
                        self.assertTrue(all(segment.video_path == result.path for segment in result.segments))
                        self.assertIn(full_text, generate.call_args_list[0].kwargs["input"])

    def test_custom_sentence_boundaries_still_override_unicode_defaults(self):
        language, _ = llm_adapter("ollama", [])
        segmenter = WordVideoSegmenter(
            min_segment_length=0.1, sentence_boundary=".!?",
            llm_adapter=language, stt_adapter=speech_adapter("elevenlabs"),
        )
        segments = segmenter.get_segments_from_words(
            (TimedWord("你好。", 0, 1, separator_after=""), TimedWord("世界。", 2, 3))
        )
        self.assertEqual(["你好。世界。"], [segment.content for segment in segments])
        self.assertEqual([(0, 3)], [(segment.start, segment.end) for segment in segments])

    def test_source_punctuation_and_whitespace_survive_both_segmenter_flows(self):
        samples = [
            ("The price is $5.", ["The", "price", "is", "$5."]),
            ("It is -5 degrees.", ["It", "is", "-5", "degrees."]),
            ("Use #Python now.", ["Use", "#Python", "now."]),
            ("Hello @world.", ["Hello", "@world."]),
            ("Score +5.", ["Score", "+5."]),
            ("Type /help.", ["Type", "/help."]),
            ("state-of-the-art.", ["state", "-of", "-the", "-art."]),
            ("and/or.", ["and", "/or."]),
            ("l’amour.", ["l", "’amour."]),
            ("你好$5世界。", ["你好", "$5", "世界。"]),
            ("Pay €5 or £6.", ["Pay", "€5", "or", "£6."]),
            ("Value −5.", ["Value", "−5."]),
            ("Keep 100% now.", ["Keep", "100%", "now."]),
            ('He said: "go!"', ["He", "said:", '"go!"']),
            ("Ele disse: “Olá!”", ["Ele", "disse:", "Olá!"]),
            ("「你好，世界！」", ["你好", "世界"]),
            ("go/go—go.", ["go", "/go", "—go."]),
            ("A\t /\u00a0B\n\u2003+C.", ["A", "/B", "+C."]),
            ("Cafe\u0301\u00a0&\tchá.", ["Café", "chá."]),
        ]
        for text, tokens in samples:
            raw_words = [
                {"text": token, "start": index + 0.1, "end": index + 0.8}
                for index, token in enumerate(tokens)
            ]
            for provider in ("whisper_local", "elevenlabs"):
                if provider == "whisper_local":
                    fixture = {
                        "text": text, "language": "pt",
                        "segments": [{"text": text, "start": 0, "end": len(tokens), "words": raw_words}],
                    }
                else:
                    fixture = {"text": text, "language_code": "por", "words": [dict(word, type="word") for word in raw_words]}
                for segmenter_type in (WordVideoSegmenter, ClusteredVideoSegmenter):
                    with self.subTest(text=text, provider=provider, segmenter=segmenter_type.__name__):
                        responses = ['{"0":"Tema"}', '{"0":"Tema"}']
                        if segmenter_type is ClusteredVideoSegmenter:
                            responses = [text, text] + responses
                        language, generate = llm_adapter("openai", responses)
                        speech = speech_adapter(provider, fixture)
                        segmenter = segmenter_type(
                            min_segment_length=1, max_subtopics=3,
                            llm_adapter=language, stt_adapter=speech,
                        )
                        result = segmenter.create_video_segments(Video("synthetic", "data/raw/synthetic.mp4"))
                        self.assertEqual(text, speech.transcribe.return_value.text)
                        self.assertEqual(
                            [(word["start"], word["end"]) for word in raw_words],
                            [(word.start, word.end) for word in speech.transcribe.return_value.words],
                        )
                        self.assertEqual([text], [segment.content for segment in result.segments])
                        expected = (0.1, len(tokens) - 0.2)
                        if provider == "whisper_local" and segmenter_type is ClusteredVideoSegmenter:
                            expected = (0, len(tokens))
                        self.assertEqual([expected], [(segment.start, segment.end) for segment in result.segments])
                        self.assertEqual("Tema", result.segments[0].video_topic)
                        self.assertEqual(result.path, result.segments[0].video_path)
                        self.assertIn(text, generate.call_args_list[0].kwargs["input"])

    def test_empty_transcription_stops_before_any_llm_call(self):
        language, create = llm_adapter("openai", [])
        speech = SimpleNamespace(
            transcribe=Mock(return_value=TranscriptionResult("", ())),
            records=[],
            preflight=Mock(),
        )
        segmenter = WordVideoSegmenter(llm_adapter=language, stt_adapter=speech)
        with self.assertRaises(NoSpeechError):
            segmenter.create_video_segments(Video("empty", "data/raw/synthetic.mp4"))
        create.assert_not_called()

    def test_injected_segmenter_needs_no_provider_sdk_or_config_file(self):
        language, _ = llm_adapter("ollama", [])
        speech = speech_adapter("whisper_local")
        with (
            patch.dict(
                sys.modules,
                {
                    "ollama": None,
                    "openai": None,
                    "whisper_timestamped": None,
                    "elevenlabs": None,
                },
            ),
            patch(
                "open_video_summary.core.segmenter.video_segmenter.load_provider_config",
                side_effect=AssertionError("unused config"),
            ),
        ):
            WordVideoSegmenter(llm_adapter=language, stt_adapter=speech)
            ClusteredVideoSegmenter(llm_adapter=language, stt_adapter=speech)


class CLIProviderTests(unittest.TestCase):
    def setUp(self):
        output = PROJECT_DIR / "outputs"
        output.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=output)
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)

    def args(self, *extra):
        return cli.build_parser().parse_args(
            [
                "segment",
                "--input",
                "data/raw/synthetic",
                "--output",
                str(self.output / "segments.json"),
                *extra,
            ]
        )

    def test_argparse_defaults_do_not_mask_environment_and_aliases_work(self):
        args = self.args()
        self.assertIsNone(args.llm_model)
        self.assertIsNone(args.stt_model)
        self.assertIsNone(args.language)
        config = load_provider_config(
            vars(args),
            environ={
                "OVS_LLM_MODEL": "environment-model",
                "OVS_STT_MODEL": "environment-stt",
            },
            env_file=self.output / ".env",
        )
        self.assertEqual("environment-model", config.llm.model)
        self.assertEqual("environment-stt", config.stt.model)
        aliases = self.args("--whisper-model", "medium", "--language", "en")
        self.assertEqual("medium", aliases.stt_model)
        self.assertEqual("en", aliases.language)

    def test_failed_preflight_is_recorded_before_transcription(self):
        llm = SimpleNamespace(
            records=[],
            preflight=Mock(
                side_effect=AuthenticationError("OpenAI authentication failed.")
            ),
            close=Mock(),
        )
        stt = speech_adapter("whisper_local")
        config = ProviderConfig(
            LLMConfig(provider="openai", model="future", api_key="private-key"),
            STTConfig(),
        )
        with (
            patch(
                "open_video_summary.utils.providers.load_provider_config",
                return_value=config,
            ),
            patch("open_video_summary.adapters.factory.create_llm", return_value=llm),
            patch("open_video_summary.adapters.factory.create_stt", return_value=stt),
            patch("shutil.which", return_value="ffmpeg"),
        ):
            with self.assertRaises(AuthenticationError):
                cli._segment(self.args())
        stt.transcribe.assert_not_called()
        llm.close.assert_called_once()
        report = (self.output / "segments_run.json").read_text(encoding="utf-8")
        self.assertNotIn("private-key", report)
        metadata = json.loads(report)
        self.assertEqual("failed", metadata["status"])
        self.assertEqual("AuthenticationError", metadata["error_type"])

    def test_cli_success_writes_portable_data_and_safe_run_metadata(self):
        llm, _ = llm_adapter(
            "openai",
            [
                '{"0":"Tema A","1":"Tema B"}',
                '{"0":"Tema A"}',
                '{"1":"Tema B"}',
                '["Global A","Global B"]',
                '{"0":"Global A"}',
                '{"1":"Global B"}',
            ],
        )
        llm.preflight = Mock()
        stt = speech_adapter("elevenlabs")
        config = ProviderConfig(
            LLMConfig(
                provider="openai",
                model="gpt-6-luna",
                reasoning_effort="high",
                api_key="private-key",
            ),
            STTConfig(
                provider="elevenlabs", model="scribe_v2", api_key="private-stt-key"
            ),
        )
        video = Video("synthetic", "data/raw/synthetic.mp4")
        clip = Mock()
        clip.__enter__ = Mock(return_value=SimpleNamespace(duration=15))
        clip.__exit__ = Mock(return_value=False)
        with (
            patch(
                "open_video_summary.utils.providers.load_provider_config",
                return_value=config,
            ),
            patch("open_video_summary.adapters.factory.create_llm", return_value=llm),
            patch("open_video_summary.adapters.factory.create_stt", return_value=stt),
            patch("shutil.which", return_value="ffmpeg"),
            patch(
                "open_video_summary.parsers.video.VideoLoader.load_videos_from_directory",
                return_value=[video],
            ),
            patch("moviepy.VideoFileClip", return_value=clip),
        ):
            cli._segment(self.args())
        metadata = json.loads(
            (self.output / "segments_run.json").read_text(encoding="utf-8")
        )
        self.assertEqual("completed", metadata["status"])
        self.assertEqual(6, len(metadata["llm_calls"]))
        self.assertEqual("high", metadata["llm_calls"][0]["sent_reasoning_effort"])
        self.assertNotIn("private-key", json.dumps(metadata))
        self.assertNotIn(TEXT, json.dumps(metadata))
        saved = json.loads((self.output / "segments.json").read_text(encoding="utf-8"))[
            0
        ]
        self.assertEqual("data/raw/synthetic.mp4", saved["segments"][0]["video_path"])
        clip.__exit__.assert_called_once()

    def test_non_segment_commands_do_not_load_provider_configuration(self):
        with (
            patch(
                "open_video_summary.utils.providers.load_provider_config",
                side_effect=AssertionError("providers unused"),
            ),
            patch.object(cli, "_cpu_settings"),
            patch.object(cli, "_summarize") as summarize,
        ):
            cli.main(["summarize", "--no-render"])
        summarize.assert_called_once()


class FilteringProviderIntegrationTests(unittest.TestCase):
    def test_boolean_answers_preserve_existing_filter_threshold(self):
        language, generate = llm_adapter(
            "ollama",
            [
                "{'0': True, '1': False}",
                "{'0': False, '1': False}",
            ],
        )
        criterion = VideoQuestionBasedFiltering(
            filter_questions=["First question?", "Second question?"],
            min_positive_answer_ratio=0.5,
            llm_adapter=language,
        )
        criterion.get_segment_frames_every_n_seconds = Mock(return_value=[])
        criterion.get_segment_byte_images = Mock(return_value=[b"synthetic-jpeg"])
        video = Video(
            "synthetic",
            "data/raw/synthetic.mp4",
            segments=[
                VideoSegment("First", 0, 1),
                VideoSegment("Second", 2, 3),
            ],
        )
        handler = SummarySegmentHandler()
        handler.set_source_videos([video])
        self.assertIs(handler, criterion.evaluate(handler))
        self.assertEqual({video.segments[0]}, handler.include)
        self.assertEqual([b"synthetic-jpeg"], generate.call_args.kwargs["images"])
        self.assertNotIn("options", generate.call_args.kwargs)

    def test_filter_creation_is_lazy_and_preserves_original_local_model_default(self):
        with patch(
            "open_video_summary.core.selection_criteria.filtering.create_configured_llm"
        ) as create:
            criterion = VideoQuestionBasedFiltering(filter_questions=["Question?"])
            create.assert_not_called()
            self.assertIs(create.return_value, criterion._get_llm())
            self.assertIs(create.return_value, criterion._get_llm())
            create.assert_called_once_with(default_models={"ollama": "ministral-3"})


if __name__ == "__main__":
    unittest.main()
