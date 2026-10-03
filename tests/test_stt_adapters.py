import copy
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import traceback
import unittest
import wave
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from open_video_summary.adapters import stt
from open_video_summary.contracts import TimedWord
from open_video_summary.errors import (
    AuthenticationError,
    ConfigurationError,
    InvalidResponseError,
    NoSpeechError,
    RateLimitError,
    ServiceTimeoutError,
    ServiceUnavailableError,
)
from open_video_summary.utils.config import PROJECT_DIR
from open_video_summary.utils.providers import STTConfig


# Synthetic Portuguese content; no research recordings or paid APIs are used.
SCRIBE_RESPONSE = {
    "text": "Olá, mundo! Outra frase.",
    "language_code": "por",
    "model_id": "scribe_v2",
    "audio_duration_secs": 3.0,
    "words": [
        {"type": "word", "text": "Olá,", "start": 0.1, "end": 0.4},
        {"type": "spacing", "text": " "},
        {"type": "word", "text": "mundo!", "start": 0.5, "end": 0.8},
        {"type": "spacing", "text": " "},
        {"type": "word", "text": "Outra", "start": 1.2, "end": 1.5},
        {"type": "spacing", "text": " "},
        {"type": "word", "text": "frase.", "start": 1.6, "end": 2.0},
    ],
}
WHISPER_RESPONSE = {
    "text": " Olá, mundo! Outra frase.",
    "language": "pt",
    "segments": [
        {
            "text": " Olá, mundo!",
            "start": 0.1,
            "end": 0.8,
            "words": [
                {"text": "Olá,", "start": 0.1, "end": 0.4, "confidence": 0.9},
                {"text": "mundo!", "start": 0.5, "end": 0.8},
            ],
        },
        {
            "text": " Outra frase.",
            "start": 1.2,
            "end": 2.0,
            "words": [
                {"text": "Outra", "start": 1.2, "end": 1.5},
                {"text": "frase.", "start": 1.6, "end": 2.0},
            ],
        },
    ],
}


class ApiFailure(Exception):
    def __init__(self, status):
        self.status_code = status
        self.body = {"detail": "secret-key and full-private-transcript"}
        super().__init__("secret-key and full-private-transcript")


class NetworkFailure(Exception):
    pass


class NetworkTimeout(NetworkFailure):
    pass


class STTAdapterTests(unittest.TestCase):
    def setUp(self):
        output = PROJECT_DIR / "outputs"
        output.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=output)
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.audio_path = self.output / "audio.wav"
        self.write_wav(self.audio_path)
        self.config = STTConfig(
            provider="elevenlabs",
            model="scribe_v2",
            language="pt",
            api_key="synthetic-test-key",
            timeout_seconds=12.0,
            max_attempts=3,
            retry_backoff_seconds=0.1,
        )

    @staticmethod
    def write_wav(path):
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\0\0" * 48000)

    @contextmanager
    def remote(self, response=None, *, errors=(), config=None):
        responses = list(errors)
        fixture = copy.deepcopy(SCRIBE_RESPONSE if response is None else response)
        uploads = []

        def convert(**kwargs):
            upload = kwargs["file"]
            uploads.append(upload)
            self.assertEqual(0, upload.tell())
            self.assertTrue(upload.read(4))
            value = responses.pop(0) if responses else copy.deepcopy(fixture)
            if isinstance(value, Exception):
                raise value
            return value

        client = SimpleNamespace(
            speech_to_text=SimpleNamespace(convert=Mock(side_effect=convert))
        )
        client_class = Mock(return_value=client)
        transport = MagicMock()
        transport.__enter__.return_value = transport
        httpx = SimpleNamespace(
            Client=Mock(return_value=transport),
            TimeoutException=NetworkTimeout,
            TransportError=NetworkFailure,
        )
        modules = {
            "elevenlabs.client": SimpleNamespace(ElevenLabs=client_class),
            "httpx": httpx,
        }
        with (
            patch.object(stt, "_require_dependency"),
            patch.object(stt, "_version", return_value="synthetic-sdk-version"),
            patch.object(
                stt.importlib, "import_module", side_effect=modules.__getitem__
            ) as imports,
            patch.object(stt.time, "sleep") as sleep,
        ):
            adapter = stt.ElevenLabsSTT(config or self.config)
            yield SimpleNamespace(
                adapter=adapter,
                client=client,
                client_class=client_class,
                httpx=httpx,
                transport=transport,
                uploads=uploads,
                imports=imports,
                sleep=sleep,
            )

    @contextmanager
    def local(self, response=None):
        audio = [0.0] * 48000
        model = object()
        whisper = SimpleNamespace(
            load_model=Mock(return_value=model),
            load_audio=Mock(return_value=audio),
            transcribe=Mock(return_value=copy.deepcopy(response or WHISPER_RESPONSE)),
        )
        config = STTConfig(model="base", language="pt")
        with (
            patch.object(stt, "_require_dependency"),
            patch.object(stt.shutil, "which", return_value="ffmpeg"),
            patch.object(stt, "_version", return_value="1.15.8"),
            patch.object(
                stt,
                "_whisper_languages",
                return_value={
                    code: code
                    for code in ("pt", "es", "en", "fr", "de", "zh", "jw", "haw", "yue")
                },
            ),
            patch.object(
                stt.importlib, "import_module", return_value=whisper
            ) as imports,
        ):
            yield stt.LocalWhisperSTT(config), whisper, model, audio, imports

    def assert_safe_failure(self, callback, error_type):
        with self.assertRaises(error_type) as raised:
            callback()
        rendered = "".join(traceback.format_exception(raised.exception))
        self.assertNotIn("secret-key", rendered)
        self.assertNotIn("full-private-transcript", rendered)
        return raised.exception

    def test_adapter_import_and_constructors_do_not_import_optional_or_ml_modules(self):
        code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'whisper', 'whisper_timestamped', 'elevenlabs'}:
        raise AssertionError('unexpected optional or ML import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from open_video_summary.adapters.stt import LocalWhisperSTT, ElevenLabsSTT
from open_video_summary.utils.providers import STTConfig
LocalWhisperSTT(STTConfig())
ElevenLabsSTT(STTConfig(provider='elevenlabs', model='scribe_v2'))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)

    def test_local_whisper_is_lazy_cpu_and_preserves_native_segments(self):
        with self.local() as (adapter, whisper, model, audio, imports):
            adapter.preflight()
            imports.assert_not_called()
            whisper.load_model.assert_not_called()
            first = adapter.transcribe(self.audio_path)
            second = adapter.transcribe(self.audio_path, language="es")
            whisper.load_model.assert_called_once_with(
                "base", device="cpu", download_root=str(PROJECT_DIR / ".cache/whisper")
            )
            self.assertEqual("whisper_timestamped", imports.call_args.args[0])
            whisper.transcribe.assert_called_with(
                model, audio, language="es", fp16=False, verbose=None
            )
        self.assertEqual("Olá, mundo! Outra frase.", first.text)
        self.assertEqual("pt", first.language)
        self.assertEqual(
            [(0.1, 0.8), (1.2, 2.0)], [(s.start, s.end) for s in first.segments]
        )
        self.assertEqual(
            ["Olá, mundo!", "Outra frase."], [s.text for s in first.segments]
        )
        self.assertEqual(first.words[:2], first.segments[0].words)
        self.assertEqual("whisper_local", first.metadata.provider)
        self.assertEqual("es", second.metadata.requested_language)
        self.assertEqual("es", second.metadata.sent_language)
        self.assertEqual(3.0, first.metadata.audio_duration_seconds)

    def test_local_iso_alias_is_translated_before_model_load_and_recorded(self):
        with self.local() as (adapter, whisper, model, audio, imports):
            adapter.config = replace(adapter.config, language="POR")
            adapter.preflight()
            imports.assert_not_called()
            whisper.load_model.assert_not_called()
            result = adapter.transcribe(self.audio_path)
            whisper.transcribe.assert_called_once_with(
                model, audio, language="pt", fp16=False, verbose=None
            )
        self.assertEqual("POR", result.metadata.requested_language)
        self.assertEqual("pt", result.metadata.sent_language)
        self.assertEqual("pt", result.metadata.reported_language)

    def test_local_language_aliases_follow_installed_capabilities(self):
        languages = {
            code: code
            for code in ("pt", "en", "fr", "zh", "de", "jw", "haw", "yue", "xyz")
        }
        with patch.object(stt, "_whisper_languages", return_value=languages):
            for requested, sent in {
                "por": "pt", "eng": "en", "fra": "fr", "fre": "fr",
                "zho": "zh", "chi": "zh", "deu": "de", "ger": "de",
                "jav": "jw", "jv": "jw", "haw": "haw", "yue": "yue",
                "xyz": "xyz",
            }.items():
                with self.subTest(requested=requested):
                    self.assertEqual(sent, stt._whisper_language(requested))
            # A reliable ISO alias still needs support in the selected SDK.
            with self.assertRaises(ConfigurationError):
                stt._whisper_language("spa")
        with patch.object(stt, "_whisper_languages") as registry:
            self.assertIsNone(stt._whisper_language(None))
            registry.assert_not_called()

    def test_local_unsupported_language_fails_before_model_or_audio(self):
        with self.local() as (adapter, whisper, _, _, imports):
            for language in ("zzz", "ace", "aa", "pt-BR"):
                with self.subTest(language=language):
                    adapter.config = replace(adapter.config, language=language)
                    with self.assertRaises(ConfigurationError):
                        adapter.preflight()
            adapter.config = replace(adapter.config, language="pt")
            with self.assertRaises(ConfigurationError):
                adapter.transcribe("missing-input.mp4", language="zzz")
            whisper.load_model.assert_not_called()
            whisper.load_audio.assert_not_called()
            whisper.transcribe.assert_not_called()
            imports.assert_not_called()

    def test_local_registry_is_selected_only_and_sdk_failure_is_safe(self):
        with patch.object(
            stt.importlib,
            "import_module",
            return_value=SimpleNamespace(LANGUAGES={"pt": "portuguese"}),
        ) as imports:
            self.assertEqual("pt", stt._whisper_language("por"))
            imports.assert_called_once_with("whisper.tokenizer")
        with patch.object(stt.importlib, "import_module", side_effect=ImportError):
            with self.assertRaises(ConfigurationError):
                stt._whisper_language("pt")

    def test_scribe_call_honors_model_language_and_configured_timeout(self):
        config = replace(self.config, model="custom-future-scribe-model")
        with self.remote(config=config) as fake:
            result = fake.adapter.transcribe(self.audio_path)
            fake.httpx.Client.assert_called_once_with(timeout=12.0)
            fake.client_class.assert_called_once_with(
                api_key="synthetic-test-key", timeout=12.0, httpx_client=fake.transport
            )
            kwargs = fake.client.speech_to_text.convert.call_args.kwargs
            self.assertEqual("custom-future-scribe-model", kwargs["model_id"])
            self.assertEqual("pt", kwargs["language_code"])
            self.assertEqual("word", kwargs["timestamps_granularity"])
            self.assertFalse(kwargs["tag_audio_events"])
            self.assertFalse(kwargs["diarize"])
            self.assertEqual({"max_retries": 0}, kwargs["request_options"])
            self.assertTrue(fake.uploads[0].closed)
            fake.transport.__exit__.assert_called_once()
            self.assertEqual(
                ["elevenlabs.client", "httpx"],
                [c.args[0] for c in fake.imports.call_args_list],
            )
        self.assertEqual("por", result.language)
        self.assertEqual((), result.segments)
        self.assertEqual("custom-future-scribe-model", result.metadata.requested_model)
        self.assertEqual("scribe_v2", result.metadata.returned_model)
        self.assertEqual("synthetic-sdk-version", result.metadata.sdk_version)
        self.assertEqual(3.0, result.metadata.audio_duration_seconds)
        self.assertGreaterEqual(result.metadata.duration_seconds, 0)
        self.assertEqual(1, result.metadata.attempts)
        self.assertEqual("pt", result.metadata.sent_language)
        record = json.dumps(asdict(result.metadata))
        self.assertNotIn("Olá", record)
        self.assertNotIn("synthetic-test-key", record)

    def test_sdk_objects_stay_inside_adapter_and_missing_reported_model_is_unknown(
        self,
    ):
        response = copy.deepcopy(SCRIBE_RESPONSE)
        response.pop("model_id")
        response["words"] = [SimpleNamespace(**word) for word in response["words"]]
        with self.remote(response=SimpleNamespace(**response)) as fake:
            result = fake.adapter.transcribe(self.audio_path)
        self.assertTrue(all(isinstance(word, TimedWord) for word in result.words))
        self.assertIsNone(result.metadata.returned_model)

    def test_autodetection_and_override_language_are_passed_explicitly(self):
        with self.remote(config=replace(self.config, language=None)) as fake:
            first = fake.adapter.transcribe(self.audio_path)
            self.assertIsNone(
                fake.client.speech_to_text.convert.call_args.kwargs["language_code"]
            )
            second = fake.adapter.transcribe(self.audio_path, language="pt")
            self.assertEqual(
                "pt",
                fake.client.speech_to_text.convert.call_args.kwargs["language_code"],
            )
        self.assertIsNone(first.metadata.requested_language)
        self.assertEqual("pt", second.metadata.requested_language)

    def test_remote_language_normalization_does_not_import_local_capabilities(self):
        with (
            self.remote(config=replace(self.config, language="POR")) as fake,
            patch.object(
                stt, "_whisper_languages", side_effect=AssertionError("local import")
            ),
        ):
            fake.adapter.preflight()
            fake.imports.assert_not_called()
            result = fake.adapter.transcribe(self.audio_path)
            self.assertEqual(
                "por", fake.client.speech_to_text.convert.call_args.kwargs["language_code"]
            )
        self.assertEqual("POR", result.metadata.requested_language)
        self.assertEqual("por", result.metadata.sent_language)
        self.assertEqual("por", result.metadata.reported_language)

    def test_spaces_punctuation_and_events_preserve_speech_intervals(self):
        response = {
            "text": "(risos)  Olá ,  mundo !",
            "language_code": "pt",
            "words": [
                {"type": "audio_event", "text": "(risos)", "start": 0, "end": 0.1},
                {"type": "spacing", "text": "  "},
                {"type": "word", "text": "Olá", "start": 0.2, "end": 0.5},
                {"type": "word", "text": ",", "start": None, "end": None},
                {"type": "spacing", "text": "\n "},
                {"type": "word", "text": "mundo", "start": 0.8, "end": 1.1},
                {"type": "word", "text": "!"},
            ],
        }
        with self.remote(response=response) as fake:
            result = fake.adapter.transcribe(self.audio_path)
        self.assertEqual("Olá, mundo!", result.text)
        self.assertEqual(
            (TimedWord("Olá,", 0.2, 0.5), TimedWord("mundo!", 0.8, 1.1)), result.words
        )

    def test_top_level_punctuation_is_retained_without_creating_word_times(self):
        response = copy.deepcopy(SCRIBE_RESPONSE)
        for word in response["words"]:
            if word["type"] == "word":
                word["text"] = word["text"].strip(",!.")
        with self.remote(response=response) as fake:
            result = fake.adapter.transcribe(self.audio_path)
        self.assertEqual(SCRIBE_RESPONSE["text"], result.text)
        self.assertEqual(
            ["Olá,", "mundo!", "Outra", "frase."], [w.text for w in result.words]
        )
        self.assertEqual(
            [(0.1, 0.4), (0.5, 0.8), (1.2, 1.5), (1.6, 2.0)],
            [(w.start, w.end) for w in result.words],
        )

    def test_word_punctuation_is_preserved_when_top_level_text_omits_it(self):
        response = copy.deepcopy(SCRIBE_RESPONSE)
        response["text"] = "Olá mundo Outra frase"
        with self.remote(response=response) as fake:
            result = fake.adapter.transcribe(self.audio_path)
        self.assertEqual(SCRIBE_RESPONSE["text"], result.text)
        self.assertEqual("mundo!", result.words[1].text)
        self.assertEqual((0.5, 0.8), (result.words[1].start, result.words[1].end))

    def test_separate_quotes_keep_their_position_and_original_speech_times(self):
        response = {
            "text": 'Ele disse: " Olá ! " Depois.',
            "language_code": "pt",
            "words": [
                {"type": "word", "text": "Ele", "start": 0.1, "end": 0.3},
                {"type": "word", "text": "disse:", "start": 0.4, "end": 0.7},
                {"type": "word", "text": '"'},
                {"type": "word", "text": "Olá", "start": 0.8, "end": 1.1},
                {"type": "word", "text": "!"},
                {"type": "word", "text": '"'},
                {"type": "word", "text": "Depois.", "start": 1.3, "end": 1.6},
            ],
        }
        with self.remote(response=response) as fake:
            result = fake.adapter.transcribe(self.audio_path)
        self.assertEqual('Ele disse: "Olá!" Depois.', result.text)
        self.assertEqual(TimedWord('"Olá!"', 0.8, 1.1), result.words[2])

    def test_event_label_equal_to_spoken_text_does_not_remove_spoken_punctuation(self):
        response = {
            "text": "Olá, Olá",
            "language_code": "pt",
            "words": [
                {"type": "word", "text": "Olá,", "start": 0.1, "end": 0.5},
                {"type": "spacing", "text": " "},
                {"type": "audio_event", "text": "Olá", "start": 0.6, "end": 0.9},
            ],
        }
        with self.remote(response=response) as fake:
            result = fake.adapter.transcribe(self.audio_path)
        self.assertEqual("Olá,", result.text)
        self.assertEqual((TimedWord("Olá,", 0.1, 0.5),), result.words)

    def test_synthetic_delayed_audio_extraction_preserves_video_time_reference(self):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            self.skipTest("FFmpeg is unavailable for the synthetic extraction check")
        tone_path = self.output / "tone.wav"
        with wave.open(str(tone_path), "wb") as tone:
            tone.setnchannels(1)
            tone.setsampwidth(2)
            tone.setframerate(16000)
            tone.writeframes(
                b"".join(
                    struct.pack(
                        "<h", round(12000 * math.sin(2 * math.pi * 300 * index / 16000))
                    )
                    for index in range(8000)
                )
            )
        video_path = self.output / "delayed.mkv"
        subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=16x16:r=10:d=1.5",
                "-itsoffset",
                "0.5",
                "-i",
                str(tone_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "ffv1",
                "-c:a",
                "pcm_s16le",
                str(video_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=20,
        )
        with stt._upload_audio(video_path, 20.0) as extracted:
            with wave.open(str(extracted), "rb") as audio:
                rate = audio.getframerate()
                samples = struct.unpack(
                    "<" + "h" * audio.getnframes(), audio.readframes(audio.getnframes())
                )
            first_audible = next(
                index for index, value in enumerate(samples) if abs(value) > 100
            )
            self.assertAlmostEqual(0.5, first_audible / rate, delta=0.01)
            self.assertTrue(all(value == 0 for value in samples[: int(rate * 0.49)]))
            self.assertTrue(
                any(abs(value) > 100 for value in samples[int(rate * 0.51) :])
            )
        self.assertFalse(extracted.parent.exists())

    def test_empty_and_event_only_responses_raise_no_speech_once(self):
        responses = [
            {"text": "", "words": [], "language_code": "pt"},
            {
                "text": " (risos) ",
                "words": [{"type": "audio_event", "text": "(risos)"}],
            },
        ]
        for response in responses:
            with (
                self.subTest(response=response),
                self.remote(response=response) as fake,
            ):
                self.assert_safe_failure(
                    lambda: fake.adapter.transcribe(self.audio_path), NoSpeechError
                )
                self.assertEqual(1, fake.client.speech_to_text.convert.call_count)
                fake.sleep.assert_not_called()
                self.assertEqual("NoSpeechError", fake.adapter.records[-1].status)
                self.assertTrue(all(upload.closed for upload in fake.uploads))
        with self.local(response={"text": "", "segments": [], "language": "pt"}) as (
            adapter,
            *_,
        ):
            self.assert_safe_failure(
                lambda: adapter.transcribe(self.audio_path), NoSpeechError
            )

    def test_invalid_responses_are_rejected_without_retries(self):
        mutations = [
            lambda r: r.pop("text"),
            lambda r: r.pop("words"),
            lambda r: r.update(text="palavras diferentes"),
            lambda r: r.update(text="Olá. mundo! Outra frase."),
            lambda r: r.update(text=""),
            lambda r: r.update(words=[]),
            lambda r: r["words"][0].update(type="unknown"),
            lambda r: r["words"][1].update(text="unexpected content"),
            lambda r: r["words"][0].update(text="duas palavras"),
            lambda r: r["words"][0].update(start=None),
            lambda r: r["words"][0].update(start="0.1"),
            lambda r: r["words"][0].update(start=True),
            lambda r: r["words"][0].update(start=-0.1),
            lambda r: r["words"][0].update(start=float("nan")),
            lambda r: r["words"][0].update(start=10**500),
            lambda r: r["words"][0].update(end=float("inf")),
            lambda r: r["words"][0].update(end=0.0),
            lambda r: r["words"][2].update(start=0.3),
            lambda r: r["words"][4].update(start=0.0),
            lambda r: r.update(language_code=5),
            lambda r: r.update(model_id=5),
            lambda r: r.update(audio_duration_secs=-1),
        ]
        for index, mutate in enumerate(mutations):
            response = copy.deepcopy(SCRIBE_RESPONSE)
            mutate(response)
            with self.subTest(case=index), self.remote(response=response) as fake:
                self.assert_safe_failure(
                    lambda: fake.adapter.transcribe(self.audio_path),
                    InvalidResponseError,
                )
                self.assertEqual(1, fake.client.speech_to_text.convert.call_count)
                fake.sleep.assert_not_called()
                fake.transport.__exit__.assert_called_once()
                self.assertTrue(all(upload.closed for upload in fake.uploads))

    def test_invalid_local_word_and_segment_timestamps_are_rejected(self):
        mutations = [
            lambda r: r["segments"][0].update(start=-1),
            lambda r: r["segments"][0].update(end=float("nan")),
            lambda r: r["segments"][0]["words"][0].update(start=0.0),
            lambda r: r["segments"][0]["words"][1].update(end=0.9),
            lambda r: r["segments"][1].update(start=0.7),
            lambda r: r["segments"][0].update(words=[]),
        ]
        for index, mutate in enumerate(mutations):
            response = copy.deepcopy(WHISPER_RESPONSE)
            mutate(response)
            with (
                self.subTest(case=index),
                self.local(response=response) as (adapter, whisper, *_),
            ):
                self.assert_safe_failure(
                    lambda: adapter.transcribe(self.audio_path), InvalidResponseError
                )
                self.assertEqual(1, whisper.transcribe.call_count)

    def test_transient_failures_retry_with_fresh_uploads_and_bounded_backoff(self):
        failures = [
            ApiFailure(429),
            ApiFailure(503),
            ApiFailure(408),
            NetworkTimeout(),
            TimeoutError(),
            NetworkFailure(),
        ]
        for failure in failures:
            with (
                self.subTest(failure=type(failure).__name__),
                self.remote(errors=[failure, failure]) as fake,
            ):
                result = fake.adapter.transcribe(self.audio_path)
                self.assertEqual(3, fake.client.speech_to_text.convert.call_count)
                self.assertEqual(
                    [0.1, 0.2], [call.args[0] for call in fake.sleep.call_args_list]
                )
                self.assertEqual(3, result.metadata.attempts)
                self.assertEqual(3, len({id(upload) for upload in fake.uploads}))
                self.assertTrue(all(upload.closed for upload in fake.uploads))
        with self.remote(errors=[ApiFailure(503)] * 3) as fake:
            self.assert_safe_failure(
                lambda: fake.adapter.transcribe(self.audio_path),
                ServiceUnavailableError,
            )
            self.assertEqual(3, fake.client.speech_to_text.convert.call_count)
            self.assertEqual(2, fake.sleep.call_count)
            self.assertEqual(3, fake.adapter.records[-1].attempts)

    def test_exhausted_rate_and_timeout_failures_use_internal_errors(self):
        for error, expected in [
            (ApiFailure(429), RateLimitError),
            (NetworkTimeout(), ServiceTimeoutError),
        ]:
            with (
                self.subTest(error=type(error).__name__),
                self.remote(errors=[error] * 3) as fake,
            ):
                self.assert_safe_failure(
                    lambda: fake.adapter.transcribe(self.audio_path), expected
                )
                self.assertEqual(3, fake.client.speech_to_text.convert.call_count)

    def test_auth_validation_and_unexpected_sdk_failures_do_not_retry_or_leak(self):
        cases = [
            (ApiFailure(401), AuthenticationError),
            (ApiFailure(403), AuthenticationError),
            (ApiFailure(402), ConfigurationError),
            (ApiFailure(404), ConfigurationError),
            (ApiFailure(422), ConfigurationError),
            (
                ValueError("secret-key and full-private-transcript"),
                InvalidResponseError,
            ),
            (
                RuntimeError("secret-key and full-private-transcript"),
                ServiceUnavailableError,
            ),
        ]
        for failure, expected in cases:
            with (
                self.subTest(status=getattr(failure, "status_code", None)),
                self.remote(errors=[failure]) as fake,
            ):
                self.assert_safe_failure(
                    lambda: fake.adapter.transcribe(self.audio_path), expected
                )
                self.assertEqual(1, fake.client.speech_to_text.convert.call_count)
                fake.sleep.assert_not_called()
                fake.transport.__exit__.assert_called_once()
                self.assertTrue(fake.uploads[0].closed)

    def test_video_extraction_keeps_source_timeline_and_cleans_up_after_every_outcome(
        self,
    ):
        video = self.output / "source.mp4"
        video.write_bytes(b"synthetic-video")
        for failure in (None, ApiFailure(401), "invalid-response"):
            extracted = []

            def extract(command, **kwargs):
                self.assertEqual(str(video), command[command.index("-i") + 1])
                self.assertNotIn("-ss", command)
                self.assertNotIn("-t", command)
                self.assertIn("-copyts", command)
                self.assertIn("-start_at_zero", command)
                self.assertIn("aresample=async=1:first_pts=0", command)
                self.assertEqual(12.0, kwargs["timeout"])
                extracted.append(Path(command[-1]))
                self.write_wav(extracted[-1])

            response = copy.deepcopy(SCRIBE_RESPONSE)
            if failure == "invalid-response":
                response["words"][0]["start"] = -1
            errors = [failure] if isinstance(failure, Exception) else []
            with (
                self.subTest(failure=str(failure)),
                self.remote(response=response, errors=errors) as fake,
                patch.object(stt.shutil, "which", return_value="ffmpeg"),
                patch.object(stt.subprocess, "run", side_effect=extract),
            ):
                if failure:
                    expected = (
                        AuthenticationError
                        if isinstance(failure, Exception)
                        else InvalidResponseError
                    )
                    self.assert_safe_failure(
                        lambda: fake.adapter.transcribe(video), expected
                    )
                else:
                    fake.adapter.transcribe(video)
                self.assertTrue(all(upload.closed for upload in fake.uploads))
            self.assertFalse(extracted[0].exists())
            self.assertFalse(extracted[0].parent.exists())
            self.assertTrue(video.is_file())

    def test_extraction_failure_and_timeout_remove_partial_files_without_upload(self):
        video = self.output / "source.mp4"
        video.write_bytes(b"synthetic-video")
        for expected, timeout in [
            (ConfigurationError, False),
            (ServiceTimeoutError, True),
        ]:
            extracted = []

            def extract(command, **kwargs):
                extracted.append(Path(command[-1]))
                extracted[-1].write_bytes(b"partial")
                if timeout:
                    raise subprocess.TimeoutExpired(command, kwargs["timeout"])
                raise subprocess.CalledProcessError(
                    1, command, stderr=b"secret-key and full-private-transcript"
                )

            with (
                self.subTest(timeout=timeout),
                self.remote() as fake,
                patch.object(stt.shutil, "which", return_value="ffmpeg"),
                patch.object(stt.subprocess, "run", side_effect=extract),
            ):
                self.assert_safe_failure(
                    lambda: fake.adapter.transcribe(video), expected
                )
                fake.client.speech_to_text.convert.assert_not_called()
                self.assertEqual(0, fake.adapter.records[-1].attempts)
            self.assertFalse(extracted[0].parent.exists())

    def test_missing_ffmpeg_is_only_required_for_video_extraction(self):
        video = self.output / "source.mp4"
        video.write_bytes(b"synthetic-video")
        with (
            self.remote() as fake,
            patch.object(stt.shutil, "which", return_value=None),
        ):
            fake.adapter.preflight()
            fake.adapter.transcribe(self.audio_path)
            self.assert_safe_failure(
                lambda: fake.adapter.transcribe(video), ConfigurationError
            )
            self.assertEqual(1, fake.client.speech_to_text.convert.call_count)

    def test_invalid_direct_configuration_fails_before_import_or_upload(self):
        configs = [
            replace(self.config, api_key=None),
            replace(self.config, model=""),
            replace(self.config, timeout_seconds=0),
            replace(self.config, timeout_seconds=float("nan")),
            replace(self.config, max_attempts=0),
            replace(self.config, max_attempts=True),
            replace(self.config, retry_backoff_seconds=-1),
            replace(self.config, language="invalid"),
        ]
        for config in configs:
            with self.subTest(config=config), self.remote(config=config) as fake:
                self.assert_safe_failure(
                    lambda: fake.adapter.transcribe(self.audio_path), ConfigurationError
                )
                fake.imports.assert_not_called()
                fake.client.speech_to_text.convert.assert_not_called()

    def test_missing_dependency_and_missing_media_use_internal_configuration_error(
        self,
    ):
        with patch.object(stt.importlib.util, "find_spec", return_value=None):
            self.assert_safe_failure(
                lambda: stt.ElevenLabsSTT(self.config).preflight(), ConfigurationError
            )
            self.assert_safe_failure(
                lambda: stt.LocalWhisperSTT(STTConfig()).preflight(), ConfigurationError
            )
        with self.remote() as fake:
            self.assert_safe_failure(
                lambda: fake.adapter.transcribe(self.output / "missing.wav"),
                ConfigurationError,
            )
            fake.imports.assert_not_called()

    def test_unreadable_upload_closes_transport_without_calling_api(self):
        with (
            self.remote() as fake,
            patch.object(Path, "open", side_effect=PermissionError("secret-key")),
        ):
            self.assert_safe_failure(
                lambda: fake.adapter.transcribe(self.audio_path), ConfigurationError
            )
            fake.client.speech_to_text.convert.assert_not_called()
            fake.transport.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
