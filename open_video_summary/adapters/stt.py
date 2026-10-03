"""Speech providers translated into project-owned text and timing records."""

import importlib
import importlib.metadata
import importlib.util
import math
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
import wave
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from open_video_summary.contracts import (
    ServiceMetadata,
    TimedWord,
    TranscriptSegment,
    TranscriptionResult,
)
from open_video_summary.errors import (
    AuthenticationError,
    ConfigurationError,
    InvalidResponseError,
    NoSpeechError,
    ProviderError,
    RateLimitError,
    ServiceTimeoutError,
    ServiceUnavailableError,
)
from open_video_summary.utils.config import PROJECT_DIR
from open_video_summary.utils.paths import project_path
from open_video_summary.utils.providers import STTConfig


_AUDIO_SUFFIXES = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}
_OPEN_PUNCTUATION = "([{¿¡“«"


def _field(value, name, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _require_dependency(package: str) -> None:
    try:
        available = importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        available = False
    if not available:
        raise ConfigurationError(f"The selected STT provider requires {package}.")


def _validate_config(config: STTConfig) -> None:
    if not isinstance(config.model, str) or not config.model.strip():
        raise ConfigurationError("An STT model must be configured.")
    if (
        isinstance(config.timeout_seconds, bool)
        or not isinstance(config.timeout_seconds, (int, float))
        or not math.isfinite(config.timeout_seconds)
        or config.timeout_seconds <= 0
    ):
        raise ConfigurationError("The STT timeout must be finite and positive.")
    if (
        isinstance(config.max_attempts, bool)
        or not isinstance(config.max_attempts, int)
        or config.max_attempts < 1
    ):
        raise ConfigurationError("STT max_attempts must be a positive integer.")
    if (
        isinstance(config.retry_backoff_seconds, bool)
        or not isinstance(config.retry_backoff_seconds, (int, float))
        or not math.isfinite(config.retry_backoff_seconds)
        or config.retry_backoff_seconds < 0
    ):
        raise ConfigurationError(
            "The STT retry backoff must be finite and nonnegative."
        )


def _language(value: str | None) -> str | None:
    if value is not None and (
        not isinstance(value, str) or re.fullmatch(r"[a-zA-Z]{2,3}", value) is None
    ):
        raise ConfigurationError("The STT language must be an ISO-639 code or None.")
    return value


def _media_path(value: str | Path) -> Path:
    try:
        path = project_path(value)
        if not path.is_file():
            raise ConfigurationError(
                "The media file does not exist or is not readable."
            )
        return path
    except (OSError, TypeError, ValueError):
        raise ConfigurationError(
            "The media path is invalid or is not readable."
        ) from None


def _text(value) -> str:
    if not isinstance(value, str):
        raise InvalidResponseError("STT returned a non-text transcript or token.")
    value = " ".join(value.split())
    value = re.sub(r"\s+([,.;:!?…%\)\]\}”»])", r"\1", value)
    return re.sub(r"([\(\[\{¿¡“«])\s+", r"\1", value)


def _lexical_key(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFC", value)
        if character.isalnum() or unicodedata.category(character).startswith("M")
    )


def _merge_punctuation(word: str, transcript_token: str) -> str:
    def split(token):
        letters, punctuation = [], [""]
        for character in unicodedata.normalize("NFC", token):
            if _lexical_key(character):
                letters.append(character)
                punctuation.append("")
            else:
                punctuation[-1] += character
        return letters, punctuation

    letters, original = split(word)
    _, reported = split(transcript_token)
    merged = []
    for left, right in zip(original, reported):
        if left and right and left != right:
            raise InvalidResponseError(
                "STT transcript and words have conflicting punctuation."
            )
        merged.append(left or right)
    return "".join(
        punctuation + (letters[index] if index < len(letters) else "")
        for index, punctuation in enumerate(merged)
    )


def _punctuation_opens(token: str, quoted: set[str]) -> bool:
    opens = token[0] in _OPEN_PUNCTUATION
    if token[0] in "\"'":
        opens = token[0] not in quoted
    lexical = [
        index for index, character in enumerate(token) if _lexical_key(character)
    ]
    decorations = (
        token if not lexical else token[: lexical[0]] + token[lexical[-1] + 1 :]
    )
    for quote in "\"'":
        if decorations.count(quote) % 2:
            if quote in quoted:
                quoted.remove(quote)
            else:
                quoted.add(quote)
    return opens


def _timestamp(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidResponseError("STT returned a missing or invalid timestamp.")
    try:
        value = float(value)
    except OverflowError:
        raise InvalidResponseError(
            "STT timestamps must be finite and nonnegative."
        ) from None
    if not math.isfinite(value) or value < 0:
        raise InvalidResponseError("STT timestamps must be finite and nonnegative.")
    return value


def _interval(item, previous_end: float = 0.0) -> tuple[float, float]:
    start, end = _timestamp(_field(item, "start")), _timestamp(_field(item, "end"))
    if end < start or start < previous_end:
        raise InvalidResponseError(
            "STT timestamps are reversed, overlapping, or unordered."
        )
    return start, end


def _words(
    items, *, elevenlabs: bool = False
) -> tuple[tuple[TimedWord, ...], list[tuple[str, int]]]:
    if not isinstance(items, (list, tuple)):
        raise InvalidResponseError("STT did not return a list of timed words.")
    words, events, prefix = [], [], ""
    quoted = set()
    previous_end = 0.0
    for item in items:
        kind = _field(item, "type") if elevenlabs else "word"
        token = _text(_field(item, "text"))
        if kind == "audio_event":
            events.append((token, len(words)))
            continue
        if kind == "spacing":
            if token:
                raise InvalidResponseError("STT returned a non-space spacing token.")
            continue
        if kind != "word":
            raise InvalidResponseError("STT returned an unknown token type.")
        if not token:
            continue
        opens = _punctuation_opens(token, quoted)
        if not _lexical_key(token):
            # Punctuation has no speech timing of its own. Keep the adjacent
            # spoken word's interval rather than manufacture an interval.
            if words and not opens:
                words[-1] = replace(words[-1], text=words[-1].text + token)
            else:
                prefix += token
            continue
        if " " in token:
            raise InvalidResponseError("STT returned multiple words with one interval.")
        start, end = _interval(item, previous_end)
        words.append(TimedWord(text=prefix + token, start=start, end=end))
        prefix, previous_end = "", end
    if prefix and words:
        words[-1] = replace(words[-1], text=words[-1].text + prefix)
    return tuple(words), events


def _align_text(
    words: tuple[TimedWord, ...],
    transcript,
    events: list[tuple[str, int]] | None = None,
) -> tuple[str, tuple[TimedWord, ...]]:
    text = unicodedata.normalize("NFC", _text(transcript))
    cursor, word_index, event_spans = 0, 0, []
    for event, words_before in events or ():
        if not event:
            continue
        # Match events after their preceding speech tokens. Removing the first
        # equal string globally could discard spoken text with the same label.
        while word_index < words_before:
            expected = _lexical_key(words[word_index].text)
            for letter in expected:
                while cursor < len(text) and not _lexical_key(text[cursor]):
                    cursor += 1
                if cursor >= len(text) or text[cursor] != letter:
                    raise InvalidResponseError(
                        "STT events and timed speech do not align."
                    )
                cursor += 1
            word_index += 1
        event = unicodedata.normalize("NFC", event)
        location = text.find(event, cursor)
        if location < 0:
            continue
        if _lexical_key(text[cursor:location]):
            raise InvalidResponseError("STT events and timed speech do not align.")
        event_spans.append((location, location + len(event)))
        cursor = location + len(event)
    for start, end in reversed(event_spans):
        text = text[:start] + text[end:]
    text = _text(text)
    if not words:
        if _lexical_key(text):
            raise InvalidResponseError(
                "STT returned speech without aligned word timestamps."
            )
        raise NoSpeechError("No speech was returned by the selected STT provider.")
    if not text:
        raise InvalidResponseError("STT returned timed speech without transcript text.")

    tokens, prefix, quoted = [], "", set()
    for token in text.split():
        opens = _punctuation_opens(token, quoted)
        if _lexical_key(token):
            tokens.append(prefix + token)
            prefix = ""
        elif tokens and not opens:
            tokens[-1] += token
        else:
            prefix += token
    if prefix and tokens:
        tokens[-1] += prefix
    if len(tokens) != len(words) or any(
        _lexical_key(token) != _lexical_key(word.text)
        for token, word in zip(tokens, words)
    ):
        raise InvalidResponseError("STT transcript text and timed words do not align.")
    aligned = tuple(
        replace(word, text=_merge_punctuation(word.text, token))
        for token, word in zip(tokens, words)
    )
    return _text(" ".join(word.text for word in aligned)), aligned


def _reported_language(response, field: str) -> str | None:
    value = _field(response, field)
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise InvalidResponseError("STT returned an invalid language value.")
    return value


def _reported_model(response) -> str | None:
    value = _field(response, "model_id", _field(response, "model"))
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise InvalidResponseError("STT returned an invalid model value.")
    return value


def _local_result(response) -> TranscriptionResult:
    raw_segments = _field(response, "segments")
    if not isinstance(raw_segments, (list, tuple)):
        raise InvalidResponseError("Local Whisper did not return transcript segments.")
    segments, all_words, previous_end = [], [], 0.0
    for item in raw_segments:
        segment_text = _text(_field(item, "text"))
        raw_words = _field(item, "words", [])
        if not segment_text and raw_words == []:
            continue
        start, end = _interval(item, previous_end)
        words, _ = _words(raw_words)
        segment_text, words = _align_text(words, segment_text)
        if words[0].start < start or words[-1].end > end:
            raise InvalidResponseError(
                "Local Whisper words fall outside their segment."
            )
        segments.append(TranscriptSegment(segment_text, start, end, words))
        all_words.extend(words)
        previous_end = end
    text, words = _align_text(tuple(all_words), _field(response, "text"))
    # Keep native utterance boundaries for the clustered segmenter.
    cursor = 0
    for index, segment in enumerate(segments):
        count = len(segment.words)
        segments[index] = replace(segment, words=words[cursor : cursor + count])
        cursor += count
    return TranscriptionResult(
        text=text,
        words=words,
        language=_reported_language(response, "language"),
        segments=tuple(segments),
    )


def _elevenlabs_result(response) -> TranscriptionResult:
    words, events = _words(_field(response, "words"), elevenlabs=True)
    text, words = _align_text(words, _field(response, "text"), events)
    return TranscriptionResult(
        text=text, words=words, language=_reported_language(response, "language_code")
    )


def _wav_duration(path: Path) -> float | None:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes() / audio.getframerate()
    except (OSError, EOFError, wave.Error, ZeroDivisionError):
        return None


@contextmanager
def _upload_audio(path: Path, timeout: float):
    if path.suffix.lower() in _AUDIO_SUFFIXES:
        yield path
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ConfigurationError(
            "FFmpeg is required to extract audio from this media file."
        )
    with tempfile.TemporaryDirectory(prefix="ovs-stt-") as directory:
        output = Path(directory) / "audio.wav"
        # Preserve gaps and a delayed audio stream relative to the video. There
        # is no seek, crop, or speech-dependent offset applied to the source.
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-copyts",
            "-start_at_zero",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-af",
            "aresample=async=1:first_pts=0",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise ServiceTimeoutError(
                "Audio extraction exceeded the configured STT timeout."
            ) from None
        except (OSError, subprocess.CalledProcessError):
            raise ConfigurationError(
                "Audio extraction failed. Check FFmpeg and the media audio stream."
            ) from None
        if not output.is_file() or output.stat().st_size == 0:
            raise ConfigurationError("Audio extraction did not produce readable audio.")
        yield output


def _remote_failure(error: Exception, httpx) -> tuple[ProviderError, bool]:
    if isinstance(error, ProviderError):
        return error, False
    if isinstance(error, (TimeoutError, httpx.TimeoutException)):
        return ServiceTimeoutError("ElevenLabs transcription timed out."), True
    if isinstance(error, httpx.TransportError):
        return ServiceUnavailableError("ElevenLabs could not be reached."), True
    status = getattr(error, "status_code", None)
    if status in (401, 403):
        return (
            AuthenticationError(
                "ElevenLabs rejected the configured credentials or access."
            ),
            False,
        )
    if status == 429:
        return RateLimitError("ElevenLabs rate or concurrency limit was reached."), True
    if status in (408, 504):
        return ServiceTimeoutError("ElevenLabs transcription timed out."), True
    if isinstance(status, int) and status >= 500:
        return (
            ServiceUnavailableError(
                "ElevenLabs transcription is temporarily unavailable."
            ),
            True,
        )
    if isinstance(status, int) and 400 <= status < 500:
        return (
            ConfigurationError(
                "ElevenLabs rejected the transcription request. Check model, language, media, and account access."
            ),
            False,
        )
    if isinstance(error, ValueError) or (
        isinstance(status, int) and 200 <= status < 300
    ):
        return (
            InvalidResponseError(
                "ElevenLabs returned an invalid transcription response."
            ),
            False,
        )
    if isinstance(error, (TypeError, AttributeError)):
        return (
            ConfigurationError(
                "The installed ElevenLabs SDK is incompatible with the STT adapter."
            ),
            False,
        )
    return ServiceUnavailableError("ElevenLabs transcription failed."), False


class _STTAdapter:
    def __init__(self, config: STTConfig):
        self.config = config
        self.records: list[ServiceMetadata] = []

    def _metadata(
        self,
        started: float,
        language: str | None,
        sdk_version: str | None,
        *,
        response=None,
        result: TranscriptionResult | None = None,
        attempts: int = 1,
        status: str = "completed",
        audio_duration: float | None = None,
    ) -> ServiceMetadata:
        metadata = ServiceMetadata(
            provider=self.config.provider,
            requested_model=self.config.model,
            returned_model=_reported_model(response) if result is not None else None,
            requested_language=language,
            reported_language=result.language if result is not None else None,
            duration_seconds=time.perf_counter() - started,
            audio_duration_seconds=audio_duration,
            adapter_version="1",
            sdk_version=sdk_version,
            attempts=attempts,
            status=status,
        )
        self.records.append(metadata)
        return metadata


class LocalWhisperSTT(_STTAdapter):
    """Lazy CPU Whisper with its native utterance boundaries retained."""

    def __init__(self, config: STTConfig):
        super().__init__(config)
        self._model = None

    def preflight(self) -> None:
        _validate_config(self.config)
        _language(self.config.language)
        _require_dependency("whisper_timestamped")
        if shutil.which("ffmpeg") is None:
            raise ConfigurationError("Local Whisper requires FFmpeg on PATH.")

    def transcribe(
        self, media_path: str | Path, language: str | None = None
    ) -> TranscriptionResult:
        self.preflight()
        language = _language(self.config.language if language is None else language)
        path = _media_path(media_path)
        started, sdk_version = time.perf_counter(), _version("whisper-timestamped")
        audio_duration = None
        try:
            try:
                whisper = importlib.import_module("whisper_timestamped")
            except ImportError:
                raise ConfigurationError(
                    "Local Whisper dependencies could not be loaded."
                ) from None
            if self._model is None:
                self._model = whisper.load_model(
                    self.config.model,
                    device="cpu",
                    download_root=str(PROJECT_DIR / ".cache/whisper"),
                )
            audio = whisper.load_audio(str(path))
            audio_duration = len(audio) / 16000
            response = whisper.transcribe(
                self._model,
                audio,
                language=language,
                fp16=False,
                verbose=None,
            )
            result = _local_result(response)
            metadata = self._metadata(
                started,
                language,
                sdk_version,
                response=response,
                result=result,
                audio_duration=audio_duration,
            )
            return replace(result, metadata=metadata)
        except Exception as error:
            if isinstance(error, ProviderError):
                failure = error
            elif isinstance(error, TimeoutError):
                failure = ServiceTimeoutError(
                    "Local Whisper timed out while loading or transcribing."
                )
            elif isinstance(
                error, (ValueError, FileNotFoundError, subprocess.CalledProcessError)
            ):
                failure = ConfigurationError(
                    "Local Whisper could not load the configured model or media."
                )
            else:
                failure = ServiceUnavailableError(
                    "Local Whisper could not complete transcription."
                )
            self._metadata(
                started,
                language,
                sdk_version,
                status=type(failure).__name__,
                audio_duration=audio_duration,
            )
            raise failure from None


class ElevenLabsSTT(_STTAdapter):
    """Synchronous Scribe upload with explicit timeouts and bounded retries."""

    def preflight(self) -> None:
        _validate_config(self.config)
        _language(self.config.language)
        if not isinstance(self.config.api_key, str) or not self.config.api_key.strip():
            raise ConfigurationError(
                "ELEVENLABS_API_KEY must be configured for ElevenLabs STT."
            )
        _require_dependency("elevenlabs")

    def transcribe(
        self, media_path: str | Path, language: str | None = None
    ) -> TranscriptionResult:
        self.preflight()
        language = _language(self.config.language if language is None else language)
        path = _media_path(media_path)
        started, sdk_version = time.perf_counter(), _version("elevenlabs")
        attempts, audio_duration = 0, None
        try:
            try:
                client_class = importlib.import_module("elevenlabs.client").ElevenLabs
                httpx = importlib.import_module("httpx")
            except (ImportError, AttributeError):
                raise ConfigurationError(
                    "The ElevenLabs SDK dependencies could not be loaded."
                ) from None
            with _upload_audio(path, self.config.timeout_seconds) as audio_path:
                audio_duration = _wav_duration(audio_path)
                with httpx.Client(timeout=self.config.timeout_seconds) as transport:
                    client = client_class(
                        api_key=self.config.api_key,
                        timeout=self.config.timeout_seconds,
                        httpx_client=transport,
                    )
                    for attempts in range(1, self.config.max_attempts + 1):
                        try:
                            upload = audio_path.open("rb")
                        except OSError:
                            raise ConfigurationError(
                                "The transcription audio file could not be read."
                            ) from None
                        try:
                            with upload:
                                response = client.speech_to_text.convert(
                                    file=upload,
                                    model_id=self.config.model,
                                    language_code=language,
                                    timestamps_granularity="word",
                                    tag_audio_events=False,
                                    diarize=False,
                                    request_options={"max_retries": 0},
                                )
                            break
                        except Exception as error:
                            failure, retryable = _remote_failure(error, httpx)
                            if not retryable or attempts == self.config.max_attempts:
                                raise failure from None
                            time.sleep(
                                min(
                                    self.config.retry_backoff_seconds
                                    * 2 ** (attempts - 1),
                                    30.0,
                                )
                            )
            result = _elevenlabs_result(response)
            reported_duration = _field(response, "audio_duration_secs")
            if reported_duration is not None:
                audio_duration = _timestamp(reported_duration)
            metadata = self._metadata(
                started,
                language,
                sdk_version,
                response=response,
                result=result,
                attempts=attempts,
                audio_duration=audio_duration,
            )
            return replace(result, metadata=metadata)
        except Exception as error:
            failure = (
                error
                if isinstance(error, ProviderError)
                else ServiceUnavailableError(
                    "ElevenLabs transcription could not complete."
                )
            )
            self._metadata(
                started,
                language,
                sdk_version,
                attempts=attempts,
                status=type(failure).__name__,
                audio_duration=audio_duration,
            )
            raise failure from None
