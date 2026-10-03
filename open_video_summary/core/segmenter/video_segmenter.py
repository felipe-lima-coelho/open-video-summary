import itertools
import unicodedata
from typing import Optional
from math import ceil
from abc import abstractmethod

from open_video_summary.adapters.factory import create_llm, create_stt
from open_video_summary.contracts import (
    GenerationRequest,
    LanguageModel,
    OutputSpec,
    SpeechToText,
    TimedWord,
    TranscriptSegment,
    TranscriptionResult,
)
from open_video_summary.core.segmenter.prompts import VideoSegmenterPrompts
from open_video_summary.entities.video import Video, VideoSegment
from open_video_summary.errors import NoSpeechError
from open_video_summary.handlers.segment import SegmentsCluster
from open_video_summary.utils.paths import project_path
from open_video_summary.utils.providers import load_provider_config


def _sentence_end(text: str, punctuation: str = ".!?。！？") -> bool:
    text = text.rstrip()
    while text and (
        text[-1] in "\"'”’»)]}"
        or unicodedata.category(text[-1]) in {"Pe", "Pf"}
    ):
        text = text[:-1].rstrip()
    return text.endswith(tuple(punctuation))


def _words_text(words: list[TimedWord] | tuple[TimedWord, ...]) -> str:
    if not words:
        return ""
    return "".join(
        word.text + word.separator_after for word in words[:-1]
    ) + words[-1].text


class BaseVideoSegmenter:
    @abstractmethod
    def create_video_segments(self, video: Video, language: str | None = None) -> Video:
        pass


class TopicsBasedVideoSegmenter(BaseVideoSegmenter):
    def __init__(
        self,
        whisper_model: str | None = None,
        min_segment_length: int = 10,
        max_segment_length: Optional[int] = None,
        max_subtopics: Optional[int] = None,
        prompts_template: VideoSegmenterPrompts | None = None,
        llm_adapter: LanguageModel | None = None,
        stt_adapter: SpeechToText | None = None,
    ) -> None:
        self.whisper_model = whisper_model
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.max_subtopics = max_subtopics
        self.prompts_template = prompts_template or VideoSegmenterPrompts()
        if llm_adapter is None or stt_adapter is None:
            config = load_provider_config({"stt_model": whisper_model})
            llm_adapter = llm_adapter or create_llm(config.llm)
            stt_adapter = stt_adapter or create_stt(config.stt)
        self.llm_adapter = llm_adapter
        self.stt_adapter = stt_adapter
        self._providers_ready = False

    def preflight(self) -> None:
        if not self._providers_ready:
            self.stt_adapter.preflight()
            self.llm_adapter.preflight()
            self._providers_ready = True

    def transcribe_video(
        self, video_path: str, language: str | None = None
    ) -> TranscriptionResult:
        self.preflight()
        result = self.stt_adapter.transcribe(
            project_path(video_path), language=language
        )
        if not result.words or not result.text.strip():
            raise NoSpeechError("The input audio contains no timestamped speech.")
        return result

    def calculate_max_subtopics(
        self,
        video_duration_in_sec: int,
        min_topics: int = 3,
        exponent: float = 0.5,
        scale: float = 2.0,
    ) -> int:
        minutes = ceil(video_duration_in_sec / 60)
        return ceil(min_topics + scale * (minutes**exponent - 1))

    def get_llm_response(
        self,
        prompt: str,
        pattern: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        output = (
            OutputSpec(kind="pattern", pattern=pattern) if pattern else OutputSpec()
        )
        return self.llm_adapter.generate(
            GenerationRequest(prompt=prompt, output=output, temperature=temperature)
        ).value.strip()

    def load_video_topics(self, full_document: str, video: Video) -> dict[str, str]:
        max_subtopics = self.max_subtopics
        if max_subtopics is None:
            from moviepy import VideoFileClip

            with VideoFileClip(str(project_path(video.path))) as clip:
                max_subtopics = self.calculate_max_subtopics(
                    video_duration_in_sec=int(clip.duration)
                )
        topics_prompt = self.prompts_template.generate_subtopics.format(
            full_video_transcript=full_document, max_subtopics=max_subtopics
        )
        return self.llm_adapter.generate(
            GenerationRequest(
                prompt=topics_prompt,
                output=OutputSpec(kind="topics", max_items=max_subtopics),
            )
        ).value

    def load_global_topics(self, videos: list[Video]) -> list[str]:
        all_topics = list(itertools.chain.from_iterable([v.topics for v in videos]))
        topics_prompt = self.prompts_template.get_global_topics.format(
            topics_collection=all_topics
        )
        return self.llm_adapter.generate(
            GenerationRequest(
                prompt=topics_prompt,
                output=OutputSpec(kind="string_list"),
            )
        ).value

    def adjust_segments_order(self, segments: list[VideoSegment]) -> list[VideoSegment]:
        adjusted_segments = []
        for i, seg in enumerate(segments):
            seg.order = i
            adjusted_segments.append(seg)
        return adjusted_segments


class WordVideoSegmenter(TopicsBasedVideoSegmenter):
    def __init__(
        self,
        whisper_model: str | None = None,
        min_segment_length: int = 10,
        max_segment_length: Optional[int] = None,
        max_subtopics: Optional[int] = None,
        sentence_boundary: str = ".!?。！？",
        prompts_template: VideoSegmenterPrompts | None = None,
        llm_adapter: LanguageModel | None = None,
        stt_adapter: SpeechToText | None = None,
    ) -> None:
        super().__init__(
            whisper_model=whisper_model,
            min_segment_length=min_segment_length,
            max_segment_length=max_segment_length,
            max_subtopics=max_subtopics,
            prompts_template=prompts_template,
            llm_adapter=llm_adapter,
            stt_adapter=stt_adapter,
        )
        self.sentence_boundary = sentence_boundary

    def get_words_from_segments(
        self, segments: list[TranscriptSegment]
    ) -> list[TimedWord]:
        return list(
            itertools.chain.from_iterable(segment.words for segment in segments)
        )

    def get_segments_from_words(
        self, words: list[TimedWord] | tuple[TimedWord, ...]
    ) -> list[VideoSegment]:
        segments: list[VideoSegment] = []
        curr_seg_data: dict = {"start": None, "end": None, "words": []}

        for word in words:
            if curr_seg_data["start"] is None:
                curr_seg_data["start"] = word.start
            curr_seg_data["end"] = word.end
            curr_seg_data["words"].append(word)
            if (
                _sentence_end(word.text, self.sentence_boundary)
                and curr_seg_data["end"] - curr_seg_data["start"]
                >= self.min_segment_length
            ):
                segments.append(
                    VideoSegment(
                        content=_words_text(curr_seg_data["words"]),
                        start=curr_seg_data["start"],
                        end=curr_seg_data["end"],
                    )
                )

                curr_seg_data = {"start": None, "end": None, "words": []}

        if curr_seg_data["words"]:
            segments.append(
                VideoSegment(
                    content=_words_text(curr_seg_data["words"]),
                    start=curr_seg_data["start"],
                    end=curr_seg_data["end"],
                )
            )

        return segments

    def classify_segment_topics(
        self,
        segments: list[VideoSegment],
        topics: dict[str, str],
        local_topics: bool = True,
    ) -> list[VideoSegment]:
        classified_segments: list[VideoSegment] = []

        for segment in segments:
            prompt = self.prompts_template.classify_subtopic.format(
                content=segment.content, topics=topics
            )
            segment_topic = self.llm_adapter.generate(
                GenerationRequest(
                    prompt=prompt,
                    output=OutputSpec(kind="topic", topic_ids=tuple(topics)),
                )
            ).value
            topic_id = next(iter(segment_topic))
            if local_topics:
                segment.video_topic = topics[topic_id]
            else:
                segment.global_topic = topics[topic_id]

            previous = classified_segments[-1] if classified_segments else None
            if (
                local_topics
                and previous is not None
                and previous.video_topic == segment.video_topic
                and (
                    self.max_segment_length is None
                    or (segment.end - previous.start) <= self.max_segment_length
                )
            ):
                previous.content += f" {segment.content}"
                previous.end = segment.end
            else:
                classified_segments.append(segment)

        return classified_segments

    def create_video_segments(self, video: Video, language: str | None = None) -> Video:
        transcription = self.transcribe_video(video.path, language)
        video_segments = self.get_segments_from_words(transcription.words)

        # Adding topics to video dataclass
        if not video.topics:
            video_topics = self.load_video_topics(transcription.text, video)
            video.topics = list(video_topics.values())
        else:
            video_topics = {str(i): topic for i, topic in enumerate(video.topics)}

        video_segments = self.classify_segment_topics(video_segments, video_topics)
        video_segments = self.adjust_segments_order(video_segments)
        for segment in video_segments:
            segment.video_path = video.path
        video.segments = video_segments
        return video

    def create_videos_segments(
        self,
        videos: list[Video],
        language: str | None = None,
        global_topics: Optional[list[str] | dict[str, str]] = None,
    ) -> list[Video]:
        videos = [self.create_video_segments(video, language) for video in videos]

        if not global_topics:
            global_topics = self.load_global_topics(videos)

        if isinstance(global_topics, list):
            global_topics = {str(i): topic for i, topic in enumerate(global_topics)}

        for video in videos:
            video.segments = self.classify_segment_topics(
                video.segments, global_topics, local_topics=False
            )

        return videos


class ClusteredVideoSegmenter(TopicsBasedVideoSegmenter):
    def __init__(
        self,
        whisper_model: str | None = None,
        min_segment_length: int = 10,
        max_segment_length: int = 300,
        max_subtopics: Optional[int] = None,
        segment_overlap_ratio: float = 0.5,
        max_phrase_pause_interval: float = 0.7,
        prompts_template: VideoSegmenterPrompts | None = None,
        llm_adapter: LanguageModel | None = None,
        stt_adapter: SpeechToText | None = None,
    ) -> None:
        super().__init__(
            whisper_model=whisper_model,
            min_segment_length=min_segment_length,
            max_segment_length=max_segment_length,
            max_subtopics=max_subtopics,
            prompts_template=prompts_template,
            llm_adapter=llm_adapter,
            stt_adapter=stt_adapter,
        )
        self.segment_overlap_ratio = segment_overlap_ratio
        self.max_phrase_pause_interval = max_phrase_pause_interval

    def transcript_segments(
        self, result: TranscriptionResult
    ) -> list[TranscriptSegment]:
        """Keep native local boundaries; group words when a provider has none."""
        if result.segments:
            return list(result.segments)
        segments = []
        words = []
        for word in result.words:
            words.append(word)
            if _sentence_end(word.text):
                segments.append(
                    TranscriptSegment(
                        text=_words_text(words),
                        start=words[0].start,
                        end=words[-1].end,
                        words=tuple(words),
                    )
                )
                words = []
        if words:
            segments.append(
                TranscriptSegment(
                    text=_words_text(words),
                    start=words[0].start,
                    end=words[-1].end,
                    words=tuple(words),
                )
            )
        return segments

    def list_overlapping_clusters(
        self, segmented_document: list[TranscriptSegment]
    ) -> list[SegmentsCluster]:
        cluster_list, cluster = [], SegmentsCluster()
        num_segments = len(segmented_document)

        overlap_seconds = int(self.min_segment_length * self.segment_overlap_ratio)

        # Create overlaping segment cluster_list
        for i, s in enumerate(segmented_document):
            raw_segment = VideoSegment(
                order=i,
                start=s.start,
                end=s.end,
                content=s.text.strip(),
            )
            cluster.append(raw_segment)

            # Checking if current segment is the last one
            if i + 1 == num_segments:
                cluster_list.append(cluster)
                break

            # Checking if next segment is too close
            next_seg = segmented_document[i + 1]
            if (next_seg.start - raw_segment.end) < self.max_phrase_pause_interval:
                continue

            # Checking if segment matches ending criteria
            if cluster.duration >= self.min_segment_length and _sentence_end(
                cluster.last.content
            ):
                cluster_list.append(cluster)
                cluster = cluster.next_overlaping_cluster(overlap_seconds)

        return cluster_list

    def classify_segment_topics(
        self, cluster_list: list[SegmentsCluster], topics: dict[str, str]
    ) -> list[VideoSegment]:
        min_segments: dict[int, dict] = {}

        for cluster in cluster_list:
            prompt = self.prompts_template.classify_subtopic.format(
                content=cluster.content, topics=topics
            )
            cluster_topic = self.llm_adapter.generate(
                GenerationRequest(
                    prompt=prompt,
                    output=OutputSpec(kind="topic", topic_ids=tuple(topics)),
                )
            ).value
            topic_id = next(iter(cluster_topic))

            for cluster_seg in cluster.segments:
                if cluster_seg.order is not None and (
                    cluster_seg.order not in min_segments
                    or cluster.duration
                    < min_segments[cluster_seg.order]["cluster_duration"]
                ):
                    cluster_seg.video_topic = topics[topic_id]
                    min_segments[cluster_seg.order] = {
                        "segment": cluster_seg,
                        "cluster_duration": cluster.duration,
                    }

        return [min_seg["segment"] for min_seg in min_segments.values()]

    def fuse_similar_segments(
        self, min_segments: list[VideoSegment]
    ) -> list[VideoSegment]:
        video_segments, video_segment = [], None
        for current_segment in min_segments:
            if video_segment is None:
                video_segment = current_segment
                continue

            segment_duration = video_segment.end - video_segment.start
            current_duration = current_segment.end - current_segment.start
            combined_duration = segment_duration + current_duration

            # Extend current segment
            if (video_segment.video_topic == current_segment.video_topic) and (
                self.max_segment_length is not None
                and combined_duration < self.max_segment_length
            ):
                video_segment.content += f" {current_segment.content}"
                video_segment.end = current_segment.end
                continue

            # Flush current segment to segments list and start a new one
            video_segments.append(video_segment)
            video_segment = current_segment
            started_new = True

        if video_segment is not None:
            video_segments.append(video_segment)

        return video_segments

    def fix_segments_content(
        self, segments: list[TranscriptSegment], full_video_transcription: str
    ) -> list[TranscriptSegment]:
        adjusted_segments = []
        for seg in segments:
            prompt = self.prompts_template.fix_segment_transcription.format(
                full_video_transcript=full_video_transcription, content=seg.text
            )
            adjusted_segments.append(
                TranscriptSegment(
                    text=self.get_llm_response(prompt),
                    start=seg.start,
                    end=seg.end,
                    words=seg.words,
                )
            )

        return adjusted_segments

    def fix_content(self, content: str) -> str:
        prompt = self.prompts_template.fix_full_transcription.format(content=content)
        return self.get_llm_response(prompt)

    def create_video_segments(self, video: Video, language: str | None = None) -> Video:
        transcription = self.transcribe_video(video.path, language)
        full_document = transcription.text
        segmented_document = self.transcript_segments(transcription)

        # Using LLM to fix transcription errors
        fixed_full_document = self.fix_content(full_document)
        fixed_segmented_document = self.fix_segments_content(
            segmented_document, fixed_full_document
        )

        # Adding topics to video dataclass
        if not video.topics:
            video_topics = self.load_video_topics(full_document, video)
            video.topics = list(video_topics.values())
        else:
            video_topics = {str(i): topic for i, topic in enumerate(video.topics)}

        segment_clusters = self.list_overlapping_clusters(fixed_segmented_document)
        labled_min_segments = self.classify_segment_topics(
            segment_clusters, video_topics
        )
        segments = self.fuse_similar_segments(labled_min_segments)
        segments = self.adjust_segments_order(segments)
        for segment in segments:
            segment.video_path = video.path

        # Adding segments to video dataclass
        video.segments = segments

        return video
