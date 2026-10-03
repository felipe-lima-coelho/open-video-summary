import io
from PIL import Image
from numpy import ndarray
from cv2 import cvtColor, COLOR_BGR2RGB

from open_video_summary.utils import log
from open_video_summary.entities.video import Video, VideoSegment
from open_video_summary.utils.processing.video import VideoProcessor
from open_video_summary.handlers.summary import SummarySegmentHandler
from open_video_summary.adapters.factory import create_configured_llm
from open_video_summary.contracts import GenerationRequest, LanguageModel, OutputSpec
from open_video_summary.core.selection_criteria.base import SelectionCriteria


class VideoQuestionBasedFiltering(SelectionCriteria):
    USER_QUERY_PROMPT = """
        @USER_QUERY
        {user_query}

        #####
        @EXAMPLES

        Example 1
        Input: "I want to see images that show nature, mainly different animals and trees."
        Output:
        [
            "Are there any animals in the image?",
            "Is there a tree in the image?"
        ]

        Example 2
        Input: "Show segments where airplanes and people appear."
        Output:
        [
            "Is there a person in the image?",
            "Is there an airplane in the image?",
        ]

        #####
        @OUTPUT

        [
            "Inferred question 1",
            "Inferred question 2",
            "Inferred question 3",
            ...
        ]

        #####

    Analyze the user input in @USER_QUERY and create YES or NO questions.
    The questions should be simple, concise and straightforward.
    Compound questions should not be created.
    The questions should be designed to determine whether the content of a video segment is relevant to the @USER_QUERY.
    The final output should be list of strings, with each string being a binary yes or no question.
    """

    VIDEO_QUESTIONS_PROMPT = """
        @QUESTIONS
        {questions}

        #####
        @EXAMPLES

        Example 1: The image shows a dog in a park with trees and grass.
        Input: "1. Are there any animals in the image? 2. Is there a tree in the image?"
        Output:
        {{
            "1": True,
            "2": True
        }}

        Example 2: The image shows a busy street with cars and people, but no flying vehicles.
        Input: "1. Is there a person in the image? 2. Is there an airplane in the image? 3. What color is the sky?"
        Output:
        {{
            "1": True,
            "2": False,
            "3": None
        }}

        #####
        @OUTPUT

        {{
            "1": True,
            "2": False,
            "3": True,
            ...
        }}

        #####

        Analyze the image provided and answer each question in @QUESTIONS with ONLY True or False.
        Any answer that is neither True nor False should be answered as 'None'.
        The final output should be JSON object with the number of each question as key and either True or False as value.
    """

    def __init__(
        self,
        user_query: str = "",
        filter_questions: list[str] = [],
        keyframes_interval_seconds: int = 5,
        llm_adapter: LanguageModel | None = None,
        min_positive_answer_ratio: float = 0.5,
        convert_to_rgb: bool = True,
        include_segments: bool = True,
    ) -> None:
        super().__init__(read_from="source")

        if not user_query and not filter_questions:
            raise ValueError(
                "At least one of 'user_query' or 'filter_questions' must be provided."
            )

        self.user_query = user_query
        self.filter_questions = filter_questions
        self.keyframes_interval_seconds = keyframes_interval_seconds
        self.llm_adapter = llm_adapter
        self.min_positive_answer_ratio = min_positive_answer_ratio
        self.convert_to_rgb = convert_to_rgb
        self.include_segments = include_segments

    def _get_llm(self) -> LanguageModel:
        if self.llm_adapter is None:
            self.llm_adapter = create_configured_llm(
                default_models={"ollama": "ministral-3"}
            )
        return self.llm_adapter

    def evaluate(self, handler: SummarySegmentHandler) -> SummarySegmentHandler:
        videos = [
            video
            for video in self.get_criteria_input(handler)
            if isinstance(video, Video)
        ]
        log.info(f"Retrieved {len(videos)} videos to execute {self.name} criteria.")

        if self.filter_questions and self.user_query:
            log.info(
                "Both 'user_query' and 'filter_questions' provided. 'filter_questions' will be used for filtering."
            )

        if self.user_query and not self.filter_questions:
            self.create_questions_from_user_query()

        questions_dict = {i: item for i, item in enumerate(self.filter_questions)}
        prompt = VideoQuestionBasedFiltering.VIDEO_QUESTIONS_PROMPT.format(
            questions=questions_dict
        )

        for video in videos:
            segments = self.remove_discarded(handler, video.segments)
            segments = self.remove_outputted(handler, segments)
            for segment in segments:
                answers = self.search_on_segment(segment, prompt)
                positive_answers = sum(
                    answer for answer in answers.values() if isinstance(answer, bool)
                )
                positive_ratio = positive_answers / len(self.filter_questions)

                if positive_ratio >= self.min_positive_answer_ratio:
                    log.info(
                        f"Segment [{segment.start}s - {segment.end}s] of video '{video.name}' "
                        f"passed the filtering with {positive_answers} positive answers "
                        f"out of {len(self.filter_questions)} questions "
                        f"({positive_ratio:.2%} positive ratio)."
                    )
                    action = self.include if self.include_segments else self.discard
                    action(handler, segment)

        return handler

    def search_on_segment(self, segment: VideoSegment, prompt: str) -> dict:
        segment_keyframes = self.get_segment_frames_every_n_seconds(
            video_segment=segment
        )
        segment_keyframes_bytes = self.get_segment_byte_images(
            segment_keyframes=segment_keyframes
        )

        return (
            self._get_llm()
            .generate(
                GenerationRequest(
                    prompt=prompt,
                    output=OutputSpec(
                        kind="answers",
                        answer_ids=tuple(
                            str(i) for i in range(len(self.filter_questions))
                        ),
                    ),
                    temperature=None,
                    images=tuple(segment_keyframes_bytes),
                )
            )
            .value
        )

    def get_segment_byte_images(self, segment_keyframes: list) -> list[bytes]:
        return [
            self.get_image_bytes(segment_frame) for segment_frame in segment_keyframes
        ]

    def get_segment_frames_every_n_seconds(self, video_segment: VideoSegment) -> list:
        start_second = video_segment.start
        end_second = video_segment.end

        return VideoProcessor.retrieve_video_frames(
            video_path=video_segment.video_path,
            target_fps=1 / self.keyframes_interval_seconds,
            grayscale=False,
            start_second=start_second,
            end_second=end_second,
        )

    def get_image_bytes(self, segment_frame: ndarray) -> bytes:
        buffered = io.BytesIO()
        image = Image.fromarray(cvtColor(segment_frame, COLOR_BGR2RGB), "RGB")
        image.save(buffered, format="JPEG")

        return buffered.getvalue()

    def create_questions_from_user_query(self) -> None:
        log.info(
            "Only 'user_query' provided. Generating filter questions from user query."
        )
        prompt = VideoQuestionBasedFiltering.USER_QUERY_PROMPT.format(
            user_query=self.user_query
        )
        self.filter_questions = (
            self._get_llm()
            .generate(
                GenerationRequest(
                    prompt=prompt,
                    output=OutputSpec(kind="string_list"),
                    temperature=None,
                )
            )
            .value
        )
        log.info(
            f"Generated {len(self.filter_questions)} filter questions from user query."
        )
