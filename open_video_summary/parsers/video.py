from __future__ import annotations

from pathlib import Path
from json import load, dump
from contextlib import ExitStack
from dacite import from_dict
from dataclasses import asdict
from moviepy.video.fx import FadeIn, FadeOut
from moviepy import VideoFileClip, concatenate_videoclips

from open_video_summary.entities.video import Video
from open_video_summary.utils.paths import project_path, video_paths


class VideoLoader:
    @staticmethod
    def load_videos_from_directory(
        directory: str, video_file_format: str = "mp4"
    ) -> list[Video]:
        return [
            VideoLoader.video_from_file(path.as_posix())
            for path in sorted(project_path(directory).rglob(f"*.{video_file_format}"))
        ]

    @staticmethod
    def load_videos_from_json(json_file: str) -> list[Video]:
        with project_path(json_file).open(encoding="utf-8") as file:
            videos_data = load(file)
        return list(map(VideoLoader.video_from_dict, videos_data))

    @staticmethod
    def video_from_file(video_file: str) -> Video:
        video_path = project_path(video_file)

        # Check if the file exists
        if not video_path.exists():
            raise FileNotFoundError(f"Video file {video_file} does not exist.")

        # Check if the file is a video file
        if video_path.suffix not in {".mp4", ".mov", ".avi", ".mkv"}:
            raise ValueError(f"File {video_file} is not a valid video file.")

        return Video(
            name=video_path.stem,
            path=(
                video_path.absolute() if not video_path.is_absolute() else video_path
            ).as_posix(),
        )

    @staticmethod
    def video_from_dict(video_data: dict) -> Video:
        return from_dict(data_class=Video, data=video_paths(video_data, resolve=True))


class VideoDumper:
    @staticmethod
    def dump_videos_to_json(videos: list[Video], json_file: str) -> None:
        output_path = project_path(json_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            dump(
                list(map(VideoDumper.video_to_dict, videos)),
                file,
                indent=4,
                ensure_ascii=False,
            )

    @staticmethod
    def video_to_dict(video: Video) -> dict:
        return video_paths(asdict(video))


class SummaryWriter:
    @staticmethod
    def write_video_summary(
        video: Video, fadein_seconds: float = 0.5, fadeout_seconds: float = 0.5
    ) -> None:
        fadein_fx = FadeIn(duration=fadein_seconds)
        fadeout_fx = FadeOut(duration=fadeout_seconds)

        if not video.segments:
            raise ValueError("The summary has no segments to render.")
        output_path = project_path(video.path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Reuse each reader and close it after encoding, including on errors.
        with ExitStack() as stack:
            sources = {}
            clips_list = []
            for segment in video.segments:
                source_path = project_path(segment.video_path).as_posix()
                if source_path not in sources:
                    sources[source_path] = stack.enter_context(
                        VideoFileClip(source_path)
                    )
                clip = sources[source_path].subclipped(segment.start, segment.end)
                clip = fadein_fx.apply(clip)
                clip = fadeout_fx.apply(clip)
                clips_list.append(clip)

            final_clip = concatenate_videoclips(clips_list, method="compose")
            stack.callback(final_clip.close)
            final_clip.write_videofile(
                output_path.as_posix(), codec="libx264", audio_codec="aac", threads=2
            )
