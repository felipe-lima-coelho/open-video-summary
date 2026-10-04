from typing import Optional
from math import ceil, isfinite
from open_video_summary.utils.paths import project_path
from open_video_summary.utils.processing.metrics import visual_count, visual_stage
from cv2 import (
    cvtColor,
    VideoCapture,
    CAP_PROP_FPS,
    COLOR_BGR2GRAY,
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_POS_FRAMES,
)


class VideoProcessor:
    @staticmethod
    def retrieve_segment_frames(
        video_path: str,
        start_second: int | float,
        end_second: int | float,
        target_fps: int | float = 1,
        grayscale: bool = False,
    ) -> list:
        """Sample source frames whose native timestamps are in ``[start, end)``.

        Sampling keeps the existing global frame stride and source resolution.
        Bounds use the reported floating-point FPS, including fractional rates.
        A verified frame seek avoids decoding the prefix for each interval; a
        backend that cannot report the requested position falls back to a fresh
        sequential reader. Other criteria retain their legacy retrieval method.
        """
        if (
            not isfinite(start_second)
            or not isfinite(end_second)
            or start_second < 0
            or end_second <= start_second
        ):
            raise ValueError("Visual interval must have finite 0 <= start < end.")
        if not isfinite(target_fps) or target_fps <= 0:
            raise ValueError("target_fps must be finite and positive.")

        with visual_stage("frame_decode"):
            path = project_path(video_path).as_posix()
            video = VideoCapture(path)
            decoded_frames = 0
            frames = []
            try:
                source_fps = video.get(CAP_PROP_FPS)
                frame_count = video.get(CAP_PROP_FRAME_COUNT)
                if (
                    not video.isOpened()
                    or not isfinite(source_fps)
                    or source_fps <= 0
                    or not isfinite(frame_count)
                    or frame_count < 1
                ):
                    raise ValueError(f"Cannot read video FPS/frame count: {video_path}")
                total_frames = int(frame_count)
                stride = int(int(source_fps) / target_fps)
                if stride < 1:
                    raise ValueError("target_fps exceeds the native sampling stride.")
                if start_second >= total_frames / source_fps:
                    return frames

                first = max(0, ceil(start_second * source_fps))
                # Compare timestamps directly to correct multiplication rounding
                # at an exact native-frame boundary.
                while first > 0 and (first - 1) / source_fps >= start_second:
                    first -= 1
                while first / source_fps < start_second:
                    first += 1
                first_sample = ((first + stride - 1) // stride) * stride
                if first_sample >= total_frames or first_sample / source_fps >= end_second:
                    return frames

                frame = 0
                if first_sample:
                    sought = video.set(CAP_PROP_POS_FRAMES, first_sample)
                    position = video.get(CAP_PROP_POS_FRAMES)
                    if sought and isfinite(position) and position == first_sample:
                        frame = first_sample
                        visual_count("interval_seeks")
                    else:
                        video.release()
                        video = VideoCapture(path)
                        visual_count("interval_seek_fallbacks")

                while frame < total_frames and frame / source_fps < end_second:
                    success, img = video.read()
                    if not success:
                        break
                    decoded_frames += 1
                    if frame >= first_sample and frame % stride == 0:
                        if grayscale:
                            img = cvtColor(img, COLOR_BGR2GRAY)
                        frames.append(img)
                    frame += 1
            finally:
                video.release()
                visual_count("video_reads")
                visual_count("decoded_frames", decoded_frames)
                visual_count("sampled_frames", len(frames))
        return frames

    @staticmethod
    def retrieve_video_frames(
        video_path: str,
        target_fps: int | float = 1,
        grayscale: bool = False,
        start_second: int | float = 0,
        end_second: Optional[int | float] = None,
    ) -> list:
        with visual_stage("frame_decode"):
            video = VideoCapture(project_path(video_path).as_posix())
            decoded_frames = 0
            frames = []
            try:
                source_fps = int(video.get(CAP_PROP_FPS))
                total_frames = int(video.get(CAP_PROP_FRAME_COUNT))
                end_second = end_second or (total_frames / source_fps)

                frames_interval = int(source_fps / target_fps)
                frame, success = -1, True
                while success:
                    success, img = video.read()
                    if not success:
                        break

                    decoded_frames += 1
                    frame += 1
                    if (frame / source_fps) < start_second:
                        continue

                    if (frame / source_fps) > end_second:
                        break

                    if frame % frames_interval != 0:
                        continue

                    if grayscale:
                        img = cvtColor(img, COLOR_BGR2GRAY)

                    frames.append(img)

            finally:
                video.release()
                visual_count("video_reads")
                visual_count("decoded_frames", decoded_frames)
                visual_count("sampled_frames", len(frames))
        return frames
