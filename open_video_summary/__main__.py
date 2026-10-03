"""Local CPU entry points for the existing research pipeline."""

import argparse
import json
import os
import sys

from open_video_summary.utils.config import PROJECT_DIR, ModelPaths
from open_video_summary.utils.paths import project_path, portable_path


def _cpu_settings(threads: int) -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
    ):
        os.environ[name] = str(threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".cache/huggingface"))


def _summarize(args) -> None:
    from open_video_summary.parsers.video import VideoLoader, VideoDumper, SummaryWriter

    videos = VideoLoader.load_videos_from_json(args.dataset)
    if not videos or any(not video.segments for video in videos):
        raise ValueError("The dataset must contain videos with preprocessed segments.")
    for video in videos:
        for path in {video.path, *(segment.video_path for segment in video.segments)}:
            if not project_path(path).is_file():
                raise FileNotFoundError(f"Missing source video: {path}")
    if not project_path(ModelPaths.SUBJECTIVITY_CLASSIFIER).is_dir():
        raise FileNotFoundError(
            "Run 'python -m open_video_summary prepare-demo' first."
        )

    import cv2
    import torch

    cv2.setNumThreads(args.threads)
    torch.set_num_threads(args.threads)
    from open_video_summary.core.summarizers import HSMVideoSumm

    output = project_path(args.output)
    handler_path = output.with_name(f"{output.stem}_handler.json")
    metadata_path = output.with_suffix(".json")
    print(
        f"HSMVideoSumm on CPU: {len(videos)} source videos, "
        f"{sum(len(v.segments) for v in videos)} input segments.",
        flush=True,
    )
    summary = HSMVideoSumm.summarize(
        videos=videos,
        title=args.title,
        video_output_path=output.as_posix(),
        handler_output_path=handler_path.as_posix(),
    )
    if not summary.segments:
        raise ValueError("HSMVideoSumm did not select any segments.")
    if not args.no_render:
        SummaryWriter.write_video_summary(summary)
    VideoDumper.dump_videos_to_json([summary], metadata_path.as_posix())
    print(
        f"Selected {len(summary.segments)} segments; "
        f"duration {sum(s.end - s.start for s in summary.segments):.2f} seconds."
    )
    print(f"Summary metadata: {portable_path(metadata_path)}")
    print(f"Selection log: {portable_path(handler_path)}")
    if not args.no_render:
        print(f"Video summary: {portable_path(output)}")


def _doctor(args) -> None:
    from importlib.metadata import version
    import shutil
    import cv2
    import torch
    import tensorflow as tf

    packages = [
        "numpy",
        "opencv-python",
        "moviepy",
        "tensorflow",
        "sentence-transformers",
        "torch",
        "whisper-timestamped",
    ]
    versions = {name: version(name) for name in packages}
    report = {
        "python": sys.version.split()[0],
        "packages": versions,
        "device": "cpu",
        "torch_cuda_available": torch.cuda.is_available(),
        "tensorflow_gpus": len(tf.config.list_physical_devices("GPU")),
        "ffmpeg_on_path": shutil.which("ffmpeg") is not None,
        "subjectivity_model": all(
            (project_path(ModelPaths.SUBJECTIVITY_CLASSIFIER) / name).is_file()
            for name in ("config.json", "model.safetensors", "tokenizer_config.json")
        ),
        "face_cascade": not cv2.CascadeClassifier(ModelPaths.FACE_CASCADE).empty(),
    }
    print(json.dumps(report, indent=2))
    if not report["face_cascade"] or not report["subjectivity_model"]:
        raise RuntimeError("Required demo assets are missing; run prepare-demo.")


def _segment(args) -> None:
    from open_video_summary.adapters.llm import OllamaAdapter
    from open_video_summary.core.segmenter.video_segmenter import WordVideoSegmenter
    from open_video_summary.parsers.video import VideoLoader, VideoDumper

    # Check the optional service before Whisper downloads a model.
    import ollama

    client = ollama.Client()
    import httpx

    try:
        available = client.list().get("models", [])
    except (httpx.HTTPError, ollama.ResponseError) as exc:
        raise RuntimeError(
            "Ollama is not available. Start its server and install the chosen model "
            "before segmenting raw videos."
        ) from exc
    model_names = {item.get("name") for item in available}
    if (
        args.llm_model not in model_names
        and f"{args.llm_model}:latest" not in model_names
    ):
        raise RuntimeError(
            f"Ollama model '{args.llm_model}' is missing; install it in Ollama first."
        )

    videos = VideoLoader.load_videos_from_directory(args.input)
    if not videos:
        raise ValueError("The input directory contains no MP4 videos.")
    segmenter = WordVideoSegmenter(
        whisper_model=args.whisper_model,
        min_segment_length=5,
        max_segment_length=120,
        llm_adapter=OllamaAdapter(model=args.llm_model),
    )
    videos = segmenter.create_videos_segments(videos, language=args.language)
    VideoDumper.dump_videos_to_json(videos, args.output)
    print(f"Segment metadata: {portable_path(args.output)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Open Video Summary local CPU runner.")
    parser.add_argument("--threads", type=int, default=2)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare-demo", help="Extract demo videos and original classifier."
    )
    prepare.add_argument("--without-model", action="store_true")
    summarize = subparsers.add_parser(
        "summarize", help="Run the original HSMVideoSumm on segmented videos."
    )
    summarize.add_argument("--dataset", default="data/processed/bebe_real.json")
    summarize.add_argument("--output", default="outputs/bebe_real_summary.mp4")
    summarize.add_argument("--title", default="Bebe Real Summary")
    summarize.add_argument("--no-render", action="store_true")
    subparsers.add_parser(
        "doctor", help="Check the environment and required demo assets."
    )
    segment = subparsers.add_parser(
        "segment", help="Transcribe/topic-segment raw MP4s using Whisper and Ollama."
    )
    segment.add_argument("--input", required=True)
    segment.add_argument("--output", default="outputs/segments.json")
    segment.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
    )
    segment.add_argument("--llm-model", default="gemma2")
    segment.add_argument("--language", default="pt")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive.")
    _cpu_settings(args.threads)
    try:
        if args.command == "prepare-demo":
            from open_video_summary.demo import prepare_demo

            prepare_demo(download_model=not args.without_model)
        else:
            {"summarize": _summarize, "doctor": _doctor, "segment": _segment}[
                args.command
            ](args)
    except (FileNotFoundError, ValueError, RuntimeError, ConnectionError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
