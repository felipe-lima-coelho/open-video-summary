"""CPU research pipeline entry points with independently selected providers."""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone

from open_video_summary.errors import ConfigurationError
from open_video_summary.utils.config import PROJECT_DIR, ModelPaths
from open_video_summary.utils.paths import project_path, portable_path
from open_video_summary.utils.providers import load_thread_count, load_visual_scope


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

    for criterion in HSMVideoSumm.selection_criteria:
        if criterion.name == "QualityPick":
            criterion.visual_threads = args.threads
            criterion.visual_scope = getattr(args, "visual_scope", "segment")

    output = project_path(args.output)
    handler_path = output.with_name(f"{output.stem}_handler.json")
    metadata_path = output.with_suffix(".json")
    audit_path = output.with_name(f"{output.stem}_audit.json")
    print(
        f"HSMVideoSumm on CPU: {len(videos)} source videos, "
        f"{sum(len(v.segments) for v in videos)} input segments; "
        f"visual scope {getattr(args, 'visual_scope', 'segment')}.",
        flush=True,
    )
    information_kwargs = {}
    if getattr(args, "analyze_information", False) or getattr(args, "information_report", None):
        from open_video_summary.core.summarizers.information_config import configured_information_analyzer

        information_kwargs = {
            "information_analyzer": configured_information_analyzer(vars(args)),
            "information_output_path": args.information_report,
            "information_csv_path": getattr(args, "information_csv", None),
            "information_input_path": args.dataset,
        }
    summary = HSMVideoSumm.summarize(
        videos=videos,
        title=args.title,
        video_output_path=output.as_posix(),
        handler_output_path=handler_path.as_posix(),
        audit_output_path=audit_path.as_posix(),
        **information_kwargs,
    )
    if information_kwargs:
        _print_information_report(HSMVideoSumm.last_information_report, HSMVideoSumm.last_information_path)
    visual_profile_path = output.with_name(f"{output.stem}_visual_profile.json")
    visual_profile = next(
        criterion.last_profile
        for criterion in HSMVideoSumm.selection_criteria
        if criterion.name == "QualityPick"
    )
    visual_profile_path.parent.mkdir(parents=True, exist_ok=True)
    visual_profile_path.write_text(
        json.dumps(visual_profile, indent=2), encoding="utf-8"
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
    print(f"Selection audit: {portable_path(audit_path)}")
    print(f"Visual timings: {portable_path(visual_profile_path)}")
    if not args.no_render:
        print(f"Video summary: {portable_path(output)}")


def _print_information_report(report, path):
    if report is None:
        print("Information analysis: failed before a snapshot report could be built.")
        return
    occurrence_label = (
        "provisional occurrence evidence groups (alignment pending)"
        if report.counts.occurrences_provisional
        else "identified occurrences"
    )
    print(
        f"Information analysis: {report.status}; "
        f"{report.counts.unique_units} identified units, "
        f"{report.counts.occurrences} {occurrence_label}; {len(report.issues)} issues."
    )
    if path:
        print(f"Information report: {portable_path(path)}")


def _analyze_information(args):
    from open_video_summary.core.summarizers.information_config import configured_information_analyzer
    from open_video_summary.core.summarizers.information_contracts import capture_snapshot
    from open_video_summary.core.summarizers.information_io import save_information_report
    from open_video_summary.parsers.video import VideoLoader

    # This text-only path needs neither video files nor an HSM classifier.
    videos = VideoLoader.load_videos_from_json(args.dataset)
    snapshot = capture_snapshot(videos, stage_id="source_inventory")
    report = configured_information_analyzer(vars(args)).analyze(snapshot)
    reserved = [args.dataset]
    reserved.extend(video.path for video in videos)
    reserved.extend(segment.video_path for video in videos for segment in video.segments)
    destination = args.output or f"outputs/information/{report.run_id}.json"
    path = save_information_report(report, destination, reserved=reserved, csv_path=args.information_csv)
    _print_information_report(report, path)
    if report.status != "completed":
        raise RuntimeError(f"Information analysis is {report.status}; inspect the saved report's issues.")


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
    import shutil
    from open_video_summary.adapters.factory import create_llm, create_stt
    from open_video_summary.errors import ConfigurationError
    from open_video_summary.utils.providers import load_provider_config

    output = project_path(args.output)
    run_path = output.with_name(f"{output.stem}_run.json")
    started = time.monotonic()
    report = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "input": portable_path(args.input),
        "output": portable_path(output),
    }
    llm, stt = None, None
    try:
        config = load_provider_config(vars(args))
        report["settings"] = {
            "llm": {
                "provider": config.llm.provider,
                "model": config.llm.model,
                "reasoning_effort": config.llm.reasoning_effort,
                "timeout_seconds": config.llm.timeout_seconds,
            },
            "stt": {
                "provider": config.stt.provider,
                "model": config.stt.model,
                "language": config.stt.language,
                "timeout_seconds": config.stt.timeout_seconds,
            },
            "max_attempts": config.llm.max_attempts,
        }
        llm = create_llm(config.llm)
        stt = create_stt(config.stt)
        # Validate selected credentials and dependencies before any model loads.
        stt.preflight()
        if shutil.which("ffmpeg") is None:
            raise ConfigurationError(
                "FFmpeg must be on PATH to segment raw MP4 videos."
            )
        from open_video_summary.core.segmenter.video_segmenter import WordVideoSegmenter
        from open_video_summary.parsers.video import VideoLoader, VideoDumper

        segmenter = WordVideoSegmenter(
            min_segment_length=5,
            max_segment_length=120,
            llm_adapter=llm,
            stt_adapter=stt,
        )
        segmenter.preflight()
        videos = VideoLoader.load_videos_from_directory(args.input)
        if not videos:
            raise ValueError("The input directory contains no MP4 videos.")
        report["input_video_count"] = len(videos)
        videos = segmenter.create_videos_segments(videos, language=config.stt.language)
        VideoDumper.dump_videos_to_json(videos, args.output)
        report["status"] = "completed"
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        raise
    finally:
        report["duration_seconds"] = time.monotonic() - started
        report["llm_calls"] = [asdict(item) for item in llm.records] if llm else []
        report["stt_calls"] = [asdict(item) for item in stt.records] if stt else []
        for adapter in (llm, stt):
            close = getattr(adapter, "close", None)
            if close is not None:
                close()
        run_path.parent.mkdir(parents=True, exist_ok=True)
        run_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"Segment metadata: {portable_path(args.output)}")
    print(f"Run metadata: {portable_path(run_path)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open Video Summary CPU runner.")
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "CPU thread budget (default: OVS_THREADS or 2); summarize uses up to "
            "this many visual workers."
        ),
    )
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
    summarize.add_argument(
        "--visual-scope", choices=("segment", "video"), default=None,
        help="Visual features from segment intervals or full source videos (default: OVS_VISUAL_SCOPE or segment).",
    )
    summarize.add_argument("--analyze-information", action="store_true", help="Run a passive transcript inventory before Introduction; save a separate report.")
    summarize.add_argument("--information-report", default=None, help="Enable inventory and save a new JSON file under outputs/.")
    _information_arguments(summarize)
    information = subparsers.add_parser("analyze-information", help="Inventory segmented transcript text without rendering or classifier assets.")
    information.add_argument("--dataset", default="data/processed/bebe_real.json")
    information.add_argument("--output", default=None, help="New JSON under outputs/ (default: unique outputs/information/ file).")
    _information_arguments(information)
    subparsers.add_parser(
        "doctor", help="Check the environment and required demo assets."
    )
    segment = subparsers.add_parser(
        "segment", help="Transcribe/topic-segment raw MP4s with configured providers."
    )
    segment.add_argument("--input", required=True)
    segment.add_argument("--output", default="outputs/segments.json")
    segment.add_argument("--llm-provider", default=None)
    segment.add_argument("--llm-model", default=None)
    segment.add_argument(
        "--llm-reasoning-effort", dest="reasoning_effort", default=None
    )
    segment.add_argument("--llm-base-url", default=None)
    segment.add_argument("--stt-provider", default=None)
    segment.add_argument(
        "--stt-model", "--whisper-model", dest="stt_model", default=None
    )
    segment.add_argument("--stt-language", "--language", dest="language", default=None)
    segment.add_argument("--llm-timeout", type=float, default=None)
    segment.add_argument("--stt-timeout", type=float, default=None)
    segment.add_argument("--max-attempts", type=int, default=None)
    return parser


def _information_arguments(parser):
    parser.add_argument("--information-csv", default=None, help="Optional occurrence table in a new CSV under outputs/.")
    parser.add_argument("--no-information-qa", dest="information_qa", action="store_false", default=None)
    parser.add_argument("--information-max-calls", type=int, default=None)
    parser.add_argument("--information-rounds", type=int, default=None)
    parser.add_argument("--information-max-pairs", type=int, default=None)
    parser.add_argument("--information-max-candidates", type=int, default=None)
    parser.add_argument("--information-context-chars", type=int, default=None)
    parser.add_argument("--information-context-segments", type=int, default=None)
    parser.add_argument("--information-acceptance", type=float, default=None)
    parser.add_argument("--information-gap-threshold", type=float, default=None)
    parser.add_argument("--information-equivalence", type=float, default=None)
    parser.add_argument("--evaluator-provider", default=None)
    parser.add_argument("--evaluator-model", default=None)
    parser.add_argument("--evaluator-base-url", default=None)
    parser.add_argument("--evaluator-timeout", type=float, default=None)
    parser.add_argument("--evaluator-max-attempts", type=int, default=None)
    parser.add_argument("--llm-provider", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-reasoning-effort", dest="reasoning_effort", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-timeout", type=float, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive.")
    try:
        args.threads = load_thread_count(args.threads)
        if args.command == "summarize":
            args.visual_scope = load_visual_scope(args.visual_scope)
    except ConfigurationError as exc:
        parser.error(str(exc))
    _cpu_settings(args.threads)
    try:
        if args.command == "prepare-demo":
            from open_video_summary.demo import prepare_demo

            prepare_demo(download_model=not args.without_model)
        else:
            {"summarize": _summarize, "analyze-information": _analyze_information, "doctor": _doctor, "segment": _segment}[
                args.command
            ](args)
    except (FileNotFoundError, ValueError, RuntimeError, ConnectionError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
