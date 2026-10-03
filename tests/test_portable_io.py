import json
import tempfile
import unittest
from pathlib import Path

from open_video_summary.entities.video import Video, VideoSegment
from open_video_summary.handlers.summary import (
    SummarySegmentHandler,
    SummarySegmentHandlerIO,
)
from open_video_summary.parsers.video import VideoLoader, VideoDumper
from open_video_summary.utils.config import PROJECT_DIR


class PortableIOTests(unittest.TestCase):
    def setUp(self):
        output_dir = PROJECT_DIR / "outputs"
        output_dir.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=output_dir)
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.video = Video(
            name="portable",
            path=(PROJECT_DIR / "data/raw/example.mp4").as_posix(),
            segments=[
                VideoSegment(content="Acentuação científica", start=0.0, end=1.0)
            ],
        )

    def test_loader_resolves_video_and_segment_paths(self):
        video = VideoLoader.video_from_dict(
            {
                "name": "sample",
                "path": "data/raw/example.mp4",
                "segments": [
                    {
                        "content": "sample",
                        "start": 0.0,
                        "end": 1.0,
                        "video_path": "data/raw/example.mp4",
                    }
                ],
            }
        )
        expected = (PROJECT_DIR / "data/raw/example.mp4").as_posix()
        self.assertEqual(expected, video.path)
        self.assertEqual(expected, video.segments[0].video_path)

    def test_dump_load_roundtrip_uses_relative_utf8_paths(self):
        path = self.output / "videos.json"
        VideoDumper.dump_videos_to_json([self.video], path.as_posix())
        data = json.loads(path.read_text(encoding="utf-8"))[0]
        self.assertEqual("data/raw/example.mp4", data["path"])
        self.assertEqual("data/raw/example.mp4", data["segments"][0]["video_path"])
        self.assertEqual(
            [self.video], VideoLoader.load_videos_from_json(path.as_posix())
        )

    def test_summary_log_roundtrip_uses_relative_paths(self):
        handler = SummarySegmentHandler()
        handler.set_source_videos([self.video])
        handler.include_segment(self.video.segments[0], "test")
        handler.finalize()
        path = self.output / "handler.json"
        SummarySegmentHandlerIO.save(handler, path.as_posix())
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn(PROJECT_DIR.as_posix(), json.dumps(data))
        restored = SummarySegmentHandlerIO.load(path.as_posix())
        self.assertEqual(handler.source, restored.source)
        self.assertEqual(handler.output, restored.output)
        self.assertEqual(handler.agent_logs, restored.agent_logs)

    def test_versioned_datasets_have_only_relative_paths(self):
        for path in (PROJECT_DIR / "data/processed").glob("*.json"):
            for video in json.loads(path.read_text(encoding="utf-8")):
                self.assertFalse(Path(video["path"]).is_absolute(), path.name)
                for segment in video["segments"]:
                    self.assertEqual(video["path"], segment["video_path"])
                    self.assertFalse(
                        Path(segment["video_path"]).is_absolute(), path.name
                    )


if __name__ == "__main__":
    unittest.main()
