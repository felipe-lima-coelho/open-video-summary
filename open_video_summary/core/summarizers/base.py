from functools import reduce

from open_video_summary.entities.video import Video
from open_video_summary.core.selection_criteria.base import SelectionCriteria
from open_video_summary.handlers.summary import (
    SummarySegmentHandler,
    SummarySegmentHandlerIO,
)
from open_video_summary.utils.paths import project_path
from open_video_summary.utils.audit import finalize_summary_audit, save_summary_audit


class Summarizer:
    def __init__(self, selection_criteria: list[SelectionCriteria]) -> None:
        self.selection_criteria = selection_criteria
        self.last_handler = None
        self.last_audit = None

    def summarize(
        self,
        videos: list[Video],
        title: str = "",
        video_output_path: str = "output.mp4",
        handler_output_path: str = "output_handler.json",
        save_output: bool = True,
        audit_output_path: str | None = None,
        collect_audit: bool = True,
    ) -> Video:
        handler = SummarySegmentHandler(audit_enabled=collect_audit)
        handler.set_source_videos(videos)
        error = None
        try:
            handler = reduce(lambda h, c: c.evaluate(h), self.selection_criteria, handler)
            handler.finalize()
        except Exception as exc:
            error = exc
            for report in handler.audit["criteria"].values():
                if report.get("status") == "in_progress":
                    report["status"] = "failed"
            raise
        finally:
            finalize_summary_audit(handler, status="failed" if error else "completed", error=error)
            self.last_handler = handler
            self.last_audit = handler.audit if collect_audit else None
            if save_output:
                if error is None:
                    SummarySegmentHandlerIO.save(handler, handler_output_path)
                if collect_audit:
                    output = project_path(video_output_path)
                    audit_path = audit_output_path or output.with_name(f"{output.stem}_audit.json").as_posix()
                    save_summary_audit(handler.audit, audit_path)

        topics = list(set(segment.video_topic for segment in handler.output))
        return Video(
            name=title,
            segments=handler.output,
            topics=topics,
            path=video_output_path,
        )
