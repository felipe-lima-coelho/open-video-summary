from functools import reduce
import warnings
from dataclasses import replace

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
        self.last_information_report = None
        self.last_information_path = None
        self.last_information_error = None

    def summarize(
        self,
        videos: list[Video],
        title: str = "",
        video_output_path: str = "output.mp4",
        handler_output_path: str = "output_handler.json",
        save_output: bool = True,
        audit_output_path: str | None = None,
        collect_audit: bool = True,
        information_analyzer=None,
        information_output_path: str | None = None,
        information_csv_path: str | None = None,
        information_input_path: str | None = None,
        information_report_observer=None,
    ) -> Video:
        self.last_information_report = None
        self.last_information_path = None
        self.last_information_error = None
        handler = SummarySegmentHandler(audit_enabled=collect_audit)
        handler.set_source_videos(videos)
        if information_analyzer is not None:
            self._analyze_information(
                handler, information_analyzer, information_output_path,
                information_csv_path, information_input_path, video_output_path,
                handler_output_path, audit_output_path, save_output,
            )
            if information_report_observer is not None:
                try:
                    information_report_observer(self.last_information_report,
                                                self.last_information_path)
                except Exception:
                    pass
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

    def _analyze_information(self, handler, analyzer, report_path, csv_path, input_path, video_path, handler_path, audit_path, save_output):
        from open_video_summary.core.summarizers.information_analysis import failed_information_report
        from open_video_summary.core.summarizers.information_contracts import AnalysisIssue, capture_snapshot
        from open_video_summary.core.summarizers.information_io import save_information_report

        try:
            snapshot = capture_snapshot(handler.source, stage_id="before_introduction", selection_stage_order=tuple(criterion.name for criterion in self.selection_criteria))
            try:
                report = analyzer.analyze(snapshot)
            except Exception as exc:
                report = failed_information_report(snapshot, exc)
            self.last_information_report = report
            if save_output or report_path is not None:
                output = project_path(video_path)
                reserved = [input_path, video_path, handler_path, audit_path or output.with_name(f"{output.stem}_audit.json"), output.with_suffix(".json"), output.with_name(f"{output.stem}_visual_profile.json")]
                reserved.extend(video.path for video in handler.source)
                reserved.extend(segment.video_path for video in handler.source for segment in video.segments)
                destination = report_path or f"outputs/information/{report.run_id}.json"
                try:
                    self.last_information_path = save_information_report(report, destination, reserved=reserved, csv_path=csv_path)
                except Exception as exc:
                    self.last_information_error = type(exc).__name__
                    report = replace(report, status="partial" if report.units else "failed", counts=replace(report.counts, valid_zero=False), issues=report.issues + (AnalysisIssue("report_write_failed", type(exc).__name__),))
                    self.last_information_report = report
                    # Preserve failure evidence in a distinct safe location while
                    # the existing selection flow continues without a new log action.
                    try:
                        self.last_information_path = save_information_report(report, f"outputs/information/{report.run_id}_failed.json", reserved=reserved)
                    except Exception:
                        pass
            if report.status != "completed":
                warnings.warn(f"Information analysis is {report.status}; inspect last_information_report and its issues.", RuntimeWarning, stacklevel=2)
        except Exception as exc:
            self.last_information_error = type(exc).__name__
            warnings.warn(f"Information analysis failed ({type(exc).__name__}); summary selection continues.", RuntimeWarning, stacklevel=2)
