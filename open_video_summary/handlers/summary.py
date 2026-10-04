from json import dump, load
from dacite import Config, from_dict
from dataclasses import dataclass, field, asdict

from open_video_summary.utils import log
from open_video_summary.entities.summary import SummaryLog
from open_video_summary.entities.video import Video, VideoSegment
from open_video_summary.utils.paths import project_path, video_paths
from open_video_summary.utils.audit import json_values, new_summary_audit, segment_record


@dataclass
class SummarySegmentHandler:
    __source_videos: list[Video] = field(default_factory=list)
    __output: list[VideoSegment] = field(default_factory=list)
    __to_discard: set[VideoSegment] = field(default_factory=set)
    __to_include: set[VideoSegment] = field(default_factory=set)
    __to_pick: list[set[VideoSegment]] = field(default_factory=list)
    __agent_log: dict[str, SummaryLog] = field(default_factory=dict)
    __audit: dict = field(default_factory=new_summary_audit)
    audit_enabled: bool = True

    @property
    def audit(self) -> dict:
        # Old handler files have no audit field. Build their manifest lazily
        # when a caller starts collecting new evidence from the restored source.
        if self.audit_enabled and self.source and not self.__audit["sources"]:
            self._initialize_audit_manifest()
        return self.__audit

    def _initialize_audit_manifest(self) -> None:
        for video_index, video in enumerate(self.source):
            self.__audit["sources"].append({
                "video_id": f"v{video_index}",
                "video_index": video_index,
                "name": video.name,
                "path": video_paths({"path": video.path})["path"],
            })
            self.__audit["segments"].extend(
                segment_record(
                    segment, f"v{video_index}:s{segment_index}",
                    video_id=f"v{video_index}", video_index=video_index,
                    segment_index=segment_index,
                )
                for segment_index, segment in enumerate(video.segments)
            )

    def segment_id(self, segment: VideoSegment) -> str:
        """Use input positions, preserving identity even for equal source items."""
        for video_index, video in enumerate(self.source):
            for segment_index, source_segment in enumerate(video.segments):
                if segment is source_segment:
                    return f"v{video_index}:s{segment_index}"
        for video_index, video in enumerate(self.source):
            for segment_index, source_segment in enumerate(video.segments):
                if segment == source_segment:
                    return f"v{video_index}:s{segment_index}"
        # Criteria can also be used directly without source videos. These IDs
        # describe their actual first-request order within this handler.
        external = getattr(self, "_audit_external_segments", [])
        for index, item in enumerate(external):
            if segment == item:
                return f"external:s{index}"
        segment_id = f"external:s{len(external)}"
        external.append(segment)
        self._audit_external_segments = external
        if self.audit_enabled:
            self.__audit["segments"].append(segment_record(segment, segment_id))
        return segment_id

    def segment_eligibility(self, segment: VideoSegment) -> dict:
        return {
            "eligible": segment not in self.discard and segment not in self.output,
            "already_discarded": segment in self.discard,
            "already_output": segment in self.output,
            "discarded_by": [
                name for name, records in self.agent_logs.items() if segment in records.discard
            ],
            "output_by": [
                name for name, records in self.agent_logs.items() if segment in records.output
            ],
        }

    @property
    def source(self) -> list[Video]:
        return self.__source_videos

    @property
    def output(self) -> list[VideoSegment]:
        return self.__output

    @property
    def include(self) -> set[VideoSegment]:
        return self.__to_include

    @property
    def discard(self) -> set[VideoSegment]:
        return self.__to_discard

    @property
    def pick(self) -> list[set[VideoSegment]]:
        return self.__to_pick

    @property
    def agent_logs(self) -> dict[str, SummaryLog]:
        return self.__agent_log

    def __log_agent_action(
        self,
        action: str,
        agent: str,
        segment: VideoSegment | list[VideoSegment] | set[VideoSegment],
    ) -> None:
        if agent not in self.__agent_log:
            self.__agent_log[agent] = SummaryLog()
        action_item = getattr(self.__agent_log[agent], action)
        action_item.append(segment)

    def set_source_videos(self, videos: list[Video]) -> None:
        if self.__source_videos:
            error_msg = "Source Videos cannot change once they are set."
            log.error(error_msg)
            raise ValueError(error_msg)

        self.__source_videos = videos
        if self.audit_enabled:
            self._initialize_audit_manifest()
        log.info("Source videos set.")

    def add_output_segment(self, segment: VideoSegment, agent: str) -> None:
        if segment in self.__output:
            log.info("Segment is already in output.")
            return
        self.__output.append(segment)
        self.__to_discard.discard(segment)
        self.__to_include.discard(segment)
        self.__log_agent_action("output", agent, segment)
        log.info("Added video segment to output.")

    def include_segment(self, segment: VideoSegment, agent: str) -> None:
        self.__to_discard.discard(segment)
        self.__to_include.add(segment)
        self.__log_agent_action("include", agent, segment)
        log.info("Added video segment to 'include' set.")

    def discard_segment(self, segment: VideoSegment, agent: str) -> None:
        if segment in self.__output:
            log.info("Can't discard segment already in output.")
            return
        self.__to_include.discard(segment)
        self.__to_discard.add(segment)
        self.__log_agent_action("discard", agent, segment)
        log.info("Added video segment to 'discard' set.")

    def add_segments_to_pick(self, segments: set[VideoSegment], agent: str) -> None:
        self.__to_pick.append(segments)
        self.__log_agent_action("pick", agent, segments)
        log.info("Added video segments to 'pick' set.")

    def output_included_segments(self) -> None:
        log.info("Moving all 'include' segments to output.")
        for segment in self.__to_include:
            self.__output.append(segment)
        self.__to_include.clear()

    def finalize(self) -> None:
        log.info("Finalizing SummarySegmentHandler.")
        if not self.__output:
            self.output_included_segments()


class SummarySegmentHandlerIO:
    @staticmethod
    def save(handler: SummarySegmentHandler, filepath: str) -> None:
        log.info("Saving SummarySegmentHandler to disk file.")

        output_path = project_path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            dump(
                json_values(video_paths(asdict(handler))), file,
                ensure_ascii=False, indent=4, allow_nan=False,
            )

    @staticmethod
    def load(filepath: str) -> SummarySegmentHandler:
        log.info("Loading SummarySegmentHandler from disk file.")
        with project_path(filepath).open(encoding="utf-8") as file:
            return from_dict(
                SummarySegmentHandler,
                video_paths(load(file), resolve=True),
                config=Config(cast=[set]),
            )
