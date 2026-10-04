"""Immutable source snapshot and information report; no selection actions."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, is_dataclass

from open_video_summary.errors import ConfigurationError
from open_video_summary.utils.paths import portable_path


def canonical_json(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def fingerprint(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SnapshotSegment:
    id: str
    source_identity: str
    video_id: str
    video_index: int
    segment_index: int
    content: str
    start: float
    end: float
    order: int | None
    video_topic: str
    global_topic: str
    video_path: str


@dataclass(frozen=True)
class SnapshotVideo:
    id: str
    name: str
    path: str
    topics: tuple[str, ...]
    segments: tuple[SnapshotSegment, ...]


@dataclass(frozen=True)
class AnalysisSnapshot:
    """Provenance remains available; semantic context is limited to current input.

    This is a data contract, not a plugin/hook framework. A future caller must
    explicitly construct its current stage input and cannot reintroduce removed
    source segments as semantic context merely because provenance retains them.
    """

    stage_id: str
    source_fingerprint: str
    current_input_fingerprint: str
    source: tuple[SnapshotVideo, ...]
    source_order: tuple[str, ...]
    current_order: tuple[str, ...]
    allowed_context_ids: tuple[str, ...]
    selection_stage_order: tuple[str, ...]

    @property
    def source_segments(self) -> tuple[SnapshotSegment, ...]:
        return tuple(segment for video in self.source for segment in video.segments)

    @property
    def current_segments(self) -> tuple[SnapshotSegment, ...]:
        by_id = {segment.id: segment for segment in self.source_segments}
        return tuple(by_id[identifier] for identifier in self.current_order)


def capture_snapshot(
    videos,
    *,
    stage_id="before_introduction",
    selection_stage_order=(),
    current_order=None,
    allowed_context_ids=None,
):
    """Copy primitives by position, preserving equal-valued source occurrences."""
    source = []
    for video_index, video in enumerate(videos):
        if not isinstance(video.name, str) or not all(
            isinstance(topic, str) for topic in video.topics
        ):
            raise ConfigurationError(
                "Information source names and topics must be strings."
            )
        segments = []
        for segment_index, segment in enumerate(video.segments):
            if (
                not isinstance(segment.content, str)
                or type(segment.start) not in (int, float)
                or type(segment.end) not in (int, float)
                or not math.isfinite(segment.start)
                or not math.isfinite(segment.end)
                or not 0 <= segment.start <= segment.end
                or (segment.order is not None and type(segment.order) is not int)
                or not isinstance(segment.video_topic, str)
                or not isinstance(segment.global_topic, str)
            ):
                raise ConfigurationError(
                    "Information analysis requires text and valid segment timestamps."
                )
            segments.append(
                {
                    "id": f"v{video_index}:s{segment_index}",
                    "video_id": f"v{video_index}",
                    "video_index": video_index,
                    "segment_index": segment_index,
                    "content": segment.content,
                    "start": segment.start,
                    "end": segment.end,
                    "order": segment.order,
                    "video_topic": segment.video_topic,
                    "global_topic": segment.global_topic,
                    "video_path": (
                        portable_path(segment.video_path) if segment.video_path else ""
                    ),
                }
            )
        source.append(
            {
                "id": f"v{video_index}",
                "name": video.name,
                "path": portable_path(video.path) if video.path else "",
                "topics": tuple(video.topics),
                "segments": segments,
            }
        )
    source_hash = fingerprint(source)
    immutable_source = tuple(
        SnapshotVideo(
            video["id"],
            video["name"],
            video["path"],
            video["topics"],
            tuple(
                SnapshotSegment(
                    source_identity=f"{source_hash}:{segment['id']}", **segment
                )
                for segment in video["segments"]
            ),
        )
        for video in source
    )
    source_order = tuple(
        segment.id for video in immutable_source for segment in video.segments
    )
    current = source_order if current_order is None else tuple(current_order)
    allowed = current if allowed_context_ids is None else tuple(allowed_context_ids)
    if len(set(current)) != len(current) or not set(current) <= set(source_order):
        raise ConfigurationError(
            "The current information input has invalid positional identifiers."
        )
    if len(set(allowed)) != len(allowed) or not set(allowed) <= set(current):
        raise ConfigurationError(
            "Semantic context must belong to the current stage input."
        )
    if not set(current) <= set(allowed):
        raise ConfigurationError(
            "Every current target must be permitted as semantic context."
        )
    by_id = {
        segment.id: asdict(segment)
        for video in immutable_source
        for segment in video.segments
    }
    current_hash = fingerprint(
        {
            "stage_id": stage_id,
            "order": current,
            "segments": [by_id[item] for item in current],
            "allowed_context_ids": allowed,
        }
    )
    return AnalysisSnapshot(
        stage_id,
        source_hash,
        current_hash,
        immutable_source,
        source_order,
        current,
        allowed,
        tuple(selection_stage_order),
    )


@dataclass(frozen=True)
class Qualifiers:
    attribution: str | None
    negated: bool
    modality: str | None
    quantities: tuple[str, ...]
    conditions: tuple[str, ...]


@dataclass(frozen=True)
class Evidence:
    segment_id: str
    source_identity: str
    quote: str
    start_char: int
    end_char: int
    role: str
    segment_start: float
    segment_end: float
    timestamp_resolution: str = "segment"


@dataclass(frozen=True)
class InformationCandidate:
    text: str
    unit_type: str
    qualifiers: Qualifiers
    evidence: tuple[Evidence, ...]
    question: str | None
    answer: str | None
    unresolved_references: tuple[str, ...]


@dataclass(frozen=True)
class CandidateRecord:
    id: str
    target_segment_id: str
    route: str
    round: int
    candidate: InformationCandidate | None
    raw_json: str
    validation: str
    reasons: tuple[str, ...] = ()
    signals: tuple[tuple[str, float], ...] = ()
    granularity: str | None = None
    granularity_probability: float | None = None
    granularity_confidence: float | None = None


@dataclass(frozen=True)
class InformationUnit:
    id: str
    text: str
    unit_type: str
    qualifiers: Qualifiers
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class InformationOccurrence:
    id: str
    unit_id: str
    segment_id: str
    source_identity: str
    candidate_ids: tuple[str, ...]
    assertion_evidence: tuple[Evidence, ...]
    context_evidence: tuple[Evidence, ...]
    routes: tuple[str, ...]


@dataclass(frozen=True)
class InformationRelation:
    left_candidate_id: str
    right_candidate_id: str
    relation: str
    probability: float | None
    confidence: float | None
    merged: bool = False
    origin: str = "evaluator"


@dataclass(frozen=True)
class CoverageRecord:
    segment_id: str
    round: int
    represented_candidate_ids: tuple[str, ...]
    focus_markers: tuple[tuple[str, int, int], ...]
    signals: tuple[tuple[str, float], ...]
    state: str


@dataclass(frozen=True)
class AnalysisIssue:
    kind: str
    detail: str
    segment_ids: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScopeCount:
    id: str
    occurrences: int
    unique_units: int


@dataclass(frozen=True)
class InformationCounts:
    candidates: int
    accepted_candidates: int
    occurrences: int
    unique_units: int
    by_segment: tuple[ScopeCount, ...]
    by_video: tuple[ScopeCount, ...]
    valid_zero: bool


@dataclass(frozen=True)
class AnalysisCall:
    operation: str
    provider: str
    input_hash: str
    output_hash: str | None
    status: str
    metadata_json: str
    decisions_json: str | None = None


@dataclass(frozen=True)
class InformationReport:
    schema_version: int
    protocol_version: str
    run_id: str
    status: str
    snapshot: AnalysisSnapshot
    candidates: tuple[CandidateRecord, ...]
    units: tuple[InformationUnit, ...]
    occurrences: tuple[InformationOccurrence, ...]
    relations: tuple[InformationRelation, ...]
    coverage: tuple[CoverageRecord, ...]
    counts: InformationCounts
    issues: tuple[AnalysisIssue, ...]
    calls: tuple[AnalysisCall, ...]
    metadata_json: str

    def to_dict(self) -> dict:
        def export(value):
            if is_dataclass(value):
                result = {}
                for item in fields(value):
                    field_value = getattr(value, item.name)
                    if item.name == "metadata_json":
                        result["metadata"] = json.loads(field_value)
                    elif item.name == "raw_json":
                        result["raw"] = json.loads(field_value)
                    elif item.name == "decisions_json":
                        result["decisions"] = (
                            json.loads(field_value) if field_value is not None else None
                        )
                    else:
                        result[item.name] = export(field_value)
                return result
            if isinstance(value, tuple):
                return [export(item) for item in value]
            return value

        return export(self)
