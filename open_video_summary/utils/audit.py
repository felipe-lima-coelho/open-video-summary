"""Small, portable audit records built from the existing selection results."""

import json
from math import isfinite
from numbers import Integral, Real

from open_video_summary.utils.paths import portable_path, project_path, video_paths


def new_summary_audit() -> dict:
    return {
        "schema_version": 1,
        "status": "in_progress",
        "sources": [],
        "segments": [],
        "criteria": {},
        "outcome": {},
    }


def segment_record(segment, segment_id, **indices) -> dict:
    return {
        "segment_id": segment_id,
        **indices,
        "video_path": portable_path(segment.video_path) if segment.video_path else "",
        "order": segment.order,
        "start": segment.start,
        "end": segment.end,
        "video_topic": segment.video_topic,
        "global_topic": segment.global_topic,
    }


def finite_number(value):
    value = float(value)
    return value if isfinite(value) else None


def json_values(value):
    """Represent unavailable numeric values as null, never JSON NaN/Infinity."""
    if isinstance(value, dict):
        return {str(key): json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_values(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return finite_number(value)
    return value


def finalize_summary_audit(handler, *, status="completed", error=None) -> None:
    if not handler.audit_enabled:
        return
    audit = handler.audit
    actions = {}
    for criterion, records in handler.agent_logs.items():
        for action in ("output", "include", "discard"):
            for segment in getattr(records, action):
                actions.setdefault(handler.segment_id(segment), []).append(
                    {"criterion": criterion, "action": action}
                )
    output_ids = [handler.segment_id(segment) for segment in handler.output]
    included_ids = {handler.segment_id(segment) for segment in handler.include}
    discarded_ids = {handler.segment_id(segment) for segment in handler.discard}
    redundancy = audit["criteria"].get("ContentBasedRedundancy", {})
    visual = audit["criteria"].get("QualityPick", {})
    clustered = {
        segment_id
        for cluster in redundancy.get("clusters", [])
        for segment_id in cluster["segment_ids"]
    }
    candidates = {}
    for cluster in visual.get("clusters", []):
        for candidate in cluster["candidates"]:
            candidates.setdefault(candidate["segment_id"], []).append(
                (cluster["cluster_id"], candidate)
            )

    outcomes = []
    for segment in audit["segments"]:
        segment_id = segment["segment_id"]
        positions = [index for index, item in enumerate(output_ids) if item == segment_id]
        state = (
            "output" if positions else "discarded" if segment_id in discarded_ids
            else "included" if segment_id in included_ids else "not_selected"
        )
        reasons = []
        if not positions:
            reasons.extend(
                {"criterion": action["criterion"], "reason": "recorded_discard"}
                for action in actions.get(segment_id, [])
                if action["action"] == "discard"
            )
            if redundancy and segment_id not in clustered:
                reasons.append(
                    {"criterion": "ContentBasedRedundancy", "reason": "not_in_any_cluster"}
                )
            for cluster_id, candidate in candidates.get(segment_id, []):
                if not candidate.get("chosen"):
                    reasons.append({
                        "criterion": "QualityPick",
                        "cluster_id": cluster_id,
                        "reason": candidate.get("exclusion_reason") or "not_chosen_by_rank",
                    })
            if any(candidate.get("chosen") for _, candidate in candidates.get(segment_id, [])):
                reasons.append({
                    "criterion": "QualityPick",
                    "reason": "quality_choice_not_in_final_output",
                })
        outcomes.append({
            "segment_id": segment_id,
            "final_state": state,
            "output_positions": positions,
            "recorded_actions": actions.get(segment_id, []),
            "exclusion_reasons": reasons,
        })
    audit["status"] = status
    audit["outcome"] = {"output_segment_ids": output_ids, "segments": outcomes}
    if error is not None:
        audit["error_type"] = type(error).__name__


def save_summary_audit(audit: dict, filepath: str) -> None:
    output = project_path(filepath)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_values(video_paths(audit)), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
