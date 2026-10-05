"""Atomic, non-overwriting local report exports under the outputs directory."""

import csv
import io
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from open_video_summary.errors import ConfigurationError
from open_video_summary.utils.paths import project_path


def validate_information_destination(path, *, reserved=(), suffix=".json") -> Path:
    destination = project_path(path).resolve()
    outputs = project_path("outputs").resolve()
    if not destination.is_relative_to(outputs) or destination.suffix.lower() != suffix:
        raise ConfigurationError(
            f"Information reports must be {suffix} files under outputs/."
        )
    for item in reserved:
        if not item:
            continue
        protected = project_path(item).resolve()
        if destination == protected or (
            destination.exists()
            and protected.exists()
            and os.path.samefile(destination, protected)
        ):
            raise ConfigurationError(
                "The information report aliases a protected input or summary artifact."
            )
    if destination.exists():
        raise ConfigurationError(
            "The information destination already exists; choose a new report path."
        )
    return destination


def _atomic_create(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=".information-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-directory hard link publishes the complete file atomically and
        # fails if another writer created the destination. replace() would clobber.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def information_csv(report) -> str:
    stream = io.StringIO(newline="")
    names = [
        "status",
        "unit_id",
        "text",
        "unit_type",
        "qualifier_state",
        "qualifiers",
        "representative_candidate_id",
        "occurrence_id",
        "alignment_state",
        "segment_id",
        "source_identity",
        "segment_start",
        "segment_end",
        "assertion_quotes",
        "context_quotes",
    ]
    writer = csv.DictWriter(stream, fieldnames=names)
    writer.writeheader()
    by_unit = {unit.id: unit for unit in report.units}
    for occurrence in report.occurrences:
        unit = by_unit[occurrence.unit_id]
        evidence = occurrence.assertion_evidence[0]
        writer.writerow(
            {
                "status": report.status,
                "unit_id": unit.id,
                "text": unit.text,
                "unit_type": unit.unit_type,
                "qualifier_state": unit.qualifier_state,
                "qualifiers": json.dumps(
                    asdict(unit.qualifiers) if unit.qualifiers is not None else None,
                    ensure_ascii=False,
                ),
                "representative_candidate_id": unit.representative_candidate_id,
                "occurrence_id": occurrence.id,
                "alignment_state": occurrence.alignment_state,
                "segment_id": occurrence.segment_id,
                "source_identity": occurrence.source_identity,
                "segment_start": evidence.segment_start,
                "segment_end": evidence.segment_end,
                "assertion_quotes": json.dumps(
                    [item.quote for item in occurrence.assertion_evidence],
                    ensure_ascii=False,
                ),
                "context_quotes": json.dumps(
                    [item.quote for item in occurrence.context_evidence],
                    ensure_ascii=False,
                ),
            }
        )
    if not report.occurrences:
        writer.writerow({"status": report.status})
    return stream.getvalue()


def save_information_report(report, path, *, reserved=(), csv_path=None) -> Path:
    destination = validate_information_destination(path, reserved=reserved)
    csv_destination = None
    if csv_path:
        csv_destination = validate_information_destination(
            csv_path, reserved=tuple(reserved) + (destination,), suffix=".csv"
        )
    payload = (
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    )
    # Validate both destinations before publishing either export.
    if csv_destination is not None:
        _atomic_create(csv_destination, information_csv(report))
    _atomic_create(destination, payload)
    return destination
