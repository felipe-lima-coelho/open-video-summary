"""Portable run provenance and exact inventory diagnostics, never semantic proofs."""

import hashlib
import importlib.metadata
import platform
from itertools import combinations
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from open_video_summary.core.summarizers.information_contracts import canonical_json, fingerprint


def _endpoint(adapter):
    value = getattr(getattr(adapter, "config", None), "base_url", None)
    if not value:
        return None
    parsed = urlsplit(value)
    # User info, fragments and query strings can contain credentials.
    host = parsed.hostname or ""
    if parsed.port:
        host += f":{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def analysis_manifest(snapshot, config, templates, generator, evaluator, provider_settings):
    """Hash source/code/settings without paths, secrets, clocks or run IDs."""
    package = Path(__file__).resolve().parents[2]
    paths = [*sorted((package / "core" / "summarizers").glob("information_*.py")),
             *(package / "adapters" / name for name in ("llm.py", "information_llm.py", "typesafe.py", "information_schema.py")),
             package / "contracts.py", package / "utils" / "providers.py"]
    source_files = {path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in paths}
    dependencies = {}
    for name in ("openai", "httpx", "pydantic"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    manifest = {
        "version": 2,
        "source_fingerprint": snapshot.source_fingerprint,
        "current_input_fingerprint": snapshot.current_input_fingerprint,
        "settings": asdict(config),
        "effective_execution": {"target_workers": config.target_workers,
            "pair_workers": config.pair_workers, "pair_batch_size": config.pair_batch_items},
        "templates": templates,
        "code_hash": fingerprint(source_files),
        "source_files": source_files,
        "runtime": {"python": platform.python_version(), "platform": platform.system(),
                    "machine": platform.machine(), "dependencies": dependencies},
        "generator": {**provider_settings(generator), "endpoint": _endpoint(generator)},
        "evaluator": {**provider_settings(evaluator), "endpoint": _endpoint(evaluator)},
        "sampling": (generator.sampling_settings() if callable(getattr(generator, "sampling_settings", None))
            else {"requested_temperature": 0.0, "seed": None,
                "note": "Ollama sends temperature. The general OpenAI Responses adapter omits it "
                        "and does not expose a seed. Serial execution controls order, not provider determinism."}),
    }
    return {"analysis_signature": fingerprint(manifest), "manifest": manifest,
            "guarantee": "Matching manifests identify comparable settings, not identical fresh provider outputs."}


def inventory_payload(units, occurrences, candidates):
    """Retain exact Portuguese meanings and anchors while discarding positional IDs."""
    records = {record.id: record for record in candidates}
    rows = []
    for unit in units:
        claims = sorted({records[identifier].candidate.text for identifier in unit.candidate_ids})
        anchors = []
        for occurrence in occurrences:
            if occurrence.unit_id != unit.id:
                continue
            anchors.append({"segment_id": occurrence.segment_id, "source_identity": occurrence.source_identity,
                "assertion_evidence": [asdict(item) for item in occurrence.assertion_evidence],
                "context_evidence": [asdict(item) for item in occurrence.context_evidence]})
        rows.append({"text": unit.text, "unit_type": unit.unit_type, "claims": claims,
                     "qualifiers": asdict(unit.qualifiers) if unit.qualifiers is not None else None,
                     "qualifier_state": unit.qualifier_state,
                     "anchors": sorted(anchors, key=canonical_json)})
    return sorted(rows, key=canonical_json)


def reliability_diagnostics(issues, candidates, relations, coverage_foci, unexamined):
    kinds = {item.kind for item in issues}
    budget_kinds = sorted(kind for kind in kinds if "budget" in kind or "limit" in kind)
    return {
        "budget_limits": budget_kinds,
        "provider_failures": sorted(kind for kind in kinds if "failed" in kind or kind == "provider_stopped"),
        "unreviewed_atomicity_candidates": sum(item.validation == "accepted" and item.inventory_role != "decomposed_parent"
            and item.atomicity_state == "not_reviewed" for item in candidates),
        "uncertain_atomicity_candidates": sum(item.validation == "accepted" and item.inventory_role != "decomposed_parent"
            and item.atomicity_state in {"compound", "uncertain"} for item in candidates),
        "unconfirmed_equivalence_pairs": sum(item.equivalence_origin == "confirmation_pending" for item in relations),
        "uncertain_equivalence_pairs": sum(item.equivalence_state == "uncertain" for item in relations),
        "unexamined_pairs": unexamined,
        "unresolved_source_foci": sum(item.state != "covered" for item in coverage_foci),
    }


def compare_information_reports(reports):
    """Compare exact words and anchors; these overlaps are not semantic accuracy."""
    rows, inventories = [], []
    for report in reports:
        units = {canonical_json({"text": item["text"], "unit_type": item["unit_type"],
                                  "qualifiers": item.get("qualifiers")}) for item in report["units"]}
        anchors = {canonical_json({key: evidence[key] for key in
                                   ("source_identity", "segment_id", "start_char", "end_char", "quote")})
                   for item in report["occurrences"] for evidence in item["assertion_evidence"]}
        metadata = report.get("metadata", {})
        rows.append({"run_id": report.get("run_id"), "status": report["status"],
            "counts_provisional": report["counts"].get("counts_provisional", report["status"] != "completed"),
            "unique_units": report["counts"]["unique_units"], "occurrences": report["counts"]["occurrences"],
            "analysis_signature": metadata.get("reproducibility", {}).get("analysis_signature"),
            "content_digest": metadata.get("content_digest"),
            "reliability_diagnostics": metadata.get("reliability_diagnostics", {})})
        inventories.append((units, anchors))
    pairs = []
    for left, right in combinations(range(len(rows)), 2):
        result = {"left": left, "right": right}
        for index, name in enumerate(("exact_unit_wording", "exact_assertion_anchor")):
            a, b = inventories[left][index], inventories[right][index]
            result[name] = {"intersection": len(a & b), "union": len(a | b),
                "jaccard": len(a & b) / len(a | b) if a | b else 1.0,
                "only_left": sorted(a - b), "only_right": sorted(b - a)}
        pairs.append(result)
    signatures = [item["analysis_signature"] for item in rows]
    return {"runs": rows, "pairs": pairs,
            "matching_manifests": bool(signatures) and all(signatures) and len(set(signatures)) == 1,
            "interpretation": "Exact wording and anchor overlap expose inventory changes even when counts agree. They do not measure semantic equivalence, accuracy, coverage or completeness; partial counts remain provisional."}
