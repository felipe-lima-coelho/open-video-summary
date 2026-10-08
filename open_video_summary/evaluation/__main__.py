"""Offline command for reference evaluation and same-reference ablation tables."""

import argparse
import json
import sys

from open_video_summary.utils.paths import project_path
from .information import (
    compare_information_evaluations,
    evaluate_information_files,
    save_information_evaluation,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate information inventories offline; no provider calls."
    )
    parser.add_argument("--report", help="Frozen information report JSON.")
    parser.add_argument("--reference", help="Versioned source-grounded reference JSON.")
    parser.add_argument(
        "--alignment", help="Optional fingerprint-bound reviewed alignment JSON."
    )
    parser.add_argument(
        "--allow-protocol-mismatch",
        action="store_true",
        help="Allow an explicitly diagnostic cross-protocol evaluation.",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="EVALUATION",
        help="Tabulate prior same-reference evaluations.",
    )
    parser.add_argument(
        "--output", required=True, help="New JSON artifact under outputs/."
    )
    args = parser.parse_args(argv)
    if args.compare:
        if (
            args.report
            or args.reference
            or args.alignment
            or args.allow_protocol_mismatch
        ):
            parser.error(
                "--compare cannot be combined with report/reference/alignment arguments."
            )
    elif not args.report or not args.reference:
        parser.error("--report and --reference are required for evaluation.")
    try:
        if args.compare:
            result = compare_information_evaluations(
                [
                    json.loads(project_path(path).read_text(encoding="utf-8-sig"))
                    for path in args.compare
                ]
            )
        else:
            result = evaluate_information_files(
                args.report,
                args.reference,
                args.alignment,
                allow_protocol_mismatch=args.allow_protocol_mismatch,
            )
        destination = save_information_evaluation(result, args.output)
    except (ValueError, OSError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2
    print(f"Saved offline evaluation: {destination}")
    if not args.compare:
        metrics = result["metrics"]
        for name in ("unit_coverage", "occurrence_coverage"):
            values = metrics[name]
            print(
                f"{name}: {values['covered']}/{values['reference_total']} verified; "
                f"{values['confirmed_omitted']} confirmed omitted; "
                f"{values['confirmed_partial']} partial; {values['unresolved']} unresolved"
            )
        fidelity = metrics["source_fidelity"]
        print(
            f"Source fidelity: {fidelity['verified_supported']}/{fidelity['report_total']} verified; "
            f"{fidelity['confirmed_false_positive']} confirmed unsupported; "
            f"{fidelity['confirmed_partial']} partial; {fidelity['unresolved']} unresolved"
        )
        if result["provenance"]["diagnostic_only"]:
            print(
                "Diagnostic reference-relative result; annotation/protocol limitations are recorded in provenance."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
