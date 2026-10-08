# Offline information reference evaluation

The evaluator compares a frozen analysis report with a versioned inventory of
contextual propositions. It runs without API keys, provider imports, model
weights, network calls, or new dependencies. It measures agreement with the
annotated reference; an agent annotation is not human gold or scientific proof
of extraction accuracy.

```powershell
.\.venv\Scripts\python.exe -m open_video_summary.evaluation `
  --report outputs/information/run.json `
  --reference data/evaluation/information/reference.json `
  --alignment outputs/evaluation/reviewed_alignment.json `
  --output outputs/evaluation/run_evaluation.json
```

`--alignment` is optional. Relative paths resolve from the repository root.
Outputs must be new JSON files under ignored `outputs/`; existing artifacts are
never overwritten. The command validates source fingerprints, transcript text,
source identities, timestamps, reference scope, protocol and literal evidence.
It records canonical content fingerprints and byte hashes for loaded files.

The default requires identical protocol IDs. Use `--allow-protocol-mismatch`
only to inspect a legacy report against a newer protocol. Such a result is
explicitly diagnostic and cannot enter the same-protocol ablation table.

## Reference contract

The reference envelope has `schema_version: 1`, `protocol`, `provenance`,
`snapshot`, and `units`. Extra annotation, ambiguity and gap fields are retained
in the reference fingerprint and can document decisions without adding counted
units.

`protocol` requires `id`, `version`, and an explicit `granularity` rule. Follow
the same rules for every mode: one complete contextual proposition per unit;
conditions remain attached to their consequences; attribution, polarity,
quantities and modality remain in the proposition; independent properties may
be separate; a decomposed parent and all its components cannot count as an
additional independent unit. Reference context resolves a proposition rather
than inventing an assertion. Corrections preserve the originally communicated
proposition and its occurrence.

`provenance` must state `annotator_type` and `reviewer_status`. The corpus also
records annotator identity, annotation date, source path, source byte hash,
scope, protocol, and unresolved cases. `snapshot` follows the report snapshot:
`source_fingerprint`, `current_order`, and `source` videos with positional source
segment IDs, immutable identities, original transcript content and timestamps.
The reference scope must be contained in the report's current input.

Each reference unit has a stable `id`, complete `text`, optional `qualifiers`,
and `occurrences`. Each occurrence has a globally unique `id`, `segment_id`,
`assertion_evidence`, and optional `context_evidence`. Evidence stores a literal
`quote` with `[start_char, end_char)` Unicode character offsets; context anchors
must name their own segment when it differs. Assertion anchors belong to the
occurrence's segment. The evaluator rejects nonliteral reference evidence.
Explicit `scorable: false` on a unit or occurrence keeps the record and its
`exclusion_reason` visible while removing it from coverage denominators. A unit
exclusion also excludes its occurrences. Non-scorable identity/antecedent
components may be documented in annotations without excluding a generic
proposition that remains scorable; review must honor that stated component scope.

## Reviewed alignment contract

An alignment is separate from the frozen reference and must bind the exact
report and reference content. Create the envelope with the public fingerprint
function:

```python
from open_video_summary.evaluation import (
    evaluate_information_report, evaluation_fingerprint,
)

alignment = {
    "schema_version": 1,
    "report_fingerprint": evaluation_fingerprint(report),
    "reference_fingerprint": evaluation_fingerprint(reference),
    "provenance": {
        "annotator_type": "agent",  # Use "human" only for an actual human review.
        "reviewer_status": "reviewed",
        "reviewer": "independent reviewer identity",
        "model_generated": False,
    },
    "unit_links": [{
        "reference_unit_ids": ["R1"], "report_unit_ids": ["u0"],
        "relation": "equivalent", "review_status": "verified",
        "reason": "Complete meaning and its source scope were independently inspected.",
    }],
}
result = evaluate_information_report(report, reference, alignment)
```

A provider's own support/equivalence output cannot validate itself. Unreviewed
model-generated mappings remain unverified even if they carry a proposed
`verified` label. Independent review must be explicitly recorded with
`independently_reviewed: true` and a reviewer identity. This is an auditable
provenance declaration, not an automatic guarantee of reviewer independence.

Every reviewed row needs `review_status: "verified"`. Other rows remain visible
without contributing to measured matches or errors. Missing mappings are
unresolved, not omissions or false positives.

| Alignment field | Meaning |
| --- | --- |
| `unit_links` | `reference_unit_ids`, `report_unit_ids`, and a relation. `equivalent` is one complete meaning to one unit; `split` is one reference unnecessarily separated into several report units; `merged` is several reference units combined into one report unit; `partial` preserves only part of a reference; `unresolved` makes no verified claim. |
| `compound_plus_components` relation | Requires at least two reference units, `parent_report_unit_ids`, `component_report_unit_ids`, and `decomposition_verified: true`. Parents and components must be disjoint, present among counted report units, and collectively equal `report_unit_ids`. Only a verified decomposition counts the parent as extra. |
| `occurrence_links` | `reference_occurrence_ids`, `report_occurrence_ids`, relation `equivalent`, `split_anchor`, `merged_occurrences`, or `unresolved`. Each complete match needs a complete unit alignment and overlapping assertion anchors in the same source utterance. |
| `reference_decisions` | `reference_unit_id`, state `omitted`, `partial`, or `unresolved`, and a reason. `omitted` and `partial` mean exhaustive inspection of the entire current report for that reference found no full match. A lexical search miss is insufficient. |
| `report_decisions` | `report_unit_id`, state `unsupported`, `partial`, `duplicate`, `supported_unreferenced`, or `unresolved`, and a reason. A duplicate names `duplicate_of`; a supported unit missing from the reference is not automatically unsupported. |
| `reference_occurrence_decisions` | `reference_occurrence_id` and state `omitted`, `partial`, or `unresolved`, under the same exhaustive-review rule. |
| `report_occurrence_decisions` | `report_occurrence_id` and state `unsupported`, `duplicate`, or `unresolved`. An invalid literal citation is reported separately from a semantic false positive. |
| `qualifier_errors` on a unit link | Entries with `field` (`conditions`, `negated`, `quantities`, `modality`, `attribution`), `kind` (`content` or `annotation`), and `detail`. A content error requires a partial alignment, not complete equivalence. Unknown canonical qualifiers are not automatically annotation errors. |

The only automatic semantic shortcut is **identical complete proposition text
plus identical valid assertion anchors**, with required reference context also
cited. Normalization changes Unicode composition and whitespace only; it does
not remove accents, punctuation, numbers, negation or modality. Identical text
must identify exactly one reference unit. A paraphrase or broader/narrower
assertion span needs review. Citation overlap alone cannot recover a different
proposition. Repeated assertions at different character spans remain separate
occurrences, even inside one segment. Repeated discovery routes at the same
verified assertion can be diagnosed as duplicate occurrences.

## Metrics and interpretation

Unit coverage and occurrence coverage have separate denominators. `covered`
includes complete verified alignments; `confirmed_omitted` and
`confirmed_partial` require explicit exhaustive review; everything else remains
`unresolved`. The lower bound is covered / reference total; the upper bound
adds unresolved references as possible matches. Resolved-only recall is also
reported, with the reviewed fraction, so a tiny reviewed subset cannot look
like complete coverage. Empty denominators yield `null`.

Source fidelity reports verified supported units, confirmed unsupported units,
partial units and unresolved units. It is separate from canonical granularity:
a source-supported compound or duplicate can still produce the wrong inventory
count. Occurrence fidelity separately reports source matches and invalid
citations. Unmatched output is never silently counted as a false positive.
A reviewed duplicate declaration alone does not prove source fidelity: two
unsupported outputs may duplicate one another. The claim needs a supported
reference alignment independently of the count diagnostic.

Consolidation records verified duplicate units/occurrences, unnecessary splits,
improper merges, and compound parents counted alongside their components. The
per-reference and per-output records preserve each mapping and review note.
The integrity audit compares stored aggregate/scope counts with actual report
records; it does not trust a reported total as ground truth. Partial-run flags,
uncertain equivalence decisions, unresolved decompositions and coverage foci
remain visible alongside reference metrics.

## Criterion labels and ablations

Optional alignment `candidate_labels` may independently label accepted and
rejected controls:

```json
{
  "candidate_id": "c0",
  "expected_validation": "accepted",
  "expected_granularity": "atomic",
  "review_status": "verified",
  "selection_basis": "source-based control; not a random sample",
  "gates": {"support": true, "conditions": true, "attribution": null}
}
```

The tables compare expected validation/granularity with recorded outcomes.
`expected_validation` can be `accepted`, `rejected`, or `uncertain`; the report's
more specific states such as `needs_review` remain explicit matrix columns.
Each labeled binary gate reports TP/FP/TN/FN at the **recorded** acceptance
threshold, reliability bins, a Brier score and a descriptive sample calibration
error. Missing signals/thresholds and null labels remain unscored. Record
`sampling`, particularly for targeted controls. These diagnostics do not
calibrate a threshold or establish population reliability from a selected,
small or agent-annotated sample. No test-data threshold tuning occurs.

Metadata preserves `direct`, `qa`, `hybrid`, and `hybrid_coverage` modes
(`hybrid_recovery` is a consumer alias), generation/evaluator identity, settings,
prompt hashes, template versions, elapsed time, calls, attempts and reported
tokens. A legacy inferred mode is explicitly marked. Missing token usage is
not treated as zero cost; monetary cost is left unspecified.

```powershell
.\.venv\Scripts\python.exe -m open_video_summary.evaluation `
  --compare outputs/evaluation/direct.json outputs/evaluation/qa.json `
            outputs/evaluation/hybrid.json outputs/evaluation/recovery.json `
  --output outputs/evaluation/ablation_comparison.json
```

The comparison requires the same reference fingerprint, source, segment scope,
and protocol. It preserves review coverage and partial-run status rather than
ranking modes automatically. Changing the evaluator provider can be tabulated
with the same contract. A single baseline diagnosis or deterministic regression
is not an empirical ablation study.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_information_reference_evaluation -v
```

These regressions use synthetic Portuguese controls with sockets disabled.
They test actual bad counts, repeated/overlapping anchors, unsupported claims,
reviewed omissions, split/merge and verified decomposition errors, explicit
qualifier labels, unverified model mappings, gate FP/FN diagnostics, protocol
isolation, fingerprints, safe exports and the standalone command.

## Frozen news baseline diagnosis

The source-only news reference was frozen at commit `0d6bb72` before a separate
agent inspected the legacy report. Its canonical content fingerprint is
`c61c63a9159968532124a99f7a367a8c6b5e858ffbeed027a8ed0be1434e6922`.
There are 53 annotation records and 56 occurrences; two ambiguous identity-bound
records are excluded, leaving 51 eligible units and 54 eligible occurrences.

The saved local alignment is
`outputs/evaluation/google_huawei_baseline_20261008_alignment.json`; its result
is `outputs/evaluation/google_huawei_baseline_20261008_evaluation.json`. These
ignored artifacts bind the original report fingerprint
`5e4937b9298b49fc39d4aa84583a4ca9aef7f66d11f66cdf8e709a7ca2992aeb`.
The report was **partial**, with 36 units and 37 occurrences, under protocol v1.
The reference uses v2, so this is explicitly a cross-protocol diagnosis.

Under this bounded, agent-reviewed alignment, 33/51 eligible units are covered,
17 have reviewed omissions and one remains unresolved: coverage bounds are
64.7–66.7%. Occurrences have 34/54 covered, 18 reviewed omissions and two
unresolved: bounds are 63.0–66.7%. Source support is verified for 34/36 output
units; two remain unresolved and no unsupported output unit is confirmed.
The count diagnosis identifies two extra units (one Android-ownership duplicate
and one decree parent counted with its components), four merged groups and
three extra source occurrences. The decree decomposition explicitly records
the agent's contextual emergency-rationale interpretation for `u27`; it is
available for independent review, not assumed human adjudication.

Ten deliberately selected source-based candidate controls produce diagnostic
gate confusion and reliability tables. They include rejected/repair cases and
known source-supported cases with low scores. This selected, agent-reviewed
sample is **not representative accuracy or calibration evidence**, and no
threshold was changed. The unit/occurrence omissions above concern the counted
inventory, even when an unaccepted candidate had discovered the content.

Evaluate a final v2/schema-7 report against the frozen reference with:

```powershell
.\.venv\Scripts\python.exe -m open_video_summary.evaluation `
  --report outputs/information/final_schema7_report.json `
  --reference data/evaluation/information/google_huawei_reference_v1.json `
  --output outputs/evaluation/final_schema7_initial.json
```

Create a **new** reviewed alignment bound to that final report and the same
reference fingerprint, using the API example above. Independently inspect
paraphrases, qualifier scope, granularity and each source occurrence. Save it as
a new artifact and rerun with `--alignment`; do not rewrite the source reference
to fit the output or reuse baseline links against a new report. The fingerprint
checks prevent that reuse. Add only verified matches and exhaustive reviewed
nonmatches, keeping uncertain cases explicit. The initial exact-only evaluation
will ordinarily have broad bounds until this independent review is complete.

The real core v2 is also exercised against the separate synthetic reference by
`tests/test_information_reference_core_integration.py`: scripted generation and
evaluation, sockets blocked, schema 7, five units and five occurrences matched.
This verifies report/harness compatibility and preserves the frozen reference;
it does not measure provider accuracy.
