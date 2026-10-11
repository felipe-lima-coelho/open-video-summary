# Information inventory reliability

The information analyzer builds a source-grounded inventory before HSM summary
selection. Its report does not add or alter selection decisions. The analyzer
describes propositions communicated by the transcript; it does not establish
whether those propositions are true outside the source.

Repeatability, completeness and correctness are separate properties. Serial
execution removes worker scheduling as a source of within-run ordering changes.
It does not make a hosted or local generative service return the same answer on
every fresh request. A completed run means that configured operations finished
without recorded pending issues; it does not prove that the transcript was
exhaustively inventoried or that accepted meanings are correct.

## Repeatable run setup

Use the same source dataset and scope, analysis mode, evaluator and generator
models, thresholds, budgets, endpoint and code revision for each run. Set the
evaluator credential in the process environment or root `.env`, then give every
run a new report path:

```powershell
$env:OVS_EVALUATOR_API_KEY = "your-evaluator-key"
.\.venv\Scripts\python.exe -m open_video_summary analyze-information `
  --dataset data/processed/bebe_real.json `
  --output outputs/information/bebe_reliability_run01.json `
  --llm-provider ollama --llm-model gemma2 `
  --evaluator-provider typesafe --evaluator-model jev-1.13.0 `
  --information-reproducible
```

Repeat the same command in a fresh process, changing only the output filename.
`--information-reproducible` (or `OVS_INFORMATION_REPRODUCIBLE=true`) sets
effective source-target workers and pair workers to one, and uses one comparison
per pair request. It preserves configured call limits and thresholds. The report's
`metadata.effective_execution` records observed workers; the manifest records the
profile's effective configured execution. It does not set a provider seed or
replay/cache answers across runs.

When information analysis selects OpenAI `gpt-6-luna` (including an exact dated
snapshot), its dedicated generator always sends `seed=42`, `temperature=0`, and
`reasoning_effort="none"`. This module overrides a different general LLM effort
for that model. It uses Chat Completions with the same strict output schemas:
the Responses API has no supported seed field. Other OpenAI models and other
pipeline modules retain their existing adapter and configuration; Ollama is
unchanged. Unsupported sampling settings stop the run without dropping a
parameter or falling back to a different API. The fixed profile is independent
of the serial execution option above.

For this profile, each provider attempt records the requested and sent seed,
sent temperature and effort, API endpoint, returned model and
`system_fingerprint` when supplied. The manifest records the effective sampling
policy and endpoint. Chat Completions does not report the effective reasoning
effort, so that reported field remains null. OpenAI describes its deprecated
seed parameter as best effort and does not guarantee deterministic outputs,
even with a fixed seed; the returned fingerprint can also be absent. See the
[model parameter guide](https://developers.openai.com/api/docs/guides/latest-model),
[Chat Completions request reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create),
and [Responses request reference](https://developers.openai.com/api/reference/resources/responses/methods/create).

The report stores `metadata.reproducibility.analysis_signature` and its
`manifest`. The manifest describes source and current-input fingerprints,
settings, effective execution, template and code hashes, selected runtime
versions, provider/model settings, and the sampling parameters exposed by the
adapters. It omits credentials, timestamps and run IDs; endpoint user information,
query strings and fragments are removed. Matching signatures identify comparable
run conditions, not identical provider responses.

`metadata.content_digest` fingerprints the canonical inventory payload: unit
text and type, candidate claim texts, verified qualifier data, and exact source
assertion/context anchors. Generated unit, candidate and occurrence IDs do not
drive this digest. It does not hash every report field or the raw provider
exchanges. Equal digests mean those included records match; they do not prove
completeness, semantic correctness or equivalence under paraphrase. Inspect the
full reports for candidate decisions, relations, coverage details and call
records.
`metadata.reliability_diagnostics` lists unresolved budget/provider stops,
unreviewed or uncertain atomicity, unconfirmed or uncertain equivalence,
unexamined pairs, and unresolved source foci. Interpret it alongside report
status, issues, provisional count flags, and the manifest.

## Source and prompt limits

Long-target direct and QA discovery uses disjoint original-source windows. A
failed or candidate-cap-saturated window is split while the per-target window
limit and shared call budget permit; splitting stops at the minimum window size
or configured invocation limit. Remaining windows are reported as pending.
Evidence offsets still refer to the unchanged original transcript. A successful
window or a lower prompt size does not establish source coverage.

Recovery prompts are measured after assembly. They retain the complete original
target, every accepted active meaning, any verified qualifiers, and literal
evidence.
Focus-specific repair state omits unresolved proposals unrelated to that focus
while preserving every proposal in the report. The request keeps cited reference
context and drops optional uncited neighboring context first if it exceeds the
configured character limit. It never truncates required transcript text; if the
complete required prompt is still too large, it remains unresolved and is
reported. These bounds prevent silent content clipping, but may leave work
unfinished under the configured budget.

## Granularity and equivalence

The counted object is one complete contextual proposition, including its
attribution, modality, negation, conditions, quantities and scope. Multiple
discovery routes are provenance, not additional propositions. A source sentence
can contain zero, one or several such propositions; a source slot is never a
unit count. Unresolved alternatives stay visible with provisional counts.

An independently discovered coverage focus that passes the complete candidate
validation may enter the inventory verbatim when it lacks a verified active
correspondence. The `coverage_promotion` candidate retains the focus's exact
text, evidence, proposed qualifiers, raw proposal and validation signals;
`validation_reused_from` and `recovery_focus_id` identify the original focus.
It still receives independent atomicity review and ordinary consolidation.
Unvalidated foci are never promoted, and this route cannot restore a verified
decomposed parent. Promotion does not establish exhaustive coverage.

Atomicity and parent/parts review are two bounded checks. By default, each target
can use up to 8 independent atomicity audit requests, with at most 8 distinct
accepted meanings in each request, plus up to 8 joint parent/parts reviews. Both
limits use `--information-granularity-checks` (range 0–64); the requests also
consume the shared logical-call budget and remain subject to context and provider
limits. The report records `atomicity_review_requests` separately from
`granularity_checks`.

These judges receive explicit claim-text fields: `parent_claim_text` and indexed
`component_claim_texts` for joint review, and `claim0_text`, `claim1_text`, etc. for
atomicity. Questions name those fields directly. Separate `source_scopes` retain
marked assertion spans and each claim's own cited reference context for reference
resolution, governing qualifiers and occurrence fidelity. Source assertions that
the claim does not communicate cannot enlarge its meaning. Candidate annotations,
IDs and other provenance are not additional propositions. An identical statement
or complete paraphrase cannot be a proper component merely because its source
contains other facts. Decomposition generation likewise targets only the explicit
parent claim; original candidates, raw scores and evidence remain in the report.

`CandidateRecord.atomicity_state` is `not_reviewed`, `atomic`, `compound`, or
`uncertain`, with audit signals and the representative review candidate retained
on the record. Existing granularity proposals and support validation remain
separate. When these checks are enabled, accepted active units without confirmed
atomicity remain in the report with
`inventory_state="provisional_granularity"`; the run stays partial or becomes
failed if analysis stops with a provider failure. A verified decomposition must still preserve the
parent's entire meaning, qualifiers and source occurrence. Setting
`--information-granularity-checks 0` is an explicit audit ablation: it skips both
atomicity and joint parent/parts checks, so it provides no independent
atomicity-based reliability evidence.

A gray abstract splittability decision can now request concrete component
witnesses within the same joint-review budget. Each exact parent meaning has
at most one generation attempt. Newly discovered semantic peers can prompt a
new bounded joint review; repeated route or annotation variants cannot.
Only the full concrete parent/parts verification can authorize replacement,
including its unchanged parent-compound, collective preservation, independence,
distinctness, source-occurrence and qualifier gates. Gray or failed concrete
checks retain the parent and proposed parts as unresolved alternatives.

The exact duplicate shortcut applies only when validated complete candidate text
and complete evidence match, ignoring evidence list order and repeated identical
evidence entries. Unverified type labels do not add meaning: all raw labels and
qualifier annotations remain on their candidates. A different asserted occurrence
or cited reference scope still requires semantic evaluation. Canonical source
coordinates, literal wording and complete text determine scheduling and the
representative, so discovery order alone does not choose a different grouping.
Every other proposed semantic merge needs an individual
source-scoped follow-up with directional entailment checks in both directions.
The analyzer prioritizes prospective equivalents and uncertain pair decisions
under `--information-relation-adjudications` (default 256, at most one follow-up
per pair); prospective equivalence confirmations are prioritized before other
uncertain relation follow-ups. Follow-ups also consume the pair and logical-call
budgets. An
unconfirmed prospective merge stays separate and provisional; similarity,
shared evidence and a primary relation label do not authorize it. Final groups
still require complete-link compatibility, so each pair of group members must
have a confirmed equivalence decision. Unexamined or unresolved relations can
leave one utterance represented by multiple units.

## Jev relation prompt trial

On 2026-10-11, a candidate inspired by the
[Jev 1.13 guidance](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)
used explicit claim fields and shorter directional questions (trial commit
`3f3dd29`). A predeclared diagnostic compared v10 and that v11 candidate on eight
fixed relation cases, with three fresh repetitions per variant: 48 evaluator
calls, all completed, with no generator calls and an unchanged 0.90 threshold.
These selected cases and expected labels are diagnostic controls, not an
independent reference inventory or a population accuracy estimate.

V10 identified all 9 expected-equivalent and 15 expected-distinct repetitions.
V11 identified 3 expected-equivalent repetitions; the other 6 were uncertain
(the Bebe exact-wording case and a synonym case, three repetitions each).
Both variants identified all 15 expected-distinct repetitions without a false
merge. The trial therefore supplied no measured benefit and lost resolved
equivalences on these controls. The analyzer retains v10; the OpenAI fixed
sampling profile remains unchanged. This result does not rule out other
applications of Jev's guidance or establish broader model performance.

The original trial remains in Git history. Local requests, responses and analysis
are preserved under
`outputs/evaluation/repeatability_tuning_20261011T150112Z/jev_ab/`; the predeclared
fixtures are under `outputs/evaluation/jev_ab_predeclared_20261011T150237Z/`.

## Comparing results

For a comparison, keep and report every run's signature, digest, status, settings,
counts and diagnostics. A changed signature means a recorded manifest field
differs. A matching signature with a changed digest shows that the canonical
inventory records differ under those conditions; inspect changed proposition
text, qualifiers, source anchors and pair decisions. A matching digest shows
repeatability of the hashed records for those runs only. Partial or failed runs
remain incomplete even if their digests match. A signature only covers fields in
the manifest; it cannot identify provider-side changes that the manifest does
not expose.

Use a separately authored and reviewed, versioned reference inventory to assess
source fidelity, unit granularity and occurrence alignment. The repository's
[offline reference evaluation guide](information_reference_evaluation.md)
defines that contract. Do not use the analyzer's own support, atomicity or
equivalence answers as a gold label for its output. Record annotator provenance,
protocol, scope, ambiguous cases and sampling; keep unresolved comparisons
unresolved instead of scoring them as errors or successes.

These distinctions follow the scope of related evaluation work. [FActScore](https://arxiv.org/abs/2305.14251)
evaluates fine-grained factual precision; that does not by itself establish
exhaustive source coverage. [SummaC](https://aclanthology.org/2022.tacl-1.10/)
shows why input granularity matters for NLI-based summary inconsistency detection,
but does not validate this source-level inventory protocol. [InferBR](https://aclanthology.org/2024.lrec-main.793/)
supplies Portuguese NLI data, not
an accuracy estimate for the selected evaluator on these transcripts. TypeSafe's
[confidence guidance](https://docs.typesafe.ai/confidence) describes answer
probabilities and confidence signals; those values are routing signals here,
not measured correctness rates or Portuguese calibration evidence.
