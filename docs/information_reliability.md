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

The exact duplicate shortcut applies only when validated candidate text, type and
complete evidence match. Every other proposed semantic merge needs an individual
source-scoped follow-up with directional entailment checks in both directions.
Relation template v11 presents `left_claim_text` and `right_claim_text` before
separate `source_scopes` containing each claim's complete original marked target
and its own cited reference segments. Primary comparisons and directional
follow-ups share that representation. In a batch, every independently keyed
question names its own pair and claim IDs. Source material resolves references
and governing scope; its other assertions never enlarge the compared claims.
Six shorter field-specific questions retain both directional checks, correction
and incompatibility checks, and the separate complete-meaning check. Thresholds,
Choice order, conflict handling and merging rules are unchanged.

This follows the [Jev 1.13 guidance](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)
to use explicit inputs and focused semantic questions instead of asking one
judge to recover the comparison target from a broader task. It reduces an
avoidable ambiguity in the payload; offline payload tests establish its
construction, not better model accuracy. Fixed-input fresh provider comparisons
and full fresh runs are separate evidence. Identical wording with different
anchors still requires source-aware confirmation; neither the prompt change
nor a high primary probability is an automatic equivalence proof.

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
