# Open Video Summary

Research code for video and multi-video summarization. HSMVideoSumm combines
transcript and topic segments with selection criteria for introductions,
subjectivity, redundancy, visual quality, and chronology.

## Windows setup

Use **Python 3.11 x64**. Python 3.13 is not compatible with the TensorFlow 2.17
used by this project. The setup uses CPU PyTorch and works without a GPU. From
the repository root, run these commands in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
.\.venv\Scripts\python.exe -m open_video_summary doctor
```

The script creates `.venv`, installs the pinned versions in
`requirements-windows.lock`, installs the project in editable mode, and prepares
the sample assets. It can be run again; existing videos and model files are
reused. `-ExecutionPolicy Bypass` applies only to that PowerShell process.

The first setup downloads about 662 MB for the original subjectivity classifier,
207 MB for CPU PyTorch, and 382 MB for Windows TensorFlow. Allow a few GB of disk
space. The classifier ZIP's SHA256 is checked before extraction. The three sample
videos are in the tracked `data/raw/bebe_real.zip` file (35 MB).

## Run the bundled summary

```powershell
.\.venv\Scripts\python.exe -m open_video_summary summarize
```

This runs HSMVideoSumm on the three videos and 13 pre-segmented
clips in `data/processed/bebe_real.json`. The transcripts and topics are already
in that versioned dataset, so the command does not need Ollama, OpenAI, Whisper,
ElevenLabs, or an API key. HSMVideoSumm, video processing, and rendering run
locally on the CPU, with a CPU thread budget of two by default.

The command writes:

- `outputs/bebe_real_summary.mp4`: the summary video with audio;
- `outputs/bebe_real_summary.json`: selected clips and timestamps;
- `outputs/bebe_real_summary_handler.json`: selection decisions;
- `outputs/bebe_real_summary_audit.json`: numerical scores, all text similarities, and exclusion reasons;
- `outputs/bebe_real_summary_visual_profile.json`: visual stage timings and counters;
- `app.log`: execution log.

Selected clips and duration depend on the algorithm. Running the bundled example
checks that the pipeline works; it is not a scientific evaluation.

### Visual processing and timing

Choose the CPU budget with the global `--threads` option **before** `summarize`:

```powershell
.\.venv\Scripts\python.exe -m open_video_summary --threads 4 summarize
```

Set `OVS_THREADS=4` in the repository-root `.env` to use that budget by default.
The CLI reads that file from any working directory. An explicit `--threads`
value overrides the process environment and `.env`; otherwise, `OVS_THREADS`
from the process environment takes priority over `.env`. Blank values are treated
as absent, and the final default is two threads.

For visual extraction, the runner starts at most `min(threads, unique requests)`
Windows-compatible processes. Each process decodes one source interval or full video and runs SIFT
and keyframe matching with one OpenCV/BLAS/OpenMP thread. A budget of one uses
the serial path. A request is a distinct source interval in `segment` scope or a
distinct source video in `video` scope. Each evaluation creates and closes
its own pool; startup and numerical-library imports can outweigh parallel work
on small inputs.

Other summarization stages keep the native-library thread settings selected by
`--threads`. KMeans fits run in the parent, after the active extraction window
finishes, so those two compute phases do not multiply the thread budget.
FFmpeg rendering still uses two threads independently. Codec helper threads and
library housekeeping threads mean this is not a limit on the OS thread count.
The CLI sets the usual OpenMP/BLAS environment limits itself. The already-pinned
`threadpoolctl` dependency also limits native pools inside workers when the
calling Python program imported them before Windows spawned the child.

`QualityPick` keeps SIFT descriptors in memory for reuse during one evaluation.
The cache holds at most 256 MiB of
descriptor arrays and is released after the evaluation, including on failure.
It keys entries by canonical file path, file metadata, scope, and extraction settings,
including start/end times in `segment` scope. Repeated identical intervals share
descriptors; different intervals from the same source have separate entries. In
`video` scope, segments from the same source share full-video descriptors;
larger entries are recomputed, and least recently used entries are evicted when
needed. Sampled frames and active candidate-group/KMeans arrays require additional
memory beyond this cache capacity. Parallel workers each hold their own sampled
frames and SIFT working memory. Prefetch holds at most `min(workers + 1, unique requests)`
pending tasks, including one extra task so a worker can continue when a later request
finishes before the first requested result. Completed descriptors wait in temporary
numeric `.npy` files until their original request order. These files contain no
pickled objects and are deleted on consumption or pool cleanup, including on
failure. Futures retain metadata rather than arrays.
If a source changes or a previously consumed entry needs recomputing after cache
eviction, prefetch closes and the remaining requests use the serial path.
Custom feature extractors retain their original serial per-segment calls.
Python callers opt into workers with `QualityPick(visual_threads=4, ...)`; the
Python API defaults to serial extraction. On Windows, call a parallel evaluation
inside the usual `if __name__ == "__main__":` guard in an importable Python script.
Setting `max_descriptor_cache_bytes=0` disables both reuse and speculative extraction.

The default visual scope is now `segment`. It extracts frames from each
candidate's source path in the interval `[start, end)`. Bounds use the source's
floating-point FPS, including fractional rates. Frame seeks are checked; a
backend that cannot report the requested position uses a fresh sequential
reader. Resolution stays native. The nominal one-frame-per-second sampling
keeps the existing global stride: source frame indices divisible by
`int(native_fps)`. At fractional rates this is an approximate one FPS schedule.
SIFT still excludes the first and last sampled frames (`frames[1:-1]`) within
the extracted interval. This changes descriptor inputs and can change summary
selection.

Set `OVS_VISUAL_SCOPE=video` in the root `.env` for the legacy full-source scope.
`summarize --visual-scope segment|video` overrides the process environment, then
root `.env`, then the default `segment`. A blank value is absent; a blank process
value masks the file value and uses the default. `video` keeps the previous full
video sampling, integer-FPS boundary behavior, and per-source descriptor reuse.
Introduction and subjectivity retain their existing frame retrieval in both modes.
Python callers select the scope with `QualityPick(visual_scope="segment", ...)`
or `QualityPick(visual_scope="video", ...)`.

For example, save two separate local validation runs on the same preprocessed data:

```powershell
.\.venv\Scripts\python.exe -m open_video_summary summarize `
  --dataset data/processed/catedral_notre_dame.json --visual-scope segment `
  --output outputs/cathedral_segment/summary.mp4 --no-render
.\.venv\Scripts\python.exe -m open_video_summary summarize `
  --dataset data/processed/catedral_notre_dame.json --visual-scope video `
  --output outputs/cathedral_video/summary.mp4 --no-render
```

Each candidate group still fits its own 300-word KMeans dictionary with the same
descriptor order and multiplicities. SIFT reuses one detector per extraction. Descriptor
matching keeps the same dot products, thresholds, and NumPy sort behavior for
ties and NaNs; repeated reverse comparisons reuse their exact previous result.

The visual profile records total `QualityPick` elapsed time, including process
startup, imports, temporary-file I/O, waiting, and cleanup. `stage_seconds` holds
exclusive times in the parent: extraction stages when serial, or worker waiting
and descriptor loading when parallel, plus KMeans fit/prediction, BoVW dataframe
construction, and quality ranking. Their sum can be smaller than the total
because it excludes some orchestration and logging.
`worker_stage_seconds` separately sums work across processes: frame decoding
(including sampling and grayscale conversion), SIFT detection, keyframe matching,
and descriptor assembly. These overlapping times must not be added to the parent
stages or treated as elapsed time. `workers` records task PIDs, monotonic start/end
times, extraction CPU time, native thread limits, and import/spool timing.
`worker_cpu_seconds` sums extraction CPU time; it excludes process startup and
imports, whose full cost is included in the total elapsed time.
Counters include actual video reads, decoded/sampled frames, SIFT frames, cache
hits/misses, KMeans fits, and the pending-task limit and peak. The profile contains
no transcript or video pixels.
Python callers can read `QualityPick.last_profile`; the CLI saves it alongside
the summary and also logs it in `app.log`.

The existing KMeans default has no fixed random seed, and candidates are sets.
Independent runs can therefore differ even with unchanged settings. Controlled
comparisons must hold the candidate order and random state constant and record
thread settings. Parallel extraction consumes the same arrays in the original
candidate order and leaves random seeds, dictionary fitting, and ranking in the
parent unchanged. It does not make the existing algorithm deterministic.

### Selection audit

Every saved summary also gets `<summary_stem>_audit.json`, including with
`--no-render`. This separate audit uses `schema_version: 1` and contains no
transcripts, pixels, prompts, or credentials. Source paths inside the repository
are root-relative POSIX paths. Segment IDs such as `v0:s2` refer to input video
and segment positions, with source path, original order, start/end, and topics in
the manifest. IDs for direct criteria calls without source videos use the actual
first-request order (`external:s0`, etc.).

| Field | Meaning |
| --- | --- |
| `sources`, `segments` | Source identity and input-order manifest; candidate timestamps stay original in either scope |
| `criteria.ContentBasedRedundancy.raw_matrix` | Complete pandas correlation result before any selection mask; rows/columns follow `matrix_segment_ids` |
| `pair_decisions` | Every ordered matrix cell, with its value/origin, filter reasons, eligibility, video-pair maximum, maximum tie flag, and actual cluster memberships |
| `cluster_decisions`, `clusters`, `segment_cluster_memberships` | Edge-processing steps and final connected-component memberships; each segment belongs to at most one group |
| `criteria.QualityPick.clusters` | Actual candidate order, descriptor counts and extraction scope/bounds, visual-word weights, summed score, full rank order, exact ties, and chosen flags |
| `outcome` | Final output order and per-segment state, recorded actions, and exclusion reasons |

Text coverage includes all computed off-diagonal similarities, even those below
the strict `>` threshold or within one video. Pandas calculates each unordered
off-diagonal pair once and mirrors it. Its diagonal is forced to 1; it is labelled
`pandas_diagonal` and is not a measured self similarity. The audit preserves that
distinction, structural mirrors, and unavailable values (`null`). It records the
existing masks, then the maximum among retained pairs for each ordered video pair,
then prior discard/output eligibility and the resulting grouping. These are the
actual stages used for selection; logging does not recompute TF-IDF or similarities.

Content redundancy groups are connected components of the eligible selected
pairs. A segment can connect indirectly: pairs A-B, A-C, and B-D form one group
with A, B, C, and D. A later pair joining two existing groups merges them. Groups
follow the first encountered pair in each final component, and their members
remain sets. In `cluster_decisions`, `action` describes the step when the pair
was processed; `cluster_id` refers to the final group after all merges. Earlier
`create_cluster` steps can therefore reference the same final group.

The visual score is the existing BoVW sum:
`sum(term_frequency * log10(dictionary_size / candidate_document_frequency))`.
It is a ranking score within its fitted candidate group, not a confidence or a
calibrated image-quality probability. Exact ties use pandas `nlargest`'s existing
first-candidate rule; both the order and all tied IDs are recorded. Missing word
weights are `null` and are skipped by the existing sum. In `segment` scope, an
interval without SIFT descriptors has zero word terms and score 0 when the group
can still fit the unchanged dictionary. Fewer than 300 total descriptors fails
explicitly and saves a partial audit with the failure reason; the vocabulary is
never reduced automatically. Empty, reversed, negative, and nonfinite interval
bounds fail explicitly. An interval beyond the source or without a sampled frame
has no descriptors. The legacy `video` scope retains its empty-SIFT error behavior.

Unchosen candidates are recorded in this audit without adding discard actions to
the existing handler. `recorded_actions` reflects the existing handler logs;
candidate rank decisions provide the additional visual exclusion evidence.
KMeans and set iteration remain unchanged, so independent runs can vary even in
the same scope. Mode comparisons need the same candidate order, random state,
data, and thread settings to attribute a difference to scope alone.

Library calls to `Summarizer.summarize(...)` save the audit by default; set
`audit_output_path` to choose its path. With `save_output=False`, inspect
`summarizer.last_audit` and `summarizer.last_handler` in memory. Set
`collect_audit=False` to disable capture. Direct criterion calls expose their
report through `criterion.last_audit` and `handler.audit`. Old handler JSON files
remain readable. Audit capture adds no model run and does not change selection or
consume random state.

## Inventory information before summary selection

Information analysis is opt-in. It runs **once before Introduction** in the
existing HSM pipeline, using all source transcript segments while the selection
handler's output is still empty. It returns a separate information report and
adds no include, discard, pick, or output decisions. The original `summarize`
command continues to work without provider credentials when this option is absent.

The generator uses the existing Ollama/OpenAI adapters. An independently selected
evaluator checks support, qualifiers and granularity, audits the original text
for possible omissions, and compares candidate meanings. Its provider selects the
API service: the currently implemented evaluator provider is `typesafe`, and
`jev-1.13.0` is its pinned model. Set the evaluator role's key in the process
environment or root `.env`, then use a **new report path**:

```powershell
$env:OVS_EVALUATOR_API_KEY = "your-evaluator-key"
.\.venv\Scripts\python.exe -m open_video_summary summarize `
  --dataset data/processed/bebe_real.json `
  --output outputs/bebe_real_summary.mp4 `
  --information-report outputs/bebe_real_information_run01.json `
  --information-csv outputs/bebe_real_information_run01.csv `
  --llm-provider ollama --llm-model gemma2 `
  --evaluator-provider typesafe --evaluator-model jev-1.13.0
```

The summary still needs its normal video and classifier assets. Ollama must have
the configured generator model available. Alternatively, set `OPENAI_API_KEY`
and replace the generator options with `--llm-provider openai --llm-model gpt-6-luna
--llm-reasoning-effort high`. With either generator, TypeSafe/Jev receives the
selected transcript context, candidates and evaluation questions through HTTPS. OpenAI
also receives extraction prompts and transcript context when selected; an Ollama
server receives them at its configured endpoint. No source video or audio is sent
by the inventory analyzer. API services can incur usage charges.

`summarize --analyze-information` enables the same hook and automatically chooses
`outputs/information/<run-id>.json`. `--information-report` also enables it.
Explicit report files are never overwritten, including when the HSM singleton is
used twice. Reports and optional CSV tables must live under `outputs/`; resolved
aliases of the input dataset, source videos and summary artifacts are refused.
Writes publish a complete UTF-8 file atomically. If analysis fails or reaches a
limit, summary selection proceeds and the CLI prints the analysis status. A
report destination failure is visible in `last_information_error` and triggers a
separate failure report under `outputs/information/` when that directory is writable.

For text-only inspection, use the additional command below. It reads a segmented
dataset without checking MP4 existence, loading the subjectivity classifier,
running Whisper, or requiring FFmpeg:

```powershell
.\.venv\Scripts\python.exe -m open_video_summary analyze-information `
  --dataset data/processed/bebe_real.json `
  --output outputs/bebe_real_information_text_run01.json `
  --llm-provider ollama --llm-model gemma2 `
  --evaluator-provider typesafe --evaluator-model jev-1.13.0
```

This command exits nonzero for a partial or failed analysis after saving its
report. Missing `OVS_EVALUATOR_API_KEY` creates a failed report before contacting
the generator; it is never interpreted as a successful inventory of zero units.
An unsupported evaluator provider or invalid evaluator settings instead produce a
configuration error before the analysis begins. The passive HSM continuation
applies to analysis failures after a valid configuration has been constructed.

### Protocol and report interpretation

`contextual-propositions-v2` inventories verbal content **as communicated**, not
the external truth of a claim. A unit preserves entity context, attribution,
negation, modality, numbers, conditions and exceptions. Independent properties
become separate units; a condition remains with its consequent. Joint source
review checks proposed parent/parts representations even when an earlier
single-candidate decision called the parent atomic. A parent becomes
`inventory_role: "decomposed_parent"` only when the verified components preserve
its complete meaning, governing qualifiers and source occurrence, and are
distinct independent contents. It remains in `candidates` as provenance but
does not add a third unit to its two parts. Specificity or overlapping quotations
alone cannot establish this decomposition.

Incomplete, conflicting or uncertain alternatives remain visible in
`decompositions`; affected units carry `inventory_state:
"provisional_granularity"` and `decomposition_ids`. Their summed count is
provisional, with `provisional_granularity_units` reported separately. A later
pair decision that conflicts with verified component distinctness restores the
parent as a visible alternative. This favors traceable uncertainty over silently
dropping residual meaning or conditions.

Coverage also has a source-content ledger in `coverage_foci`. An independent
generator traverses the original target without the accepted inventory, proposes
contents with literal evidence, and passes each proposal through the same
support, qualifier and atomicity checks. Each stable focus records its full
proposal validation, verified candidate correspondences, final unit IDs, and a
bounded repair history. A focus can be `missing`, `partial`, `covered` or
`uncertain`. Only complete meaning at the same source occurrence closes a gap;
sharing a whole-sentence quote, adding a duplicate, or reformulating an existing
claim does not. Source foci do not count as candidate units or occurrences.

The separate open audit still traverses the original source for content that
never became a focus. No-progress and budget stops leave unresolved records
visible; even a completed run does not prove exhaustive coverage. Reports use
schema 8, retain model and prompt versions, and treat probability/confidence as
routing signals rather than measured application accuracy.

For matched experiments, `--information-mode direct`, `qa`, `hybrid`, and
`hybrid_coverage` select direct discovery, QA discovery, both, or both plus the
source ledger/audit and gap recovery. All four share source snapshots, candidate
validation, joint granularity, literal repairs and complete-link consolidation.
Only `hybrid_coverage` performs coverage recovery. An explicit mode owns route
selection; without one, the existing QA switch remains effective and coverage
stays enabled. Reports record `metadata.experimental_mode` and every configured
limit. Use the same protocol, reference scope, evaluator and budgets when
comparing quality and cost; an agent-authored development fixture is not a human
reference inventory or a scientific calibration.

Direct extraction and optional anchored question/answer extraction initially
receive the same source context without seeing each other's output. Questions
and answers are discovery aids; they are not units to add to the count. Every
candidate must carry literal quotations and proposed offsets, with assertion
evidence in its target segment. Code checks Unicode character slices and existing
IDs. A matching supplied anchor is retained, including for repeated quotations.
If offsets drift, code resolves only a **unique exact quotation in the same
permitted source segment**. No fuzzy matching or first-occurrence guess is used;
ambiguous and nonliteral evidence remains rejected. The candidate's `raw` output
retains supplied offsets, and `evidence_resolutions` records each resolved anchor.
Every resolved candidate still requires semantic validation. Timestamps are
copied from source segments, with segment-level resolution, never estimated by a
model. Context citations can resolve a reference but create no occurrence.

Jev receives readable assertion quotations, the whole original target, whole
source segments for cited reference context, and the proposed claim. The
source-owned scope exposes governing conditions, attribution, modality and
negation outside a generator-selected quote. It does not authorize new assertions
from reference context. Timing metadata and extraction instructions are excluded.
Focused anchor binding first checks that the selected wording identifies the
candidate's property or event. That request contains only assertion wording and
claim plus literal cited reference context, which can resolve references but
cannot supply a new assertion. Exact quote/claim
identity, or one contiguous exact candidate passage covering every selected
assertion interval, can establish binding in code. Matching only a different
occurrence or combining disjoint passages cannot. This proves binding only;
source-scope fidelity and atomicity checks still run. Six independent checks gate
content support and qualifier fidelity, together with atomicity and applicable
QA checks. An empty generated annotation never disables
these source-based checks. An atomic claim need not include unrelated independent
facts in a longer source passage; coverage auditing handles those omissions.

Binding can also be established when the candidate retains one complete selected
source sentence verbatim while adding a contextual description. Only the
sentence's terminal punctuation may be omitted; internal wording is preserved,
and a mid-sentence fragment does not qualify. This proves ownership of the
retained assertion only. Added entity descriptions, conditions, quantities and
other content still require every source-fidelity and atomicity check; context
alone never creates an occurrence. Reports distinguish this binding origin as
`literal_assertion_sentence`.

After both independent discovery routes, a bounded source-literal repair can
copy a complete selected assertion sentence into a new candidate when an
expanded or paraphrased proposal remains unresolved. The original proposal is
retained and the new `literal_recovery` record identifies it through
`literal_repair_of`. The copy keeps its cited reference context, passes every
fidelity and atomicity check, and cannot omit governing wording from that
sentence. Fragments, unchanged proposals, exact duplicates and candidates from
outside their discovery window are skipped. Compound sentences remain subject to
repair; source copying alone never establishes a valid unit. At most four such
repairs run per target by default, sharing the global call cap. Each complete
source assertion gets at most one attempt even when routes cite different
reference context or propose different annotations for that same assertion.

Auxiliary annotation quality is audited separately from content acceptance at the
same threshold. Candidate `proposed_qualifiers` and raw generator fields remain
proposals, with `annotation_state`, reasons and signals. A supported proposition
can be counted while its proposed annotations remain uncertain or invalid.
Canonical units expose `qualifiers: null` and `qualifier_state: "unknown"` in that
case; null does not mean that the proposition has no qualifiers. Verified canonical
fields belong to the unit's `representative_candidate_id` and its chosen text;
they are not borrowed from another paraphrase. Unverified proposals do not affect
coverage, pair comparison, matching or duplicate shortcuts. Attribution names a
reported speaker or claimant, rather than an actor merely performing an action.
Choice options retain stable IDs with descriptions. Content and equivalence
thresholds remain unchanged.

Within an immutable source target, completed validation can be reused only for
the same resolved evidence, claim, QA question/answer, proposed annotations and
source scope. Each repeated candidate retains its own raw output and route, with
`validation_reused_from` tracing the original decision. Errors are not cached;
changed content or evidence requires new checks. Reuse also retains negative and
uncertain decisions rather than seeking a different probability for unchanged
content. Focused binding can add a logical call for a nonliteral paraphrase;
reuse and free literal binding avoid redundant calls. The configured global
call limit is unchanged, and any unexecuted work remains visible.

Coverage checks read the original target and accepted propositions, including
markers for quantities, negations, conditions and references. These markers are
hints, not an alternative word-count measure or a content filter. A full sentence
appearing as evidence does not mark all its propositions as represented. Possible
gaps trigger additional extraction and validation; an uncertain audit, no progress
or an exhausted budget remains visible.

Pair comparisons preserve complement, specificity, contradiction and correction
relations. Only sufficiently strong equivalence decisions merge meanings, and
every member of a merged group must be pairwise compatible. Equal text in different
contexts still needs evaluation. Exact validated duplicates with identical
evidence can be deduplicated in code, independently of auxiliary annotations.
These free duplicate checks run before paid comparisons even if the logical-call
budget is exhausted. Shared assertion anchors and textual similarity prioritize
the remaining comparisons; priority alone never establishes equivalence.
Unexamined pairs remain explicit and can leave one utterance in multiple units.
The report-level `counts.counts_provisional` flag marks both whole-input totals
and the by-segment/by-video aggregates provisional whenever analysis is partial
or failed. The CSV repeats this flag on each row. The per-scope
`occurrences_provisional` flags and aggregate `counts.occurrences_provisional`
have a narrower meaning: they identify unresolved alignment from overlapping,
nonidentical assertion anchors retained within a semantic unit. A false
alignment flag therefore does not make counts final while `counts_provisional`
is true.
Occurrence deduplication requires the same
set of literal assertion anchors (source identity, segment and character offsets);
context citations do not determine identity. Repetitions retain distinct
positional source identities and assertion spans. Different overlapping anchors,
including contained quotes, remain separate evidence groups with
`alignment_state="unresolved_overlap"` and an alignment issue. Their total is
provisional and may overcount utterances; `counts.occurrences_provisional=true`
and the corresponding segment/video flags expose this alignment uncertainty.
Overlap alone never merges occurrences. These groups can still belong to one
semantic unit, so the distinct unit count remains separate from occurrence
alignment.

JSON is the canonical report. It includes the frozen source snapshot, source and
current-input fingerprints, source/current order, selection stage order,
`stage_id="before_introduction"`, candidate validation and reasons, contextual
units, assertion evidence groups and their alignment state, context evidence,
pair relations, joint decompositions, individual source foci, coverage records,
counts by segment/video/whole input and unresolved issues. Calls record requested
and returned models, input/output hashes, usage when available, latency and retry
attempts. Report schema 8 retains nullable supplied offsets in evidence-resolution
records, separates content validation from annotation audit, and keeps canonical
qualifiers nullable. Canonical evidence offsets remain integers.
Source data and older reports are not rewritten. Generator and evaluator template hashes, the evaluator
template version, protocol version and settings are retained without keys.
The optional CSV is an occurrence evidence table with alignment state, report
count provisionality, canonical qualifier state, nullable qualifiers and
representative candidate ID; the JSON contains the full evidence. When counts
are provisional, the CLI labels both unit and occurrence totals provisional.
It adds “alignment pending” to occurrence groups when anchor alignment is the
specific unresolved evidence issue.

`completed` means the configured operations finished without recorded pending
issues. It is **not proof of complete extraction or calibrated semantic accuracy**.
`partial` retains accepted units alongside pending work and provisional counts;
`failed` distinguishes unusable analysis from a valid empty result. Only a
completed empty inventory has `counts.valid_zero=true`. Jev's Noul value is a
probability of a yes answer; Choice's selected-option probability and its separate
confidence metric are recorded distinctly. Neither is a measured accuracy rate in
this application. The defaults below need validation on Portuguese annotations.

### Analysis bounds and library contract

Non-secret CLI settings override the process environment, then root `.env`, then
defaults. Credentials follow process environment > root `.env` and have no CLI
flag, matching the project's other services. A blank process key masks the file
key. The generator keeps the existing `OVS_LLM_*` settings and selected provider
credentials. Evaluator settings use the independent `OVS_EVALUATOR_*` role;
`OVS_EVALUATOR_API_KEY` is routed only to the selected evaluator and never falls
back to generator/transcription keys. Provider constructors, model defaults and
endpoint defaults are selected through `EVALUATOR_PROVIDERS` in the existing
adapter factory. Only `typesafe` is currently implemented; registering another
adapter requires the typed evaluator contract rather than the generative LLM
interface. Unsupported names raise an explicit configuration error.

TypeSafe defaults to the pinned `jev-1.13.0`, not a moving latest alias;
`OVS_EVALUATOR_MODEL` or `--evaluator-model` can explicitly select another model.
The stdlib HTTP adapter implements the official
[TypeSafe API](https://docs.typesafe.ai/api); model version and confidence semantics
are described in [models](https://docs.typesafe.ai/models) and
[confidence](https://docs.typesafe.ai/confidence).

| CLI option | Environment setting | Default |
| --- | --- | --- |
| `--no-information-qa` | `OVS_INFORMATION_QA` (`true`/`false`) | QA enabled |
| `--information-mode` | `OVS_INFORMATION_MODE` | Unset: existing QA switch plus coverage; explicit `direct`, `qa`, `hybrid`, `hybrid_coverage` modes override that switch |
| `--information-granularity-checks` | `OVS_INFORMATION_GRANULARITY_CHECKS` | 8 joint parent/parts reviews per target (0–64); 0 disables joint review |
| `--information-coverage-foci` | `OVS_INFORMATION_COVERAGE_FOCI` | 16 independent source foci per target (0–256); 0 selects the legacy segment audit |
| `--information-gap-repairs` | `OVS_INFORMATION_GAP_REPAIRS` | 2 attempts per source focus (0–10), also bounded by target rounds and global calls |
| `--information-concurrency` | `OVS_INFORMATION_CONCURRENCY` | 2 independent source targets (range 1–8) |
| `--information-max-calls` | `OVS_INFORMATION_MAX_CALLS` | 256 logical calls |
| `--information-rounds` | `OVS_INFORMATION_ROUNDS` | 2 recovery rounds per target |
| `--information-pair-concurrency` | `OVS_INFORMATION_PAIR_CONCURRENCY` | 8 comparison requests in flight (range 1–8) |
| `--information-pair-batch-size` | `OVS_INFORMATION_PAIR_BATCH_SIZE` | 4 independently keyed pair decisions per request (range 1–8); 1 enables the individual baseline |
| `--information-relation-adjudications` | `OVS_INFORMATION_RELATION_ADJUDICATIONS` | Up to 256 individual source-scoped follow-ups for uncertain relations, at most one per pair |
| `--information-direct-window-chars` | `OVS_INFORMATION_DIRECT_WINDOW_CHARS` | 240 original-source characters per direct discovery window for longer targets (range 80–4000) |
| `--information-max-direct-windows` | `OVS_INFORMATION_MAX_DIRECT_WINDOWS` | At most 16 direct generation invocations per long target, including failed-window decomposition (range 1–64) |
| `--information-literal-repairs` | `OVS_INFORMATION_LITERAL_REPAIRS` | At most 4 complete source-sentence repairs per target after discovery (range 0–64; 0 disables) |
| `--information-max-pairs` | `OVS_INFORMATION_MAX_PAIRS` | `auto`: remaining global calls times bounded batch capacity; an integer caps paid pair decisions, including follow-ups and 0 |
| `--information-qa-window-chars` | `OVS_INFORMATION_QA_WINDOW_CHARS` | 240 original-source characters per independent QA discovery window (range 80–4000) |
| `--information-max-qa-windows` | `OVS_INFORMATION_MAX_QA_WINDOWS` | At most 16 QA generation invocations per target, including failed-window decomposition (range 1–64) |
| `--information-max-candidates` | `OVS_INFORMATION_MAX_CANDIDATES` | 16 per generation; bounded direct and QA windows emit at most 4 |
| `--information-context-chars` | `OVS_INFORMATION_CONTEXT_CHARS` | 24000 characters per prompt, or evaluation context plus questions |
| `--information-context-segments` | `OVS_INFORMATION_CONTEXT_SEGMENTS` | Up to 4 additional current segments from the same video |
| `--information-acceptance` | `OVS_INFORMATION_ACCEPTANCE` | 0.85 for each support/qualifier signal and atomicity |
| `--information-gap-threshold` | `OVS_INFORMATION_GAP_THRESHOLD` | 0.65 yes signals a gap; at most 0.35 means no gap signaled |
| `--information-equivalence` | `OVS_INFORMATION_EQUIVALENCE` | 0.90 selected relation probability |
| `--evaluator-provider` | `OVS_EVALUATOR_PROVIDER` | `typesafe` |
| `--evaluator-model` | `OVS_EVALUATOR_MODEL` | `jev-1.13.0` for TypeSafe |
| No CLI credential flag | `OVS_EVALUATOR_API_KEY` | Unset; required when analysis runs |
| `--evaluator-base-url` | `OVS_EVALUATOR_BASE_URL` | Selected provider endpoint; `https://api.typesafe.ai` for TypeSafe |
| `--evaluator-timeout` | `OVS_EVALUATOR_TIMEOUT_SECONDS` | 30 seconds per HTTP attempt |
| `--evaluator-max-attempts` | `OVS_EVALUATOR_MAX_ATTEMPTS` | 2 attempts; TypeSafe allows at most 5 |
| `--evaluator-operation-timeout` | `OVS_EVALUATOR_OPERATION_TIMEOUT_SECONDS` | 90 seconds for the complete evaluation operation, including admission and retries |
| `--evaluator-request-limit` | `OVS_EVALUATOR_REQUEST_LIMIT` | 60 physical requests per window |
| `--evaluator-token-limit` | `OVS_EVALUATOR_TOKEN_LIMIT` | 80000 estimated input tokens per window |
| `--evaluator-rate-window` | `OVS_EVALUATOR_RATE_WINDOW_SECONDS` | 1 second |
| `--evaluator-limit-group` | `OVS_EVALUATOR_LIMIT_GROUP` | Model identifier; explicitly group models sharing a service limit |
| `--evaluator-token-request-overhead` | `OVS_EVALUATOR_TOKEN_REQUEST_OVERHEAD` | 256 predicted input tokens for unknown request templates (range 0–100000) |
| `--evaluator-token-question-overhead` | `OVS_EVALUATOR_TOKEN_QUESTION_OVERHEAD` | 128 predicted input tokens per question for unknown question templates (range 0–100000) |
| `--llm-operation-timeout` | `OVS_LLM_OPERATION_TIMEOUT_SECONDS` | 360 seconds for the complete generation operation |
| `--llm-max-output-tokens` | `OVS_LLM_MAX_OUTPUT_TOKENS` | 16384 OpenAI output tokens, including reasoning; Ollama generation remains unchanged |
| `--llm-request-limit` | `OVS_LLM_REQUEST_LIMIT` | Unset; optional physical request ceiling per window |
| `--llm-token-limit` | `OVS_LLM_TOKEN_LIMIT` | Unset; optional estimated input plus reserved output token ceiling per window |
| `--llm-rate-window` | `OVS_LLM_RATE_WINDOW_SECONDS` | 60 seconds; OpenAI header limits describe minute windows |
| No CLI switch | `OVS_LLM_LEARN_RATE_LIMITS` | `true`; learn OpenAI limits from response headers |
| `--llm-organization`, `--llm-project` | `OVS_LLM_ORGANIZATION`, `OVS_LLM_PROJECT` | Unset; sent to OpenAI and included in the controller scope |
| `--llm-limit-group` | `OVS_LLM_LIMIT_GROUP` | Model identifier; set a common group for a shared model family |

The initial `TYPESAFE_API_KEY`, `OVS_JEV_*` and `--jev-*` configuration names were
replaced, with no compatibility aliases or hidden fallback. Rename the key to
`OVS_EVALUATOR_API_KEY`, the model/endpoint/timeout/attempt settings to their
`OVS_EVALUATOR_*` equivalents, and any CLI options to `--evaluator-*`.

The call budget counts logical generation/evaluation invocations. Bounded provider
retries add physical requests: the generator uses `--max-attempts` /
`OVS_MAX_ATTEMPTS` (default 3), while the evaluator uses its separate attempt limit.
Provider preflight/model discovery is separate from that budget. Retry records and token
usage, when supplied by the service, allow operational cost accounting; no fixed
currency cost is inferred. Large original targets are not truncated to fit a
request: they become pending work. Pair-budget exhaustion prevents further
automatic merges and marks unique-unit counts provisional. Free exact validated
proofs run first and consume neither paid comparisons nor logical calls. `auto`
removes the former independent 160-comparison default without increasing the
global budget. The library default remains 256 calls; `.env.example` explicitly
sets 1000. Reports distinguish paid pair-decision attempts (`paid_pair_comparisons`),
logical comparison requests (`pair_requests`), individual relation follow-ups
(`relation_adjudications`), and reused exact-proposition relations. Physical retry
attempts and token usage remain separate. A batch saves requests; it does not
turn four independent semantic decisions into one paid pair decision.

Only candidates with identical accepted claim text, type and complete resolved
evidence reuse a representative pair's decision. Every other pair is still
examined; lexical similarity sets order and never establishes a relation.
Reused relation rows name their representative pair, and original candidate and
occurrence evidence remain intact. Groups still require equivalence for every
member pair, including reused exact-proposition decisions; contradictory edges
prevent transitive merges.

A borderline eight-class Choice receives at most one individual follow-up,
within the same global and paid-pair limits. Six binary questions jointly check
directional entailment, incompatible scope, explicit correction and same complete
meaning. The cutoff remains unchanged and is inclusive within two floating-point
representation steps, so arithmetic at 0.10/0.90 does not create false uncertainty.
Initial selected class/probability and all follow-up signals are retained;
`adjudication_strength` describes the subtype's weakest defining signal, while
`equivalence_strength` records the decisive equivalence criterion. Neither is
a calibrated relation probability.

Schema 6 records `equivalence_state` separately as `equivalent`, `distinct` or
`uncertain`. Because equivalence requires entailment in both directions, a decisive
negative in either direction establishes distinctness even when complementary
versus more-specific remains unknown. A decisive same-complete-meaning check can
resolve gray directional signals; conflicting checks remain uncertain. Subtype
uncertainty stays in the relation, metadata and CLI; it does not make otherwise
resolved inventory counts provisional. Genuine equivalence uncertainty still
blocks merges and keeps counts provisional. The follow-up is one fixed request,
with no repeated wording search until a preferred answer appears.
The primary distribution also records `primary_equivalent_probability`,
`primary_distinct_probability` and `primary_uncertain_probability`. Primary
subtype and family decisions divide by the complete raw distribution total,
recorded as `primary_probability_total`, to account for provider rounding.
`initial_probability` and the call's Choice record retain the original scores.
The distinct family sums the six mutually exclusive known non-equivalence
classes; the uncertain class contributes to the denominator but not that family.
This normalized sum can reach the existing equivalence threshold while
the descriptive subtype remains unresolved. Follow-up checks still run within
their limits; a decisive contradiction between primary family and follow-up
evidence keeps the result uncertain. Family mass is a model output aggregate,
not a measured accuracy rate, and its use is identified by
`equivalence_origin: "primary_relation_family"`.
Primary comparisons run in waves that keep up to one-third of remaining calls,
capped by the remaining follow-up allowance, available for those checks. Later
waves reclaim unused reserve. The report records the final representative-pair
dimension and planned initial request count; neither a batch nor a reserve
guarantees that newly discovered content fits the configured budget.

Direct discovery splits targets longer than its configured window size before
generation, avoiding a repeated monolithic request for the entire long target.
Shorter targets keep the ordinary direct request. Its independent invocation
bound includes failed-window decomposition, and `direct_source_windows` records
the original offsets, status and attempt number. Pending direct windows remain
visible and make the inventory provisional.

Independent QA discovery traverses disjoint source windows with at most four
candidates per generation, retaining the full original target and cited context
for references and governing qualifiers. Assertion evidence must start in its
assigned window, and quotes/offsets always belong to the original source. A timed
out window is decomposed under the same provider deadlines and global call cap;
the configured QA invocation limit includes those retries. Pending windows are
reported. QA receives no direct-route candidates. A recoverable route failure
still permits the original-source coverage audit and preserves valid evidence
from the other route. Window completion does not prove semantic coverage.
Both discovery routes retain the complete original target and use source-owned
offsets, including when a sentence or its governing condition crosses a window
boundary. They share the existing global call cap and provider controls; direct
windows do not receive QA output, and QA does not receive direct output.
`discovery_window_declarations` retains the generator's reported window issues
beside trusted request bounds, source-slice hashes and lengths. A generator's
incorrect echo of the input bounds does not replace the code-owned source slice
or reject a valid in-window assertion. This includes the generator's
`discovery_window_offset_mismatch` and `inconsistent_window_offsets` diagnostics
when the supplied bounds and literal slice verify against the original target.
The rule applies only to that target's window metadata; evidence and reference
problems remain operative. Actual outside-window assertions remain
unresolved, with their original and resolved offsets preserved.

Independent source-focus generation uses the `information_foci` output schema.
It copies precise literal quotes and may leave both character offsets `null` for
a quote that occurs exactly once in its cited segment. Code locates that exact
Unicode substring and records the supplied nulls and resolved integer offsets.
Repeated quotes still require correct supplied integer offsets identifying their
own occurrence; missing or incorrect bounds never select the first occurrence.
Mixed null/integer pairs, nonliteral quotes, normalization and evidence from a
forbidden source remain rejected. Other generation routes retain their existing
integer-offset schemas. The source, model, reasoning effort, output ceiling,
provider deadlines and all semantic acceptance checks remain unchanged. This
removes unnecessary model-side coordinate calculation; real latency and quality
must still be measured for each experiment.

Recovery requests share repeated literal citations through one `evidence_table`.
Exact duplicate projected candidates share an `ids` list. Every accepted claim,
verified qualifier and essential citation is retained, together with the full
original source and repair reasons. Repeated provenance, timestamps and numeric
diagnostics remain in the report. This bounds repeated state without truncating
research content. A request still exceeding the configured context cap records
its operation, size, limit and remaining call budget separately from call
exhaustion.

Coverage requests group exact repeats of accepted text, type and full source
evidence, retaining every contributing candidate ID. Original report records and
coverage provenance remain intact. The source ledger advances only when a focus
acquires a fuller verified correspondence. A defective focus hypothesis can be
revised under the same ID only after a source-scoped semantic check confirms a
complete repair of the same content; its original proposal stays in history.
Extra duplicate rows or reformulations do not count as progress. Distinct
assertion spans remain distinct occurrences. With the ledger explicitly disabled
(`--information-coverage-foci 0`), the legacy segment loop instead tracks new
exact accepted identities or newly verified annotations. Neither stopping rule
proves completeness; unresolved coverage keeps counts provisional.

Configured CLI analyses process independent source segments concurrently, using
separate provider clients for each worker. The default is two targets; set
`--information-concurrency 4` or `OVS_INFORMATION_CONCURRENCY=4` to increase it,
up to eight. After the first primary discovery request succeeds, the first
independent QA request and initial source-focus request may fetch responses ahead
when conservative target-prefix call headroom is available. Their exact prompts
and source context stay unchanged; responses, candidate validation, adaptive
window recovery, IDs and subsequent audits are applied in the original order.
Each independent provider endpoint has the configured physical-request bound,
including foreground and lookahead generation. Jev validation can proceed while
OpenAI generation occupies its slots. Roles using the same provider endpoint
share a conservative cap across accounts, models and rate groups; unidentified
services also share slots. Existing provider rate controls remain independent of
this concurrency cap. Local queue waiting occurs before the
provider operation deadline starts. Lookahead uses private native adapter forks;
non-forkable adapters, serial runs and constrained budgets retain ordinary
admission. Report metadata records submitted, started, consumed and discarded
lookahead requests, queue waits by role and disabled reasons. The prefix headroom guard
is target-local; other concurrent targets can still deplete the global budget.
Started responses left unused after a stop remain audited and keep the report
partial. Permanent provider failures stop dependent dispatch through the existing
shared controls, so their timing can affect which earlier work finishes.

Operations within each target keep their dependency order. Pair
consolidation runs afterward with up to eight requests, each carrying at most
four independently keyed pair decisions. Each question explicitly names its pair
block and both candidate IDs; only that pair's marked original source scopes and
cited reference context authorize a decision. The character budget reduces batch
size when necessary, including individual requests. Set batch size 1 for an
individual comparison baseline. Logical calls are reserved in deterministic
priority order before dispatch, and relations, calls and groups follow that order
regardless of completion order. All workers share the logical-call cap, so target
coverage at cap exhaustion can depend on scheduling. Isolated failed comparisons
remain unexamined while other admitted comparisons can finish.

Every physical send, including retries and all Jev validation, anchor, coverage
and pair requests, passes through one shared process-local controller for its
provider, endpoint, credential, organization/project and limit group. Native
worker forks keep private clients and records while sharing this controller.
Configured analyzers share an explicit `RequestScope`; library callers combining
separate adapters can pass the same scope. The controller does not coordinate
other processes or external account users. A model family sharing a provider
limit needs an explicit common limit group; project headers alone do not provide
cross-group or organization-wide enforcement.

Jev starts at 60 requests/s and 80000 estimated input tokens/s, leaving headroom
against the user organization's 80 requests/s and 100000 input tokens/s allowance.
OpenAI account limits are not assumed. Valid response headers establish local
ceilings at 90% of their published request/token limits, conservatively reconcile
remaining/reset allowances with in-flight reservations, and can only tighten
explicit ceilings. Optional project token headers further tighten the group.
Unknown or absent request-limit headers use a local discovery pace of four
requests/s while allowing concurrent in-flight calls. This fallback is not a
guarantee about unknown account limits. Header learning uses minute windows;
disable it when configuring another OpenAI window duration.

Token predictions use the complete serialized UTF-8 request length plus a
64-token envelope allowance, including questions, criteria, state and output
schema. Provider templates are unknown, so this is not an upper bound or exact
tokenization. Jev also predicts 256 tokens per request and 128 per question for
unknown template overhead; both allowances are configurable. If reported input
usage exceeds its prediction, the shared controller raises future predictions to
125% of that observed ratio. Queued sends and retries use the current multiplier.
Reported excess is charged to the shared window, including for invalid decisions
with valid usage. Late excess feedback adds debt for a full window after arrival;
lower usage never refunds capacity. Actual Jev output usage is reported separately
and does not consume its input-only token window. Records expose base prediction,
multiplier before/after feedback, original reservation, accounted tokens, excess
adjustment and whether usage is reported, unknown or invalid.

OpenAI additionally reserves its explicit `max_output_tokens` ceiling (including
reasoning); incomplete responses are rejected and recorded. A request predicted
larger than a configured or learned token window fails visibly before sending,
without truncating source text. Adjust the output budget, overhead or limit
explicitly when appropriate. Actual provider usage remains separate from admission
predictions. Prediction error can affect already sent requests before feedback
arrives, and neither prediction headroom nor feedback controls other account
consumers; these local controls are not an account-wide usage guarantee.

A temporary 429 publishes a cooldown for both new requests and retries in the
affected controller. Already sent requests finish; recovery is paced gradually
over ten seconds. An OpenAI cooldown does not pause Jev. Isolated timeouts back
off only their operation. Three service-unavailable failures within 30 seconds
pause the affected group for at least five seconds and then ramp its pace.
`Retry-After` seconds and HTTP dates are minimum waits and are never clipped;
`retry-after-ms` is also accepted as a compatibility fallback. Exponential backoff
adds up to 25% jitter (backoff capped at 10 seconds for generation, 30 for
evaluation). If a required wait cannot fit the remaining operation deadline,
the operation fails with `RequestDeadlineError` without an early retry. Admission
and backoff waits are interruptible, and terminal authentication/quota feedback
wakes queued operations. Current OpenAI credit/spend/usage limit errors and model
configuration failures stop new dependent analysis requests. A target-specific
invalid request does not cancel other targets.
The dependent run uses a separate `RunStoppedError` signal to stop retrying other
roles after a terminal failure; the original failure remains in report metadata,
and independent controller scopes are not marked terminal by that run signal.

Retries are bounded by both attempts and a total operation deadline. The existing
120-second generation and 30-second evaluation per-attempt defaults are clamped
to remaining time. A shared cooldown cannot make an intrinsically long generation
succeed within its individual timeout; choose finite timeout/deadline settings
explicitly. OpenAI SDK retries are disabled, including injected real SDK clients.
Already sent sockets retain their transport timeouts and are not forcibly aborted.
Attempt records expose physical sends, retry/error reasons, admission/backoff wait
time and reasons, estimated reservations, exact usage when returned, active
limits and output/deadline settings for operational auditing.
Service classifications and header semantics follow the official
[OpenAI error guide](https://developers.openai.com/api/docs/guides/error-codes),
[OpenAI rate limits](https://developers.openai.com/api/docs/guides/rate-limits)
and [TypeSafe API](https://docs.typesafe.ai/api#handling-rate-limits).

The CLI prints target, operation, provider-attempt, retry, and elapsed-time updates
while analysis runs, plus a ten-second heartbeat during long provider calls. For
`summarize --analyze-information`, it prints the saved report path before selection
starts so the inventory result is visible even while the summary continues.

Library callers can pass `information_analyzer=InformationAnalyzer(generator,
evaluator, config)` to the existing `Summarizer.summarize(...)`. Optional
`information_output_path`, `information_csv_path` and `information_input_path`
control exports and protected dataset identity. An optional
`information_report_observer(report, path)` receives the finished report after
its export and before selection criteria run. With `save_output=False` and no
explicit report path, inspect the immutable `last_information_report` in memory.
The return type remains `Video`, and report state is reset for every invocation.
`InformationAnalyzer(..., progress=observer)` optionally emits frozen
`AnalysisProgress` events; without an observer the library emits no progress.
Observers must return promptly; callback exceptions do not change the analysis.
Injected clients or transports without a safe `fork()` run serially, with actual
concurrency recorded in report metadata. Interrupting a concurrent run cancels
queued targets, prevents further logical calls and retry attempts in active
targets, and waits for already sent bounded requests and client cleanup; it does
not forcibly abort their sockets.

The neutral `AnalysisSnapshot -> InformationReport` contract lives alongside the
analyzer in `core/summarizers/`. Original source provenance and allowed current
stage context are distinct: removed source content cannot be reintroduced solely
through provenance. A future plugin that consumes another module's stage result
is **not implemented**; there is no generic multi-hook framework or stage delta
tracking in this change.

Offline tests use synthetic Portuguese fixtures, mocked generators and evaluators,
and real adapter serializers with fake HTTP transport. They check integration and
invariants, not scientific fidelity, coverage or superiority. Run them with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_information*.py" -v
.\.venv\Scripts\python.exe -m unittest tests.test_typesafe_adapter -v
```

## Choose transcription and topic models

Raw videos can use a local or hosted provider for each stage:

| Stage | Local default | Hosted option | Data sent to a hosted service |
| --- | --- | --- | --- |
| Transcription (STT) | Whisper `base` on this computer | ElevenLabs `scribe_v2` | Audio extracted from each video |
| Topic classification (LLM) | Ollama `gemma2` on this computer | OpenAI `gpt-6-luna` using Responses | Transcript text and the topic-classification prompt |

Video reading, audio extraction, segmentation, HSMVideoSumm selection, and
rendering stay local. OpenAI receives transcript text for topic classification;
ElevenLabs receives audio for transcription. The hosted options need internet
access, the matching API key, and an account with access to the selected model.
Provider use can incur charges. The project does not send the full video file to
either provider.

The OpenAI adapter uses the Responses API and structured output. GPT-6 Luna
supports the Responses API, structured output, and reasoning effort values
`none`, `low`, `medium`, `high`, `xhigh`, and `max`. The adapter omits temperature
to use each model's accepted generation defaults. A blank `OVS_LLM_REASONING_EFFORT` leaves the setting
to the model (GPT-6 Luna currently defaults to `medium`); `none` is an explicit
request to disable reasoning. See the
[GPT-6 Luna model page](https://developers.openai.com/api/docs/models/gpt-6-luna),
[reasoning guide](https://developers.openai.com/api/docs/guides/reasoning), and
[structured outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).
Known model capabilities are checked before requests: [GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra)
and [GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol)
accept `low`, `medium`, `high`, `xhigh`, and `max`; [GPT-5](https://developers.openai.com/api/docs/models/gpt-5)
accepts `minimal`, `low`, `medium`, and `high`. Unrecognized future model IDs are
sent to the service, whose compatibility errors become internal configuration errors.

ElevenLabs uses Scribe v2 for hosted transcription. `pt` selects Portuguese; set
`OVS_STT_LANGUAGE=auto` to let the service detect the language. The local Whisper
adapter also accepts `auto`. It maps ISO aliases such as `por` to `pt`, using
the [ISO language code list](https://www.loc.gov/standards/iso639-2/php/code_list.php),
and validates them against the installed Whisper tokenizer before loading model
weights or audio. Native codes such as `haw` and `yue` remain available when the
installed tokenizer supports them. Run metadata preserves the requested code,
the translated code sent to the provider, and the language reported in its response.
Timed words keep the source separators, so languages and mixed text without
spaces retain their spelling and original word timestamps. Internal transcript
whitespace and punctuation keep their source positions. Leading and trailing
utterance whitespace is trimmed, including residue at transcript edges after
removing audio events. Word-token punctuation
supplements a source gap only when that gap has no punctuation; conflicting
punctuation is rejected. Valid reported model and language fields remain in run
metadata even when the transcript is invalid. The default sentence
boundaries include `.`, `!`, `?`, `。`, `！`, and `？`; custom boundaries remain configurable.
See the [speech-to-text API](https://elevenlabs.io/docs/api-reference/speech-to-text/convert)
and [Scribe language and capability guide](https://elevenlabs.io/docs/overview/capabilities/speech-to-text).

## Configure providers

Copy the example to a root `.env` file, then edit the provider and model values
you want to use:

```powershell
Copy-Item .env.example .env
notepad .env
```

The example selects the local defaults and leaves both API keys blank. Set only
the key for each hosted provider you choose. Do not commit `.env` or put real
keys in `.env.example`.

Settings follow this priority: explicit CLI option, process environment, root
`.env`, then the configured default. Blank optional values are treated as absent;
`OVS_STT_LANGUAGE=auto` explicitly requests language detection. If you choose a
different provider in `.env`, update its model too. Leaving the model unset uses
`gemma2` for Ollama, `gpt-6-luna` for OpenAI, `base` for local Whisper, or
`scribe_v2` for ElevenLabs. A blank `OVS_LLM_BASE_URL` uses the selected provider's
default endpoint (`http://localhost:11434` for Ollama,
`https://api.openai.com/v1` for OpenAI).

The supported settings and defaults are:

| Setting | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | empty | OpenAI credential |
| `ELEVENLABS_API_KEY` | empty | ElevenLabs credential |
| `OVS_LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `OVS_LLM_MODEL` | provider default | `gemma2` or `gpt-6-luna` |
| `OVS_LLM_REASONING_EFFORT` | model default | OpenAI effort; accepted values depend on the model (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`) |
| `OVS_LLM_BASE_URL` | provider default | Ollama/OpenAI API endpoint or a compatible gateway |
| `OVS_LLM_TIMEOUT_SECONDS` | `120` | LLM request timeout |
| `OVS_STT_PROVIDER` | `whisper_local` | `whisper_local` or `elevenlabs` |
| `OVS_STT_MODEL` | provider default | Whisper model or `scribe_v2` |
| `OVS_STT_LANGUAGE` | `pt` | Language code; `auto` enables detection |
| `OVS_STT_TIMEOUT_SECONDS` | `120` | ElevenLabs HTTP and FFmpeg audio-extraction timeout; it does not interrupt synchronous local Whisper inference |
| `OVS_MAX_ATTEMPTS` | `3` | Maximum total attempts for LLM generation and transient hosted STT failures |
| `OVS_THREADS` | `2` | CPU thread budget used by the CLI; an explicit global `--threads` value takes priority |
| `OVS_VISUAL_SCOPE` | `segment` | Visual extraction from each source interval or legacy full source video (`video`); summarize `--visual-scope` takes priority |

For example, use OpenAI with a compatible endpoint by setting
`OVS_LLM_BASE_URL=https://gateway.example/v1`. That endpoint must implement the
Responses API and structured output used by this adapter. A Chat Completions-only
endpoint is not sufficient. Authentication errors, unsupported model names,
missing Responses routes, and rejected structured-output parameters need a
credential, model, or endpoint correction; retries will not fix them. Only
transient remote network, timeout, or rate-limit failures are retried, within
the configured attempt count. Local Whisper inference errors do not retry.
Check a 401/403 for credentials or access, a 404 for the model or route,
a 400 for unsupported request features, and a 429 for rate limits or quota.
`OVS_MAX_ATTEMPTS` caps LLM generation requests, including retries for malformed
or schema-invalid output. SDK-level automatic retries are disabled so this cap
stays predictable. ElevenLabs retries only transient HTTP failures; local
Whisper model-loading and inference errors do not retry.

## Segment raw videos

`segment` accepts either local or hosted providers for each stage. These
PowerShell examples show all four pairings. Set the needed key in `.env` or in
the current PowerShell session before running a hosted provider.

**Ollama and local Whisper** (the defaults):

```powershell
ollama pull gemma2
.\.venv\Scripts\python.exe -m open_video_summary segment `
  --input data/raw/my_dataset `
  --output outputs/my_dataset_segments.json `
  --llm-provider ollama --llm-model gemma2 `
  --stt-provider whisper_local --stt-model base --stt-language pt
```

Install and start the [Ollama server](https://ollama.com/download); the Python
package installs only its client. Whisper downloads its selected model the first
time it transcribes. `tiny` is a smaller local option; `medium` and `large` need
more memory and time.

**OpenAI and local Whisper:**

```powershell
$env:OPENAI_API_KEY = "your-key"
.\.venv\Scripts\python.exe -m open_video_summary segment `
  --input data/raw/my_dataset `
  --output outputs/my_dataset_segments.json `
  --llm-provider openai --llm-model gpt-6-luna --llm-reasoning-effort low `
  --stt-provider whisper_local --stt-model base --stt-language pt
```

**Ollama and ElevenLabs:**

```powershell
$env:ELEVENLABS_API_KEY = "your-key"
.\.venv\Scripts\python.exe -m open_video_summary segment `
  --input data/raw/my_dataset `
  --output outputs/my_dataset_segments.json `
  --llm-provider ollama --llm-model gemma2 `
  --stt-provider elevenlabs --stt-model scribe_v2 --stt-language pt
```

**OpenAI and ElevenLabs:**

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:ELEVENLABS_API_KEY = "your-key"
.\.venv\Scripts\python.exe -m open_video_summary segment `
  --input data/raw/my_dataset `
  --output outputs/my_dataset_segments.json `
  --llm-provider openai --llm-model gpt-6-luna --llm-reasoning-effort low `
  --stt-provider elevenlabs --stt-model scribe_v2 --stt-language pt
```

You can set a provider endpoint and request limits on the command line. Explicit
CLI options override environment settings:

```powershell
.\.venv\Scripts\python.exe -m open_video_summary segment `
  --input data/raw/my_dataset --output outputs/my_dataset_segments.json `
  --llm-provider openai --llm-model gpt-6-luna `
  --llm-base-url https://gateway.example/v1 `
  --stt-provider elevenlabs --stt-model scribe_v2 --stt-language auto `
  --llm-timeout 120 --stt-timeout 180 --max-attempts 3
```

The legacy `--whisper-model` option aliases `--stt-model`, and `--language`
aliases `--stt-language`. Existing local commands such as
`--whisper-model base --llm-model gemma2 --language pt` continue to work.

Each segment run writes a `<output_stem>_run.json` file next to its output. For
example, `outputs/my_dataset_segments.json` gets
`outputs/my_dataset_segments_run.json`. The run record captures requested,
sent, and service-reported provider/model/reasoning/language settings, elapsed
time, SDK and adapter versions, attempt status, and audio duration when known.
It does not include API keys, transcript text, prompts, or audio bytes.

`segment` also requires the FFmpeg executable on `PATH`, regardless of provider
choice. The Python `ffmpeg` package does not install that executable. Check it
with `ffmpeg -version`; on Windows, install it with
`winget install -e --id Gyan.FFmpeg` and open a new PowerShell window.

## More commands

Check the environment, prepare the sample assets, and run the bundled example:

```powershell
.\.venv\Scripts\python.exe -m open_video_summary doctor
.\.venv\Scripts\python.exe -m open_video_summary prepare-demo
.\.venv\Scripts\python.exe -m open_video_summary summarize
```

`prepare-demo` extracts the tracked videos and downloads the original trained
subjectivity classifier. Use `--without-model` to extract only the videos.
`summarize` accepts another already segmented dataset:

```powershell
.\.venv\Scripts\python.exe -m open_video_summary summarize `
  --dataset data/processed/my_dataset.json `
  --output outputs/my_summary.mp4
```

To run from another directory, invoke the clone's Python executable by its path.
Paths passed to the API and CLI are interpreted from the repository root. Video
paths inside JSON files are saved as portable, root-relative paths.

## Data and notebooks

Do not check in `.venv`, downloaded models, caches, generated summaries, or
extracted raw videos. Keep generated results under `outputs/`. The clone has
metadata for five datasets, but only `bebe_real` includes MP4 files. Other videos
are available from the [original dataset directory](https://drive.google.com/drive/folders/1y19ih3j36UqXlWFcgyxNXsNWluE3lky6?usp=drive_link).

Open notebooks with the `.venv/Scripts/python.exe` kernel.
`notebooks/hsmvideosumm.ipynb` prepares the same sample assets and runs the
summary. `notebooks/video-segmenter.ipynb` uses local Whisper and Ollama and
writes new results to `outputs`. `notebooks/llm-evaluations.ipynb` needs its own
evaluation datasets, Ollama models, and, where applicable, Kaggle access.

`requirements-windows.lock` pins the Windows x64 / Python 3.11 CPU environment.
It includes `tensorflow-intel==2.17.1`, required by the Windows TensorFlow wheel,
and `tensorflow-io-gcs-filesystem==0.31.0`, which has a compatible Windows
wheel. The setup script installs `torch==2.6.0+cpu` from the PyTorch CPU index
before installing that lock file. For other environments, use Poetry with a
compatible Python version.
