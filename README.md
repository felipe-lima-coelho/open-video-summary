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

The generator uses the existing Ollama/OpenAI adapters. A separate TypeSafe Jev
evaluator checks support, qualifiers and granularity, audits the original text
for possible omissions, and compares candidate meanings. Set the evaluator key
in the process environment or root `.env`, then use a **new report path**:

```powershell
$env:TYPESAFE_API_KEY = "your-typesafe-key"
.\.venv\Scripts\python.exe -m open_video_summary summarize `
  --dataset data/processed/bebe_real.json `
  --output outputs/bebe_real_summary.mp4 `
  --information-report outputs/bebe_real_information_run01.json `
  --information-csv outputs/bebe_real_information_run01.csv `
  --llm-provider ollama --llm-model gemma2
```

The summary still needs its normal video and classifier assets. Ollama must have
the configured generator model available. Alternatively, set `OPENAI_API_KEY`
and replace the last line with `--llm-provider openai --llm-model gpt-6-luna
--llm-reasoning-effort high`. With either generator, Jev receives the selected
transcript context, candidates and evaluation questions through HTTPS. OpenAI
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
  --llm-provider ollama --llm-model gemma2
```

This command exits nonzero for a partial or failed analysis after saving its
report. Missing `TYPESAFE_API_KEY` creates a failed report before contacting the
generator; it is never interpreted as a successful inventory of zero units.

### Protocol and report interpretation

`contextual-propositions-v1` inventories verbal content **as communicated**, not
the external truth of a claim. A unit preserves entity context, attribution,
negation, modality, numbers, conditions and exceptions. Independent properties
become separate units; a condition remains with its consequent. Compound
candidates flagged by the evaluator are excluded and offered to the bounded
recovery route, so their parts cannot silently add to an already counted compound.

Direct extraction and optional anchored question/answer extraction initially
receive the same source context without seeing each other's output. Questions
and answers are discovery aids; they are not units to add to the count. Every
candidate must carry literal quote offsets, with assertion evidence in its target
segment. Code checks Unicode character slices and existing IDs; timestamps are
copied from source segments, with segment-level resolution, never estimated by a
model. Context citations can resolve a reference but create no occurrence.

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
evidence can be deduplicated in code. Repetitions retain distinct positional source
identities and assertion spans; overlapping evidence from two routes does not add
another occurrence. Ambiguous broad quotes cannot join two known repetitions and
are marked for review.

JSON is the canonical report. It includes the frozen source snapshot, source and
current-input fingerprints, source/current order, selection stage order,
`stage_id="before_introduction"`, candidate validation and reasons, contextual
units, assertion occurrences, context evidence, pair relations, coverage records,
counts by segment/video/whole input and unresolved issues. Calls record requested
and returned models, input/output hashes, usage when available, latency and retry
attempts. Prompt hashes, protocol version and settings are retained without keys.
The optional CSV is an occurrence table; the JSON contains the full evidence.

`completed` means the configured operations finished without recorded pending
issues. It is **not proof of complete extraction or calibrated semantic accuracy**.
`partial` retains accepted units alongside pending work and provisional counts;
`failed` distinguishes unusable analysis from a valid empty result. Only a
completed empty inventory has `counts.valid_zero=true`. Jev's Noul value is a
probability of a yes answer; Choice's selected-option probability and its separate
confidence metric are recorded distinctly. Neither is a measured accuracy rate in
this application. The defaults below need validation on Portuguese annotations.

### Analysis bounds and library contract

CLI settings override the process environment, then root `.env`, then defaults.
The generator keeps the existing `OVS_LLM_*` settings. Jev defaults to the pinned
`jev-1.13.0`, not a moving latest alias; `OVS_JEV_MODEL` or `--jev-model` can
explicitly select another model. The stdlib HTTP adapter implements the official
[TypeSafe API](https://docs.typesafe.ai/api); model version and confidence semantics
are described in [models](https://docs.typesafe.ai/models) and
[confidence](https://docs.typesafe.ai/confidence).

| CLI option | Environment setting | Default |
| --- | --- | --- |
| `--no-information-qa` | `OVS_INFORMATION_QA` (`true`/`false`) | QA enabled |
| `--information-max-calls` | `OVS_INFORMATION_MAX_CALLS` | 256 logical calls |
| `--information-rounds` | `OVS_INFORMATION_ROUNDS` | 2 recovery rounds per target |
| `--information-max-pairs` | `OVS_INFORMATION_MAX_PAIRS` | 160 semantic pair comparisons |
| `--information-max-candidates` | `OVS_INFORMATION_MAX_CANDIDATES` | 16 per target and extraction route |
| `--information-context-chars` | `OVS_INFORMATION_CONTEXT_CHARS` | 24000 characters per prompt, or evaluation context plus questions |
| `--information-context-segments` | `OVS_INFORMATION_CONTEXT_SEGMENTS` | Up to 4 additional current segments from the same video |
| `--information-acceptance` | `OVS_INFORMATION_ACCEPTANCE` | 0.85 for each support/qualifier signal and atomicity |
| `--information-gap-threshold` | `OVS_INFORMATION_GAP_THRESHOLD` | 0.65 yes signals a gap; at most 0.35 means no gap signaled |
| `--information-equivalence` | `OVS_INFORMATION_EQUIVALENCE` | 0.90 selected relation probability |
| `--jev-model` | `OVS_JEV_MODEL` | `jev-1.13.0` |
| `--jev-base-url` | `OVS_JEV_BASE_URL` | `https://api.typesafe.ai` |
| `--jev-timeout` | `OVS_JEV_TIMEOUT_SECONDS` | 30 seconds per HTTP attempt |
| `--jev-max-attempts` | `OVS_JEV_MAX_ATTEMPTS` | 2 attempts, at most 5 |

The call budget counts logical generation/evaluation invocations. Bounded provider
retries add physical requests: the generator uses `--max-attempts` /
`OVS_MAX_ATTEMPTS` (default 3), while Jev uses its separate attempt limit. Provider
preflight/model discovery is separate from that budget. Retry records and token
usage, when supplied by the service, allow operational cost accounting; no fixed
currency cost is inferred. Large original targets are not truncated to fit a
request: they become pending work. Pair-budget exhaustion prevents further
automatic merges and marks unique-unit counts provisional.

Library callers can pass `information_analyzer=InformationAnalyzer(generator,
evaluator, config)` to the existing `Summarizer.summarize(...)`. Optional
`information_output_path`, `information_csv_path` and `information_input_path`
control exports and protected dataset identity. With `save_output=False` and no
explicit report path, inspect the immutable `last_information_report` in memory.
The return type remains `Video`, and report state is reset for every invocation.

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
