# Information-analysis plugin

Information analysis remains optional and observational. The initial inventory
runs before Introduction, copies all source transcript segments into an immutable
snapshot, and adds no summary decisions. Schema 11 adds `questions` and
`stage_reports`; all original candidates, units, evidence, counts and reliability
diagnostics remain available. Summary order, transcripts, source identities,
timestamps, topics and selection actions are unchanged.

## Enable and inject

The existing CLI switches enable the inventory and plugin together:

```powershell
.\.venv\Scripts\python.exe -m open_video_summary summarize `
  --dataset data/processed/bebe_real.json `
  --output outputs/summary.mp4 `
  --information-report outputs/information/summary_run01.json
```

Existing provider/evaluator configuration applies. No stage analysis or provider
configuration is loaded when no analyzer is attached. Text-only
`analyze-information` still produces the initial inventory and questions without
running selection criteria.

Library injection accepts the existing analyzer or its explicit plugin wrapper:

```python
from open_video_summary.core.selection_criteria.introduction import Introduction
from open_video_summary.core.selection_criteria.quality import QualityPick
from open_video_summary.core.summarizers.base import Summarizer
from open_video_summary.core.summarizers.information_config import configured_information_analyzer
from open_video_summary.core.summarizers.information_plugin import InformationAnalysisPlugin

analyzer = configured_information_analyzer()
plugin = InformationAnalysisPlugin(analyzer)
summarizer = Summarizer(criteria, information_analyzer=plugin)
video = summarizer.summarize(videos, save_output=False)
report = summarizer.last_information_report

# Existing summarize(..., information_analyzer=analyzer) remains supported.
# Override a stage's evaluator while keeping the shared initial question inventory.
intro = Introduction(information_analyzer=analyzer)
# Explicit opt-out overrides the pipeline default.
quality = QualityPick("ContentBasedRedundancy", information_analyzer=False)

# Direct module evaluation creates a fresh initial inventory and stage context.
intro.evaluate(handler)
module_report = intro.last_information_report
```

Direct calls need source videos set on the handler to obtain positional evidence.
Per-module injection outside an enabled pipeline applies to that module. A
pipeline-level analyzer attaches to all its existing and future modules by default,
including modules that do not inherit `SelectionCriteria`. A `summarize` call with
`information_analyzer=None` or `False` disables the pipeline-level default;
explicit individual injections remain individual injections. A constructor's
unspecified analyzer inherits the pipeline setting; explicitly passing `None` or
`False` excludes that module.

Subclasses inherit the hook automatically, including constructors that omit
`super().__init__`. Existing positional arguments and additional `evaluate`
arguments are forwarded unchanged. Inherited or `super().evaluate()` calls do not
run the observer twice. A custom criterion that retains only selected candidates
can override `information_removed_segments(handler)` to return its effective
removed segments after successful evaluation. That hook reports outcomes and must
not alter handler actions. The default hook reports no additional removals.

## Exclusions and limits

Set exact names in the process environment or root `.env`:

```dotenv
OVS_INFORMATION_EXCLUDE_MODULES=Introduction,QualityPick
OVS_INFORMATION_STAGE_MAX_CALLS=128
```

Names match the criterion's `name`, its class name, or its fully qualified Python
class name, for example
`open_video_summary.core.selection_criteria.quality.QualityPick`. Matching is case
sensitive; comma-separated whitespace is stripped; no substring or wildcard
matching occurs. Exclusions suppress that module's stage report and evaluator
calls. The initial inventory remains enabled. Effective removal outcomes still
describe availability for subsequent stages in the same run, including an excluded
QualityPick's non-winners. They never transfer to another run or module instance.

`InformationAnalysisConfig(excluded_modules=("Introduction",), stage_max_calls=32)`
and `InformationAnalysisPlugin(analyzer, excluded_modules=("QualityPick",))` offer
library exclusions. Environment exclusions always apply. The stage cap is separate
from the initial `max_calls` cap. Each stage uses at most `stage_max_calls` logical
evaluator requests; zero leaves matches pending without sending a request. Existing
character limits, bounded adapter retries, provider request/token limits and
timeouts also apply. No new generation requests run after selection modules.

## Questions and answers

`questions` retains accepted QA discovery pairs with their `qa_anchor` and
`qa_consistency` probabilities, unit IDs, candidate IDs, occurrence IDs and exact
assertion/reference evidence. Different answers or units are never merged merely
because question text matches. Multiple queries for one unit add no information.

Direct-only units, including `OVS_INFORMATION_QA=false`, receive a query labeled
`origin="templated_proposition"` that quotes the complete accepted proposition.
Its answer keeps source-language wording and certainty. This fallback does not
invent a natural question, translate the proposition, add model requests or enable
independent QA discovery. Accepted units without usable occurrences are listed in
`metadata.question_inventory.uncovered_unit_ids`.

Each stage compares **every initial question with every newly removed source
transcript**, including source occurrences missed by the initial inventory.
`metadata.planned_comparisons` records the full question/source product. Budget,
context-limit and provider failures leave explicit `not_evaluated` pairs and a
`pending_comparisons` count; zero budget leaves the whole product pending without
evaluator preflight or provider requests.

Each pair has a stage-local `occurrence_id` identifying its support candidate,
`origin="stage_question_source"`, and exact literal full-transcript assertion
evidence at `[0, len(content))` with `evidence_granularity="source_segment"`. These
are source-segment support records, not localized answer spans or new initial
inventory occurrences. `initial_occurrence_ids` retains any corresponding initial
anchors as optional provenance and never determines which pairs are checked.
Quotes, source identities, offsets and segment timestamps must match the frozen
source. Multiple queries/source matches do not increment initial inventory counts.

The evaluator sees the discarded target and up to `context_segments` nearby
discarded segments from the same video as reference context. Only target assertion
wording can support the answer; reference context can resolve references but cannot
supply a new assertion. The proposed baseline answer is a hypothesis to validate,
never evidence for itself. Original provenance is not permission to import a
retained or previously removed reference into that scope. Missing grounding must
remain unsupported or uncertain; no substituted context or cropped transcript is
used. Whole-segment evidence does not itself establish semantic support.

Matches retain raw probabilities for support, governing conditions/exceptions,
negation, quantities, modality, attribution, question answerability and complete
answer consistency. Every gate must meet `acceptance_threshold` to obtain
`state="answerable"` and a supported `answer`. A decisive negative gate yields
`not_answerable`; gray or missing signals remain uncertain or pending. In
particular, a contradiction or loss of a source's uncertainty cannot satisfy a
baseline question merely through topical similarity. These are model probability
signals, not calibrated accuracy or proof of external truth.

The fixed question set still depends on the initial inventory; this pass does not
discover new question meanings or units. Pending pairs, bounded discarded reference
context and probabilistic decisions prevent a complete-recall claim. Retained
support diagnostics use known initial occurrences and do not reanalyze retained
transcripts. Initial partial results, unknown equivalence and provisional
granularity stay visible.

## Removal and potential loss

A module reports only its new effective removals at its successful stage boundary.
Ordinary discards use the current handler state, cross-checked against new action
history. A segment discarded then output or re-included in that same stage is
retained. `pick`, grouping, ordering and moving selected segments alone remove no
information. Equal-valued source segments retain their distinct positional IDs:
`explicit_discard_action_ids` records exact acted-on source objects, while
`discarded_segment_ids` records all positions effectively excluded under the
handler's existing value-equality semantics.

QualityPick additionally reports the union of evaluated cluster candidates minus
the union of all actual stage winners. Winning any overlapping cluster keeps a
segment. Previously included/output segments that remain guaranteed are retained;
previous removals are not new losses. `metadata.outcome_removed_segment_ids`
distinguishes these ranked-out outcomes from explicit handler discard actions.
This observation works with summary auditing disabled and does not change the
existing visual scores, ties, chosen winners or decision log.

`loss.removed_unit_ids` includes initial semantic units with an inventoried
assertion occurrence removed **or a newly supported discarded question/source
match**, even if that source occurrence was missed initially. Repeated occurrences,
source aliases and multiple questions contribute once per baseline unit. Initial
inventory counts stay unchanged. A remaining known initial occurrence, with all
its required context still available, places its unit in `retained_unit_ids`,
preventing a potential-loss claim.
`potentially_lost_unit_ids` contains units with a supported discarded answer and no
such remaining occurrence. Failed/pending/gray matches on any discarded source
can widen the provisional risk bound even without an initial occurrence there;
context-only impacts
remain in `unresolved_unit_ids` and `context_affected_unit_ids`.

`potentially_lost_unique_units` is the nominal estimate for those known initial
groups. `potential_loss_upper_bound` also includes unresolved known units;
`counts_provisional` preserves initial inventory and stage uncertainty. Unresolved
semantic duplicates or granularity alternatives can make these numbers overcount;
they are not an upper bound on uncatalogued source information. The lower bound is
zero and `definitely_lost_unique_units` is null because complete inventory coverage
is never proven. These diagnostics describe a module boundary. Later custom
criteria can restore segments, and this report makes no final-summary loss claim.

## Persistence and failures

The initial JSON is saved before selection and remains available through the
existing `information_report_observer(report, path)` callback. After each analyzed
stage, its immutable appendix is added to `stage_reports`, and
`Summarizer.last_information_report` is updated. Direct modules expose the same
combined structure in `last_information_report`. CSV exports describe the initial
inventory.

Initial creation refuses existing destinations and protected dataset/video/summary
aliases. A same-run stage append validates the exact saved previous content,
unchanged complete initial inventory, and append-only stage history, then publishes
the complete JSON atomically. Replaced/edited or multiply linked files are
preserved. Report files stay under `outputs/`. Write failure is visible in
`last_information_error` and the stage's `stage_report_write_failed` issue; a fresh
failure artifact under `outputs/information/` is attempted. In-memory results
remain available even if that fallback cannot be written.

Analyzer failures stay in the separate observational report. Selection exceptions
keep their original type and propagate normally; an observer cannot retry a
criterion, roll back its decisions or mask its exception. All run context and
budgets are fresh for sequential calls. Offline tests cover these contracts with
synthetic transcripts, fake typed decisions and mocked visual features; no live
LLM scientific evaluation is implied.
