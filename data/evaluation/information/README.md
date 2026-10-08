# Information reference inventory

This directory contains a source-anchored reference inventory for the 16 transcript segments in `data/processed/google_huawei_segments.json`. It supports bounded evaluation of proposition discovery, contextual fidelity, granularity, recurring occurrences, and omissions. The current draft has 53 annotated unit records and 55 source occurrences; 14 parent sentences are linked to their components and excluded from the unit count. These are protocol annotation counts, not a measured or absolute count of information. The fixture is agent-produced and awaits independent review; it is not human gold data, a calibrated benchmark, or a claim of exhaustive scientific truth.

## Annotation protocol

The inventory follows `contextual-propositions-v2`. A unit is one independently assessable proposition expressed by the transcript, contextualized enough to identify its subject and preserve its communicated scope. An annotation records the speaker's claim, not whether the claim is true in the world.

- Split coordinated or adjacent assertions when each can be assessed independently. Record the source sentence as a decomposition parent, link the component units, and count only those components.
- Keep one conditional rule together with its condition and consequent. Do not promote a condition, reason, purpose, or hypothetical event into an asserted event.
- Preserve attribution, negation, modality, quantities, time, scope, and qualifying circumstances. Treat reported allegations and opinions as attributed claims.
- Resolve a reference only from cited source context. When identity remains unclear, retain the bounded proposition with a placeholder and record the ambiguity; do not guess a named entity.
- Group occurrences across sources only when the full proposition and its qualifiers match. Similar topic, shared entity, or related wording is not sufficient.
- Cite exact spans using zero-based Unicode character offsets in the source segment. Assertion evidence identifies the occurrence; context evidence must preserve any governing subject, antecedent, attribution, time, or other qualifier needed to support its complete meaning. A short assertion span is acceptable only with that context. Parent evidence does not implicitly supply uncited context to a component.
- Keep source-content decomposition and repeated occurrences as annotations, not additional information units. A parent sentence is not counted alongside its component units.

## Provenance and limits

`google_huawei_reference_v1.json` embeds the source snapshot and records both its SHA-256 file digest and the analysis snapshot fingerprint. Its scope is exactly the 16 transcript segments (4 from `jornal_da_band`, 6 from `jornal_nacional`, and 6 from `sbt_brasil`). The annotations were prepared directly from that source file without consulting generated inventories or model outputs.

The original agent annotation has undergone source-context remediation by a separate agent after two independent reviews identified fidelity defects. The current revision awaits a fresh independent review; prior review findings are not approval of this revision. The remediation re-read all 53 unit records and 55 occurrences against all 16 original segments without consulting generated inventories or baseline predictions. `coverage_audit` records that source pass, not a scientific completeness result.

The seven-entry `gaps` ledger records unresolved references and identity limits. Two units (`gh-u015` and `gh-u016`) have unresolved referents and are explicitly non-scorable for strict unit alignment, leaving 51 eligible records. The generic claims in `gh-u011`, `gh-u028`, `gh-u035`, `gh-u041`, and `gh-u048` remain scorable, but the unnamed entities, actor of the 5G-focus aside, evaluation target, manufacturer identity, and accusation antecedent are not. The Samsung comparison in `gh-u053` is separate from Huawei's repeated second-place rank in `gh-u002`; only two sources explicitly identify Huawei as second-largest. The v2:s1 rank phrase remains inside the attributed claim in `gh-u041` and is not a separate Huawei-rank occurrence.

The remediation preserves governing time in the proposition and its evidence (`gh-u020`, `gh-u039`), explicitly cites the missing Huawei/Google/phone subjects (`gh-u053`, `gh-u024`, `gh-u037`), restores decision antecedents and attribution, and retains the interviewee's stated reasons without guessing the target of his evaluation. The limited update permission does not become a complete repeal of the business prohibition. Huawei's platform goal remains an attributed purpose, not an achieved result. Annotation explanations stay in metadata; decomposition parents and source context remain uncounted.

An empty list of known missing units is not evidence that the transcript or annotation is complete. For evaluation, omissions should be computed as unmatched eligible reference units. Do not report these results as calibrated accuracy or as a definitive information count.

The transcript is Portuguese. Evidence preserves the source text exactly, including its Unicode accents and punctuation; comparisons must use the recorded source offsets rather than re-encoding displayed terminal output.

No held-out human-reviewed contrast split is claimed. The inventory contains no provider outputs, model decisions, or fabricated reviewer decisions. A reviewer can independently inspect the source spans and revise the annotations before the fixture is used for calibration or scientific claims.

## Synthetic protocol checks

`synthetic_contrast_cases_v1.json` is a separate, agent-authored fixture with three Portuguese source segments, five units, and two non-counted decomposition parents. It contrasts independent actions, a condition-bound negated rule, an attributed possibility, and a statement that a causal link was not confirmed. Its embedded source manifest has its own SHA-256 digest and snapshot fingerprint. Timestamps are placeholders because these are text-only examples.

These examples check protocol behavior only. They are not held-out data, a statistically independent benchmark, or evidence of performance on real transcripts, and their counts must not be pooled with the Google/Huawei corpus.
