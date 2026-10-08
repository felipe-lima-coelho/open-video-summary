"""Opt-in information-analysis settings, loaded only for an analysis request."""

import math
from dataclasses import dataclass

from open_video_summary.errors import ConfigurationError


@dataclass(frozen=True)
class InformationAnalysisConfig:
    qa_enabled: bool = True
    experimental_mode: str | None = None
    max_granularity_checks: int = 8
    max_coverage_foci: int = 16
    max_gap_repairs: int = 2
    max_calls: int = 256
    max_coverage_rounds: int = 2
    max_literal_repairs: int = 4
    max_pair_comparisons: int | None = None
    pair_concurrency: int = 8
    pair_batch_size: int = 4
    max_relation_adjudications: int = 256
    direct_window_chars: int = 240
    max_direct_windows: int = 16
    qa_window_chars: int = 240
    max_qa_windows: int = 16
    max_candidates_per_route: int = 16
    max_context_chars: int = 24000
    context_segments: int = 4
    acceptance_threshold: float = 0.85
    gap_threshold: float = 0.65
    equivalence_threshold: float = 0.90
    concurrency: int = 2

    def __post_init__(self):
        if not isinstance(self.qa_enabled, bool):
            raise ConfigurationError("Information QA must be a boolean.")
        if self.experimental_mode not in {None, "direct", "qa", "hybrid", "hybrid_coverage"}:
            raise ConfigurationError("Information mode must be direct, qa, hybrid or hybrid_coverage.")
        limits = {
            "max_granularity_checks": (0, 64),
            "max_coverage_foci": (0, 256),
            "max_gap_repairs": (0, 10),
            "concurrency": (1, 8),
            "pair_concurrency": (1, 8),
            "pair_batch_size": (1, 8),
            "max_relation_adjudications": (0, 10000),
            "direct_window_chars": (80, 4000),
            "max_direct_windows": (1, 64),
            "qa_window_chars": (80, 4000),
            "max_qa_windows": (1, 64),
            "max_calls": (1, 10000),
            "max_coverage_rounds": (0, 10),
            "max_literal_repairs": (0, 64),
            "max_pair_comparisons": (0, 10000),
            "max_candidates_per_route": (1, 256),
            "max_context_chars": (4000, 200000),
            "context_segments": (0, 64),
        }
        for name, (minimum, maximum) in limits.items():
            value = getattr(self, name)
            if name == "max_pair_comparisons" and value is None:
                continue
            if type(value) is not int or not minimum <= value <= maximum:
                raise ConfigurationError(
                    f"Information {name} must be between {minimum} and {maximum}."
                )
        for name in ("acceptance_threshold", "gap_threshold", "equivalence_threshold"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.5 < value < 1
            ):
                raise ConfigurationError(
                    f"Information {name} must be finite and between 0.5 and 1."
                )

    @property
    def discovery_routes(self):
        if self.experimental_mode == "direct":
            return ("direct",)
        if self.experimental_mode == "qa":
            return ("qa",)
        if self.experimental_mode is not None:
            return ("direct", "qa")
        return ("direct", "qa") if self.qa_enabled else ("direct",)

    @property
    def coverage_enabled(self):
        return self.experimental_mode in {None, "hybrid_coverage"}

    @property
    def effective_mode(self):
        return self.experimental_mode or ("hybrid_coverage" if self.qa_enabled else "direct_coverage")


def configured_information_analyzer(overrides=None, *, environ=None, env_file=None, progress=None):
    """Construct independently selected generation and typed evaluation roles."""
    from open_video_summary.adapters.factory import create_evaluator, create_llm
    from open_video_summary.core.summarizers.information_analysis import (
        InformationAnalyzer,
    )
    from open_video_summary.utils.providers import (
        load_environment_values,
        load_evaluator_config,
        load_provider_config,
    )

    supplied = overrides or {}
    values = load_environment_values(environ=environ, env_file=env_file)

    def get(argument, variable, default, convert=str):
        value = supplied.get(argument)
        if value is None:
            value = values.get(variable)
        if value is None or str(value).strip() == "":
            return default
        try:
            return convert(value)
        except (ValueError, TypeError):
            raise ConfigurationError(
                f"{variable} has an invalid information-analysis value."
            ) from None

    def boolean(value):
        if isinstance(value, bool):
            return value
        lowered = str(value).lower()
        if lowered not in {"true", "false", "1", "0"}:
            raise ValueError()
        return lowered in {"true", "1"}

    def pair_limit(value):
        return None if str(value).strip().lower() == "auto" else int(value)

    settings = InformationAnalysisConfig(
        qa_enabled=get("information_qa", "OVS_INFORMATION_QA", True, boolean),
        experimental_mode=get("information_mode", "OVS_INFORMATION_MODE", None),
        max_granularity_checks=get("information_granularity_checks", "OVS_INFORMATION_GRANULARITY_CHECKS", 8, int),
        max_coverage_foci=get("information_coverage_foci", "OVS_INFORMATION_COVERAGE_FOCI", 16, int),
        max_gap_repairs=get("information_gap_repairs", "OVS_INFORMATION_GAP_REPAIRS", 2, int),
        concurrency=get("information_concurrency", "OVS_INFORMATION_CONCURRENCY", 2, int),
        pair_concurrency=get("information_pair_concurrency", "OVS_INFORMATION_PAIR_CONCURRENCY", 8, int),
        pair_batch_size=get("information_pair_batch_size", "OVS_INFORMATION_PAIR_BATCH_SIZE", 4, int),
        max_relation_adjudications=get("information_relation_adjudications", "OVS_INFORMATION_RELATION_ADJUDICATIONS", 256, int),
        direct_window_chars=get("information_direct_window_chars", "OVS_INFORMATION_DIRECT_WINDOW_CHARS", 240, int),
        max_direct_windows=get("information_max_direct_windows", "OVS_INFORMATION_MAX_DIRECT_WINDOWS", 16, int),
        qa_window_chars=get("information_qa_window_chars", "OVS_INFORMATION_QA_WINDOW_CHARS", 240, int),
        max_qa_windows=get("information_max_qa_windows", "OVS_INFORMATION_MAX_QA_WINDOWS", 16, int),
        max_calls=get("information_max_calls", "OVS_INFORMATION_MAX_CALLS", 256, int),
        max_coverage_rounds=get("information_rounds", "OVS_INFORMATION_ROUNDS", 2, int),
        max_literal_repairs=get("information_literal_repairs", "OVS_INFORMATION_LITERAL_REPAIRS", 4, int),
        max_pair_comparisons=get(
            "information_max_pairs", "OVS_INFORMATION_MAX_PAIRS", None, pair_limit
        ),
        max_candidates_per_route=get(
            "information_max_candidates", "OVS_INFORMATION_MAX_CANDIDATES", 16, int
        ),
        max_context_chars=get(
            "information_context_chars", "OVS_INFORMATION_CONTEXT_CHARS", 24000, int
        ),
        context_segments=get(
            "information_context_segments", "OVS_INFORMATION_CONTEXT_SEGMENTS", 4, int
        ),
        acceptance_threshold=get(
            "information_acceptance", "OVS_INFORMATION_ACCEPTANCE", 0.85, float
        ),
        gap_threshold=get(
            "information_gap_threshold", "OVS_INFORMATION_GAP_THRESHOLD", 0.65, float
        ),
        equivalence_threshold=get(
            "information_equivalence", "OVS_INFORMATION_EQUIVALENCE", 0.90, float
        ),
    )
    evaluator_config = load_evaluator_config(
        supplied, environ=values, env_file=env_file
    )
    generator = create_llm(
        load_provider_config(supplied, environ=values, env_file=env_file).llm
    )
    evaluator = create_evaluator(evaluator_config)
    from open_video_summary.utils.request_control import RequestScope
    scope = RequestScope()
    for adapter in (generator, evaluator):
        if callable(getattr(adapter, "set_request_scope", None)):
            adapter.set_request_scope(scope)
    return InformationAnalyzer(generator, evaluator, settings, progress=progress)
