"""Opt-in information-analysis settings, loaded only for an analysis request."""

import math
from dataclasses import dataclass

from open_video_summary.errors import ConfigurationError


@dataclass(frozen=True)
class InformationAnalysisConfig:
    qa_enabled: bool = True
    max_calls: int = 256
    max_coverage_rounds: int = 2
    max_pair_comparisons: int = 160
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
        limits = {
            "concurrency": (1, 8),
            "max_calls": (1, 10000),
            "max_coverage_rounds": (0, 10),
            "max_pair_comparisons": (0, 10000),
            "max_candidates_per_route": (1, 256),
            "max_context_chars": (4000, 200000),
            "context_segments": (0, 64),
        }
        for name, (minimum, maximum) in limits.items():
            value = getattr(self, name)
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

    settings = InformationAnalysisConfig(
        qa_enabled=get("information_qa", "OVS_INFORMATION_QA", True, boolean),
        concurrency=get("information_concurrency", "OVS_INFORMATION_CONCURRENCY", 2, int),
        max_calls=get("information_max_calls", "OVS_INFORMATION_MAX_CALLS", 256, int),
        max_coverage_rounds=get("information_rounds", "OVS_INFORMATION_ROUNDS", 2, int),
        max_pair_comparisons=get(
            "information_max_pairs", "OVS_INFORMATION_MAX_PAIRS", 160, int
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
    return InformationAnalyzer(generator, create_evaluator(evaluator_config), settings, progress=progress)
