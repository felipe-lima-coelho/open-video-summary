"""Opt-in information-analysis settings, loaded only for an analysis request."""

import math
import os
from dataclasses import dataclass
from pathlib import Path

from open_video_summary.errors import ConfigurationError
from open_video_summary.utils.config import PROJECT_DIR


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

    def __post_init__(self):
        if not isinstance(self.qa_enabled, bool):
            raise ConfigurationError("Information QA must be a boolean.")
        limits = {
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


def configured_information_analyzer(overrides=None, *, environ=None, env_file=None):
    """Construct the existing generator plus a separately typed Jev evaluator."""
    from dotenv import dotenv_values
    from open_video_summary.adapters.factory import create_llm
    from open_video_summary.adapters.typesafe import TypeSafeConfig, TypeSafeEvaluator
    from open_video_summary.core.summarizers.information_analysis import (
        InformationAnalyzer,
    )
    from open_video_summary.utils.providers import load_provider_config

    supplied = overrides or {}
    path = Path(env_file) if env_file is not None else PROJECT_DIR / ".env"
    values = dict(dotenv_values(path, interpolate=False)) if path.is_file() else {}
    values.update(os.environ if environ is None else environ)

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
    evaluator_config = TypeSafeConfig(
        api_key=values.get("TYPESAFE_API_KEY") or None,
        model=get("jev_model", "OVS_JEV_MODEL", "jev-1.13.0"),
        base_url=get("jev_base_url", "OVS_JEV_BASE_URL", "https://api.typesafe.ai"),
        timeout_seconds=get("jev_timeout", "OVS_JEV_TIMEOUT_SECONDS", 30.0, float),
        max_attempts=get("jev_max_attempts", "OVS_JEV_MAX_ATTEMPTS", 2, int),
    )
    generator = create_llm(
        load_provider_config(supplied, environ=values, env_file=path).llm
    )
    return InformationAnalyzer(generator, TypeSafeEvaluator(evaluator_config), settings)
