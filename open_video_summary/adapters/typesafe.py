"""HTTP adapter for TypeSafe's legacy non-generative evaluator API."""

from dataclasses import dataclass, field
from collections.abc import Mapping
import math

from open_video_summary.adapters.evaluation_http import (
    HTTPEvaluator, HTTPEvaluatorConfig, Transport, TransportResponse,
    _stdlib_transport, _valid_probability, _read_usage,
    _PROBABILITY_SUM_TOLERANCE,
)
from open_video_summary.contracts import (
    ChoiceResult, EvaluationMetadata, EvaluationResult, NoulResult,
)
from open_video_summary.errors import InvalidResponseError

DEFAULT_TYPESAFE_MODEL = "jev-1.13.0"
_ENDPOINT_PATH = "/v1/systemone"


@dataclass(frozen=True)
class TypeSafeConfig(HTTPEvaluatorConfig):
    """Pinned TypeSafe defaults; explicit model and HTTP settings are supported."""

    provider_name = "TypeSafe"
    model: str = DEFAULT_TYPESAFE_MODEL
    base_url: str = "https://api.typesafe.ai"
    request_limit: float = 60.0
    token_limit: int = 80000
    rate_window_seconds: float = 1.0
    provider: str = field(default="typesafe", init=False)


class TypeSafeEvaluator(HTTPEvaluator):
    """Evaluate boolean and choice questions with TypeSafe System One."""

    provider_name = "TypeSafe"

    def _default_transport(self):
        return _stdlib_transport

    def _request_url(self):
        return self.config.base_url.rstrip("/") + _ENDPOINT_PATH

    def _request_payload(self, context, questions):
        return {"state": context, "model": self.config.model, "questions": questions}

    def _parse_response(
        self,
        response: TransportResponse,
        noul_ids: tuple[str, ...],
        choice_options: Mapping[str, tuple[str, ...]],
        *, document=None,
    ) -> tuple[
        dict[str, NoulResult | ChoiceResult], str | None, int | None, int | None
    ]:
        if document is None:
            document = self._read_document(response)
        answers = document.get("answers")
        expected_ids = set(noul_ids).union(choice_options)
        if not isinstance(answers, dict) or set(answers) != expected_ids:
            raise InvalidResponseError(
                "TypeSafe returned missing or unexpected question IDs."
            )

        normalized: dict[str, NoulResult | ChoiceResult] = {}
        noul_set = set(noul_ids)
        for identifier, answer in answers.items():
            if not isinstance(answer, dict):
                raise InvalidResponseError("TypeSafe returned an invalid answer.")
            if identifier in noul_set:
                value = answer.get("noul")
                if answer.get("type") != "noul" or not _valid_probability(value):
                    raise InvalidResponseError(
                        "TypeSafe returned an invalid boolean score."
                    )
                normalized[identifier] = NoulResult(identifier, float(value))
                continue

            options = choice_options[identifier]
            selected = answer.get("choice")
            probabilities = answer.get("probabilities")
            confidence = answer.get("confidence")
            if answer.get("type") != "choice" or selected not in options:
                raise InvalidResponseError(
                    "TypeSafe selected an unknown choice option."
                )
            if not isinstance(probabilities, dict) or set(probabilities) != set(
                options
            ):
                raise InvalidResponseError(
                    "TypeSafe returned invalid choice probabilities."
                )
            if any(not _valid_probability(probabilities[key]) for key in options):
                raise InvalidResponseError(
                    "TypeSafe returned invalid choice probabilities."
                )
            total = sum(float(probabilities[key]) for key in options)
            if not math.isclose(
                total, 1.0, rel_tol=0.0, abs_tol=_PROBABILITY_SUM_TOLERANCE
            ):
                raise InvalidResponseError(
                    "TypeSafe choice probabilities do not sum to one."
                )
            if confidence is not None and not _valid_probability(confidence):
                raise InvalidResponseError(
                    "TypeSafe returned invalid choice confidence."
                )
            normalized[identifier] = ChoiceResult(
                identifier,
                selected,
                tuple((key, float(probabilities[key])) for key in options),
                None if confidence is None else float(confidence),
            )

        returned_model = document.get("model")
        if returned_model is not None and (
            not isinstance(returned_model, str) or not returned_model.strip()
        ):
            raise InvalidResponseError("TypeSafe returned an invalid model identifier.")
        input_tokens, output_tokens = _read_usage(document.get("usage"))
        return normalized, returned_model, input_tokens, output_tokens
