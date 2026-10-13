"""OpenAI Decisions REST adapter for probability-based transcript evaluation."""

import json
import math
from dataclasses import dataclass, field

from open_video_summary.adapters.evaluation_http import (
    HTTPEvaluator, HTTPEvaluatorConfig, _read_usage, _valid_probability,
    _PROBABILITY_SUM_TOLERANCE,
)
from open_video_summary.contracts import ChoiceResult, NoulResult
from open_video_summary.errors import (
    ConfigurationError, InvalidResponseError, ProviderConfigurationError,
)

DEFAULT_DECISIONS_MODEL = "gpt-6-luna"


@dataclass(frozen=True)
class DecisionsConfig(HTTPEvaluatorConfig):
    """Independent OpenAI evaluator settings; model identifiers are configurable."""

    provider_name = "OpenAI Decisions"
    model: str = DEFAULT_DECISIONS_MODEL
    base_url: str = "https://api.openai.com/v1"
    learn_rate_limits: bool = True
    organization: str | None = None
    project: str | None = None
    provider: str = field(default="openai", init=False)

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.learn_rate_limits, bool):
            raise ConfigurationError("OpenAI Decisions learn_rate_limits must be a boolean.")
        for name in ("organization", "project"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip() or value != value.strip()
                or any(ord(character) < 32 or ord(character) == 127 for character in value)
            ):
                raise ConfigurationError(f"OpenAI Decisions {name} must be a nonempty header value.")


class OpenAIDecisionsEvaluator(HTTPEvaluator):
    """Map domain booleans and choices to documented typed Decisions choices.

    Boolean questions use choices with literal ``true``/``false`` values. The
    domain receives P(true), preserving its existing probability thresholds.
    No generative response, predicate threshold, or textual confidence is used.
    """

    provider_name = "OpenAI Decisions"

    def _request_url(self):
        return self.config.base_url.rstrip("/") + "/decisions"

    def _request_headers(self):
        headers = super()._request_headers()
        if self.config.organization is not None:
            headers["OpenAI-Organization"] = self.config.organization
        if self.config.project is not None:
            headers["OpenAI-Project"] = self.config.project
        return headers

    def _build_questions(self, context, noul, choice):
        questions, noul_ids, choice_options = super()._build_questions(context, noul, choice)
        if any(len(options) < 2 for options in choice_options.values()):
            raise ValueError("OpenAI Decisions choice questions require 2 to 255 unique strings.")
        return questions, noul_ids, choice_options

    def _request_payload(self, context, questions):
        wire_questions = []
        for identifier, question in questions.items():
            if question["type"] == "noul":
                choices = [
                    {"value": True, "description": "The answer to the question is yes."},
                    {"value": False, "description": "The answer to the question is no."},
                ]
            else:
                choices = [{"value": value, "description": description}
                           for value, description in question["criteria"].items()]
            wire_questions.append({"type": "choice", "name": identifier,
                                   "instructions": question["instructions"], "choices": choices})
        return {"model": self.config.model, "input": context, "questions": wire_questions}

    def _parse_response(self, response, noul_ids, choice_options, *, document=None):
        if document is None:
            document = self._read_document(response)
        answers = document.get("answers")
        identifiers = (*noul_ids, *choice_options)
        if not isinstance(answers, list) or len(answers) != len(identifiers):
            raise InvalidResponseError("OpenAI Decisions returned missing or unexpected answers.")
        normalized = {}
        for identifier, answer in zip(identifiers, answers):
            if isinstance(answer, dict) and answer.get("type") == "refusal":
                raise InvalidResponseError("OpenAI Decisions refused an evaluation question.")
            if (not isinstance(answer, dict) or answer.get("name") != identifier
                    or answer.get("type") != "choice"):
                raise InvalidResponseError("OpenAI Decisions returned an invalid or reordered answer.")
            options = (True, False) if identifier in noul_ids else choice_options[identifier]
            expected = {(type(value), value) for value in options}
            selected = answer.get("choice")
            if type(selected) not in {str, bool} or (type(selected), selected) not in expected:
                raise InvalidResponseError("OpenAI Decisions selected an unknown choice value.")
            distribution = answer.get("probabilities")
            if not isinstance(distribution, list) or len(distribution) != len(options):
                raise InvalidResponseError("OpenAI Decisions returned incomplete choice probabilities.")
            probabilities = {}
            for item in distribution:
                if not isinstance(item, dict):
                    raise InvalidResponseError("OpenAI Decisions returned invalid choice probabilities.")
                value, probability = item.get("value"), item.get("probability")
                if type(value) not in {str, bool} or not _valid_probability(probability):
                    raise InvalidResponseError("OpenAI Decisions returned invalid choice probabilities.")
                key = (type(value), value)
                if key not in expected or key in probabilities:
                    raise InvalidResponseError("OpenAI Decisions returned unexpected or duplicate choices.")
                probabilities[key] = float(probability)
            if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0.0,
                                abs_tol=_PROBABILITY_SUM_TOLERANCE):
                raise InvalidResponseError("OpenAI Decisions choice probabilities do not sum to one.")
            confidence = answer.get("confidence")
            if not _valid_probability(confidence):
                raise InvalidResponseError("OpenAI Decisions returned invalid choice confidence.")
            if identifier in noul_ids:
                normalized[identifier] = NoulResult(identifier, probabilities[(bool, True)])
            else:
                normalized[identifier] = ChoiceResult(identifier, selected,
                    tuple((value, probabilities[(str, value)]) for value in options), float(confidence))
        returned_model = document.get("model")
        if not isinstance(returned_model, str) or not returned_model.strip():
            raise InvalidResponseError("OpenAI Decisions returned an invalid model identifier.")
        input_tokens, output_tokens = _read_usage(document.get("usage"), self.provider_name)
        return normalized, returned_model, input_tokens, output_tokens

    def _http_error(self, status_code, body=b""):
        try:
            document = json.loads(body)
            detail = document.get("error", document) if isinstance(document, dict) else {}
            detail = detail if isinstance(detail, dict) else {}
        except (ValueError, TypeError):
            detail = {}
        if status_code == 429:
            terminal = {"insufficient_quota", "billing_hard_limit_reached", "quota_exceeded",
                        "billing_limit_exceeded", "usage_limit_reached", "credit_balance_exhausted",
                        "organization_spend_limit_exceeded", "project_spend_limit_exceeded",
                        "organization_usage_limit_exceeded"}
            if any(isinstance(detail.get(name), str) and detail[name] in terminal
                   for name in ("code", "type")):
                return ProviderConfigurationError("OpenAI Decisions quota or billing limit was reached.")
        if status_code in {400, 404, 409, 422} and detail.get("param") == "model":
            return ProviderConfigurationError(
                f"OpenAI Decisions rejected the configured model (HTTP {status_code}).")
        return super()._http_error(status_code, body)
