"""Offline Decisions wire-contract tests on synthetic data, not model evaluation."""

import copy
import json
import unittest
from dataclasses import asdict, replace
from threading import Event
from unittest.mock import patch

from open_video_summary.adapters.decisions import (
    DEFAULT_DECISIONS_MODEL, DecisionsConfig, OpenAIDecisionsEvaluator,
)
from open_video_summary.adapters.evaluation_http import TransportResponse
from open_video_summary.contracts import ChoiceResult, NoulResult
from open_video_summary.errors import (
    AuthenticationError, ConfigurationError, InvalidResponseError,
    ProviderConfigurationError, RequestCancelledError,
)
from tests.test_typesafe_adapter import FakeTransport


def answer(name="supported", *, selected=True, probability=.9):
    return {"type": "choice", "name": name, "choice": selected,
            "confidence": probability if selected else 1 - probability,
            "probabilities": [{"value": True, "probability": probability},
                              {"value": False, "probability": 1 - probability}]}


def response(answers=None, *, model=DEFAULT_DECISIONS_MODEL, usage=None, headers=None):
    document = {"model": model, "answers": [answer()] if answers is None else answers}
    if usage is not None:
        document["usage"] = usage
    return TransportResponse(200, headers or {}, json.dumps(document).encode("utf-8"))


class DecisionsAdapterTests(unittest.TestCase):
    def setUp(self):
        self.config = DecisionsConfig(api_key="offline-evaluator-private", timeout_seconds=4.5,
                                      retry_backoff_seconds=0, max_attempts=1)

    def evaluate(self, transport, **settings):
        evaluator = OpenAIDecisionsEvaluator(replace(self.config, **settings), transport=transport)
        return evaluator, evaluator.evaluate("Uma fonte sintética.", noul={"supported": "Supported?"})

    def test_documented_payload_preserves_context_names_instructions_and_choice_descriptions(self):
        criteria = {"atomic": "One contextual proposition.", "compound": "Independent properties."}
        choice = {"type": "choice", "name": "granularity", "choice": "atomic", "confidence": .97,
                  "probabilities": [{"value": "compound", "probability": .03},
                                    {"value": "atomic", "probability": .97}]}
        transport = FakeTransport(response([answer(), choice], model="gpt-6-luna-snapshot",
            usage={"input_tokens": 21, "output_tokens": 8, "total_tokens": 29}))
        evaluator = OpenAIDecisionsEvaluator(self.config, transport=transport)
        result = evaluator.evaluate("Resumo sintético em português.", noul={"supported": "Supported?"},
                                    choice={"granularity": ("Classify the claim.", criteria)})
        sent = transport.calls[0]
        self.assertEqual("https://api.openai.com/v1/decisions", sent["url"])
        self.assertEqual("Bearer offline-evaluator-private", sent["headers"]["Authorization"])
        self.assertEqual("application/json", sent["headers"]["Content-Type"])
        self.assertEqual(4.5, sent["timeout"])
        self.assertEqual({"model": "gpt-6-luna", "input": "Resumo sintético em português.", "questions": [
            {"type": "choice", "name": "supported", "instructions": "Supported?", "choices": [
                {"value": True, "description": "The answer to the question is yes."},
                {"value": False, "description": "The answer to the question is no."}]},
            {"type": "choice", "name": "granularity", "instructions": "Classify the claim.",
             "choices": [{"value": value, "description": description} for value, description in criteria.items()]},
        ]}, json.loads(sent["body"]))
        self.assertEqual((NoulResult("supported", .9),), result.noul)
        self.assertEqual((ChoiceResult("granularity", "atomic", (("atomic", .97), ("compound", .03)), .97),), result.choice)
        self.assertEqual(("openai", "gpt-6-luna", "gpt-6-luna-snapshot", 21, 8, 2, "reported"), (
            result.metadata.provider, result.metadata.requested_model, result.metadata.returned_model,
            result.metadata.input_tokens, result.metadata.output_tokens,
            result.metadata.question_count, result.metadata.usage_status))
        self.assertEqual([result.metadata], evaluator.records)
        self.assertNotIn(self.config.api_key, repr(self.config))
        self.assertNotIn(self.config.api_key, repr(result))
        self.assertNotIn(self.config.api_key.encode(), sent["body"])

    def test_boolean_probability_is_p_true_even_when_false_is_selected(self):
        _, result = self.evaluate(FakeTransport(response([answer(selected=False, probability=.12)])))
        self.assertAlmostEqual(.12, result.noul[0].probability)
        self.assertIsNone(result.noul[0].confidence)

    def test_literal_string_true_false_options_are_distinct_from_boolean_questions(self):
        categorical = {"type": "choice", "name": "literal", "choice": "false", "confidence": .8,
                       "probabilities": [{"value": "true", "probability": .2},
                                         {"value": "false", "probability": .8}]}
        transport = FakeTransport(response([answer(), categorical]))
        result = OpenAIDecisionsEvaluator(self.config, transport=transport).evaluate(
            "Synthetic labels.", noul={"supported": "Supported?"},
            choice={"literal": ("Which literal?", ("true", "false"))})
        self.assertEqual("false", result.choice[0].selected)
        questions = json.loads(transport.calls[0]["body"])["questions"]
        self.assertIs(True, questions[0]["choices"][0]["value"])
        self.assertEqual("true", questions[1]["choices"][0]["value"])

    def test_explicit_future_model_and_gateway_pass_verbatim_without_fallback(self):
        transport = FakeTransport(response(model="future-decision-model"))
        _, result = self.evaluate(transport, model="future-decision-model", base_url="https://gateway.example/tenant/v1/")
        self.assertEqual("https://gateway.example/tenant/v1/decisions", transport.calls[0]["url"])
        self.assertEqual("future-decision-model", json.loads(transport.calls[0]["body"])["model"])
        self.assertEqual("future-decision-model", result.metadata.requested_model)

    def test_cli_model_and_role_headers_reach_decisions_factory(self):
        from open_video_summary.__main__ import build_parser
        from open_video_summary.adapters.factory import create_evaluator
        from open_video_summary.utils.providers import load_evaluator_config

        args = build_parser().parse_args(["analyze-information", "--evaluator-provider", "openai",
            "--evaluator-model", "future-decision-model", "--evaluator-organization", "cli-org",
            "--evaluator-project", "cli-project", "--no-evaluator-learn-rate-limits"])
        config = load_evaluator_config(vars(args), environ={"OVS_EVALUATOR_API_KEY": "offline-role-key",
            "OVS_EVALUATOR_MODEL": "environment-model", "OPENAI_API_KEY": "offline-generator-key"},
            env_file="outputs/nonexistent-decisions-test.env")
        evaluator = create_evaluator(config)
        self.assertIsInstance(evaluator, OpenAIDecisionsEvaluator)
        self.assertEqual(("future-decision-model", "offline-role-key", "cli-org", "cli-project", False), (
            evaluator.config.model, evaluator.config.api_key, evaluator.config.organization,
            evaluator.config.project, evaluator.config.learn_rate_limits))

    def test_organization_project_headers_and_controller_scope_are_independent(self):
        transport = FakeTransport(response())
        evaluator, _ = self.evaluate(transport, organization="fixture-org", project="fixture-project")
        self.assertEqual("fixture-org", transport.calls[0]["headers"]["OpenAI-Organization"])
        self.assertEqual("fixture-project", transport.calls[0]["headers"]["OpenAI-Project"])
        other = OpenAIDecisionsEvaluator(replace(self.config, project="different-project"),
                                       request_scope=evaluator.request_scope)
        self.assertIsNot(evaluator.controller, other.controller)

    def test_default_unknown_limits_learn_valid_openai_minute_headers(self):
        self.assertIsNone(self.config.request_limit)
        self.assertIsNone(self.config.token_limit)
        self.assertEqual(60, self.config.rate_window_seconds)
        evaluator, result = self.evaluate(FakeTransport(response(headers={
            "x-ratelimit-limit-requests": "10000", "x-ratelimit-limit-tokens": "1000000"})))
        self.assertEqual((9000, 900000, 60), (result.metadata.request_limit,
            result.metadata.token_limit, result.metadata.rate_window_seconds))
        self.assertTrue(evaluator.controller.snapshot()["header_learning"])

    def test_header_learning_can_be_disabled_for_custom_windows(self):
        _, result = self.evaluate(FakeTransport(response(headers={"x-ratelimit-limit-requests": "2"})),
                                 learn_rate_limits=False, request_limit=10, rate_window_seconds=1)
        self.assertEqual(10, result.metadata.request_limit)
        self.assertEqual(1, result.metadata.rate_window_seconds)

    def test_choices_require_two_to_255_unique_values_before_sending(self):
        for options in ((), ("only",), tuple(str(index) for index in range(256)), ("a", "a")):
            with self.subTest(size=len(options)):
                transport = FakeTransport()
                evaluator = OpenAIDecisionsEvaluator(self.config, transport=transport)
                with self.assertRaises(ValueError):
                    evaluator.evaluate("Synthetic.", choice={"q": ("Classify.", options)})
                self.assertEqual([], transport.calls)
                self.assertEqual([], evaluator.records)
        choices = tuple(str(index) for index in range(255))
        transport = FakeTransport(response([{"type": "choice", "name": "q", "choice": "0", "confidence": 1,
            "probabilities": [{"value": option, "probability": 1 if option == "0" else 0} for option in choices]}]))
        result = OpenAIDecisionsEvaluator(self.config, transport=transport).evaluate(
            "Synthetic.", choice={"q": ("Classify.", choices)})
        self.assertEqual(255, len(result.choice[0].probabilities))

    def test_refusal_missing_extra_reordered_names_and_wrong_types_are_rejected(self):
        invalid_answers = [[], [answer(), answer()], [dict(answer(), name="unexpected")],
                           [{"type": "refusal", "name": None}],
                           [dict(answer(), type="predicate")], [dict(answer(), choice="true")],
                           [dict(answer(), choice=1)], [None]]
        for answers in invalid_answers:
            with self.subTest(answers=answers):
                transport = FakeTransport(response(answers))
                evaluator = OpenAIDecisionsEvaluator(self.config, transport=transport)
                with self.assertRaises(InvalidResponseError):
                    evaluator.evaluate("Synthetic.", noul={"supported": "Supported?"})
                self.assertEqual(1, len(transport.calls))
                self.assertEqual("invalid_response", evaluator.records[-1].status)
        transport = FakeTransport(response([answer("second"), answer("first")]))
        with self.assertRaisesRegex(InvalidResponseError, "reordered"):
            OpenAIDecisionsEvaluator(self.config, transport=transport).evaluate(
                "Synthetic.", noul={"first": "First?", "second": "Second?"})

    def test_missing_duplicate_unknown_or_invalid_probabilities_and_confidence_are_rejected(self):
        invalid = [dict(answer(), probabilities=None), dict(answer(), probabilities=[]),
                   dict(answer(), probabilities=[{"value": True, "probability": .9}] * 2),
                   dict(answer(), probabilities=[{"value": "true", "probability": .9},
                                                {"value": False, "probability": .1}]),
                   dict(answer(), probabilities=[{"value": 1, "probability": .9},
                                                {"value": False, "probability": .1}]),
                   dict(answer(), probabilities=[{"value": True, "probability": .9},
                                                {"value": False, "probability": .9}])]
        for value in (True, -1, 1.1, float("nan"), float("inf"), "0.9", None):
            invalid.append(dict(answer(), confidence=value))
            altered = answer()
            altered["probabilities"][0]["probability"] = value
            invalid.append(altered)
        for value in invalid:
            with self.subTest(answer=value), self.assertRaises(InvalidResponseError):
                self.evaluate(FakeTransport(response([value])))

    def test_duplicate_json_and_invalid_model_are_rejected(self):
        for bad in (TransportResponse(200, {}, b'{"model":"gpt-6-luna","answers":[],"answers":[]}'),
                    response(model=None), response(model="")):
            with self.subTest(body=bad.body), self.assertRaises(InvalidResponseError):
                self.evaluate(FakeTransport(bad))

    def test_refusal_and_invalid_usage_still_account_for_valid_reported_input(self):
        for answers, usage in (([{"type": "refusal", "name": "supported"}], {"input_tokens": 10000, "output_tokens": 8}),
                               ([answer()], {"input_tokens": 10000, "output_tokens": True})):
            with self.subTest(usage=usage):
                transport = FakeTransport(response(answers, usage=usage))
                evaluator = OpenAIDecisionsEvaluator(self.config, transport=transport)
                with self.assertRaises(InvalidResponseError):
                    evaluator.evaluate("Synthetic.", noul={"supported": "Supported?"})
                self.assertEqual(10000, evaluator.records[-1].accounted_tokens)
                self.assertGreater(evaluator.records[-1].token_adjustment, 0)
                self.assertEqual(10000, evaluator.records[-1].input_tokens)

    def test_authentication_quota_and_unsupported_model_errors_stop_without_retry(self):
        errors = [(401, {}, AuthenticationError), (429, {"code": "credit_balance_exhausted"}, ProviderConfigurationError),
                  (429, {"type": "project_spend_limit_exceeded"}, ProviderConfigurationError),
                  (404, {}, ProviderConfigurationError), (400, {"param": "model"}, ProviderConfigurationError)]
        for status, details, expected in errors:
            with self.subTest(status=status, details=details):
                transport = FakeTransport(TransportResponse(status, {}, json.dumps({"error": {
                    **details, "message": "offline-evaluator-private and private transcript"}}).encode()))
                evaluator = OpenAIDecisionsEvaluator(replace(self.config, max_attempts=2), transport=transport)
                with self.assertRaises(expected) as caught:
                    evaluator.evaluate("Synthetic.", noul={"supported": "Supported?"})
                self.assertEqual(1, len(transport.calls))
                self.assertNotIn("private", str(caught.exception))
                self.assertNotIn(self.config.api_key, repr(evaluator.records))

    def test_rate_limit_retry_retains_identical_model_payload_and_attempt_audit(self):
        transport = FakeTransport(TransportResponse(429, {"Retry-After": "1"}, b'{}'), response())
        waits = []
        evaluator = OpenAIDecisionsEvaluator(replace(self.config, max_attempts=2), transport=transport,
                                           sleep=waits.append)
        result = evaluator.evaluate("Synthetic.", noul={"supported": "Supported?"})
        self.assertEqual(["rate_limited", "success"], [record.status for record in evaluator.records])
        self.assertEqual(2, result.metadata.attempts)
        self.assertEqual(transport.calls[0]["body"], transport.calls[1]["body"])
        self.assertAlmostEqual(1, sum(waits), places=2)

    def test_local_key_config_and_cancellation_checks_prevent_network_sends(self):
        transport = FakeTransport()
        evaluator = OpenAIDecisionsEvaluator(DecisionsConfig(), transport=transport)
        with self.assertRaisesRegex(AuthenticationError, "OpenAI Decisions API key"):
            evaluator.evaluate("Synthetic.", noul={"supported": "Supported?"})
        event = Event()
        event.set()
        evaluator = OpenAIDecisionsEvaluator(self.config, transport=transport)
        evaluator.cancel_event = event
        with self.assertRaises(RequestCancelledError):
            evaluator.evaluate("Synthetic.", noul={"supported": "Supported?"})
        self.assertEqual([], transport.calls)
        for settings in ({"base_url": "http://remote.example"}, {"max_attempts": 6},
                         {"learn_rate_limits": "true"}, {"project": "project\ninvalid"}):
            with self.subTest(settings=settings), self.assertRaises(ConfigurationError):
                replace(self.config, **settings)

    def test_native_forks_share_controller_but_keep_metadata_and_transport_ownership(self):
        evaluator = OpenAIDecisionsEvaluator(self.config)
        with patch("open_video_summary.adapters.evaluation_http._stdlib_transport", FakeTransport(response())):
            worker = evaluator.fork()
        self.assertIs(evaluator.request_scope, worker.request_scope)
        self.assertIs(evaluator.controller, worker.controller)
        result = worker.evaluate("Synthetic.", noul={"supported": "Supported?"})
        self.assertEqual([], evaluator.records)
        self.assertEqual([result.metadata], worker.records)
        with self.assertRaises(ConfigurationError):
            OpenAIDecisionsEvaluator(self.config, transport=FakeTransport()).fork()

    def test_information_analyzer_uses_real_decisions_transport_and_preserves_sources(self):
        from open_video_summary.core.summarizers.information_analysis import InformationAnalyzer
        from open_video_summary.core.summarizers.information_config import InformationAnalysisConfig
        from open_video_summary.core.summarizers.information_contracts import capture_snapshot
        from tests.test_information_analysis import ScriptedGenerator, candidate, videos

        sent = []

        def transport(url, *, headers, body, timeout):
            payload = json.loads(body)
            sent.append(payload)
            answers = []
            for question in payload["questions"]:
                choices = [item["value"] for item in question["choices"]]
                if type(choices[0]) is bool:
                    answers.append(answer(question["name"], probability=.99))
                else:
                    selected = "atomic" if "atomic" in choices else choices[0]
                    answers.append({"type": "choice", "name": question["name"], "choice": selected,
                        "confidence": .99, "probabilities": [{"value": value,
                            "probability": .99 if value == selected else .01 / (len(choices) - 1)} for value in choices]})
            return response(answers)

        originals = videos(["O backup é diário."])
        frozen = copy.deepcopy(asdict(originals[0]))
        evaluator = OpenAIDecisionsEvaluator(self.config, transport=transport)
        generator = ScriptedGenerator({("v0:s0", "direct"): [candidate("v0:s0", originals[0].segments[0].content)]})
        report = InformationAnalyzer(generator, evaluator, InformationAnalysisConfig(
            qa_enabled=False, experimental_mode="direct", concurrency=1, max_calls=20,
            max_granularity_checks=0, max_literal_repairs=0, max_coverage_foci=0,
        )).analyze(capture_snapshot(originals))
        self.assertEqual("completed", report.status)
        self.assertEqual(1, report.counts.unique_units)
        self.assertEqual(frozen, asdict(originals[0]))
        exported = report.to_dict()
        self.assertEqual("openai", exported["metadata"]["evaluator_provider"])
        self.assertEqual("gpt-6-luna", exported["metadata"]["evaluator"]["model"])
        self.assertIn("adapters/decisions.py", exported["metadata"]["reproducibility"]["manifest"]["source_files"])
        self.assertTrue(any(question["name"] == "granularity" for payload in sent for question in payload["questions"]))
        self.assertNotIn(self.config.api_key, json.dumps(exported))


if __name__ == "__main__":
    unittest.main()
