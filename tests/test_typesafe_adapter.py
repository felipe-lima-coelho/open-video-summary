import json
import unittest

from open_video_summary.adapters.typesafe import (
    DEFAULT_TYPESAFE_MODEL,
    ChoiceResult,
    NoulResult,
    TransportResponse,
    TypeSafeConfig,
    TypeSafeEvaluator,
)
from open_video_summary.errors import (
    AuthenticationError,
    ConfigurationError,
    InvalidResponseError,
    RateLimitError,
    ServiceTimeoutError,
)


def response(
    answers=None,
    *,
    status=200,
    model="jev-1.13.0",
    usage=None,
    headers=None,
):
    document = {"answers": answers if answers is not None else {}}
    if model is not None:
        document["model"] = model
    if usage is not None:
        document["usage"] = usage
    return TransportResponse(
        status_code=status,
        headers={} if headers is None else headers,
        body=json.dumps(document, allow_nan=True).encode("utf-8"),
    )


class FakeTransport:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, url, *, headers, body, timeout):
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "body": body,
                "timeout": timeout,
            }
        )
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class TypeSafeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.key = "typesafe-test-key"
        self.config = TypeSafeConfig(
            api_key=self.key,
            base_url="https://api.typesafe.ai",
            timeout_seconds=4.5,
            retry_backoff_seconds=0.25,
        )

    def test_posts_verified_payload_and_normalizes_both_answer_types(self):
        transport = FakeTransport(
            response(
                {
                    "boolean": {"type": "noul", "noul": 0.875},
                    "decision": {
                        "type": "choice",
                        "choice": "retain",
                        "probabilities": {"retain": 0.8, "remove": 0.2},
                        "confidence": 0.8,
                    },
                },
                usage={"input_tokens": 21, "output_tokens": 8},
            )
        )
        evaluator = TypeSafeEvaluator(self.config, transport=transport)

        result = evaluator.evaluate(
            "Resumo de uma entrevista em português.",
            noul={"boolean": "Does the transcript mention an introduction?"},
            choice={
                "decision": (
                    "Should this segment be retained?",
                    ("retain", "remove"),
                )
            },
        )

        self.assertEqual(1, len(transport.calls))
        call = transport.calls[0]
        self.assertEqual("https://api.typesafe.ai/v1/systemone", call["url"])
        self.assertEqual(f"Bearer {self.key}", call["headers"]["Authorization"])
        self.assertEqual("application/json", call["headers"]["Content-Type"])
        self.assertEqual(4.5, call["timeout"])
        self.assertNotIn(self.key.encode(), call["body"])
        body = json.loads(call["body"])
        self.assertEqual("Resumo de uma entrevista em português.", body["state"])
        self.assertEqual(DEFAULT_TYPESAFE_MODEL, body["model"])
        self.assertEqual(
            {
                "boolean": {
                    "type": "noul",
                    "instructions": "Does the transcript mention an introduction?",
                },
                "decision": {
                    "type": "choice",
                    "instructions": "Should this segment be retained?",
                    "criteria": {"retain": "retain", "remove": "remove"},
                },
            },
            body["questions"],
        )
        self.assertEqual((NoulResult("boolean", 0.875),), result.noul)
        self.assertEqual(
            (
                ChoiceResult(
                    "decision",
                    "retain",
                    (("retain", 0.8), ("remove", 0.2)),
                    0.8,
                ),
            ),
            result.choice,
        )
        self.assertEqual("jev-1.13.0", result.metadata.returned_model)
        self.assertEqual(21, result.metadata.input_tokens)
        self.assertEqual(8, result.metadata.output_tokens)
        self.assertEqual(1, result.metadata.attempts)
        self.assertEqual("success", result.metadata.status)
        self.assertEqual([result.metadata], evaluator.records)

    def test_config_is_credential_optional_and_repr_hides_key(self):
        no_key = TypeSafeConfig()
        evaluator = TypeSafeEvaluator(no_key, transport=FakeTransport())
        with self.assertRaises(AuthenticationError):
            evaluator.preflight()
        self.assertNotIn(self.key, repr(self.config))
        self.assertEqual(DEFAULT_TYPESAFE_MODEL, no_key.model)

    def test_preflight_checks_key_without_network_call(self):
        transport = FakeTransport()
        evaluator = TypeSafeEvaluator(TypeSafeConfig(), transport=transport)
        with self.assertRaises(AuthenticationError) as caught:
            evaluator.evaluate("private transcript", noul={"q": "Check it."})
        self.assertNotIn("private transcript", str(caught.exception))
        self.assertEqual([], transport.calls)
        self.assertEqual([], evaluator.records)

    def test_config_rejects_invalid_endpoint_and_retry_settings(self):
        invalid_settings = (
            {"base_url": "http://api.typesafe.ai"},
            {"base_url": "https://user:password@example.test"},
            {"base_url": "https://example.test:bad"},
            {"max_attempts": 0},
            {"max_attempts": 6},
            {"timeout_seconds": float("inf")},
            {"timeout_seconds": 0},
            {"retry_backoff_seconds": float("nan")},
        )
        for overrides in invalid_settings:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ConfigurationError):
                    TypeSafeConfig(**overrides)

    def test_custom_explicit_model_is_preserved_in_request_and_metadata(self):
        config = TypeSafeConfig(api_key=self.key, model="jev-1.12.0")
        transport = FakeTransport(
            response({"q": {"type": "noul", "noul": 0.4}}, model="jev-1.12.0")
        )
        result = TypeSafeEvaluator(config, transport=transport).evaluate(
            "text", noul={"q": "Question?"}
        )
        self.assertEqual("jev-1.12.0", json.loads(transport.calls[0]["body"])["model"])
        self.assertEqual("jev-1.12.0", result.metadata.requested_model)

    def test_retries_rate_limit_with_capped_retry_after_and_records_failure(self):
        transport = FakeTransport(
            TransportResponse(
                429,
                {"Retry-After": "900"},
                b'{"error":"private service details"}',
            ),
            response({"q": {"type": "noul", "noul": 0.5}}),
        )
        waits = []
        evaluator = TypeSafeEvaluator(
            self.config, transport=transport, sleep=waits.append
        )

        result = evaluator.evaluate("private transcript", noul={"q": "Question?"})

        self.assertEqual(2, len(transport.calls))
        self.assertEqual([30.0], waits)
        self.assertEqual(
            ["rate_limited", "success"], [r.status for r in evaluator.records]
        )
        self.assertEqual(2, result.metadata.attempts)
        self.assertNotIn("private service details", repr(evaluator.records))
        self.assertNotIn(self.key, repr(evaluator.records))

    def test_timeout_retries_then_raises_sanitized_provider_error(self):
        transport = FakeTransport(
            TimeoutError(f"timed out with key={self.key} and private transcript"),
            TimeoutError(f"timed out with key={self.key} and private transcript"),
        )
        waits = []
        evaluator = TypeSafeEvaluator(
            self.config, transport=transport, sleep=waits.append
        )
        with self.assertRaises(ServiceTimeoutError) as caught:
            evaluator.evaluate("private transcript", noul={"q": "Question?"})
        self.assertEqual(2, len(transport.calls))
        self.assertEqual([0.25], waits)
        self.assertNotIn(self.key, str(caught.exception))
        self.assertNotIn("private transcript", str(caught.exception))
        self.assertNotIn(self.key, repr(evaluator.records))
        self.assertEqual(["timeout", "timeout"], [r.status for r in evaluator.records])

    def test_authentication_and_unprocessable_errors_do_not_retry(self):
        for status, expected_error in (
            (401, AuthenticationError),
            (422, ConfigurationError),
        ):
            with self.subTest(status=status):
                transport = FakeTransport(
                    TransportResponse(status, {}, b"private transcript and key")
                )
                evaluator = TypeSafeEvaluator(self.config, transport=transport)
                with self.assertRaises(expected_error):
                    evaluator.evaluate("private transcript", noul={"q": "Question?"})
                self.assertEqual(1, len(transport.calls))
                self.assertEqual(1, len(evaluator.records))
                self.assertNotIn("private transcript", repr(evaluator.records))

    def test_invalid_answer_ids_probabilities_and_choice_are_rejected(self):
        bad_answers = (
            {"unexpected": {"type": "noul", "noul": 0.4}},
            {"q": {"type": "noul", "noul": float("nan")}},
            {
                "q": {
                    "type": "choice",
                    "choice": "unknown",
                    "probabilities": {"a": 0.4, "b": 0.6},
                    "confidence": 0.5,
                }
            },
            {
                "q": {
                    "type": "choice",
                    "choice": "a",
                    "probabilities": {"a": 0.4, "b": 0.7},
                    "confidence": 0.5,
                }
            },
            {
                "q": {
                    "type": "choice",
                    "choice": "a",
                    "probabilities": {"a": 0.4, "b": 0.6},
                    "confidence": 1.1,
                }
            },
        )
        for answers in bad_answers:
            with self.subTest(answers=answers):
                transport = FakeTransport(response(answers))
                evaluator = TypeSafeEvaluator(self.config, transport=transport)
                with self.assertRaises(InvalidResponseError):
                    if answers.get("q", {}).get("type") == "choice":
                        evaluator.evaluate("text", choice={"q": ("Pick.", ("a", "b"))})
                    else:
                        evaluator.evaluate("text", noul={"q": "Question?"})
                self.assertEqual(1, len(evaluator.records))
                self.assertEqual("invalid_response", evaluator.records[0].status)

    def test_choice_probability_usage_may_be_absent_only_when_usage_unavailable(self):
        transport = FakeTransport(
            response(
                {
                    "q": {
                        "type": "choice",
                        "choice": "a",
                        "probabilities": {"a": 0.51, "b": 0.49},
                        "confidence": 0.51,
                    }
                },
                usage={"input_tokens": 12},
            )
        )
        result = TypeSafeEvaluator(self.config, transport=transport).evaluate(
            "text", choice={"q": ("Pick.", ("a", "b"))}
        )
        self.assertIsNone(result.metadata.output_tokens)

        no_usage_transport = FakeTransport(
            response(
                {
                    "q": {
                        "type": "choice",
                        "choice": "a",
                        "probabilities": {"a": 0.51, "b": 0.49},
                        "confidence": 0.51,
                    }
                },
            )
        )
        absent = TypeSafeEvaluator(self.config, transport=no_usage_transport).evaluate(
            "text", choice={"q": ("Pick.", ("a", "b"))}
        )
        self.assertIsNone(absent.metadata.input_tokens)
        self.assertIsNone(absent.metadata.output_tokens)

    def test_choice_distribution_accepts_small_rounding_drift(self):
        transport = FakeTransport(
            response(
                {
                    "q": {
                        "type": "choice",
                        "choice": "a",
                        "probabilities": {"a": 0.505, "b": 0.505},
                        "confidence": 0.505,
                    }
                }
            )
        )
        result = TypeSafeEvaluator(self.config, transport=transport).evaluate(
            "text", choice={"q": ("Pick.", ("a", "b"))}
        )
        self.assertEqual((("a", 0.505), ("b", 0.505)), result.choice[0].probabilities)

    def test_duplicate_question_ids_and_bad_input_never_reach_transport(self):
        transport = FakeTransport()
        evaluator = TypeSafeEvaluator(self.config, transport=transport)
        with self.assertRaises(ValueError):
            evaluator.evaluate(
                "text",
                noul={"same": "Question?"},
                choice={"same": ("Pick.", ("a", "b"))},
            )
        self.assertEqual([], transport.calls)


if __name__ == "__main__":
    unittest.main()
