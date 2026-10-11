"""Fixed sampling for information analysis without changing other LLM callers."""

import json

from open_video_summary.adapters.llm import OpenAIAdapter, _value
from open_video_summary.errors import ConfigurationError, InvalidResponseError, ProviderConfigurationError


class InformationOpenAIAdapter(OpenAIAdapter):
    """Use Chat Completions because Responses has no supported seed parameter.

    The seed is a best-effort provider control, not a determinism guarantee.
    Each inherited worker fork owns its client and physical-attempt metadata.
    """

    seed = 42
    temperature = 0.0
    api_endpoint = "chat.completions"

    def _validate_config(self):
        super()._validate_config()
        if (self.config.provider != "openai"
                or not self._known_alias(self.model, "gpt-6-luna")
                or self.config.reasoning_effort != "none"):
            raise ConfigurationError(
                "The information sampling profile requires OpenAI gpt-6-luna "
                "(or a dated snapshot) with reasoning effort none."
            )

    def sampling_settings(self):
        return {"api_endpoint": self.api_endpoint, "requested_temperature": self.temperature,
                "seed": self.seed, "reasoning_effort": self.config.reasoning_effort,
                "note": "Every attempt sends the same seed and temperature. OpenAI seed "
                        "is deprecated and best effort; identical fresh outputs are not guaranteed. "
                        "Unsupported settings fail visibly without a parameter-dropping fallback."}

    def _reset_generation_state(self):
        self._reported_fingerprint = None

    def _generation_fields(self, request):
        return {"temperature_sent": self.temperature if self._request_sent else None,
                "requested_seed": self.seed, "seed_sent": self.seed if self._request_sent else None,
                "system_fingerprint": self._reported_fingerprint,
                "api_endpoint": self.api_endpoint}

    def _generate_once(self, request):
        if request.images or request.temperature != self.temperature:
            raise ConfigurationError(
                "Information generation requires text input and temperature 0."
            )
        kwargs = {"model": self.model,
                  "messages": [{"role": "user", "content": request.prompt}],
                  "store": False, "max_completion_tokens": self.config.max_output_tokens,
                  "reasoning_effort": self.config.reasoning_effort,
                  "temperature": self.temperature, "seed": self.seed}
        response_format = self._format(request.output)
        if response_format:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                key: value for key, value in response_format.items() if key != "type"}}
        client = self._get_client()
        self._admit(json.dumps(kwargs, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                    output_tokens=self.config.max_output_tokens)
        if callable(getattr(client, "with_options", None)):
            client = client.with_options(timeout=self._attempt_timeout, max_retries=0)
        kwargs["timeout"] = self._attempt_timeout
        self._request_sent = True
        completions = client.chat.completions
        raw = getattr(completions, "with_raw_response", None)
        try:
            if raw is not None:
                raw_response = raw.create(**kwargs)
                self._response_headers = raw_response.headers
                response = raw_response.parse()
            else:
                response = completions.create(**kwargs)
                self._response_headers = _value(response, "headers")
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            body = getattr(exc, "body", None)
            parameter = _value(_value(body, "error", body), "param")
            if status in {400, 422} and parameter in {"seed", "temperature", "reasoning_effort"}:
                raise ProviderConfigurationError(
                    f"OpenAI rejected the fixed information sampling profile (parameter: {parameter}). "
                    "The requested settings were not dropped."
                ) from None
            raise
        usage = _value(response, "usage")
        self._usage = (_value(usage, "prompt_tokens"), _value(usage, "completion_tokens"))
        self._reported_model = _value(response, "model")
        self._reported_fingerprint = _value(response, "system_fingerprint")
        choices = _value(response, "choices", []) or []
        if len(choices) != 1 or _value(choices[0], "finish_reason") != "stop":
            raise InvalidResponseError("OpenAI returned an incomplete or failed information response.")
        message = _value(choices[0], "message")
        if _value(message, "refusal"):
            raise InvalidResponseError("OpenAI refused the information response.")
        text = _value(message, "content")
        if not isinstance(text, str) or not text.strip():
            raise InvalidResponseError("OpenAI returned no information response text.")
        # Chat Completions does not report the effective reasoning effort.
        return text, self._reported_model, None
