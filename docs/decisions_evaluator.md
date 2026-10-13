# OpenAI Decisions evaluator

Opt-in information analysis uses OpenAI Decisions as its default typed evaluator:
`openai` selects the service and `gpt-6-luna` selects its model. The generator's
`--llm-model` remains independent. The public beta currently documents only
`gpt-6-luna`; selecting another model requires that the Decisions endpoint support
it. The adapter accepts explicit model identifiers without a local allowlist and
reports a provider error if the service rejects one.

Use an OpenAI-issued API key in `OVS_EVALUATOR_API_KEY`. This is a separate role
credential: `OPENAI_API_KEY` is used for generation and is never an evaluator
fallback. Credentials may be set in the process environment or the root `.env`.

```powershell
$env:OVS_EVALUATOR_API_KEY = "your-openai-evaluator-key"
.\.venv\Scripts\python.exe -m open_video_summary analyze-information `
  --dataset data/processed/bebe_real.json `
  --output outputs/bebe_real_information_decisions_run01.json `
  --llm-provider ollama --llm-model gemma2 `
  --evaluator-provider openai --evaluator-model gpt-6-luna
```

Change only `--evaluator-model` to select a different supported evaluator model.
Alternatively set `OVS_EVALUATOR_MODEL` in the environment or `.env`. Settings
resolve in this order: CLI, process environment, root `.env`, provider default.
Blank model settings use the provider default. Explicit `typesafe` still defaults
to `jev-1.13.0`; choose `--evaluator-provider typesafe` and set this role's TypeSafe
key to run the legacy API. When switching an existing `.env`, update its explicit
provider and model together and remove legacy rate overrides if they no longer
describe the selected service.

For a checkout whose `.env` still explicitly selects Jev, these process settings
override its provider, model and legacy rate settings without editing credentials
in the file. Set the new role key locally; never reuse a TypeSafe key for OpenAI.
Empty local ceilings let valid OpenAI headers establish account limits. Copy the
same nonsecret values into `.env` if the switch should persist across shells.

```powershell
$env:OVS_EVALUATOR_PROVIDER = "openai"
$env:OVS_EVALUATOR_MODEL = "gpt-6-luna"
$env:OVS_EVALUATOR_API_KEY = "your-openai-evaluator-key"
$env:OVS_EVALUATOR_RATE_WINDOW_SECONDS = "60"
$env:OVS_EVALUATOR_REQUEST_LIMIT = ""
$env:OVS_EVALUATOR_TOKEN_LIMIT = ""
$env:OVS_EVALUATOR_LEARN_RATE_LIMITS = "true"
```

The adapter posts the existing transcript context and evaluation questions to
`POST /v1/decisions` using the documented REST schema. It maps existing boolean
questions to `choice` questions with literal boolean alternatives and reads
P(true) directly, so the analysis probability cutoffs retain their meaning.
Categorical choices retain the original string identifiers, descriptions and
probability distribution. Answers must preserve question names and order, contain
every expected alternative once, and provide finite probabilities in [0, 1]
summing to one within the existing 0.02 tolerance. Refusals and malformed results
fail visibly. There is no fallback to generated text or another model.

The optional analyzer does not change HSM selection rules, transcripts, segment
identity, timestamps or labels. Different evaluator models can change information
validation, relations and counts; compare reports with their recorded provider,
requested/returned model and reproducibility manifest. Historical Jev results
remain historical observations, rather than measurements of Decisions.

`OVS_EVALUATOR_BASE_URL` / `--evaluator-base-url` defaults to
`https://api.openai.com/v1`; a compatible gateway base must include its version or
tenant prefix, and the adapter appends `/decisions`. The HTTP implementation uses
Python's standard library, so the existing OpenAI SDK pin and dependency locks
need no changes. The documented Decisions Python SDK requires 3.26.0 or later;
this adapter does not require SDK support for that endpoint.

Requests retain the existing bounded retry, timeout, cancellation, worker ownership
and usage-accounting controls. OpenAI defaults to a 60-second rate window and
unknown request/token ceilings until valid response headers establish them.
`OVS_EVALUATOR_REQUEST_LIMIT` and `OVS_EVALUATOR_TOKEN_LIMIT` set optional local
ceilings. `OVS_EVALUATOR_LEARN_RATE_LIMITS=true` controls header learning; disable
it with `--no-evaluator-learn-rate-limits` when using a different window duration.
Organization and project headers use `OVS_EVALUATOR_ORGANIZATION` and
`OVS_EVALUATOR_PROJECT`, or their `--evaluator-organization` / `--evaluator-project`
flags. They are independent of the generator role and participate in controller
scope isolation. TypeSafe retains its existing 60 requests / 80000 estimated input
tokens per second defaults and does not learn OpenAI headers.

The API receives transcript text, candidates and questions during an authorized
analysis run; the information analyzer sends no video or audio. Tests use synthetic
transcripts and injected offline transports. No live Decisions request or model
quality evaluation is part of this integration's validation.

Official sources: [Decisions guide](https://developers.openai.com/api/docs/guides/decisions),
[REST request and response reference](https://developers.openai.com/api/reference/resources/decisions/methods/create),
and [Python SDK reference](https://developers.openai.com/api/reference/python/resources/decisions/methods/create).
