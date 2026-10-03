# Open Video Summary

Research code for video and multi-video summarization. HSMVideoSumm combines
transcript and topic segments with selection criteria for introductions,
subjectivity, redundancy, visual quality, and chronology.

## Windows setup

Use **Python 3.11 x64**. Python 3.13 is not compatible with the TensorFlow 2.17
used by this project. The setup uses CPU PyTorch and works without a GPU. From
the repository root, run these commands in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
.\.venv\Scripts\python.exe -m open_video_summary doctor
```

The script creates `.venv`, installs the pinned versions in
`requirements-windows.lock`, installs the project in editable mode, and prepares
the sample assets. It can be run again; existing videos and model files are
reused. `-ExecutionPolicy Bypass` applies only to that PowerShell process.

The first setup downloads about 662 MB for the original subjectivity classifier,
207 MB for CPU PyTorch, and 382 MB for Windows TensorFlow. Allow a few GB of disk
space. The classifier ZIP's SHA256 is checked before extraction. The three sample
videos are in the tracked `data/raw/bebe_real.zip` file (35 MB).

## Run the bundled summary

```powershell
.\.venv\Scripts\python.exe -m open_video_summary summarize
```

This runs the original HSMVideoSumm on the three videos and 13 pre-segmented
clips in `data/processed/bebe_real.json`. The transcripts and topics are already
in that versioned dataset, so the command does not need Ollama, OpenAI, Whisper,
ElevenLabs, or an API key. HSMVideoSumm, video processing, and rendering run
locally on the CPU, using two threads by default.

The command writes:

- `outputs/bebe_real_summary.mp4`: the summary video with audio;
- `outputs/bebe_real_summary.json`: selected clips and timestamps;
- `outputs/bebe_real_summary_handler.json`: selection decisions;
- `outputs/bebe_real_summary_visual_profile.json`: visual stage timings and counters;
- `app.log`: execution log.

Selected clips and duration depend on the algorithm. Running the bundled example
checks that the pipeline works; it is not a scientific evaluation.

### Visual processing and timing

`QualityPick` keeps full-video SIFT descriptors in memory for reuse by segments
from the same source during one evaluation. The cache holds at most 256 MiB of
descriptor arrays and is released after the evaluation, including on failure.
It keys entries by canonical file path, file metadata, and extraction settings;
larger entries are recomputed, and least recently used entries are evicted when
needed. Sampled frames and active candidate-group/KMeans arrays require additional
memory beyond this cache capacity. Custom feature extractors retain their original
per-segment calls.
Python callers can set `QualityPick(max_descriptor_cache_bytes=0, ...)` to
disable the cache.

Extraction still uses the full source video at its original resolution, one
sampled frame per second, and the same first/last-frame exclusion. Each candidate
group still fits its own 300-word KMeans dictionary with the same descriptor
order and multiplicities. SIFT reuses one detector per extraction. Descriptor
matching keeps the same dot products, thresholds, and NumPy sort behavior for
ties and NaNs; repeated reverse comparisons reuse their exact previous result.

The visual profile records total `QualityPick` elapsed time separately from
exclusive substage times: frame decoding (including sampling and grayscale
conversion), SIFT detection, keyframe matching, descriptor assembly, KMeans fit,
KMeans prediction, BoVW dataframe construction, and quality ranking. Their sum
can be smaller than the total because it excludes orchestration and logging.
Counters include actual video reads, decoded/sampled frames, SIFT frames, cache
hits/misses, and KMeans fits. The profile contains no transcript or video pixels.
Python callers can read `QualityPick.last_profile`; the CLI saves it alongside
the summary and also logs it in `app.log`.

The existing KMeans default has no fixed random seed, and candidates are sets.
Independent runs can therefore differ even with unchanged settings. Controlled
comparisons must hold the candidate order, random state, and thread settings
constant. The CLI continues to use two CPU threads by default; use the existing
global `--threads` option before `summarize` when measuring another setting.

## Choose transcription and topic models

Raw videos can use a local or hosted provider for each stage:

| Stage | Local default | Hosted option | Data sent to a hosted service |
| --- | --- | --- | --- |
| Transcription (STT) | Whisper `base` on this computer | ElevenLabs `scribe_v2` | Audio extracted from each video |
| Topic classification (LLM) | Ollama `gemma2` on this computer | OpenAI `gpt-6-luna` using Responses | Transcript text and the topic-classification prompt |

Video reading, audio extraction, segmentation, HSMVideoSumm selection, and
rendering stay local. OpenAI receives transcript text for topic classification;
ElevenLabs receives audio for transcription. The hosted options need internet
access, the matching API key, and an account with access to the selected model.
Provider use can incur charges. The project does not send the full video file to
either provider.

The OpenAI adapter uses the Responses API and structured output. GPT-6 Luna
supports the Responses API, structured output, and reasoning effort values
`none`, `low`, `medium`, `high`, `xhigh`, and `max`. The adapter omits temperature
to use each model's accepted generation defaults. A blank `OVS_LLM_REASONING_EFFORT` leaves the setting
to the model (GPT-6 Luna currently defaults to `medium`); `none` is an explicit
request to disable reasoning. See the
[GPT-6 Luna model page](https://developers.openai.com/api/docs/models/gpt-6-luna),
[reasoning guide](https://developers.openai.com/api/docs/guides/reasoning), and
[structured outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).
Known model capabilities are checked before requests: [GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra)
and [GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol)
accept `low`, `medium`, `high`, `xhigh`, and `max`; [GPT-5](https://developers.openai.com/api/docs/models/gpt-5)
accepts `minimal`, `low`, `medium`, and `high`. Unrecognized future model IDs are
sent to the service, whose compatibility errors become internal configuration errors.

ElevenLabs uses Scribe v2 for hosted transcription. `pt` selects Portuguese; set
`OVS_STT_LANGUAGE=auto` to let the service detect the language. The local Whisper
adapter also accepts `auto`. It maps ISO aliases such as `por` to `pt`, using
the [ISO language code list](https://www.loc.gov/standards/iso639-2/php/code_list.php),
and validates them against the installed Whisper tokenizer before loading model
weights or audio. Native codes such as `haw` and `yue` remain available when the
installed tokenizer supports them. Run metadata preserves the requested code,
the translated code sent to the provider, and the language reported in its response.
Timed words keep the source separators, so languages and mixed text without
spaces retain their spelling and original word timestamps. Internal transcript
whitespace and punctuation keep their source positions. Leading and trailing
utterance whitespace is trimmed, including residue at transcript edges after
removing audio events. Word-token punctuation
supplements a source gap only when that gap has no punctuation; conflicting
punctuation is rejected. Valid reported model and language fields remain in run
metadata even when the transcript is invalid. The default sentence
boundaries include `.`, `!`, `?`, `。`, `！`, and `？`; custom boundaries remain configurable.
See the [speech-to-text API](https://elevenlabs.io/docs/api-reference/speech-to-text/convert)
and [Scribe language and capability guide](https://elevenlabs.io/docs/overview/capabilities/speech-to-text).

## Configure providers

Copy the example to a root `.env` file, then edit the provider and model values
you want to use:

```powershell
Copy-Item .env.example .env
notepad .env
```

The example selects the local defaults and leaves both API keys blank. Set only
the key for each hosted provider you choose. Do not commit `.env` or put real
keys in `.env.example`.

Settings follow this priority: explicit CLI option, process environment, root
`.env`, then provider default. Blank optional values are treated as absent;
`OVS_STT_LANGUAGE=auto` explicitly requests language detection. If you choose a
different provider in `.env`, update its model too. Leaving the model unset uses
`gemma2` for Ollama, `gpt-6-luna` for OpenAI, `base` for local Whisper, or
`scribe_v2` for ElevenLabs. A blank `OVS_LLM_BASE_URL` uses the selected provider's
default endpoint (`http://localhost:11434` for Ollama,
`https://api.openai.com/v1` for OpenAI).

The supported settings and defaults are:

| Setting | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | empty | OpenAI credential |
| `ELEVENLABS_API_KEY` | empty | ElevenLabs credential |
| `OVS_LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `OVS_LLM_MODEL` | provider default | `gemma2` or `gpt-6-luna` |
| `OVS_LLM_REASONING_EFFORT` | model default | OpenAI effort; accepted values depend on the model (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`) |
| `OVS_LLM_BASE_URL` | provider default | Ollama/OpenAI API endpoint or a compatible gateway |
| `OVS_LLM_TIMEOUT_SECONDS` | `120` | LLM request timeout |
| `OVS_STT_PROVIDER` | `whisper_local` | `whisper_local` or `elevenlabs` |
| `OVS_STT_MODEL` | provider default | Whisper model or `scribe_v2` |
| `OVS_STT_LANGUAGE` | `pt` | Language code; `auto` enables detection |
| `OVS_STT_TIMEOUT_SECONDS` | `120` | ElevenLabs HTTP and FFmpeg audio-extraction timeout; it does not interrupt synchronous local Whisper inference |
| `OVS_MAX_ATTEMPTS` | `3` | Maximum total attempts for LLM generation and transient hosted STT failures |

For example, use OpenAI with a compatible endpoint by setting
`OVS_LLM_BASE_URL=https://gateway.example/v1`. That endpoint must implement the
Responses API and structured output used by this adapter. A Chat Completions-only
endpoint is not sufficient. Authentication errors, unsupported model names,
missing Responses routes, and rejected structured-output parameters need a
credential, model, or endpoint correction; retries will not fix them. Only
transient remote network, timeout, or rate-limit failures are retried, within
the configured attempt count. Local Whisper inference errors do not retry.
Check a 401/403 for credentials or access, a 404 for the model or route,
a 400 for unsupported request features, and a 429 for rate limits or quota.
`OVS_MAX_ATTEMPTS` caps LLM generation requests, including retries for malformed
or schema-invalid output. SDK-level automatic retries are disabled so this cap
stays predictable. ElevenLabs retries only transient HTTP failures; local
Whisper model-loading and inference errors do not retry.

## Segment raw videos

`segment` accepts either local or hosted providers for each stage. These
PowerShell examples show all four pairings. Set the needed key in `.env` or in
the current PowerShell session before running a hosted provider.

**Ollama and local Whisper** (the defaults):

```powershell
ollama pull gemma2
.\.venv\Scripts\python.exe -m open_video_summary segment `
  --input data/raw/my_dataset `
  --output outputs/my_dataset_segments.json `
  --llm-provider ollama --llm-model gemma2 `
  --stt-provider whisper_local --stt-model base --stt-language pt
```

Install and start the [Ollama server](https://ollama.com/download); the Python
package installs only its client. Whisper downloads its selected model the first
time it transcribes. `tiny` is a smaller local option; `medium` and `large` need
more memory and time.

**OpenAI and local Whisper:**

```powershell
$env:OPENAI_API_KEY = "your-key"
.\.venv\Scripts\python.exe -m open_video_summary segment `
  --input data/raw/my_dataset `
  --output outputs/my_dataset_segments.json `
  --llm-provider openai --llm-model gpt-6-luna --llm-reasoning-effort low `
  --stt-provider whisper_local --stt-model base --stt-language pt
```

**Ollama and ElevenLabs:**

```powershell
$env:ELEVENLABS_API_KEY = "your-key"
.\.venv\Scripts\python.exe -m open_video_summary segment `
  --input data/raw/my_dataset `
  --output outputs/my_dataset_segments.json `
  --llm-provider ollama --llm-model gemma2 `
  --stt-provider elevenlabs --stt-model scribe_v2 --stt-language pt
```

**OpenAI and ElevenLabs:**

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:ELEVENLABS_API_KEY = "your-key"
.\.venv\Scripts\python.exe -m open_video_summary segment `
  --input data/raw/my_dataset `
  --output outputs/my_dataset_segments.json `
  --llm-provider openai --llm-model gpt-6-luna --llm-reasoning-effort low `
  --stt-provider elevenlabs --stt-model scribe_v2 --stt-language pt
```

You can set a provider endpoint and request limits on the command line. Explicit
CLI options override environment settings:

```powershell
.\.venv\Scripts\python.exe -m open_video_summary segment `
  --input data/raw/my_dataset --output outputs/my_dataset_segments.json `
  --llm-provider openai --llm-model gpt-6-luna `
  --llm-base-url https://gateway.example/v1 `
  --stt-provider elevenlabs --stt-model scribe_v2 --stt-language auto `
  --llm-timeout 120 --stt-timeout 180 --max-attempts 3
```

The legacy `--whisper-model` option aliases `--stt-model`, and `--language`
aliases `--stt-language`. Existing local commands such as
`--whisper-model base --llm-model gemma2 --language pt` continue to work.

Each segment run writes a `<output_stem>_run.json` file next to its output. For
example, `outputs/my_dataset_segments.json` gets
`outputs/my_dataset_segments_run.json`. The run record captures requested,
sent, and service-reported provider/model/reasoning/language settings, elapsed
time, SDK and adapter versions, attempt status, and audio duration when known.
It does not include API keys, transcript text, prompts, or audio bytes.

`segment` also requires the FFmpeg executable on `PATH`, regardless of provider
choice. The Python `ffmpeg` package does not install that executable. Check it
with `ffmpeg -version`; on Windows, install it with
`winget install -e --id Gyan.FFmpeg` and open a new PowerShell window.

## More commands

Check the environment, prepare the sample assets, and run the bundled example:

```powershell
.\.venv\Scripts\python.exe -m open_video_summary doctor
.\.venv\Scripts\python.exe -m open_video_summary prepare-demo
.\.venv\Scripts\python.exe -m open_video_summary summarize
```

`prepare-demo` extracts the tracked videos and downloads the original trained
subjectivity classifier. Use `--without-model` to extract only the videos.
`summarize` accepts another already segmented dataset:

```powershell
.\.venv\Scripts\python.exe -m open_video_summary summarize `
  --dataset data/processed/my_dataset.json `
  --output outputs/my_summary.mp4
```

To run from another directory, invoke the clone's Python executable by its path.
Paths passed to the API and CLI are interpreted from the repository root. Video
paths inside JSON files are saved as portable, root-relative paths.

## Data and notebooks

Do not check in `.venv`, downloaded models, caches, generated summaries, or
extracted raw videos. Keep generated results under `outputs/`. The clone has
metadata for five datasets, but only `bebe_real` includes MP4 files. Other videos
are available from the [original dataset directory](https://drive.google.com/drive/folders/1y19ih3j36UqXlWFcgyxNXsNWluE3lky6?usp=drive_link).

Open notebooks with the `.venv/Scripts/python.exe` kernel.
`notebooks/hsmvideosumm.ipynb` prepares the same sample assets and runs the
summary. `notebooks/video-segmenter.ipynb` uses local Whisper and Ollama and
writes new results to `outputs`. `notebooks/llm-evaluations.ipynb` needs its own
evaluation datasets, Ollama models, and, where applicable, Kaggle access.

`requirements-windows.lock` pins the Windows x64 / Python 3.11 CPU environment.
It includes `tensorflow-intel==2.17.1`, required by the Windows TensorFlow wheel,
and `tensorflow-io-gcs-filesystem==0.31.0`, which has a compatible Windows
wheel. The setup script installs `torch==2.6.0+cpu` from the PyTorch CPU index
before installing that lock file. For other environments, use Poetry with a
compatible Python version.
