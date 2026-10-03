# Repository guidance

These instructions apply throughout this repository. Follow the user's request and higher-priority instructions first; use any more specific applicable guidance for its scope.

## Project map

- This is research code for video and multi-video summarization. The existing `HSMVideoSumm` combines transcript/topic segments with introduction, subjectivity, redundancy, visual-quality, chronology, and filtering criteria.
- `open_video_summary/core/summarizers` coordinates summaries; `core/selection_criteria` contains selection rules; `core/segmenter` creates segments; `entities`, `handlers`, and `parsers` define and persist video, segment, summary, and decision data.
- `open_video_summary/__main__.py` is the local CLI. `data/processed/*.json` contains versioned segmented datasets. `notebooks/` contains research and evaluation workflows. Read the relevant module or notebook before changing its behavior.

## Environment and commands

- On Windows, use Python 3.11 x64 and the CPU environment created by `scripts/setup_windows.ps1`. From the repository root:

  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
  .\.venv\Scripts\python.exe -m open_video_summary doctor
  .\.venv\Scripts\python.exe -m unittest discover -s tests -v
  ```

- Use `.\.venv\Scripts\python.exe` for project commands after setup. The Windows lock installs CPU PyTorch; do not assume CUDA or require a GPU.
- `summarize` runs HSMVideoSumm on already segmented input. The bundled `bebe_real` example uses three videos and 13 precomputed segments. `prepare-demo` prepares its sample assets and trained subjectivity classifier.
- `segment` processes raw MP4s and requires FFmpeg on `PATH`, a running Ollama server with the selected model, and Whisper weights. Use pre-segmented data when segmentation is outside the task.
- Run the narrowest relevant tests for code changes. Do not treat a pipeline smoke run or the bundled example as scientific evaluation; describe only datasets, settings, and results that were actually used or measured.

## Research data and behavior

- Preserve source video identity, transcript text, segment order and timestamps, topic labels, selection criteria, and recorded decisions unless the requested change calls for altering them. Explain the reason when a change can affect those values or summary selection.
- Do not silently regenerate or normalize versioned datasets, prompts, model outputs, or notebook results. Keep generated results under `outputs/`; review notebook diffs so unrelated cell output or formatting does not obscure the change.
- The sample transcript and labels are Portuguese research content. Write new documentation, comments, and docstrings in English, while preserving source-language research content.
- Keep paths portable across clones. `utils/config.py` locates the repository root, and `utils/paths.py` resolves relative paths there regardless of the current directory. Persist paths inside datasets as root-relative POSIX paths for both `path` and nested segment `video_path` fields; preserve absolute paths only for files outside the repository.

## Dependencies and generated files

- Keep changes focused and avoid unrelated formatting churn. Follow the existing code and test style.
- Declare direct dependencies in `pyproject.toml`. When changing dependencies, justify the need and update `poetry.lock` and, when the Windows install is affected, `requirements-windows.lock` consistently. Preserve the Windows CPU PyTorch install step in `scripts/setup_windows.ps1`.
- Do not commit `.venv`, downloaded model weights, caches, generated summaries, or extracted raw videos. These belong in ignored local paths such as `.venv/`, `models/`, `.cache/`, and `outputs/`. `data/raw/bebe_real.zip` is an existing tracked demo asset; check tracking status before touching other raw data.
