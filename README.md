# Open Video Summary

Research project on video and multi-video summarization. The library combines
transcript and topic segmentation with selection criteria such as introduction,
subjectivity, redundancy, visual quality, and chronology.

## Local test on Windows

Use **Python 3.11 x64**. Python 3.13 is not compatible with the TensorFlow 2.17
used by this project. The setup below uses the CPU and works with an integrated
AMD GPU. You do not need to activate the virtual environment or install Poetry
for this test.

From the repository root, run these commands in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
.\.venv\Scripts\python.exe -m open_video_summary summarize
```

The first command creates `.venv`, installs the versions pinned in
`requirements-windows.lock`, installs the project in editable mode, and prepares
the notebook's original assets. It can be run again: existing videos and the
model are reused. `-ExecutionPolicy Bypass` applies only to that PowerShell
process.

The first setup downloads approximately 662 MB for the original subjectivity
classifier, in addition to the Python dependencies. The PyTorch CPU package
download is about 207 MB, and the Windows TensorFlow download is about 382 MB.
Allow a few GB of disk space. The model ZIP's SHA256 is checked before
extraction. The three sample videos are already in the clone at
`data/raw/bebe_real.zip` (35 MB).

The `summarize` command runs the **original HSMVideoSumm** on the three videos and
the 13 segments in `data/processed/bebe_real.json`. It uses the trained
classifier and the existing research criteria. The example's transcripts and
topics have already been calculated, so this test does not need an Ollama
server, an API key, or a Whisper download. Processing and encoding use the CPU,
with two threads by default. During testing, consider closing applications that
use a lot of memory.

The results are:

- `outputs/bebe_real_summary.mp4`: summary video with audio;
- `outputs/bebe_real_summary.json`: selected segments and their timestamps;
- `outputs/bebe_real_summary_handler.json`: selection criteria decisions;
- `app.log`: execution log.

The duration and selected segments depend on the algorithm's choices. Running
the example checks that the pipeline works; scientific quality requires
evaluation with the research metrics and data.

To check the environment or repeat only the setup:

```powershell
.\.venv\Scripts\python.exe -m open_video_summary doctor
.\.venv\Scripts\python.exe -m open_video_summary prepare-demo
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Paths and reproducibility in another clone

All relative paths in the API, CLI, and JSON files are interpreted from the
repository root, which the package code locates. For example,
`data/raw/bebe_real/jornal_nacional.mp4` does not depend on the user, the drive
letter, or the working directory. The loader resolves both the video's `path`
and the segments' `video_path`. Exporters write paths inside the project as
root-relative paths in UTF-8. Absolute paths outside the repository are also
supported.

Do not check in `.venv`, downloaded models, caches, or results. Copy the videos
needed for an experiment into `data/raw/<dataset>/` when sharing it. The clone
includes metadata for five datasets, but only `bebe_real` includes MP4s. Other
videos are available in the [original dataset directory](https://drive.google.com/drive/folders/1y19ih3j36UqXlWFcgyxNXsNWluE3lky6?usp=drive_link).

To test another dataset that has already been segmented:

```powershell
.\.venv\Scripts\python.exe -m open_video_summary summarize --dataset data/processed/meu_conjunto.json --output outputs/meu_resumo.mp4
```

To run from another directory, invoke the clone's Python executable using its
path. Arguments remain relative to the project root. Repeat the editable install
if you move the folder or create another clone.

## New videos: Whisper and Ollama

Creating segments from raw MP4s is an additional step. It requires the `ffmpeg`
executable on PATH, a running [Ollama](https://ollama.com/download) server, and a
model available on that server. The Python `ollama` package installed by the
project is an API client; it does not install the server. The Python `ffmpeg`
package also does not replace the executable.

Check FFmpeg with `ffmpeg -version`. To install it on Windows, run
`winget install -e --id Gyan.FFmpeg` and open a new PowerShell window.

After installing the Ollama server, for example:

```powershell
ollama pull gemma2
.\.venv\Scripts\python.exe -m open_video_summary segment --input data/raw/meu_conjunto --output outputs/meu_conjunto_segments.json --whisper-model base --llm-model gemma2
.\.venv\Scripts\python.exe -m open_video_summary summarize --dataset outputs/meu_conjunto_segments.json --output outputs/meu_resumo.mp4
```

`gemma2` is the model used by the original adapter and requires an additional
download of several GB; the setup does not download it. The CLI checks the
server and model before loading Whisper. Whisper `base` is a lighter starting
option for the CPU and downloads its weights during the first transcription.
`tiny` is also available. The original notebook used `medium`; choose
`--whisper-model medium` to use that larger model. The model choice can change
the transcripts and results. The Whisper cache is stored in `.cache/whisper` at
the project root.

## Notebooks and versions

Open the notebooks in your Jupyter editor with the `.venv/Scripts/python.exe`
kernel. `notebooks/hsmvideosumm.ipynb` prepares the same assets and runs the
summary. `notebooks/video-segmenter.ipynb` uses `base` and only the video dataset
included in the clone; it requires Whisper and Ollama and writes new results to
`outputs`. `notebooks/llm-evaluations.ipynb` requires its own evaluation
datasets, Ollama models, and, where applicable, Kaggle access. These additional
resources are not needed for the local HSMVideoSumm test.

`requirements-windows.lock` was derived from `poetry.lock` for Windows x64 and
Python 3.11. It preserves the project's versions, uses PyTorch `2.6.0+cpu`, adds
`tensorflow-intel==2.17.1`, which is required by the Windows TensorFlow wheel,
and uses `tensorflow-io-gcs-filesystem==0.31.0`, which has a wheel for this
platform. The original lock file contains `0.37.1`, which has no Windows wheel.
For other environments, the original workflow remains `poetry install` with a
compatible Python version.
