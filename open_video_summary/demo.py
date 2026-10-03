"""Prepare the original sample data and classifier used by the research notebook."""

import shutil
import stat
import hashlib
from pathlib import PurePosixPath
from zipfile import ZipFile, is_zipfile

from open_video_summary.utils.config import PROJECT_DIR, ModelPaths

SUBJECTIVITY_MODEL_ID = "1nvUTbcjvG3E3LEOG4ccDUZb75KqrmC4Q"
SUBJECTIVITY_MODEL_SHA256 = (
    "9fdc5dcee61c0261f3fc993ea890d4646dc5c6e2081a2f8d970677fe429b82ab"
)


def _extract_archive(archive, destination, *, expected_folder: str) -> None:
    """Extract ordinary files inside the expected dataset/model folder only."""
    destination = destination.resolve()
    with ZipFile(archive) as zipped:
        entries = []
        for item in zipped.infolist():
            name = PurePosixPath(item.filename.replace("\\", "/"))
            target = (destination / str(name)).resolve()
            mode = item.external_attr >> 16
            if (
                not name.parts
                or name.parts[0] != expected_folder
                or not target.is_relative_to(destination / expected_folder)
                or stat.S_ISLNK(mode)
                or target.suffix.lower()
                in {".exe", ".dll", ".bat", ".cmd", ".ps1", ".py"}
            ):
                raise ValueError(f"Unexpected archive entry: {item.filename}")
            entries.append((item, target))

        for item, target in entries:
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not target.exists() or target.stat().st_size != item.file_size:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".tmp")
                with zipped.open(item) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)
                temporary.replace(target)


def prepare_demo(*, download_model: bool = True) -> None:
    """Idempotent setup; the versioned segment JSON is never replaced by the ZIP."""
    archive = PROJECT_DIR / "data/raw/bebe_real.zip"
    if not archive.is_file():
        raise FileNotFoundError("The clone must include data/raw/bebe_real.zip.")
    _extract_archive(archive, PROJECT_DIR / "data/raw", expected_folder="bebe_real")
    print("Sample videos available in data/raw/bebe_real.")

    model_path = PROJECT_DIR / "models/subjectivity_classifier.mdl"
    if (model_path / "config.json").is_file() and any(
        (model_path / name).is_file()
        and (model_path / name).stat().st_size == 711443456
        for name in ("model.safetensors", "pytorch_model.bin")
    ):
        print("Original subjectivity classifier already available.")
        return
    if not download_model:
        print("Subjectivity classifier not downloaded (HSMVideoSumm requires it).")
        return

    model_archive = PROJECT_DIR / "models/subjectivity_classifier.mdl.zip"
    model_archive.parent.mkdir(parents=True, exist_ok=True)
    if not model_archive.exists() or not is_zipfile(model_archive):
        import gdown

        print("Downloading the original subjectivity classifier (~662 MB).")
        result = gdown.download(
            id=SUBJECTIVITY_MODEL_ID,
            output=str(model_archive),
            resume=True,
            use_cookies=False,
        )
        if not result:
            raise RuntimeError(
                "Could not download the classifier from the notebook's link."
            )
    if not is_zipfile(model_archive):
        raise ValueError("The downloaded classifier is not a ZIP archive.")
    with model_archive.open("rb") as file:
        checksum = hashlib.file_digest(file, "sha256").hexdigest()
    if checksum != SUBJECTIVITY_MODEL_SHA256:
        raise ValueError(
            "The classifier ZIP differs from the original research artifact (SHA256 mismatch)."
        )
    _extract_archive(
        model_archive,
        PROJECT_DIR / "models",
        expected_folder="subjectivity_classifier.mdl",
    )
    if not (model_path / "config.json").is_file():
        raise ValueError(
            f"Classifier configuration missing in {ModelPaths.SUBJECTIVITY_CLASSIFIER}."
        )
    print("Original classifier available in models/subjectivity_classifier.mdl.")
