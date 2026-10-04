"""Summarization coordinators; create the existing HSM singleton on first use."""

__all__ = ["HSMVideoSumm"]


def __getattr__(name):
    if name != "HSMVideoSumm":
        raise AttributeError(name)
    from open_video_summary.utils.config import ModelPaths
    from open_video_summary.core.summarizers.base import Summarizer
    from open_video_summary.classifiers.image import CascadeFaceDetector
    from open_video_summary.core.selection_criteria.quality import QualityPick
    from open_video_summary.core.selection_criteria.introduction import Introduction
    from open_video_summary.classifiers.text import TransformersSubjectivityClassifier
    from open_video_summary.core.selection_criteria.chronology import (
        ClusterBasedChronology,
    )
    from open_video_summary.core.selection_criteria.redundancy import (
        ContentBasedRedundancy,
    )
    from open_video_summary.core.selection_criteria.subjectivity import (
        ObjectContentSubjectivity,
    )

    summarizer = Summarizer(
        selection_criteria=[
            Introduction(),
            ObjectContentSubjectivity(
                subjectivity_classifier=TransformersSubjectivityClassifier(
                    model_path=ModelPaths.SUBJECTIVITY_CLASSIFIER
                ),
                object_detector=CascadeFaceDetector(
                    classifier_path=ModelPaths.FACE_CASCADE
                ),
            ),
            ContentBasedRedundancy(),
            QualityPick(source_criteria="ContentBasedRedundancy"),
            ClusterBasedChronology(cluster_criteria="ContentBasedRedundancy"),
        ]
    )
    globals()[name] = summarizer
    return summarizer
