import json
import os
from collections import OrderedDict
from contextvars import ContextVar
from typing import Callable
from numpy import ndarray
from pandas import DataFrame

from open_video_summary.utils import log
from open_video_summary.entities.video import VideoSegment
from open_video_summary.utils.processing.video import VideoProcessor
from open_video_summary.handlers.summary import SummarySegmentHandler
from open_video_summary.core.selection_criteria.base import SelectionCriteria
from open_video_summary.utils.processing.image import BagOfVisualWords, ImageProcessor
from open_video_summary.utils.paths import project_path
from open_video_summary.utils.processing.metrics import (
    VisualProfile,
    collect_visual_profile,
    visual_count,
    visual_maximum,
    visual_stage,
)


class _DescriptorCache:
    """Bound retained descriptor bytes; each evaluation owns its own cache."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.items = OrderedDict()
        self.nbytes = 0

    def get(self, key):
        if key not in self.items:
            visual_count("cache_misses")
            return None
        self.items.move_to_end(key)
        visual_count("cache_hits")
        return self.items[key]

    def put(self, key, descriptors: ndarray) -> None:
        if descriptors.nbytes > self.max_bytes:
            visual_count("cache_oversized")
            return
        while self.nbytes + descriptors.nbytes > self.max_bytes:
            _, removed = self.items.popitem(last=False)
            self.nbytes -= removed.nbytes
            visual_count("cache_evictions")
        self.items[key] = descriptors
        self.nbytes += descriptors.nbytes
        visual_maximum("cache_peak_bytes", self.nbytes)

    def clear(self) -> None:
        self.items.clear()
        self.nbytes = 0


_descriptor_cache: ContextVar[_DescriptorCache | None] = ContextVar(
    "visual_descriptor_cache", default=None
)


class QualityPick(SelectionCriteria):
    def __init__(
        self,
        source_criteria: str,
        top_n_segments: int = 1,
        bovw_dict_size: int = 300,
        features_extractor: Callable = ImageProcessor.ks_sift,
        max_descriptor_cache_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if max_descriptor_cache_bytes < 0:
            raise ValueError("max_descriptor_cache_bytes must be non-negative.")
        super().__init__(read_from="pick", source_criteria=source_criteria)
        self.top_n_segments = top_n_segments
        self.bovw_dict_size = bovw_dict_size
        self.features_extractor = features_extractor
        self.max_descriptor_cache_bytes = max_descriptor_cache_bytes
        self.last_profile = None

    def evaluate(self, handler: SummarySegmentHandler) -> SummarySegmentHandler:
        profile = VisualProfile()
        cache = _DescriptorCache(self.max_descriptor_cache_bytes)
        cache_token = _descriptor_cache.set(cache)
        try:
            with collect_visual_profile(profile):
                clusters = [
                    cluster
                    for cluster in self.get_criteria_input(handler)
                    if isinstance(cluster, set)
                ]
                visual_count("clusters", len(clusters))
                log.info(
                    f"Retrieved {len(clusters)} cluster to execute {self.name} criteria."
                )

                for cluster in clusters:
                    seg_features = self.extract_segments_visual_features(cluster)
                    df = self.get_bovw_dataframe(seg_features)

                    log.info(
                        f"Retrieving top-{self.top_n_segments} segments from cluster."
                    )

                    with visual_stage("quality_ranking"):
                        df["histogram_sum"] = df.sum(axis=1)
                        top_segments = df.nlargest(
                            self.top_n_segments, columns="histogram_sum"
                        ).index.to_list()

                    # Discarding whole cluster and including only best-quality segment
                    map(lambda s: self.discard(handler, s), cluster)
                    for segment in top_segments:
                        self.include(handler, segment)
        finally:
            _descriptor_cache.reset(cache_token)
            cache.clear()
            self.last_profile = profile.as_dict()
            self.last_profile["settings"] = {
                "scope": "full_video",
                "target_fps": 1,
                "grayscale": True,
                "resolution": "source",
                "bovw_dict_size": self.bovw_dict_size,
                "max_descriptor_cache_bytes": self.max_descriptor_cache_bytes,
                "features_extractor": (
                    "ks_sift"
                    if self.features_extractor is ImageProcessor.ks_sift
                    else "custom"
                ),
            }
            log.info(f"Visual quality profile: {json.dumps(self.last_profile)}")

        return handler

    def extract_segments_visual_features(
        self, segments: set[VideoSegment]
    ) -> dict[VideoSegment, ndarray]:
        log.info("Extracting visual features from segments.")
        cache = _descriptor_cache.get()
        result = {}
        for segment in segments:
            visual_count("segment_feature_requests")
            key = self._descriptor_cache_key(segment, cache)
            descriptors = cache.get(key) if key is not None else None
            if descriptors is None:
                if key is None:
                    visual_count("cache_bypasses")
                frames = VideoProcessor.retrieve_video_frames(
                    segment.video_path, grayscale=True
                )
                if self.features_extractor is ImageProcessor.ks_sift:
                    descriptors = self.features_extractor(frames)
                else:
                    with visual_stage("custom_feature_extraction"):
                        descriptors = self.features_extractor(frames)
                del frames
                if key is not None:
                    cache.put(key, descriptors)
            result[segment] = descriptors
        return result

    def _descriptor_cache_key(self, segment, cache):
        # Custom extractors may be stateful or random, so retain their call semantics.
        if (
            cache is None
            or cache.max_bytes == 0
            or self.features_extractor is not ImageProcessor.ks_sift
        ):
            return None
        try:
            path = project_path(segment.video_path).resolve()
            stat = path.stat()
        except OSError:
            return None
        return (
            os.path.normcase(str(path)),
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            self.features_extractor,
            1,  # Full-resolution, one-frame-per-second grayscale extraction.
            True,
            "full_video",
        )

    def get_bovw_dataframe(
        self, segments_features: dict[VideoSegment, list]
    ) -> DataFrame:
        log.info(
            f"Generating Bag-of-Visual-Words for {len(segments_features)} segments."
        )
        bovw = BagOfVisualWords(
            items=segments_features,
            dict_size=self.bovw_dict_size,
        )
        log.info("Fitting KMeans algorithm for Bag-of-Visual-Words generated.")
        bovw.fit_kmeans()

        return bovw.generate_bovw_dataframe()
