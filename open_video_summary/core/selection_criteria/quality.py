import json
import os
from collections import OrderedDict
from contextlib import nullcontext
from contextvars import ContextVar
from typing import Callable
from numbers import Integral
from math import isfinite
from numpy import ndarray
from pandas import DataFrame

from open_video_summary.utils import log
from open_video_summary.entities.video import VideoSegment
from open_video_summary.utils.processing.video import VideoProcessor
from open_video_summary.handlers.summary import SummarySegmentHandler
from open_video_summary.core.selection_criteria.base import SelectionCriteria
from open_video_summary.utils.processing.image import BagOfVisualWords, ImageProcessor
from open_video_summary.utils.paths import project_path
from open_video_summary.utils.audit import finite_number
from open_video_summary.utils.providers import VisualScope
from open_video_summary.utils.processing.parallel import DescriptorWorkers
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
        self.metadata = {}
        self.nbytes = 0

    def get(self, key):
        if key not in self.items:
            visual_count("cache_misses")
            return None
        self.items.move_to_end(key)
        visual_count("cache_hits")
        return self.items[key]

    def put(self, key, descriptors: ndarray, metadata=None) -> None:
        if descriptors.nbytes > self.max_bytes:
            visual_count("cache_oversized")
            return
        while self.nbytes + descriptors.nbytes > self.max_bytes:
            removed_key, removed = self.items.popitem(last=False)
            self.metadata.pop(removed_key, None)
            self.nbytes -= removed.nbytes
            visual_count("cache_evictions")
        self.items[key] = descriptors
        self.metadata[key] = metadata
        self.nbytes += descriptors.nbytes
        visual_maximum("cache_peak_bytes", self.nbytes)

    def clear(self) -> None:
        self.items.clear()
        self.metadata.clear()
        self.nbytes = 0


_descriptor_cache: ContextVar[_DescriptorCache | None] = ContextVar(
    "visual_descriptor_cache", default=None
)
_descriptor_workers: ContextVar[DescriptorWorkers | None] = ContextVar(
    "visual_descriptor_workers", default=None
)


class QualityPick(SelectionCriteria):
    def __init__(
        self,
        source_criteria: str,
        top_n_segments: int = 1,
        bovw_dict_size: int = 300,
        features_extractor: Callable = ImageProcessor.ks_sift,
        max_descriptor_cache_bytes: int = 256 * 1024 * 1024,
        visual_threads: int = 1,
        visual_scope: VisualScope = "segment",
    ) -> None:
        if max_descriptor_cache_bytes < 0:
            raise ValueError("max_descriptor_cache_bytes must be non-negative.")
        if visual_threads < 1:
            raise ValueError("visual_threads must be positive.")
        super().__init__(read_from="pick", source_criteria=source_criteria)
        self.top_n_segments = top_n_segments
        self.bovw_dict_size = bovw_dict_size
        self.features_extractor = features_extractor
        self.max_descriptor_cache_bytes = max_descriptor_cache_bytes
        self.visual_threads = visual_threads
        self.visual_scope = visual_scope
        self.last_profile = None
        self.last_audit = None
        self._last_extraction_details = {}

    @property
    def visual_scope(self) -> VisualScope:
        return self._visual_scope

    @visual_scope.setter
    def visual_scope(self, value: VisualScope) -> None:
        if value not in {"segment", "video"}:
            raise ValueError("visual_scope must be 'segment' or 'video'.")
        self._visual_scope = value

    def evaluate(self, handler: SummarySegmentHandler) -> SummarySegmentHandler:
        profile = VisualProfile()
        cache = _DescriptorCache(self.max_descriptor_cache_bytes)
        cache_token = _descriptor_cache.set(cache)
        self._last_extraction_details = {}
        self.last_audit = (
            {
                "status": "in_progress",
                "source_criteria": self.source_criteria,
                "scope": self.visual_scope,
                "metric": {
                    "name": "histogram_sum",
                    "formula": "sum(tf * log10(dictionary_size / candidate_document_frequency))",
                    "direction": "larger_is_selected",
                    "missing_word_weight": "null; pandas sum skips missing weights",
                    "tie_comparison": "exact numeric equality",
                    "tie_break": "pandas nlargest keep=first in actual candidate order",
                    "top_n_segments": self.top_n_segments,
                    "dictionary_size": self.bovw_dict_size,
                    "empty_descriptors": "zero word terms and score 0 when the dictionary can be fit",
                    "confidence": "not a confidence or calibrated image-quality probability",
                },
                "extraction": {
                    "scope": self.visual_scope,
                    "interval_bounds": "[start, end)" if self.visual_scope == "segment" else "full source",
                    "target_fps": 1,
                    "sampling": "global source frame index modulo int(native_fps); native resolution",
                    "timestamp_fps": "native floating-point FPS" if self.visual_scope == "segment" else "legacy integer FPS",
                    "grayscale": True,
                    "sift_frames": "sampled frames [1:-1]",
                    "features_extractor": "ks_sift" if self.features_extractor is ImageProcessor.ks_sift else "custom",
                },
                "clusters": [],
            }
            if handler.audit_enabled else None
        )
        if self.last_audit is not None:
            handler.audit["criteria"][self.name] = self.last_audit
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

                workers = self._create_descriptor_workers(clusters, cache)
                with workers if workers is not None else nullcontext():
                    workers_token = _descriptor_workers.set(workers)
                    try:
                        for cluster_index, cluster in enumerate(clusters):
                            cluster_audit = self._start_cluster_audit(handler, cluster_index, cluster)
                            seg_features = None
                            try:
                                seg_features = self.extract_segments_visual_features(cluster)
                                if workers is not None:
                                    workers.wait_before_fit()
                                if cluster_audit is not None:
                                    for candidate, segment in zip(cluster_audit["candidates"], seg_features):
                                        candidate.update(self._last_extraction_details.get(segment, {}))
                                        candidate["descriptor_count"] = len(seg_features[segment])
                                df = self.get_bovw_dataframe(seg_features)
                            except Exception as exc:
                                if cluster_audit is not None:
                                    cluster_audit["status"] = "failed"
                                    cluster_audit["error_type"] = type(exc).__name__
                                    if str(exc).startswith("Visual interval must"):
                                        reason = "invalid_interval"
                                    elif str(exc).startswith("Cannot read video FPS/frame count"):
                                        reason = "invalid_source_video_metadata"
                                    elif seg_features is not None and sum(len(value) for value in seg_features.values()) < self.bovw_dict_size:
                                        reason = "insufficient_descriptors_for_dictionary"
                                    else:
                                        reason = "feature_extraction_or_dictionary_failed"
                                    cluster_audit["failure_reason"] = reason
                                    for candidate, segment in zip(cluster_audit["candidates"], cluster):
                                        candidate.update(self._last_extraction_details.get(segment, {}))
                                        candidate["exclusion_reason"] = cluster_audit["failure_reason"]
                                raise

                            log.info(
                                f"Retrieving top-{self.top_n_segments} segments from cluster."
                            )

                            with visual_stage("quality_ranking"):
                                df["histogram_sum"] = df.sum(axis=1)
                                top_segments = df.nlargest(
                                    self.top_n_segments, columns="histogram_sum"
                                ).index.to_list()

                            if cluster_audit is not None:
                                self._finish_cluster_audit(handler, cluster_audit, df, top_segments)

                            # Discarding whole cluster and including only best-quality segment
                            map(lambda s: self.discard(handler, s), cluster)
                            for segment in top_segments:
                                self.include(handler, segment)
                        if self.last_audit is not None:
                            self.last_audit["status"] = "completed"
                    finally:
                        _descriptor_workers.reset(workers_token)
        finally:
            if self.last_audit is not None and self.last_audit["status"] == "in_progress":
                self.last_audit["status"] = "failed"
            _descriptor_cache.reset(cache_token)
            cache.clear()
            self.last_profile = profile.as_dict()
            self.last_profile["settings"] = {
                "scope": self.visual_scope,
                "target_fps": 1,
                "grayscale": True,
                "resolution": "source",
                "bovw_dict_size": self.bovw_dict_size,
                "max_descriptor_cache_bytes": self.max_descriptor_cache_bytes,
                "visual_threads": self.visual_threads,
                "worker_native_threads": 1,
                "features_extractor": (
                    "ks_sift"
                    if self.features_extractor is ImageProcessor.ks_sift
                    else "custom"
                ),
            }
            log.info(f"Visual quality profile: {json.dumps(self.last_profile)}")

        return handler

    def _start_cluster_audit(self, handler, cluster_index, cluster):
        if self.last_audit is None:
            return None
        cluster_id = f"{self.name}:cluster{cluster_index}"
        report = {
            "cluster_id": cluster_id,
            "source_cluster_id": f"{self.source_criteria}:cluster{cluster_index}",
            "status": "in_progress",
            "candidates": [
                {
                    "segment_id": handler.segment_id(segment),
                    "candidate_index": index,
                    "score": None,
                    "rank": None,
                    "chosen": False,
                    "descriptor_count": None,
                    "interval_status": (
                        "not_used" if self.visual_scope == "video" else "valid"
                        if isfinite(segment.start) and isfinite(segment.end)
                        and 0 <= segment.start < segment.end else "invalid"
                    ),
                }
                for index, segment in enumerate(cluster)
            ],
        }
        self.last_audit["clusters"].append(report)
        return report

    def _finish_cluster_audit(self, handler, report, df, top_segments):
        ranked = df.nlargest(len(df), columns="histogram_sum").index.to_list()
        report["ranking_segment_ids"] = [handler.segment_id(segment) for segment in ranked]
        report["chosen_segment_ids"] = [handler.segment_id(segment) for segment in top_segments]
        by_id = {candidate["segment_id"]: candidate for candidate in report["candidates"]}
        scores = df["histogram_sum"]
        ties = []
        recorded_scores = []
        for segment in df.index:
            score = scores.loc[segment]
            candidate = by_id[handler.segment_id(segment)]
            tied = [item for item in df.index if scores.loc[item] == score]
            candidate.update({
                "score": finite_number(score),
                "score_status": "finite" if finite_number(score) is not None else "nonfinite",
                "rank": ranked.index(segment) + 1,
                "chosen": segment in top_segments,
                "exclusion_reason": None if segment in top_segments else "not_chosen_by_rank",
                "descriptor_status": "no_descriptors" if candidate["descriptor_count"] == 0 else "available",
                "tied_segment_ids": [handler.segment_id(item) for item in tied],
                "visual_word_weights": [
                    {"visual_word": int(word) if isinstance(word, Integral) else str(word), "weight": finite_number(df.loc[segment, word])}
                    for word in df.columns if word != "histogram_sum"
                ],
            })
            if len(tied) > 1 and score not in recorded_scores:
                ties.append({"score": finite_number(score), "segment_ids": candidate["tied_segment_ids"]})
                recorded_scores.append(score)
        report["ties"] = ties
        report["status"] = "completed"

    def _create_descriptor_workers(self, clusters, cache):
        if (
            self.visual_threads <= 1
            or cache.max_bytes == 0
            or self.features_extractor is not ImageProcessor.ks_sift
        ):
            return None
        sources = {}
        for cluster in clusters:
            for segment in cluster:
                key = self._descriptor_cache_key(segment, cache)
                if key is not None:
                    sources.setdefault(key, (
                        key[0], self.visual_scope,
                        segment.start if self.visual_scope == "segment" else None,
                        segment.end if self.visual_scope == "segment" else None,
                    ))
        if len(sources) < 2:
            return None
        return DescriptorWorkers(sources, workers=self.visual_threads)

    def extract_segments_visual_features(
        self, segments: set[VideoSegment]
    ) -> dict[VideoSegment, ndarray]:
        log.info("Extracting visual features from segments.")
        cache = _descriptor_cache.get()
        workers = _descriptor_workers.get()
        result = {}
        for segment in segments:
            visual_count("segment_feature_requests")
            self._last_extraction_details[segment] = {
                "scope": self.visual_scope,
                "start": segment.start if self.visual_scope == "segment" else None,
                "end": segment.end if self.visual_scope == "segment" else None,
            }
            key = self._descriptor_cache_key(segment, cache)
            descriptors = cache.get(key) if key is not None else None
            metadata = cache.metadata.get(key) if descriptors is not None else None
            if descriptors is None:
                if key is None:
                    visual_count("cache_bypasses")
                if workers is not None and key is not None:
                    descriptors = workers.get(key)
                    metadata = workers.last_extraction
                if descriptors is None:
                    frames = (
                        VideoProcessor.retrieve_segment_frames(
                            segment.video_path, segment.start, segment.end, grayscale=True
                        )
                        if self.visual_scope == "segment"
                        else VideoProcessor.retrieve_video_frames(segment.video_path, grayscale=True)
                    )
                    self._last_extraction_details[segment]["sampled_frames"] = len(frames)
                    if self.features_extractor is ImageProcessor.ks_sift:
                        descriptors = (
                            self.features_extractor(frames, allow_empty=True)
                            if self.visual_scope == "segment"
                            else self.features_extractor(frames)
                        )
                    else:
                        with visual_stage("custom_feature_extraction"):
                            descriptors = self.features_extractor(frames)
                    metadata = {
                        "scope": self.visual_scope,
                        "start": segment.start if self.visual_scope == "segment" else None,
                        "end": segment.end if self.visual_scope == "segment" else None,
                        "sampled_frames": len(frames),
                        "descriptor_count": len(descriptors),
                    }
                    del frames
                if key is not None:
                    cache.put(key, descriptors, metadata)
            self._last_extraction_details[segment] = metadata if isinstance(metadata, dict) else {}
            samples = self._last_extraction_details[segment].get("sampled_frames")
            self._last_extraction_details[segment]["extraction_status"] = (
                "available" if len(descriptors) else "no_sampled_frames" if samples == 0
                else "no_inner_frames" if samples is not None and samples <= 2 else "no_descriptors"
            )
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
            self.visual_scope,
            segment.start if self.visual_scope == "segment" else None,
            segment.end if self.visual_scope == "segment" else None,
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
        if self.visual_scope == "segment":
            total = sum(len(value) for value in segments_features.values())
            if total < self.bovw_dict_size:
                raise ValueError(
                    f"Visual dictionary needs {self.bovw_dict_size} descriptors; "
                    f"the candidate group contains {total}."
                )
        log.info("Fitting KMeans algorithm for Bag-of-Visual-Words generated.")
        bovw.fit_kmeans()

        return bovw.generate_bovw_dataframe()
