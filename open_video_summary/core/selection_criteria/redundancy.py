from pandas import DataFrame
from numpy import equal, tril
from math import isnan
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

from open_video_summary.utils import log
from open_video_summary.utils.helpers import custom_cosine
from open_video_summary.entities.video import Video, VideoSegment
from open_video_summary.handlers.summary import SummarySegmentHandler
from open_video_summary.core.selection_criteria.base import SelectionCriteria
from open_video_summary.utils.audit import finite_number


class TopicBasedRedundancy(SelectionCriteria):
    TOPIC_TYPES = {"global_topic", "video_topic"}

    def __init__(self, topic_type: str = "global_topic") -> None:
        if topic_type not in self.TOPIC_TYPES:
            raise ValueError(
                f"Invalid topic_type '{topic_type}'. Must be one of {self.TOPIC_TYPES}."
            )

        super().__init__(read_from="source")
        self.topic_type = topic_type

    def evaluate(self, handler: SummarySegmentHandler) -> SummarySegmentHandler:
        videos = [
            video
            for video in self.get_criteria_input(handler)
            if isinstance(video, Video)
        ]
        log.info(f"Found {len(videos)} videos to execute {self.name} criteria.")

        redundancy_clusters = self.cluster_by_topic(videos, handler)

        for cluster in redundancy_clusters:
            log.info(
                f"Including cluster with {len(cluster)} elements to be chosen from."
            )
            self.pick(handler, cluster)

        return handler

    def cluster_by_topic(
        self, videos: list[Video], handler: SummarySegmentHandler
    ) -> list[set[VideoSegment]]:
        log.info("Clustering videos by topic.")
        topic_clusters: dict[str, set[VideoSegment]] = {}

        for video in videos:
            for segment in video.segments:
                if (
                    segment in handler.discard
                    or segment in handler.output
                    or not (topic := getattr(segment, self.topic_type, None))
                ):
                    continue

                if topic not in topic_clusters:
                    topic_clusters[topic] = set()

                topic_clusters[topic].add(segment)

        return list(topic_clusters.values())


class ContentBasedRedundancy(SelectionCriteria):
    def __init__(
        self,
        reference_time_sec: int = 785,
        base_threshold: float = 0.17,
    ) -> None:
        super().__init__(read_from="source")
        self.reference_time_sec = reference_time_sec
        self.base_threshold = base_threshold
        self.last_audit = None

    def evaluate(self, handler: SummarySegmentHandler) -> SummarySegmentHandler:
        videos = [
            video
            for video in self.get_criteria_input(handler)
            if isinstance(video, Video)
        ]
        log.info(f"Found {len(videos)} videos to execute {self.name} criteria.")

        threshold = self.calc_min_threshold(videos)
        self.last_audit = (
            {
                "status": "in_progress",
                "metric": "custom_cosine dot product of existing TF-IDF vectors",
                "tfidf": {"use_idf": True, "smooth_idf": False, "input": "all source segments, including introductions"},
                "threshold": threshold,
                "threshold_operator": ">",
                "reference_time_sec": self.reference_time_sec,
                "base_threshold": self.base_threshold,
                "coverage": {
                    "matrix": "complete pandas corr result before selection masks",
                    "computed": "upper off-diagonal cells; one custom_cosine calculation per unordered pair with sufficient finite terms",
                    "mirrored": "lower off-diagonal cells mirror the computed value",
                    "diagonal": "pandas forces 1; self similarity is not calculated by custom_cosine",
                    "unavailable": "non-finite values are null",
                },
                "segments": [
                    {"segment_id": handler.segment_id(segment), **handler.segment_eligibility(segment)}
                    for video in videos for segment in video.segments
                ],
                "clusters": [],
                "cluster_decisions": [],
            }
            if handler.audit_enabled else None
        )
        if self.last_audit is not None:
            handler.audit["criteria"][self.name] = self.last_audit

        bow_df = self.get_bow_df(videos)
        correlations = self.get_correlations_df(
            bow_df, threshold=threshold, audit_report=self.last_audit, handler=handler
        )
        redundancies = self.get_redundancies(correlations, audit_report=self.last_audit)
        redundancy_clusters = self.cluster_segments(
            handler, redundancies, videos, audit_report=self.last_audit
        )

        for cluster in redundancy_clusters:
            log.info(
                f"Including cluster with {len(cluster)} elements to be chosen from."
            )
            self.pick(handler, cluster)

        if self.last_audit is not None:
            self.last_audit["status"] = "completed"

        return handler

    def calc_min_threshold(self, videos: list[Video]) -> float:
        set_time = sum(video.segments[-1].end for video in videos)
        diff = (set_time - self.reference_time_sec) / self.reference_time_sec
        return self.base_threshold + self.base_threshold * diff

    def get_bow_df(self, videos: list[Video]) -> DataFrame:
        log.info("Generating bag-of-words DataFrame for videos found.")
        items = {
            (vid_index, seg_index): segment.content
            for vid_index, video in enumerate(videos)
            for seg_index, segment in enumerate(video.segments)
        }

        index_names = ["video_index", "segment_index"]
        index_df = DataFrame(items.keys(), columns=index_names)

        sentences = list(items.values())

        vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=False)
        tfidf_data = vectorizer.fit_transform(sentences)
        tfidf_df = DataFrame(
            tfidf_data.toarray(), columns=vectorizer.get_feature_names_out()
        )

        return index_df.join(tfidf_df).set_index(index_names)

    def get_correlations_df(
        self, bow_df: DataFrame, threshold: float, *, audit_report=None, handler=None
    ) -> DataFrame:
        log.info("Calculating correlations DataFrame from bag-of-words.")
        correlations = bow_df.T.corr(custom_cosine)

        # Disregarding same-video comparisons
        is_same_video = equal.outer(
            correlations.index.get_level_values("video_index"),
            correlations.columns.get_level_values("video_index"),
        )

        # Keeping only the upper diagonal of the pairwise comparisons
        is_upper_diagonal = tril(correlations) > 0

        # Keeping only similarities greater than treshold
        is_gt_threshold = correlations.gt(threshold)

        if audit_report is not None:
            order = list(correlations.index)
            segment_ids = [
                handler.segment_id(handler.source[video_index].segments[segment_index])
                for video_index, segment_index in order
            ]
            audit_report["matrix_segment_ids"] = segment_ids
            audit_report["raw_matrix"] = [
                [finite_number(value) for value in row] for row in correlations.to_numpy()
            ]
            eligibility = {item["segment_id"]: item for item in audit_report["segments"]}
            pair_decisions = []
            for row, row_key in enumerate(order):
                for col, col_key in enumerate(order):
                    value = correlations.iloc[row, col]
                    finite = finite_number(value) is not None
                    present = not isnan(value)
                    same_video = bool(is_same_video[row, col])
                    lower_positive = bool(is_upper_diagonal[row, col])
                    above_threshold = bool(is_gt_threshold.iloc[row, col])
                    reasons = []
                    if not finite:
                        reasons.append("nonfinite_similarity")
                    if same_video:
                        reasons.append("same_video")
                    if lower_positive:
                        reasons.append("lower_triangle_positive_mask")
                    if not above_threshold:
                        reasons.append("not_strictly_above_threshold")
                    row_eligibility = eligibility[segment_ids[row]]
                    col_eligibility = eligibility[segment_ids[col]]
                    for prefix, item in (("row", row_eligibility), ("column", col_eligibility)):
                        if item["already_discarded"]:
                            reasons.append(f"{prefix}_already_discarded")
                        if item["already_output"]:
                            reasons.append(f"{prefix}_already_output")
                    pair_decisions.append({
                        "pair_id": f"r{row}:c{col}",
                        "row": row,
                        "column": col,
                        "row_segment_id": segment_ids[row],
                        "column_segment_id": segment_ids[col],
                        "video_pair": [int(row_key[0]), int(col_key[0])],
                        "value": finite_number(value),
                        "value_status": (
                            "finite" if finite else "nan" if not present
                            else "positive_infinity" if value > 0 else "negative_infinity"
                        ),
                        "origin": (
                            "unavailable" if not present else "pandas_diagonal"
                            if row == col else "computed" if row < col else "mirrored"
                        ),
                        "retained_after_filters": present and not (same_video or lower_positive or not above_threshold),
                        "eligible_for_clustering": row_eligibility["eligible"] and col_eligibility["eligible"],
                        "video_pair_maximum": None,
                        "is_video_pair_maximum": False,
                        "cluster_ids": [],
                        "exclusion_reasons": reasons,
                    })
            audit_report["pair_decisions"] = pair_decisions

        correlations = (
            correlations.mask(is_same_video | is_upper_diagonal | ~is_gt_threshold)
            .dropna(axis="index", how="all")
            .dropna(axis="columns", how="all")
        )
        correlations = correlations.melt(ignore_index=False).dropna(subset=["value"])
        correlations.columns = ["video_index_col", "segment_index_col", "value"]
        correlations.reset_index(inplace=True)

        return correlations

    def get_redundancies(self, correlations: DataFrame, *, audit_report=None) -> list[list[tuple[int, int]]]:
        log.info("Finding redundant segments from correlations.")
        redundancies = correlations[
            correlations.groupby(["video_index", "video_index_col"])["value"].transform(
                max
            )
            == correlations["value"]
        ]

        if audit_report is not None:
            maxima = correlations.groupby(["video_index", "video_index_col"])["value"].max().to_dict()
            audit_report["video_pair_maxima"] = [
                {"video_pair": [int(left), int(right)], "value": finite_number(value), "tie_operator": "=="}
                for (left, right), value in maxima.items()
            ]
            for pair in audit_report["pair_decisions"]:
                maximum = maxima.get(tuple(pair["video_pair"]))
                value = pair["value"]
                if value is None:
                    value = (
                        float("inf") if pair["value_status"] == "positive_infinity"
                        else float("-inf") if pair["value_status"] == "negative_infinity"
                        else float("nan")
                    )
                pair["video_pair_maximum"] = finite_number(maximum) if maximum is not None else None
                pair["is_video_pair_maximum"] = bool(
                    pair["retained_after_filters"] and maximum is not None
                    and value == maximum
                )
                if pair["retained_after_filters"] and not pair["is_video_pair_maximum"]:
                    pair["exclusion_reasons"].append("not_video_pair_maximum")

        # Setting video index and segment index as one tuple object
        redundancies["video"] = tuple(
            zip(redundancies["video_index"], redundancies["segment_index"])
        )
        redundancies["match"] = tuple(
            zip(redundancies["video_index_col"], redundancies["segment_index_col"])
        )
        return redundancies[["video", "match"]].values.tolist()

    def cluster_segments(
        self,
        handler: SummarySegmentHandler,
        redundancies: list[list[tuple[int, int]]],
        videos: list[Video],
        *,
        audit_report=None,
    ) -> list[set[VideoSegment]]:
        clusters: list[set[VideoSegment]] = []
        locations: dict[VideoSegment, int] = {}
        pair_lookup = {}
        if audit_report is not None:
            pair_lookup = {
                (pair["row_segment_id"], pair["column_segment_id"]): pair
                for pair in audit_report["pair_decisions"]
            }

        for item_a, item_b in redundancies:
            segment_a = videos[item_a[0]].segments[item_a[1]]
            segment_b = videos[item_b[0]].segments[item_b[1]]
            decision = None
            pair = None
            if audit_report is not None:
                ids = (handler.segment_id(segment_a), handler.segment_id(segment_b))
                pair = pair_lookup[ids]
                decision = {
                    "pair_id": pair["pair_id"],
                    "segment_ids": list(ids),
                    "action": "skipped_ineligible",
                    "cluster_id": None,
                }
                audit_report["cluster_decisions"].append(decision)

            # Disregarding redundancies which one of the elements was either discarded or outputted
            if (
                segment_a in handler.discard
                or segment_a in handler.output
                or segment_b in handler.discard
                or segment_b in handler.output
            ):
                continue

            if segment_a in locations:
                clusters[locations[segment_a]].add(segment_b)
                locations[segment_b] = locations[segment_a]
                cluster_index, action = locations[segment_a], "add_to_row_segment_cluster"
            elif segment_b in locations:
                clusters[locations[segment_b]].add(segment_a)
                locations[segment_a] = locations[segment_b]
                cluster_index, action = locations[segment_b], "add_to_column_segment_cluster"
            else:
                clusters.append({segment_a, segment_b})
                locations[segment_a] = len(clusters) - 1
                cluster_index, action = len(clusters) - 1, "create_cluster"
            if decision is not None:
                decision["action"] = action
                decision["cluster_id"] = f"{self.name}:cluster{cluster_index}"

        if audit_report is not None:
            audit_report["clusters"] = [
                {
                    "cluster_id": f"{self.name}:cluster{index}",
                    "segment_ids": [handler.segment_id(segment) for segment in cluster],
                }
                for index, cluster in enumerate(clusters)
            ]
            memberships = {}
            for cluster in audit_report["clusters"]:
                for segment_id in cluster["segment_ids"]:
                    memberships.setdefault(segment_id, []).append(cluster["cluster_id"])
            audit_report["segment_cluster_memberships"] = memberships
            for pair in audit_report["pair_decisions"]:
                pair["cluster_ids"] = [
                    cluster_id
                    for cluster_id in memberships.get(pair["row_segment_id"], [])
                    if cluster_id in memberships.get(pair["column_segment_id"], [])
                ]

        return clusters
