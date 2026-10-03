from numpy import argmin, argsort, count_nonzero, dot, isnan, transpose
from open_video_summary.entities.image import Keyframe
from open_video_summary.utils.processing.metrics import visual_count


def _best_match_index(sim):
    """Keep NumPy's original sort tie/NaN policy without sorting unique maxima."""
    negative = -sim
    if not negative.size or negative.dtype.kind not in "fiuc":
        return argsort(negative)[0]
    best = argmin(negative)
    if isnan(negative[best]) or count_nonzero(negative == negative[best]) != 1:
        visual_count("full_sort_fallbacks")
        return argsort(negative)[0]
    return best


class KeyframeHandler:
    @staticmethod
    def num_matches(kf: Keyframe, other: Keyframe, threshold: float = 0.95) -> int:
        num_match = 0
        d1_t, d2_t = map(transpose, (kf.descriptor, other.descriptor))
        reverse_matches = {}
        forward_dots, reverse_dots, reverse_reuses = 0, 0, 0

        for i, desc in enumerate(kf.descriptor):
            sim = dot(desc, d2_t)
            forward_dots += 1
            self_match = _best_match_index(sim)

            if sim[self_match] >= threshold:
                if self_match not in reverse_matches:
                    match_feature = other.descriptor[self_match]
                    sim_check = dot(match_feature, d1_t)
                    reverse_dots += 1
                    other_match = _best_match_index(sim_check)
                    reverse_matches[self_match] = (
                        other_match,
                        sim_check[other_match] >= threshold,
                    )
                else:
                    reverse_reuses += 1
                other_match, passes_threshold = reverse_matches[self_match]
                num_match += passes_threshold and (other_match == i)

        visual_count("keyframe_pairs")
        visual_count("forward_dots", forward_dots)
        visual_count("reverse_dots", reverse_dots)
        visual_count("reverse_reuses", reverse_reuses)
        return num_match

    @staticmethod
    def is_keyframe(
        keyframe: Keyframe,
        keyframe_list: list[Keyframe],
        min_keypoints_diff_ratio: float = 0.6,
        min_descriptors_diff_ratio: float = 0.1,
    ) -> bool:
        if not keyframe_list:
            return True

        return sum(
            (
                abs(kf.descriptor_size - kf.descriptor_size)
                >= kf.descriptor_size * min_keypoints_diff_ratio
            )
            or (
                KeyframeHandler.num_matches(keyframe, kf)
                < (min_descriptors_diff_ratio * kf.descriptor_size)
            )
            for kf in keyframe_list
        ) == len(keyframe_list)
