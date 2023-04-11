"""
This file contains the time-constrained minimum permutation word error rate
"""

import typing

from meeteval.wer.wer.error_rate import ErrorRate
from meeteval.wer.wer.cp import CPErrorRate
from typing import List, Dict

from meeteval.wer.utils import _items

if typing.TYPE_CHECKING:
    from meeteval._typing import TypedDict

    class Segment(TypedDict):
        words: str
        start_time: 'int | float'
        end_time: 'int | float'


__all__ = ['time_constrained_minimum_permutation_word_error_rate', 'time_constrained_siso_word_error_rate']

# pseudo-timestamp strategies
def equidistant_intervals(interval, count):
    """Divides the interval into `count` equally sized intervals"""
    if count == 0:
        return []
    elif count == 1:
        return [interval]
    interval_length = (interval[1] - interval[0]) / count
    return [(interval[0] + i * interval_length, interval[0] + (i + 1) * interval_length) for i in range(count)]


def equidistant_points(interval, count):
    """Places `count` points (intervals of size zero) in `interval` with equal distance"""
    if count == 0:
        return []
    elif count == 1:
        return [((interval[0] + interval[1]) / 2,) * 2]
    interval_length = (interval[1] - interval[0]) / count

    return [(interval[0] + (i + 0.5) * interval_length,) * 2 for i in range(count)]


def full_segment(interval, count):
    """Outputs `interval` for each word"""
    return [interval] * count


pseudo_word_level_strategies = dict(
    equidistant_intervals=equidistant_intervals,
    equidistant_points=equidistant_points,
    full_segment=full_segment,
)


def apply_collar(timestamps, collar):
    return [(max(t[0] - collar, 0), t[1] + collar) for t in timestamps]


def _get_words_and_intervals(segments: 'List[Segment]', pseudo_word_level_strategy, collar):
    pseudo_word_level_strategy = pseudo_word_level_strategies[pseudo_word_level_strategy]

    words = []
    intervals = []

    for segment in segments:
        segment_words = segment['words'].split()
        words.extend(segment_words)
        segment_timings = pseudo_word_level_strategy((segment['start_time'], segment['end_time']), len(segment_words))
        intervals.extend(segment_timings)
        assert len(segment_words) == len(segment_timings), (segment_words, segment_timings)

    intervals = [(float(i[0]), float(i[1])) for i in intervals]

    return words, apply_collar(intervals, collar)


def _map(fn, x):
    if isinstance(x, dict):
        return {k: fn(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return [fn(v) for v in x]
    else:
        raise TypeError()


def _check_timing_annotations(t, k):
    import numpy as np
    t = np.array(t)
    if np.any(t[1:, 0] < t[:-1, 0]):
        raise ValueError(f'The time annotations should be sorted by start time')
    overlaps = []
    for s1, s2 in zip(t[1:], t[:-1]):
        if s1[0] < s2[1] and s2[0] < s1[1]:
            overlaps.append((s1, s2))
    if len(overlaps):
        import warnings
        warnings.warn(
            f'A speaker ({k}) overlaps with itself. '
            f'This can lead to contradictions between pseudo-word-level timings and word order. '
            f'An exception will be raised later when such a contradiction occurs. '
        )


def _time_constrained_siso_error_rate(
        reference, hypothesis, reference_timing, hypothesis_timing
):
    from meeteval.wer.matching.cy_levenshtein import time_constrained_levenshtein_distance_with_alignment

    result = time_constrained_levenshtein_distance_with_alignment(
        reference, hypothesis, reference_timing, hypothesis_timing
    )

    return ErrorRate(
        result['total'],
        len(reference),
        insertions=result['insertions'],
        deletions=result['deletions'],
        substitutions=result['substitutions'],
    )


def time_constrained_siso_word_error_rate(
        reference: 'List[Segment]',
        hypothesis: 'List[Segment]',
        reference_pseudo_word_level_timing='full_segment',
        hypothesis_pseudo_word_level_timing='equidistant_intervals',
        reference_collar: int = 0,
        hypothesis_collar: int = 0,
):
    reference_words, reference_timing = _get_words_and_intervals(
        reference, reference_pseudo_word_level_timing, reference_collar
    )
    hypothesis_words, hypothesis_timing = _get_words_and_intervals(
        hypothesis, hypothesis_pseudo_word_level_timing, hypothesis_collar
    )
    _check_timing_annotations(reference_timing, 'reference')
    _check_timing_annotations(hypothesis_timing, 'hypothesis')
    return _time_constrained_siso_error_rate(reference_words, hypothesis_words, reference_timing, hypothesis_timing)


def time_constrained_minimum_permutation_word_error_rate(
        reference: 'List[List[Segment]] | Dict[str, List[Segment]]',
        hypothesis: 'List[List[Segment]] | Dict[str, List[Segment]]',
        reference_pseudo_word_level_timing='full_segment',
        hypothesis_pseudo_word_level_timing='equidistant_intervals',
        reference_collar: int = 0,
        hypothesis_collar: int = 0,
) -> CPErrorRate:
    from meeteval.wer.matching.cy_levenshtein import time_constrained_levenshtein_distance
    from meeteval.wer.wer.cp import _cp_word_error_rate

    for k, v in _items(reference):
        _check_timing_annotations([(s['start_time'], s['end_time']) for s in v], f'reference {k}')
    for k, v in _items(hypothesis):
        _check_timing_annotations([(s['start_time'], s['end_time']) for s in v], f'hypothesis {k}')

    # Convert segments into lists of words and word-level timings
    reference = _map(lambda x: _get_words_and_intervals(x, reference_pseudo_word_level_timing, reference_collar),
                     reference)
    hypothesis = _map(lambda x: _get_words_and_intervals(x, hypothesis_pseudo_word_level_timing, hypothesis_collar),
                      hypothesis)

    return _cp_word_error_rate(
        reference, hypothesis,
        distance_fn=lambda tt, et: time_constrained_levenshtein_distance(tt[0], et[0], tt[1], et[1]),
        siso_error_rate=lambda tt, et: _time_constrained_siso_error_rate(tt[0], et[0], tt[1], et[1]),
        missing=[[], []],
    )
