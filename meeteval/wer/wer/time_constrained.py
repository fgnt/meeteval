"""
This file contains the time-constrained minimum permutation word error rate
"""
import itertools
import typing
from dataclasses import dataclass, replace

from meeteval.io.stm import STM
from meeteval.io.seglst import SegLST, seglst_map, asseglst, SegLstSegment
from meeteval.wer.wer.error_rate import ErrorRate, SelfOverlap
from meeteval.wer.wer.cp import CPErrorRate
import logging

logger = logging.getLogger('time_constrained')

if typing.TYPE_CHECKING:
    from meeteval._typing import TypedDict


    class Segment(TypedDict):
        words: str
        start_time: 'int | float'
        end_time: 'int | float'
        speaker: typing.Optional[str]

__all__ = [
    'time_constrained_minimum_permutation_word_error_rate',
    'time_constrained_siso_word_error_rate',
    'tcp_word_error_rate_multifile',
    'apply_collar',
    'get_pseudo_word_level_timings',
    'align',
]


# pseudo-timestamp strategies
def equidistant_intervals(interval, words):
    """Divides the interval into `count` equally sized intervals
    """
    count = len(words)
    if count == 0:
        return []
    elif count == 1:
        return [interval]
    interval_length = (interval[1] - interval[0]) / count
    return [
        (
            interval[0] + i * interval_length,
            interval[0] + (i + 1) * interval_length
        ) for i in range(count)
    ]


def equidistant_points(interval, words):
    """Places `count` points (intervals of size zero) in `interval` with equal distance"""
    count = len(words)
    if count == 0:
        return []
    elif count == 1:
        return [((interval[0] + interval[1]) / 2,) * 2]
    interval_length = (interval[1] - interval[0]) / count

    return [
        (interval[0] + i * interval_length + interval_length / 2,) * 2
        for i in range(count)
    ]


def character_based(interval, words):
    """Divides the interval into one interval per word where the size of the interval is
    proportional to the word length in characters."""
    if len(words) == 0:
        return []
    elif len(words) == 1:
        return [interval]
    import numpy as np
    word_lengths = np.asarray([len(w) for w in words])
    end_points = np.cumsum(word_lengths)
    total_num_characters = end_points[-1]
    character_length = (interval[1] - interval[0]) / total_num_characters
    return [
        (
            interval[0] + character_length * start,
            interval[0] + character_length * end
        )
        for start, end in zip([0] + list(end_points[:-1]), end_points)
    ]


def character_based_points(interval, words):
    """Places points in the center of the character-based intervals"""
    intervals = character_based(interval, words)
    intervals = [
        ((interval[1] + interval[0]) / 2,) * 2
        for interval in intervals
    ]
    return intervals


def full_segment(interval, words):
    """Outputs `interval` for each word"""
    return [interval] * len(words)


def no_segmentation(interval, words):
    if len(words) != 1:
        if len(words) > 1:
            raise ValueError(
                f'No segmentation strategy was specified, but the number of words '
                f'is {len(words)} instead of 1 for segment {interval}.\n'
                f'If "{" ".join(words)}" is indeed a single word, open an issue at '
                f'github.com/fgnt/meeteval/issues/new'
            )
        else:
            raise ValueError(
                f'No segmentation strategy was specified (i.e., exactly one '
                f'timestamp per word is required), but a segment ({interval}) '
                f'contains no word.'
            )
    assert len(words) == 1
    return [interval]


pseudo_word_level_strategies = {
    'equidistant_intervals': equidistant_intervals,
    'equidistant_points': equidistant_points,
    'full_segment': full_segment,
    'character_based': character_based,
    'character_based_points': character_based_points,
    'none': no_segmentation,
    None: no_segmentation,
}


def _check_timing_annotations(t, k):
    import numpy as np
    t = np.array(t)
    if np.any(t[1:, 0] < t[:-1, 0]):
        raise ValueError(f'The time annotations must be sorted by start time')
    overlaps = []
    for s1, s2 in zip(t[1:], t[:-1]):
        if s1[0] < s2[1] and s2[0] < s1[1]:
            overlaps.append((s1, s2))
    if len(overlaps):
        import warnings
        k = k or 'unknown'
        warnings.warn(
            f'A speaker ({k}) overlaps with itself. '
            f'This can lead to contradictions between pseudo-word-level '
            f'timings and word order. An exception will be raised later when '
            f'such a contradiction occurs. '
        )


def _time_constrained_siso_error_rate(
        reference: SegLST, hypothesis: SegLST, prune='auto'
):
    from meeteval.wer.matching.cy_levenshtein import time_constrained_levenshtein_distance_with_alignment

    reference = reference.filter(lambda s: s['words'])
    hypothesis = hypothesis.filter(lambda s: s['words'])
    reference_words = reference.T['words']
    reference_timing = list(zip(reference.T['start_time'], reference.T['end_time']))
    hypothesis_words = hypothesis.T['words']
    hypothesis_timing = list(zip(hypothesis.T['start_time'], hypothesis.T['end_time']))

    result = time_constrained_levenshtein_distance_with_alignment(
        reference_words, hypothesis_words, reference_timing, hypothesis_timing,
        prune=prune
    )

    return ErrorRate(
        result['total'],
        len(reference_words),
        insertions=result['insertions'],
        deletions=result['deletions'],
        substitutions=result['substitutions'],
        reference_self_overlap=None,
        hypothesis_self_overlap=None
    )


@dataclass
class TimeMarkedTranscript:
    transcript: 'list[str]'
    timings: 'list[tuple[float, float]]'

    def to_seglst(self):
        return SegLST([{
            'words': transcript,
            'start_time': timing[0],
            'end_time': timing[1],
        } for transcript, timing in zip(self.transcript, self.timings)])

    @classmethod
    def new(cls, d):
        d = asseglst(d)
        return cls(
            transcript=[s['words'] for s in d],
            timings=[(s['start_time'], s['end_time']) for s in d],
        )

    @classmethod
    def merge(cls, *t):
        return TimeMarkedTranscript(
            transcript=[w for tt in t for w in tt.transcript],
            timings=[timing for tt in t for timing in tt.timings]
        )

    def has_self_overlaps(self):
        last_end = 0
        for t in sorted(self.timings):
            if last_end > t[0]:
                return True
            last_end = t[1]
        return False

    def get_self_overlap(self):
        """
        Returns the self-overlap of the transcript.

        ▇
         ▇
          ▇
        >>> TimeMarkedTranscript(['a', 'b', 'c'], [(0, 1), (1, 2), (2, 3)]).get_self_overlap()
        SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=3)

        ▇
        ▇
        >>> TimeMarkedTranscript(['a', 'b'], [(0,1), (0,1)]).get_self_overlap()
        SelfOverlap(overlap_rate=1.0, overlap_time=1, total_time=1)

        ▇
        ▇
        ▇
        >>> TimeMarkedTranscript(['a', 'b', 'c'], [(0,1), (0,1), (0,1)]).get_self_overlap()
        SelfOverlap(overlap_rate=2.0, overlap_time=2, total_time=1)

        ▇▇▇▇▇▇▇▇▇▇
         ▇
           ▇
        >>> TimeMarkedTranscript(['a', 'b', 'c'], [(0,10), (1,2), (3,4)]).get_self_overlap()
        SelfOverlap(overlap_rate=0.2, overlap_time=2, total_time=10)

        ▇▇▇▇
         ▇▇
          ▇▇▇
        >>> TimeMarkedTranscript(['a', 'b', 'c'], [(0,4), (1,3), (2,5)]).get_self_overlap()
        SelfOverlap(overlap_rate=0.8, overlap_time=4, total_time=5)

        >>> TimeMarkedTranscript(['a', 'b', 'c'], [(0, 1), (0.5, 1.5), (1, 2)]).get_self_overlap()
        SelfOverlap(overlap_rate=0.5, overlap_time=1.0, total_time=2.0)
        """
        latest_end = 0
        self_overlap = 0
        total = 0
        for t in sorted(self.timings):
            if latest_end > t[0]:
                self_overlap += min(latest_end, t[1]) - t[0]
            total += max(0, t[1] - latest_end)
            latest_end = max(latest_end, t[1])
        return SelfOverlap(self_overlap, total)

    @classmethod
    def create(cls, data: 'TimeMarkedTranscriptLike') -> 'TimeMarkedTranscript':
        if isinstance(data, TimeMarkedTranscript):
            return data
        elif isinstance(data, STM):
            return cls.from_stm(data)
        elif isinstance(data, list) and (len(data) == 0 or isinstance(data[0], dict)):
            return cls.from_segment_dicts(data)
        else:
            raise TypeError(data)

    @classmethod
    def from_stm(cls, stm: STM) -> 'TimeMarkedTranscript':
        speaker_ids = stm.grouped_by_speaker_id().keys()
        assert len(speaker_ids) == 1, 'Only single-speaker STMs are supported'
        time_marked_transcript = cls(
            [l.transcript for l in stm.lines],
            [(l.begin_time, l.end_time) for l in stm.lines],
        )
        return time_marked_transcript

    @classmethod
    def from_segment_dicts(cls, data: 'list[Segment]') -> 'TimeMarkedTranscript':
        if len(data) == 0:
            return cls([], [])
        if 'speaker' in data[0]:
            assert all(d['speaker'] == data[0]['speaker'] for d in data[1:])
        time_marked_transcript = cls(
            transcript=[s['words'] for s in data],
            timings=[(s['start_time'], s['end_time']) for s in data],
        )
        return time_marked_transcript

    def _repr_pretty_(self, p, cycle):
        """
        >>> tmt = TimeMarkedTranscript(['abc b', 'c d e f'], [(0, 4), (4, 8)])
        >>> tmt
        TimeMarkedTranscript(transcript=['abc b', 'c d e f'], timings=[(0, 4), (4, 8)])
        >>> from IPython.lib.pretty import pprint
        >>> pprint(tmt, max_width=30)
        TimeMarkedTranscript(
            transcript=['abc b',
             'c d e f'],
            timings=[(0, 4), (4, 8)]
        )
        """
        if cycle:
            p.text(f'{self.__class__.__name__}(...)')
        else:
            txt = f'{self.__class__.__name__}('
            with p.group(4, txt, ''):
                keys = self.__dataclass_fields__.keys()
                for i, k in enumerate(keys):
                    if i:
                        p.breakable(sep=' ')
                    else:
                        p.breakable(sep='')
                    p.text(f'{k}=')
                    p.pretty(getattr(self, k))
                    if i != len(keys) - 1:
                        p.text(',')
            p.breakable('')
            p.text(')')


# Annotation for input
TimeMarkedTranscriptLike = 'TimeMarkedTranscript | STM | list[Segment]'


@seglst_map()
def apply_collar(s: SegLST, collar: float):
    """
    Adds a collar to begin and end times.

    Works with any format that is convertible to SegLST and back, such as `STM` and `RTTM`.

    >>> apply_collar(SegLST([{'start_time': 0, 'end_time': 1}]), 1)
    SegLST(segments=[{'start_time': -1, 'end_time': 2}])
    >>> print(apply_collar(STM.parse('X 1 A 0 1 a b'), 1).dumps())
    X 1 A -1 2 a b
    <BLANKLINE>
    """
    return s.map(lambda s: {**s, 'start_time': s['start_time'] - collar, 'end_time': s['end_time'] + collar})


@seglst_map()
def get_pseudo_word_level_timings(t: SegLST, strategy: str) -> SegLST:
    """
    Takes a transcript with segment-level annotations and outputs a transcript
    with estimated word-level annotations.

    Choices for `strategy`:
        - `'equidistant_intervals`': Divide segment-level timing into equally
                sized intervals
        - `'equidistant_points`': Place time points equally spaded int the
                segment-level intervals
        - `'full_segment`': Use the full segment for each word that belongs to
                that segment
        - `'character_based`': Estimate the word length based on the number
                of characters
        - `'character_based_points`': Estimates the word length based on the
                number of characters and creates a point in the center of each
                word
        - `'none`' or `None`: Do not estimate word-level timings but assume
                that the provided timings are already given on a word level.

    >>> from IPython.lib.pretty import pprint
    >>> from meeteval.io.seglst import SegLST
    >>> s = SegLST([{'words': 'abc b', 'start_time': 0, 'end_time': 4}, {'words': 'c d e f', 'start_time': 4, 'end_time': 8}])
    >>> pprint(get_pseudo_word_level_timings(s, 'full_segment'))
    SegLST([{'words': 'abc', 'start_time': 0, 'end_time': 4},
            {'words': 'b', 'start_time': 0, 'end_time': 4},
            {'words': 'c', 'start_time': 4, 'end_time': 8},
            {'words': 'd', 'start_time': 4, 'end_time': 8},
            {'words': 'e', 'start_time': 4, 'end_time': 8},
            {'words': 'f', 'start_time': 4, 'end_time': 8}])
    >>> pprint(get_pseudo_word_level_timings(s, 'equidistant_points'))
    SegLST([{'words': 'abc', 'start_time': 1.0, 'end_time': 1.0},
            {'words': 'b', 'start_time': 3.0, 'end_time': 3.0},
            {'words': 'c', 'start_time': 4.5, 'end_time': 4.5},
            {'words': 'd', 'start_time': 5.5, 'end_time': 5.5},
            {'words': 'e', 'start_time': 6.5, 'end_time': 6.5},
            {'words': 'f', 'start_time': 7.5, 'end_time': 7.5}])
    >>> pprint(get_pseudo_word_level_timings(s, 'equidistant_intervals'))
    SegLST([{'words': 'abc', 'start_time': 0.0, 'end_time': 2.0},
            {'words': 'b', 'start_time': 2.0, 'end_time': 4.0},
            {'words': 'c', 'start_time': 4.0, 'end_time': 5.0},
            {'words': 'd', 'start_time': 5.0, 'end_time': 6.0},
            {'words': 'e', 'start_time': 6.0, 'end_time': 7.0},
            {'words': 'f', 'start_time': 7.0, 'end_time': 8.0}])
    >>> word_level = get_pseudo_word_level_timings(s, 'character_based')
    >>> pprint(word_level)
    SegLST([{'words': 'abc', 'start_time': 0.0, 'end_time': 3.0},
            {'words': 'b', 'start_time': 3.0, 'end_time': 4.0},
            {'words': 'c', 'start_time': 4.0, 'end_time': 5.0},
            {'words': 'd', 'start_time': 5.0, 'end_time': 6.0},
            {'words': 'e', 'start_time': 6.0, 'end_time': 7.0},
            {'words': 'f', 'start_time': 7.0, 'end_time': 8.0}])
    >>> pprint(get_pseudo_word_level_timings(word_level, 'none'))   # Copies over the timings since word-level timings are already assumed
    SegLST([{'words': 'abc', 'start_time': 0.0, 'end_time': 3.0},
            {'words': 'b', 'start_time': 3.0, 'end_time': 4.0},
            {'words': 'c', 'start_time': 4.0, 'end_time': 5.0},
            {'words': 'd', 'start_time': 5.0, 'end_time': 6.0},
            {'words': 'e', 'start_time': 6.0, 'end_time': 7.0},
            {'words': 'f', 'start_time': 7.0, 'end_time': 8.0}])
    >>> pprint(get_pseudo_word_level_timings(s, 'character_based_points'))
    SegLST([{'words': 'abc', 'start_time': 1.5, 'end_time': 1.5},
            {'words': 'b', 'start_time': 3.5, 'end_time': 3.5},
            {'words': 'c', 'start_time': 4.5, 'end_time': 4.5},
            {'words': 'd', 'start_time': 5.5, 'end_time': 5.5},
            {'words': 'e', 'start_time': 6.5, 'end_time': 6.5},
            {'words': 'f', 'start_time': 7.5, 'end_time': 7.5}])

    Works with any format that is convertible to SegLST and back, for example STM:
    >>> print(get_pseudo_word_level_timings(STM.new(s, session_id='dummy', speaker='dummy'), 'character_based_points').dumps())
    dummy 1 dummy 1.5 1.5 abc
    dummy 1 dummy 3.5 3.5 b
    dummy 1 dummy 4.5 4.5 c
    dummy 1 dummy 5.5 5.5 d
    dummy 1 dummy 6.5 6.5 e
    dummy 1 dummy 7.5 7.5 f
    <BLANKLINE>
    """
    pseudo_word_level_strategy = pseudo_word_level_strategies[strategy]

    def get_words(s):
        res = []
        words = s['words'].split()
        if not words:  # Make sure that we don't drop a speaker
            words = ['']
        for w, (start, end) in zip(
                words,
                pseudo_word_level_strategy((s['start_time'], s['end_time']), words)
        ):
            res.append({**s, 'words': w, 'start_time': start, 'end_time': end})
        return res

    return t.flatmap(get_words)


@seglst_map(required_keys=('start_time', 'end_time'))
def remove_overlaps(
        t: SegLST,
        max_overlap: float = 0.4,
        warn_message: str = None,
) -> SegLST:
    """
    Remove overlaps between words or segments in a transcript.

    Note: Sorts the segments by begin time.

    Args:
        t: SegLST object to remove overlaps from
        max_overlap: maximum allowed relative overlap between words or segments.
            Raises a `ValueError` when more overlap is found.
        warn_message: if not None, a warning is printed when overlaps are corrected.
    """
    last: 'typing.Optional[SegLstSegment]' = None

    def correct(s):
        nonlocal last
        if last and last['end_time'] > s['start_time']:
            if warn_message is not None:
                import warnings
                warnings.warn(warn_message)
            overlap = last['end_time'] - s['start_time']
            if overlap > max_overlap * (s['start_time'] - last['start_time']):
                import numpy as np
                raise ValueError(
                    f'Overlapping segments exceed max allowed relative overlap. '
                    f'Segment {last} overlaps with {s}. '
                    f'{overlap} > {max_overlap * (s["end_time"] - last["start_time"])} '
                    f'relative overlap: {np.divide(overlap, (s["end_time"] - last["end_time"]))}'
                )
            center = (last['end_time'] + s['start_time']) / 2
            assert center > last['start_time'], (center, last['start_time'])
            last['start_time'] = last['start_time']
            last['end_time'] = center
            assert last['end_time'] > last['start_time'], last
        last = s
        return s

    return t.sorted('start_time').map(correct)


def sort_and_validate(segments: SegLST, sort, pseudo_word_level_timing, name, warn=True):
    """
    Args:
        segments:
        sort: How to sort words/segments. Options:
            - `True`: sort by segment start time and assert that the word-level
                        timings are sorted by start time
            - `False`: do not sort and do not check word order
            - `'segment'`: sort segments by start time and do not check word order
            - `'word'`: sort words by start time
        pseudo_word_level_timing:
        name:

    >>> segments = SegLST([{'words': 'c d', 'start_time': 1, 'end_time': 3}, {'words': 'a b', 'start_time': 0, 'end_time': 3}])
    >>> sort_and_validate(segments, True, 'character_based', 'test')
    Traceback (most recent call last):
    ...
    ValueError: The order of word-level timings contradicts the segment-level order in test: 2 of 4 times.
    Consider setting sort to False or "segment" or "word".
    >>> sort_and_validate(segments, False, 'character_based', 'test')
    SegLST(segments=[{'words': 'c', 'start_time': 1.0, 'end_time': 2.0}, {'words': 'd', 'start_time': 2.0, 'end_time': 3.0}, {'words': 'a', 'start_time': 0.0, 'end_time': 1.5}, {'words': 'b', 'start_time': 1.5, 'end_time': 3.0}])
    >>> sort_and_validate(segments, 'segment', 'character_based', 'test')
    SegLST(segments=[{'words': 'a', 'start_time': 0.0, 'end_time': 1.5}, {'words': 'b', 'start_time': 1.5, 'end_time': 3.0}, {'words': 'c', 'start_time': 1.0, 'end_time': 2.0}, {'words': 'd', 'start_time': 2.0, 'end_time': 3.0}])
    >>> sort_and_validate(segments, 'word', 'character_based', 'test')
    SegLST(segments=[{'words': 'a', 'start_time': 0.0, 'end_time': 1.5}, {'words': 'c', 'start_time': 1.0, 'end_time': 2.0}, {'words': 'b', 'start_time': 1.5, 'end_time': 3.0}, {'words': 'd', 'start_time': 2.0, 'end_time': 3.0}])
    """
    if sort not in (True, False, 'segment', 'word'):
        raise ValueError(
            f'Invalid value for sort: {sort}. Choose one of True, False, '
            f'"segment", "word"'
        )

    for s in segments:
        if s['end_time'] < s['start_time']:
            raise ValueError(
                f'The end time of an interval must be larger than the start '
                f'time. Found {s} in {name}'
            )

    if sort in (True, 'segment', 'word'):
        segments = segments.sorted('start_time')

    words = get_pseudo_word_level_timings(segments, pseudo_word_level_timing)

    # Check whether words are sorted by start time
    words_sorted = words.sorted('start_time')
    if warn and words_sorted != words:
        # This check should be fast because `sorted` doesn't change the identity
        # of the contained objects (so `words_sorted[0] is words[0] == True`
        # when they are sorted).
        contradictions = [a != b for a, b in zip(words_sorted, words)]
        try:
            session_ids = f' (session ids: {sorted(set(segments.T["session_id"]))})'
        except KeyError:
            session_ids = ''
        msg = (
            f'The order of word-level timings contradicts the segment-level '
            f'order in {name}: {sum(contradictions)} of {len(contradictions)} '
            f'times.{session_ids}'
        )
        if sort is not True:
            logger.warning(msg)
        else:
            raise ValueError(
                f'{msg}\nConsider setting sort to False or "segment" or "word".'
            )

    if sort == 'word':
        words = words_sorted

    return words


def get_self_overlap(d: SegLST):
    """
    Returns the self-overlap of the transcript.

    ▇
     ▇
      ▇
    >>> get_self_overlap([{'words': 'a', 'start_time': 0, 'end_time': 1}, {'words': 'b', 'start_time': 1, 'end_time': 2}, {'words': 'c', 'start_time': 2, 'end_time': 3}])
    SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=3)

    ▇
    ▇
    >>> get_self_overlap([{'words': 'a', 'start_time': 0, 'end_time': 1}] * 2)
    SelfOverlap(overlap_rate=1.0, overlap_time=1, total_time=1)

    ▇
    ▇
    ▇
    >>> get_self_overlap([{'words': 'a', 'start_time': 0, 'end_time': 1}] * 3)
    SelfOverlap(overlap_rate=2.0, overlap_time=2, total_time=1)

    ▇▇▇▇▇▇▇▇▇▇
     ▇
       ▇
    >>> get_self_overlap([{'words': 'a', 'start_time': 0, 'end_time': 10}, {'words': 'b', 'start_time': 1, 'end_time': 2}, {'words': 'c', 'start_time': 3, 'end_time': 4}])
    SelfOverlap(overlap_rate=0.2, overlap_time=2, total_time=10)

    ▇▇▇▇
     ▇▇
      ▇▇▇
    >>> get_self_overlap([{'words': 'a', 'start_time': 0, 'end_time': 4}, {'words': 'b', 'start_time': 1, 'end_time': 3}, {'words': 'c', 'start_time': 2, 'end_time': 5}])
    SelfOverlap(overlap_rate=0.8, overlap_time=4, total_time=5)

    >>> get_self_overlap([{'words': 'a', 'start_time': 0, 'end_time': 1}, {'words': 'b', 'start_time': 0.5, 'end_time': 1.5}, {'words': 'c', 'start_time': 1, 'end_time': 2}])
    SelfOverlap(overlap_rate=0.5, overlap_time=1.0, total_time=2.0)
    """
    d = asseglst(d, required_keys=('start_time', 'end_time'), py_convert=None)
    latest_end = 0
    self_overlap = 0
    total = 0
    for t in sorted(d, key=lambda x: x['start_time']):
        if latest_end > t['start_time']:
            self_overlap += min(latest_end, t['end_time']) - t['start_time']
        total += max(0, t['end_time'] - latest_end)
        latest_end = max(latest_end, t['end_time'])
    return SelfOverlap(self_overlap, total)


def time_constrained_siso_levenshtein_distance(
        reference: 'SegLST', hypothesis: 'SegLST'
) -> int:
    from meeteval.wer.matching.cy_levenshtein import time_constrained_levenshtein_distance

    # Ignore empty segments
    reference = reference.filter(lambda s: s['words'])
    hypothesis = hypothesis.filter(lambda s: s['words'])

    return time_constrained_levenshtein_distance(
        reference=reference.T['words'],
        hypothesis=hypothesis.T['words'],
        reference_timing=list(zip(reference.T['start_time'], reference.T['end_time'])),
        hypothesis_timing=list(zip(hypothesis.T['start_time'], hypothesis.T['end_time'])),
    )


def time_constrained_siso_word_error_rate(
        reference: 'SegLST',
        hypothesis: 'SegLST',
        reference_pseudo_word_level_timing='character_based',
        hypothesis_pseudo_word_level_timing='character_based_points',
        collar: int = 0,
        reference_sort='segment',
        hypothesis_sort='segment',
):
    """
    Time-constrained word error rate for single-speaker transcripts.

    Args:
        reference: reference transcript
        hypothesis: hypothesis transcript
        reference_pseudo_word_level_timing: strategy for pseudo-word level
            timing for reference
        hypothesis_pseudo_word_level_timing: strategy for pseudo-word level
            timing for hypothesis
        collar: collar applied to hypothesis pseudo-word level timings
        reference_sort: How to sort the reference. Options: 'segment',
            'word', True, False. See below
        hypothesis_sort: How to sort the reference. Options: 'segment',
            'word', True, False. See below
        
    Reference / Hypothesis sorting options:
    - True: sort by segment start time and assert that the word-level timings
        are sorted by start time
    - False: do not sort and don't check word order
    - 'segment': sort by segment start time and don't check word order
    - 'word': sort by word start time

    >>> time_constrained_siso_word_error_rate(
    ... [{'words': 'a b', 'start_time': 0, 'end_time': 2},  {'words': 'c d', 'start_time': 0, 'end_time': 2}],
    ... [{'words': 'a', 'start_time': 0, 'end_time': 1}])
    ErrorRate(error_rate=0.75, errors=3, length=4, insertions=0, deletions=3, substitutions=0, reference_self_overlap=SelfOverlap(overlap_rate=1.0, overlap_time=2, total_time=2), hypothesis_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1))
    """
    # Convert to SegLST. Disallow Python conversions since there is currently
    # no way to get the timings from a Python structure.
    reference = asseglst(
        reference,
        required_keys=('start_time', 'end_time', 'words'),
        py_convert=None
    )
    hypothesis = asseglst(
        hypothesis,
        required_keys=('start_time', 'end_time', 'words'),
        py_convert=None
    )

    # Only single-speaker transcripts are supported, but we can here have
    # multiple segments, e.g., for word-level transcripts
    assert 'speaker' not in reference.T.keys() or len(reference.unique('speaker')) <= 1, 'Only single-speaker transcripts are supported'
    assert 'speaker' not in hypothesis.T.keys() or len(hypothesis.unique('speaker')) <= 1, 'Only single-speaker transcripts are supported'

    _reference = sort_and_validate(
        reference,
        reference_sort,
        reference_pseudo_word_level_timing,
        'reference'
    )
    _hypothesis = sort_and_validate(
        hypothesis,
        hypothesis_sort,
        hypothesis_pseudo_word_level_timing,
        'hypothesis'
    )
    _hypothesis = apply_collar(_hypothesis, collar)

    er = _time_constrained_siso_error_rate(_reference, _hypothesis)

    # pseudo_word_level_timing and collar change the time stamps,
    # hence calculate the overlap with the original time stamps
    er = replace(
        er,
        reference_self_overlap=get_self_overlap(reference),
        hypothesis_self_overlap=get_self_overlap(hypothesis),
    )
    return er


def time_constrained_minimum_permutation_word_error_rate(
        reference: 'SegLST',
        hypothesis: 'SegLST',
        *,
        reference_pseudo_word_level_timing='character_based',
        hypothesis_pseudo_word_level_timing='character_based_points',
        collar: int = 0,
        reference_sort='segment',
        hypothesis_sort='segment',
) -> CPErrorRate:
    """
    Time-constrained minimum permutation word error rate for single-speaker
    transcripts.

    Args:
        reference: reference transcript
        hypothesis: hypothesis transcript
        reference_pseudo_word_level_timing: strategy for pseudo-word level
            timing for reference
        hypothesis_pseudo_word_level_timing: strategy for pseudo-word level
            timing for hypothesis
        collar: collar applied to hypothesis pseudo-word level timings
        reference_sort: How to sort the reference. Options: 'segment', 'word',
            True, False. See below
        hypothesis_sort: How to sort the reference. Options: 'segment', 'word',
            True, False. See below
        
    Reference / Hypothesis sorting options:
    - True: sort by segment start time and assert that the word-level timings
            are sorted by start time
    - False: do not sort and don't check word order
    - 'segment': sort by segment start time and don't check word order
    - 'word': sort by word start time
    """
    from meeteval.wer.wer.cp import _cp_error_rate

    reference = asseglst(
        reference,
        required_keys=('start_time', 'end_time', 'words', 'speaker'),
        py_convert=None
    )
    hypothesis = asseglst(
        hypothesis,
        required_keys=('start_time', 'end_time', 'words', 'speaker'),
        py_convert=None
    )

    reference = reference.groupby('speaker')
    hypothesis = hypothesis.groupby('speaker')

    # Compute self-overlap for ref and hyp before converting to words and
    # applying the collar. This is required later
    # Add zero to work with empty reference and/or hypothesis
    reference_self_overlap = sum(
        [get_self_overlap(v) for v in reference.values()]
    ) + SelfOverlap.zero()
    hypothesis_self_overlap = sum(
        [get_self_overlap(v) for v in hypothesis.values()]
    ) + SelfOverlap.zero()

    # Convert segments into lists of words and word-level timings
    reference = {
        k: sort_and_validate(
            v,
            reference_sort,
            reference_pseudo_word_level_timing,
            f'reference speaker "{k}"'
        )
        for k, v in reference.items()
    }
    hypothesis = {
        k: sort_and_validate(
            v,
            hypothesis_sort,
            hypothesis_pseudo_word_level_timing,
            f'hypothesis speaker "{k}"'
        )
        for k, v in hypothesis.items()
    }

    reference = SegLST.merge(*reference.values())
    hypothesis = SegLST.merge(*hypothesis.values())

    hypothesis = apply_collar(hypothesis, collar)

    # Convert into integer representation to save some computation later.
    # `'words'` contains a single word only.
    sym2int = {v: i for i, v in enumerate([
        segment['words']
        for segment in itertools.chain(reference, hypothesis)
        if segment['words']
    ], start=1)}
    sym2int[''] = 0

    reference = reference.map(lambda s: {**s, 'words': sym2int[s['words']]})
    hypothesis = hypothesis.map(lambda s: {**s, 'words': sym2int[s['words']]})

    er = _cp_error_rate(
        reference, hypothesis,
        distance_fn=time_constrained_siso_levenshtein_distance,
        siso_error_rate=_time_constrained_siso_error_rate,
    )
    er = replace(
        er,
        reference_self_overlap=reference_self_overlap,
        hypothesis_self_overlap=hypothesis_self_overlap,
    )
    return er


tcp_word_error_rate = time_constrained_minimum_permutation_word_error_rate


def tcp_word_error_rate_multifile(
        reference, hypothesis,
        reference_pseudo_word_level_timing='character_based',
        hypothesis_pseudo_word_level_timing='character_based_points',
        collar: int = 0,
        reference_sort='segment',
        hypothesis_sort='segment',
) -> 'dict[str, CPErrorRate]':
    """
    Computes the tcpWER for each example in the reference and hypothesis files.
    See `time_constrained_minimum_permutation_word_error_rate` for details.
    
    To compute the overall WER, use
    `sum(tcp_word_error_rate_multifile(r, h).values())`.
    """
    from meeteval.io.seglst import apply_multi_file
    r = apply_multi_file(lambda r, h: time_constrained_minimum_permutation_word_error_rate(
        r, h,
        reference_pseudo_word_level_timing=reference_pseudo_word_level_timing,
        hypothesis_pseudo_word_level_timing=hypothesis_pseudo_word_level_timing,
        collar=collar,
        reference_sort=reference_sort,
        hypothesis_sort=hypothesis_sort,
    ), reference, hypothesis)
    return r


def index_alignment_to_kaldi_alignment(alignment, reference, hypothesis, eps='*'):
    return [
        (eps if a is None else reference[a], eps if b is None else hypothesis[b])
        for a, b in alignment
    ]


def align(
        reference: SegLST, hypothesis: SegLST,
        *,
        reference_pseudo_word_level_timing='character_based',
        hypothesis_pseudo_word_level_timing='character_based_points',
        collar: int = 0,
        style='words',
        reference_sort='segment',
        hypothesis_sort='segment',
):
    """
    Align two transcripts, similar to `kaldialign.align`, but with time
    constraint.

    Note that empty segments are ignored / skipped for the alignment.

    Args:
        reference: reference transcript
        hypothesis: hypothesis transcript
        reference_pseudo_word_level_timing: strategy for pseudo-word level
                timing for reference
        hypothesis_pseudo_word_level_timing: strategy for pseudo-word level
                timing for hypothesis
        collar: collar applied to hypothesis pseudo-word level timings
        style: Alignment output style. Can be one of
            - 'words' or 'kaldi': Output in the style of `kaldialign.align`
            - 'index': Output indices of the reference and hypothesis words
                instead of the words. Empty segments are included in the index,
                so aligning `('', 'a')` would give index `1` for `'a'`. Note
                that the indices index words, not segments, and that word
                indices do not necessarily correspond to the index of the
                segment in the input. If you want the indices to be valid for
                your input, make sure to pass word-level timings and set
                `reference_pseudo_word_level_timing=None` and/or
                `hypothesis_pseudo_word_level_timing=None`.
            - 'seglst': Output the (seglst) segments for each word. Note: Empty
                segments are ignored
        reference_sort: How to sort the reference. Options: 'segment', 'word',
            True, False. See below
        hypothesis_sort: How to sort the reference. Options: 'segment', 'word',
            True, False. See below
        
    Reference / Hypothesis sorting options:
    - True: sort by segment start time and assert that the word-level timings
            are sorted by start time
    - False: do not sort and don't check word order
    - 'segment': sort by segment start time and don't check word order
    - 'word': sort by word start time

    >>> from pprint import pprint
    >>> align(
    ... [{'words': 'a', 'start_time': 0, 'end_time': 1}, {'words': 'b', 'start_time': 1, 'end_time': 2}, {'words': 'c', 'start_time': 2, 'end_time': 3}],
    ... [{'words': 'a', 'start_time': 0, 'end_time': 1}, {'words': 'b', 'start_time': 1, 'end_time': 2}, {'words': 'c', 'start_time': 3, 'end_time': 4}])
    [('a', 'a'), ('b', 'b'), ('c', '*'), ('*', 'c')]
    >>> align(
    ... [{'words': 'a', 'start_time': 0, 'end_time': 1}, {'words': 'b', 'start_time': 1, 'end_time': 2}, {'words': 'c', 'start_time': 2, 'end_time': 3}],
    ... [{'words': 'a', 'start_time': 0, 'end_time': 1}, {'words': 'b', 'start_time': 1, 'end_time': 2}, {'words': 'c', 'start_time': 3, 'end_time': 4}],
    ... collar=1)
    [('a', 'a'), ('b', 'b'), ('c', 'c')]
    >>> align(
    ... [{'words': 'a b', 'start_time': 0, 'end_time': 1}, {'words': 'c', 'start_time': 1, 'end_time': 2}, {'words': 'd e', 'start_time': 2, 'end_time': 3}],
    ... [{'words': 'a', 'start_time': 0, 'end_time': 1}, {'words': 'b c', 'start_time': 1, 'end_time': 2}, {'words': 'e f', 'start_time': 3, 'end_time': 4}], collar=1)
    [('a', 'a'), ('b', 'b'), ('c', 'c'), ('d', '*'), ('e', 'e'), ('*', 'f')]
    >>> align(
    ... [{'words': 'a b', 'start_time': 0, 'end_time': 1}, {'words': 'c', 'start_time': 1, 'end_time': 2}, {'words': 'd e', 'start_time': 2, 'end_time': 3}],
    ... [{'words': 'a', 'start_time': 0, 'end_time': 1}, {'words': 'b c', 'start_time': 1, 'end_time': 2}, {'words': 'e f', 'start_time': 3, 'end_time': 4}],
    ... collar=1, style='index')
    [(0, 0), (1, 1), (2, 2), (3, None), (4, 3), (None, 4)]
    >>> pprint(align(
    ... [{'words': 'a', 'start_time': 0, 'end_time': 1}, {'words': 'b', 'start_time': 1, 'end_time': 2}, {'words': 'c', 'start_time': 2, 'end_time': 3}],
    ... [{'words': 'a', 'start_time': 0, 'end_time': 1}, {'words': 'b', 'start_time': 1, 'end_time': 2}, {'words': 'c', 'start_time': 3, 'end_time': 4}], style='seglst'))
    [({'end_time': 1, 'start_time': 0, 'words': 'a'},
      {'end_time': 0.5, 'start_time': 0.5, 'words': 'a'}),
     ({'end_time': 2, 'start_time': 1, 'words': 'b'},
      {'end_time': 1.5, 'start_time': 1.5, 'words': 'b'}),
     ({'end_time': 3, 'start_time': 2, 'words': 'c'}, None),
     (None, {'end_time': 3.5, 'start_time': 3.5, 'words': 'c'})]

    Empty segments / words are ignored
     >>> pprint(align(
     ... [{'words': '', 'start_time': 0, 'end_time': 1}, {'words': 'a', 'start_time': 1, 'end_time': 2}],
     ... [{'words': 'a', 'start_time': 1, 'end_time': 2}, {'words': '', 'start_time': 2, 'end_time': 3}]
     ... ))
     [('a', 'a')]
    >>> pprint(align(
    ... [{'words': '', 'start_time': 0, 'end_time': 1}, {'words': 'a', 'start_time': 1, 'end_time': 2}],
    ... [{'words': 'a', 'start_time': 1, 'end_time': 2}, {'words': '', 'start_time': 2, 'end_time': 3}],
    ... style='index'))
    [(1, 0)]

    Any additional attributes are passed through when style='seglst'
    >>> align([{'words': 'a', 'start_time': 0, 'end_time': 1, 'custom_data': [1, 2, 3]}], [], style='seglst')
    [({'words': 'a', 'start_time': 0, 'end_time': 1, 'custom_data': [1, 2, 3]}, None)]
    >>> from meeteval.io.stm import STM, STMLine
    >>> pprint(align(STM([STMLine.parse('ex 1 A 0 1 a', parse_float=float)]), STM([STMLine.parse('ex 1 B 0 1 a', parse_float=float)]), style='seglst'))
    [({'channel': 1,
       'end_time': 1,
       'session_id': 'ex',
       'speaker': 'A',
       'start_time': 0,
       'words': 'a'},
      {'channel': 1,
       'end_time': 0.5,
       'session_id': 'ex',
       'speaker': 'B',
       'start_time': 0.5,
       'words': 'a'})]
    """
    reference = asseglst(
        reference,
        required_keys=('start_time', 'end_time', 'words'),
        py_convert=None
    )
    hypothesis = asseglst(
        hypothesis,
        required_keys=('start_time', 'end_time', 'words'),
        py_convert=None
    )
    reference = sort_and_validate(
        reference,
        reference_sort,
        reference_pseudo_word_level_timing,
        'reference'
    )
    hypothesis = sort_and_validate(
        hypothesis,
        hypothesis_sort,
        hypothesis_pseudo_word_level_timing,
        'hypothesis'
    )

    # Add index for tracking across filtering operations. This is only required
    # for the index style since all other styles can be constructed from seglst
    # without the index. Especially for `style = 'seglst'` we want to keep
    # identity
    if style == 'index':
        reference = SegLST(
            [{**s, '__align_index': i} for i, s in enumerate(reference)]
        )
        hypothesis = SegLST(
            [{**s, '__align_index': i} for i, s in enumerate(hypothesis)]
        )

    # Ignore empty segments
    reference = reference.filter(lambda s: s['words'])
    hypothesis = hypothesis.filter(lambda s: s['words'])

    hypothesis_ = apply_collar(hypothesis, collar=collar)

    # Compute the alignment with Cython code
    reference_words = reference.T['words']
    reference_timing = list(zip(reference.T['start_time'], reference.T['end_time']))
    hypothesis_words = hypothesis_.T['words']
    hypothesis_timing = list(zip(hypothesis_.T['start_time'], hypothesis_.T['end_time']))

    from meeteval.wer.matching.cy_levenshtein import time_constrained_levenshtein_distance_with_alignment
    alignment = time_constrained_levenshtein_distance_with_alignment(
        reference_words, hypothesis_words, reference_timing, hypothesis_timing
    )['alignment']

    # Convert "local" (relative to filtered words) indices to segments
    alignment = [
        (None if a is None else reference[a], None if b is None else hypothesis[b])
        for a, b in alignment
    ]

    if style == 'index':
        # Use the "global" (before filtering) index so that it corresponds to
        # the input when the input already consists of words
        alignment = [
            (None if a is None else a['__align_index'],
             None if b is None else b['__align_index'])
            for a, b in alignment
        ]
    elif style in ('kaldi', 'words'):
        alignment = [
            ('*' if a is None else a['words'],
             '*' if b is None else b['words'])
            for a, b in alignment
        ]
    elif style == 'seglst':
        pass  # Already in correct format
    elif style != 'index':
        raise ValueError(f'Unknown alignment style: {style}')

    return alignment
