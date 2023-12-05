"""
This file contains the time-constrained minimum permutation word error rate
"""
import itertools
import typing
from dataclasses import dataclass, replace

from meeteval.io.stm import STM
from meeteval.io.seglst import Tidy, tidy_map, convert_to_tidy, tidy_args
from meeteval.wer.wer.error_rate import ErrorRate, SelfOverlap
from meeteval.wer.wer.cp import CPErrorRate
from typing import List, Dict
import logging

from meeteval.wer.utils import _values, _map, _keys

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
    'tcp_word_error_rate_stm'
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
    return [(interval[0] + i * interval_length, interval[0] + (i + 1) * interval_length) for i in range(count)]


def equidistant_points(interval, words):
    """Places `count` points (intervals of size zero) in `interval` with equal distance"""
    count = len(words)
    if count == 0:
        return []
    elif count == 1:
        return [((interval[0] + interval[1]) / 2,) * 2]
    interval_length = (interval[1] - interval[0]) / count

    return [(interval[0] + (i + 0.5) * interval_length,) * 2 for i in range(count)]


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
    return [(interval[0] + character_length * start, interval[0] + character_length * end) for start, end in
            zip([0] + list(end_points[:-1]), end_points)]


def character_based_points(interval, words):
    """Places points in the center of the character-based intervals"""
    intervals = character_based(interval, words)
    intervals = [((interval[1] + interval[0]) / 2,) * 2 for interval in intervals]
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
                f'No segmentation strategy was specified (i.e., exactly one timestamp per word is required), '
                f'but a segment ({interval}) contains no word.'
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
            f'This can lead to contradictions between pseudo-word-level timings and word order. '
            f'An exception will be raised later when such a contradiction occurs. '
        )


def _time_constrained_siso_error_rate(
        reference: Tidy, hypothesis: Tidy, prune='auto'
):
    from meeteval.wer.matching.cy_levenshtein import time_constrained_levenshtein_distance_with_alignment

    reference_words = [s['words'] for s in reference if s['words']]
    reference_timing = [(s['start_time'], s['end_time']) for s in reference if s['words']]
    hypothesis_words = [s['words'] for s in hypothesis if s['words']]
    hypothesis_timing = [(s['start_time'], s['end_time']) for s in hypothesis if s['words']]

    result = time_constrained_levenshtein_distance_with_alignment(
        reference_words, hypothesis_words, reference_timing, hypothesis_timing, prune=prune
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
    transcript: List[str]
    timings: List[typing.Tuple[float, float]]

    def segments(self):
        return [{
            'words': transcript,
            'start_time': timing[0],
            'end_time': timing[1],
        } for transcript, timing in zip(self.transcript, self.timings)]

    @classmethod
    def convert(cls, d):
        d = convert_to_tidy(d)
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
    def from_segment_dicts(cls, data: 'List[Segment]') -> 'TimeMarkedTranscript':
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
TimeMarkedTranscriptLike = 'TimeMarkedTranscript | STM | List[Segment]'


@tidy_map()
def apply_collar(s: Tidy, collar: float):
    """
    >>> apply_collar([{'start_time': 0, 'end_time': 1}], 1)
    [{'start_time': -1, 'end_time': 2}]
    """
    return s.map(
        lambda s: {**s, 'start_time': s['start_time'] - collar, 'end_time': s['end_time'] + collar})


@tidy_map()
def get_pseudo_word_level_timings(t: Tidy, strategy: str) -> Tidy:
    """
    TODO: Remove TimeMarkedTranscript from testcases

    >>> from IPython.lib.pretty import pprint
    >>> s = TimeMarkedTranscript(['abc b', 'c d e f'], [(0, 4), (4, 8)])
    >>> pprint(get_pseudo_word_level_timings(s, 'full_segment'))
    TimeMarkedTranscript(
        transcript=['abc', 'b', 'c', 'd', 'e', 'f'],
        timings=[(0, 4), (0, 4), (4, 8), (4, 8), (4, 8), (4, 8)]
    )
    >>> pprint(get_pseudo_word_level_timings(s, 'equidistant_points'))
    TimeMarkedTranscript(
        transcript=['abc', 'b', 'c', 'd', 'e', 'f'],
        timings=[(1.0, 1.0),
         (3.0, 3.0),
         (4.5, 4.5),
         (5.5, 5.5),
         (6.5, 6.5),
         (7.5, 7.5)]
    )
    >>> pprint(get_pseudo_word_level_timings(s, 'equidistant_intervals'))
    TimeMarkedTranscript(
        transcript=['abc', 'b', 'c', 'd', 'e', 'f'],
        timings=[(0.0, 2.0),
         (2.0, 4.0),
         (4.0, 5.0),
         (5.0, 6.0),
         (6.0, 7.0),
         (7.0, 8.0)]
    )
    >>> word_level = get_pseudo_word_level_timings(s, 'character_based')
    >>> pprint(word_level)
    TimeMarkedTranscript(
        transcript=['abc', 'b', 'c', 'd', 'e', 'f'],
        timings=[(0.0, 3.0),
         (3.0, 4.0),
         (4.0, 5.0),
         (5.0, 6.0),
         (6.0, 7.0),
         (7.0, 8.0)]
    )
    >>> pprint(get_pseudo_word_level_timings(word_level, 'none'))   # Copies over the timings since word-level timings are already assumed
    TimeMarkedTranscript(
        transcript=['abc', 'b', 'c', 'd', 'e', 'f'],
        timings=[(0.0, 3.0),
         (3.0, 4.0),
         (4.0, 5.0),
         (5.0, 6.0),
         (6.0, 7.0),
         (7.0, 8.0)]
    )
    >>> pprint(get_pseudo_word_level_timings(s, 'character_based_points'))
    TimeMarkedTranscript(
        transcript=['abc', 'b', 'c', 'd', 'e', 'f'],
        timings=[(1.5, 1.5),
         (3.5, 3.5),
         (4.5, 4.5),
         (5.5, 5.5),
         (6.5, 6.5),
         (7.5, 7.5)]
    )
    """
    pseudo_word_level_strategy = pseudo_word_level_strategies[strategy]

    def get_words(s):
        res = []
        words = s['words'].split()
        if not words:  # Make sure that we don't drop a speaker
            words = ['']
        for w, (start, end) in zip(words, pseudo_word_level_strategy((s['start_time'], s['end_time']), words)):
            res.append({**s, 'words': w, 'start_time': start, 'end_time': end})
        return res

    return t.flatmap(get_words)


@tidy_map()
def remove_overlaps(
        t: Tidy,
        max_overlap: float = 0.4,
        warn_message: str = None,
) -> Tidy:
    """
    Remove overlaps between words or segments in a transcript.

    Args:
        s: TimeMarkedTranscript
        max_overlap: maximum allowed relative overlap between words or segments.
            Raises a `ValueError` when more overlap is found.
        warn_message: if not None, a warning is printed when overlaps are corrected.
    """
    last = None

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

    return t.map(correct)


def sort_and_validate(segments: Tidy, sort, pseudo_word_level_timing, name):
    """
    Args:
        segments:
        sort: How to sort words/segments. Options:
            - `True`: sort by segment start time and assert that the word-level timings are sorted by start time
            - `False`: do not sort and do not check word order
            - `'segment'`: sort segments by start time and do not check word order
            - `'word'`: sort words by start time
        pseudo_word_level_timing:
        name:

    >>> segments = [{'words': 'c d', 'start_time': 1, 'end_time': 3}, {'words': 'a b', 'start_time': 0, 'end_time': 3}]
    >>> sort_and_validate(segments, True, 'character_based', 'test')
    Traceback (most recent call last):
    ...
    ValueError: The order of word-level timings contradicts the segment-level order in test: 2 of 4 times.
    Consider setting sort to False or "segment" or "word".
    >>> sort_and_validate(segments, False, 'character_based', 'test')
    ([{'words': 'c', 'start_time': 1.0, 'end_time': 2.0}, {'words': 'd', 'start_time': 2.0, 'end_time': 3.0}, {'words': 'a', 'start_time': 0.0, 'end_time': 1.5}, {'words': 'b', 'start_time': 1.5, 'end_time': 3.0}], False)
    >>> sort_and_validate(segments, 'segment', 'character_based', 'test')
    ([{'words': 'a', 'start_time': 0.0, 'end_time': 1.5}, {'words': 'b', 'start_time': 1.5, 'end_time': 3.0}, {'words': 'c', 'start_time': 1.0, 'end_time': 2.0}, {'words': 'd', 'start_time': 2.0, 'end_time': 3.0}], False)
    >>> sort_and_validate(segments, 'word', 'character_based', 'test')
    ([{'words': 'a', 'start_time': 0.0, 'end_time': 1.5}, {'words': 'c', 'start_time': 1.0, 'end_time': 2.0}, {'words': 'b', 'start_time': 1.5, 'end_time': 3.0}, {'words': 'd', 'start_time': 2.0, 'end_time': 3.0}], True)
    """
    if sort not in (True, False, 'segment', 'word'):
        raise ValueError(f'Invalid value for sort: {sort}. Choose one of True, False, "segment", "word"')

    for s in segments:
        if s['end_time'] < s['start_time']:
            raise ValueError(f'The end time of an interval must be larger than the start time. Found {s} in {name}')

    if sort in (True, 'segment', 'word'):
        segments = segments.sorted('start_time')

    words = get_pseudo_word_level_timings(segments, pseudo_word_level_timing)

    # Check whether words are sorted by start time
    words_sorted = words.sorted('start_time')
    # TODO: only check relevant keys? That would speed things up if the user provides a lot of custom keys
    if words_sorted != words:
        contradictions = [a != b for a, b in zip(words_sorted, words)]
        msg = (
            f'The order of word-level timings contradicts the segment-level order in {name}: '
            f'{sum(contradictions)} of {len(contradictions)} times.'
        )
        if sort is not True:
            logger.warning(msg)
        else:
            raise ValueError(f'{msg}\nConsider setting sort to False or "segment" or "word".')

    if sort == 'word':
        words = words_sorted

    return words


@tidy_args('start_time', 'end_time')
def get_self_overlap(d: Tidy):
    latest_end = 0
    self_overlap = 0
    total = 0
    for t in sorted(d, key=lambda x: x['start_time']):
        if latest_end > t['start_time']:
            self_overlap += min(latest_end, t['end_time']) - t['start_time']
        total += max(0, t['end_time'] - latest_end)
        latest_end = max(latest_end, t['end_time'])
    return SelfOverlap(self_overlap, total)


def time_constrained_siso_levenshtein_distance(reference: 'Tidy', hypothesis: 'Tidy') -> int:
    from meeteval.wer.matching.cy_levenshtein import time_constrained_levenshtein_distance

    reference_words = [s['words'] for s in reference if s['words']]
    reference_timing = [(s['start_time'], s['end_time']) for s in reference if s['words']]
    hypothesis_words = [s['words'] for s in hypothesis if s['words']]
    hypothesis_timing = [(s['start_time'], s['end_time']) for s in hypothesis if s['words']]

    return time_constrained_levenshtein_distance(reference_words, hypothesis_words, reference_timing, hypothesis_timing)


def time_constrained_siso_word_error_rate(
        reference: 'TidyData',
        hypothesis: 'TidyData',
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
        reference_pseudo_word_level_timing: strategy for pseudo-word level timing for reference
        hypothesis_pseudo_word_level_timing: strategy for pseudo-word level timing for hypothesis
        collar: collar applied to hypothesis pseudo-word level timings
        reference_sort: How to sort the reference. Options: 'segment', 'word', True, False. See below
        hypothesis_sort: How to sort the reference. Options: 'segment', 'word', True, False. See below
        
    Reference / Hypothesis sorting options:
    - True: sort by segment start time and assert that the word-level timings are sorted by start time
    - False: do not sort and don't check word order
    - 'segment': sort by segment start time and don't check word order
    - 'word': sort by word start time

    >>> time_constrained_siso_word_error_rate(TimeMarkedTranscript(['a b', 'c d'], [(0,2), (0,2)]), TimeMarkedTranscript(['a'], [(0,1)]))
    ErrorRate(error_rate=0.75, errors=3, length=4, insertions=0, deletions=3, substitutions=0, reference_self_overlap=SelfOverlap(overlap_rate=1.0, overlap_time=2, total_time=2), hypothesis_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1))
    """
    # TODO: assert single speaker?

    _reference = sort_and_validate(reference, reference_sort, reference_pseudo_word_level_timing)
    _hypothesis = sort_and_validate(hypothesis, hypothesis_sort, hypothesis_pseudo_word_level_timing)
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


@tidy_args('speaker', 'start_time', 'end_time', 'words')
def time_constrained_minimum_permutation_word_error_rate(
        reference: 'Tidy',
        hypothesis: 'Tidy',
        *,
        reference_pseudo_word_level_timing='character_based',
        hypothesis_pseudo_word_level_timing='character_based_points',
        collar: int = 0,
        reference_sort='segment',
        hypothesis_sort='segment',
) -> CPErrorRate:
    """
    Time-constrained minimum permutation word error rate for single-speaker transcripts.

    Args:
        reference: reference transcript
        hypothesis: hypothesis transcript
        reference_pseudo_word_level_timing: strategy for pseudo-word level timing for reference
        hypothesis_pseudo_word_level_timing: strategy for pseudo-word level timing for hypothesis
        collar: collar applied to hypothesis pseudo-word level timings
        reference_sort: How to sort the reference. Options: 'segment', 'word', True, False. See below
        hypothesis_sort: How to sort the reference. Options: 'segment', 'word', True, False. See below
        
    Reference / Hypothesis sorting options:
    - True: sort by segment start time and assert that the word-level timings are sorted by start time
    - False: do not sort and don't check word order
    - 'segment': sort by segment start time and don't check word order
    - 'word': sort by word start time
    """
    from meeteval.wer.wer.cp import _cp_error_rate

    reference = reference.groupby('speaker')
    hypothesis = hypothesis.groupby('speaker')

    # Compute self-overlap for ref and hyp before converting to words and applying the collar.
    # This is required later
    reference_self_overlap = sum([get_self_overlap(v) for v in reference.values()], start=SelfOverlap(0, 0))
    hypothesis_self_overlap = sum([get_self_overlap(v) for v in hypothesis.values()], start=SelfOverlap(0, 0))

    # Convert segments into lists of words and word-level timings
    reference = {k: sort_and_validate(v, reference_sort, reference_pseudo_word_level_timing, f'reference speaker "{k}"')
                 for k, v in reference.items()}
    hypothesis = {
        k: sort_and_validate(v, hypothesis_sort, hypothesis_pseudo_word_level_timing, f'hypothesis speaker "{k}"') for
        k, v in hypothesis.items()}

    reference = Tidy.merge(*reference.values())
    hypothesis = Tidy.merge(*hypothesis.values())

    hypothesis = apply_collar(hypothesis, collar)

    # Convert into integer representation to save some computation later. `'words'` contains a single word only.
    sym2int = {v: i for i, v in enumerate({
        segment['words'] for segment in itertools.chain(reference, hypothesis)
    })}

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


def tcp_word_error_rate_stm(
        reference_stm: 'STM', hypothesis_stm: 'STM',
        reference_pseudo_word_level_timing='character_based',
        hypothesis_pseudo_word_level_timing='character_based_points',
        collar: int = 0,
        reference_sort='segment',
        hypothesis_sort='segment',
) -> 'Dict[str, CPErrorRate]':
    """
    Computes the tcpWER for each example in the reference and hypothesis STM files.
    See `time_constrained_minimum_permutation_word_error_rate` for details.
    
    To compute the overall WER, use `sum(tcp_word_error_rate_stm(r, h).values())`.
    """
    from meeteval.io.stm import apply_stm_multi_file
    r = apply_stm_multi_file(lambda r, h: time_constrained_minimum_permutation_word_error_rate(
        r, h,
        reference_pseudo_word_level_timing=reference_pseudo_word_level_timing,
        hypothesis_pseudo_word_level_timing=hypothesis_pseudo_word_level_timing,
        collar=collar,
        reference_sort=reference_sort,
        hypothesis_sort=hypothesis_sort,
    ), reference_stm, hypothesis_stm)
    return r


def index_alignment_to_kaldi_alignment(alignment, reference, hypothesis, eps='*'):
    return [
        (eps if a is None else reference[a], eps if b is None else hypothesis[b])
        for a, b in alignment
    ]


@tidy_args('words', 'start_time', 'end_time')
def align(
        reference: Tidy, hypothesis: Tidy,
        *,
        reference_pseudo_word_level_timing='character_based',
        hypothesis_pseudo_word_level_timing='character_based_points',
        collar: int = 0,
        style='words',
        reference_sort='segment',
        hypothesis_sort='segment',
):
    """
    Align two time-marked transcripts, similar to `kaldialign.align`, but with time constriant.

    Args:
        reference: reference transcript
        hypothesis: hypothesis transcript
        reference_pseudo_word_level_timing: strategy for pseudo-word level timing for reference
        hypothesis_pseudo_word_level_timing: strategy for pseudo-word level timing for hypothesis
        collar: collar applied to hypothesis pseudo-word level timings
        style: Alignment output style. Can be one of
            - 'words' or 'kaldi': Output in the style of `kaldialign.align`
            - 'index': Output indices of the reference and hypothesis words instead of the words
            - 'tidy': Output the (tidy) segments for each word
        reference_sort: How to sort the reference. Options: 'segment', 'word', True, False. See below
        hypothesis_sort: How to sort the reference. Options: 'segment', 'word', True, False. See below
        
    Reference / Hypothesis sorting options:
    - True: sort by segment start time and assert that the word-level timings are sorted by start time
    - False: do not sort and don't check word order
    - 'segment': sort by segment start time and don't check word order
    - 'word': sort by word start time

    >>> from pprint import pprint
    >>> align(TimeMarkedTranscript('a b c'.split(), [(0,1), (1,2), (2,3)]), TimeMarkedTranscript('a b c'.split(), [(0,1), (1,2), (3,4)]))
    [('a', 'a'), ('b', 'b'), ('c', '*'), ('*', 'c')]
    >>> align(TimeMarkedTranscript('a b c'.split(), [(0,1), (1,2), (2,3)]), TimeMarkedTranscript('a b c'.split(), [(0,1), (1,2), (3,4)]), collar=1)
    [('a', 'a'), ('b', 'b'), ('c', 'c')]
    >>> align(TimeMarkedTranscript(['a b', 'c', 'd e'], [(0,1), (1,2), (2,3)]), TimeMarkedTranscript(['a', 'b c', 'e f'], [(0,1), (1,2), (3,4)]), collar=1)
    [('a', 'a'), ('b', 'b'), ('c', 'c'), ('d', '*'), ('e', 'e'), ('*', 'f')]
    >>> align(TimeMarkedTranscript(['a b', 'c', 'd e'], [(0,1), (1,2), (2,3)]), TimeMarkedTranscript(['a', 'b c', 'e f'], [(0,1), (1,2), (3,4)]), collar=1, style='index')
    [(0, 0), (1, 1), (2, 2), (3, None), (4, 3), (None, 4)]
    >>> pprint(align(TimeMarkedTranscript('a b c'.split(), [(0,1), (1,2), (2,3)]), TimeMarkedTranscript('a b c'.split(), [(0,1), (1,2), (3,4)]), style='tidy'))
    [({'end_time': 1, 'start_time': 0, 'words': 'a'},
      {'end_time': 0.5, 'start_time': 0.5, 'words': 'a'}),
     ({'end_time': 2, 'start_time': 1, 'words': 'b'},
      {'end_time': 1.5, 'start_time': 1.5, 'words': 'b'}),
     ({'end_time': 3, 'start_time': 2, 'words': 'c'}, None),
     (None, {'end_time': 3.5, 'start_time': 3.5, 'words': 'c'})]

    Any additional attributes are passed through when style='tidy'
    >>> align([{'words': 'a', 'start_time': 0, 'end_time': 1, 'custom_data': [1, 2, 3]}], [], style='tidy')
    [({'words': 'a', 'start_time': 0, 'end_time': 1, 'custom_data': [1, 2, 3]}, None)]
    >>> from meeteval.io.stm import STM, STMLine
    >>> pprint(align(STM([STMLine.parse('ex 1 A 0 1 a')]), STM([STMLine.parse('ex 1 B 0 1 a')]), style='tidy'))
    [({'end_time': 1,
       'session_id': 'ex',
       'speaker': 'A',
       'start_time': 0,
       'words': 'a'},
      {'end_time': 0.5,
       'session_id': 'ex',
       'speaker': 'B',
       'start_time': 0.5,
       'words': 'a'})]

    >>> align([{'start_time': 40.3219375, 'end_time': 40.611802364864865, 'words': 'THEN'}], [{'start_time': 16.208375, 'end_time': 20.080375, 'words': ''}], style='tidy')
    """
    reference = sort_and_validate(reference, reference_sort, reference_pseudo_word_level_timing, 'reference')
    hypothesis = sort_and_validate(hypothesis, hypothesis_sort, hypothesis_pseudo_word_level_timing, 'hypothesis')
    hypothesis_ = apply_collar(hypothesis, collar=collar)

    # TODO: Handle empty segments correctly
    reference_words = [s['words'] for s in reference]
    reference_timing = [(s['start_time'], s['end_time']) for s in reference]
    hypothesis_words = [s['words'] for s in hypothesis_]
    hypothesis_timing = [(s['start_time'], s['end_time']) for s in hypothesis_]

    from meeteval.wer.matching.cy_levenshtein import time_constrained_levenshtein_distance_with_alignment
    alignment = time_constrained_levenshtein_distance_with_alignment(
        reference_words, hypothesis_words, reference_timing, hypothesis_timing
    )['alignment']

    if style in ('kaldi', 'words'):
        alignment = index_alignment_to_kaldi_alignment(alignment, reference_words, hypothesis_words)
    elif style == 'tidy':
        alignment = [
            (None if a is None else reference[a],
             None if b is None else hypothesis[b])
            for a, b in alignment
        ]
    elif style != 'index':
        raise ValueError(f'Unknown alignment style: {style}')

    return alignment
