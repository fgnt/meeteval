"""
This file contains the time-constrained minimum permutation word error rate
"""
import itertools
import typing
from dataclasses import dataclass

from meeteval.io.stm import STM
from meeteval.wer.wer.error_rate import ErrorRate
from meeteval.wer.wer.cp import CPErrorRate
from typing import List, Dict

from meeteval.wer.utils import _values, _map, _keys

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
        reference, hypothesis, reference_timing, hypothesis_timing
):
    from meeteval.wer.matching.cy_levenshtein import time_constrained_levenshtein_distance_with_alignment

    result = time_constrained_levenshtein_distance_with_alignment(
        reference, hypothesis, reference_timing, hypothesis_timing,
    )

    return ErrorRate(
        result['total'],
        len(reference),
        insertions=result['insertions'],
        deletions=result['deletions'],
        substitutions=result['substitutions'],
    )


@dataclass
class TimeMarkedTranscript:
    transcript: List[str]
    timings: List[typing.Tuple[float, float]]

    def check(self, msg):
        _check_timing_annotations(self.timings, msg)

    def has_self_overlaps(self):
        last_end = 0
        for t in sorted(self.timings):
            if last_end > t[0]:
                return True
            last_end = t[1]
        return False

    def get_self_overlap_amount(self):
        last_end = 0
        self_overlap = 0
        for t in self.timings:
            if last_end > t[0]:
                self_overlap += last_end - t[0]
            last_end = t[1]
        return self_overlap

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
        time_marked_transcript.check(next(iter(speaker_ids)))
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
        time_marked_transcript.check(None)
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


def get_pseudo_word_level_timings(
        s: TimeMarkedTranscript,
        strategy: str,
        collar: float = 0,
) -> TimeMarkedTranscript:
    """
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

    all_words = []
    word_level_timings = []

    for words, interval in zip(s.transcript, s.timings):
        words = words.split()  # Get words form segment
        segment_timings = pseudo_word_level_strategy(interval, words)
        word_level_timings.extend(segment_timings)
        all_words.extend(words)
        assert len(words) == len(segment_timings), (words, segment_timings)

    word_level_timings = [(max(t[0] - collar, 0), t[1] + collar) for t in word_level_timings]
    return TimeMarkedTranscript(all_words, word_level_timings)


def remove_overlaps(
        s: TimeMarkedTranscript,
        max_overlap: float = 0.4,
        warn_message: str = None,
) -> TimeMarkedTranscript:
    """
    Remove overlaps between words or segments in a transcript.

    Args:
        s: TimeMarkedTranscript
        max_overlap: maximum allowed relative overlap between words or segments.
            Raises a `ValueError` when more overlap is found.
        warn_message: if not None, a warning is printed when overlaps are corrected.
    """
    corrected_timings = []
    for t in s.timings:
        if corrected_timings and corrected_timings[-1][1] > t[0]:
            if warn_message is not None:
                import warnings
                warnings.warn(warn_message)
            last = corrected_timings[-1]
            overlap = last[1] - t[0]
            if overlap > max_overlap * (t[1] - last[0]):
                import numpy as np
                raise ValueError(
                    f'Overlapping segments exceed max allowed relative overlap. '
                    f'Segment {last} overlaps with {t}. '
                    f'{overlap} > {max_overlap * (t[1] - last[0])} '
                    f'relative overlap: {np.divide(overlap, (t[1] - last[-1]))}'
                )
            center = (last[-1] + t[0]) / 2
            corrected_timings[-1] = (last[0], center)
            t = (center, t[1])

            assert t[1] > t[0], t
            assert last[1] > last[0], last

        corrected_timings.append(t)
    return TimeMarkedTranscript(s.transcript, corrected_timings)


def sort_segments(s: TimeMarkedTranscript):
    import numpy as np
    order = np.argsort(np.asarray(s.timings)[:, 0])
    return TimeMarkedTranscript(
        [s.transcript[int(i)] for i in order],
        [s.timings[int(i)] for i in order],
    )


def _prepare_transcript(segments, sort, pseudo_word_level_timing, collar=0):
    # Check that all timings are valid
    if len(segments.transcript) != len(segments.timings):
        raise ValueError(
            f'Number of words does not match number of timings {len(segments.transcript)} != {len(segments.timings)}')

    for t in segments.timings:
        if t[1] < t[0]:
            raise ValueError(f'The end time of an interval must be larger than the start time. Found {t}')

    prune = True

    # Find self-overlap. Print warning if self-overlap is found.
    if segments.has_self_overlaps():
        self_overlap_amount = segments.get_self_overlap_amount()

        import warnings
        warnings.warn(f'Found self-overlaps in transcript. Overlap amount: {self_overlap_amount}')

    assert sort in (True, False, 'segment', 'word')
    if sort in (True, 'segment', 'word'):
        segments = sort_segments(segments)

    words = get_pseudo_word_level_timings(segments, pseudo_word_level_timing, collar)
    # Check whether words are sorted by start time
    words_sorted = sort_segments(words)
    if words_sorted != words:
        if sort is not True:
            import warnings
            warnings.warn(f'The order of word-level timings contradicts the segment-level order')
            prune = False
        else:
            raise ValueError(
                f'The order of word-level timings contradicts the segment-level order!\n'
                f'Consider setting sort to False or "segment" or "word".\n'
            )

    if sort == 'word':
        words = words_sorted
        prune = True  # Pruning always works when words are sorted by start time
    return words, prune


def time_constrained_siso_word_error_rate(
        reference: TimeMarkedTranscriptLike,
        hypothesis: TimeMarkedTranscriptLike,
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
    """
    reference = TimeMarkedTranscript.create(reference)
    hypothesis = TimeMarkedTranscript.create(hypothesis)

    try:
        reference, _ = _prepare_transcript(reference, reference_sort, reference_pseudo_word_level_timing)
    except ValueError as e:
        raise ValueError(f'Invalid reference') from e
    try:
        hypothesis, _ = _prepare_transcript(
            hypothesis, hypothesis_sort,
            hypothesis_pseudo_word_level_timing, collar
        )
    except ValueError as e:
        raise ValueError(f'Invalid hypothesis') from e

    return _time_constrained_siso_error_rate(
        reference.transcript, hypothesis.transcript,
        reference.timings, hypothesis.timings,
    )


def time_constrained_minimum_permutation_word_error_rate(
        reference: 'List[TimeMarkedTranscriptLike] | Dict[str, TimeMarkedTranscriptLike] | STM',
        hypothesis: 'List[TimeMarkedTranscriptLike] | Dict[str, TimeMarkedTranscriptLike] | STM',
        reference_pseudo_word_level_timing='character_based',
        hypothesis_pseudo_word_level_timing='character_based',
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
    from meeteval.wer.matching.cy_levenshtein import time_constrained_levenshtein_distance_v3
    from meeteval.wer.wer.cp import _cp_error_rate

    if isinstance(reference, STM):
        reference = reference.grouped_by_speaker_id()
    if isinstance(hypothesis, STM):
        hypothesis = hypothesis.grouped_by_speaker_id()

    reference = _map(TimeMarkedTranscript.create, reference)
    hypothesis = _map(TimeMarkedTranscript.create, hypothesis)

    # Convert segments into lists of words and word-level timings
    allow_prune = True
    for k in _keys(reference):
        try:
            reference[k], p = _prepare_transcript(reference[k], reference_sort, reference_pseudo_word_level_timing)
            allow_prune = allow_prune and p
        except ValueError as e:
            raise ValueError(f'Invalid reference for speaker {k}') from e

    for k in _keys(hypothesis):
        try:
            hypothesis[k], p = _prepare_transcript(hypothesis[k], hypothesis_sort, hypothesis_pseudo_word_level_timing,
                                                   collar=collar)
            allow_prune = allow_prune and p
        except ValueError as e:
            raise ValueError(f'Invalid hypothesis for speaker {k}') from e

    sym2int = {v: i for i, v in enumerate({
        word for words in itertools.chain(_values(reference), _values(hypothesis)) for word in words.transcript
    })}

    reference = _map(lambda x: TimeMarkedTranscript([sym2int[s] for s in x.transcript], x.timings), reference)
    hypothesis = _map(lambda x: TimeMarkedTranscript([sym2int[s] for s in x.transcript], x.timings), hypothesis)

    return _cp_error_rate(
        reference, hypothesis,
        distance_fn=lambda tt, et: time_constrained_levenshtein_distance_v3(
            tt.transcript, et.transcript, tt.timings, et.timings
        ),
        siso_error_rate=lambda tt, et: _time_constrained_siso_error_rate(
            tt.transcript, et.transcript, tt.timings, et.timings
        ),
        missing=TimeMarkedTranscript([], []),
    )


tcp_word_error_rate = time_constrained_minimum_permutation_word_error_rate


def tcp_word_error_rate_stm(
        reference_stm: 'STM', hypothesis_stm: 'STM',
        reference_pseudo_word_level_timing='character_based',
        hypothesis_pseudo_word_level_timing='character_based',
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
    return apply_stm_multi_file(lambda r, h: time_constrained_minimum_permutation_word_error_rate(
        r, h,
        reference_pseudo_word_level_timing=reference_pseudo_word_level_timing,
        hypothesis_pseudo_word_level_timing=hypothesis_pseudo_word_level_timing,
        collar=collar,
        reference_sort=reference_sort,
        hypothesis_sort=hypothesis_sort,
    ), reference_stm, hypothesis_stm)


def index_alignment_to_kaldi_alignment(alignment, reference, hypothesis, eps='*'):
    return [
        (eps if a is None else reference[a], eps if b is None else hypothesis[b])
        for a, b in alignment
    ]


def align(
        reference: TimeMarkedTranscriptLike, hypothesis: TimeMarkedTranscriptLike,
        reference_pseudo_word_level_timing='character_based',
        hypothesis_pseudo_word_level_timing='character_based',
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
            - 'words_and_times': Output the time interval (pseudo-word-level, without collar) with each word
        reference_sort: How to sort the reference. Options: 'segment', 'word', True, False. See below
        hypothesis_sort: How to sort the reference. Options: 'segment', 'word', True, False. See below
        
    Reference / Hypothesis sorting options:
    - True: sort by segment start time and assert that the word-level timings are sorted by start time
    - False: do not sort and don't check word order
    - 'segment': sort by segment start time and don't check word order
    - 'word': sort by word start time

    >>> align(TimeMarkedTranscript('a b c'.split(), [(0,1), (1,2), (2,3)]), TimeMarkedTranscript('a b c'.split(), [(0,1), (1,2), (3,4)]))
    [('a', 'a'), ('b', 'b'), ('c', '*'), ('*', 'c')]
    >>> align(TimeMarkedTranscript('a b c'.split(), [(0,1), (1,2), (2,3)]), TimeMarkedTranscript('a b c'.split(), [(0,1), (1,2), (3,4)]), collar=1)
    [('a', 'a'), ('b', 'b'), ('c', 'c')]
    >>> align(TimeMarkedTranscript(['a b', 'c', 'd e'], [(0,1), (1,2), (2,3)]), TimeMarkedTranscript(['a', 'b c', 'e f'], [(0,1), (1,2), (3,4)]), collar=1)
    [('a', 'a'), ('b', 'b'), ('c', 'c'), ('d', '*'), ('e', 'e'), ('*', 'f')]
    >>> align(TimeMarkedTranscript(['a b', 'c', 'd e'], [(0,1), (1,2), (2,3)]), TimeMarkedTranscript(['a', 'b c', 'e f'], [(0,1), (1,2), (3,4)]), collar=1, style='index')
    [(0, 0), (1, 1), (2, 2), (3, None), (4, 3), (None, 4)]
    >>> align(TimeMarkedTranscript(['a b', 'c', 'd e'], [(0,1), (1,2), (2,3)]), TimeMarkedTranscript(['a', 'b c', 'e f'], [(0,1), (1,2), (3,4)]), collar=1, style='words_and_times')
    [(('a', (0.0, 0.5)), ('a', (0, 1))), (('b', (0.5, 1.0)), ('b', (1.0, 1.5))), (('c', (1, 2)), ('c', (1.5, 2.0))), (('d', (2.0, 2.5)), ('*', (2.0, 2.5))), (('e', (2.5, 3.0)), ('e', (3.0, 3.5))), (('*', (3.5, 4.0)), ('f', (3.5, 4.0)))]
    """
    reference = TimeMarkedTranscript.create(reference)
    hypothesis = TimeMarkedTranscript.create(hypothesis)

    try:
        reference_, _ = _prepare_transcript(reference, reference_sort, reference_pseudo_word_level_timing)
    except ValueError as e:
        raise ValueError('Invalid reference') from e
    try:
        hypothesis_, _ = _prepare_transcript(hypothesis, hypothesis_sort, hypothesis_pseudo_word_level_timing, collar)
    except ValueError as e:
        raise ValueError('Invalid hypothesis') from e

    from meeteval.wer.matching.cy_levenshtein import time_constrained_levenshtein_distance_with_alignment
    alignment = time_constrained_levenshtein_distance_with_alignment(
        reference_.transcript, hypothesis_.transcript,
        reference_.timings, hypothesis_.timings,
    )['alignment']

    if style in ('kaldi', 'words'):
        alignment = index_alignment_to_kaldi_alignment(alignment, reference_.transcript, hypothesis_.transcript)
    elif style == 'words_and_times':
        hypothesis_, _ = _prepare_transcript(hypothesis, hypothesis_sort, hypothesis_pseudo_word_level_timing, collar=0)
        reference_ = list(zip(reference_.transcript, reference_.timings))
        hypothesis_ = list(zip(hypothesis_.transcript, hypothesis_.timings))

        alignment = [
            (('*', hypothesis_[b][1]) if a is None else reference_[a],
             ('*', reference_[a][1]) if b is None else hypothesis_[b])
            for a, b in alignment
        ]
    elif style != 'index':
        raise ValueError(f'Unknown alignment style: {style}')

    return alignment
