import functools
import typing

import dataclasses

import meeteval
from meeteval.wer.preprocess import preprocess
from meeteval.wer.wer.time_constrained import _time_constrained_siso_error_rate

if typing.TYPE_CHECKING:
    from meeteval.io import STM
    from meeteval.wer import OrcErrorRate

__all__ = [
    'time_constrained_orc_wer',
    'time_constrained_orc_wer_multifile',
    'greedy_time_constrained_orc_wer',
    'greedy_time_constrained_orc_wer_multifile',
]


def time_constrained_orc_wer(
        reference,
        hypothesis,
        collar,
        reference_pseudo_word_level_timing='character_based',
        hypothesis_pseudo_word_level_timing='character_based_points',
        hypothesis_sort='segment',
        reference_sort='segment',
) -> 'OrcErrorRate':
    """
    The time-constrained version of the ORC-WER (tcORC-WER).

    Special cases where the reference or hypothesis is empty
    >>> time_constrained_orc_wer([], [], collar=5)
    OrcErrorRate(errors=0, length=0, insertions=0, deletions=0, substitutions=0, assignment=())
    >>> time_constrained_orc_wer([], [{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': 'a', 'speaker': 'A'}], collar=5)
    OrcErrorRate(errors=1, length=0, insertions=1, deletions=0, substitutions=0, hypothesis_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1), assignment=())
    >>> time_constrained_orc_wer([{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': 'a', 'speaker': 'A'}], [], collar=5)
    OrcErrorRate(error_rate=1.0, errors=1, length=1, insertions=0, deletions=1, substitutions=0, reference_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1), assignment=('dummy',))
    >>> time_constrained_orc_wer([{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': '', 'speaker': 'A'}], [{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': 'a', 'speaker': 'A'}], collar=5)
    OrcErrorRate(errors=1, length=0, insertions=1, deletions=0, substitutions=0, reference_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1), hypothesis_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1), assignment=('A',))
    >>> time_constrained_orc_wer([{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': 'a b', 'speaker': 'A'}], [{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': 'a d', 'speaker': 'A'}], collar=5)
    OrcErrorRate(error_rate=0.5, errors=1, length=2, insertions=0, deletions=0, substitutions=1, reference_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1), hypothesis_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1), assignment=('A',))
    """
    from meeteval.wer.wer.orc import _orc_error_rate

    if reference_sort == 'word':
        raise ValueError(
            'reference_sort="word" is not supported for time-constrained '
            'ORC-WER because the ORC-WER assumes that an utterance appears'
            'continuously in the reference.'
        )

    def matching(reference, hypothesis):
        # Compute the time-constrained ORC distance
        from meeteval.wer.matching.cy_time_constrained_orc_matching import time_constrained_orc_levenshtein_distance
        return time_constrained_orc_levenshtein_distance(
            [segment.T['words'] for segment in reference.groupby('segment_index').values()],
            [stream.T['words'] for stream in hypothesis.values()],
            [list(zip(segment.T['start_time'], segment.T['end_time'])) for segment in
             reference.groupby('segment_index').values()],
            [list(zip(stream.T['start_time'], stream.T['end_time'])) for stream in hypothesis.values()],
        )

    # Drop segment index in reference. It will get a new one after merging by speakers
    reference = meeteval.io.asseglst(reference)
    reference = reference.map(lambda x: {k: v for k, v in x.items() if k != 'segment_index'})

    reference, hypothesis, ref_self_overlap, hyp_self_overlap = preprocess(
        reference, hypothesis,
        keep_keys=('words', 'segment_index', 'speaker', 'start_time', 'end_time'),
        reference_sort=reference_sort,
        hypothesis_sort=hypothesis_sort,
        reference_pseudo_word_level_timing=reference_pseudo_word_level_timing,
        hypothesis_pseudo_word_level_timing=hypothesis_pseudo_word_level_timing,
        segment_representation='word',
        segment_index='segment',
        remove_empty_segments=False,
        collar=collar,
    )

    er = _orc_error_rate(reference, hypothesis, matching, _time_constrained_siso_error_rate)
    er = dataclasses.replace(
        er,
        reference_self_overlap=ref_self_overlap,
        hypothesis_self_overlap=hyp_self_overlap,
    )
    return er


def time_constrained_orc_wer_multifile(
        reference: 'STM', hypothesis: 'STM',
        *,
        collar,
        reference_pseudo_word_level_timing='character_based',
        hypothesis_pseudo_word_level_timing='character_based_points',
        hypothesis_sort='segment',
        reference_sort='segment',
        partial=False,
) -> 'dict[str, OrcErrorRate]':
    from meeteval.io.seglst import apply_multi_file
    r = apply_multi_file(lambda r, h: time_constrained_orc_wer(
        r, h,
        reference_pseudo_word_level_timing=reference_pseudo_word_level_timing,
        hypothesis_pseudo_word_level_timing=hypothesis_pseudo_word_level_timing,
        collar=collar,
        hypothesis_sort=hypothesis_sort,
        reference_sort=reference_sort,
    ), reference, hypothesis, partial=partial)
    return r


def greedy_time_constrained_orc_wer(
        reference,
        hypothesis,
        *,
        collar,
        reference_pseudo_word_level_timing='character_based',
        hypothesis_pseudo_word_level_timing='character_based_points',
        reference_sort='segment',
        hypothesis_sort='segment',
):
    """
    Special cases where the reference or hypothesis is empty
    >>> greedy_time_constrained_orc_wer([], [], collar=5)
    OrcErrorRate(errors=0, length=0, insertions=0, deletions=0, substitutions=0, assignment=())
    >>> greedy_time_constrained_orc_wer([], [{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': 'a', 'speaker': 'A'}], collar=5)
    OrcErrorRate(errors=1, length=0, insertions=1, deletions=0, substitutions=0, hypothesis_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1), assignment=())
    >>> greedy_time_constrained_orc_wer([{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': 'a', 'speaker': 'A'}], [], collar=5)
    OrcErrorRate(error_rate=1.0, errors=1, length=1, insertions=0, deletions=1, substitutions=0, reference_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1), assignment=('dummy',))
    >>> greedy_time_constrained_orc_wer([{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': '', 'speaker': 'A'}], [{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': 'a', 'speaker': 'A'}], collar=5)
    OrcErrorRate(errors=1, length=0, insertions=1, deletions=0, substitutions=0, reference_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1), hypothesis_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1), assignment=('A',))
    >>> greedy_time_constrained_orc_wer([{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': 'a b', 'speaker': 'A'}], [{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': 'a d', 'speaker': 'A'}], collar=5)
    OrcErrorRate(error_rate=0.5, errors=1, length=2, insertions=0, deletions=0, substitutions=1, reference_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1), hypothesis_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1), assignment=('A',))
    """
    from meeteval.wer.matching.greedy_combination_matching import greedy_time_constrained_combination_matching, \
        initialize_assignment
    from meeteval.wer.wer.orc import _orc_error_rate

    def matching(reference, hypothesis):
        """Use the mimo matching algorithm. Convert inputs and outputs between the formats"""
        distance, assignment = greedy_time_constrained_combination_matching(
            [list(zip(*r)) for r in reference.T['words', 'start_time', 'end_time']],
            [[w for words in stream.T['words', 'start_time', 'end_time'] for w in zip(*words)] for stream in
             hypothesis.values()],
            initial_assignment=initialize_assignment(reference, hypothesis, initialization='tcp'),
        )
        return distance, assignment

    def siso(reference, hypothesis):
        return _time_constrained_siso_error_rate(
            reference.flatmap(lambda x: [
                {**x, 'words': w, 'start_time': s, 'end_time': e}
                for w, s, e in zip(x['words'], x['start_time'], x['end_time'])
            ]),
            hypothesis.flatmap(lambda x: [
                {**x, 'words': w, 'start_time': s, 'end_time': e}
                for w, s, e in zip(x['words'], x['start_time'], x['end_time'])
            ]),
        )

    # Drop segment index in reference. It will get a new one after merging by speakers
    reference = meeteval.io.asseglst(reference)
    reference = reference.map(lambda x: {k: v for k, v in x.items() if k != 'segment_index'})

    reference, hypothesis, ref_self_overlap, hyp_self_overlap = preprocess(
        reference, hypothesis,
        keep_keys=('words', 'segment_index', 'speaker', 'start_time', 'end_time'),
        reference_sort=reference_sort,
        hypothesis_sort=hypothesis_sort,
        segment_representation='segment',
        segment_index='segment',
        remove_empty_segments=False,
        convert_to_int=True,
        reference_pseudo_word_level_timing=reference_pseudo_word_level_timing,
        hypothesis_pseudo_word_level_timing=hypothesis_pseudo_word_level_timing,
        collar=collar,
    )

    er = _orc_error_rate(reference, hypothesis, matching, siso)
    er = dataclasses.replace(
        er,
        reference_self_overlap=ref_self_overlap,
        hypothesis_self_overlap=hyp_self_overlap,
    )
    return er


def greedy_time_constrained_orc_wer_multifile(
        reference: 'STM', hypothesis: 'STM',
        *,
        collar,
        reference_pseudo_word_level_timing='character_based',
        hypothesis_pseudo_word_level_timing='character_based_points',
        hypothesis_sort='segment',
        reference_sort='segment',
        partial=False,
) -> 'dict[str, OrcErrorRate]':
    from meeteval.io.seglst import apply_multi_file
    r = apply_multi_file(lambda r, h: greedy_time_constrained_orc_wer(
        r, h,
        reference_pseudo_word_level_timing=reference_pseudo_word_level_timing,
        hypothesis_pseudo_word_level_timing=hypothesis_pseudo_word_level_timing,
        collar=collar,
        hypothesis_sort=hypothesis_sort,
        reference_sort=reference_sort,
    ), reference, hypothesis, partial=partial)
    return r
