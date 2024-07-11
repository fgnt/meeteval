import functools
import typing


if typing.TYPE_CHECKING:
    from meeteval.io import STM
    from meeteval.wer import OrcErrorRate

__all__ = [
    'time_constrained_orc_wer',
    'time_constrained_orc_wer_multifile',
]


def time_constrained_orc_wer(
        reference,
        hypothesis,
        collar=0,
        reference_pseudo_word_level_timing='character_based',
        hypothesis_pseudo_word_level_timing='character_based_points',
        hypothesis_sort='segment',
        reference_sort='segment',
) -> OrcErrorRate:
    """
    The time-constrained version of the ORC-WER (tcORC-WER).

    Special cases where the reference or hypothesis is empty
    >>> time_constrained_orc_wer([], [])
    OrcErrorRate(errors=0, length=0, insertions=0, deletions=0, substitutions=0, assignment=())
    >>> time_constrained_orc_wer([], [{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': 'a', 'speaker': 'A'}])
    OrcErrorRate(errors=1, length=0, insertions=1, deletions=0, substitutions=0, hypothesis_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1), assignment=())
    >>> time_constrained_orc_wer([{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': 'a', 'speaker': 'A'}], [])
    OrcErrorRate(error_rate=1.0, errors=1, length=1, insertions=0, deletions=1, substitutions=0, reference_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1), assignment=('dummy',))
    >>> time_constrained_orc_wer([{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': '', 'speaker': 'A'}], [{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': 'a', 'speaker': 'A'}])
    OrcErrorRate(errors=1, length=0, insertions=1, deletions=0, substitutions=0, reference_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1), hypothesis_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1), assignment=('A',))
    >>> time_constrained_orc_wer([{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': 'a b', 'speaker': 'A'}], [{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': 'a d', 'speaker': 'A'}])
    OrcErrorRate(error_rate=0.5, errors=1, length=2, insertions=0, deletions=0, substitutions=1, reference_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1), hypothesis_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1), assignment=('A',))
    """
    from meeteval.wer.wer.orc import _orc_error_rate
    from meeteval.wer.wer.time_constrained import preprocess_time_constrained

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

    return _orc_error_rate(
        reference, hypothesis,
        functools.partial(
            preprocess_time_constrained,
            reference, hypothesis, collar,
            reference_pseudo_word_level_timing, hypothesis_pseudo_word_level_timing,
            reference_sort, hypothesis_sort,
            convert_to_int=False
        ),
        matching
    )


def time_constrained_orc_wer_multifile(
        reference: 'STM', hypothesis: 'STM',
        reference_pseudo_word_level_timing='character_based',
        hypothesis_pseudo_word_level_timing='character_based_points',
        collar: int = 0,
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
