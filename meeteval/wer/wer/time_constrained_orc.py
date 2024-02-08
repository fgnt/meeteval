import meeteval
from meeteval.wer import combine_error_rates
from meeteval.wer.wer.orc import OrcErrorRate, apply_orc_assignment
from meeteval.wer.wer.time_constrained import get_self_overlap
from meeteval.wer.wer.utils import check_single_filename

import typing

if typing.TYPE_CHECKING:
    from meeteval.io import STM
    from meeteval.wer.wer.cp import CPErrorRate

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
):
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
    if reference_sort == 'word':
        raise ValueError(
            'reference_sort="word" is not supported for time-constrained '
            'ORC-WER because the ORC-WER assumes that an utterance appears'
            'continuously in the reference.'
        )

    # Convert to seglst
    original_reference = reference = meeteval.io.asseglst(reference)
    original_hypothesis = hypothesis = meeteval.io.asseglst(hypothesis)
    check_single_filename(reference, hypothesis)

    # Add a segment index to the reference so that we can later find words that
    # come from the same segment. Do this before removing empty segments so that
    # we can still find the original segments in the reference after the
    # assignment.
    for i, s in enumerate(reference):
        s['segment_index'] = i

    # Group by stream. For ORC-WER, only hypothesis must be grouped
    hypothesis = hypothesis.groupby('speaker')

    # Calculate self-overlap before modifying the segments
    reference_self_overlap = sum(
        [get_self_overlap(h) for h in reference.groupby('speaker').values()]
    ) if len(reference) > 0 else None
    hypothesis_self_overlap = sum(
        [get_self_overlap(h) for h in hypothesis.values()]
    ) if len(hypothesis) > 0 else None

    # Remove empty segments
    reference_missing_segments = reference.filter(lambda s: s['words'] == '')
    reference = reference.filter(lambda s: s['words'] != '')
    hypothesis = {
        k: h.filter(lambda s: s['words'] != '')
        for k, h in hypothesis.items()
    }

    # Time-constrained preprocessing
    from meeteval.wer.wer.time_constrained import sort_and_validate, apply_collar

    # Note: We will very likely have overlaps that result in word-order changes
    # in the reference because we treat all speakers as one stream. This is no
    # issue for the ORC alignment algorithm! We do not want to spam the user
    # with meaningless warnings here, thus warn=False.
    reference = sort_and_validate(
        reference,
        reference_sort,
        reference_pseudo_word_level_timing,
        f'reference',
        warn=False,
    )
    hypothesis = {
        k: apply_collar(sort_and_validate(
            h,
            hypothesis_sort,
            hypothesis_pseudo_word_level_timing,
            f'hypothesis speaker "{k}"'
        ), collar) for k, h in hypothesis.items()
    }

    # Compute the time-constrained ORC distance
    from meeteval.wer.matching.cy_time_constrained_orc_matching import time_constrained_orc_levenshtein_distance
    distance, assignment = time_constrained_orc_levenshtein_distance(
        [segment.T['words'] for segment in reference.groupby('segment_index').values()],
        [stream.T['words'] for stream in hypothesis.values()],
        [list(zip(segment.T['start_time'], segment.T['end_time'])) for segment in
         reference.groupby('segment_index').values()],
        [list(zip(stream.T['start_time'], stream.T['end_time'])) for stream in hypothesis.values()],
    )

    # Translate the assignment from hypothesis index to stream id
    hypothesis_keys = list(hypothesis.keys()) or ['dummy']
    assignment = [hypothesis_keys[h] for h in assignment]

    # Apply assignment
    reference_new, _ = apply_orc_assignment(assignment, reference, hypothesis)

    # Put the original segments back by inserting empty segments that were
    # removed in the beginning. Sort by segment_index to get the assignment in
    # the same order as the input to this function.
    # TODO: Estimate the stream for the missing segments
    reference_missing_segments = reference_missing_segments.map(
        lambda s: {**s, 'speaker': hypothesis_keys[0]}
    )
    reference_new = meeteval.io.SegLST.merge(
        reference_new, reference_missing_segments
    ).sorted('segment_index')
    assignment = tuple([
        v[0]['speaker']
        for v in reference_new.groupby('segment_index').values()
    ])

    # Apply the assignment to the original reference for the consistency check.
    original_reference, _ = apply_orc_assignment(assignment, original_reference, original_hypothesis)

    # Group by speaker
    reference_new = original_reference.groupby('speaker')
    original_hypothesis = original_hypothesis.groupby('speaker')

    # Consistency check: Compute WER with the siso algorithm after applying the
    # assignment and compare the result with the distance from the ORC algorithm
    from meeteval.wer.wer.time_constrained import time_constrained_siso_word_error_rate
    er = combine_error_rates(*[
        time_constrained_siso_word_error_rate(
            reference_new.get(k, meeteval.io.SegLST([])),
            original_hypothesis.get(k, meeteval.io.SegLST([])),
            reference_pseudo_word_level_timing=reference_pseudo_word_level_timing,
            hypothesis_pseudo_word_level_timing=hypothesis_pseudo_word_level_timing,
            collar=collar,
            reference_sort=reference_sort,
            hypothesis_sort=hypothesis_sort,
        )
        for k in set(hypothesis.keys()) | set(reference_new.keys())
    ])
    length = len(reference)
    assert er.length == length, (length, er)
    assert er.errors == distance, (distance, er, assignment)

    return OrcErrorRate(
        er.errors, er.length,
        insertions=er.insertions,
        deletions=er.deletions,
        substitutions=er.substitutions,
        assignment=tuple(assignment),
        reference_self_overlap=reference_self_overlap,
        hypothesis_self_overlap=hypothesis_self_overlap,
    )


def time_constrained_orc_wer_multifile(
        reference: 'STM', hypothesis: 'STM',
        reference_pseudo_word_level_timing='character_based',
        hypothesis_pseudo_word_level_timing='character_based_points',
        collar: int = 0,
        hypothesis_sort='segment',
        reference_sort='segment',
        partial=False,
) -> 'dict[str, CPErrorRate]':
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
