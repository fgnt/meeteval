import meeteval
from meeteval.wer import ErrorRate
from meeteval.wer.wer.orc import OrcErrorRate
from meeteval.wer.wer.utils import check_single_filename

import typing

if typing.TYPE_CHECKING:
    from meeteval.io import STM
    from meeteval.wer.wer.cp import CPErrorRate


def time_constrained_orc_wer(
        reference,
        hypothesis,
        collar=0,
        reference_pseudo_word_level_timing='character_based',
        hypothesis_pseudo_word_level_timing='character_based_points',
        hypothesis_sort='segment',
):
    """
    The time-constrained version of the ORC-WER (tcORC-WER).

    Special cases where the reference or hypothesis is empty
    >>> time_constrained_orc_wer([], [])
    OrcErrorRate(errors=0, length=0, insertions=0, deletions=0, substitutions=0, assignment=())
    >>> time_constrained_orc_wer([], [{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': 'a', 'speaker': 'A'}])
    OrcErrorRate(errors=1, length=0, insertions=1, deletions=0, substitutions=0, assignment=())
    >>> time_constrained_orc_wer([{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': 'a', 'speaker': 'A'}], [])
    OrcErrorRate(error_rate=1.0, errors=1, length=1, insertions=0, deletions=1, substitutions=0, assignment=())
    """
    # Convert to seglst
    reference = meeteval.io.asseglst(reference)
    hypothesis = meeteval.io.asseglst(hypothesis)
    check_single_filename(reference, hypothesis)

    # Remove empty segments
    reference = reference.filter(lambda s: s['words'] != '')
    hypothesis = hypothesis.filter(lambda s: s['words'] != '')

    # Add a segment index to the reference so that we can later find words that
    # come from the same segment
    for i, s in enumerate(reference):
        s['segment_index'] = i

    # Group by stream
    hypothesis = hypothesis.groupby('speaker')

    # Time-constrained preprocessing
    from meeteval.wer.wer.time_constrained import sort_and_validate, apply_collar
    reference = sort_and_validate(
        reference,
        'segment',
        reference_pseudo_word_level_timing,
        f'reference'
    )
    hypothesis = {
        k: apply_collar(sort_and_validate(
            h,
            hypothesis_sort,
            hypothesis_pseudo_word_level_timing,
            f'hypothesis speaker "{k}"'
        ), collar) for k, h in hypothesis.items()
    }

    # TODO: compute self-overlap

    from meeteval.wer.matching.cy_time_constrained_orc_matching import time_constrained_orc_levenshtein_distance

    distance, assignment = time_constrained_orc_levenshtein_distance(
        [segment.T['words'] for segment in reference.groupby('segment_index').values()],
        [stream.T['words'] for stream in hypothesis.values()],
        [list(zip(segment.T['start_time'], segment.T['end_time'])) for segment in reference.groupby('segment_index').values()],
        [list(zip(stream.T['start_time'], stream.T['end_time'])) for stream in hypothesis.values()],
    )

    hypothesis_keys = list(hypothesis.keys())
    assignment = [hypothesis_keys[h] for r, h in assignment]

    # Apply assignment in seglst format
    r_ = list(reference.groupby('segment_index').values()) # Shallow copy because we pop later
    if assignment:
        reference_new = []
        for h in assignment:
            for w in r_.pop(0):
                reference_new.append({**w, 'speaker': h})
        reference_new = meeteval.io.SegLST(reference_new).groupby('speaker')
    else:
        reference_new = reference.groupby('speaker')

    from meeteval.wer.wer.time_constrained import _time_constrained_siso_error_rate
    er = sum([
        _time_constrained_siso_error_rate(
            reference_new.get(k, meeteval.io.SegLST([])),
            hypothesis.get(k, meeteval.io.SegLST([])),
        )
        for k in set(hypothesis.keys()) | set(reference_new.keys())
    ], start=ErrorRate(0, 0, 0, 0, 0, None, None))
    length = len(reference)
    assert er.length == length, (length, er)
    assert er.errors == distance, (distance, er, assignment)

    return OrcErrorRate(
        er.errors, er.length,
        insertions=er.insertions,
        deletions=er.deletions,
        substitutions=er.substitutions,
        assignment=tuple(assignment),
        reference_self_overlap=None,
        hypothesis_self_overlap=None,
    )

def time_constrained_orc_wer_multifile(
        reference: 'STM', hypothesis: 'STM',
        reference_pseudo_word_level_timing='character_based',
        hypothesis_pseudo_word_level_timing='character_based_points',
        collar: int = 0,
        hypothesis_sort='segment',
) -> 'dict[str, CPErrorRate]':
    from meeteval.io.seglst import apply_multi_file
    r = apply_multi_file(lambda r, h: time_constrained_orc_wer(
        r, h,
        reference_pseudo_word_level_timing=reference_pseudo_word_level_timing,
        hypothesis_pseudo_word_level_timing=hypothesis_pseudo_word_level_timing,
        collar=collar,
        hypothesis_sort=hypothesis_sort,
    ), reference, hypothesis)
    return r