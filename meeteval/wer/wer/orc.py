import dataclasses

import meeteval
from meeteval.io.seglst import  SegLST, asseglistconvertible
from meeteval.wer.wer.error_rate import ErrorRate, combine_error_rates
from meeteval.wer.wer.siso import _seglst_siso_error_rate
from meeteval.wer.utils import _keys
from meeteval.io.stm import STM

__all__ = [
    'OrcErrorRate',
    'orc_word_error_rate',
    'orc_word_error_rate_multifile',
    'apply_orc_assignment'
]


@dataclasses.dataclass(frozen=True, repr=False)
class OrcErrorRate(ErrorRate):
    """
    >>> OrcErrorRate(0, 10, 0, 0, 0, None, None, (0, 1))
    OrcErrorRate(error_rate=0.0, errors=0, length=10, insertions=0, deletions=0, substitutions=0, assignment=(0, 1))
    >>> OrcErrorRate(0, 10, 0, 0, 0, None, None, (0, 1)) + OrcErrorRate(10, 10, 0, 0, 10, None, None, (1, 0, 1))
    ErrorRate(error_rate=0.5, errors=10, length=20, insertions=0, deletions=0, substitutions=10)
    """
    assignment: 'tuple[int, ...]'

    def apply_assignment(self, reference, hypothesis):
        """
        >>> OrcErrorRate(0, 10, 0, 0, 0, None, None, (0, 1)).apply_assignment(['a', 'b'], ['a', 'b'])
        ([['a'], ['b']], ['a', 'b'])
        >>> ref = meeteval.io.STM.parse('X 1 A 0.0 1.0 a b\\nX 1 A 1.0 2.0 c d\\nX 1 B 0.0 2.0 e f\\n')
        >>> hyp = meeteval.io.STM.parse('X 1 1 0.0 2.0 c d\\nX 1 0 0.0 2.0 a b e f\\n')
        >>> ref, hyp = OrcErrorRate(0, 10, 0, 0, 0, None, None, (0, 1, 1)).apply_assignment(ref, hyp)
        >>> print(ref.dumps())
        X 1 0 0.0 1.0 a b
        X 1 1 1.0 2.0 c d
        X 1 1 0.0 2.0 e f
        <BLANKLINE>
        >>> print(hyp.dumps())
        X 1 1 0.0 2.0 c d
        X 1 0 0.0 2.0 a b e f
        <BLANKLINE>
        """
        return apply_orc_assignment(self.assignment, reference, hypothesis)

def orc_word_error_rate_multifile(
        reference,
        hypothesis,
        partial=False,
) -> 'dict[str, OrcErrorRate]':
    """
    Computes the ORC WER for each example in the reference and hypothesis files.

    To compute the overall WER, use
    `sum(orc_word_error_rate_multifile(r, h).values())`.
    """
    from meeteval.io.seglst import apply_multi_file
    return apply_multi_file(
        orc_word_error_rate, reference, hypothesis,
        partial=partial
    )


def orc_word_error_rate(reference, hypothesis):
    """
    The Optimal Reference Combination (ORC) WER, implemented efficiently.

    # All correct on a single channel
    >>> orc_word_error_rate(['a b', 'c d', 'e f'], ['a b c d e f'])
    OrcErrorRate(error_rate=0.0, errors=0, length=6, insertions=0, deletions=0, substitutions=0, assignment=(0, 0, 0))

    # All correct on two channels
    >>> orc_word_error_rate(['a b', 'c d', 'e f'], ['a b', 'c d e f'])
    OrcErrorRate(error_rate=0.0, errors=0, length=6, insertions=0, deletions=0, substitutions=0, assignment=(0, 1, 1))

    # One utterance is split
    >>> er = orc_word_error_rate(['a', 'c d', 'e'], ['a c', 'd e'])
    >>> er
    OrcErrorRate(error_rate=0.5, errors=2, length=4, insertions=1, deletions=1, substitutions=0, assignment=(0, 0, 1))
    >>> er.apply_assignment(['a', 'c d', 'e'], ['a c', 'd e'])
    ([['a', 'c d'], ['e']], ['a c', 'd e'])

    >>> orc_word_error_rate(STM.parse('X 1 A 0.0 1.0 a b\\nX 1 B 0.0 2.0 e f\\nX 1 A 1.0 2.0 c d\\n'), STM.parse('X 1 1 0.0 2.0 c d\\nX 1 0 0.0 2.0 a b e f\\n'))
    OrcErrorRate(error_rate=0.0, errors=0, length=6, insertions=0, deletions=0, substitutions=0, assignment=('0', '0', '1'))

    >>> er = orc_word_error_rate(['a', 'c d', 'e'], {'A': 'a c', 'B': 'd e'})
    >>> er
    OrcErrorRate(error_rate=0.5, errors=2, length=4, insertions=1, deletions=1, substitutions=0, assignment=('A', 'A', 'B'))
    >>> er.apply_assignment(['a', 'c d', 'e'], {'A': 'a c', 'B': 'd e'})
    ({'A': ['a', 'c d'], 'B': ['e']}, {'A': 'a c', 'B': 'd e'})

    >>> orc_word_error_rate([{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': '', 'speaker': 'A'}], [{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': 'a', 'speaker': 'A'}])
    OrcErrorRate(errors=1, length=0, insertions=1, deletions=0, substitutions=0, assignment=('A',))
    """
    # Convert to seglst
    reference = meeteval.io.asseglst(reference)
    hypothesis = meeteval.io.asseglst(hypothesis)
    from meeteval.wer.wer.utils import check_single_filename
    check_single_filename(reference, hypothesis)

    # Add a segment index to the reference so that we can later find words that
    # come from the same segment. Do this before removing empty segments so that
    # we can still find the original segments in the reference after the
    # assignment.
    for i, s in enumerate(reference):
        s['segment_index'] = i

    # Group by stream. For ORC-WER, only hypothesis must be grouped
    hypothesis = hypothesis.groupby('speaker')

    # Safety check: The complexity explodes for large numbers of speakers
    if len(hypothesis) > 10:
        raise RuntimeError(
            f'Are you sure?\n'
            f'Found a total of {len(hypothesis)} speakers in the input.\n'
            f'This indicates a mistake in the input, or does your use-case '
            f'really require scoring with that many speakers?\n'
            f'See https://github.com/fgnt/meeteval/blob/main/doc/num_speaker_limits.md for details.'
        )

    # Remove empty segments. We need to merge the empty
    # reference segments back in the end to obtain the correct
    # assignment
    reference_missing_segments = reference.filter(lambda s: s['words'] == '')
    reference = reference.filter(lambda s: s['words'] != '')
    hypothesis = {
        k: h.filter(lambda s: s['words'] != '')
        for k, h in hypothesis.items()
    }

    # Split into words
    def split_words(d: 'SegLST'):
        return d.flatmap(
            lambda s: [
                {**s, 'words': w}
                for w in (s['words'].split() if s['words'].strip() else [''])
            ])
    reference = split_words(reference)
    hypothesis = {k: split_words(v) for k, v in hypothesis.items()}

    # Compute the ORC distance
    from meeteval.wer.matching.mimo_matching import mimo_matching
    distance, assignment = mimo_matching(
        [[segment.T['words'] for segment in reference.groupby('segment_index').values()]],
        [stream.T['words'] for stream in hypothesis.values()],
    )

    # Translate the assignment from hypothesis index to stream id
    # Fill with a dummy stream if hypothesis is empty
    hypothesis_keys = list(hypothesis.keys()) or ['dummy']
    assignment = [hypothesis_keys[h] for _, h in assignment]

    # Apply assignment
    reference_new, _ = apply_orc_assignment(assignment, reference, None)

    # Put the original segments back by inserting empty segments that were
    # removed in the beginning
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

    # Group by speaker
    reference_new = reference_new.groupby('speaker')

    # Consistency check: Compute WER with the siso algorithm after applying the
    # assignment and compare the result with the distance from the ORC algorithm
    er = combine_error_rates(*[
        _seglst_siso_error_rate(
            reference_new.get(k, meeteval.io.SegLST([])),
            hypothesis.get(k, meeteval.io.SegLST([])),
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
        assignment=assignment,
        reference_self_overlap=None,
        hypothesis_self_overlap=None,
    )


def apply_orc_assignment(
        assignment: 'list[int | str] | tuple[int | str]',
        reference: 'list[list[str]] | dict[str, list[str]] | SegLST',
        hypothesis: 'list[str] | dict[str] | SegLST',
):
    """
    Apply ORC assignment so that the reference streams match the hypothesis streams.

    Computing the standard WER on the output of this function yields the same
    result as the ORC-WER on the input of this function.

    Arguments:
        assignment: The assignment of reference utterances to the hypothesis
            streams. The length of the assignment must match the number of
            utterances in the reference. The assignment is a list of stream
            labels, one entry for each utterance.
        reference: The reference utterances. This can be a list of lists of
            utterances, or a SegLST object. If it is a SegLST object, the
            "segment_index" field is used to group the utterances, if present.
        hypothesis: Is passed thorugh unchanged but used to determine the format
            of the reference output if it is not SegLST.


    >>> assignment = ('A', 'A', 'B')
    >>> apply_orc_assignment(assignment, ['a', 'c d', 'e'], {'A': 'a c', 'B': 'd e'})
    ({'A': ['a', 'c d'], 'B': ['e']}, {'A': 'a c', 'B': 'd e'})

    >>> assignment = (0, 0, 1)
    >>> apply_orc_assignment(assignment, ['a', 'c d', 'e'], ['a c', 'd e'])
    ([['a', 'c d'], ['e']], ['a c', 'd e'])

    >>> assignment = ('A', )
    >>> apply_orc_assignment(assignment, ['a'], {'A': 'b', 'B': 'c'})
    ({'A': ['a'], 'B': []}, {'A': 'b', 'B': 'c'})

    >>> ref = meeteval.io.STM.parse('X 1 A 0.0 1.0 a b\\nX 1 A 1.0 2.0 c d\\nX 1 B 0.0 2.0 e f\\n')
    >>> hyp = meeteval.io.STM.parse('X 1 1 0.0 2.0 c d\\nX 1 0 0.0 2.0 a b e f\\n')
    >>> ref, hyp = apply_orc_assignment((0, 1, 1), ref, hyp)
    >>> print(ref.dumps())
    X 1 0 0.0 1.0 a b
    X 1 1 1.0 2.0 c d
    X 1 1 0.0 2.0 e f
    <BLANKLINE>
    >>> print(hyp.dumps())
    X 1 1 0.0 2.0 c d
    X 1 0 0.0 2.0 a b e f
    <BLANKLINE>
    """
    if reference != []:     # Special case where we don't want to handle [] as SegLST
        try:
            r_conv = asseglistconvertible(reference, py_convert=False)
        except Exception:
            pass
        else:
            reference = r_conv.to_seglst()

            if 'segment_index' in reference.T.keys():
                reference = reference.groupby('segment_index').values()
            else:
                reference = [[r] for r in reference]

            assert len(reference) == len(assignment), (len(reference), len(assignment))
            reference = meeteval.io.SegLST([
                {**s, 'speaker': a}
                for r, a in zip(reference, assignment)
                for s in r
            ])

            return r_conv.new(reference), hypothesis

    reference_new = {k: [] for k in _keys(hypothesis)}

    assert len(reference) == len(assignment), (len(reference), len(assignment))
    for r, a in zip(reference, assignment):
        reference_new[a].append(r)

    if isinstance(hypothesis, dict):
        return dict(reference_new), hypothesis
    elif isinstance(hypothesis, (list, tuple)):
        return type(hypothesis)(reference_new.values()), hypothesis
    else:
        raise TypeError(type(hypothesis), hypothesis)
