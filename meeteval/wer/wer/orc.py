import dataclasses
import functools
import typing

import meeteval
from meeteval.io.py import NestedStructure
from meeteval.io.seglst import asseglistconvertible
from meeteval.wer.wer.error_rate import ErrorRate, combine_error_rates
from meeteval.wer.wer.siso import _seglst_siso_error_rate, _siso_error_rate
from meeteval.wer.utils import _keys
from meeteval.io.stm import STM

__all__ = [
    'OrcErrorRate',
    'orc_word_error_rate',
    'orc_word_error_rate_multifile',
    'apply_orc_assignment',
    'greedy_orc_word_error_rate',
    'greedy_orc_word_error_rate_multifile',
]

from meeteval.wer.preprocess import preprocess

if typing.TYPE_CHECKING:
    from meeteval.io.seglst import SegLST


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
        reference_sort='segment_if_available',
        hypothesis_sort='segment_if_available',
) -> 'dict[str, OrcErrorRate]':
    """
    Computes the ORC WER for each example in the reference and hypothesis files.

    To compute the overall WER, use
    `sum(orc_word_error_rate_multifile(r, h).values())`.
    """
    from meeteval.io.seglst import apply_multi_file
    return apply_multi_file(
        functools.partial(
            orc_word_error_rate,
            reference_sort=reference_sort,
            hypothesis_sort=hypothesis_sort,
        ), reference, hypothesis,
        partial=partial
    )


def _orc_error_rate(
        reference, hypothesis,
        matching_fn,
        siso_error_rate,
):
    # Remove empty segments from the reference because they can increase
    # the computational complexity of the ORC algorithm, but they do not 
    # contribute to the distance
    total_num_segments = len(set(reference.T['segment_index']))
    reference = reference.filter(lambda x: x['words'] != '')
    hypothesis = hypothesis.filter(lambda x: x['words'] != '')
    num_filtered_segments = len(reference.unique('segment_index'))

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

    # Compute the ORC distance
    distance, assignment = matching_fn(reference, hypothesis)
    assert len(assignment) == num_filtered_segments, (len(assignment), num_filtered_segments)

    # Translate the assignment from hypothesis index to stream id
    # Fill with a dummy stream if hypothesis is empty
    hypothesis_keys = list(hypothesis.keys()) or ['dummy']
    assignment = [hypothesis_keys[a] for a in assignment]

    # Apply assignment
    reference_new, _ = apply_orc_assignment(assignment, reference, None)

    # Consistency check: Compute WER with the siso algorithm after applying the
    # assignment and compare the result with the distance from the ORC algorithm
    reference_new_grouped = reference_new.groupby('speaker')
    er = combine_error_rates(*[
        siso_error_rate(
            reference_new_grouped.get(k, meeteval.io.SegLST([])),
            hypothesis.get(k, meeteval.io.SegLST([])),
        )
        for k in set(hypothesis.keys()) | set(reference_new_grouped.keys())
    ])
    length = sum([len(s['words']) if isinstance(s['words'], list) else 1 for s in reference])
    assert er.length == length, (length, er)
    if distance is not None:
        # The matching function can in some cases not compute the distance
        assert er.errors == distance, (distance, er, assignment)

    # Get the assignment in the order of segment_index
    assignment = [a for _, a in sorted(zip(reference.groupby('segment_index').keys(), assignment))]

    # Insert labels for empty segments that got removed
    if len(assignment) != total_num_segments:
        # Make sure this assignment is reproducible by sorting the keys
        dummy_key = sorted(hypothesis_keys)[0]
        assignment = sorted([(r, a) for a, r in zip(assignment, sorted(reference.unique('segment_index')))])
        for i in range(total_num_segments):
            if i >= len(assignment) or assignment[i][0] != i:
                assignment.insert(i, (i, dummy_key))
        assignment = [a[1] for a in assignment]

        assert len(assignment) == total_num_segments, (len(assignment), total_num_segments)

    return OrcErrorRate(
        er.errors, er.length,
        insertions=er.insertions,
        deletions=er.deletions,
        substitutions=er.substitutions,
        assignment=tuple(assignment),
        reference_self_overlap=None,
        hypothesis_self_overlap=None
    )


def orc_word_error_rate(
        reference,
        hypothesis,
        reference_sort='segment_if_available',
        hypothesis_sort='segment_if_available',
):
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
    OrcErrorRate(error_rate=0.0, errors=0, length=6, insertions=0, deletions=0, substitutions=0, reference_self_overlap=SelfOverlap(overlap_rate=Decimal('0E+1'), overlap_time=0, total_time=Decimal('4.0')), hypothesis_self_overlap=SelfOverlap(overlap_rate=Decimal('0E+1'), overlap_time=0, total_time=Decimal('4.0')), assignment=('0', '0', '1'))

    >>> er = orc_word_error_rate(['a', 'c d', 'e'], {'A': 'a c', 'B': 'd e'})
    >>> er
    OrcErrorRate(error_rate=0.5, errors=2, length=4, insertions=1, deletions=1, substitutions=0, assignment=('A', 'A', 'B'))
    >>> er.apply_assignment(['a', 'c d', 'e'], {'A': 'a c', 'B': 'd e'})
    ({'A': ['a', 'c d'], 'B': ['e']}, {'A': 'a c', 'B': 'd e'})

    >>> orc_word_error_rate([{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': '', 'speaker': 'A'}], [{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': 'a', 'speaker': 'A'}])
    OrcErrorRate(errors=1, length=0, insertions=1, deletions=0, substitutions=0, reference_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1), hypothesis_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1), assignment=('A',))
    """
    from meeteval.wer.matching.mimo_matching import mimo_matching

    def matching(reference, hypothesis):
        """Use the mimo matching algorithm. Convert inputs and outputs between the formats"""
        distance, assignment = mimo_matching(
            [reference.T['words']],
            [[w for words in stream.T['words'] for w in words] for stream in hypothesis.values()],
        )
        assignment = [a for _, a in assignment]
        return distance, assignment

    def siso(reference, hypothesis):
        return _siso_error_rate(
            [w for words in reference.T['words'] for w in words],
            [w for words in hypothesis.T['words'] for w in words],
        )

    # Drop segment index in reference. It will get a new one after merging by speakers
    reference = meeteval.io.asseglst(reference)
    reference = reference.map(lambda x: {k: v for k, v in x.items() if k != 'segment_index'})

    reference, hypothesis, ref_self_overlap, hyp_self_overlap = preprocess(
        reference, hypothesis,
        keep_keys=('words', 'segment_index', 'speaker'),
        reference_sort=reference_sort,
        hypothesis_sort=hypothesis_sort,
        segment_representation='segment',
        segment_index='segment',
        remove_empty_segments=False,
    )
    er = _orc_error_rate(reference, hypothesis, matching, siso)
    er = dataclasses.replace(
        er,
        reference_self_overlap=ref_self_overlap,
        hypothesis_self_overlap=hyp_self_overlap,
    )
    return er


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
    if reference != []:  # Special case where we don't want to handle [] as SegLST
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


def greedy_orc_word_error_rate(
        reference: 'SegLST',
        hypothesis: 'SegLST',
        reference_sort='segment_if_available',
        hypothesis_sort='segment_if_available',
):
    """
    # All correct on a single channel
    >>> greedy_orc_word_error_rate(['a b', 'c d', 'e f'], ['a b c d e f'])
    OrcErrorRate(error_rate=0.0, errors=0, length=6, insertions=0, deletions=0, substitutions=0, assignment=(0, 0, 0))

    # All correct on two channels
    >>> greedy_orc_word_error_rate(['a b', 'c d', 'e f'], ['a b', 'c d e f'])
    OrcErrorRate(error_rate=0.0, errors=0, length=6, insertions=0, deletions=0, substitutions=0, assignment=(0, 1, 1))

    # One utterance is split
    >>> er = greedy_orc_word_error_rate(['a', 'c d e'], ['a c', 'd e'])
    >>> er
    OrcErrorRate(error_rate=0.5, errors=2, length=4, insertions=1, deletions=1, substitutions=0, assignment=(0, 1))
    >>> er.apply_assignment(['a', 'c d e'], ['a c', 'd e'])
    ([['a'], ['c d e']], ['a c', 'd e'])

    >>> greedy_orc_word_error_rate(STM.parse('X 1 A 0.0 1.0 a b\\nX 1 B 0.0 2.0 e f\\nX 1 A 1.0 2.0 c d\\n'), STM.parse('X 1 1 0.0 2.0 c d\\nX 1 0 0.0 2.0 a b e f\\n'))
    OrcErrorRate(error_rate=0.0, errors=0, length=6, insertions=0, deletions=0, substitutions=0, reference_self_overlap=SelfOverlap(overlap_rate=Decimal('0E+1'), overlap_time=0, total_time=Decimal('4.0')), hypothesis_self_overlap=SelfOverlap(overlap_rate=Decimal('0E+1'), overlap_time=0, total_time=Decimal('4.0')), assignment=('0', '0', '1'))

    >>> er = greedy_orc_word_error_rate(['a', 'c d e'], {'A': 'a c', 'B': 'd e'})
    >>> er
    OrcErrorRate(error_rate=0.5, errors=2, length=4, insertions=1, deletions=1, substitutions=0, assignment=('A', 'B'))
    >>> er.apply_assignment(['a', 'c d e'], {'A': 'a c', 'B': 'd e'})
    ({'A': ['a'], 'B': ['c d e']}, {'A': 'a c', 'B': 'd e'})

    >>> greedy_orc_word_error_rate([{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': '', 'speaker': 'A'}], [{'session_id': 'a', 'start_time': 0, 'end_time': 1, 'words': 'a', 'speaker': 'A'}])
    OrcErrorRate(errors=1, length=0, insertions=1, deletions=0, substitutions=0, reference_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1), hypothesis_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=1), assignment=('A',))

    """
    from meeteval.wer.matching.greedy_combination_matching import greedy_combination_matching, initialize_assignment

    def matching(reference, hypothesis):
        """Use the mimo matching algorithm. Convert inputs and outputs between the formats"""
        distance, assignment = greedy_combination_matching(
            reference.T['words'],
            [[w for words in stream.T['words'] for w in words] for stream in hypothesis.values()],
            initial_assignment=initialize_assignment(reference, hypothesis, initialization='cp'),
        )
        return distance, assignment

    def siso(reference, hypothesis):
        return _siso_error_rate(
            [w for words in reference.T['words'] for w in words],
            [w for words in hypothesis.T['words'] for w in words],
        )

    # Drop segment index in reference. It will get a new one after merging by speakers
    reference = meeteval.io.asseglst(reference)
    reference = reference.map(lambda x: {k: v for k, v in x.items() if k != 'segment_index'})

    reference, hypothesis, ref_self_overlap, hyp_self_overlap = preprocess(
        reference, hypothesis,
        keep_keys=('words', 'segment_index', 'speaker'),
        reference_sort=reference_sort,
        hypothesis_sort=hypothesis_sort,
        segment_representation='segment',
        segment_index='segment',
        remove_empty_segments=False,
        convert_to_int=True,
    )
    er = _orc_error_rate(reference, hypothesis, matching, siso)
    er = dataclasses.replace(
        er,
        reference_self_overlap=ref_self_overlap,
        hypothesis_self_overlap=hyp_self_overlap,
    )
    return er


def greedy_orc_word_error_rate_multifile(
        reference,
        hypothesis,
        partial=False,
        reference_sort='segment_if_available',
        hypothesis_sort='segment_if_available',
) -> 'dict[str, OrcErrorRate]':
    """
    Computes the ORC WER for each example in the reference and hypothesis files.

    To compute the overall WER, use
    `sum(orc_word_error_rate_multifile(r, h).values())`.
    """
    from meeteval.io.seglst import apply_multi_file
    return apply_multi_file(
        functools.partial(
            greedy_orc_word_error_rate,
            reference_sort=reference_sort,
            hypothesis_sort=hypothesis_sort,
        ), reference, hypothesis,
        partial=partial
    )
