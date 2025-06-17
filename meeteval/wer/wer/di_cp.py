import dataclasses
import functools
from typing import Tuple

import meeteval
from meeteval.io.seglst import SegLST
from meeteval.wer.wer.error_rate import ErrorRate


__all__ = [
    'DICPErrorRate',
    'greedy_di_cp_word_error_rate',
    'greedy_di_cp_word_error_rate_multifile',
    'greedy_di_tcp_word_error_rate',
    'greedy_di_tcp_word_error_rate_multifile',
    'apply_dicp_assignment',
]


@dataclasses.dataclass(frozen=True)
class DICPErrorRate(ErrorRate):
    assignment: Tuple[int, ...]

    def apply_assignment(self, reference, hypothesis):
        return apply_dicp_assignment(self.assignment, reference, hypothesis)

    @classmethod
    def from_dict(cls, d):
        d.pop('error_rate', None)
        return cls(**d)


def greedy_di_cp_word_error_rate(
        reference,
        hypothesis,
        reference_sort='segment_if_available',
        hypothesis_sort='segment_if_available',
):
    """
    Computes the DI-cpWER with a greedy algorithm

    >>> reference = SegLST([
    ...     {'segment_index': 0, 'speaker': 'A', 'words': 'a'},
    ...     {'segment_index': 1, 'speaker': 'A', 'words': 'b'},
    ...     {'segment_index': 2, 'speaker': 'B', 'words': 'c'},
    ...     {'segment_index': 3, 'speaker': 'B', 'words': 'd'},
    ... ])
    >>> hypothesis = SegLST([
    ...     {'segment_index': 0, 'speaker': 'A', 'words': 'a'},
    ...     {'segment_index': 1, 'speaker': 'A', 'words': 'b'},
    ...     {'segment_index': 2, 'speaker': 'B', 'words': 'c'},
    ...     {'segment_index': 3, 'speaker': 'B', 'words': 'd'},
    ... ])
    >>> greedy_di_cp_word_error_rate(reference, hypothesis)
    DICPErrorRate(error_rate=0.0, errors=0, length=4, insertions=0, deletions=0, substitutions=0, reference_self_overlap=None, hypothesis_self_overlap=None, assignment=('A', 'A', 'B', 'B'))
    >>> hypothesis = SegLST([
    ...     {'segment_index': 0, 'speaker': 'A', 'words': 'a'},
    ...     {'segment_index': 1, 'speaker': 'B', 'words': 'b'},
    ...     {'segment_index': 2, 'speaker': 'A', 'words': 'c'},
    ...     {'segment_index': 3, 'speaker': 'B', 'words': 'd'},
    ... ])
    >>> greedy_di_cp_word_error_rate(reference, hypothesis)
    DICPErrorRate(error_rate=0.0, errors=0, length=4, insertions=0, deletions=0, substitutions=0, reference_self_overlap=None, hypothesis_self_overlap=None, assignment=('A', 'A', 'B', 'B'))
    >>> hypothesis = SegLST([
    ...     {'segment_index': 0, 'speaker': 'A', 'words': 'a b'},
    ...     {'segment_index': 2, 'speaker': 'A', 'words': 'b c d'},
    ... ])
    >>> greedy_di_cp_word_error_rate(reference, hypothesis)
    DICPErrorRate(error_rate=0.25, errors=1, length=4, insertions=1, deletions=0, substitutions=0, reference_self_overlap=None, hypothesis_self_overlap=None, assignment=('A', 'B'))
    """

    # The assignment of the DI-cpWER is equal to the assignment of the ORC-WER
    # with swapped arguments (reference <-> hypothesis)
    er = meeteval.wer.wer.orc.greedy_orc_word_error_rate(
        hypothesis, reference,
        hypothesis_sort, reference_sort
    )

    # The error rate object can be constructed just from the ORC-WER error rate
    # object. Insertions and deletions are swapped, the length is different.
    return DICPErrorRate(
        er.errors, sum([len(s['words'].split()) for s in reference]),
        insertions=er.deletions,
        deletions=er.insertions,
        substitutions=er.substitutions,
        assignment=er.assignment,
        reference_self_overlap=er.hypothesis_self_overlap,
        hypothesis_self_overlap=er.reference_self_overlap,
    )


def greedy_di_cp_word_error_rate_multifile(
        reference,
        hypothesis,
        partial=False,
        reference_sort='segment_if_available',
        hypothesis_sort='segment_if_available',
) -> 'dict[str, DICPErrorRate]':
    """
    Computes the (Greedy) DI-cpWER for each example in the reference and hypothesis files.

    To compute the overall WER, use
    `sum(greedy_di_cp_word_error_rate_multifile(r, h).values())`.
    """
    from meeteval.io.seglst import apply_multi_file
    return apply_multi_file(
        functools.partial(
            greedy_di_cp_word_error_rate,
            reference_sort=reference_sort,
            hypothesis_sort=hypothesis_sort,
        ), reference, hypothesis,
        partial=partial
    )


def greedy_di_tcp_word_error_rate(
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
    Computes the DI-tcpWER (time-constrained DI-cpWER) with a greedy algorithm

    >>> reference = SegLST([
    ...     {'segment_index': 0, 'speaker': 'A', 'words': 'a', 'start_time': 0.0, 'end_time': 1.0},
    ...     {'segment_index': 1, 'speaker': 'A', 'words': 'b', 'start_time': 1.0, 'end_time': 2.0},
    ...     {'segment_index': 2, 'speaker': 'B', 'words': 'c', 'start_time': 2.0, 'end_time': 3.0},
    ...     {'segment_index': 3, 'speaker': 'B', 'words': 'd', 'start_time': 3.0, 'end_time': 4.0},
    ... ])
    >>> greedy_di_tcp_word_error_rate(reference, reference, collar=0)
    DICPErrorRate(error_rate=0.0, errors=0, length=4, insertions=0, deletions=0, substitutions=0, reference_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=6.0), hypothesis_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=6.0), assignment=('A', 'A', 'B', 'B'))
    >>> hypothesis = SegLST([
    ...     {'segment_index': 0, 'speaker': 'A', 'words': 'a b', 'start_time': 0.0, 'end_time': 2.0},
    ...     {'segment_index': 2, 'speaker': 'A', 'words': 'b c d', 'start_time': 1.0, 'end_time': 4.0},
    ... ])
    >>> greedy_di_tcp_word_error_rate(reference, hypothesis, collar=0)
    DICPErrorRate(error_rate=0.25, errors=1, length=4, insertions=1, deletions=0, substitutions=0, reference_self_overlap=SelfOverlap(overlap_rate=0.0, overlap_time=0, total_time=6.0), hypothesis_self_overlap=SelfOverlap(overlap_rate=0.25, overlap_time=1.0, total_time=4.0), assignment=('A', 'B'))
    """

    # The assignment of the DI-tcpWER is equal to the assignment of the tcORC-WER
    # with swapped arguments (reference <-> hypothesis)
    er = meeteval.wer.wer.time_constrained_orc.greedy_time_constrained_orc_wer(
        hypothesis, reference,
        reference_pseudo_word_level_timing=hypothesis_pseudo_word_level_timing,
        hypothesis_pseudo_word_level_timing=reference_pseudo_word_level_timing,
        collar=collar,
        reference_sort=hypothesis_sort, hypothesis_sort=reference_sort
    )

    # The error rate object can be constructed just from the ORC-WER error rate
    # object. Insertions and deletions are swapped, the length is different.
    return DICPErrorRate(
        er.errors, sum([len(s['words'].split()) for s in reference]),
        insertions=er.deletions,
        deletions=er.insertions,
        substitutions=er.substitutions,
        assignment=er.assignment,
        reference_self_overlap=er.hypothesis_self_overlap,
        hypothesis_self_overlap=er.reference_self_overlap,
    )


def greedy_di_tcp_word_error_rate_multifile(
        reference,
        hypothesis,
        *,
        collar,
        partial=False,
        reference_pseudo_word_level_timing='character_based',
        hypothesis_pseudo_word_level_timing='character_based_points',
        reference_sort='segment',
        hypothesis_sort='segment',
) -> 'dict[str, DICPErrorRate]':
    """
    Computes the (Greedy) DI-tcpWER for each example in the reference and hypothesis files.

    To compute the overall WER, use
    `sum(greedy_di_tcp_word_error_rate_multifile(r, h).values())`.
    """
    from meeteval.io.seglst import apply_multi_file
    return apply_multi_file(
        functools.partial(
            greedy_di_tcp_word_error_rate,
            reference_pseudo_word_level_timing=reference_pseudo_word_level_timing,
            hypothesis_pseudo_word_level_timing=hypothesis_pseudo_word_level_timing,
            collar=collar,
            reference_sort=reference_sort,
            hypothesis_sort=hypothesis_sort,
        ), reference, hypothesis,
        partial=partial
    )


def apply_dicp_assignment(
        assignment: 'list[int | str] | tuple[int | str]',
        reference: 'list[list[str]] | dict[str, list[str]] | SegLST',
        hypothesis: 'list[str] | dict[str] | SegLST',
):
    """
        Apply DI-cp assignment so that the hypothesis streams match the reference streams.

        Computing the standard WER on the output of this function yields the same
        result as the DI-cpWER on the input of this function.

        Arguments:
            assignment: The assignment of hypothesis segments to the reference
                streams. The length of the assignment must match the number of
                segments in the hypothesis. The assignment is a list of stream
                labels, one entry for each stream.
            reference: Is passed thorugh unchanged but used to determine the format
                of the hypothesis output if it is not SegLST.
            hypothesis: The hypothesis segments. This can be a list of lists of
                segments, or a SegLST object. If it is a SegLST object, the
                "segment_index" field is used to group the segments, if present.

        >>> assignment = ('A', 'A', 'B')
        >>> apply_dicp_assignment(assignment, {'A': 'a c', 'B': 'd e'}, ['a', 'c d', 'e'])
        ({'A': 'a c', 'B': 'd e'}, {'A': ['a', 'c d'], 'B': ['e']})

        >>> assignment = (0, 0, 1)
        >>> apply_dicp_assignment(assignment, ['a c', 'd e'], ['a', 'c d', 'e'])
        (['a c', 'd e'], [['a', 'c d'], ['e']])

        >>> assignment = ('A', )
        >>> apply_dicp_assignment(assignment, {'A': 'b', 'B': 'c'}, ['a'])
        ({'A': 'b', 'B': 'c'}, {'A': ['a'], 'B': []})

        >>> ref = meeteval.io.STM.parse('X 1 A 0.0 1.0 a b\\nX 1 A 1.0 2.0 c d\\nX 1 B 0.0 2.0 e f\\n')
        >>> hyp = meeteval.io.STM.parse('X 1 1 0.0 2.0 c d\\nX 1 0 0.0 2.0 a b e f\\n')
        >>> ref, hyp = apply_dicp_assignment((0, 1, 1), hyp, ref)
        >>> print(ref.dumps())
        X 1 1 0.0 2.0 c d
        X 1 0 0.0 2.0 a b e f
        <BLANKLINE>
        >>> print(hyp.dumps())
        X 1 0 0.0 1.0 a b
        X 1 1 1.0 2.0 c d
        X 1 1 0.0 2.0 e f
        <BLANKLINE>
        """
    # The assignment is identical to the ORC assignment, but with
    # reference and hypothesis swapped.
    from meeteval.wer.wer.orc import apply_orc_assignment
    hypothesis, reference = apply_orc_assignment(assignment, hypothesis, reference)
    return reference, hypothesis
