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
]


@dataclasses.dataclass(frozen=True)
class DICPErrorRate(ErrorRate):
    assignment: Tuple[int, ...]

    def apply_assignment(self, reference, hypothesis):
        if reference != []:  # Special case where we don't want to handle [] as SegLST
            from meeteval.io.seglst import SegLST, asseglistconvertible
            import meeteval
            try:
                h_conv = asseglistconvertible(hypothesis, py_convert=False)
            except:
                pass
            else:
                hypothesis = h_conv.to_seglst()

                if 'segment_index' in hypothesis.T.keys():
                    hypothesis = hypothesis.groupby('segment_index').values()
                else:
                    hypothesis = [[r] for r in hypothesis]

                assert len(hypothesis) == len(self.assignment), (len(hypothesis), len(self.assignment))
                hypothesis = meeteval.io.SegLST([
                    {**s, 'speaker': a}
                    for r, a in zip(hypothesis, self.assignment)
                    for s in r
                ])

                return reference, h_conv.new(hypothesis)
        h = {}
        for a, h_ in zip(self.assignment, hypothesis):
            h.setdefault(a, []).append(h_)
        return reference, h

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
) -> 'dict[str, OrcErrorRate]':
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
