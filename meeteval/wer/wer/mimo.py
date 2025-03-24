import dataclasses
import functools
from typing import Iterable, Any

from meeteval.io.seglst import asseglistconvertible, SegLST
from meeteval.wer.preprocess import add_segment_index, preprocess
from meeteval.wer.wer.error_rate import ErrorRate
from meeteval.wer.wer.siso import _siso_error_rate
from meeteval.wer.utils import _keys, _items, _values

__all__ = [
    'MimoErrorRate', 'mimo_word_error_rate', 'apply_mimo_assignment',
    'mimo_word_error_rate_multifile'
]

from meeteval.io import STM


@dataclasses.dataclass(frozen=True, repr=False)
class MimoErrorRate(ErrorRate):
    """
    >>> MimoErrorRate(0, 10, 0, 0, 0, None, None, [(0, 0)])
    MimoErrorRate(error_rate=0.0, errors=0, length=10, insertions=0, deletions=0, substitutions=0, assignment=[(0, 0)])
    >>> MimoErrorRate(0, 10, 0, 0, 0, None, None, [(0, 0)]) + MimoErrorRate(10, 10, 0, 0, 10, None, None, [(0, 0)])
    ErrorRate(error_rate=0.5, errors=10, length=20, insertions=0, deletions=0, substitutions=10)
    """
    assignment: 'tuple[int, ...]'

    def apply_assignment(self, reference, hypothesis):
        return apply_mimo_assignment(
            self.assignment,
            reference=reference,
            hypothesis=hypothesis,
        )


def mimo_error_rate(
        reference: 'list[list[Iterable]] | dict[Any, list[Iterable]]',
        hypothesis: 'list[Iterable] | dict[Iterable]',
):
    if max(len(hypothesis), len(reference)) > 10:
        num_speakers = max(len(hypothesis), len(reference))
        raise RuntimeError(
            f'Are you sure?\n'
            f'Found a total of {num_speakers} speakers in the input.\n'
            f'This indicates a mistake in the input, or does your use-case '
            f'really require scoring with that many speakers?\n'
            f'See https://github.com/fgnt/meeteval/blob/main/doc/num_speaker_limits.md '
            f'for details.'
        )

    from meeteval.wer.matching.mimo_matching import mimo_matching

    distance, assignment = mimo_matching(
        [_values(r) for r in _values(reference)],
        _values(hypothesis)
    )

    reference_keys = _keys(reference)
    hypothesis_keys = _keys(hypothesis)
    assignment = [(reference_keys[r], hypothesis_keys[h]) for r, h in assignment]

    reference_new, hypothesis = apply_mimo_assignment(
        assignment,
        reference=reference,
        hypothesis=hypothesis,
    )

    er = sum([
        _siso_error_rate(
            [t for r_ in r for t in r_],  # Create list with all words from one speaker
            h,
        )
        for r, h in zip(_values(reference_new), _values(hypothesis))
    ])
    assert er.errors == distance, (distance, er)

    return MimoErrorRate(
        er.errors, er.length,
        insertions=er.insertions,
        deletions=er.deletions,
        substitutions=er.substitutions,
        assignment=assignment,
        reference_self_overlap=None,
        hypothesis_self_overlap=None,
    )


def mimo_word_error_rate(
        reference,
        hypothesis,
        reference_sort='segment_if_available',
        hypothesis_sort='segment_if_available',
) -> MimoErrorRate:
    """
    The Multiple Input speaker, Multiple Output channel (MIMO) WER.

    >>> mimo_word_error_rate([['a b c d e f']], ['a b c d e f'])
    MimoErrorRate(error_rate=0.0, errors=0, length=6, insertions=0, deletions=0, substitutions=0, assignment=[(0, 0)])

    # All correct, utterance order between speakers can change
    >>> mimo_word_error_rate([['a b', 'c d'], ['e f']], ['a b', 'e f c d'])
    MimoErrorRate(error_rate=0.0, errors=0, length=6, insertions=0, deletions=0, substitutions=0, assignment=[(1, 1), (0, 0), (0, 1)])
    >>> mimo_word_error_rate([['a b', 'c d'], ['e f']], ['a b', 'c d e f'])
    MimoErrorRate(error_rate=0.0, errors=0, length=6, insertions=0, deletions=0, substitutions=0, assignment=[(0, 0), (0, 1), (1, 1)])
    >>> mimo_word_error_rate([['a b', 'c d'], ['e f']], ['c d', 'a b e f'])
    MimoErrorRate(error_rate=0.0, errors=0, length=6, insertions=0, deletions=0, substitutions=0, assignment=[(0, 1), (1, 1), (0, 0)])

    >>> mimo_word_error_rate({'A': ['a b', 'c d'], 'B': ['e f']},
    ...                      {'O1': 'c d', 'O2': 'a b e f'})
    MimoErrorRate(error_rate=0.0, errors=0, length=6, insertions=0, deletions=0, substitutions=0, assignment=[('A', 'O2'), ('B', 'O2'), ('A', 'O1')])

    >>> mimo_word_error_rate(STM.parse('X 1 A 0.0 1.0 a b\\nX 1 A 1.0 2.0 c d\\nX 1 B 0.0 2.0 e f\\n'), STM.parse('X 1 1 0.0 2.0 c d\\nX 1 0 0.0 2.0 a b e f\\n'))
    MimoErrorRate(error_rate=0.0, errors=0, length=6, insertions=0, deletions=0, substitutions=0, reference_self_overlap=SelfOverlap(overlap_rate=Decimal('0E+1'), overlap_time=0, total_time=Decimal('4.0')), hypothesis_self_overlap=SelfOverlap(overlap_rate=Decimal('0E+1'), overlap_time=0, total_time=Decimal('4.0')), assignment=[('A', '0'), ('B', '0'), ('A', '1')])
    """
    reference, hypothesis, ref_self_overlap, hyp_self_overlap = preprocess(
        reference, hypothesis,
        remove_empty_segments=False,
        reference_sort=reference_sort,
        hypothesis_sort=hypothesis_sort,
    )

    # Convert to dict of lists of words and remove empty words here.
    reference = {
        k: [
            [word for words in segment.T['words'] for word in words if word != '']
            for segment in v.groupby('segment_index').values()
        ]
        for k, v in reference.groupby('speaker').items()
    }
    hypothesis = {
        k: [word for words in v.T['words'] for word in words if word != '']
        for k, v in hypothesis.groupby('speaker').items()
    }

    # Call core function
    return dataclasses.replace(
        mimo_error_rate(reference, hypothesis),
        reference_self_overlap=ref_self_overlap,
        hypothesis_self_overlap=hyp_self_overlap,
    )


def mimo_word_error_rate_multifile(
        reference,
        hypothesis,
        partial=False,
        reference_sort='segment_if_available',
        hypothesis_sort='segment_if_available',
) -> 'dict[str, MimoErrorRate]':
    """
    Computes the MIMO WER for each example in the reference and hypothesis
    files.

    To compute the overall WER, use
    `sum(mimo_word_error_rate_multifile(r, h).values())`.
    """
    from meeteval.io.seglst import apply_multi_file
    return apply_multi_file(
        functools.partial(
            mimo_word_error_rate,
            reference_sort=reference_sort,
            hypothesis_sort=hypothesis_sort,
        ),
        reference, hypothesis, partial=partial
    )


def apply_mimo_assignment(
        assignment: 'list[tuple]',
        reference: 'list[list[Any]] | dict[list[Any]]',
        hypothesis: 'list[Any] | dict[Any, Any]',
):
    """
    Apply the assignment to the reference and hypothesis.
    Only the reference is modifed.

    The labels and the segment order is modifed in reference according 
    to the assignment. The order change is important when segments 
    overlap!

    >>> assignment = [('A', 'O2'), ('B', 'O2'), ('A', 'O1')]
    >>> reference = {'A': ['a b', 'c d'], 'B': ['e f']}
    >>> hypothesis = {'O1': 'c d', 'O2': 'a b e f'}
    >>> apply_mimo_assignment(assignment, reference, hypothesis)
    ({'O1': ['c d'], 'O2': ['a b', 'e f']}, {'O1': 'c d', 'O2': 'a b e f'})

    >>> assignment = [(0, 1), (1, 1), (0, 0)]
    >>> reference = [['a b', 'c d'], ['e f']]
    >>> hypothesis = ['c d', 'a b e f']
    >>> apply_mimo_assignment(assignment, reference, hypothesis)
    ([['c d'], ['a b', 'e f']], ['c d', 'a b e f'])

    >>> assignment = [('A', 'O1'), ('A', 'O1')]
    >>> reference = {'A': ['a b', 'c d']}
    >>> hypothesis = {'O1': 'c d', 'O2': 'a b e f'}
    >>> apply_mimo_assignment(assignment, reference, hypothesis)
    ({'O1': ['a b', 'c d'], 'O2': []}, {'O1': 'c d', 'O2': 'a b e f'})

    >>> reference = STM.parse('X 1 A 0.0 1.0 a b\\nX 1 A 1.0 2.0 c d\\n')
    >>> hypothesis = STM.parse('X 1 O1 0.0 2.0 c d\\nX 1 O0 0.0 2.0 a b e f\\n')
    >>> reference, hypothesis = apply_mimo_assignment(assignment, reference, hypothesis)
    >>> print(reference.dumps())
    X 1 O1 0.0 1.0 a b
    X 1 O1 1.0 2.0 c d
    <BLANKLINE>
    >>> print(hypothesis.dumps())
    X 1 O1 0.0 2.0 c d
    X 1 O0 0.0 2.0 a b e f
    <BLANKLINE>
    """

    try:
        r_conv = asseglistconvertible(reference, py_convert=None)
    except Exception:
        pass
    else:
        reference = r_conv.to_seglst()
        
        # Order is given by segment_index, or the order in which the 
        # segments were added to the list, if segment_index is not 
        # available
        if 'segment_index' in reference.T.keys():
            reference = reference.sorted('segment_index')
        else:
            reference = add_segment_index(reference)

        reference = {
            k: list(v.groupby('segment_index').values())
            for k, v in reference.groupby('speaker').items()
        }
        new_reference = []
        for r, h in assignment:
            segments = reference[r].pop(0)
            for s in segments:
                new_reference.append({**s, 'speaker': h})

        return r_conv.new(new_reference), hypothesis

    reference_new = {k: [] for k in _keys(hypothesis)}
    # convert to list and copy
    reference = {k: list(v) for k, v in _items(reference)}

    for r, h in assignment:
        reference_new[h].append(reference[r].pop(0))

    if isinstance(hypothesis, dict):
        return dict(reference_new), hypothesis
    elif isinstance(hypothesis, (list, tuple)):
        return type(hypothesis)(reference_new.values()), hypothesis
    else:
        raise TypeError(type(hypothesis), hypothesis)
