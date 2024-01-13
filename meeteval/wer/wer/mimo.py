import dataclasses
from typing import Iterable, Any

from meeteval.io.seglst import asseglst
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


def mimo_word_error_rate(reference, hypothesis) -> MimoErrorRate:
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
    MimoErrorRate(error_rate=0.0, errors=0, length=6, insertions=0, deletions=0, substitutions=0, assignment=[('A', '0'), ('B', '0'), ('A', '1')])
    """
    reference = asseglst(reference)
    hypothesis = asseglst(hypothesis)

    # Sort by start time if the start time is available
    # TODO: implement something like reference_sort from time_constrained.py?
    if 'start_time' in reference.T.keys():
        reference = reference.sorted('start_time')
    if 'start_time' in hypothesis.T.keys():
        hypothesis = hypothesis.sorted('start_time')

    # Convert to dict of lists of words
    reference = {
        k: [s['words'].split() for s in v if s['words'] != '']
        for k, v in reference.groupby('speaker').items()
    }
    hypothesis = {
        k: [w for s in v if s['words'] != '' for w in s['words'].split()]
        for k, v in hypothesis.groupby('speaker').items()
    }

    # Call core function
    return mimo_error_rate(reference, hypothesis)


def mimo_word_error_rate_multifile(
        reference,
        hypothesis
) -> 'dict[str, MimoErrorRate]':
    """
    Computes the MIMO WER for each example in the reference and hypothesis
    files.

    To compute the overall WER, use
    `sum(mimo_word_error_rate_multifile(r, h).values())`.
    """
    from meeteval.io.seglst import apply_multi_file
    return apply_multi_file(mimo_word_error_rate, reference, hypothesis)


def apply_mimo_assignment(
        assignment: 'list[tuple]',
        reference: 'list[list[Any]] | dict[list[Any]]',
        hypothesis: 'list[Any] | dict[Any, Any]',
):
    """
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
    """
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
