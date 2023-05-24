import dataclasses
from typing import Tuple, List, Dict

from .error_rate import ErrorRate
from .siso import siso_word_error_rate
from ..utils import _keys, _items


__all__ = ['MimoErrorRate', 'mimo_word_error_rate', 'apply_mimo_assignment']


@dataclasses.dataclass(frozen=True)
class MimoErrorRate(ErrorRate):
    """
    >>> MimoErrorRate(0, 10, 0, 0, 0, [(0, 0)])
    MimoErrorRate(errors=0, length=10, insertions=0, deletions=0, substitutions=0, error_rate=0.0, assignment=[(0, 0)])
    >>> MimoErrorRate(0, 10, 0, 0, 0, [(0, 0)]) + MimoErrorRate(10, 10, 0, 0, 10, [(0, 0)])
    ErrorRate(errors=10, length=20, insertions=0, deletions=0, substitutions=10, error_rate=0.5)
    """
    assignment: Tuple[int, ...]


def mimo_word_error_rate(
        reference: 'List[List[str]] | Dict[List[str]]',
        hypothesis: 'List[str] | Dict[str]',
) -> MimoErrorRate:
    """
    The Multiple Input speaker, Multiple Output channel (MIMO) WER.

    >>> mimo_word_error_rate([['a b c d e f']], ['a b c d e f'])
    MimoErrorRate(errors=0, length=6, insertions=0, deletions=0, substitutions=0, error_rate=0.0, assignment=[(0, 0)])

    # All correct, utterance order between speakers can change
    >>> mimo_word_error_rate([['a b', 'c d'], ['e f']], ['a b', 'e f c d'])
    MimoErrorRate(errors=0, length=6, insertions=0, deletions=0, substitutions=0, error_rate=0.0, assignment=[(1, 1), (0, 0), (0, 1)])
    >>> mimo_word_error_rate([['a b', 'c d'], ['e f']], ['a b', 'c d e f'])
    MimoErrorRate(errors=0, length=6, insertions=0, deletions=0, substitutions=0, error_rate=0.0, assignment=[(0, 0), (0, 1), (1, 1)])
    >>> mimo_word_error_rate([['a b', 'c d'], ['e f']], ['c d', 'a b e f'])
    MimoErrorRate(errors=0, length=6, insertions=0, deletions=0, substitutions=0, error_rate=0.0, assignment=[(0, 1), (1, 1), (0, 0)])

    >>> mimo_word_error_rate({'A': ['a b', 'c d'], 'B': ['e f']},
    ...                      {'O1': 'c d', 'O2': 'a b e f'})
    MimoErrorRate(errors=0, length=6, insertions=0, deletions=0, substitutions=0, error_rate=0.0, assignment=[('A', 'O2'), ('B', 'O2'), ('A', 'O1')])

    """
    if max(len(hypothesis), len(reference)) > 20:
        num_speakers = max(len(hypothesis), len(reference))
        raise RuntimeError(
            f'Are you sure?\n'
            f'Found a total of {num_speakers} speakers in the input.\n'
            f'This indicates a mistake in the input, or does your use-case '
            f'really require scoring with that many speakers?\n'
            f'See https://github.com/fgnt/meeteval/blob/main/doc/num_speaker_limits.md for details.'
        )

    from meeteval.wer.matching.mimo_matching import mimo_matching

    def to_list(obj):
        if isinstance(obj, dict):
            return obj.values()
        else:
            return obj

    def get_keys(obj):
        if isinstance(obj, dict):
            return list(obj.keys())
        else:
            return list(range(len(obj)))

    reference_keys = get_keys(reference)
    hypothesis_keys = get_keys(hypothesis)

    reference_values = [to_list(r) for r in to_list(reference)]
    hypothesis_values = to_list(hypothesis)

    reference_words = [
        [u.split() for u in ref_chn]
        for ref_chn in reference_values
    ]
    hypothesis_words = [h.split() for h in hypothesis_values]
    distance, assignment = mimo_matching(reference_words, hypothesis_words)

    assignment = [
        (reference_keys[r], hypothesis_keys[h]) for r, h in assignment]

    reference_new, hypothesis = apply_mimo_assignment(
        assignment,
        reference=reference,
        hypothesis=hypothesis,
    )

    er = sum([
        siso_word_error_rate(
            ' '.join(r),  # Concatenate all utterances from one speaker
            h,
        )
        for r, h in zip(to_list(reference_new), to_list(hypothesis))
    ])
    assert er.errors == distance, (distance, er)

    return MimoErrorRate(
        er.errors, er.length,
        insertions=er.insertions,
        deletions=er.deletions,
        substitutions=er.substitutions,
        assignment=assignment,
    )


def apply_mimo_assignment(
        assignment: 'List[tuple]',
        reference: 'List[List[str]] | Dict[List[str]]',
        hypothesis: 'List[str] | Dict[str]',
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
