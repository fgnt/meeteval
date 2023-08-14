import dataclasses
from typing import Tuple, List, Dict, Iterable, Any

from meeteval.wer.wer.error_rate import ErrorRate
from meeteval.wer.wer.siso import siso_word_error_rate, _siso_error_rate
from meeteval.wer.utils import _keys, _items, _values, _map

__all__ = ['MimoErrorRate', 'mimo_word_error_rate', 'apply_mimo_assignment', 'mimo_word_error_rate_stm']

from meeteval.io import STM


@dataclasses.dataclass(frozen=True)
class MimoErrorRate(ErrorRate):
    """
    >>> MimoErrorRate(0, 10, 0, 0, 0, [(0, 0)])
    MimoErrorRate(errors=0, length=10, insertions=0, deletions=0, substitutions=0, error_rate=0.0, assignment=[(0, 0)])
    >>> MimoErrorRate(0, 10, 0, 0, 0, [(0, 0)]) + MimoErrorRate(10, 10, 0, 0, 10, [(0, 0)])
    ErrorRate(errors=10, length=20, insertions=0, deletions=0, substitutions=10, error_rate=0.5)
    """
    assignment: Tuple[int, ...]


def mimo_error_rate(
    reference: 'List[List[Iterable]] | Dict[Any, List[Iterable]]',
    hypothesis: 'List[Iterable] | Dict[Iterable]',
):
    if max(len(hypothesis), len(reference)) > 10:
        num_speakers = max(len(hypothesis), len(reference))
        raise RuntimeError(
            f'Are you sure?\n'
            f'Found a total of {num_speakers} speakers in the input.\n'
            f'This indicates a mistake in the input, or does your use-case '
            f'really require scoring with that many speakers?\n'
            f'See https://github.com/fgnt/meeteval/blob/main/doc/num_speaker_limits.md for details.'
        )

    from meeteval.wer.matching.mimo_matching import mimo_matching

    distance, assignment = mimo_matching([_values(r) for r in _values(reference)], _values(hypothesis))

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
    )





def mimo_word_error_rate(
        reference: 'List[List[str] | Dict[Any, str]] | Dict[List[str], Dict[Any, str]] | STM',
        hypothesis: 'List[str] | Dict[str] | STM',
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
    if isinstance(reference, STM) or isinstance(hypothesis, STM):
        from meeteval.wer.wer.utils import _check_valid_input_files
        _check_valid_input_files(reference, hypothesis)
        reference = {
            speaker_id: r.utterance_transcripts()
            for speaker_id, r in reference.grouped_by_speaker_id().items()
        }
        hypothesis = {
            speaker_id: h.merged_transcripts()
            for speaker_id, h in hypothesis.grouped_by_speaker_id().items()
        }

    reference = _map(lambda x: _map(str.split, x), reference)
    hypothesis = _map(str.split, hypothesis)
    return mimo_error_rate(reference, hypothesis)


def mimo_word_error_rate_stm(reference_stm: 'STM', hypothesis_stm: 'STM') -> 'Dict[str, MimoErrorRate]':
    """
    Computes the MIMO WER for each example in the reference and hypothesis STM files.

    To compute the overall WER, use `sum(mimo_word_error_rate_stm(r, h).values())`.
    """
    from meeteval.io.stm import apply_stm_multi_file
    return apply_stm_multi_file(mimo_word_error_rate, reference_stm, hypothesis_stm)


def apply_mimo_assignment(
        assignment: 'List[tuple]',
        reference: 'List[List[Any]] | Dict[List[Any]]',
        hypothesis: 'List[Any] | Dict[Any, Any]',
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
