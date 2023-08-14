import collections
import dataclasses
import typing
from typing import Tuple, List, Dict, Iterable, Any

from .error_rate import ErrorRate
from .siso import _siso_error_rate
from ..utils import _items, _keys, _values, _map
from meeteval.io.stm import STM

__all__ = ['OrcErrorRate', 'orc_word_error_rate', 'orc_word_error_rate_stm', 'apply_orc_assignment']


@dataclasses.dataclass(frozen=True)
class OrcErrorRate(ErrorRate):
    """
    >>> OrcErrorRate(0, 10, 0, 0, 0, (0, 1))
    OrcErrorRate(errors=0, length=10, insertions=0, deletions=0, substitutions=0, error_rate=0.0, assignment=(0, 1))
    >>> OrcErrorRate(0, 10, 0, 0, 0, (0, 1)) + OrcErrorRate(10, 10, 0, 0, 10, (1, 0, 1))
    ErrorRate(errors=10, length=20, insertions=0, deletions=0, substitutions=10, error_rate=0.5)
    """
    assignment: Tuple[int, ...]

    def apply_assignment(self, reference, hypothesis):
        ref = collections.defaultdict(list)

        assert len(reference) == len(self.assignment), (len(reference), len(self.assignment))
        for r, a in zip(reference, self.assignment):
            ref[a].append(r)

        if isinstance(hypothesis, dict):
            ref = dict(ref)
        elif isinstance(hypothesis, list):
            ref = list(ref.values())
        elif isinstance(hypothesis, tuple):
            ref = list(ref.values())
        else:
            raise TypeError(type(hypothesis), hypothesis)

        return ref, hypothesis


def orc_word_error_rate_stm(reference_stm: 'STM', hypothesis_stm: 'STM') -> 'Dict[str, OrcErrorRate]':
    """
    Computes the ORC WER for each example in the reference and hypothesis STM files.

    To compute the overall WER, use `sum(orc_word_error_rate_stm(r, h).values())`.
    """
    from meeteval.io.stm import apply_stm_multi_file
    return apply_stm_multi_file(orc_word_error_rate, reference_stm, hypothesis_stm)


def orc_error_rate(
        reference: 'List[Iterable]',
        hypothesis: 'List[Iterable] | Dict[Any, Iterable]',
):
    # Safety check: The complexity explodes for large numbers of speakers
    if len(hypothesis) > 10:
        raise RuntimeError(
            f'Are you sure?\n'
            f'Found a total of {len(hypothesis)} speakers in the input.\n'
            f'This indicates a mistake in the input, or does your use-case '
            f'really require scoring with that many speakers?\n'
            f'See https://github.com/fgnt/meeteval/blob/main/doc/num_speaker_limits.md for details.'
        )

    from meeteval.wer.matching.mimo_matching import mimo_matching_v3
    distance, assignment = mimo_matching_v3([reference], _values(hypothesis))
    assignment = tuple([_keys(hypothesis)[x[1]] for x in assignment])

    reference_new, hypothesis = apply_orc_assignment(
        assignment,
        reference=reference,
        hypothesis=hypothesis,
    )

    er = sum([
        _siso_error_rate(
            [t for r_ in r for t in r_],  # Create list with all words from one speaker
            hypothesis[speaker],
        )
        for speaker, r in _items(reference_new)
    ])
    assert er.errors == distance, (distance, er)

    return OrcErrorRate(
        er.errors, er.length,
        insertions=er.insertions,
        deletions=er.deletions,
        substitutions=er.substitutions,
        assignment=assignment,
    )



def orc_word_error_rate(
        reference: 'List[str] | STM',
        hypothesis: 'List[str] | dict[str] | STM',
) -> OrcErrorRate:
    """
    The Optimal Reference Combination (ORC) WER, implemented efficiently.

    # All correct on a single channel
    >>> orc_word_error_rate(['a b', 'c d', 'e f'], ['a b c d e f'])
    OrcErrorRate(errors=0, length=6, insertions=0, deletions=0, substitutions=0, error_rate=0.0, assignment=(0, 0, 0))

    # All correct on two channels
    >>> orc_word_error_rate(['a b', 'c d', 'e f'], ['a b', 'c d e f'])
    OrcErrorRate(errors=0, length=6, insertions=0, deletions=0, substitutions=0, error_rate=0.0, assignment=(0, 1, 1))

    # One utterance is split
    >>> er = orc_word_error_rate(['a', 'c d', 'e'], ['a c', 'd e'])
    >>> er
    OrcErrorRate(errors=2, length=4, insertions=1, deletions=1, substitutions=0, error_rate=0.5, assignment=(0, 0, 1))
    >>> er.apply_assignment(['a', 'c d', 'e'], ['a c', 'd e'])
    ([['a', 'c d'], ['e']], ['a c', 'd e'])

    >>> er = orc_word_error_rate(['a', 'c d', 'e'], {'A': 'a c', 'B': 'd e'})
    >>> er
    OrcErrorRate(errors=2, length=4, insertions=1, deletions=1, substitutions=0, error_rate=0.5, assignment=('A', 'A', 'B'))
    >>> er.apply_assignment(['a', 'c d', 'e'], {'A': 'a c', 'B': 'd e'})
    ({'A': ['a', 'c d'], 'B': ['e']}, {'A': 'a c', 'B': 'd e'})
    """
    if isinstance(reference, STM) or isinstance(hypothesis, STM):
        from meeteval.wer.wer.utils import _check_valid_input_files
        _check_valid_input_files(reference, hypothesis)
        reference = reference.utterance_transcripts()
        hypothesis = {
            speaker_id: h_.merged_transcripts()
            for speaker_id, h_ in hypothesis.grouped_by_speaker_id().items()
        }

    reference_words = [r.split() for r in reference]
    hypothesis_words = _map(str.split, hypothesis)
    return orc_error_rate(reference_words, hypothesis_words)


def apply_orc_assignment(
        assignment: 'List[tuple]',
        reference: 'List[str]',
        hypothesis: 'List[str] | dict[str]',
):
    """
    >>> assignment = ('A', 'A', 'B')
    >>> apply_orc_assignment(assignment, ['a', 'c d', 'e'], {'A': 'a c', 'B': 'd e'})
    ({'A': ['a', 'c d'], 'B': ['e']}, {'A': 'a c', 'B': 'd e'})

    >>> assignment = (0, 0, 1)
    >>> apply_orc_assignment(assignment, ['a', 'c d', 'e'], ['a c', 'd e'])
    ([['a', 'c d'], ['e']], ['a c', 'd e'])

    >>> assignment = ('A', )
    >>> apply_orc_assignment(assignment, ['a'], {'A': 'b', 'B': 'c'})
    ({'A': ['a'], 'B': []}, {'A': 'b', 'B': 'c'})
    """
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
