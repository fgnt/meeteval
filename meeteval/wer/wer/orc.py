import collections
import dataclasses
from typing import Iterable, Any

from meeteval.io.seglst import asseglst, SegLST
from meeteval.wer.wer.error_rate import ErrorRate, combine_error_rates
from meeteval.wer.wer.siso import _siso_error_rate
from meeteval.wer.utils import _items, _keys, _values, _map
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


def orc_word_error_rate_multifile(
        reference,
        hypothesis
) -> 'dict[str, OrcErrorRate]':
    """
    Computes the ORC WER for each example in the reference and hypothesis files.

    To compute the overall WER, use
    `sum(orc_word_error_rate_multifile(r, h).values())`.
    """
    from meeteval.io.seglst import apply_multi_file
    return apply_multi_file(orc_word_error_rate, reference, hypothesis)


def orc_error_rate(
        reference: 'list[Iterable]',
        hypothesis: 'list[Iterable] | dict[Any, Iterable]',
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

    from meeteval.wer.matching.mimo_matching import mimo_matching
    distance, assignment = mimo_matching([reference], _values(hypothesis))
    hypothesis_keys = list(_keys(hypothesis))
    assignment = tuple([hypothesis_keys[x[1]] for x in assignment])

    reference_new, hypothesis = apply_orc_assignment(
        assignment,
        reference=reference,
        hypothesis=hypothesis,
    )

    er = combine_error_rates(*[
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
        hypothesis_self_overlap=None,
        reference_self_overlap=None,
    )



def orc_word_error_rate(
        reference: 'SegLST', hypothesis: 'SegLST'
) -> OrcErrorRate:
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

    >>> orc_word_error_rate(STM.parse('X 1 A 0.0 1.0 a b\\nX 1 A 1.0 2.0 c d\\nX 1 B 0.0 2.0 e f\\n'), STM.parse('X 1 1 0.0 2.0 c d\\nX 1 0 0.0 2.0 a b e f\\n'))
    OrcErrorRate(error_rate=0.0, errors=0, length=6, insertions=0, deletions=0, substitutions=0, assignment=('0', '0', '1'))

    >>> er = orc_word_error_rate(['a', 'c d', 'e'], {'A': 'a c', 'B': 'd e'})
    >>> er
    OrcErrorRate(error_rate=0.5, errors=2, length=4, insertions=1, deletions=1, substitutions=0, assignment=('A', 'A', 'B'))
    >>> er.apply_assignment(['a', 'c d', 'e'], {'A': 'a c', 'B': 'd e'})
    ({'A': ['a', 'c d'], 'B': ['e']}, {'A': 'a c', 'B': 'd e'})
    """
    # Convert to SegLST
    reference = asseglst(reference, required_keys=('words',))
    hypothesis = asseglst(hypothesis, required_keys=('words', 'speaker'))

    if 'start_time' in reference.T.keys():
        reference = reference.sorted('start_time')
    if 'start_time' in hypothesis.T.keys():
        hypothesis = hypothesis.sorted('start_time')

    reference = [w.split() for w in reference.T['words'] if w]
    hypothesis = {
        speaker: [w for s in seglst if s['words'] for w in s['words'].split()]
        for speaker, seglst in hypothesis.groupby('speaker').items()
    }

    # if isinstance(reference, STM) or isinstance(hypothesis, STM):
    #     from meeteval.wer.wer.utils import _check_valid_input_files
    #     _check_valid_input_files(reference, hypothesis)
    #     reference = reference.utterance_transcripts()
    #     hypothesis = {
    #         speaker_id: h_.merged_transcripts()
    #         for speaker_id, h_ in hypothesis.grouped_by_speaker_id().items()
    #     }
    #
    # reference_words = [r.split() for r in reference]
    # hypothesis_words = _map(str.split, hypothesis)
    return orc_error_rate(reference, hypothesis)


def apply_orc_assignment(
        assignment: 'list[tuple]',
        reference: 'list[str]',
        hypothesis: 'list[str] | dict[str]',
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
