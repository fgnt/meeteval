"""
This file contains python wrapper functions for different WER definitions
"""
from dataclasses import dataclass, field
from typing import Hashable, List, Tuple, Optional


@dataclass(frozen=True)
class ErrorRate:
    """
    This class represents an error rate. It bundles statistics over the errors
    and makes sure that no wrong arithmetic operations can be performed on
    error rates (e.g., a simple mean).

    This class is frozen because an error rate should not change after it
    has been computed.
    """
    errors: int
    length: int
    error_rate: int = field(init=False)

    @classmethod
    def zero(cls):
        """
        The "neutral element" for error rates.
        Useful as a starting point in sum.
        """
        return ErrorRate(0, 0)

    def __post_init__(self):
        if self.errors < 0:
            raise ValueError()
        if self.length < 0:
            raise ValueError()

        # We have to use object.__setattr__ in frozen dataclass.
        # The alternative would be a property named `error_rate` and a custom
        # repr
        object.__setattr__(
            self, 'error_rate',
            self.errors / self.length if self.length > 0 else None
        )
        assert self.length == 0 or self.error_rate >= 0

    def __add__(self, other: 'ErrorRate'):
        """Combines two error rates"""
        if not isinstance(other, ErrorRate):
            raise ValueError()
        # Return the base class here. Meta information can become
        # meaningless and should be handled in subclasses
        return ErrorRate(
            self.errors + other.errors,
            self.length + other.length,
        )


def combine_error_rates(*error_rates: ErrorRate) -> ErrorRate:
    """
    >>> combine_error_rates(ErrorRate(10, 10), ErrorRate(0, 10))
    ErrorRate(errors=10, length=20, error_rate=0.5)
    >>> combine_error_rates(ErrorRate(10, 10))
    ErrorRate(errors=10, length=10, error_rate=1.0)
    >>> combine_error_rates(*([ErrorRate(10, 10)]*10))
    ErrorRate(errors=100, length=100, error_rate=1.0)
    """
    if len(error_rates) == 1:
        return error_rates[0]
    return sum(error_rates, start=error_rates[0].zero())


def _siso_error_rate(
        reference: List[Hashable],
        hypothesis: List[Hashable]
) -> ErrorRate:
    import editdistance
    return ErrorRate(
        editdistance.distance(reference, hypothesis),
        len(reference)
    )


def siso_word_error_rate(
        reference: str,
        hypothesis: str
) -> ErrorRate:
    """
    The "standard" Single Input speaker, Single Output speaker (SISO) WER.

    This WER definition is what is generally called "WER". It just matches one
    word string against another word string.

    >>> siso_word_error_rate('a b c', 'a b c')
    ErrorRate(errors=0, length=3, error_rate=0.0)
    >>> siso_word_error_rate('a b', 'c d')
    ErrorRate(errors=2, length=2, error_rate=1.0)
    """
    return _siso_error_rate(
        reference.split(),
        hypothesis.split()
    )


def siso_character_error_rate(
        reference: str,
        hypothesis: str,
):
    """
    >>> siso_character_error_rate('abc', 'abc')
    ErrorRate(errors=0, length=3, error_rate=0.0)
    """
    return _siso_error_rate(
        list(reference), list(hypothesis)
    )


@dataclass(frozen=True)
class MimoErrorRate(ErrorRate):
    """
    >>> OrcErrorRate(0, 10, (0, 1)) + OrcErrorRate(10, 10, (1, 0, 1))
    ErrorRate(errors=10, length=20, error_rate=0.5)
    """
    assignment: Tuple[int, ...]


def mimo_word_error_rate(
        reference: List[List[str]],
        hypothesis: List[str],
) -> MimoErrorRate:
    """
    The Multiple Input speaker, Multiple Output channel (MIMO) WER.

    >>> mimo_word_error_rate([['a b c d e f']], ['a b c d e f'])
    ErrorRate(errors=0, length=6, error_rate=0.0)

    # All correct, utterance order between speakers can change
    >>> mimo_word_error_rate([['a b', 'c d'], ['e f']], ['a b', 'e f c d'])
    ErrorRate(errors=0, length=6, error_rate=0.0)
    >>> mimo_word_error_rate([['a b', 'c d'], ['e f']], ['a b', 'c d e f'])
    ErrorRate(errors=0, length=6, error_rate=0.0)
    >>> mimo_word_error_rate([['a b', 'c d'], ['e f']], ['c d', 'a b e f'])
    ErrorRate(errors=0, length=6, error_rate=0.0)

    """
    from meeteval.wer.matching.mimo_matching import mimo_matching_v3
    reference = [
        [u.split() for u in ref_chn]
        for ref_chn in reference
    ]
    length = sum([
        sum([len(u) for u in ref_chn])
        for ref_chn in reference
    ])
    hypothesis = [h.split() for h in hypothesis]
    distance, assignment = mimo_matching_v3(reference, hypothesis)
    return MimoErrorRate(distance, length, assignment)


@dataclass(frozen=True)
class OrcErrorRate(ErrorRate):
    """
    >>> OrcErrorRate(0, 10, (0, 1)) + OrcErrorRate(10, 10, (1, 0, 1))
    ErrorRate(errors=10, length=20, error_rate=0.5)
    """
    assignment: Tuple[int, ...]


def orc_word_error_rate(
        reference: List[str],
        hypothesis: List[str],
):
    """
    The Optimal Reference Combination (ORC) WER, implemented efficiently.

    # All correct on a single channel
    >>> orc_word_error_rate(['a b', 'c d', 'e f'], ['a b c d e f'])
    OrcErrorRate(errors=0, length=6, error_rate=0.0, assignment=(0, 0, 0))

    # All correct on two channels
    >>> orc_word_error_rate(['a b', 'c d', 'e f'], ['a b', 'c d e f'])
    OrcErrorRate(errors=0, length=6, error_rate=0.0, assignment=(0, 1, 1))

    # One utterance is split
    >>> orc_word_error_rate(['a', 'c d', 'e'], ['a c', 'd e'])
    OrcErrorRate(errors=2, length=4, error_rate=0.5, assignment=(0, 0, 1))
    """
    from meeteval.wer.matching.mimo_matching import mimo_matching_v3
    reference = [r.split() for r in reference]
    length = sum([len(r) for r in reference])
    hypothesis = [h.split() for h in hypothesis]
    distance, assignment = mimo_matching_v3([reference], hypothesis)
    assignment = tuple([x[1] for x in assignment])
    return OrcErrorRate(distance, length, assignment)


@dataclass(frozen=True)
class CPErrorRate(ErrorRate):
    """
    Error rate statistics wrapper for the cpWER. Tracks the number of missed,
    false-allarm and scored speakers in addition to word-level errors.

    >>> combine_error_rates(CPErrorRate(0, 10, 1, 0, 3), CPErrorRate(5, 10, 0, 1, 3))
    CPErrorRate(errors=5, length=20, error_rate=0.25, missed_speaker=1, falarm_speaker=1, scored_speaker=6)
    """
    missed_speaker: int
    falarm_speaker: int
    scored_speaker: int
    assignment: Optional[Tuple[int, ...]] = None

    @classmethod
    def zero(cls):
        return cls(0, 0, 0, 0, 0)

    def __add__(self, other: 'CPErrorRate'):
        if not isinstance(other, self.__class__):
            raise ValueError()

        return CPErrorRate(
            self.errors + other.errors,
            self.length + other.length,
            self.missed_speaker + other.missed_speaker,
            self.falarm_speaker + other.falarm_speaker,
            self.scored_speaker + other.scored_speaker,
        )


def cp_word_error_rate(
        reference: List[str],
        hypothesis: List[str],
) -> CPErrorRate:
    """
    The Concatenated minimum Permutation WER (cpWER).

    Each element in `reference` represents a reference speaker.
    Each element in `hypothesis` represents an estimated speaker.

    This implementation uses the Hungarian algorithm, so it works for large
    numbers of speakers.

    The number of over- / under-estimated speakers is tracked and returned
    by the CPErrorRate class. When returned from this function, only one
    (missed_speaker or falarm_speaker) can be unequal to 0, but tracking them
    individually makes when averaging over multiple examples.

    >>> cp_word_error_rate(['a b c', 'd e f'], ['a b c', 'd e f'])
    CPErrorRate(errors=0, length=6, error_rate=0.0, missed_speaker=0, falarm_speaker=0, scored_speaker=2)
    >>> cp_word_error_rate(['a b', 'c d'], ['a b', 'c d', 'e f'])
    CPErrorRate(errors=2, length=4, error_rate=0.5, missed_speaker=0, falarm_speaker=1, scored_speaker=2)
    >>> cp_word_error_rate(['a', 'b', 'c d'], ['a', 'b'])
    """
    import editdistance
    import scipy.optimize
    import numpy as np

    reference = [r.split() for r in reference]
    hypothesis = [h.split() for h in hypothesis]

    cost_matrix = np.array([
        [
            editdistance.eval(tt, et)
            for et in hypothesis
        ]
        for tt in reference
    ])

    # Find the best permutation with hungarian algorithm
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    distances = cost_matrix[row_ind, col_ind]
    distances = list(distances)

    # Handle over-/under-estimation
    if len(hypothesis) > len(reference):
        # Over-estimation: Add full length of over-estimated hypotheses
        # to distance
        for i in sorted(set(range(len(hypothesis))) - set(col_ind)):
            distances.append(len(hypothesis[i]))
    elif len(hypothesis) < len(reference):
        # Under-estimation: Add full length of the unused references
        for i in sorted(set(range(len(reference))) - set(row_ind)):
            distances.append(len(reference[i]))

    # Compute WER from distance
    distance = sum(distances)
    length = sum(map(len, reference))

    return CPErrorRate(
        int(distance), length,
        missed_speaker=max(0, len(reference) - len(hypothesis)),
        falarm_speaker=max(0, len(hypothesis) - len(reference)),
        scored_speaker=len(reference),
        assignment=tuple(map(int, col_ind)),
    )
