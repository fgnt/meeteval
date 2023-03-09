"""
This file contains python wrapper functions for different WER definitions
"""
import itertools
import typing
import string
import collections

import dataclasses
from dataclasses import dataclass, field
from meeteval._typing import Hashable, List, Tuple, Optional, Dict, Literal

if typing.TYPE_CHECKING:
    from meeteval.io.stm import STM


__all__ = [
    'ErrorRate',
    'mimo_word_error_rate',
    'orc_word_error_rate',
    'orc_word_error_rate_stm',
    'cp_word_error_rate',
    'siso_word_error_rate',
    'siso_character_error_rate',
    'MimoErrorRate',
    'OrcErrorRate',
    'CPErrorRate',
    'combine_error_rates',
]


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

    insertions: int
    deletions: int
    substitutions: int

    error_rate: int = field(init=False)


    @classmethod
    def zero(cls):
        """
        The "neutral element" for error rates.
        Useful as a starting point in sum.
        """
        return ErrorRate(0, 0, 0, 0, 0)

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
        errors = self.insertions + self.deletions + self.substitutions
        if errors != self.errors:
            raise RuntimeError(
                f'ins {self.insertions} + del {self.deletions} + sub {self.substitutions} != errors {self.errors}')

    def __radd__(self, other: 'int') -> 'ErrorRate':
        if isinstance(other, int) and other == 0:
            # Special case to support sum.
            return self
        return NotImplemented

    def __add__(self, other: 'ErrorRate') -> 'ErrorRate':
        """Combines two error rates"""
        if not isinstance(other, ErrorRate):
            raise NotImplemented
        # Return the base class here. Meta information can become
        # meaningless and should be handled in subclasses
        return ErrorRate(
            self.errors + other.errors,
            self.length + other.length,
            insertions=self.insertions + other.insertions,
            deletions=self.deletions + other.deletions,
            substitutions=self.substitutions + other.substitutions,
        )

    @classmethod
    def from_dict(self, d: dict):
        """
        >>> ErrorRate.from_dict(dataclasses.asdict(ErrorRate(1, 1, 0, 0, 1)))
        ErrorRate(errors=1, length=1, insertions=0, deletions=0, substitutions=1, error_rate=1.0)
        >>> ErrorRate.from_dict(dataclasses.asdict(CPErrorRate(1, 1, 0, 0, 1, 1, 1, 1)))
        CPErrorRate(errors=1, length=1, insertions=0, deletions=0, substitutions=1, error_rate=1.0, missed_speaker=1, falarm_speaker=1, scored_speaker=1, assignment=None)
        >>> ErrorRate.from_dict(dataclasses.asdict(OrcErrorRate(1, 1, 0, 0, 1, (0, 1))))
        OrcErrorRate(errors=1, length=1, insertions=0, deletions=0, substitutions=1, error_rate=1.0, assignment=(0, 1))
        >>> ErrorRate.from_dict(dataclasses.asdict(MimoErrorRate(1, 1, 0, 0, 1, [(0, 1)])))
        MimoErrorRate(errors=1, length=1, insertions=0, deletions=0, substitutions=1, error_rate=1.0, assignment=[(0, 1)])
        """
        # For backward compatibility, set default values.
        d.setdefault('insertions', None)
        d.setdefault('deletions', None)
        d.setdefault('substitutions', None)

        if d.keys() == {
                'errors', 'length', 'error_rate',
                'insertions', 'deletions', 'substitutions',
        }:
            return ErrorRate(
                errors=d['errors'],
                length=d['length'],
                insertions=d['insertions'],
                deletions=d['deletions'],
                substitutions=d['substitutions'],
            )

        if d.keys() == {
            'errors', 'length', 'error_rate',
            'insertions', 'deletions', 'substitutions',
            'missed_speaker', 'falarm_speaker', 'scored_speaker',
            'assignment',
        }:
            return CPErrorRate(
                errors=d['errors'], length=d['length'],
                insertions=d['insertions'],
                deletions=d['deletions'],
                substitutions=d['substitutions'],
                missed_speaker=d['missed_speaker'],
                falarm_speaker=d['falarm_speaker'],
                scored_speaker=d['scored_speaker'],
                assignment=d['assignment'],
            )

        if d.keys() == {
            'errors', 'length', 'error_rate',
            'insertions', 'deletions', 'substitutions',
            'assignment',
        }:
            if isinstance(d['assignment'][0], (tuple, list)):
                XErrorRate = MimoErrorRate
            else:
                XErrorRate = OrcErrorRate

            return XErrorRate(
                errors=d['errors'],
                length=d['length'],
                insertions=d['insertions'],
                deletions=d['deletions'],
                substitutions=d['substitutions'],
                assignment=d['assignment'],
            )
        raise ValueError(d.keys(), d)


def combine_error_rates(*error_rates: ErrorRate) -> ErrorRate:
    """
    >>> combine_error_rates(ErrorRate(10, 10, 0, 0, 10), ErrorRate(0, 10, 0, 0, 0))
    ErrorRate(errors=10, length=20, insertions=0, deletions=0, substitutions=10, error_rate=0.5)
    >>> combine_error_rates(ErrorRate(10, 10, 0, 0, 10))
    ErrorRate(errors=10, length=10, insertions=0, deletions=0, substitutions=10, error_rate=1.0)
    >>> combine_error_rates(*([ErrorRate(10, 10, 0, 0, 10)]*10))
    ErrorRate(errors=100, length=100, insertions=0, deletions=0, substitutions=100, error_rate=1.0)
    """
    if len(error_rates) == 1:
        return error_rates[0]
    return sum(error_rates, error_rates[0].zero())


def _siso_error_rate(
        reference: List[Hashable],
        hypothesis: List[Hashable]
) -> ErrorRate:
    import kaldialign

    try:
        result = kaldialign.edit_distance(reference, hypothesis)
    except TypeError:
        raise TypeError(type(reference), type(hypothesis), type(reference[0]), type(hypothesis[0]), reference[0], hypothesis[0])

    return ErrorRate(
        result['total'],
        len(reference),
        insertions=result['ins'],
        deletions=result['del'],
        substitutions=result['sub'],
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
    ErrorRate(errors=0, length=3, insertions=0, deletions=0, substitutions=0, error_rate=0.0)
    >>> siso_word_error_rate('a b', 'c d')
    ErrorRate(errors=2, length=2, insertions=0, deletions=0, substitutions=2, error_rate=1.0)
    >>> siso_word_error_rate(reference='This is wikipedia', hypothesis='This wikipedia')  # Deletion example from https://en.wikipedia.org/wiki/Word_error_rate
    ErrorRate(errors=1, length=3, insertions=0, deletions=1, substitutions=0, error_rate=0.3333333333333333)
    """
    return _siso_error_rate(
        reference.split(),
        hypothesis.split()
    )


def siso_character_error_rate(
        reference: str,
        hypothesis: str,
) -> ErrorRate:
    """
    >>> siso_character_error_rate('abc', 'abc')
    ErrorRate(errors=0, length=3, insertions=0, deletions=0, substitutions=0, error_rate=0.0)
    """
    return _siso_error_rate(
        list(reference), list(hypothesis)
    )


@dataclass(frozen=True)
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
    from meeteval.wer.matching.mimo_matching import mimo_matching_v3

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
    distance, assignment = mimo_matching_v3(reference_words, hypothesis_words)

    assignment = [
        (reference_keys[r], hypothesis_keys[h]) for r, h in assignment]

    from meeteval.wer.assignment import apply_mimo_assignment
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


@dataclass(frozen=True)
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


def orc_word_error_rate_stm(
        reference_stm: 'STM',
        hypothesis_stm: 'STM',
) -> 'Dict[OrcErrorRate]':
    reference = reference_stm.grouped_by_filename()
    hypothesis = hypothesis_stm.grouped_by_filename()
    assert reference.keys() == hypothesis.keys(), (reference.keys(), hypothesis.keys())

    ret = {}
    for filename in reference_stm:
        r = reference[filename].utterance_transcripts()
        h = {
            speaker_id: h_.merged_transcripts()
            for speaker_id, h_ in hypothesis[filename].grouped_by_speaker_id().items()
        }
        ret[filename] = orc_word_error_rate(reference=r, hypothesis=h)

    return ret


def orc_word_error_rate(
        reference: 'List[str]',
        hypothesis: 'List[str] | dict[str]',
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
    from meeteval.wer.matching.mimo_matching import mimo_matching_v3
    reference_words = [r.split() for r in reference]
    # length = sum([len(r) for r in reference])
    hypothesis_keys, hypothesis_words = zip(*[
        (k, h.split()) for k, h in _items(hypothesis)])
    distance, assignment = mimo_matching_v3([reference_words], hypothesis_words)
    assignment = tuple([hypothesis_keys[x[1]] for x in assignment])

    from meeteval.wer.assignment import apply_orc_assignment
    reference_new, hypothesis = apply_orc_assignment(
        assignment,
        reference=reference,
        hypothesis=hypothesis,
    )

    er = sum([
        siso_word_error_rate(
            ' '.join(r),  # Concatenate all utterances from one speaker
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


@dataclass(frozen=True)
class CPErrorRate(ErrorRate):
    """
    Error rate statistics wrapper for the cpWER. Tracks the number of missed,
    false-allarm and scored speakers in addition to word-level errors.

    >>> CPErrorRate(0, 10, 0, 0, 0, 1, 0, 3)
    CPErrorRate(errors=0, length=10, insertions=0, deletions=0, substitutions=0, error_rate=0.0, missed_speaker=1, falarm_speaker=0, scored_speaker=3, assignment=None)
    >>> combine_error_rates(CPErrorRate(0, 10, 0, 0, 0, 1, 0, 3), CPErrorRate(5, 10, 0, 0, 5, 0, 1, 3))
    CPErrorRate(errors=5, length=20, insertions=0, deletions=0, substitutions=5, error_rate=0.25, missed_speaker=1, falarm_speaker=1, scored_speaker=6, assignment=None)
    """
    missed_speaker: int
    falarm_speaker: int
    scored_speaker: int
    # assignment: Optional[Tuple[int, ...]] = None
    assignment: Optional[Tuple['int | str', ...]] = None

    @classmethod
    def zero(cls):
        return cls(0, 0, 0, 0, 0, 0, 0, 0)

    def __add__(self, other: 'CPErrorRate'):
        if not isinstance(other, self.__class__):
            raise ValueError()

        return CPErrorRate(
            self.errors + other.errors,
            self.length + other.length,
            insertions=self.insertions + other.insertions,
            deletions=self.deletions + other.deletions,
            substitutions=self.substitutions + other.substitutions,
            missed_speaker=self.missed_speaker + other.missed_speaker,
            falarm_speaker=self.falarm_speaker + other.falarm_speaker,
            scored_speaker=self.scored_speaker + other.scored_speaker,
        )

    def apply_assignment(
            self,
            reference: dict,
            hypothesis: dict,
            style: 'Literal["hyp", "ref"]' = 'ref',
            fallback_keys=string.ascii_letters,
            missing='',
    ):
        """
        Apply the assignment, so that reference and hypothesis have the same
        keys.

        >>> from IPython.lib.pretty import pprint

        # The assignment is not valid, but contains all tests (e.g. 'O2' and 'C'
          could be assigned to each other to reduce the cpWER).
        >>> assignment = [('A', 'O1'), ('B', 'O3'), (None, 'O2'), ('C', None)]

        >>> er = CPErrorRate(1, 1, 0, 0, 1, 1, 1, 1, assignment)
        >>> reference = {'A': 'Atext', 'B': 'Btext', 'C': 'Ctext'}
        >>> hypothesis = {'O1': 'O1text', 'O2': 'O2text', 'O3': 'O3text'}
        >>> pprint(er.apply_assignment(reference, hypothesis, style='hyp'))
        ({'O1': 'Atext', 'O3': 'Btext', 'O2': '', 'a': 'Ctext'},
         {'O1': 'O1text', 'O3': 'O3text', 'O2': 'O2text', 'a': ''})

        >>> pprint(er.apply_assignment(reference, hypothesis, style='ref'))
        ({'A': 'Atext', 'B': 'Btext', 'a': '', 'C': 'Ctext'},
         {'A': 'O1text', 'B': 'O3text', 'a': 'O2text', 'C': ''})

        """
        from meeteval.wer.assignment import apply_cp_assignment
        return apply_cp_assignment(
            self.assignment,
            reference=reference,
            hypothesis=hypothesis,
            style=style,
            fallback_keys=fallback_keys,
            missing=missing,
        )


def cp_word_error_rate(
        reference: 'List[str] | Dict[str, str]',
        hypothesis: 'List[str] | Dict[str, str]',
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
    CPErrorRate(errors=0, length=6, insertions=0, deletions=0, substitutions=0, error_rate=0.0, missed_speaker=0, falarm_speaker=0, scored_speaker=2, assignment=((0, 0), (1, 1)))
    >>> cp_word_error_rate(['a b', 'c d'], ['a b', 'c d', 'e f'])
    CPErrorRate(errors=2, length=4, insertions=2, deletions=0, substitutions=0, error_rate=0.5, missed_speaker=0, falarm_speaker=1, scored_speaker=2, assignment=((0, 0), (1, 1), (None, 2)))
    >>> cp_word_error_rate(['a', 'b', 'c d'], ['a', 'b'])
    CPErrorRate(errors=2, length=4, insertions=0, deletions=2, substitutions=0, error_rate=0.5, missed_speaker=1, falarm_speaker=0, scored_speaker=3, assignment=((0, 0), (1, 1), (2, None)))

    >>> cp_word_error_rate({'r0': 'a', 'r1': 'b', 'r2': 'c d'}, {'h0': 'a', 'h1': 'b'})
    CPErrorRate(errors=2, length=4, insertions=0, deletions=2, substitutions=0, error_rate=0.5, missed_speaker=1, falarm_speaker=0, scored_speaker=3, assignment=(('r0', 'h0'), ('r1', 'h1'), ('r2', None)))
    >>> er = cp_word_error_rate({'r0': 'a', 'r1': 'b', 'r2': 'c'}, {'h0': 'b', 'h1': 'c', 'h2': 'd', 'h3': 'a'})
    >>> er
    CPErrorRate(errors=1, length=3, insertions=1, deletions=0, substitutions=0, error_rate=0.3333333333333333, missed_speaker=0, falarm_speaker=1, scored_speaker=3, assignment=(('r0', 'h3'), ('r1', 'h0'), ('r2', 'h1'), (None, 'h2')))
    >>> er.apply_assignment({'r0': 'a', 'r1': 'b', 'r2': 'c'}, {'h0': 'b', 'h1': 'c', 'h2': 'd', 'h3': 'a'})
    ({'r0': 'a', 'r1': 'b', 'r2': 'c', 'a': ''}, {'r0': 'a', 'r1': 'b', 'r2': 'c', 'a': 'd'})
    """
    import editdistance
    import scipy.optimize
    import numpy as np

    if isinstance(hypothesis, dict):
        hypothesis_keys = list(hypothesis.keys())
        hypothesis_values = list(hypothesis.values())
    else:
        hypothesis_keys = list(range(len(hypothesis)))
        hypothesis_values = hypothesis
    if isinstance(reference, dict):
        reference_keys = list(reference.keys())
        reference_values = list(reference.values())
    else:
        reference_keys = list(range(len(reference)))
        reference_values = reference

    try:
        reference_words = [r.split() for r in reference_values]
    except AttributeError:
        raise ValueError(reference)
    hypothesis_words = [h.split() for h in hypothesis_values]

    cost_matrix = np.array([
        [
            editdistance.eval(tt, et)
            for et in hypothesis_words
        ]
        for tt in reference_words
    ])

    # Find the best permutation with hungarian algorithm
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    distances = cost_matrix[row_ind, col_ind]
    distances = list(distances)

    # Handle over-/under-estimation
    if len(hypothesis_words) > len(reference_words):
        # Over-estimation: Add full length of over-estimated hypotheses
        # to distance
        none_assigned = sorted(set(range(len(hypothesis_words))) - set(col_ind))
        for i in none_assigned:
            distances.append(len(hypothesis_words[i]))
        col_ind = [*col_ind, *none_assigned]
    elif len(hypothesis_words) < len(reference_words):
        # Under-estimation: Add full length of the unused references
        none_assigned = sorted(set(range(len(reference_words))) - set(row_ind))
        for i in none_assigned:
            distances.append(len(reference_words[i]))
        row_ind = [*row_ind, *none_assigned]

    # Compute WER from distance
    distance = sum(distances)

    assignment = tuple([
        (
            reference_keys[r] if r is not None else r,
            hypothesis_keys[c] if c is not None else c,
        )
        for r, c in itertools.zip_longest(row_ind, col_ind)
    ])

    missed_speaker = max(0, len(reference) - len(hypothesis))
    falarm_speaker = max(0, len(hypothesis) - len(reference))

    from meeteval.wer.assignment import apply_cp_assignment
    reference_new, hypothesis_new = apply_cp_assignment(
        assignment,
        reference=reference,
        hypothesis=hypothesis,
    )

    er = sum([
        siso_word_error_rate(r, hypothesis_new[speaker])
        for speaker, r in _items(reference_new)
    ])
    assert distance == er.errors, (distance, er)

    return CPErrorRate(
        er.errors, er.length,
        insertions=er.insertions,
        deletions=er.deletions,
        substitutions=er.substitutions,
        missed_speaker=missed_speaker,
        falarm_speaker=falarm_speaker,
        scored_speaker=len(reference),
        assignment=assignment,
    )


def _items(obj):
    if isinstance(obj, dict):
        return obj.items()
    elif isinstance(obj, (tuple, list)):
        return enumerate(obj)
    else:
        raise TypeError(type(obj), obj)
