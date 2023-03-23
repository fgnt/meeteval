import dataclasses
import itertools
import string
from typing import Optional, Tuple

from .error_rate import ErrorRate
from .siso import siso_word_error_rate
from ..utils import _items

__all__ = ['CPErrorRate', 'cp_word_error_rate', 'apply_cp_assignment']


@dataclasses.dataclass(frozen=True)
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


def apply_cp_assignment(
        assignment: 'List[tuple]',
        reference: dict,
        hypothesis: dict,
        style: 'Literal["hyp", "ref"]' = 'ref',
        fallback_keys=string.ascii_letters,
        missing='',
):
    """
    Apply the assignment, so that reference and hypothesis have the same
    keys.

    The code is roughly:
        if style == 'ref':
            hypothesis = {
                r_key: hypothesis[h_key]
                for r_key, h_key in assignment
            }
        elif style == 'hyp':
            reference = {
                h_key: reference[r_key]
                for r_key, h_key in assignment
            }
    On top of this, this code takes care of different number of speakers.
    In that case, a fallback_key is used and inserted if nessesary.
    Missing speakers get `missing` as value.

    >>> from IPython.lib.pretty import pprint

    >>> def test(assignment):
    ...     reference = {k: f'{k}ref' for k, _ in assignment if k is not None}
    ...     hypothesis = {k: f'{k}hyp' for _, k in assignment if k is not None}
    ...     pprint(apply_cp_assignment(assignment, reference, hypothesis, style='hyp'))
    ...     pprint(apply_cp_assignment(assignment, reference, hypothesis, style='ref'))

    >>> test([('A', 'O1'), ('B', 'O3'), (None, 'O2')])
    ({'O1': 'Aref', 'O3': 'Bref', 'O2': ''},
     {'O1': 'O1hyp', 'O3': 'O3hyp', 'O2': 'O2hyp'})
    ({'A': 'Aref', 'B': 'Bref', 'a': ''},
     {'A': 'O1hyp', 'B': 'O3hyp', 'a': 'O2hyp'})

    >>> test([('A', 'A')])  # Same keys, matching assignment
    ({'A': 'Aref'}, {'A': 'Ahyp'})
    ({'A': 'Aref'}, {'A': 'Ahyp'})
    >>> test([('A', 'O')])  # Different keys
    ({'O': 'Aref'}, {'O': 'Ohyp'})
    ({'A': 'Aref'}, {'A': 'Ohyp'})
    >>> test([('A', 'B'), ('B', 'A')])  # Swap keys
    ({'B': 'Aref', 'A': 'Bref'}, {'B': 'Bhyp', 'A': 'Ahyp'})
    ({'A': 'Aref', 'B': 'Bref'}, {'A': 'Bhyp', 'B': 'Ahyp'})
    >>> test([('A', 'M'), ('B', 'N'), ('M', 'O')])  # Key confusion
    ({'M': 'Aref', 'N': 'Bref', 'O': 'Mref'},
     {'M': 'Mhyp', 'N': 'Nhyp', 'O': 'Ohyp'})
    ({'A': 'Aref', 'B': 'Bref', 'M': 'Mref'},
     {'A': 'Mhyp', 'B': 'Nhyp', 'M': 'Ohyp'})
    >>> test([('A', 'M'), ('M', None)])  # Key confusion fallback
    ({'M': 'Aref', 'a': 'Mref'}, {'M': 'Mhyp', 'a': ''})
    ({'A': 'Aref', 'M': 'Mref'}, {'A': 'Mhyp', 'M': ''})

    >>> def test_list(assignment):
    ...     reference = {k: f'{k}ref' for k, _ in assignment if k is not None}
    ...     hypothesis = {k: f'{k}hyp' for _, k in assignment if k is not None}
    ...     reference = list(dict(sorted(reference.items())).values())
    ...     hypothesis = list(dict(sorted(hypothesis.items())).values())
    ...     pprint(apply_cp_assignment(assignment, reference, hypothesis, style='hyp'))
    ...     pprint(apply_cp_assignment(assignment, reference, hypothesis, style='ref'))

    >>> test_list([(0, 0)])
    (['0ref'], ['0hyp'])
    (['0ref'], ['0hyp'])
    >>> test_list([(0, 0), (1, None)])
    (['0ref', '1ref'], ['0hyp', ''])
    (['0ref', '1ref'], ['0hyp', ''])
    >>> test_list([(0, 0), (None, 1)])
    (['0ref', ''], ['0hyp', '1hyp'])
    (['0ref', ''], ['0hyp', '1hyp'])
    """
    assert assignment, assignment

    if isinstance(reference, dict) and isinstance(hypothesis, dict):
        # Check for valid keys
        assert None not in reference, reference.keys()
        assert None not in hypothesis, hypothesis.keys()

        r_keys, h_keys = zip(*assignment)
        r_keys = set(filter(None, r_keys))
        h_keys = set(filter(None, h_keys))
        assert r_keys == set(reference.keys()), (r_keys, reference.keys(), assignment)
        assert h_keys == set(hypothesis.keys()), (h_keys, hypothesis.keys(), assignment)

        fallback_keys = iter([
            k
            for k in fallback_keys
            if k not in reference.keys()
            if k not in hypothesis.keys()
        ])

        reference_new = {}
        hypothesis_new = {}

        def get(obj, key, default):
            return obj.get(key, default)
    elif isinstance(reference, (tuple, list)) and isinstance(hypothesis, (tuple, list)):
        fallback_keys = iter(range(
            min(len(reference), len(hypothesis)),
            max(len(reference), len(hypothesis)),
        ))

        max_len = len({k for ks in assignment for k in ks if k is not None})
        reference_new = [missing] * max_len
        hypothesis_new = [missing] * max_len

        def get(obj, key, default):
            if key is None or key >= len(obj):
                return default
            return obj[key]
    else:
        raise TypeError(type(reference), type(hypothesis))

    for r_key, h_key in assignment:
        assert r_key is not None or h_key is not None, (r_key, h_key)

    if style == 'hyp':
        # Change the keys of the reference to those of the hypothesis
        def get_key(r_key, h_key):
            if h_key is not None:
                return h_key
            return next(fallback_keys)
    elif style == 'ref':
        def get_key(r_key, h_key):
            if r_key is not None:
                return r_key
            return next(fallback_keys)
    else:
        raise ValueError(f'{style!r} not in ["ref", "hyp"]')

    for r_key, h_key in assignment:
        k = get_key(r_key, h_key)
        reference_new[k] = get(reference, r_key, missing)
        hypothesis_new[k] = get(hypothesis, h_key, missing)

    return reference_new, hypothesis_new