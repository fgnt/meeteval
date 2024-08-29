import dataclasses
import functools
import itertools
import string
import typing

import meeteval
from meeteval._typing import Literal
from meeteval.io.seglst import SegLST, asseglst, asseglistconvertible
from meeteval.wer.preprocess import preprocess

from meeteval.wer.wer.error_rate import ErrorRate

if typing.TYPE_CHECKING:
    T = typing.TypeVar('T')
    from typing import Optional, Any


__all__ = [
    'CPErrorRate',
    'cp_word_error_rate',
    'apply_cp_assignment',
    'cp_word_error_rate_multifile'
]


@dataclasses.dataclass(frozen=True, repr=False)
class CPErrorRate(ErrorRate):
    """
    Error rate statistics wrapper for the cpWER. Tracks the number of missed,
    false-allarm and scored speakers in addition to word-level errors.

    >>> CPErrorRate(0, 10, 0, 0, 0, None, None, 1, 0, 3)
    CPErrorRate(error_rate=0.0, errors=0, length=10, insertions=0, deletions=0, substitutions=0, missed_speaker=1, falarm_speaker=0, scored_speaker=3)
    >>> from meeteval.wer.wer.error_rate import combine_error_rates
    >>> combine_error_rates(CPErrorRate(0, 10, 0, 0, 0, None, None, 1, 0, 3), CPErrorRate(5, 10, 0, 0, 5, None, None, 0, 1, 3))
    CPErrorRate(error_rate=0.25, errors=5, length=20, insertions=0, deletions=0, substitutions=5, missed_speaker=1, falarm_speaker=1, scored_speaker=6)
    """
    missed_speaker: int
    falarm_speaker: int
    scored_speaker: int
    # assignment: 'Optional[tuple[int, ...]]' = None
    assignment: 'Optional[tuple[int | str | Any, ...]]' = None

    @classmethod
    def zero(cls):
        return cls(0, 0, 0, 0, 0, None, None, 0, 0, 0)

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
            reference_self_overlap=(
                self.reference_self_overlap + other.reference_self_overlap
                if self.reference_self_overlap is not None
                   and other.reference_self_overlap is not None
                else None
            ),
            hypothesis_self_overlap=(
                self.hypothesis_self_overlap + other.hypothesis_self_overlap
                if self.hypothesis_self_overlap is not None
                   and other.hypothesis_self_overlap is not None
                else None
            ),
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

        >>> er = CPErrorRate(1, 1, 0, 0, 1, None, None, 1, 1, 1, assignment)
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
        reference: 'SegLST',
        hypothesis: 'SegLST',
        reference_sort='segment_if_available',
        hypothesis_sort='segment_if_available',
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
    CPErrorRate(error_rate=0.0, errors=0, length=6, insertions=0, deletions=0, substitutions=0, missed_speaker=0, falarm_speaker=0, scored_speaker=2, assignment=((0, 0), (1, 1)))
    >>> cp_word_error_rate(['a b', 'c d'], ['a b', 'c d', 'e f'])
    CPErrorRate(error_rate=0.5, errors=2, length=4, insertions=2, deletions=0, substitutions=0, missed_speaker=0, falarm_speaker=1, scored_speaker=2, assignment=((0, 0), (1, 1), (None, 2)))
    >>> cp_word_error_rate(['a', 'b', 'c d'], ['a', 'b'])
    CPErrorRate(error_rate=0.5, errors=2, length=4, insertions=0, deletions=2, substitutions=0, missed_speaker=1, falarm_speaker=0, scored_speaker=3, assignment=((0, 0), (1, 1), (2, None)))

    >>> cp_word_error_rate({'r0': 'a', 'r1': 'b', 'r2': 'c d'}, {'h0': 'a', 'h1': 'b'})
    CPErrorRate(error_rate=0.5, errors=2, length=4, insertions=0, deletions=2, substitutions=0, missed_speaker=1, falarm_speaker=0, scored_speaker=3, assignment=(('r0', 'h0'), ('r1', 'h1'), ('r2', None)))
    >>> er = cp_word_error_rate({'r0': 'a', 'r1': 'b', 'r2': 'c'}, {'h0': 'b', 'h1': 'c', 'h2': 'd', 'h3': 'a'})
    >>> er
    CPErrorRate(error_rate=0.3333333333333333, errors=1, length=3, insertions=1, deletions=0, substitutions=0, missed_speaker=0, falarm_speaker=1, scored_speaker=3, assignment=(('r0', 'h3'), ('r1', 'h0'), ('r2', 'h1'), (None, 'h2')))
    >>> er.apply_assignment({'r0': 'a', 'r1': 'b', 'r2': 'c'}, {'h0': 'b', 'h1': 'c', 'h2': 'd', 'h3': 'a'})
    ({'r0': 'a', 'r1': 'b', 'r2': 'c', 'a': ''}, {'r0': 'a', 'r1': 'b', 'r2': 'c', 'a': 'd'})

    >>> cp_word_error_rate({'r0': 'a b'}, {'h0': 'z', 'h1': 'a e f'})  # Special case for overestimation, that was buggy in the past.
    CPErrorRate(error_rate=1.5, errors=3, length=2, insertions=2, deletions=0, substitutions=1, missed_speaker=0, falarm_speaker=1, scored_speaker=1, assignment=(('r0', 'h1'), (None, 'h0')))

    Whitespace is ignored
    >>> cp_word_error_rate({'r0': 'a b', 'r1': 'k'}, {'h0': ' ', 'h1': 'a e f'})  # Special case for overestimation, that was buggy in the past.
    CPErrorRate(error_rate=1.0, errors=3, length=3, insertions=1, deletions=1, substitutions=1, missed_speaker=0, falarm_speaker=0, scored_speaker=2, assignment=(('r0', 'h1'), ('r1', 'h0')))

    >>> from meeteval.io import STM, STMLine
    >>> r = STM([STMLine.parse('file1 0 r0 0 1 Hello World')])
    >>> h = STM([STMLine.parse('file1 0 h0 0 1 Hello World')])
    >>> cp_word_error_rate(r, h)
    CPErrorRate(error_rate=0.0, errors=0, length=2, insertions=0, deletions=0, substitutions=0, reference_self_overlap=SelfOverlap(overlap_rate=Decimal('0'), overlap_time=0, total_time=Decimal('1')), hypothesis_self_overlap=SelfOverlap(overlap_rate=Decimal('0'), overlap_time=0, total_time=Decimal('1')), missed_speaker=0, falarm_speaker=0, scored_speaker=1, assignment=(('r0', 'h0'),))

    >>> cp_word_error_rate(['a b c'.split(), 'd e f'.split()], ['a b c'.split(), 'd e f'.split()])
    CPErrorRate(error_rate=0.0, errors=0, length=6, insertions=0, deletions=0, substitutions=0, missed_speaker=0, falarm_speaker=0, scored_speaker=2, assignment=((0, 0), (1, 1)))

    """
    reference = asseglst(reference, required_keys=('speaker', 'words'))
    hypothesis = asseglst(hypothesis, required_keys=('speaker', 'words'))
    reference, hypothesis, ref_self_overlap, hyp_self_overlap = preprocess(
        reference, hypothesis,
        keep_keys=('speaker', 'words'),
        remove_empty_segments=False,
        reference_sort=reference_sort,
        hypothesis_sort=hypothesis_sort,
        segment_representation='segment',
    )

    from meeteval.wer.wer.siso import _siso_error_rate
    from meeteval.wer.matching.cy_levenshtein import levenshtein_distance

    error_rate = _minimum_permutation_word_error_rate(
        {k: [w for words in v.T['words'] for w in words] for k, v in reference.groupby('speaker').items()},
        {k: [w for words in v.T['words'] for w in words] for k, v in hypothesis.groupby('speaker').items()},
        distance_fn=levenshtein_distance,
        siso_error_rate=_siso_error_rate,
    )

    return dataclasses.replace(
        error_rate,
        reference_self_overlap=ref_self_overlap,
        hypothesis_self_overlap=hyp_self_overlap,
    )


def cp_word_error_rate_multifile(
        reference, hypothesis, partial=False,
        reference_sort='segment_if_available',
        hypothesis_sort='segment_if_available',
) -> 'dict[str, CPErrorRate]':
    """
    Computes the cpWER for each example in the reference and hypothesis STM files.

    To compute the overall WER, use `sum(cp_word_error_rate_multifile(r, h).values())`.
    """
    from meeteval.io.seglst import apply_multi_file
    return apply_multi_file(
        functools.partial(
            cp_word_error_rate,
            reference_sort=reference_sort,
            hypothesis_sort=hypothesis_sort,
        ),
        reference, hypothesis,
        partial=partial
    )


def _minimum_permutation_assignment(
        reference: 'dict[str, T]',
        hypothesis: 'dict[str, T]',
        distance_fn: 'callable[[T, T], int]',
        missing: 'T' = SegLST([])
) -> '(tuple[int], int)':
    """
    Compute the best (lowest distance) assignment of reference and hypothesis
    speakers based on `distance_fn`.

    Returns:
        The assignment and the distance
        The score matrix. Shape (reference hypothesis)

    >>> _minimum_permutation_assignment({}, {}, lambda x, y: 0)
    ((), 0, array([], dtype=float64))
    >>> _minimum_permutation_assignment({}, {'spkA': meeteval.io.SegLST([])}, lambda x, y: 1)
    (((None, 'spkA'),), 1, array([[1]]))
    >>> _minimum_permutation_assignment({'spkA': meeteval.io.SegLST([])}, {}, lambda x, y: 1)
    ((('spkA', None),), 1, array([[1]]))
    """
    import scipy.optimize
    import numpy as np

    cost_matrix = np.array([
        [
            distance_fn(tt, et)
            for et, _ in itertools.zip_longest(
            hypothesis.values(),
            reference.values(),  # ignored, "padding" for underestimation
            fillvalue=missing,
        )
        ]
        for tt, _ in itertools.zip_longest(
            reference.values(),
            hypothesis.values(),  # ignored, "padding" for overestimation
            fillvalue=missing,
        )
    ])

    if cost_matrix.size == 0:
        return (), 0, cost_matrix

    # Find the best permutation with hungarian algorithm
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    distances = cost_matrix[row_ind, col_ind]
    distances = list(distances)

    # Compute WER from distance
    distance = sum(distances)

    # need `dict.get` of the keys for overestimation
    reference_keys = dict(enumerate(reference.keys()))
    hypothesis_keys = dict(enumerate(hypothesis.keys()))

    assignment = tuple([
        (reference_keys.get(r), hypothesis_keys.get(c))
        for r, c in itertools.zip_longest(row_ind, col_ind)
    ])
    return assignment, int(distance), cost_matrix


def _minimum_permutation_word_error_rate(
        reference: 'dict[str, T]',
        hypothesis: 'dict[str, T]',
        distance_fn: 'callable[[T, T], int]',
        siso_error_rate: 'callable[[T, T], ErrorRate]',
        missing: 'T' = SegLST([]),
) -> CPErrorRate:
    """
    Computes the WER for the best (minimum error rate) assignment of reference
    to hypothesis speakers.

    Statistics (insertions, deletions, ...) are copied from the output of
    `siso_error_rate` after resolving the assignment.

    Performs no preprocessing. This means:
    - reference and hypothesis must be dicts of SegLSTs matching the input
        format of `distance_fn` and `siso_error_rate`
    - no self-overlap is computed
    """
    if max(len(hypothesis), len(reference)) > 20:
        num_speakers = max(len(hypothesis), len(reference))
        raise RuntimeError(
            f'Are you sure?\n'
            f'Found a total of {num_speakers} speakers in the input '
            f'(Reference: {len(reference)}, Hypothesis: {len(hypothesis)}).\n'
            f'This indicates a mistake in the input, or does your use-case '
            f'really require scoring with that many speakers?\n'
            f'See https://github.com/fgnt/meeteval/blob/main/doc/num_speaker_limits.md '
            f'for details.'
        )

    assignment, distance, _ = _minimum_permutation_assignment(
        reference, hypothesis, distance_fn, missing
    )

    missed_speaker = max(0, len(reference) - len(hypothesis))
    falarm_speaker = max(0, len(hypothesis) - len(reference))

    if assignment:
        reference_new, hypothesis_new = apply_cp_assignment(
            assignment,
            reference=reference,
            hypothesis=hypothesis,
            missing=missing,
        )
    else:
        reference_new, hypothesis_new = reference, hypothesis

    er = sum([
        siso_error_rate(r, hypothesis_new[speaker])
        for speaker, r in reference_new.items()
    ], start=ErrorRate.zero())

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
        reference_self_overlap=None,
        hypothesis_self_overlap=None,
    )


def apply_cp_assignment(
        assignment: 'list[tuple[Any, ...]] | tuple[tuple[Any, ...], ...]',
        reference: 'dict | list | tuple | SegLST',
        hypothesis: 'dict | list | tuple | SegLST',
        style: 'Literal["hyp", "ref"]' = 'ref',
        fallback_keys=string.ascii_letters,
        missing='',
):
    """
    Apply the assignment so that reference and hypothesis have the same keys.

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

    Also works for anything convertible to SegLST
    >>> r, h = apply_cp_assignment(
    ...     [('rA', 'hB'), ('rB', 'hA')],
    ...     meeteval.io.STM.parse(
    ...         'file1 0 rA 0 1 Hello World\\n'
    ...         'file1 0 rB 0 1 Goodbye'
    ...     ),
    ...     meeteval.io.STM.parse(
    ...         'file1 0 hB 0 1 Hello World\\n'
    ...         'file1 0 hA 0 1 Goodbye'
    ...     ),
    ...     style='ref'
    ... )
    >>> print(r.dumps())
    file1 0 rA 0 1 Hello World
    file1 0 rB 0 1 Goodbye
    <BLANKLINE>
    >>> print(h.dumps())
    file1 0 rA 0 1 Hello World
    file1 0 rB 0 1 Goodbye
    <BLANKLINE>
    """
    assert assignment, assignment

    try:
        r_conv = asseglistconvertible(reference, py_convert=None)
        h_conv = asseglistconvertible(hypothesis, py_convert=None)
    except Exception:
        # This is a Python structure
        pass
    else:
        reference = r_conv.to_seglst()
        hypothesis = h_conv.to_seglst()

        # Check for valid keys
        r_keys, h_keys = zip(*assignment)
        r_keys = set(r_keys) - {None}
        h_keys = set(h_keys) - {None}
        assert r_keys == reference.unique('speaker'), (r_keys, reference.unique('speaker'), assignment)
        assert h_keys == hypothesis.unique('speaker'), (h_keys, hypothesis.unique('speaker'), assignment)

        fallback_keys_iter = iter([
            k
            for k in fallback_keys
            if k not in r_keys
            if k not in h_keys
        ])

        try:
            if style == 'hyp':
                assignment = {
                    r: h if h is not None else next(fallback_keys_iter)
                    for r, h in assignment
                    if r is not None
                }
                # Change the keys of the reference to those of the hypothesis
                reference = reference.map(lambda s: {
                    **s, 'speaker': assignment[s['speaker']]
                })
            elif style == 'ref':
                assignment = {
                    h: r if r is not None else next(fallback_keys_iter)
                    for r, h in assignment
                    if h is not None
                }
                hypothesis = hypothesis.map(lambda s: {
                    **s, 'speaker': assignment[s['speaker']]
                })
            else:
                raise ValueError(f'{style!r} not in ["ref", "hyp"]')
        except StopIteration:
            raise RuntimeError(
                f'Too few fallback keys provided! '
                f'There are more over-/under-estimated speakers '
                f'than fallback_keys in {fallback_keys}'
            )

        return r_conv.new(reference), h_conv.new(hypothesis)

    if isinstance(reference, dict) and isinstance(hypothesis, dict):
        # Check for valid keys
        assert None not in reference, reference.keys()
        assert None not in hypothesis, hypothesis.keys()

        r_keys, h_keys = zip(*assignment)
        # CB: Why do we need to remove None?
        r_keys = set(r_keys) - {None}
        h_keys = set(h_keys) - {None}
        assert r_keys == set(reference.keys()), (r_keys, reference.keys(), assignment)
        assert h_keys == set(hypothesis.keys()), (h_keys, hypothesis.keys(), assignment)

        fallback_keys_iter = iter([
            k
            for k in fallback_keys
            if k not in reference.keys()
            if k not in hypothesis.keys()
        ])

        reference_new = {}
        hypothesis_new = {}

        def get(obj, key, default):
            return obj.get(key, default)
    elif (isinstance(reference, (tuple, list))
          and isinstance(hypothesis, (tuple, list))):
        fallback_keys_iter = iter(range(
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
            return next(fallback_keys_iter)
    elif style == 'ref':
        def get_key(r_key, h_key):
            if r_key is not None:
                return r_key
            return next(fallback_keys_iter)
    else:
        raise ValueError(f'{style!r} not in ["ref", "hyp"]')

    for r_key, h_key in assignment:
        try:
            k = get_key(r_key, h_key)
        except StopIteration:
            raise RuntimeError(
                f'Too few fallback keys provided! '
                f'There are more over-/under-estimated speakers '
                f'than fallback_keys in {fallback_keys}'
            )
        reference_new[k] = get(reference, r_key, missing)
        hypothesis_new[k] = get(hypothesis, h_key, missing)

    return reference_new, hypothesis_new
