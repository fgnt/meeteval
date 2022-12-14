import string
import collections
import typing
from meeteval._typing import Hashable, List, Tuple, Optional, Dict, Literal


__all__ = [
    'apply_mimo_assignment',
    'apply_orc_assignment',
    'apply_cp_assignment',
]


def apply_mimo_assignment(
        assignment: 'List[tuple]',
        reference: 'List[str]',
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
    """
    if isinstance(reference, (tuple, list)):
        reference = {
            k: list(v)  # convert to list and copy
            for k, v in enumerate(reference)
        }
        reference_new = [
            []
            for _ in hypothesis
        ]
    else:
        reference = {
            k: list(v)  # convert to list and copy
            for k, v in reference.items()
        }
        reference_new = {
            k: []
            for k in hypothesis.keys()
        }

    for r, h in assignment:
        reference_new[h].append(reference[r].pop(0))

    return reference_new, hypothesis




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
    """
    ref = collections.defaultdict(list)

    assert len(reference) == len(assignment), (len(reference), len(assignment))
    for r, a in zip(reference, assignment):
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
    elif isinstance(reference, (tuple, list)) and isinstance(hypothesis, (tuple, list)):
        fallback_keys = iter(range(
            min(len(reference), len(hypothesis)),
            max(len(reference), len(hypothesis)),
        ))
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

    if isinstance(reference, dict) and isinstance(hypothesis, dict):
        reference_new = {}
        hypothesis_new = {}

        def get(obj, key, default):
            return obj.get(key, default)

    elif isinstance(reference, (tuple, list)) and isinstance(hypothesis, (tuple, list)):
        max_len = len({k for ks in assignment for k in ks if k is not None})
        reference_new = [missing] * max_len
        hypothesis_new = [missing] * max_len

        def get(obj, key, default):
            if key is None:
                return default
            return obj[key] if len(obj) > key else default
    else:
        raise TypeError(type(reference), type(hypothesis))

    for r_key, h_key in assignment:
        k = get_key(r_key, h_key)
        reference_new[k] = get(reference, r_key, missing)
        hypothesis_new[k] = get(hypothesis, h_key, missing)

    return reference_new, hypothesis_new
