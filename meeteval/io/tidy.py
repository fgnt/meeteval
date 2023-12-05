"""
A "tidy" or "wide" representation of transcripts.

This format can be converted from and to any other format (e.g., STM, CTM(Group), CSV, Chime7 JSON, plain Python structures, ...).

The name is inspired by R's definition of tidy data:
 - https://cran.r-project.org/web/packages/tidyr/vignettes/tidy-data.html#:~:text=Tidy%20data%20is%20a%20standard,with%20observations%2C%20variables%20and%20types.
 - https://vita.had.co.nz/papers/tidy-data.pdf

"""
import dataclasses
import functools
from typing import List, TypedDict, Callable

from cached_property import cached_property


class TidySegment(TypedDict, total=False):
    """
    A segment in tidy format. This is a pure-Python data structure.

    Note:
        We do not define an enum with all these keys for speed reasons
    """
    session_id: str
    start_time: float
    end_time: float
    words: str
    speaker: str
    segment_index: int

    # Unused but by MeetEval but present in some file formats. They are defined here for compatibility and
    # conversion in both directions
    channel: int
    confidence: float


@dataclasses.dataclass(frozen=True)
class Tidy:
    """
    A collection of segments in tidy format. This is a pure-Python data structure and the input type to most
    functions in MeetEval that process transcript segments.
    """
    segments: 'List[TidySegment]'

    @cached_property
    def keys(self):
        """
        Keys that are common among all segments
        """
        if not self.segments:
            return set()
        return set.intersection(*[set(s.keys()) for s in self])

    def unique(self, key):
        """
        Returns the unique values for `key` among all segments.

        TODO: cache
        """
        return set([s[key] for s in self.segments])

    def __iter__(self):
        return iter(self.segments)

    def __getitem__(self, item):
        return self.segments[item]

    def __len__(self):
        return len(self.segments)

    def __add__(self, other):
        if isinstance(other, Tidy):
            return Tidy(self.segments + other.segments)
        return NotImplemented

    def groupby(self, key):
        """
        >>> t = convert_to_tidy(['a b c', 'd e f', 'g h i'], keys=('speaker',))
        >>> t.segments
        [{'words': 'a b c', 'speaker': 0}, {'words': 'd e f', 'speaker': 1}, {'words': 'g h i', 'speaker': 2}]

        >>> from pprint import pprint
        >>> pprint(t.groupby('speaker')) # doctest: +ELLIPSIS
        {0: Tidy(segments=[{'speaker': 0, 'words': 'a b c'}],
                 inversion_fn=...),
         1: Tidy(segments=[{'speaker': 1, 'words': 'd e f'}],
                 inversion_fn=...),
         2: Tidy(segments=[{'speaker': 2, 'words': 'g h i'}],
                 inversion_fn=...}
        >>> t.groupby('speaker')[0].invert()
        ['a b c']
        """
        return {k: Tidy(g) for k, g in groupby(self.segments, key=key).items()}

    def sorted(self, key):
        """
        Returns a copy of this object with the segments sorted by `key`.
        """
        return Tidy(sorted(self.segments, key=_get_key(key)))

    def map(self, fn):
        """
        Applies `fn` to all segments and returns a new `Tidy` object with the results.
        """
        return Tidy([fn(s) for s in self.segments])

    def flatmap(self, fn):
        """
        Applies `fn` to all segments, flattens the output and returns a new `Tidy` object with the results.

        Example:
            >>> Tidy([{'words': 'a b c'}]).flatmap(lambda x: [{'words': w} for w in x['words'].split()])
            Tidy(segments=[{'words': 'a'}, {'words': 'b'}, {'words': 'c'}])
        """
        return Tidy([s for t in self.segments for s in fn(t)])

    def filter(self, fn):
        """
        Applies `fn` to all segments and returns a new `Tidy` object with the segments for which `fn` returns true.
        """
        return Tidy([s for s in self.segments if fn(s)])

    @classmethod
    def merge(cls, *t):
        """
        Merges multiple `Tidy` objects into one by concatenating all segments.
        """
        return Tidy([s for t_ in t for s in t_.segments])

    def to_tidy(self):
        return self

    @classmethod
    def from_tidy(cls, d: 'Tidy', **defaults):
        if defaults:
            d = Tidy([{**defaults, **s} for s in d])
        return d


def _to_convertible(d):
    """
    Converts `d` into a structure that is convertible to the tidy format, i.e., that
    has `to_tidy` (and often `from_tidy`) defined.
    """
    # Already convertible
    if hasattr(d, 'to_tidy'):
        return d

    # Chime7 format / List of `TidySegment`s
    if isinstance(d, list) and (len(d) == 0 or isinstance(d[0], dict) and 'words' in d[0]):
        # TODO: Conversion back?
        return Tidy(d)

    # TODO: pandas DataFrame

    # Convert Python structures
    if isinstance(d, (list, tuple, dict, str)):
        # We support a fixed structure for automatic conversion, as to prevent confusions.
        # If the user wants to use a different format, they can convert it manually.
        from meeteval.io.py import NestedStructure
        return NestedStructure(d, level_keys=('speaker', 'segment_index'))

    raise NotImplementedError(f'No conversion implemented for {type(d)}!')


def convert_to_tidy(d, *, required_keys=()) -> 'Tidy':
    """
    Converts an object `d` into tidy data format. `d` can be anything convertible to the tidy format.

    Python structures
    >>> convert_to_tidy('a b c')
    Tidy(segments=[{'words': 'a b c'}])

    Data formats are also converted
    >>> from meeteval.io.stm import STM, STMLine
    >>> stm = STM([STMLine('ex', 1, 'A', 0, 1, 'a b c')])
    >>> convert_to_tidy(stm).segments
    [{'session_id': 'ex', 'channel': 1, 'speaker': 'A', 'start_time': 0, 'end_time': 1, 'words': 'a b c'}]

    The Tidy representation can be converted back to its original representation
    >>> stm.convert(convert_to_tidy(stm))
    STM(lines=[STMLine(filename='ex', channel=1, speaker_id='A', begin_time=0, end_time=1, transcript='a b c')])

    And modified before inversion
    >>> tidy = convert_to_tidy(stm)
    >>> tidy.segments[0][keys.WORDS] = 'x y z'
    >>> stm.convert(tidy)
    STM(lines=[STMLine(filename='ex', channel=1, speaker_id='A', begin_time=0, end_time=1, transcript='x y z')])

    >>> from meeteval.io.py import NestedStructure
    >>> p = NestedStructure({'A': 'a b c', 'B': 'd e f'}, required_keys=('speaker',))
    >>> tidy = convert_to_tidy(p)
    >>> tidy.segments[0]['speaker'] = 'C'
    >>> p.convert(tidy)
    {'C': 'a b c', 'B': 'd e f'}

    In some cases, (lossless) inversion is not possible after modification.
    Here, words are concatenated, but the original representation cannot be restored.
    >>> p = NestedStructure(['a b c', 'd e f'], required_keys=('speaker',))
    >>> tidy = convert_to_tidy(p)
    >>> tidy.segments[1]['speaker'] = 0
    >>> p.convert(tidy)
    ['a b c d e f']
    """
    assert isinstance(required_keys, tuple), required_keys

    # Get a type that is convertible
    d = _to_convertible(d)

    t = d.to_tidy()

    # Check that `t` has all required keys
    if len(t) and not set(required_keys).issubset(t.keys):
        required_keys = set(required_keys)
        raise ValueError(
            f'Some required keys are not present in the converted data structure!\n'
            f'Required: {required_keys}, found: {t.keys}, missing: {required_keys - t.keys}'
        )
    return t


def _get_key(key):
    import operator
    if callable(key) or key is None:
        return key
    elif isinstance(key, (str, int)):
        return operator.itemgetter(key)
    else:
        raise TypeError(f'Invalid type for key: {type(key)}')


def groupby(
        iterable,
        key=None,
        default_key=None,
):
    """
    A non-lazy variant of `itertools.groupby` with advanced features.

    Copied from `paderbox.utils.iterable.groupby`.

    Args:
        iterable: Iterable to group
        key: Determines by what to group. Can be:
            - `None`: Use the iterables elements as keys directly
            - `callable`: Gets called with every element and returns the group
                key
            - `str`, or `int`: Use `__getitem__` on elements in `iterable`
                to obtain the key
            - `Iterable`: Provides the keys. Has to have the same length as
                `iterable`.

    Examples:
        >>> groupby('ab'*3)
        {'a': ['a', 'a', 'a'], 'b': ['b', 'b', 'b']}
        >>> groupby(range(10), lambda x: x%2)
        {0: [0, 2, 4, 6, 8], 1: [1, 3, 5, 7, 9]}
        >>> groupby(({'a': x%2, 'b': x} for x in range(3)), 'a')
        {0: [{'a': 0, 'b': 0}, {'a': 0, 'b': 2}], 1: [{'a': 1, 'b': 1}]}
        >>> groupby(['abc', 'bd', 'abd', 'cdef', 'c'], 0)
        {'a': ['abc', 'abd'], 'b': ['bd'], 'c': ['cdef', 'c']}
        >>> groupby(range(10), list(range(5))*2)
        {0: [0, 5], 1: [1, 6], 2: [2, 7], 3: [3, 8], 4: [4, 9]}
        >>> groupby('abc', ['a'])
        Traceback (most recent call last):
            ...
        ValueError: zip() argument 2 is shorter than argument 1
        >>> groupby('abc', {})
        Traceback (most recent call last):
            ...
        TypeError: Invalid type for key: <class 'dict'>
    """
    import collections
    import itertools

    groups = collections.defaultdict(list)
    try:
        for key, group in itertools.groupby(iterable, _get_key(key)):
            groups[key].extend(group)
    except KeyError:
        if default_key is None:
            raise
        else:
            assert len(groups) == 0, (groups, iterable, key)
            return iterable
    return dict(groups)


def tidy_args(*required_keys):
    """
    Decorator for a function that require tidy data input to automatically convert other input formats to `Tidy`.

    Automatically converts all positional args to `Tidy` if they are not already `Tidy` and checks if all `keys` are
    present in the structure.

    Arguments:
        required_keys: The keys that must be present in the data structure.
    """
    from functools import wraps
    from inspect import signature

    def _tidy_args_wrapper(fn):
        s = signature(fn)

        num_parameters = len([p for p in s.parameters.values() if p.kind in (
            p.POSITIONAL_ONLY,
            p.POSITIONAL_OR_KEYWORD,
            p.VAR_POSITIONAL,
        )])
        if not required_keys:
            rkeys = ((),) * num_parameters
        elif required_keys and isinstance(required_keys[0], str):
            rkeys = (required_keys,) * num_parameters
        else:
            assert len(required_keys) == num_parameters, (required_keys, num_parameters)
            rkeys = required_keys

        @wraps(fn)
        def _wrapped_tidy(*args, **kwargs):
            bound_args = s.bind(*args, **kwargs)
            args = [
                convert_to_tidy(a, required_keys=k)
                for (a, k) in zip(bound_args.args, rkeys)
            ]
            return fn(*args, **bound_args.kwargs)

        return _wrapped_tidy

    return _tidy_args_wrapper


def tidy_map(*, required_keys=()):
    """
    Decorator to for a function that can be applied to anything convertible to Tidy (and invertible).

    >>> tidy_map(required_keys=('speaker',))(lambda x: x.map(lambda x: {**x, 'speaker': 'X'}))({'A': 'a b c', 'B': 'd e f'})
    {'X': 'a b c d e f'}
    """

    def _tidy_map(fn):
        @functools.wraps(fn)
        def __tidy_map(arg, *args, **kwargs):
            c = _to_convertible(arg)
            arg = convert_to_tidy(c, required_keys=required_keys)
            arg = fn(arg, *args, **kwargs)
            return c.from_tidy(arg)

        return __tidy_map

    return _tidy_map


def apply_multi_file(
        fn: 'Callable[[Tidy, Tidy], ErrorRate]',
        reference, hypothesis,
        *,
        allowed_empty_examples_ratio=0.1
):
    """
    Applies a function individually to all sessions / files.

    `reference` and `hypothesis` must be convertible to `Tidy`. If they are a Python structure, the first level
    is interpreted as the session / file key.

    Pass in data in tidy format. The function is applied to each session individually.
    # >>> from meeteval.wer.wer.error_rate import ErrorRate
    # >>> ref = [{keys.SESSION: 'S1', keys.WORDS: 'A'}, {keys.SESSION: 'S2', keys.WORDS: 'B'}]
    # >>> hyp = [{keys.SESSION: 'S1', keys.WORDS: 'A'}, {keys.SESSION: 'S2', keys.WORDS: 'C'}]
    # >>> apply_multi_file(lambda r, h: ErrorRate(r != h, 0, 0, 0, r != h, None, None), ref, hyp)
    # CombinedErrorRate(errors=1, length=0, insertions=0, deletions=0, substitutions=1, details=...)
    #
    # >>> @tidy_args()
    # ... def fn(ref, hyp):
    # ...     return ErrorRate(ref != hyp, 0, 0, 0, ref != hyp, None, None)
    # >>> apply_multi_file(fn, ref, hyp)
    # CombinedErrorRate(errors=1, length=0, insertions=0, deletions=0, substitutions=1, details=...)

    >>> from meeteval.wer.wer.cp import cp_word_error_rate
    >>> from pprint import pprint
    >>> ref = [['a b c', 'd e f'], ['g h i']]
    >>> hyp = [['a b c'], ['d e f', 'g h i']]
    >>> er = apply_multi_file(cp_word_error_rate, ref, hyp)
    >>> er
    CombinedErrorRate(error_rate=0.6666666666666666, errors=6, length=9, insertions=3, deletions=3, substitutions=0, details=...)
    >>> pprint(er.details)
    {0: CPErrorRate(error_rate=0.5, errors=3, length=6, insertions=0, deletions=3, substitutions=0, missed_speaker=1, falarm_speaker=0, scored_speaker=2, assignment=((0, 0), (1, None))),
     1: CPErrorRate(error_rate=1.0, errors=3, length=3, insertions=3, deletions=0, substitutions=0, missed_speaker=0, falarm_speaker=1, scored_speaker=1, assignment=((0, 1), (None, 0)))}
    """
    import logging
    # TODO: support having no session keys. Support python structures with three levels (session, speaker, segment)
    reference = convert_to_tidy(reference, required_keys=('session_id',)).groupby('session_id')
    hypothesis = convert_to_tidy(hypothesis, required_keys=('session_id',)).groupby('session_id')

    # Check session keys. Print a warning if they differ and raise an exception when they differ too much
    if reference.keys() != hypothesis.keys():
        h_minus_r = list(set(hypothesis.keys()) - set(reference.keys()))
        r_minus_h = list(set(reference.keys()) - set(hypothesis.keys()))

        ratio = len(r_minus_h) / len(reference.keys())

        if h_minus_r:
            # This is a warning, because missing in reference is not a problem,
            # we can safely ignore it. Missing in hypothesis is a problem,
            # because we cannot distinguish between silence and missing.
            logging.warning(
                f'Keys of reference and hypothesis differ\n'
                f'hypothesis - reference: e.g. {h_minus_r[:5]} (Total: {len(h_minus_r)} of {len(reference)})\n'
                f'Drop them.',
            )
            hypothesis = {
                k: v
                for k, v in hypothesis.items()
                if k not in h_minus_r
            }

        if len(r_minus_h) == 0:
            pass
        elif ratio <= allowed_empty_examples_ratio:
            logging.warning(
                f'Missing {ratio * 100:.3} % = {len(r_minus_h)}/{len(reference.keys())} of recordings in hypothesis.\n'
                f'Please check your system, if it ignored some recordings or predicted no transcriptions for some recordings.\n'
                f'Continue with the assumption, that the system predicted silence for the missing recordings.',
            )
        else:
            raise RuntimeError(
                'Keys of reference and hypothesis differ\n'
                f'hypothesis - reference: e.g. {h_minus_r[:5]} (Total: {len(h_minus_r)} of {len(hypothesis)})\n'
                f'reference - hypothesis: e.g. {r_minus_h[:5]} (Total: {len(r_minus_h)} of {len(reference)})'
            )


    results = {}
    for session in reference.keys():
        results[session] = fn(reference[session], hypothesis[session])

    from meeteval.wer.wer.error_rate import CombinedErrorRate
    return CombinedErrorRate.from_error_rates(results)
