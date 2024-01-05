import dataclasses
import functools
import typing

from meeteval.io.base import BaseABC
from meeteval.io.py import NestedStructure
from meeteval._typing import TypedDict

if typing.TYPE_CHECKING:
    from meeteval.wer.wer.error_rate import ErrorRate
    from typing import Callable, Iterable, Any



class SegLstSegment(TypedDict, total=False):
    """
    A segment.

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
class SegLST(BaseABC):
    """
    A collection of segments in SegLST format. This the input type to most
    functions in MeetEval that process transcript segments.
    """
    segments: 'list[SegLstSegment]'

    # Caches
    _unique = None

    @property
    class T:
        """
        The "transpose" of the segments, i.e., a mapping that maps keys to lists of values.

        The name `T` is inspired by the `T` in `pandas.DataFrame.T` and `numpy.ndarray.T`.
        """
        def __init__(self, outer):
            self._outer = outer

        def keys(self):
            """
            The keys that are common among all segments
            """
            return set.intersection(*[set(s.keys()) for s in self._outer.segments])

        def __getitem__(self, key):
            """
            Returns the values for `key` of all segments as a list.
            """
            return [s[key] for s in self._outer.segments]

    def unique(self, key) -> 'set[Any]':
        """
        Returns the unique values for `key` among all segments.
        """
        return set([s[key] for s in self.segments])

    def __iter__(self):
        return iter(self.segments)

    def __getitem__(self, item):
        return self.segments[item]

    def __len__(self):
        return len(self.segments)

    def __add__(self, other):
        if isinstance(other, SegLST):
            return SegLST(self.segments + other.segments)
        return NotImplemented

    def groupby(self, key) -> 'dict[Any, SegLST]':
        """
        >>> t = asseglst(['a b c', 'd e f', 'g h i'])
        >>> t.segments
        [{'words': 'a b c', 'segment_index': 0, 'speaker': 0}, {'words': 'd e f', 'segment_index': 0, 'speaker': 1}, {'words': 'g h i', 'segment_index': 0, 'speaker': 2}]

        >>> from pprint import pprint
        >>> pprint(t.groupby('speaker')) # doctest: +ELLIPSIS
        {0: SegLST(segments=[{'words': 'a b c', 'segment_index': 0, 'speaker': 0}]),
         1: SegLST(segments=[{'words': 'd e f', 'segment_index': 0, 'speaker': 1}]),
         2: SegLST(segments=[{'words': 'g h i', 'segment_index': 0, 'speaker': 2}])}
        """
        return {k: SegLST(g) for k, g in groupby(self.segments, key=key).items()}

    def sorted(self, key) -> 'SegLST':
        """
        Returns a copy of this object with the segments sorted by `key`.
        """
        return SegLST(sorted(self.segments, key=_get_key(key)))

    def map(self, fn: 'Callable[[SegLstSegment], SegLstSegment]') -> 'SegLST':
        """
        Applies `fn` to all segments and returns a new `SegLST` object with the results.
        """
        return SegLST([fn(s) for s in self.segments])

    def flatmap(self, fn: 'Callable[[list[SegLstSegment]], Iterable[SegLstSegment]]') -> 'SegLST':
        """
        Returns a new `SegLST` by applying `fn`, which is exptected to return an iterable of `SegLstSegment`s,
        to all segments and flattening the output.

        The name is inspired by other programming languages (e.g., JavaScript, Rust, Java, Scala) where
        flatmap is a common operation on lists / arrays / iterators. In data loading frameworks,
        this operation is known as map followed by unbatch.

        Example: Split utterances into words
            >>> SegLST([{'words': 'a b c'}]).flatmap(lambda x: [{'words': w} for w in x['words'].split()])
            SegLST(segments=[{'words': 'a'}, {'words': 'b'}, {'words': 'c'}])
        """
        return SegLST([s for t in self.segments for s in fn(t)])

    def filter(self, fn: 'Callable[[SegLstSegment], bool]') -> 'SegLST':
        """
        Applies `fn` to all segments and returns a new `SegLST` object with the segments for which `fn` returns true.
        """
        return SegLST([s for s in self.segments if fn(s)])

    @classmethod
    def merge(cls, *t) -> 'SegLST':
        """
        Merges multiple `SegLST` objects into one by concatenating all segments.
        """
        return SegLST([s for t_ in t for s in t_.segments])

    def to_seglst(self) -> 'SegLST':
        return self

    @classmethod
    def new(cls, d, **defaults) -> 'SegLST':
        d = asseglst(d)
        if defaults:
            d = d.map(lambda s: {**defaults, **s})
        return d

    def _repr_pretty_(self, p, cycle):
        """
        >>> from IPython.lib.pretty import pprint
        >>> pprint(SegLST([{'words': 'a b c', 'segment_index': 0, 'speaker': 0}]))
        SegLST([{'words': 'a b c', 'segment_index': 0, 'speaker': 0}])
        >>> pprint(SegLST([{'words': 'a b c', 'segment_index': 0, 'speaker': 0}, {'words': 'd e f', 'segment_index': 0, 'speaker': 1}, {'words': 'g h i', 'segment_index': 0, 'speaker': 2}]))
        SegLST([{'words': 'a b c', 'segment_index': 0, 'speaker': 0},
                {'words': 'd e f', 'segment_index': 0, 'speaker': 1},
                {'words': 'g h i', 'segment_index': 0, 'speaker': 2}])
        """
        name = self.__class__.__name__
        with p.group(len(name) + 1, name + '(', ')'):
            if cycle:
                p.text('...')
            else:
                p.pretty(list(self.segments))


def asseglistconvertible(d, *, py_convert=NestedStructure):
    """
    Converts `d` into a structure that is convertible to the SegLST format, i.e., that
    has `to_seglst` (and often `new`) defined.
    """
    # Already convertible
    if hasattr(d, 'to_seglst'):
        return d

    # Chime7 format / List of `SegLstSegment`s
    if isinstance(d, list) and (len(d) == 0 or isinstance(d[0], dict) and 'words' in d[0]):
        # TODO: Conversion back to list of segments (Python structure)?
        return SegLST(d)

    # TODO: pandas DataFrame

    # Convert Python structures
    if isinstance(d, (list, tuple, dict, str)):
        if py_convert is None:
            raise TypeError(f'Cannot convert {type(d)} to SegLST with py_convert={py_convert!r}!')
        # TODO: Conversion back to Python structure?
        return py_convert(d)

    raise NotImplementedError(f'No conversion implemented for {type(d)}!')


def asseglst(d, *, required_keys=(), py_convert=NestedStructure) -> 'SegLST':
    """
    Converts an object `d` into SegLST data format. `d` can be anything convertible to the SegLST format.
    Returns `d` if `isinstance(d, SegLST)`.

    Python structures have to have one or two nested levels. The first level is interpreted as the speaker key and the
    second level as the segment key.
    >>> asseglst(['a b c'])
    SegLST(segments=[{'words': 'a b c', 'segment_index': 0, 'speaker': 0}])
    >>> asseglst([['a b c', 'd e f'], ['g h i']])
    SegLST(segments=[{'words': 'a b c', 'segment_index': 0, 'speaker': 0}, {'words': 'd e f', 'segment_index': 1, 'speaker': 0}, {'words': 'g h i', 'segment_index': 0, 'speaker': 1}])
    >>> asseglst({'A': ['a b c', 'd e f'], 'B': ['g h i']})
    SegLST(segments=[{'words': 'a b c', 'segment_index': 0, 'speaker': 'A'}, {'words': 'd e f', 'segment_index': 1, 'speaker': 'A'}, {'words': 'g h i', 'segment_index': 0, 'speaker': 'B'}])

    Data formats are also converted
    >>> from meeteval.io.stm import STM, STMLine
    >>> stm = STM.parse('ex 1 A 0 1 a b c')
    >>> asseglst(stm).segments
    [{'session_id': 'ex', 'channel': 1, 'speaker': 'A', 'start_time': 0, 'end_time': 1, 'words': 'a b c'}]

    The SegLST representation can be converted back to its original representation
    >>> print(stm.new(stm).dumps())
    ex 1 A 0 1 a b c
    <BLANKLINE>

    And modified before inversion
    >>> s = asseglst(stm)
    >>> s.segments[0]['words'] = 'x y z'
    >>> print(stm.new(s).dumps())
    ex 1 A 0 1 x y z
    <BLANKLINE>
    """
    assert isinstance(required_keys, tuple), required_keys

    # Exit early if already in the correct format
    if isinstance(d, SegLST):
        return d

    # Get a type that is convertible to SegLST
    d = asseglistconvertible(d, py_convert=py_convert)

    t = d.to_seglst()

    # Check that `t` has all required keys
    if len(t) and not set(required_keys).issubset(t.T.keys()):
        required_keys = set(required_keys)
        raise ValueError(
            f'Some required keys are not present in the converted data structure!\n'
            f'Required: {required_keys}, found: {t.T.keys()}, missing: {required_keys - t.T.keys()}'
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


def seglst_map(*, required_keys=(), py_convert=NestedStructure):
    """
    Decorator to for a function that takes a (single) `SegLST` object as input and returns a (single) `SegLST` object
    as output. Automatically converts the input to `SegLST` and converts the returned value back to its original type.

    >>> @seglst_map(required_keys=('speaker',))
    ... def fn(seglst, *, speaker='X'):
    ...     return seglst.map(lambda x: {**x, 'speaker': speaker})
    >>> from meeteval.io.stm import STM
    >>> fn(STM.parse('X 1 A 0 1 a b c'))
    STM(lines=[STMLine(filename='X', channel=1, speaker_id='X', begin_time=0, end_time=1, transcript='a b c')])
    >>> from meeteval.io.rttm import RTTM
    >>> fn(RTTM.parse('SPEAKER CMU_20020319-1400_d01_NONE 1 130.430000 2.350 <NA> <NA> juliet <NA> <NA>'))
    RTTM(lines=[RTTMLine(type='SPEAKER', filename='CMU_20020319-1400_d01_NONE', channel='1', begin_time=Decimal('130.430000'), duration=Decimal('2.350000'), othography='<NA>', speaker_type='<NA>', speaker_id='X', confidence='<NA>', signal_look_ahead_time='<NA>')])
    >>> fn({'A': 'abc', 'B': 'def'}).structure
    {'X': 'abc def'}
    """

    def _seglst_map(fn):
        @functools.wraps(fn)
        def _seglst_map(arg, *args, **kwargs):
            c = asseglistconvertible(arg, py_convert=py_convert)
            arg = asseglst(c, required_keys=required_keys)
            arg = fn(arg, *args, **kwargs)
            return c.new(arg)

        return _seglst_map

    return _seglst_map


def apply_multi_file(
        fn: 'Callable[[SegLST, SegLST], ErrorRate]',
        reference, hypothesis,
        *,
        allowed_empty_examples_ratio=0.1
):
    """
    Applies a function individually to all sessions / files.

    `reference` and `hypothesis` must be convertible to `SegLST`. If they are a Python structure, the first level
    is interpreted as the session / file key.

    >>> from meeteval.wer.wer.cp import cp_word_error_rate
    >>> from pprint import pprint
    >>> ref = [['a b c', 'd e f'], ['g h i']]
    >>> hyp = [['a b c'], ['d e f', 'g h i']]
    >>> er = apply_multi_file(cp_word_error_rate, ref, hyp)
    >>> pprint(er)
    {0: CPErrorRate(error_rate=0.5, errors=3, length=6, insertions=0, deletions=3, substitutions=0, missed_speaker=1, falarm_speaker=0, scored_speaker=2, assignment=((0, 0), (1, None))),
     1: CPErrorRate(error_rate=1.0, errors=3, length=3, insertions=3, deletions=0, substitutions=0, missed_speaker=0, falarm_speaker=1, scored_speaker=1, assignment=((0, 1), (None, 0)))}
    """
    import logging
    reference = asseglst(
        reference, required_keys=('session_id',),
        py_convert=lambda p: NestedStructure(p, ('session_id', 'speaker', 'segment_id'))
    ).groupby('session_id')
    hypothesis = asseglst(
        hypothesis, required_keys=('session_id',),
        py_convert=lambda p: NestedStructure(p, ('session_id', 'speaker', 'segment_id'))
    ).groupby('session_id')

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

    return results
