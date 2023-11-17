"""
A "tidy" or "wide" representation of transcripts.

This format can be converted from and to any other format (e.g., STM, CTM(Group), CSV, Chime7 JSON, plain Python structures, ...).

The name is inspired by R's definition of tidy data:
 - https://cran.r-project.org/web/packages/tidyr/vignettes/tidy-data.html#:~:text=Tidy%20data%20is%20a%20standard,with%20observations%2C%20variables%20and%20types.
 - https://vita.had.co.nz/papers/tidy-data.pdf


TODO: Define a class that wraps a list of segments or keep Python types?
"""
import functools
from typing import Union, Dict, Any, List, TypedDict, Tuple, Callable

from meeteval.io import STM, CTM
from meeteval.io.ctm import CTMGroup
from meeteval.wer.utils import _map, _items, _values, _keys


class keys:
    """
    Defines all keys that are used somewhere in MeetEval.

    Note: They correspond to the format used for Chime7.
    """
    SESSION = 'session_id'
    START_TIME = 'start_time'
    END_TIME = 'end_time'
    WORDS = 'words'
    SPEAKER = 'speaker'
    SEGMENT = 'segment_index'   # For MIMO / ORC WER

    # Unused but by MeetEval but present in some file formats. They are defined here for compatibility and
    # conversion in both directions
    CHANNEL = 'channel'
    CONFIDENCE = 'confidence'


def _convert_segment(d):
    """
    Converts a segment `d` into tidy data format. The segment can be anything convertible to the tidy format.
    """
    if isinstance(d, str):
        return {keys.WORDS: d}
    if isinstance(d, dict) and keys.WORDS in d:
        return d
    if isinstance(d, list):
        assert all(isinstance(i, str) for i in d), d
        return _convert_segment(' '.join(d))
    raise ValueError(f'{d} is not convertible to tidy segment format.')


def convert_to_tidy_and_back(d, *, groups=(keys.SPEAKER,)) -> 'Tuple[Tidy, Callable[[Tidy], Any]]':
    """
    Converts a structure `d` into tidy data format. The structure can be anything convertible to the tidy format.

    TODO: name

    >>> def convert_cycle(d, groups=(keys.SPEAKER,)): tidy, conv = convert_to_tidy_and_back(d, groups=groups); r = conv(tidy); assert r == d, (r, d); return r
    >>> convert_cycle('a b c')
    'a b c'
    >>> convert_cycle(['a b c', 'd e f'])
    ['a b c', 'd e f']
    >>> convert_cycle({'A': 'a b c', 'B': 'd e f'})
    {'A': 'a b c', 'B': 'd e f'}
    >>> convert_cycle(['a b c', 'd e f'], groups=(keys.SESSION,))
    ['a b c', 'd e f']
    >>> convert_cycle({'ex1': {'A': 'a b c', }, 'ex2': {'C': 'd e f'}}, groups=(keys.SESSION, keys.SPEAKER))
    {'ex1': {'A': 'a b c'}, 'ex2': {'C': 'd e f'}}
    >>> from meeteval.io.stm import STM, STMLine
    >>> convert_cycle(STM([STMLine('ex', 1, 'A', 0, 1, 'a b c')]))
    STM(lines=[STMLine(filename='ex', channel=1, speaker_id='A', begin_time=0, end_time=1, transcript='a b c')])

    Modifying the tidy representation
    >>> tidy, conv = convert_to_tidy_and_back(['a b c', 'd e f'])
    >>> tidy[0][keys.WORDS] = 'x y z'
    >>> conv(tidy)
    ['x y z', 'd e f']

    >>> tidy, conv = convert_to_tidy_and_back({'A': 'a b c', 'B': 'd e f'})
    >>> tidy[0][keys.SPEAKER] = 'C'
    >>> conv(tidy)
    {'C': 'a b c', 'B': 'd e f'}
    """
    assert isinstance(groups, tuple) or groups is None, groups

    # Already wide format
    if isinstance(d, list) and len(d) > 0 and isinstance(d[0], dict):
        return d, lambda x: x

    # STM, CTM, CTMGroup, custom formats
    # Support any format that is convertible to segments, including user classes that define `segments`.
    # TODO: rename segments -> tidy?
    if hasattr(d, 'segments'):
        return d.segments(), d.__class__.convert

    # A single wide format dict / segment (TODO: can this be confusing?)
    if isinstance(d, dict) and keys.WORDS in d:
        def convert(tidy):
            assert len(tidy) == 1, tidy
            return tidy[0]
        return [d], convert

    # A single string
    if isinstance(d, str):
        def convert(tidy):
            # After modification, it can happen that the tidy representation contains multiple segments
            # assert len(tidy) == 1, tidy
            return ' '.join(s[keys.WORDS] for s in tidy)
        return [_convert_segment(d)], convert

    # A list or dict of things convertible to wide, grouped by speaker / "groups" key
    if isinstance(d, (list, dict)):
        if len(d) == 0:
            return [], lambda x: type(d)(x)
        if groups is None:
            raise ValueError(f'{d} cannot be converted with no group keys given!')

        # Convert individual elements and flatten anything in there
        converted = {}
        conversion_functions = {}
        for k in _keys(d):
            converted[k], conversion_functions[k] = convert_to_tidy_and_back(d[k], groups=groups[1:])

        if groups:
            # TODO: Assert that all of them are equal (how?)
            conversion = conversion_functions[next(iter(conversion_functions.keys()))]
            group = groups[0]
            def convert(tidy):
                tidy = groupby(tidy, group)
                if isinstance(d, dict):
                    return {
                        k: conversion(v)
                        for k, v in tidy.items()
                    }
                else:
                    return [conversion(v) for _, v in sorted(tidy.items())]
        else:
            def convert(tidy):
                raise ValueError("Conversion not possible with groups = None")


        # Set or check (when already set) the speaker / group key. Make sure that
        #  - The group keys are unique
        #  - All items in a groups have the same value for the group key
        if groups:
            group = groups[0]
            seen_group_values = set()
            for i, items in _items(converted):
                speaker = None
                for v in items:
                    if group not in v:
                        v[group] = i

                    if speaker is None:
                        speaker = v[group]
                    else:
                        assert v[group] == speaker, ('Groups may only contain a single speaker', v, speaker)
                if speaker in seen_group_values:
                    raise ValueError('Overlapping groups!')
                seen_group_values.add(speaker)
        return [v for l in _values(converted) for v in l], convert

    raise TypeError(d)

def convert_to_tidy(d, *, groups=(keys.SPEAKER,)):
    """
        Converting Python structures for cpWER, ORC WER and MIMO WER
    >>> convert_to_tidy('a b c')
    [{'words': 'a b c'}]
    >>> convert_to_tidy(['a b c', 'd e f']) # Multiple items default to speakers
    [{'words': 'a b c', 'speaker': 0}, {'words': 'd e f', 'speaker': 1}]
    >>> convert_to_tidy({'A': 'a b c', 'B': 'd e f'})
    [{'words': 'a b c', 'speaker': 'A'}, {'words': 'd e f', 'speaker': 'B'}]
    >>> convert_to_tidy(['a b c', 'd e f'], groups=(keys.SESSION,)) # But can also be any other keys
    [{'words': 'a b c', 'session_id': 0}, {'words': 'd e f', 'session_id': 1}]
    >>> convert_to_tidy({'ex1': {'A': 'a b c', }, 'ex2': {'C': 'd e f'}}, groups=(keys.SESSION, keys.SPEAKER)) # Or multiple levels
    [{'words': 'a b c', 'speaker': 'A', 'session_id': 'ex1'}, {'words': 'd e f', 'speaker': 'C', 'session_id': 'ex2'}]

    Nested structures beyond the specified groups keys are flattened (TODO: does this make sense or is this confusing? This is currently broken. I think this should be specified through the groups parameter!)
    >>> convert_to_tidy({'A': ['abc', 'def']})
    [{'words': 'abc', 'speaker': 'A'}, {'words': 'def', 'speaker': 'A'}]
    >>> convert_to_tidy({'A': [{'x': 'abc'}, {'y': 'def'}]})
    [{'x': 'abc', 'speaker': 'A'}, {'y': 'def', 'speaker': 'A'}]

    Data formats are also converted
    >>> from meeteval.io.stm import STM, STMLine
    >>> convert_to_tidy(STM([STMLine('ex', 1, 'A', 0, 1, 'a b c')]))
    [{'start_time': 0, 'end_time': 1, 'words': 'a b c', 'speaker': 'A', 'session_id': 'ex'}]
    >>> from meeteval.io.ctm import CTM, CTMLine, CTMGroup
    >>> convert_to_tidy(CTMGroup({'A': CTM([CTMLine('ex', 1, 0, 1, 'a')])}))

    >>> convert_to_tidy(['a b', 'c d'])
    [{'words': 'a b', 'speaker': 0}, {'words': 'c d', 'speaker': 1}]
    >>> convert_to_tidy(['a b', 'c d', 'e f'])
    [{'words': 'a b', 'speaker': 0}, {'words': 'c d', 'speaker': 1}, {'words': 'e f', 'speaker': 2}]
    """
    tidy, _ = convert_to_tidy_and_back(d, groups=groups)
    return tidy

# Helper functions for working with tidy data
def groupby(
        iterable,
        key = None,
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
    import operator
    import collections
    import itertools
    if callable(key) or key is None:
        key_fn = key
    elif isinstance(key, (str, int)):
        key_fn = operator.itemgetter(key)
    elif not isinstance(key, collections.Mapping):
        value_getter = operator.itemgetter(0)
        groups = collections.defaultdict(list)
        for key, group in itertools.groupby(zip(iterable, key, strict=True), operator.itemgetter(1)):
            groups[key].extend(map(value_getter, group))
        return dict(groups)
    else:
        raise TypeError(f'Invalid type for key: {type(key)}')

    groups = collections.defaultdict(list)
    for key, group in itertools.groupby(iterable, key_fn):
        groups[key].extend(group)
    return dict(groups)

# Type annotation for anything that is convertible to the tidy format
# TODO: complete this list
TidyData = Union[str, Dict[str, Any], List[Dict[str, Any]], STM, CTM, CTMGroup]

TidySegment = TypedDict('TidySegment', {
    keys.SESSION: str,
    keys.START_TIME: float,
    keys.END_TIME: float,
    keys.WORDS: str,
    keys.SPEAKER: str,
    keys.CHANNEL: int,
    keys.SEGMENT: int,
    keys.CONFIDENCE: float,
}, total=False)

Tidy = List[TidySegment]


def tidy_args(num_args=None, *, groups=()):
    """
    Decorator for function that require tidy data input to automatically convert other input formats to tidy data.
    """
    from functools import wraps
    from inspect import signature

    def _tidy_args_wrapper(fn):
        s = signature(fn)
        @wraps(fn)
        def _wrapped_tidy(*args, **kwargs):
            bound_args = s.bind(*args, **kwargs)
            args = list(bound_args.args)
            for i in range(len(args) if num_args is None else num_args):
                args[i] = convert_to_tidy(args[i], groups=groups)
            return fn(*args, **bound_args.kwargs)
        return _wrapped_tidy
    return _tidy_args_wrapper


def tidy_map(fn):
    @functools.wraps(fn)
    def _tidy_map(arg, *args, **kwargs):
        arg, conv = convert_to_tidy_and_back(arg)
        arg = fn(arg, *args, **kwargs)
        return conv(arg)
    return _tidy_map

def extract_key(d, key, ignore_missing=False):
    return [s[key] for s in d if s[key] or not ignore_missing]