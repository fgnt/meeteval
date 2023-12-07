import dataclasses
import typing
from typing import Any, List, Tuple

from meeteval.io.base import BaseABC

if typing.TYPE_CHECKING:
    from typing import Self
    from meeteval.io.seglst import SegLST


def _convert_python_structure(structure, *, keys=(), final_key='words', final_types=str):
    from meeteval.io.seglst import SegLST
    from meeteval.wer.utils import _keys

    def _convert(d, index=0):
        # A single string is converted to a single segment
        # If the final key is not set, we set it to 0
        if isinstance(d, str):
            segment = {final_key: d}

            # TODO: Allow multiple levels of defaults?
            if len(keys) > index + 1:
                raise ValueError("Key mismatch")

            if index < len(keys):
                segment[keys[-1]] = 0
            return [segment], (str,)

        # A structure. `all_keys[index]` is the key for this level
        if isinstance(d, (list, dict, tuple)):
            if len(d) == 0:
                # Special case where no structure information is available
                # This case is not invertible!
                # TODO: represent this by `None` or `''` as to make it (at least partially) invertible?
                #  Problem: how to get the keys of further nesting levels?
                return [], None if len(keys) > index else (type(d),)

            # Check if we have a key for this level. If not, raise an exception
            if index >= len(keys):
                # Only raise the exception if the final key doesn't match the `final_types`
                if final_types is not None and not isinstance(d, final_types):
                    raise ValueError(
                        f'{structure} cannot be converted because it contains more nested levels than keys given! '
                        f'keys={keys!r}, '
                        f'final_key={final_key!r}, final_types={final_types!r}'
                    )
                return [{final_key: d}], (type(d),)

            key = keys[index]
            converted = {}
            types = {}

            for k in _keys(d):
                converted[k], types[k] = _convert(d[k], index=index + 1)

            # We can only convert back if all items in can be converted.
            # Return None if not invertible.
            types = {k: types[k] for k in types.keys() if types[k] is not None}

            if len(set(types.values())) == 1:
                types = (type(d),) + next(iter(types.values()))
            else:
                types = None

            # Set or check (when already set) the key. Make sure that
            #  - The group keys are unique
            #  - All items in a group have the same value for `key`
            for k, segments in converted.items():
                for s in segments:
                    s.setdefault(key, k)
                assert set(s[key] for s in segments) == {k} or len(segments) == 0, (k, segments)
            assert set(c[0][key] for c in converted.values() if c).issubset(
                set(converted.keys())), f'Group values are not unique! {converted}'
            return [v for l in converted.values() for v in l], types
        raise TypeError(d)

    segments, types = _convert(structure)
    return SegLST(segments), types


def _invert_python_structure(t: 'SegLST', types, keys):
    if len(types) != len(keys):
        if len(types) < len(keys):
            # _convert_python_structure adds a dummy key if the final key is not set. We have to remove it here
            # again.
            for k in keys[len(types) - 1:-1]:
                if len(t.unique(k)) != 1:
                    raise ValueError(
                        f'Cannot convert SegLST to Tidy with t.keys={t.keys!r}, types={types!r} and keys={keys!r}. '
                        f'Each non-unique key must have a type, otherwise the structure is not convertible.'
                    )
            keys = keys[:len(types) - 1] + (keys[-1],)
        else:
            raise ValueError(
                f'Cannot convert SegLST to Python structure with '
                f't.keys={t.keys!r}, types={types!r} and keys={keys!r}.'
            )

    if len(types) == 1:
        # After modification, it can happen that the SegLST representation contains multiple segments.
        # We concatenate here to keep some sort of "old" behavior.
        #
        # Note: This inversion gives back the original when the SegLST representation is not modified,
        # but it can give a different representation when it was modified.
        # TODO: Would it be better to keep a list? That would make the above statement true, but it would
        #  change the format, e.g., from str to List[str] and is not guaranteed to do the same at
        #  all nesting levels
        if keys[0] == 'words':
            words = [s['words'] for s in t]
            if types[0] == str:
                return ' '.join(words)
            else:
                return types[0](words)
        assert len(t) == 1, t
        return types[0](t.segments[0][keys[0]])
    groups = {k: _invert_python_structure(v, types[1:], keys[1:]) for k, v in t.groupby(keys[0]).items()}
    if types[0] in (list, tuple):
        groups = types[0](groups.values())
    return groups


@dataclasses.dataclass(frozen=True)
class NestedStructure(BaseABC):
    """
    Wraps a Python structure where the structure levels represent keys.

    Example structure for cpWER:
        ```python
        structure = {
            'Alice': ['segment1', 'segment2'],   # Speaker 1: Alice
            'Bob': ['segment1', 'segment2'],   # Speaker 2: Bob
        }
        NestedStructure(structure, ('speaker', 'segment'))
        ```
    """
    structure: Any
    level_keys: 'List[str, ...] | Tuple[str, ...]' = ('speaker', 'segment_index')
    final_key: 'str' = 'words'
    final_types: 'type | List[type] | Tuple[type]' = str

    # Cache variables
    _types = None
    _used_keys = None
    _seglst = None

    def new(self, t: 'SegLST', **defaults) -> 'Self':
        """
        This is usually a classmethod, but here, it's an instance method
        because we need `keys` and `types` for conversion.

        >>> def convert_cycle(structure, keys, mod=None):
        ...     s = NestedStructure(structure, keys)
        ...     t = s.to_seglst()
        ...     if mod:
        ...         t = mod(t)
        ...     return s.new(t).structure
        >>> convert_cycle('a b c', keys=())
        'a b c'
        >>> convert_cycle(['a b c', 'd e f'], keys=('speaker',))
        ['a b c', 'd e f']
        >>> convert_cycle({'A': 'a b c', 'B': 'd e f'}, keys=('speaker',))
        {'A': 'a b c', 'B': 'd e f'}
        >>> s = NestedStructure({'B': 'd e f', 'A': 'a b c'}, level_keys=('speaker',))
        >>> s2 = NestedStructure(['a b c', 'd e f'], level_keys=('speaker',))
        >>> s.new(s2.to_seglst()).structure
        {0: 'a b c', 1: 'd e f'}
        >>> s2.new(s.to_seglst()).structure   # Keys are sorted when converting to list (TODO: is this expected behavior?)
        ['d e f', 'a b c']

        Empty structures are only invertible if all keys can be inferred and empty nesting levels get lost (TODO)
        >>> convert_cycle([], keys=())
        []
        >>> convert_cycle([], keys=('speaker',))
        Traceback (most recent call last):
         ...
        ValueError: Cannot convert to Python structure because this structure is not invertible.
        >>> convert_cycle([[]], keys=('speaker', 'segment_index'))
        Traceback (most recent call last):
         ...
        ValueError: Cannot convert to Python structure because this structure is not invertible.
        >>> convert_cycle([['abc'], []], keys=('speaker', 'segment_index'))
        [['abc']]
        """
        if self.types is None:
            raise ValueError('Cannot convert to Python structure because this structure is not invertible.')
        from meeteval.io.seglst import asseglst
        t = asseglst(t).map(lambda x: {**defaults, **x})
        return NestedStructure(_invert_python_structure(t, self.types, self.level_keys + (self.final_key,)),
                               self.level_keys, self.final_key, self.final_types)

    @property
    def types(self):
        if self._seglst is None:
            self.to_seglst()
        return self._types

    def to_seglst(self):
        """
        Converting Python structures for cpWER, ORC WER and MIMO WER
        >>> NestedStructure('a b c', level_keys=()).to_seglst()
        SegLST(segments=[{'words': 'a b c'}])

        Structure levels are interpreted as the keys in keys.
        >>> NestedStructure('a b c', level_keys=('speaker',)).to_seglst()
        SegLST(segments=[{'words': 'a b c', 'speaker': 0}])
        >>> NestedStructure(['a b c', 'd e f'], level_keys=('speaker',)).to_seglst()
        SegLST(segments=[{'words': 'a b c', 'speaker': 0}, {'words': 'd e f', 'speaker': 1}])
        >>> NestedStructure({'A': 'a b c', 'B': 'd e f'}, level_keys=('speaker',)).to_seglst()
        SegLST(segments=[{'words': 'a b c', 'speaker': 'A'}, {'words': 'd e f', 'speaker': 'B'}])
        >>> NestedStructure({'ex1': {'A': 'a b c', }, 'ex2': {'C': 'd e f'}}, level_keys=('session_id', 'speaker')).to_seglst()
        SegLST(segments=[{'words': 'a b c', 'speaker': 'A', 'session_id': 'ex1'}, {'words': 'd e f', 'speaker': 'C', 'session_id': 'ex2'}])

        All keys in `keys` must be present, otherwise an exception is raised
        >>> NestedStructure('a b c', level_keys=('speaker', 'channel')).to_seglst()
        Traceback (most recent call last):
          ...
        ValueError: Key mismatch

        Empty structures

        Empty structures result in empty `SegLST` objects
        >>> NestedStructure([], level_keys=('speaker',)).to_seglst()
        SegLST(segments=[])
        >>> NestedStructure({}, level_keys=('speaker',)).to_seglst()
        SegLST(segments=[])
        >>> NestedStructure([{}], level_keys=('speaker',)).to_seglst()
        SegLST(segments=[])

        Empty nested structures are allowed but not represented in the `SegLST` format (TODO)
        >>> NestedStructure([['ab'], []], level_keys=('speaker', 'segment_index')).to_seglst()
        SegLST(segments=[{'words': 'ab', 'segment_index': 0, 'speaker': 0}])

        The last key is filled with a dummy key
        >>> NestedStructure(['a b c', 'd e f'], level_keys=('speaker', 'channel')).to_seglst()
        SegLST(segments=[{'words': 'a b c', 'channel': 0, 'speaker': 0}, {'words': 'd e f', 'channel': 0, 'speaker': 1}])

        Nested structures beyond the specified groups are by default not allowed. With `ensure_word_is_string=False`,
        you can have nested structures. But be careful, this can lead to unexpected results!
        >>> NestedStructure({'A': ['abc', 'def']}, level_keys=(), final_types=None).to_seglst()
        SegLST(segments=[{'words': {'A': ['abc', 'def']}}])
        >>> NestedStructure({'A': ['abc', 'def']}, level_keys=('speaker',), final_types=None).to_seglst()
        SegLST(segments=[{'words': ['abc', 'def'], 'speaker': 'A'}])
        >>> NestedStructure({'A': [{'x': 'abc'}, {'y': 'def'}]}, level_keys=(), final_types=None).to_seglst()
        SegLST(segments=[{'words': {'A': [{'x': 'abc'}, {'y': 'def'}]}}])
        """
        if self._seglst is None:
            self.__dict__['_seglst'], self.__dict__['_types'] = _convert_python_structure(
                self.structure,
                keys=self.level_keys,
                final_key=self.final_key,
                final_types=self.final_types,
            )
        return self._seglst
