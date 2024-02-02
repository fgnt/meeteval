import abc
import io
import os
import sys
import typing
import operator
from pathlib import Path
import contextlib
import dataclasses
from dataclasses import dataclass
from itertools import groupby
import decimal

if typing.TYPE_CHECKING:
    from typing import Self
    from meeteval.io.seglst import SegLstSegment, SegLST
    from meeteval.io.uem import UEM, UEMLine
    from meeteval.io.stm import STM, STMLine
    from meeteval.io.ctm import CTM, CTMLine
    from meeteval.io.rttm import RTTM, RTTMLine

    LineSubclasses = 'UEMLine | STMLine | CTMLine | RTTMLine'
    Subclasses = 'UEM | STM | CTM | RTTM'



class BaseABC:
    @classmethod
    def new(cls, d, **defaults):
        # Example code:
        # from meeteval.io.seglst import asseglst
        # seglst = asseglst(d).map(lambda s: {**defaults, **s})
        # ... (convert seglst to cls)
        raise NotImplementedError(cls)

    def to_seglst(self):
        raise NotImplementedError()


@dataclass(frozen=True)
class BaseLine:
    @classmethod
    def parse(cls, line: str) -> 'Self':
        raise NotImplementedError(cls)

    @classmethod
    def from_dict(cls, segment: 'SegLstSegment') -> 'Self':
        raise NotImplementedError(cls)

    def to_seglst_segment(self) -> 'SegLstSegment':
        raise NotImplementedError(self)

    def serialize(self):
        raise NotImplementedError(type(self))

    def has_intersection(self, other: 'LineSubclasses'):
        """
        Returns True, when lines have an intersection, otherwise False.

        >>> from meeteval.io.stm import STMLine
        >>> from meeteval.io.rttm import RTTMLine
        >>> from meeteval.io.uem import UEMLine
        >>> from meeteval.io.ctm import CTMLine

        # Test different line types
        >>> s = STMLine('file', 1, 'A', 10, 12, 'words')
        >>> r = RTTMLine(filename='file', speaker_id='A', begin_time=10, duration=2)
        >>> u = UEMLine('file', 1, 10, 12)
        >>> c = CTMLine('file', 1, 10, 2, 'word')
        >>> s.has_intersection(u)
        True
        >>> u.has_intersection(s)
        True
        >>> u.has_intersection(r)
        True
        >>> u.has_intersection(u)
        True
        >>> u.has_intersection(c)
        True

        # Test all corner cases
        >>> u = UEMLine('file', '1', 4, 8)
        >>> u.has_intersection(UEMLine('file', '1', 1, 3))
        False
        >>> u.has_intersection(UEMLine('file', '1', 9, 10))
        False
        >>> u.has_intersection(UEMLine('file', '1', 5, 7))
        True
        >>> u.has_intersection(UEMLine('file', '1', 3, 5))
        True
        >>> u.has_intersection(UEMLine('file', '1', 7, 9))
        True
        >>> u.has_intersection(UEMLine('file', '1', 3, 4))
        False
        >>> u.has_intersection(UEMLine('file', '1', 4, 5))
        True
        >>> u.has_intersection(UEMLine('file', '1', 7, 8))
        True
        >>> u.has_intersection(UEMLine('file', '1', 8, 9))
        False
        """
        if not isinstance(other, BaseLine):
            raise NotImplementedError(self, other)
        if 'UEMLine' not in [self.__class__.__name__, other.__class__.__name__]:
            # Usually UEM is used together with the other types, i.e.
            # define the scoring region. Disable other comparisons for now,
            # until we find a usecase.
            raise NotImplementedError(self, other)

        assert self.filename == other.filename, (self, other)

        def get_begin_end_time(line):
            begin_time = line.begin_time
            try:
                end_time = line.end_time
            except AttributeError:
                end_time = begin_time + line.duration
            return begin_time, end_time

        self_begin_time, self_end_time = get_begin_end_time(self)
        other_begin_time, other_end_time = get_begin_end_time(other)

        return self_begin_time < other_end_time and other_begin_time < self_end_time

    def replace(self, **kwargs) -> 'Self':
        """
        Return a new object replacing specified fields with new values.

        >>> from meeteval.io.stm import STMLine
        >>> line = STMLine.parse('rec1 0 A 10 20 Hello World', parse_float=float)
        >>> line
        STMLine(filename='rec1', channel=0, speaker_id='A', begin_time=10, end_time=20, transcript='Hello World')
        >>> line.replace(speaker_id='B')
        STMLine(filename='rec1', channel=0, speaker_id='B', begin_time=10, end_time=20, transcript='Hello World')
        """
        return dataclasses.replace(self, **kwargs)


class Base(BaseABC):
    lines: 'list[LineSubclasses]'
    line_cls: 'LineSubclasses'

    def __init__(self, data):
        self.lines = data

    @classmethod
    def load(cls, file: [Path, str, io.TextIOBase, tuple, list], parse_float=decimal.Decimal) -> 'Self':
        files = file if isinstance(file, (tuple, list)) else [file]

        parsed_lines = []
        for f in files:
            with _open(f, 'r') as fd:
                parsed_lines.extend(cls.parse(fd.read(), parse_float=parse_float))

        return cls(parsed_lines)

    @classmethod
    def parse(cls, s: str, parse_float=decimal.Decimal) -> 'Self':
        # Many of the supported file-formats have different conventions for comments.
        # Below is an example for a file that doesn't have comments.
        # return cls([cls.line_cls.parse(line) for line in s.splitlines() if line.strip()])
        raise NotImplementedError

    @classmethod
    def merge(cls, *o) -> 'Self':
        assert all([type(o_) == cls for o_ in o]), o
        return cls([line for o_ in o for line in o_.lines])

    def _repr_pretty_(self, p, cycle):
        name = self.__class__.__name__
        with p.group(len(name) + 1, name + '(', ')'):
            if cycle:
                p.text('...')
            elif len(self.lines):
                p.pretty(list(self.lines))

    def dump(self, file):
        with _open(file, 'w') as fd:
            for line in self.lines:
                fd.write(line.serialize() + '\n')

    def dumps(self):
        return ''.join([line.serialize() + '\n' for line in self.lines])

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.lines[item]
        elif isinstance(item, slice):
            return self.__class__(self.lines[item])
        else:
            raise NotImplementedError(type(item), item)

    def __iter__(self):
        return iter(self.lines)

    def __len__(self):
        return len(self.lines)

    def __add__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.__class__(self.lines + other.lines)

    def groupby(self, key) -> 'dict[str, Self]':
        """
        >>> from meeteval.io.stm import STM, STMLine
        >>> stm = STM([STMLine.parse('rec1 0 A 10 20 Hello World', parse_float=float)])
        >>> stm.groupby(['filename', 'begin_time'])
        {('rec1', 10): STM(lines=[STMLine(filename='rec1', channel=0, speaker_id='A', begin_time=10, end_time=20, transcript='Hello World')])}
        >>> stm.groupby('filename')
        {'rec1': STM(lines=[STMLine(filename='rec1', channel=0, speaker_id='A', begin_time=10, end_time=20, transcript='Hello World')])}
        """
        if isinstance(key, str):
            key = operator.attrgetter(key)
        elif isinstance(key, (tuple, list)):
            key = operator.attrgetter(*key)

        return _Dict({
            filename: self.__class__(list(group))
            for filename, group in groupby(
                sorted(self.lines, key=key),
                key=key
            )
        })

    def grouped_by_filename(self) -> 'dict[str, Self]':
        return self.groupby(lambda x: x.filename)

    def grouped_by_speaker_id(self):
        return self.groupby(lambda x: x.speaker_id)

    def sorted(self, key: 'str | callable | tuple | list' = None, reverse=False):
        """
        This wrapper of sorted.

        Often a sort based on filename and begin_time is useful.
        Reason: filename matches between ref and hyp and begin_time
                should be similar. Using no key, sorts by
                filename, channel, speaker_id, begin_time, ...
                so speaker_id is used before begin_time.
                Since the true speaker_id is not known in hyp,
                it is likely that ref and hyp are hard to compare
                with simple tools (e.g. check first lines of ref and hyp.)

        Here, an example to get a reasonable sort:
            new = rttm.sorted(key=lambda x: (x.filename, x.begin_time))

        """
        if isinstance(key, str):
            attribute = key
            key = lambda x: getattr(x, attribute)
        elif isinstance(key, (tuple, list)):
            attributes = key
            key = lambda x: tuple([getattr(x, a) for a in attributes])

        return self.__class__(sorted(self.lines, key=key, reverse=reverse))

    def sorted_by_begin_time(self, reverse=False):
        return self.sorted(lambda x: x.begin_time, reverse=reverse)

    def filter(self, filter_fn):
        return self.__class__(list(filter(filter_fn, self.lines)))

    def filter_by_uem(self: 'Subclasses', uem: 'UEM', verbose=False):
        """
        Remove segments that are outside of the region that is specified by the
        uem.

        Speciall cases:
         - Partial inside: Keep
         - Missing filename in uem: Keep

        >>> from pprint import pprint
        >>> from meeteval.io.uem import UEM, UEMLine
        >>> from meeteval.io.stm import STM, STMLine
        >>> uem = UEM([UEMLine('file', 1, 10, 20)])
        >>> stm = STM([
        ...     STMLine('file', 1, 'A', 2, 6, 'words'),  # dropped
        ...     STMLine('file', 1, 'A', 8, 12, 'words'),
        ...     STMLine('file', 1, 'B', 14, 16, 'words'),
        ...     STMLine('file', 1, 'A', 18, 22, 'words'),
        ...     STMLine('file', 1, 'A', 24, 28, 'words'),  # dropped
        ...     STMLine('file2', 1, 'A', 24, 28, 'words'),
        ... ])
        >>> for line in stm.filter_by_uem(uem).lines: print(line)
        STMLine(filename='file', channel=1, speaker_id='A', begin_time=8, end_time=12, transcript='words')
        STMLine(filename='file', channel=1, speaker_id='B', begin_time=14, end_time=16, transcript='words')
        STMLine(filename='file', channel=1, speaker_id='A', begin_time=18, end_time=22, transcript='words')
        STMLine(filename='file2', channel=1, speaker_id='A', begin_time=24, end_time=28, transcript='words')
        """
        uem = {line.filename: line for line in uem}

        new = self.__class__([
            line
            for line in self.lines
            if line.filename not in uem or uem[line.filename].has_intersection(line)
            # if self._filter_by_uem_keep(line, uem)
        ])
        if verbose:
            print(f'Applied uem and reduced STM from {len(self)} to {len(new)} lines.', file=sys.stderr)
        return new

    def cut_by_uem(self: 'Subclasses', uem: 'UEM', verbose=False):
        """
        Remove segments that are outside of the region that is specified by the
        uem and shift the segments.

        Speciall cases:
         - Partial inside: Keep, but shorten interval to valid area.
         - Missing filename in uem: Keep

        >>> from pprint import pprint
        >>> from meeteval.io.uem import UEM, UEMLine
        >>> from meeteval.io.stm import STM, STMLine
        >>> uem = UEM([UEMLine('file', 1, 10, 20)])
        >>> stm = STM([
        ...     STMLine('file', 1, 'A', 2, 6, 'words'),  # dropped
        ...     STMLine('file', 1, 'A', 8, 12, 'words'),
        ...     STMLine('file', 1, 'B', 14, 16, 'words'),
        ...     STMLine('file', 1, 'A', 18, 22, 'words'),
        ...     STMLine('file', 1, 'A', 24, 28, 'words'),  # dropped
        ...     STMLine('file2', 1, 'A', 24, 28, 'words'),
        ... ])
        >>> for line in stm.cut_by_uem(uem).lines: print(line)
        STMLine(filename='file', channel=1, speaker_id='A', begin_time=0, end_time=2, transcript='words')
        STMLine(filename='file', channel=1, speaker_id='B', begin_time=4, end_time=6, transcript='words')
        STMLine(filename='file', channel=1, speaker_id='A', begin_time=8, end_time=10, transcript='words')
        STMLine(filename='file2', channel=1, speaker_id='A', begin_time=24, end_time=28, transcript='words')
        """
        uem = {line.filename: line for line in uem}
        from meeteval.io.uem import UEMLine
        new = []
        for line in self.lines:
            if line.filename in uem:
                u: UEMLine = uem[line.filename]
                if u.has_intersection(line):
                    new.append(line.replace(
                        begin_time=max(line.begin_time - u.begin_time,
                                       0),
                        end_time=min(line.end_time - u.begin_time,
                                     u.end_time - u.begin_time),
                    ))
            else:
                new.append(line)

        if verbose:
            print(f'Applied uem and reduced STM from {len(self)} to {len(new)} lines.', file=sys.stderr)
        return self.__class__(new)

    def filenames(self):
        return {x.filename for x in self.lines}

    def to_seglst(self) -> 'SegLST':
        from meeteval.io.seglst import SegLST
        return SegLST([l.to_seglst_segment() for l in self.lines])

    @classmethod
    def new(cls, s, **defaults) -> 'Self':
        from meeteval.io.seglst import asseglst
        return cls([cls.line_cls.from_dict({**defaults, **segment}) for segment in asseglst(s)])


def _open(f, mode='r'):
    if isinstance(f, io.TextIOBase):
        return contextlib.nullcontext(f)
    elif isinstance(f, str) and str(f).startswith('http'):
        import urllib.request, urllib.error
        try:
            resource = urllib.request.urlopen(str(f))
        except urllib.error.URLError as e:
            raise FileNotFoundError(f) from e
        # https://stackoverflow.com/a/19156107/5766934
        return contextlib.nullcontext(io.TextIOWrapper(
            resource, resource.headers.get_content_charset()))
    elif isinstance(f, (str, os.PathLike)):
        return open(f, mode)
    else:
        raise TypeError(type(f), f)


class _VerboseKeyError(KeyError):
    # origin: paderbox.utils.mapping.DispatchError
    def __str__(self):
        if len(self.args) == 2 and isinstance(self.args[0], str):
            item, keys = self.args
            if not keys:
                return f'Invalid option {item!r}.\n' \
                       f'Mapping is empty.'

            import difflib
            # Suggestions are sorted by their similarity.
            try:
                suggestions = difflib.get_close_matches(
                    item, keys, cutoff=0, n=100
                )
            except TypeError:
                keys = map(str, keys)
                suggestions = difflib.get_close_matches(
                    item, keys, cutoff=0, n=100
                )
            return f'Invalid option {item!r}.\n' \
                   f'Close matches: {suggestions!r}.'
        else:
            return super().__str__()


class _Dict(dict):
    """
    Is basically a dict with a better error message on key error.

    origin: paderbox.utils.mapping.Dispatcher

    >>> from meeteval.io.base import _Dict
    >>> d = _Dict(abc=1, bcd=2)
    >>> d['acd']  #doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    meeteval.io.base._VerboseKeyError: Invalid option 'acd'.
    Close matches: ['bcd', 'abc'].
    """
    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError as e:
            raise _VerboseKeyError(item, self.keys()) from None
