import io
import sys
import typing
from pathlib import Path
from typing import Dict, List, NamedTuple
from dataclasses import dataclass
from itertools import groupby

if typing.TYPE_CHECKING:
    from typing import Self
    from meeteval.io.uem import UEM


class BaseLine(NamedTuple):
    @classmethod
    def parse(cls, line: str) -> 'Self':
        raise NotImplementedError(cls)

    def serialize(self):
        raise NotImplementedError(type(self))

    def replace(self, **kwargs):
        """
        Return a new instance of the named tuple replacing specified fields with new values.

        >>> from meeteval.io.stm import STMLine
        >>> line = STMLine.parse('rec1 0 A 10 20 Hello World')
        >>> line
        STMLine(filename='rec1', channel=0, speaker_id='A', begin_time=10, end_time=20, transcript='Hello World')
        >>> line.replace(speaker_id='B')
        STMLine(filename='rec1', channel=0, speaker_id='B', begin_time=10, end_time=20, transcript='Hello World')
        """
        return self._replace(**kwargs)


@dataclass(frozen=True)
class Base:
    lines: List[BaseLine]
    line_cls = BaseLine

    @classmethod
    def _load(cls, file_descriptor) -> 'List[Self.line_cls]':
        raise NotImplementedError()

    @classmethod
    def load(cls, file: [Path, str, io.TextIOBase, tuple, list]) -> 'Self':
        files = file if isinstance(file, (tuple, list)) else [file]

        parsed_lines = []
        for f in files:
            if isinstance(f, io.TextIOBase):
                parsed_lines.extend(cls._load(f))
            elif isinstance(f, (str, Path)):
                with open(f, 'r') as fd:
                    parsed_lines.extend(cls._load(fd))
            else:
                raise TypeError(f, type(f), files)

        return cls(parsed_lines)

    def _repr_pretty_(self, p, cycle):
        name = self.__class__.__name__
        with p.group(len(name) + 1, name + '(', ')'):
            if cycle:
                p.text('...')
            elif len(self.lines):
                p.pretty(list(self.lines))

    def dump(self, file):
        with open(file, 'w') as fd:
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

    def groupby(self, key) -> Dict[str, 'Self']:
        return {
            filename: self.__class__(list(group))
            for filename, group in groupby(
                sorted(self.lines, key=key),
                key=key
            )
        }

    def grouped_by_filename(self) -> Dict[str, 'Self']:
        return self.groupby(lambda x: x.filename)

    def grouped_by_speaker_id(self):
        return self.groupby(lambda x: x.speaker_id)

    def sorted(self, key=None):
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
        return self.__class__(sorted(self.lines, key=key))

    def sorted_by_begin_time(self):
        return self.sorted(lambda x: x.begin_time)

    @staticmethod
    def _filter_by_uem_keep(line, uem: 'UEM'):
        """
        >>> from meeteval.io.uem import UEM, UEMLine
        >>> from meeteval.io.stm import STM, STMLine
        >>> uem = UEM([UEMLine('file', '1', 4, 8)])
        >>> Base._filter_by_uem_keep(STMLine('file', '1', 'A', 1, 3, ''), uem)
        False
        >>> Base._filter_by_uem_keep(STMLine('file', '1', 'A', 9, 10, ''), uem)
        False
        >>> Base._filter_by_uem_keep(STMLine('file', '1', 'A', 5, 7, ''), uem)
        True
        >>> Base._filter_by_uem_keep(STMLine('file', '1', 'A', 3, 5, ''), uem)
        True
        >>> Base._filter_by_uem_keep(STMLine('file', '1', 'A', 7, 9, ''), uem)
        True
        >>> Base._filter_by_uem_keep(STMLine('file', '1', 'A', 3, 4, ''), uem)
        False
        >>> Base._filter_by_uem_keep(STMLine('file', '1', 'A', 4, 5, ''), uem)
        True
        >>> Base._filter_by_uem_keep(STMLine('file', '1', 'A', 7, 8, ''), uem)
        True
        >>> Base._filter_by_uem_keep(STMLine('file', '1', 'A', 8, 9, ''), uem)
        False
        """
        try:
            entry: 'Self.line_cls' = uem[line.filename]
            begin_time = entry.begin_time
            end_time = entry.end_time
        except KeyError:
            # UEM is not specified for every file, missing means keep.
            return True
        else:
            # Partial overlap: Keep
            if line.begin_time < end_time and begin_time < line.end_time:
                return True
            return False

    def filter_by_uem(self, uem: 'UEM', verbose=False):
        new = self.__class__([
            line
            for line in self.lines
            if self._filter_by_uem_keep(line, uem)
        ])
        if verbose:
            print(f'Applied uem and reduced STM from {len(self)} to {len(new)} lines.', file=sys.stderr)
        return new
