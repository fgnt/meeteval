import io
from typing import TextIO, Dict, List, NamedTuple
from itertools import groupby
from pathlib import Path
from dataclasses import dataclass

try:
    from functools import cached_property
except ImportError:
    # Fallback for Python 3.7 and lower, since cached_property was added in
    # Python 3.8.
    from cached_property import cached_property  # Python 3.7


class UEMLine(NamedTuple):
    """
    un-partitioned evaluation map from file in NIST format.

    We found no definition from NIST (many people claim it is from NIST).
    The best reference we have is from dscore (uem file is older):
    https://github.com/nryant/dscore/blob/master/scorelib/uem.py

    - file id  --  file id
    - channel  --  channel (1-indexed)
    - onset  --  onset of evaluation region in seconds from beginning of file
    - offset  --  offset of evaluation region in seconds from beginning of
      file

    Since all field names from dscore don't match the typically used fienames
    from NIST and the source code of
    https://github.com/usnistgov/SCTK/blob/master/src/asclite/core/uemfilter.h
    uses the typical names, use fild names that match with STM.

    """
    filename: str
    channel: 'str | int'= '<NA>'
    begin_time: 'float | int | str' = 0
    end_time: 'float | int | str' = 0

    @classmethod
    def parse(cls, line: str) -> 'UEMLine':
        """
        >>> UEMLine.parse('S01 1 60.001 79.003')
        UEMLine(filename='S01', channel='1', begin_time=60.001, end_time=79.003)
        """
        filename, channel, begin_time, end_time = line.split()

        return UEMLine(
            filename=filename,
            channel=int(channel) if begin_time.isdigit() else channel,
            begin_time=int(begin_time) if begin_time.isdigit() else float(begin_time),  # Keep type, int or float,
            end_time=int(end_time) if end_time.isdigit() else float(end_time),  # Keep type, int or float,
        )

    def serialize(self):
        """
        >>> line = UEMLine.parse('S01 1 60.001 79.003')
        >>> line.serialize()
        'S01 1 60.001 79.003'
        """
        return (f'{self.filename} {self.channel} '
                f'{self.begin_time} {self.end_time}')

    def replace(self, **kwargs):
        """
        Return a new instance of the named tuple replacing specified fields with new values.

        >>> line = UEMLine.parse('S01 1 60.001 79.003')
        >>> line
        UEMLine(filename='S01', channel='1', begin_time=60.001, end_time=79.003)
        >>> line.replace(begin_time=0)
        UEMLine(filename='S01', channel='1', begin_time=0, end_time=79.003)
        """
        return self._replace(**kwargs)


@dataclass(frozen=True)
class UEM:
    lines: List[UEMLine]

    @cached_property
    def _key_to_index(self):
        keys = [l.filename for l in self.lines]
        assert len(keys) == len(set(keys)), sorted(keys)
        return {k: v for v, k in enumerate(keys)}

    @classmethod
    def load(cls, uem_file: [Path, str, io.TextIOBase, tuple, list]) -> 'UEM':
        def get_parsed_lines(fd):
            return [
                UEMLine.parse(line)
                for line in fd
                if len(line.strip()) > 0  # and not line.strip().startswith(';')  # Does uem allow comments?
            ]

        if not isinstance(uem_file, (tuple, list)):
            uem_file = [uem_file]

        parsed_lines = []

        for file in uem_file:
            if isinstance(file, io.TextIOBase):
                parsed_lines.extend(get_parsed_lines(file))
            elif isinstance(file, (str, Path)):
                with open(file, 'r') as fd:
                    parsed_lines.extend(get_parsed_lines(fd))
            else:
                raise TypeError(file, type(file), uem_file)

        return cls(parsed_lines)

    def _repr_pretty_(self, p, cycle):
        name = self.__class__.__name__
        with p.group(len(name) + 1, name + '(', ')'):
            if cycle:
                p.text('...')
            elif len(self.lines):
                p.pretty(list(self.lines))

    def dump(self, uem_file):
        with open(uem_file, 'w') as fd:
            for line in self.lines:
                fd.write(line.serialize() + '\n')

    def dumps(self):
        return '\n'.join([line.serialize() for line in self.lines])

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.lines[item]
        elif isinstance(item, slice):
            return self.__class__(self.lines[item])
        elif isinstance(item, str):
            return self.lines[self._key_to_index[item]]
        else:
            raise NotImplementedError(type(item), item)

    def __iter__(self):
        return iter(self.lines)

    def grouped_by_filename(self) -> Dict[str, 'UEM']:
        return {
            filename: UEM(list(group))
            for filename, group in groupby(sorted(self.lines), key=lambda x: x.filename)
        }

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
        return UEM(sorted(self.lines, key=key))

    def sorted_by_begin_time(self):
        return UEM(sorted(self.lines, key=lambda x: x.begin_time))
