import decimal
import io
from dataclasses import dataclass
from pathlib import Path

from meeteval.io.base import Base, BaseLine

import typing

if typing.TYPE_CHECKING:
    from typing import Self

try:
    from functools import cached_property
except ImportError:
    # Fallback for Python 3.7 and lower, since cached_property was added in
    # Python 3.8.
    from cached_property import cached_property  # Python 3.7


@dataclass(frozen=True)
class UEMLine(BaseLine):
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

    Since all field names from dscore don't match the typically used filenames
    from NIST and the source code of
    https://github.com/usnistgov/SCTK/blob/master/src/asclite/core/uemfilter.h
    uses the typical names, use field names that match with STM.

    """
    filename: str
    channel: 'str | int' = '<NA>'
    begin_time: 'float | int | decimal.Decimal' = 0
    end_time: 'float | int | decimal.Decimal' = 0

    @classmethod
    def parse(cls, line: str, parse_float=decimal.Decimal) -> 'UEMLine':
        """
        >>> UEMLine.parse('S01 1 60.001 79.003')
        UEMLine(filename='S01', channel=1, begin_time=Decimal('60.001'), end_time=Decimal('79.003'))
        """
        filename, channel, begin_time, end_time = line.split()

        if parse_float is float:
            def parse_float(x):
                return int(x) if x.isdigit() else float(x)

        uem_line = UEMLine(
            filename=filename,
            channel=int(channel) if channel.isdigit() else channel,
            begin_time=parse_float(begin_time),  # Keep type, int or float,
            end_time=parse_float(end_time),  # Keep type, int or float,
        )

        assert uem_line.begin_time <= uem_line.end_time, uem_line

        return uem_line

    def serialize(self):
        """
        >>> line = UEMLine.parse('S01 1 60.001 79.003')
        >>> line.serialize()
        'S01 1 60.001 79.003'
        """
        return (f'{self.filename} {self.channel} '
                f'{self.begin_time} {self.end_time}')


@dataclass(frozen=True)
class UEM(Base):
    lines: 'list[UEMLine]'
    line_cls = UEMLine

    @cached_property
    def _key_to_index(self):
        keys = [line.filename for line in self.lines]
        assert len(keys) == len(set(keys)), sorted(keys)
        return {k: v for v, k in enumerate(keys)}

    @classmethod
    def load(cls, file: [Path, str, io.TextIOBase, tuple, list], parse_float=decimal.Decimal) -> 'Self':
        uem = super().load(file, parse_float)

        # Check that there are no duplicate filenames because UEM only supports
        # a single segment per file
        if len(set([l.filename for l in uem.lines])) < len(uem.lines):
            raise ValueError(
                f'UEM file contains duplicate filenames, but only a single '
                f'scoring region per filename is supported: {uem}'
            )

        return uem

    @classmethod
    def parse(cls, s: str, parse_float=decimal.Decimal) -> 'UEM':
        uem = cls([
            UEMLine.parse(line, parse_float)
            for line in s.split('\n')
            if len(line.strip()) > 0  # and not line.strip().startswith(';')  # Does uem allow comments?
        ])

        # Check that there are no duplicate filenames because UEM only supports
        # a single segment per file
        if len(set([l.filename for l in uem.lines])) < len(uem.lines):
            raise ValueError(
                f'UEM file contains duplicate filenames, but only a single '
                f'scoring region per filename is supported: {uem}'
            )

        return uem

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.lines[self._key_to_index[item]]
        else:
            return super().__getitem__(item)
