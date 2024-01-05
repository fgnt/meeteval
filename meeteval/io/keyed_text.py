import typing
from dataclasses import dataclass

from meeteval.io.base import BaseLine, Base
from meeteval.io.seglst import SegLstSegment
import decimal


if typing.TYPE_CHECKING:
    from typing import Self


@dataclass(frozen=True)
class KeyedTextLine(BaseLine):
    filename: str
    transcript: str

    @classmethod
    def parse(cls, line: str, parse_float=decimal.Decimal) -> 'KeyedTextLine':
        """
        >>> KeyedTextLine.parse("key a transcript")
        KeyedTextLine(filename='key', transcript='a transcript')
        >>> KeyedTextLine.parse("   key  a transcript  ")
        KeyedTextLine(filename='key', transcript='a transcript')
        >>> KeyedTextLine.parse("  key ")
        KeyedTextLine(filename='key', transcript='')
        >>> KeyedTextLine.parse("key")
        KeyedTextLine(filename='key', transcript='')
        """
        line = line.strip()
        if ' ' in line:
            filename, transcript = line.split(maxsplit=1)
        else:
            filename = line
            transcript = ''
        return cls(filename, transcript)

    def serialize(self):
        return f'{self.filename} {self.transcript}'

    @classmethod
    def from_dict(cls, segment: 'SegLstSegment') -> 'Self':
        return cls(
            filename=segment['session_id'],
            transcript=segment['words'],
        )

    def to_seglst_segment(self) -> 'SegLstSegment':
        return {
            'session_id': self.filename,
            'words': self.transcript,
        }


@dataclass(frozen=True)
class KeyedText(Base):
    lines: 'list[KeyedTextLine]'
    line_cls = KeyedTextLine

    @classmethod
    def parse(cls, s: str, parse_float=decimal.Decimal) -> 'Self':
        return cls([
            KeyedTextLine.parse(line, parse_float=parse_float)
            for line in map(str.strip, s.split('\n'))
            if len(line) > 0
            # if not line.startswith(';;')
        ])

    def merged_transcripts(self) -> str:
        raise NotImplementedError()

    def utterance_transcripts(self) -> 'list[str]':
        """There is no notion of an "utterance" in CTM files."""
        raise NotImplementedError()
