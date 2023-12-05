import typing
from dataclasses import dataclass

from meeteval.io.base import BaseLine, Base
from meeteval.io.tidy import TidySegment


if typing.TYPE_CHECKING:
    from typing import Self, List


@dataclass(frozen=True)
class KeyedTextLine(BaseLine):
    filename: str
    transcript: str

    @classmethod
    def parse(cls, line: str, parse_float=float) -> 'KeyedTextLine':
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

    @classmethod
    def from_tidy(cls, segment: 'TidySegment') -> 'Self':
        return cls(
            filename=segment['session_id'],
            transcript=segment['words'],
        )

    def to_tidy(self) -> 'TidySegment':
        return {
            'session_id': self.filename,
            'words': self.transcript,
        }


@dataclass(frozen=True)
class KeyedText(Base):
    lines: 'List[KeyedTextLine]'
    line_cls = KeyedTextLine

    @classmethod
    def _load(cls, file_descriptor, parse_float) -> 'List[KeyedTextLine]':
        return [
            KeyedTextLine.parse(line, parse_float=parse_float)
            for line in map(str.strip, file_descriptor)
            if len(line) > 0
            # if not line.startswith(';;')
        ]

    def merged_transcripts(self) -> str:
        raise NotImplementedError()

    def utterance_transcripts(self) -> 'List[str]':
        """There is no notion of an "utterance" in CTM files."""
        raise NotImplementedError()
