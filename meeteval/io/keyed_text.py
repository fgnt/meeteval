from dataclasses import dataclass
from typing import List

from meeteval.io.base import BaseLine, Base


@dataclass(frozen=True)
class KeyedTextLine(BaseLine):
    filename: str
    transcript: str

    @classmethod
    def parse(cls, line: str) -> 'KeyedTextLine':
        filename, transcript = line.strip().split(maxsplit=1)
        return cls(filename, transcript)


@dataclass(frozen=True)
class KeyedText(Base):
    lines: List[KeyedTextLine]
    line_cls = KeyedTextLine

    @classmethod
    def _load(cls, file_descriptor) -> 'List[KeyedTextLine]':
        return [
            KeyedTextLine.parse(line)
            for line in map(str.strip, file_descriptor)
            if len(line) > 0
            # if not line.startswith(';;')
        ]

    def merged_transcripts(self) -> str:
        raise NotImplementedError()

    def utterance_transcripts(self) -> List[str]:
        """There is no notion of an "utterance" in CTM files."""
        raise NotImplementedError()
