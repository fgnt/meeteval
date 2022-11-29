from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import TextIO, Dict, List, Optional
from typing import NamedTuple


__all__ = [
    'CTMLine',
    'CTM',
    'CTMGroup',
]


class CTMLine(NamedTuple):
    """
    Represents one line of a CTM file, which is an ordered list of fields.

    CTM :== <filename> <channel> <begin_time> <duration> <word> [ <confidence> ]

    - filename: name of the recording
    - channel: ignored
    - begin_time: in seconds
    - duration: in seconds
    - word: A single word
    - confidence: optional and ignored

    CTM file format definition: https://www.nist.gov/system/files/documents/2021/08/03/OpenASR20_EvalPlan_v1_5.pdf
    """
    filename: str
    channel: int
    begin_time: float
    duration: float
    word: str
    confidence: Optional[int] = None

    @classmethod
    def parse(cls, line: str) -> 'CTMLine':
        filename, channel, begin_time, duration, word, *confidence = line.strip().split()
        assert len(confidence) < 2, confidence
        ctm_line = CTMLine(
            filename, int(channel), float(begin_time), float(duration), word,
            confidence[0] if confidence else 0
        )
        assert ctm_line.begin_time >= 0, ctm_line
        assert ctm_line.duration >= 0, ctm_line
        return ctm_line


@dataclass(frozen=True)
class CTM:
    lines: List[CTMLine]

    @classmethod
    def load(cls, ctm_file: Path) -> 'CTM':
        with ctm_file.open('r') as f:
            return cls([CTMLine.parse(line) for line in f if len(line) > 0])

    def grouped_by_filename(self) -> Dict[str, 'CTM']:
        return {
            filename: CTM(list(group))
            for filename, group in groupby(sorted(self.lines), key=lambda x: x.filename)
        }

    def merged_transcripts(self) -> str:
        return ' '.join([x.word for x in sorted(self.lines, key=lambda x: x.begin_time)])

    def utterance_transcripts(self) -> List[str]:
        """There is no notion of an "utterance" in CTM files."""
        raise NotImplementedError()


@dataclass(frozen=True)
class CTMGroup:
    ctms: List[CTM]

    @classmethod
    def load(cls, ctm_files):
        return cls([CTM.load(ctm_file) for ctm_file in ctm_files])

    def grouped_by_filename(self) -> Dict[str, 'CTMGroup']:
        groups = [
            ctm.grouped_by_filename() for ctm in self.ctms
        ]
        keys = groups[0].keys()

        for group in groups:
            if group.keys() != keys:
                raise ValueError('Example IDs must match across CTM files!')

        return {
            key: CTMGroup([
                g[key] for g in groups
            ])
            for key in keys
        }

    def grouped_by_speaker_id(self) -> List[CTM]:
        return self.ctms
