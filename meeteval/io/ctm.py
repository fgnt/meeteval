from dataclasses import dataclass
from typing import Dict, List, Optional
from typing import NamedTuple
from meeteval.io.base import Base, BaseLine


__all__ = [
    'CTMLine',
    'CTM',
    'CTMGroup',
]


@dataclass(frozen=True)
class CTMLine(BaseLine):
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
    channel: 'str | int'
    begin_time: float
    duration: float
    word: str
    confidence: Optional[int] = None

    @classmethod
    def parse(cls, line: str, parse_float=float) -> 'CTMLine':
        try:
            filename, channel, begin_time, duration, word, *confidence = line.strip().split()
            assert len(confidence) < 2, confidence
            ctm_line = CTMLine(
                filename,
                int(channel) if begin_time.isdigit() else channel,
                int(begin_time) if begin_time.isdigit() else parse_float(begin_time),  # Keep type, int or float
                int(duration) if duration.isdigit() else parse_float(duration),  # Keep type, int or float
                word,
                confidence[0] if confidence else None
            )
            assert ctm_line.begin_time >= 0, ctm_line
            assert ctm_line.duration >= 0, ctm_line
        except Exception as e:
            raise ValueError(f'Unable to parse CTM line: {line}') from e
        return ctm_line

    def serialize(self):
        # ToDO, when someone needs it.
        raise NotImplementedError(type(self))


@dataclass(frozen=True)
class CTM(Base):
    lines: List[CTMLine]
    line_cls = CTMLine

    @classmethod
    def _load(cls, file_descriptor, parse_float) -> 'List[CTMLine]':
        return [
            CTMLine.parse(line, parse_float=parse_float)
            for line in map(str.strip, file_descriptor)
            if len(line) > 0
            if not line.startswith(';;')
        ]

    def merged_transcripts(self) -> str:
        return ' '.join([x.word for x in sorted(self.lines, key=lambda x: x.begin_time)])

    def utterance_transcripts(self) -> List[str]:
        """There is no notion of an "utterance" in CTM files."""
        raise NotImplementedError()


@dataclass(frozen=True)
class CTMGroup:
    ctms: 'Dict[str, CTM]'

    @classmethod
    def load(cls, ctm_files, parse_float=float):
        return cls({str(ctm_file): CTM.load(ctm_file, parse_float=parse_float)
                    for ctm_file in ctm_files})

    def grouped_by_filename(self) -> Dict[str, 'CTMGroup']:
        groups = {
            k: ctm.grouped_by_filename() for k, ctm in self.ctms.items()
        }
        keys = next(iter(groups.values())).keys()

        for group in groups.values():
            if group.keys() != keys:
                raise ValueError('Example IDs must match across CTM files!')

        return {
            key: CTMGroup({
                k: g[key] for k, g in groups.items()
            })
            for key in keys
        }

    def grouped_by_speaker_id(self) -> Dict[str, CTM]:
        return self.ctms
