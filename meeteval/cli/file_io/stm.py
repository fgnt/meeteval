from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import TextIO, Dict, List, NamedTuple


class STMLine(NamedTuple):
    """
    Represents one line of an STM file, which is an ordered list of fields:

    STM :== <filename> <channel> <speaker_id> <begin_time> <end_time> <transcript>

    - filename: name of the segment (recording)
    - channel: anything (always ignored)
    - speaker_id: ID of the speaker or channel
    - begin_time: in seconds
    - end_time: in seconds
    - transcript: space-separated list of words

    The STM format definition can be found here: https://www.nist.gov/system/files/documents/2021/08/03/OpenASR20_EvalPlan_v1_5.pdf
    """
    filename: str
    channel: int
    speaker_id: str
    begin_time: float
    end_time: float
    transcript: str

    @classmethod
    def parse(cls, line: str) -> 'STMLine':
        filename, channel, speaker_id, begin_time, end_time, transcript = line.strip().split(maxsplit=5)
        stm_line = STMLine(
            filename, int(channel), speaker_id, float(begin_time), float(end_time),
            transcript
        )
        assert stm_line.begin_time >= 0, stm_line
        # We currently ignore the end time, so it's fine when it's before begin_time
        # assert stm_line.end_time >= stm_line.begin_time, stm_line
        return stm_line


@dataclass(frozen=True)
class STM:
    lines: List[STMLine]

    @classmethod
    def load(cls, stm_file: Path) -> 'STM':
        with stm_file.open('r') as f:
            return cls([
                STMLine.parse(line)
                for line in f
                if len(line.strip()) > 0 and not line.strip().startswith(';')
            ])

    def grouped_by_filename(self) -> Dict[str, 'STM']:
        return {
            filename: STM(list(group))
            for filename, group in groupby(sorted(self.lines), key=lambda x: x.filename)
        }

    def grouped_by_speaker_id(self):
        return [
            STM(list(speaker_group))
            for speaker_id, speaker_group in
            groupby(sorted(self.lines, key=lambda x: x.speaker_id), key=lambda x: x.speaker_id)
        ]

    def utterance_transcripts(self) -> List[str]:
        return [x.transcript for x in sorted(self.lines, key=lambda x: x.begin_time)]

    def merged_transcripts(self) -> str:
        return ' '.join(self.utterance_transcripts())
