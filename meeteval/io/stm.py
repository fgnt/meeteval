from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
import io
from typing import TextIO, Dict, List, NamedTuple

__all__ = [
    'STMLine',
    'STM',
]


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
    channel: 'int | str'
    speaker_id: str
    begin_time: 'float | int'
    end_time: 'float | int'
    transcript: str

    @classmethod
    def parse(cls, line: str) -> 'STMLine':
        filename, channel, speaker_id, begin_time, end_time, *transcript = line.strip().split(maxsplit=5)

        if len(transcript) == 1:
            transcript, = transcript
        elif len(transcript) == 0:
            transcript = ''  # empty transcript
        else:
            raise ValueError(line)

        stm_line = STMLine(
            filename,
            int(channel) if begin_time.isdigit() else channel,
            speaker_id,
            int(begin_time) if begin_time.isdigit() else float(begin_time),  # Keep type, int or float
            int(end_time) if begin_time.isdigit() else float(end_time),  # Keep type, int or float
            transcript
        )
        assert stm_line.begin_time >= 0, stm_line
        # We currently ignore the end time, so it's fine when it's before begin_time
        # assert stm_line.end_time >= stm_line.begin_time, stm_line
        return stm_line

    def serialize(self):
        """
        >>> line = STMLine.parse('rec1 0 A 10 20 Hello World')
        >>> line.serialize()
        'rec1 0 A 10 20 Hello World'
        """
        return (f'{self.filename} {self.channel} {self.speaker_id} '
                f'{self.begin_time} {self.end_time} {self.transcript}')

    def replace(self, **kwargs):
        """
        Return a new instance of the named tuple replacing specified fields with new values.

        >>> line = STMLine.parse('rec1 0 A 10 20 Hello World')
        >>> line
        STMLine(filename='rec1', channel=0, speaker_id='A', begin_time=10, end_time=20, transcript='Hello World')
        >>> line.replace(speaker_id='B')
        STMLine(filename='rec1', channel=0, speaker_id='B', begin_time=10, end_time=20, transcript='Hello World')
        """
        return self._replace(**kwargs)


@dataclass(frozen=True)
class STM:
    lines: List[STMLine]

    @classmethod
    def load(cls, stm_file: [Path, str, io.TextIOBase, tuple, list]) -> 'STM':
        def get_parsed_lines(fd):
            return [
                STMLine.parse(line)
                for line in fd
                if len(line.strip()) > 0 and not line.strip().startswith(';')
            ]

        if not isinstance(stm_file, (tuple, list)):
            stm_file = [stm_file]

        parsed_lines = []

        for file in stm_file:
            if isinstance(file, io.TextIOBase):
                parsed_lines.extend(get_parsed_lines(file))
            elif isinstance(file, (str, Path)):
                with open(file, 'r') as fd:
                    parsed_lines.extend(get_parsed_lines(fd))
            else:
                raise TypeError(file, type(file), stm_file)

        return cls(parsed_lines)

    def _repr_pretty_(self, p, cycle):
        name = self.__class__.__name__
        with p.group(len(name) + 1, name + '(', ')'):
            if cycle:
                p.text('...')
            elif len(self.lines):
                p.pretty(list(self.lines))

    def dump(self, stm_file):
        with open(stm_file, 'w') as fd:
            for line in self.lines:
                fd.write(line.serialize() + '\n')

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.lines[item]
        elif isinstance(item, slice):
            return self.__class__(self.lines[item])
        else:
            raise NotImplementedError(type(item), item)

    def __iter__(self):
        return iter(self.lines)

    def grouped_by_filename(self) -> Dict[str, 'STM']:
        return {
            filename: STM(list(group))
            for filename, group in groupby(sorted(self.lines), key=lambda x: x.filename)
        }

    def grouped_by_speaker_id(self):
        return {
            speaker_id: STM(list(speaker_group))
            for speaker_id, speaker_group in
            groupby(sorted(self.lines, key=lambda x: x.speaker_id), key=lambda x: x.speaker_id)
        }

    def sorted_by_begin_time(self):
        return STM(sorted(self.lines, key=lambda x: x.begin_time))

    def utterance_transcripts(self) -> List[str]:
        return [x.transcript for x in sorted(self.lines, key=lambda x: x.begin_time)]

    def merged_transcripts(self) -> str:
        return ' '.join(self.utterance_transcripts())
