import typing
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Dict, List, NamedTuple

if typing.TYPE_CHECKING:
    from meeteval.io.uem import UEM, UEMLine


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
            raise ValueError(f'Unable to parse STM line: {line}')
        try:
            stm_line = STMLine(
                filename,
                int(channel) if begin_time.isdigit() else channel,
                speaker_id,
                int(begin_time) if begin_time.isdigit() else float(begin_time),  # Keep type, int or float
                int(end_time) if end_time.isdigit() else float(end_time),  # Keep type, int or float
                transcript
            )
        except Exception as e:
            raise ValueError(f'Unable to parse STM line: {line}') from e
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
    def load(cls, stm_file: [Path, str, tuple, list]) -> 'STM':
        if isinstance(stm_file, (tuple, list)):
            return STM.merge(*[STM.load(f) for f in stm_file])

        with open(stm_file, 'r') as fd:
            return cls([
                STMLine.parse(line)
                for line in fd
                if len(line.strip()) > 0 and not line.strip().startswith(';')
            ])

    def _repr_pretty_(self, p, cycle):
        name = self.__class__.__name__
        with p.group(len(name) + 1, name + '(', ')'):
            if cycle:
                p.text('...')
            elif len(self.lines):
                p.pretty(list(self.lines))

    @classmethod
    def merge(cls, *stms) -> 'STM':
        return cls([line for stm in stms for line in stm.lines])

    def dump(self, stm_file):
        with open(stm_file, 'w') as fd:
            for line in self.lines:
                fd.write(line.serialize() + '\n')

    def dumps(self):
        return ''.join([
            line.serialize() + '\n'
            for line in self.lines
        ])

    def to_rttm(self):
        from meeteval.io.rttm import RTTM, RTTMLine

        return RTTM([
            RTTMLine(
                filename=line.filename,
                channel=line.channel,
                begin_time=line.begin_time,
                duration_time=line.end_time - line.begin_time,
                speaker_id=line.speaker_id,
                # line.transcript  RTTM doesn't support transcript
                # hence this information is dropped.
            )
            for line in self.lines
        ])

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

    def groupby(self, key) -> Dict[str, 'STM']:
        return {
            filename: STM(list(group))
            for filename, group in groupby(sorted(self.lines), key=key)
        }

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
            new = stm.sorted(key=lambda x: (x.filename, x.begin_time))

        """
        return STM(sorted(self.lines, key=key))

    def sorted_by_begin_time(self):
        return STM(sorted(self.lines, key=lambda x: x.begin_time))

    def utterance_transcripts(self) -> List[str]:
        return [x.transcript for x in sorted(self.lines, key=lambda x: x.begin_time)]

    def merged_transcripts(self) -> str:
        return ' '.join(self.utterance_transcripts())

    @staticmethod
    def _filter_by_uem_keep(line, uem: 'UEM'):
        """
        >>> from meeteval.io.uem import UEM, UEMLine
        >>> uem = UEM([UEMLine('file', '1', 4, 8)])
        >>> STM._filter_by_uem_keep(STMLine('file', '1', 'A', 1, 3, ''), uem)
        False
        >>> STM._filter_by_uem_keep(STMLine('file', '1', 'A', 9, 10, ''), uem)
        False
        >>> STM._filter_by_uem_keep(STMLine('file', '1', 'A', 5, 7, ''), uem)
        True
        >>> STM._filter_by_uem_keep(STMLine('file', '1', 'A', 3, 5, ''), uem)
        True
        >>> STM._filter_by_uem_keep(STMLine('file', '1', 'A', 7, 9, ''), uem)
        True
        >>> STM._filter_by_uem_keep(STMLine('file', '1', 'A', 3, 4, ''), uem)
        False
        >>> STM._filter_by_uem_keep(STMLine('file', '1', 'A', 4, 5, ''), uem)
        True
        >>> STM._filter_by_uem_keep(STMLine('file', '1', 'A', 7, 8, ''), uem)
        True
        >>> STM._filter_by_uem_keep(STMLine('file', '1', 'A', 8, 9, ''), uem)
        False
        """
        try:
            entry: 'UEMLine' = uem[line.filename]
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
        new = STM([
            line
            for line in self.lines
            if self._filter_by_uem_keep(line, uem)
        ])
        if verbose:
            print(f'Applied uem and reduced STM from {len(self)} to {len(new)} lines.')
        return new
