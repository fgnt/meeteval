import io
import typing
from typing import TextIO, Dict, List, NamedTuple
from itertools import groupby
from pathlib import Path
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    from meeteval.io.uem import UEM, UEMLine


class RTTMLine(NamedTuple):
    """
    Copied from https://github.com/nryant/dscore#rttm :
        Rich Transcription Time Marked (RTTM) files are space-delimited text
        files containing one turn per line, each line containing ten fields:

            Type -- segment type; should always by SPEAKER
            File ID -- file name; basename of the recording minus extension (e.g., rec1_a)
            Channel ID -- channel (1-indexed) that turn is on; should always be 1
            Turn Onset -- onset of turn in seconds from beginning of recording
            Turn Duration -- duration of turn in seconds
            Orthography Field -- should always by <NA>
            Speaker Type -- should always be <NA>
            Speaker Name -- name of speaker of turn; should be unique within scope of each file
            Confidence Score -- system confidence (probability) that information is correct; should always be <NA>
            Signal Lookahead Time -- should always be <NA>

        For instance:

            SPEAKER CMU_20020319-1400_d01_NONE 1 130.430000 2.350 <NA> <NA> juliet <NA> <NA>
            SPEAKER CMU_20020319-1400_d01_NONE 1 157.610000 3.060 <NA> <NA> tbc <NA> <NA>
            SPEAKER CMU_20020319-1400_d01_NONE 1 130.490000 0.450 <NA> <NA> chek <NA> <NA>
    """
    type: str = 'SPEAKER'
    filename: str = '<NA>'
    channel: str = '<NA>'
    begin_time: 'float | int | str' = 0
    duration_time: 'float | int | str' = 0
    othography: 'str' = '<NA>'
    speaker_type: 'str' = '<NA>'
    speaker_id: 'str' = '<NA>'
    confidence: str = '<NA>'
    signal_look_ahead_time: str = '<NA>'

    @classmethod
    def parse(cls, line: str) -> 'RTTMLine':
        """
        >>> RTTMLine.parse('SPEAKER CMU_20020319-1400_d01_NONE 1 130.430000 2.350 <NA> <NA> juliet <NA> <NA>')
        RTTMLine(type='SPEAKER', filename='CMU_20020319-1400_d01_NONE', channel='1', begin_time=130.43, duration_time=2.35, othography='<NA>', speaker_type='<NA>', speaker_id='juliet', confidence='<NA>', signal_look_ahead_time='<NA>')
        """
        type_, filename, channel, begin_time, duration_time, othography, \
        speaker_type, speaker_id, confidence, signal_look_ahead_time, \
            = line.split()

        return RTTMLine(
            type=type_,
            filename=filename,
            channel=int(channel) if begin_time.isdigit() else channel,
            begin_time=int(begin_time) if begin_time.isdigit() else float(begin_time),  # Keep type, int or float,
            duration_time=int(duration_time) if duration_time.isdigit() else float(duration_time),  # Keep type, int or float,
            othography=othography,
            speaker_type=speaker_type,
            speaker_id=speaker_id,
            confidence=confidence,
            signal_look_ahead_time=signal_look_ahead_time,
        )

    def serialize(self):
        """
        >>> line = RTTMLine.parse('SPEAKER CMU_20020319-1400_d01_NONE 1 130.430000 2.350 <NA> <NA> juliet <NA> <NA>')
        >>> line.serialize()
        'SPEAKER CMU_20020319-1400_d01_NONE 1 130.43 2.35 <NA> <NA> juliet <NA> <NA>'
        """
        return (f'{self.type} {self.filename} {self.channel} '
                f'{self.begin_time} {self.duration_time} {self.othography} '
                f'{self.speaker_type} {self.speaker_id} {self.confidence} '
                f'{self.signal_look_ahead_time}')

    def replace(self, **kwargs):
        """
        Return a new instance of the named tuple replacing specified fields with new values.

        >>> line = RTTMLine.parse('SPEAKER CMU_20020319-1400_d01_NONE 1 130.430000 2.350 <NA> <NA> juliet <NA> <NA>')
        >>> line
        RTTMLine(type='SPEAKER', filename='CMU_20020319-1400_d01_NONE', channel='1', begin_time=130.43, duration_time=2.35, othography='<NA>', speaker_type='<NA>', speaker_id='juliet', confidence='<NA>', signal_look_ahead_time='<NA>')
        >>> line.replace(speaker_id='B')
        RTTMLine(type='SPEAKER', filename='CMU_20020319-1400_d01_NONE', channel='1', begin_time=130.43, duration_time=2.35, othography='<NA>', speaker_type='<NA>', speaker_id='B', confidence='<NA>', signal_look_ahead_time='<NA>')
        """
        return self._replace(**kwargs)


@dataclass(frozen=True)
class RTTM:
    lines: List[RTTMLine]

    @classmethod
    def load(cls, rttm_file: [Path, str, io.TextIOBase, tuple, list]) -> 'RTTM':
        def get_parsed_lines(fd):
            return [
                RTTMLine.parse(line)
                for line in fd
                if len(line.strip()) > 0 and not line.strip().startswith(';')
            ]

        if not isinstance(rttm_file, (tuple, list)):
            rttm_file = [rttm_file]

        parsed_lines = []

        for file in rttm_file:
            if isinstance(file, io.TextIOBase):
                parsed_lines.extend(get_parsed_lines(file))
            elif isinstance(file, (str, Path)):
                with open(file, 'r') as fd:
                    parsed_lines.extend(get_parsed_lines(fd))
            else:
                raise TypeError(file, type(file), rttm_file)

        return cls(parsed_lines)

    def _repr_pretty_(self, p, cycle):
        name = self.__class__.__name__
        with p.group(len(name) + 1, name + '(', ')'):
            if cycle:
                p.text('...')
            elif len(self.lines):
                p.pretty(list(self.lines))

    def dump(self, rttm_file):
        with open(rttm_file, 'w') as fd:
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

    def __len__(self):
        return len(self.lines)

    def grouped_by_filename(self) -> Dict[str, 'RTTM']:
        return {
            filename: RTTM(list(group))
            for filename, group in groupby(sorted(self.lines), key=lambda x: x.filename)
        }

    def grouped_by_speaker_id(self):
        return {
            speaker_id: RTTM(list(speaker_group))
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
            new = rttm.sorted(key=lambda x: (x.filename, x.begin_time))

        """
        return RTTM(sorted(self.lines, key=key))

    def sorted_by_begin_time(self):
        return RTTM(sorted(self.lines, key=lambda x: x.begin_time))

    @staticmethod
    def _filter_by_uem_keep(line: 'RTTMLine', uem: 'UEM'):
        """
        >>> from meeteval.io.uem import UEM, UEMLine
        >>> from meeteval.io.stm import STM, STMLine
        >>> uem = UEM([UEMLine('file', RTTMLine.speaker_id, 4, 8)])
        >>> def to_rttm(*args):
        ...     return STM([STMLine(*args)]).to_rttm()[0]
        >>> RTTM._filter_by_uem_keep(to_rttm('file', '1', 'A', 1, 3, ''), uem)
        False
        >>> RTTM._filter_by_uem_keep(to_rttm('file', '1', 'A', 9, 10, ''), uem)
        False
        >>> RTTM._filter_by_uem_keep(to_rttm('file', '1', 'A', 5, 7, ''), uem)
        True
        >>> RTTM._filter_by_uem_keep(to_rttm('file', '1', 'A', 3, 5, ''), uem)
        True
        >>> RTTM._filter_by_uem_keep(to_rttm('file', '1', 'A', 7, 9, ''), uem)
        True
        >>> RTTM._filter_by_uem_keep(to_rttm('file', '1', 'A', 3, 4, ''), uem)
        False
        >>> RTTM._filter_by_uem_keep(to_rttm('file', '1', 'A', 4, 5, ''), uem)
        True
        >>> RTTM._filter_by_uem_keep(to_rttm('file', '1', 'A', 7, 8, ''), uem)
        True
        >>> RTTM._filter_by_uem_keep(to_rttm('file', '1', 'A', 8, 9, ''), uem)
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
            if line.begin_time < end_time \
                    and begin_time < line.begin_time + line.duration_time:
                return True
            return False

    def filter_by_uem(self, uem: 'UEM', verbose=False):
        new = RTTM([
            line
            for line in self.lines
            if self._filter_by_uem_keep(line, uem)
        ])
        if verbose:
            print(f'Applied uem and reduced STM from {len(self)} to {len(new)} lines.')
        return new
