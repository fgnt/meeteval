import typing
from typing import List, NamedTuple
from dataclasses import dataclass
from meeteval.io.base import Base, BaseLine


if typing.TYPE_CHECKING:
    import decimal
    from meeteval.io.uem import UEM, UEMLine


@dataclass(frozen=True)
class RTTMLine(BaseLine):
    """
    Copied from https://github.com/nryant/dscore#rttm :
        Rich Transcription Time Marked (RTTM) files are space-delimited text
        files containing one turn per line, each line containing ten fields:

            Type -- segment type; should always be SPEAKER
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
    begin_time: 'float | int | str | decimal.Decimal' = 0
    duration: 'float | int | str | decimal.Decimal' = 0
    othography: 'str' = '<NA>'
    speaker_type: 'str' = '<NA>'
    speaker_id: 'str' = '<NA>'
    confidence: str = '<NA>'
    signal_look_ahead_time: str = '<NA>'

    @classmethod
    def parse(cls, line: str, parse_float=float) -> 'RTTMLine':
        """
        >>> RTTMLine.parse('SPEAKER CMU_20020319-1400_d01_NONE 1 130.430000 2.350 <NA> <NA> juliet <NA> <NA>')
        RTTMLine(type='SPEAKER', filename='CMU_20020319-1400_d01_NONE', channel='1', begin_time=130.43, duration=2.35, othography='<NA>', speaker_type='<NA>', speaker_id='juliet', confidence='<NA>', signal_look_ahead_time='<NA>')
        """
        type_, filename, channel, begin_time, duration, othography, \
        speaker_type, speaker_id, confidence, signal_look_ahead_time, \
            = line.split()

        return RTTMLine(
            type=type_,
            filename=filename,
            channel=int(channel) if begin_time.isdigit() else channel,
            begin_time=int(begin_time) if begin_time.isdigit() else parse_float(begin_time),  # Keep type, int or float,
            duration=int(duration) if duration.isdigit() else parse_float(duration),  # Keep type, int or float,
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
                f'{self.begin_time} {self.duration} {self.othography} '
                f'{self.speaker_type} {self.speaker_id} {self.confidence} '
                f'{self.signal_look_ahead_time}')


@dataclass(frozen=True)
class RTTM(Base):
    lines: List[RTTMLine]
    line_cls = RTTMLine

    @classmethod
    def _load(cls, file_descriptor, parse_float) -> 'List[RTTMLine]':
        return [
            RTTMLine.parse(line, parse_float)
            for line in file_descriptor
            if len(line.strip()) > 0 and not line.strip().startswith(';')
        ]
