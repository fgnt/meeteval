import typing
from dataclasses import dataclass
from meeteval.io.base import Base, BaseLine
from meeteval.io.seglst import SegLstSegment
import decimal

if typing.TYPE_CHECKING:
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
            Signal Lookahead Time -- should always be <NA> (Will be added if missing)

        For instance:

            SPEAKER CMU_20020319-1400_d01_NONE 1 130.430000 2.350 <NA> <NA> juliet <NA> <NA>
            SPEAKER CMU_20020319-1400_d01_NONE 1 157.610000 3.060 <NA> <NA> tbc <NA> <NA>
            SPEAKER CMU_20020319-1400_d01_NONE 1 130.490000 0.450 <NA> <NA> chek <NA> <NA>

    Note:
      - The RTTM definition (Appendix A in "The 2009 (RT-09) Rich Transcription
        Meeting Recognition Evaluation Plan") doesn't say anything about the
        channel format or defaults, but dscore enforces a "1" for the channel
        (https://github.com/nryant/dscore#rttm),
        Hence, the default here is 1 for channel.
      - In https://catalog.ldc.upenn.edu/docs/LDC2004T12/RTTM-format-v13.pdf 
        the RTTM file is defined with 9 fields. Hence, we allow 9 and add a 10th
        field. Kaldi also uses 9 fields:
        https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/segmentation/convert_utt2spk_and_segments_to_rttm.py

    """
    type: str = 'SPEAKER'
    filename: str = '<NA>'
    channel: str = '1'
    begin_time: 'float | int | str | decimal.Decimal' = 0
    duration: 'float | int | str | decimal.Decimal' = 0
    orthography: 'str' = '<NA>'
    speaker_type: 'str' = '<NA>'
    speaker_id: 'str' = '<NA>'
    confidence: str = '<NA>'
    signal_look_ahead_time: str = '<NA>'

    @classmethod
    def parse(cls, line: str, parse_float=decimal.Decimal) -> 'RTTMLine':
        """
        >>> RTTMLine.parse('SPEAKER CMU_20020319-1400_d01_NONE 1 130.430000 2.350 <NA> <NA> juliet <NA> <NA>')
        RTTMLine(type='SPEAKER', filename='CMU_20020319-1400_d01_NONE', channel='1', begin_time=Decimal('130.430000'), duration=Decimal('2.350'), orthography='<NA>', speaker_type='<NA>', speaker_id='juliet', confidence='<NA>', signal_look_ahead_time='<NA>')
        >>> RTTMLine.parse('SPEAKER CMU_20020319-1400_d01_NONE 1 130.430000 2.350 <NA> <NA> juliet <NA>')
        RTTMLine(type='SPEAKER', filename='CMU_20020319-1400_d01_NONE', channel='1', begin_time=Decimal('130.430000'), duration=Decimal('2.350'), orthography='<NA>', speaker_type='<NA>', speaker_id='juliet', confidence='<NA>', signal_look_ahead_time='<NA>')
        """
        line_args = list(line.split())
        assert len(line_args) in [9, 10], (len(line_args), line, line_args)
        if len(line_args) < 10:
            line_args  += ['<NA>' * (10 - len(line_args))]
        type_, filename, channel, begin_time, duration, orthography, \
        speaker_type, speaker_id, confidence, signal_look_ahead_time, \
            = line_args

        if parse_float is float:
            def parse_float(x):
                return int(x) if x.isdigit() else float(x)

        return RTTMLine(
            type=type_,
            filename=filename,
            channel=int(channel) if begin_time.isdigit() else channel,
            begin_time=parse_float(begin_time),  # Keep type, int or float,
            duration=parse_float(duration),  # Keep type, int or float,
            orthography=orthography,
            speaker_type=speaker_type,
            speaker_id=speaker_id,
            confidence=confidence,
            signal_look_ahead_time=signal_look_ahead_time,
        )

    @classmethod
    def from_dict(cls, segment: 'SegLstSegment') -> 'RTTMLine':
        # TODO: read spec and handle speech segments with transcripts
        return RTTMLine(
            filename=segment['session_id'],
            channel=segment.get('channel', cls.channel),
            speaker_id=segment['speaker'],
            begin_time=segment['start_time'],
            duration=segment['end_time'] - segment['start_time'],
        )

    def to_seglst_segment(self) -> 'SegLstSegment':
        # TODO: read spec and handle speech segments with transcripts and other types
        d = {
            'session_id': self.filename,
            'speaker': self.speaker_id,
            'start_time': self.begin_time,
            'end_time': self.begin_time + self.duration,
        }

        if self.channel != self.__class__.channel:
            d['channel'] = self.channel

        return d

    def serialize(self):
        """
        >>> line = RTTMLine.parse('SPEAKER CMU_20020319-1400_d01_NONE 1 130.430000 2.350 <NA> <NA> juliet <NA> <NA>')
        >>> line.serialize()
        'SPEAKER CMU_20020319-1400_d01_NONE 1 130.430000 2.350 <NA> <NA> juliet <NA> <NA>'
        """
        return (f'{self.type} {self.filename} {self.channel} '
                f'{self.begin_time} {self.duration} {self.orthography} '
                f'{self.speaker_type} {self.speaker_id} {self.confidence} '
                f'{self.signal_look_ahead_time}')


@dataclass(frozen=True)
class RTTM(Base):
    lines: 'list[RTTMLine]'
    line_cls = RTTMLine

    @classmethod
    def parse(cls, s: str, parse_float=decimal.Decimal) -> 'RTTM':
        return cls([
            RTTMLine.parse(line, parse_float)
            for line in s.split('\n')
            if len(line.strip()) > 0 and not line.strip().startswith(';')
        ])
