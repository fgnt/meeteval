from dataclasses import dataclass
from meeteval.io.base import Base, BaseLine
import decimal

from meeteval.io.seglst import SegLstSegment

__all__ = [
    'STMLine',
    'STM',
]


@dataclass(frozen=True)
class STMLine(BaseLine):
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
    begin_time: 'float | int | decimal.Decimal'
    end_time: 'float | int | decimal.Decimal'
    transcript: str

    @classmethod
    def parse(cls, line: str, parse_float=decimal.Decimal) -> 'STMLine':
        filename, channel, speaker_id, begin_time, end_time, *transcript = line.strip().split(maxsplit=5)

        if len(transcript) == 1:
            transcript, = transcript
        elif len(transcript) == 0:
            transcript = ''  # empty transcript
        else:
            raise ValueError(f'Unable to parse STM line: {line}')
        try:
            if parse_float is float:
                def parse_float(x):
                    return int(x) if x.isdigit() else float(x)

            stm_line = STMLine(
                filename,
                int(channel) if begin_time.isdigit() else channel,
                speaker_id,
                parse_float(begin_time),
                parse_float(end_time),
                transcript
            )
        except Exception as e:
            raise ValueError(f'Unable to parse STM line: {line}') from e
        # assert stm_line.begin_time >= 0, stm_line
        # We currently ignore the end time, so it's fine when it's before begin_time
        # assert stm_line.end_time >= stm_line.begin_time, stm_line
        return stm_line

    @classmethod
    def from_dict(cls, segment: 'SegLstSegment'):
        return cls(
            filename=segment['session_id'],
            channel=segment.get('channel', 1),
            speaker_id=segment['speaker'],
            begin_time=segment['start_time'],
            end_time=segment['end_time'],
            transcript=segment['words'],
        )

    def to_seglst_segment(self) -> 'SegLstSegment':
        return {
            'session_id': self.filename,
            'channel': self.channel,
            'speaker': self.speaker_id,
            'start_time': self.begin_time,
            'end_time': self.end_time,
            'words': self.transcript,
        }

    def serialize(self):
        """
        >>> line = STMLine.parse('rec1 0 A 10 20 Hello World')
        >>> line.serialize()
        'rec1 0 A 10 20 Hello World'
        """
        return (f'{self.filename} {self.channel} {self.speaker_id} '
                f'{self.begin_time} {self.end_time} {self.transcript}')


@dataclass(frozen=True)
class STM(Base):
    lines: 'list[STMLine]'
    line_cls = STMLine

    @classmethod
    def parse(cls, s: str, parse_float=decimal.Decimal) -> 'STM':
        return cls([
            STMLine.parse(line, parse_float)
            for line in s.split('\n')
            if len(line.strip()) > 0 and not line.strip().startswith(';')
        ])

    def to_rttm(self):
        from meeteval.io.rttm import RTTM, RTTMLine

        # ToDo: Fix `line.end_time - line.begin_time`, when they are floats.
        #       Sometimes there is a small error and the error will be written
        #       to the rttm file.
        return RTTM.new(self.to_seglst())

    def to_array_interval(self, sample_rate, group=True):
        import paderbox as pb
        if group:
            return {
                f: {
                    s: v2.to_array_interval(sample_rate, group=False)
                    for s, v2 in v1.grouped_by_speaker_id().items()
                }
                for f, v1 in self.grouped_by_filename().items()
            }
        else:
            return pb.array.interval.ArrayInterval.from_pairs([
                (round(line.begin_time * sample_rate), round(line.end_time * sample_rate))
                for line in self.lines])

    def utterance_transcripts(self) -> 'list[str]':
        return [x.transcript for x in sorted(self.lines, key=lambda x: x.begin_time)]

    def merged_transcripts(self) -> str:
        return ' '.join(self.utterance_transcripts())


if __name__ == '__main__':
    def to_rttm(file):
        from pathlib import Path
        STM.load(file).to_rttm().dump(Path(file).with_suffix('.rttm'))


    import fire

    fire.Fire({
        'to_rttm': to_rttm,
    })
