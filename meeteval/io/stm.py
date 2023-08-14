import sys
import typing
from dataclasses import dataclass
from typing import List, NamedTuple
from meeteval.io.base import Base, BaseLine

if typing.TYPE_CHECKING:
    import decimal
    from meeteval.io.uem import UEM, UEMLine
    from meeteval.wer import ErrorRate

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
    def parse(cls, line: str, parse_float=float) -> 'STMLine':
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
                int(begin_time) if begin_time.isdigit() else parse_float(begin_time),  # Keep type, int or float
                int(end_time) if end_time.isdigit() else parse_float(end_time),  # Keep type, int or float
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

    def segment_dict(self):
        """Returns a segment dict in the style of Chime-7 annotations"""
        return {
            'start_time': self.begin_time,
            'end_time': self.end_time,
            'words': self.transcript,
            'speaker': self.speaker_id,
            'session_id': self.filename
        }


@dataclass(frozen=True)
class STM(Base):
    lines: List[STMLine]
    line_cls = STMLine

    @classmethod
    def _load(cls, file_descriptor, parse_float) -> 'List[STMLine]':
        return [
            STMLine.parse(line, parse_float)
            for line in file_descriptor
            if len(line.strip()) > 0 and not line.strip().startswith(';')
        ]

    @classmethod
    def merge(cls, *stms) -> 'STM':
        return cls([line for stm in stms for line in stm.lines])

    def to_rttm(self):
        from meeteval.io.rttm import RTTM, RTTMLine

        # ToDo: Fix `line.end_time - line.begin_time`, when they are floats.
        #       Sometimes there is a small error and the error will be written
        #       to the rttm file.

        return RTTM([
            RTTMLine(
                filename=line.filename,
                channel=line.channel,
                begin_time=line.begin_time,
                duration=line.end_time - line.begin_time,
                speaker_id=line.speaker_id,
                # line.transcript  RTTM doesn't support transcript
                # hence this information is dropped.
            )
            for line in self.lines
        ])

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

    def utterance_transcripts(self) -> List[str]:
        return [x.transcript for x in sorted(self.lines, key=lambda x: x.begin_time)]

    def merged_transcripts(self) -> str:
        return ' '.join(self.utterance_transcripts())

    def segments(self):
        return [l.segment_dict() for l in self]


def iter_examples(reference: 'STM', hypothesis: 'STM', *, allowed_empty_examples_ratio=0.1):
    reference = reference.grouped_by_filename()
    hypothesis = hypothesis.grouped_by_filename()

    if reference.keys() != hypothesis.keys():
        h_minus_r = list(set(hypothesis.keys()) - set(reference.keys()))
        r_minus_h = list(set(reference.keys()) - set(hypothesis.keys()))

        ratio = len(r_minus_h) / len(reference.keys())

        if h_minus_r:
            # This is a warning, because missing in reference is not a problem,
            # we can safely ignore it. Missing in hypothesis is a problem,
            # because we cannot distinguish between silence and missing.
            print(
                'WARNING: Keys of reference and hypothesis differ\n'
                f'hypothesis - reference: e.g. {h_minus_r[:5]} (Total: {len(h_minus_r)} of {len(reference)})\n'
                f'Drop them.',
                file=sys.stderr,
            )
            hypothesis = {
                k: v
                for k, v in hypothesis.items()
                if k not in h_minus_r
            }

        if len(r_minus_h) == 0 and ratio <= allowed_empty_examples_ratio:
            print(
                f'WARNING: Missing {ratio * 100:.3} % = {len(r_minus_h)}/{len(reference.keys())} of recordings in hypothesis.\n'
                f'Please check your system, if it ignored some recordings or predicted no transcriptions for some recordings.\n'
                f'Continue with the assumption, that the system predicted silence for the missing recordings.',
                file=sys.stderr
            )
        else:
            raise RuntimeError(
                'Keys of reference and hypothesis differ\n'
                f'hypothesis - reference: e.g. {h_minus_r[:5]} (Total: {len(h_minus_r)} of {len(hypothesis)})\n'
                f'reference - hypothesis: e.g. {r_minus_h[:5]} (Total: {len(r_minus_h)} of {len(reference)})'
            )

    for filename in reference:
        yield filename, reference[filename], hypothesis[filename]


def apply_stm_multi_file(
        fn: 'typing.Callable[[STM, STM], ErrorRate]',
        reference: 'STM',
        hypothesis: 'STM',
        *,
        allowed_empty_examples_ratio=0.1
):
    return {f: fn(r, h) for f, r, h in iter_examples(
        reference, hypothesis,
        allowed_empty_examples_ratio=allowed_empty_examples_ratio
    )}
