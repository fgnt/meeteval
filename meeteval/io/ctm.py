import typing
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional
from meeteval.io.base import Base, BaseLine, BaseABC
import decimal

if typing.TYPE_CHECKING:
    from typing import Self
    from meeteval.io.seglst import SegLstSegment, SegLST

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
    begin_time: 'decimal.Decimal | float'
    duration: 'decimal.Decimal | float'
    word: str
    confidence: Optional[int] = None

    @classmethod
    def parse(cls, line: str, parse_float=decimal.Decimal) -> 'CTMLine':
        try:
            # CB: Should we disable the support for missing confidence?
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
            assert ctm_line.duration >= 0, ctm_line
        except Exception as e:
            raise ValueError(f'Unable to parse CTM line: {line}') from e
        return ctm_line

    def serialize(self):
        """
        >>> line = CTMLine.parse('rec1 0 10 2 Hello 1')
        >>> line.serialize()
        'rec1 0 10 2 Hello 1'
        """
        s = f'{self.filename} {self.channel} {self.begin_time} {self.duration} {self.word}'
        if self.confidence is not None:
            s += f' {self.confidence}'
        return s

    @classmethod
    def from_dict(cls, segment: 'SegLstSegment') -> 'Self':
        # CTM only supports words as segments.
        # If this check fails, the input data was not converted to words before.
        assert ' ' not in segment['words'], segment
        return cls(
            filename=segment['session_id'],
            channel=segment['channel'],
            begin_time=segment['start_time'],
            duration=segment['end_time'] - segment['start_time'],
            word=segment['words'],
            confidence=segment.get('confidence', None),
        )

    def to_seglst(self) -> 'SegLstSegment':
        d = {
            'session_id': self.filename,
            'channel': self.channel,
            'start_time': self.begin_time,
            'end_time': self.begin_time + self.duration,
            'words': self.word,
        }
        if self.confidence is not None:
            d['confidence'] = self.confidence
        return d


@dataclass(frozen=True)
class CTM(Base):
    lines: List[CTMLine]
    line_cls = CTMLine

    @classmethod
    def parse(cls, s: str, parse_float=decimal.Decimal) -> 'Self':
        return cls([
            CTMLine.parse(line, parse_float=parse_float)
            for line in map(str.strip, s.split('\n'))
            if len(line) > 0
            if not line.startswith(';;')
        ])

    def merged_transcripts(self) -> str:
        return ' '.join([x.word for x in sorted(self.lines, key=lambda x: x.begin_time)])

    def utterance_transcripts(self) -> List[str]:
        """There is no notion of an "utterance" in CTM files."""
        raise NotImplementedError()

    @classmethod
    def new(cls, s, **defaults) -> 'Self':
        # CTM only supports a single speaker. Use CTMGroup to represent multiple speakers with this format.
        if len(s.unique('speaker')) > 1:
            raise ValueError(
                f'CTM only supports a single speaker, but found {len(s.unique("speaker"))} speakers '
                f'({s.unique("speaker")}). Use CTMGroup to represent multiple speakers with this format.'
            )
        return super().new(s, **defaults)


@dataclass(frozen=True)
class CTMGroup(BaseABC):
    ctms: 'Dict[str, CTM]'

    @classmethod
    def load(cls, ctm_files, parse_float=decimal.Decimal):
        return cls({str(ctm_file): CTM.load(ctm_file, parse_float=parse_float)
                    for ctm_file in ctm_files})

    def grouped_by_filename(self) -> Dict[str, 'CTMGroup']:
        groups = {
            k: ctm.grouped_by_filename() for k, ctm in self.ctms.items()
        }
        # keys = next(iter(groups.values())).keys()
        keys = dict.fromkeys([
            k
            for v in groups.values()
            for k in v.keys()
        ])

        for file, group in groups.items():
            if set(group.keys()) != set(keys) and len(group.keys()) <= len(keys) / 2:
                # CB: We may remove this warning in the future, once we find a
                # case, where it printed, while it is obviously not relevant.
                # It might happen, when a diraization system is used, where
                # the number of speakers is not known in advance/or different
                # between different recording.
                warnings.warn(
                    'Example IDs do not match across CTM files!\n'
                    f'{file} has {len(group.keys())}, while it should have {len(keys)}.\n'
                    f'Some missing example IDs: {list(set(keys) - set(group.keys()))[:5]}\n'
                    f'While this can happen (no estimate on a stream), it could also be a bug.\n'
                )

        return {
            key: CTMGroup({
                k: g.get(key, CTM([])) for k, g in groups.items()
            })
            for key in keys
        }

    def grouped_by_speaker_id(self) -> Dict[str, CTM]:
        return self.ctms

    @classmethod
    def new(cls, s: 'SegLST', **defaults) -> 'Self':
        from meeteval.io.seglst import asseglst
        return cls({k: CTM.new(v) for k, v in asseglst(s).map(lambda s: {**defaults, **s}).groupby('speaker').items()})

    def to_seglst(self) -> 'SegLST':
        from meeteval.io.seglst import SegLST
        return SegLST.merge(*[ctm.to_seglst().map(lambda x: {**x, 'speaker': speaker}) for speaker, ctm in self.ctms.items()])

    def to_stm(self):
        from meeteval.io import STM, STMLine
        stm = []
        for filename, v1 in self.grouped_by_filename().items():
            for speaker_id, v2 in v1.grouped_by_speaker_id().items():
                for line in v2.lines:
                    stm.append(STMLine(
                        filename=filename,
                        # The channel is usually ignored, but https://github.com/nryant/dscore assumes
                        # 1 as the default for the channel in RTTM files
                        channel=1,
                        speaker_id=speaker_id,
                        begin_time=line.begin_time,
                        end_time=line.duration,
                        transcript=line.word,
                    ))

        return STM(stm)


if __name__ == '__main__':
    def to_stm(*files):
        from pathlib import Path
        from meeteval.io.stm import STM, STMLine
        import decimal

        ctm_files = []
        stm_file = None
        for i, file in enumerate(files):
            file = Path(file)
            if file.suffix == '.ctm':
                ctm_files.append(file)
            elif file.suffix == '.stm':
                assert i == len(files) - 1, 'STM file must be last'
                stm_file = file

        if stm_file is None:
            stm_file = '-'
        else:
            assert not Path(stm_file).exists(), stm_file

        ctm_group = CTMGroup.load(ctm_files, parse_float=decimal.Decimal)
        stm = ctm_group.to_stm()
        stm.dump(stm_file)
        print(f'Wrote {len(stm)} lines to {stm_file}')

    import fire
    fire.Fire({
        'to_stm': to_stm,
    })