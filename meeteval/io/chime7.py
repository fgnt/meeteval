"""
The chime7 challenge uses <session_id>.json as filenames and the content
is a list, where the entries describe a segment, e.g.:

```
{
  "end_time": "11.370",
  "start_time": "11.000",
  "words": "so ummm",
  "speaker": "P03",
  "session_id": "S05"
}
```

To make it usable by meeteval, we provide code, that converts the chime7
specific format to standardized formats, i.e. to_stm and to_rttm.

Example call:

    python -m meeteval.io.chime7 to_stm transcriptions_scoring/dev/*.json --uem=uem/dev/all.uem > transcriptions_scoring/dev.stm

"""
import sys
from pathlib import Path
import paderbox as pb
from meeteval.io.stm import STM, STMLine
from meeteval.io.uem import UEM, UEMLine
from meeteval.io.rttm import RTTM, RTTMLine


def json_to_stm(json_content, filename):
    return STM([
        STMLine(
            filename=filename,
            channel='1',
            speaker_id=entry['speaker'],
            begin_time=float(entry['start_time']),
            end_time=float(entry['end_time']),
            transcript=entry['words'],
        )
        for entry in json_content
    ])


def json_to_rttm(json_content, filename):
    return RTTM([
        RTTMLine(
            filename=filename,
            channel='1',
            speaker_id=entry['speaker'],
            begin_time=float(entry['start_time']),
            duration_time=float(entry['end_time']) - float(entry['start_time']),
        )
        for entry in json_content
    ])


def to_stm(*jsons, uem=None):
    if uem is not None:
        uem = UEM.load(uem)

    for json in jsons:
        json = Path(json)
        stm = json_to_stm(pb.io.load(json), json.with_suffix('').name)

        if uem is not None:
            stm = stm.filter_by_uem(uem, verbose=True)

        sys.stdout.write(stm.dumps())


def to_rttm(*jsons, uem=None):
    if uem is not None:
        uem = UEM.load(uem)

    for json in jsons:
        json = Path(json)
        rttm = json_to_rttm(pb.io.load(json), json.with_suffix('').name)

        if uem is not None:
            rttm = rttm.filter_by_uem(uem, verbose=True)

        sys.stdout.write(rttm.dumps())


if __name__ == '__main__':
    import fire
    fire.Fire({
        'to_stm': to_stm,
        'to_rttm': to_rttm,
    })
