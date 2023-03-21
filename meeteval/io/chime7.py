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

    $ python -m meeteval.io.chime7 to_stm transcriptions_scoring/dev/*.json --uem=uem/dev/all.uem > transcriptions_scoring/dev.stm
    $ python -m meeteval.io.chime7 to_rttm transcriptions_scoring/dev/*.rttm --uem=uem/dev/all.uem > transcriptions_scoring/dev.rttm

When you want to convert all datasets for a chime6, dipco or mixer6, the call
would be

    python -m meeteval.io.chime7 dir_to_stms transcriptions_scoring --uemdir=uem
    python -m meeteval.io.chime7 dir_to_rttms transcriptions_scoring --uemdir=uem

where the script would globs for the files
    transcriptions_scoring/<dataset>/<session>.json
and
    uem/<dataset>/all.json


"""
import os
import sys
import decimal
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
            begin_time=decimal.Decimal(entry['start_time']),
            end_time=decimal.Decimal(entry['end_time']),
            transcript=entry['words'],
        )
        for entry in json_content
    ])


def json_to_rttm(json_content, filename):
    # Use Decimal, since float has rounding issues.
    return RTTM([
        RTTMLine(
            filename=filename,
            channel='1',
            speaker_id=entry['speaker'],
            begin_time=decimal.Decimal(entry['start_time']),
            duration_time=decimal.Decimal(entry['end_time']) - decimal.Decimal(entry['start_time']),
        )
        for entry in json_content
    ])


def to_stm(*jsons, uem=None, file=sys.stdout):
    if uem is not None:
        uem = UEM.load(uem)

    for json in jsons:
        json = Path(json)
        stm = json_to_stm(pb.io.load(json), json.with_suffix('').name)

        if uem is not None:
            stm = stm.filter_by_uem(uem, verbose=True)

        file.write(stm.dumps())


def to_rttm(*jsons, uem=None, file=sys.stdout):
    if uem is not None:
        uem = UEM.load(uem)

    for json in jsons:
        json = Path(json)
        rttm = json_to_rttm(pb.io.load(json), json.with_suffix('').name)

        if uem is not None:
            rttm = rttm.filter_by_uem(uem, verbose=True)

        file.write(rttm.dumps())


def dir_to(jsondir, uemdir=None, suffix='.stm'):
    assert suffix in ['.stm', '.rttm'], suffix
    jsondir = Path(jsondir)
    if uemdir is not None:
        uemdir = Path(uemdir)
        if not list(uemdir.glob('*/all.uem')):
            raise RuntimeError(
                f'Could not find any uem files with {uemdir}/*/all.uem')
    uem = None

    for datasetdir in jsondir.glob('*'):
        if datasetdir.is_file():
            continue
        datasetdir = Path(datasetdir)
        if uemdir is not None:
            uem = uemdir / datasetdir.name / 'all.uem'
            if not uem.exists():
                uem = None

        stm = datasetdir.with_suffix(suffix)
        jsons = [str(json) for json in datasetdir.glob('*.json')]
        if not jsons:
            if datasetdir.name == 'eval':
                print(f'Skip {datasetdir.name} it is empty.')
                continue
            assert jsons, (jsons, datasetdir)

        to_X = {'.stm': to_stm, '.rttm': to_rttm}[suffix]

        with open(stm, 'w') as fd:
            to_X(*jsons, uem=uem, file=fd)

        prefix = os.path.commonprefix(jsons)
        postfix = os.path.commonprefix([str(j)[::-1] for j in jsons])[::-1]
        middle = [j[len(prefix):len(j)-len(postfix)] for j in jsons]
        middle = ','.join(middle)
        print(f'Created {stm} from {prefix}[{middle}]{postfix})')


if __name__ == '__main__':
    import functools
    import fire
    fire.Fire({
        'to_stm': to_stm,
        'to_rttm': to_rttm,
        'dir_to_stms': functools.partial(dir_to, suffix='.stm'),
        'dir_to_rttms': functools.partial(dir_to, suffix='.rttm'),
    })
