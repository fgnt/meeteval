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
    $ python -m meeteval.io.chime7 to_rttm transcriptions_scoring/dev/*.json --uem=uem/dev/all.uem > transcriptions_scoring/dev.rttm

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
import json

import numpy as np

from meeteval.io.stm import STM, STMLine
from meeteval.io.uem import UEM, UEMLine
from meeteval.io.rttm import RTTM, RTTMLine


def _load_json(file, parse_float=decimal.Decimal):
    with open(file) as fd:
        return fix_json(json.load(fd, parse_float=parse_float))


def json_to_stm(json_content, filename):
    return STM([
        STMLine(
            filename=entry.get('session_id', filename),
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
            filename=entry.get('session_id') or filename,
            channel='1',
            speaker_id=entry['speaker'],
            begin_time=decimal.Decimal(entry['start_time']),
            duration=decimal.Decimal(entry['end_time']) - decimal.Decimal(entry['start_time']),
        )
        for entry in json_content
    ])


def fix_json(json_content):
    if isinstance(json_content, dict):
        json_content = list(json_content.values())
    for entry in json_content:
        try:
            entry['speaker']
        except KeyError:
            entry['speaker'] = entry['speaker_id']

            try:
                entry['start_time']
            except KeyError:
                entry['start_time'] = entry['start_second']

            try:
                entry['end_time']
            except KeyError:
                entry['end_time'] = entry['stop_second']

            try:
                entry['session_id']
            except KeyError:
                entry['session_id'] = entry['subset']

    return json_content


def to_stm(*jsons, uem=None, file=sys.stdout):
    if uem is not None:
        uem = UEM.load(uem, parse_float=decimal.Decimal)

    for json in jsons:
        json = Path(json)
        stm = json_to_stm(_load_json(json), json.stem)

        if uem is not None:
            stm = stm.filter_by_uem(uem, verbose=True)

        file.write(stm.dumps())


def to_rttm(*jsons, uem=None, file=sys.stdout):
    try:
        if uem is not None:
            uem = UEM.load(uem)

        for json in jsons:
            json = Path(json)
            rttm = json_to_rttm(_load_json(json), json.with_suffix('').name)

            if uem is not None:
                rttm = rttm.filter_by_uem(uem, verbose=True)

            file.write(rttm.dumps())
    except BrokenPipeError:
        if file != sys.stdout:
            raise


def fix_rttm(*rttms, force: bool = False, dry: bool = False):
    """
    Some baseline scripts produce rttm files, where the "filename"/recording ID
    is encoded in the rttm filename and inside the file the field is set to
    "<NA>". This causes two issues:
     - Files cannot be concatenated.
     - Tools like dscore consider only the field for the "filename" and ignore
       the rttm filename, hence they don't work.

    This function takes those buggy files as input and overwrites them with
    a fixed filename field.

    python -m meeteval.io.chime7 fix_rttm S0*.rttm --force --dry

    """
    import json

    # json.load(parse_float=)
    # cd egs2/chime7_task1/diar_asr1/exp/diarization/chime6/dev
    # python -m meeteval.io.chime7 fix_rttm *.rttm
    for rttm_file in rttms:
        rttm_file = Path(rttm_file)
        rttm = RTTM.load(rttm_file, parse_float=decimal.Decimal)
        rttm = rttm.grouped_by_filename()
        assert len(rttm) == 1, (rttm.keys())
        old, = rttm.keys()
        new = rttm_file.stem
        if old == new:
            continue
        if force:
            print(f'Change {old} to {rttm_file.stem} for rttm_file')
        else:
            assert list(rttm.keys()) == ['<NA>'], ('filenames:', rttm.keys(), 'use --force to ignore this error')

        rttm = RTTM([line.replace(filename=rttm_file.stem) for line in rttm[old]])
        if dry:
            print('Would write:')
            print(*rttm.dumps().split('\n')[:10], '...', sep='\n')
        else:
            rttm.dump(rttm_file)


def add_missing(*jsons, uem=None):
    if uem is not None:
        uem = UEM.load(uem)

    for json in jsons:
        json = Path(json)
        content = _load_json(json)

        rttm_file = json.with_suffix('.rttm')
        if not rttm_file.exists():
            rttm = json_to_rttm(content, json.with_suffix('').name)
            if uem is not None:
                rttm = rttm.filter_by_uem(uem, verbose=True)
            rttm_file.write_text(rttm.dumps())
            print(f'Wrote {rttm_file}')

        stm_file = json.with_suffix('.stm')
        if not stm_file.exists():
            try:
                stm = json_to_stm(content, json.with_suffix('').name)
            except KeyError as e:
                print(f'WARNING: Could not convert to stm, skipping {json}')
                print(e)
                continue
            if uem is not None:
                stm = stm.filter_by_uem(uem, verbose=True)
            stm_file.write_text(stm.dumps())
            print(f'Wrote {stm_file}')


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
        if uemdir is not None:
            uem = uemdir / datasetdir.name / 'all.uem'
            if not uem.exists():
                uem = None

        stm = datasetdir.with_suffix(suffix)
        jsons = [str(json) for json in datasetdir.glob('*.json')]
        if not jsons:
            # if datasetdir.name == 'eval':
            #     print(f'Skip {datasetdir.name} it is empty.')
            #     continue
            print(f'Ignore {datasetdir}, because it contains no jsons.')
            continue

        to_X = {'.stm': to_stm, '.rttm': to_rttm}[suffix]

        with open(stm, 'w') as fd:
            to_X(*jsons, uem=uem, file=fd)

        prefix = os.path.commonprefix(jsons)
        postfix = os.path.commonprefix([str(j)[::-1] for j in jsons])[::-1]
        middle = [j[len(prefix):len(j)-len(postfix)] for j in jsons]
        middle = ','.join(middle)
        print(f'Created {stm} from {prefix}[{middle}]{postfix})')


def stats(json):
    json = _load_json(json)

    print(f'Number of files: {len(json)}')
    print(f'Number of unique speakers: {len(set(x["speaker"] for x in json))}')
    print(f'Number of unique session_ids: {len(set(x["session_id"] for x in json))}')
    durations = list(decimal.Decimal(x["end_time"]) - decimal.Decimal(x["start_time"]) for x in json)
    print(f'Max duration: {max(durations)}')
    print(f'Min duration: {min(durations)}')
    print(f'Mean duration: {np.mean(durations)}')
    print(f'Median duration: {np.median(durations)}')


if __name__ == '__main__':
    import functools
    import fire
    fire.Fire({
        'to_stm': to_stm,
        'to_rttm': to_rttm,
        'dir_to_stms': functools.partial(dir_to, suffix='.stm'),
        'dir_to_rttms': functools.partial(dir_to, suffix='.rttm'),
        'fix_rttm': fix_rttm,
        'stats': stats,
        'add_missing': add_missing,
    })
