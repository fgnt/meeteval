"""


python -m meeteval.io.pbjson to_stm /scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css.json .

"""

import json
import typing
from pathlib import Path

from meeteval.io.base import BaseABC

if typing.TYPE_CHECKING:
    from meeteval.io.seglst import SegLST


def _load_json(file):
    with open(file) as fd:
        return json.load(fd)


from meeteval.io.stm import STM, STMLine


def zip_strict(*iterables):
    assert len(set(map(len, iterables))) == 1, list(map(len, iterables))
    yield from zip(*iterables)


def get_sample_rate(ex):
    import soundfile
    observation = ex['audio_path']
    if isinstance(observation, dict):
        observation = observation['observation']
    while True:
        if isinstance(observation, dict):
            observation = next(iter(observation.values()))
        elif isinstance(observation, (tuple, list)):
            observation = next(iter(observation))
        else:
            assert isinstance(observation, str), (type(observation), observation)
            break

    sample_rate = soundfile.info(observation).samplerate
    return sample_rate


class PBJsonUtt(BaseABC):
    """
    A JSON format where each entry/example represents a single utterance.

    Example:
    ```python
    pbjson = {
        'datasets': {
            '<dataset_name>': {
                '<example_id>': {
                    'num_samples': {
                        'original_source': <int>,   # in samples
                    },
                    'offset': <int>,  # in samples
                    'speaker_id': <str>,
                    'transcription': <str>,
                    'kaldi_transcription': <str>, # optional
                }
            }
        }
    }
    ```
    """

    def __init__(self, json, sample_rate=16000):
        self.json = json
        self.sample_rate = sample_rate

    @classmethod
    def load(cls, file):
        return cls(_load_json(file))

    @classmethod
    def new(cls, s, *, sample_rate=16000, dataset_name='default_dataset', **defaults):
        # This copies the segments (since we are going to `pop` keys later), applies defaults and makes sure
        # that the dataset_name key is set for all segments.
        from meeteval.io.seglst import asseglst
        s = asseglst(s).map(lambda x: {**defaults, **x, 'dataset_name': x['dataset_name'] or dataset_name})
        return cls({
            'datasets': {
                dataset_name: {
                    example_id: {
                        # Translate structure from SegLST to pbjson
                        'num_samples': {
                            'original_source': (example.pop('end_time') - example.pop('start_time')) * sample_rate,
                        },
                        'offset': example.pop('start_time') * sample_rate,
                        'speaker_id': example.pop('speaker_id'),
                        'transcription': example.pop('words'),
                        # Any additional keys are simply appended to the json
                        **example,
                    }
                    for example_id, example in dataset.groupby('example_id')
                }
                for dataset_name, dataset in s.groupby('dataset_name')
            }
        })

    def to_seglst(self) -> 'SegLST':
        return SegLST([
            {
                # Translate known keys
                'dataset_name': dataset_name,
                'example_id': example_id,
                'words': example.get('kaldi_transcription') or example['transcription'],
                'speaker': example['speaker_id'],
                'start_time': example['offset'] / self.sample_rate,
                'end_time': (example['offset'] + example['num_samples']['original_source']) / self.sample_rate,
                # Any other keys are appended
                **{
                    k: v
                    for k, v in example.items() if k not in {
                        'offset', 'num_samples', 'speaker_id', 'transcription',
                        'kaldi_transcription',
                    }
                },
            }
            for dataset_name, dataset in self.json['datasets'].items()
            for example_id, example in dataset.items()
        ])


def to_stm(json, out, datasets=None):
    import lazy_dataset.database
    out = Path(out)

    db = lazy_dataset.database.JsonDatabase(json)
    if datasets is None:
        datasets = db.dataset_names
    elif isinstance(datasets, str):
        datasets = [datasets]

    sample_rate = None

    for dataset_name in datasets:
        print(f'Processing {dataset_name}')
        stm_lines = []

        try:
            ds = db.get_dataset(dataset_name)
        except AssertionError:
            print(f'WARNING: Issue with {dataset_name}. Skip it.')

        for ex in ds:
            if sample_rate is None:
                sample_rate = get_sample_rate(ex)

            def add_line(speaker_id, begin_time, end_time, transcript):
                stm_lines.append(STMLine(
                    filename=ex['example_id'], channel=0,
                    speaker_id=speaker_id,
                    begin_time=begin_time / sample_rate, end_time=end_time / sample_rate,
                    transcript=transcript))

            for spk, o, n, t in zip_strict(
                    ex['speaker_id'],
                    ex['offset'],
                    ex['num_samples']['original_source'],
                    ex.get('kaldi_transcription') or ex['transcription']
            ):
                if isinstance(t, (tuple, list)):
                    for o_, n_, t_ in zip_strict(o, n, t):
                        add_line(spk, o_, o_ + n_, t_)
                else:
                    add_line(spk, o, o + n, t)

        file = out / f'{dataset_name}_ref.stm'
        STM(stm_lines).dump(file)
        print(f'Wrote {file}')


if __name__ == '__main__':
    import fire

    fire.Fire({
        'to_stm': to_stm,
    })
