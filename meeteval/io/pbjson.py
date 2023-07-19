"""


python -m meeteval.io.pbjson to_stm /scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css.json .

"""

import json
from pathlib import Path


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
                    begin_time=begin_time/sample_rate, end_time=end_time/sample_rate,
                    transcript=transcript))

            for spk, o, n, t in zip_strict(
                    ex['speaker_id'],
                    ex['offset'],
                    ex['num_samples']['original_source'],
                    ex.get('kaldi_transcription') or ex['transcription']
            ):
                if isinstance(t, (tuple, list)):
                    for o_, n_, t_ in zip_strict(o, n, t):
                        add_line(spk, o_, o_+n_, t_)
                else:
                    add_line(spk, o, o+n, t)

        file = out / f'{dataset_name}_ref.stm'
        STM(stm_lines).dump(file)
        print(f'Wrote {file}')


if __name__ == '__main__':
    import fire
    fire.Fire({
        'to_stm': to_stm,
    })

