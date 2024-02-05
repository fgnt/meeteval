"""
A file server that exposes wav files from the filesystem:
 - Only delivers wav files
    - Support slices (int -> samples, float -> seconds)
        - server_url/file.wav?start=0&stop=16000
        - server_url/file.wav::[0:16000]
 - Normalizes the audio files (Useful for far-field recordings that typically have a low volume)
 - Supports slicing

python -m meeteval.viz.file_server
"""

import io
import functools
from pathlib import Path
from aiohttp import web

import numpy as np
import soundfile


def _parse_audio_slice(path, start, stop):
    """
    See from paderbox.io.audioread.py::_parse_audio_slice for advanced parser.

    >>> _parse_audio_slice('file.wav::[1:2]', None, None)
    ('file.wav', 1, 2)
    >>> _parse_audio_slice('file.wav', '1', '2')
    ('file.wav', 1, 2)
    >>> _parse_audio_slice('file.wav::[1:10]', 2, 4)
    ('file.wav', 3, 5)
    """
    @functools.lru_cache()
    def samplerate(path):
        return soundfile.info(path).samplerate

    _last_type = None
    def to_number(*numbers):
        nonlocal _last_type
        for num in numbers:
            if num is None:
                yield num
                continue
            if isinstance(num, str):
                if '.' in num:
                    num = round(float(num) * samplerate(path))
                    cur_type = float
                else:
                    num = int(num)
                    cur_type = int
            else:
                cur_type = type(num)

            if _last_type is None:
                _last_type = cur_type
            else:
                assert _last_type == cur_type, ('All numbers must have a common type (flot or int)', path, start, stop, _last_type, cur_type)
            yield num

    start, stop = to_number(start, stop)

    if '::' in path:
        path, slice = path.split('::')
        assert slice[0] == '[' and slice[-1] == ']', slice
        start_, stop_ = to_number(*slice[1:-1].split(':'))
        if start is None and stop is None:
            start, stop = start_, stop_
        else:
            # Arguments are applied on the [...:...]
            start, stop = start + start_, stop + start_

    if _last_type is int:
        assert (stop - start) < 120, ('For stability: Limit the max duration to 2 min', start, stop, samplerate(path))
    if _last_type is float:
        assert (stop - start) < samplerate(path) * 120, ('For stability: Limit the max duration to 2 min', start, stop, samplerate(path))

    return path, start, stop


class Backend:
    def __init__(self):
        pass

    async def handle(self, request: web.Request):
        try:
            name = request.match_info.get('name', "Anonymous")
            name = '/' + name

            start = request.query.get('start', None)
            stop = request.query.get('stop', None)

            name, start, stop = _parse_audio_slice(name, start, stop)
            name = Path(name)

            if name.suffix in ['.wav']:
                data, sample_rate = soundfile.read(str(name), start=start, stop=stop)
                data = data * 0.95 / np.amax(np.abs(data))
                bytes = io.BytesIO()
                # Wav files produce in some browsers artefacts in the audio.
                # This is undesired, because it is not clear, that it is caused
                # by the browser or by the enhancement system of the user.
                # Ogg doesn't has this issue.
                # ToDo: Does ogg remove user artefacts?
                #       If yes, use another codec. Or find a reliable way to
                #       use wav files in all browsers.
                soundfile.write(bytes, data, samplerate=sample_rate, format='ogg')
                return web.Response(body=bytes.getvalue())
                # cp = subprocess.run([
                #     'sox', '--norm', str(name), '-t', 'ogg', '-'
                # ], stdout=subprocess.PIPE, check=True)
                # return web.Response(body=cp.stdout)
        except Exception:
            import traceback
            traceback.print_exc()
        return web.Response(status=web.HTTPUnauthorized().status_code)

    def main(self):
        app = web.Application()
        app.add_routes([web.get('/', self.handle),
                        web.get('/{name:.*}', self.handle),
                        ])
        web.run_app(app, port=7777)


if __name__ == '__main__':
    b = Backend()
    b.main()
