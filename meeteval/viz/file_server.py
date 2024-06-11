"""
A file server that exposes wav files from the filesystem:
 - Only delivers wav files
    - Support slices (int -> samples, float -> seconds)
        - server_url/file.wav?start=0&stop=16000
        - server_url/file.wav::[0:16_000]
        - server_url/file.wav::[0:160_000]?start=0&stop=16000
 - Normalizes the audio files (Useful for far-field recordings that typically have a low volume)

python -m meeteval.viz.file_server
"""

import io
import functools
import os
import sys
from pathlib import Path
from aiohttp import web

import numpy as np
import soundfile


def _parse_audio_slice(path, start, stop):
    """
    See from paderbox.io.audioread.py::_parse_audio_slice for advanced parser.

    Note: soundfile has only samples (they call it frames) as unit and no
          support for seconds. Hence, convert floats that are typically seconds
          to ints with sample resolution.

    >>> from unittest import mock
    >>> with mock.patch('soundfile.info', mock.MagicMock()) as patch:
    ...     patch.return_value.samplerate = 16000
    ...     _parse_audio_slice('file.wav::[1.0:2.0]', None, None)
    ('file.wav', 16000, 32000)
    >>> with mock.patch('soundfile.info', mock.MagicMock()) as patch:
    ...     patch.return_value.samplerate = 16000
    ...     _parse_audio_slice('file.wav', '3.', '5.')
    ('file.wav', 48000, 80000)
    >>> with mock.patch('soundfile.info', mock.MagicMock()) as patch:
    ...     patch.return_value.samplerate = 16000
    ...     _parse_audio_slice('file.wav::[1.:10.]', 2., 4.)
    ('file.wav', 48000, 80000)
    """
    def to_number(*numbers):
        for num in numbers:
            if num is None or isinstance(num, int):
                pass
            elif isinstance(num, float):
                num = round(num * samplerate)
            elif isinstance(num, str):
                num = round(float(num) * samplerate) if '.' in num else int(num)
            else:
                raise TypeError(type(num), num)
            yield num

    def get_sample_rate(path):
        try:
            return soundfile.info(os.fspath(path)).samplerate
        except RuntimeError:
            if Path(path).exists():
                raise
            else:
                raise FileNotFoundError(path) from None

    if '::' in path:
        path, slice = path.split('::')
        samplerate = get_sample_rate(path)
        start, stop = to_number(start, stop)
        assert slice[0] == '[' and slice[-1] == ']', slice
        start_, stop_ = to_number(*slice[1:-1].split(':'))
        if start is None and stop is None:
            start, stop = start_, stop_
        else:
            # Arguments are applied on the [...:...]
            start, stop = start + start_, stop + start_
    else:
        samplerate = get_sample_rate(path)
        start, stop = to_number(start, stop)

    if start is not None and start is not None:
        assert (stop - start) < samplerate * 120, ('For stability: Limit the max duration to 2 min', start, stop, samplerate)
    return path, start, stop


class Backend:
    def __init__(self):
        pass

    async def handle(self, request: web.Request):
        try:
            try:
                name = request.match_info['name']
            except KeyError:
                body = await request.read()
                print('\n'.join([
                    f"No name in request:",
                    f" | Method: {request.method}",
                    f" | URL: {request.url}",
                    f" | Headers: {request.headers}",
                    f" | Match info: {request.match_info}",
                    f" | Body: {body}",
                ]))
                return web.Response(status=web.HTTPUnauthorized().status_code)
            name = '/' + name
            print(f'Requested: {name}')

            start = request.query.get('start', None)
            stop = request.query.get('stop', None)

            name, start, stop = _parse_audio_slice(name, start, stop)
            name = Path(name)

            if name.suffix in ['.wav', '.flac']:
                try:
                    data, sample_rate = soundfile.read(str(name), start=start, stop=stop)
                except RuntimeError:
                    if Path(name).exists():
                        raise
                    else:
                        raise FileNotFoundError(name) from None
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
            import traceback, textwrap
            print(textwrap.indent(
                traceback.format_exc(), ' | ',
            ))
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
