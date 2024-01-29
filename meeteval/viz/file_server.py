"""
A file server, that exposes wav files from the filesystem:
 - Only delivers wav files
 - Normalizes the audio files (Useful for far files, that is typically low volumn)
 - Supports slicing

python -m meeteval.viz.file_server
"""

import io
from pathlib import Path
from aiohttp import web

import numpy as np
import soundfile


def _parse_audio_slice(path):
    """
    See from paderbox.io.audioread.py::_parse_audio_slice for advanced parser.

    >>> _parse_audio_slice('file.wav::[1:2]')
    ('file.wav', 1, 2)
    """
    start = stop = None
    if '::' in path:
        path, slice = path.split('::')
        assert slice[0] == '[' and slice[-1] == ']', slice
        start, stop = slice[1:-1].split(':')
        if '.' in start or '.' in stop:
            samplerate = soundfile.info(path).samplerate
            start = round(float(start) * samplerate)
            stop = round(float(stop) * samplerate)
        else:
            start, stop = int(start), int(stop)
    return path, start, stop


class Backend:
    def __init__(self):
        pass

    async def handle(self, request: web.Request):
        try:
            print('request:', request)
            name = request.match_info.get('name', "Anonymous")
            name = '/' + name
            # assert name.startswith('/net/'), name
            name, start, stop = _parse_audio_slice(name)
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
