"""
Deliver normalized audio files.

python -m meeteval.viz.file_server
"""

from aiohttp import web
from pathlib import Path
import io
import subprocess

import soundfile
import numpy as np


class Backend:
    def __init__(self):
        pass

    async def handle(self, request: web.Request):
        print('request:', request)
        name = request.match_info.get('name', "Anonymous")
        name = '/' + name
        # assert name.startswith('/net/'), name
        name = Path(name)
        if name.suffix in ['.wav']:
            try:
                data, sample_rate = soundfile.read(str(name))
            except Exception:
                import traceback
                traceback.print_exc()
            else:
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
