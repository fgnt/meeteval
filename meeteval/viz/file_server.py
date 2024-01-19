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
            # data, sample_rate = soundfile.read(str(name))
            # data = data * 0.50 / np.amax(np.abs(data))
            # bytes = io.BytesIO()
            # bytes.name = str(name)
            # soundfile.write(bytes, data, samplerate=sample_rate)
            # return web.Response(body=bytes.getvalue())

            cp = subprocess.run([
                'sox', '--norm', str(name), '-t', 'ogg', '-'
            ], stdout=subprocess.PIPE, check=True)
            return web.Response(body=cp.stdout)
        else:
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
