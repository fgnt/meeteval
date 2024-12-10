import logging
import re
import decimal
import shutil
import tempfile
import subprocess
import dataclasses
from pathlib import Path

import meeteval.io
from meeteval.wer.wer.error_rate import ErrorRate


def _fix_channel(r):
    return meeteval.io.rttm.RTTM([
        line.replace(channel='1')
        # Thilo puts there usually some random value, e.g. <NA> for hyp and 0
        # for ref, while dscore enforces the default to be 1
        for line in r
    ])


@dataclasses.dataclass(frozen=True)
class DiaErrorRate:
    """

    """
    error_rate: 'float | decimal.Decimal'

    scored_speaker_time: 'float | decimal.Decimal'
    missed_speaker_time: 'float | decimal.Decimal'  # deletions
    falarm_speaker_time: 'float | decimal.Decimal'  # insertions
    speaker_error_time: 'float | decimal.Decimal'  # substitutions

    @classmethod
    def zero(cls):
        return cls(0, 0, 0, 0, 0)

    def __post_init__(self):
        assert self.scored_speaker_time >= 0
        assert self.missed_speaker_time >= 0
        assert self.falarm_speaker_time >= 0
        assert self.speaker_error_time >= 0
        errors = self.speaker_error_time + self.falarm_speaker_time + self.missed_speaker_time
        error_rate = errors / self.scored_speaker_time
        if self.error_rate is None:
            object.__setattr__(self, 'error_rate', error_rate)
        else:
            # Since md-eval uses float internally, and the printed numbers are
            # rounded, it is in corner cases not possible to reproduce the
            # exact error rate, that is calculated internally by md-eval.
            # Therefore, the last digit may change. Here, an example, where it
            # happened:
            #   md-eval on all data:
            #       0.1813
            #   md-eval on each recording and then calculated:
            #       0.1812499845344880915558304980
            # Hence, we allow a small difference.
            assert abs(self.error_rate - error_rate) < 0.00007, (error_rate, self)

    def __radd__(self, other: 'int') -> 'ErrorRate':
        if isinstance(other, int) and other == 0:
            # Special case to support sum.
            return self
        return NotImplemented

    def __add__(self, other: 'DiaErrorRate'):
        if not isinstance(other, self.__class__):
            raise ValueError()

        return DiaErrorRate(
            error_rate=None,
            scored_speaker_time=self.scored_speaker_time + other.scored_speaker_time,
            missed_speaker_time=self.missed_speaker_time + other.missed_speaker_time,
            falarm_speaker_time=self.falarm_speaker_time + other.falarm_speaker_time,
            speaker_error_time=self.speaker_error_time + other.speaker_error_time,
        )


class _FilenameEscaper:
    """
    >>> import pprint
    >>> reference = meeteval.io.RTTM.parse('''
    ... SPEAKER rec.a 1 5.00 5.00 <NA> <NA> spk01 <NA>
    ... SPEAKER rec.a 1 10.00 10.00 <NA> <NA> spk00 <NA>
    ... ''')
    >>> hypothesis = meeteval.io.RTTM.parse('''
    ... SPEAKER rec.a 1 0.00 10.00 <NA> <NA> spk01 <NA>
    ... SPEAKER rec.a 1 10.00 10.00 <NA> <NA> spk00 <NA>
    ... ''')
    >>> uem = meeteval.io.UEM.parse('''
    ... rec.a 1 0.00 15.00
    ... ''')

    >>> pprint.pprint(md_eval_22_multifile(reference, hypothesis, uem=uem))  # doctest: +NORMALIZE_WHITESPACE
    {'rec.a': DiaErrorRate(error_rate=Decimal('0.50'),
                           scored_speaker_time=Decimal('10.000000'),
                           missed_speaker_time=Decimal('0.000000'),
                           falarm_speaker_time=Decimal('5.000000'),
                           speaker_error_time=Decimal('0.000000'))}

    >>> _FilenameEscaper._DISABLED = True
    >>> pprint.pprint(md_eval_22_multifile(reference, hypothesis, uem=uem))  # doctest: +NORMALIZE_WHITESPACE
    {'rec.a': DiaErrorRate(error_rate=Decimal('0.00'),
                           scored_speaker_time=Decimal('15.000000'),
                           missed_speaker_time=Decimal('0.000000'),
                           falarm_speaker_time=Decimal('0.000000'),
                           speaker_error_time=Decimal('0.000000'))}
    >>> _FilenameEscaper._DISABLED = False
    """
    _DISABLED = False
    def __init__(self):
        self.warned = False
        self.cache = {}

    def __call__(self, filename):
        if self._DISABLED:
            return filename
        if filename in self.cache:
            return self.cache[filename]
        elif '.' in filename:
            blocked = set(self.cache.values())
            for replacement in [
                '_', '__', '___', '____', '_____',
            ]:
                new = filename.replace('.', replacement)
                if new not in blocked:
                    break
            else:
                raise RuntimeError(f'Cannot find a replacement for {filename}.')

            if not self.warned:
                self.warned = True
                logging.warning(
                    f'Warning: Replace UEM filename "{filename}" by "{new}".\n'
                    f'         md-eval-22 removes the first suffix for uem filenames/session_ids but not in rttm files\n'
                    f'             (e.g., uem: some.audio.wav -> some.wav).\n'
                    f'             (e.g., rttm: some.audio.wav -> some.audio.wav).\n'
                    f"         dcores doesn't support dots in uem\n"
                    f'             (e.g., without uem file, rttm filenames/session_ids can have dots).\n'
                    f'             (e.g., with uem file, rttm cannot have dots).\n'
                    f"          -> dcores has no proper support of dots, because they use md-eval-22\n"
                    f'         In meeteval, we assume, that the uem file has the same filenames/session_ids as reference and hypothesis.\n'
                    f'          -> remove dots from filename and restore them later'
                )
            self.cache[filename] = new
            return new
        else:
            self.cache[filename] = filename
        return filename

    def escape_rttm(self, rttm):
        return rttm.__class__(
            [line.replace(filename=self(line.filename)) for line in rttm]
        )

    def escape_uem(self, uem):
        return uem.__class__(
            [line.replace(filename=self(line.filename)) for line in uem]
        )

    def restore(self, filename):
        if self._DISABLED:
            return filename
        if filename in self.cache:
            assert self.cache[filename] == filename, (self.cache[filename], filename)
            return filename
        for k, v in self.cache.items():
            if v == filename:
                return k
        raise ValueError(f'Cannot find {filename} as value in {self.cache}')



def md_eval_22_multifile(
        reference, hypothesis, collar=0, regions='all',
        uem=None
):
    """
    Computes the Diarization Error Rate (DER) and statistics using
    md-eval-22.pl.

    Args:
        reference: The reference in a format convertible to RTTM.
        hypothesis: The hypothesis in a format convertible to RTTM.
        collar: The collar in seconds.
        regions: The regions to score. Either 'all' or 'nooverlap'.
            'nooverlap' limits scoring to single-speaker regions by appending
            `-1` to the md-eval-22.pl command.
        uem: The UEM file.
    """
    if regions not in {'all', 'nooverlap'}:
        raise ValueError(
            f'Invalid regions: {regions}. '
            f'Select from "all" or "nooverlap".'
        )

    from meeteval.io.rttm import RTTM
    reference = RTTM.new(reference)
    hypothesis = RTTM.new(hypothesis)

    escaper = _FilenameEscaper()
    reference = escaper.escape_rttm(reference)
    hypothesis = escaper.escape_rttm(hypothesis)

    reference = _fix_channel(reference)
    hypothesis = _fix_channel(hypothesis)

    r = reference.grouped_by_filename()
    h = hypothesis.grouped_by_filename()

    keys = set(r.keys()) & set(h.keys())
    missing = set(r.keys()) ^ set(h.keys())
    if len(missing) > 0:
        logging.warning(f'Missing {len(missing)} filenames:', missing)
        logging.warning(f'Found {len(keys)} filenames:', keys)

    md_eval_22 = shutil.which('md-eval-22.pl')
    if not md_eval_22:
        md_eval_22 = Path(__file__).parent / 'md-eval-22.pl'
        if md_eval_22.exists():
            pass
        else:
            url = 'https://github.com/nryant/dscore/raw/master/scorelib/md-eval-22.pl'
            logging.info(f'md-eval-22.pl not found. Trying to download it from {url}.')
            import urllib.request
            urllib.request.urlretrieve(url, md_eval_22)
            logging.info(f'Wrote {md_eval_22}')

    warned = False

    def get_details(r, h, key, tmpdir, uem):
        nonlocal warned

        r_file = tmpdir / f'{key}.ref.rttm'
        h_file = tmpdir / f'{key}.hyp.rttm'
        r.dump(r_file)
        h.dump(h_file)

        cmd = [
            'perl', f'{md_eval_22}',
            '-c', f'{collar}',
            '-r', f'{r_file}',
            '-s', f'{h_file}',
        ]

        if regions == 'nooverlap':
            cmd.append('-1')

        if uem:
            uem_file = tmpdir / f'{key}.uem'
            uem = escaper.escape_uem(uem)
            uem.dump(uem_file)
            cmd.extend(['-u', f'{uem_file}'])
        elif not warned:
            warned = True
            logging.warning(f'No UEM file provided. See https://github.com/fgnt/meeteval/issues/97#issuecomment-2508140402 for details.')
        cp = subprocess.run(cmd, stdout=subprocess.PIPE,
                            check=True, universal_newlines=True)

        # SCORED SPEAKER TIME =4309.340250 secs
        # MISSED SPEAKER TIME =4309.340250 secs
        # FALARM SPEAKER TIME =0.000000 secs
        # SPEAKER ERROR TIME =0.000000 secs
        #  OVERALL SPEAKER DIARIZATION ERROR = 100.00 percent of scored speaker time  `(ALL)

        error_rate, = re.findall(r'OVERALL SPEAKER DIARIZATION ERROR = ([\d.]+) percent of scored speaker time',
                                 cp.stdout)
        length, = re.findall(r'SCORED SPEAKER TIME =([\d.]+) secs', cp.stdout)
        deletions, = re.findall(r'MISSED SPEAKER TIME =([\d.]+) secs', cp.stdout)
        insertions, = re.findall(r'FALARM SPEAKER TIME =([\d.]+) secs', cp.stdout)
        substitutions, = re.findall(r'SPEAKER ERROR TIME =([\d.]+) secs', cp.stdout)

        def convert(string):
            return decimal.Decimal(string)

        return DiaErrorRate(
            scored_speaker_time=convert(length),
            missed_speaker_time=convert(deletions),
            falarm_speaker_time=convert(insertions),
            speaker_error_time=convert(substitutions),
            error_rate=convert(error_rate) / 100,
        )

    per_reco = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for key in keys:
            per_reco[escaper.restore(key)] = get_details(r[key], h[key], key, tmpdir, uem)

        md_eval = get_details(
            meeteval.io.RTTM([line for key in keys for line in r[key]]),
            meeteval.io.RTTM([line for key in keys for line in h[key]]),
            '',
            tmpdir,
            uem,
        )
        summary = sum(per_reco.values())
        error_rate = summary.error_rate.quantize(md_eval.error_rate)
        if error_rate != md_eval.error_rate:
            raise RuntimeError(
                f'The error rate of md-eval-22.pl on all recordings '
                f'({summary.error_rate})\n'
                f'does not match the average error rate of md-eval-22.pl '
                f'applied to each recording ({md_eval.error_rate}).'
            )

    return per_reco


def md_eval_22(reference, hypothesis, collar=0, regions='all', uem=None):
    """
    Computes the Diarization Error Rate (DER) using md-eval-22.pl.
    """
    from meeteval.io.rttm import RTTM
    reference = RTTM.new(reference, filename='dummy')
    hypothesis = RTTM.new(hypothesis, filename='dummy')

    assert len(reference.filenames()) == 1, reference.filenames()
    assert len(hypothesis.filenames()) == 1, hypothesis.filenames()
    assert reference.filenames() == hypothesis.filenames(), (reference.filenames(), hypothesis.filenames())

    return md_eval_22_multifile(
        reference, hypothesis, collar, regions=regions, uem=uem
    )[list(reference.filenames())[0]]
