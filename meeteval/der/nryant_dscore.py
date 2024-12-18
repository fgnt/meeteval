import logging
import sys
import subprocess
import tempfile
import decimal
from pathlib import Path


def _dscore_multifile(
        reference, hypothesis, collar=0, regions='all',
        uem=None
):
    """
    dscore produces a table with the final scores, but we need
    the details. Hence, call dscore only to compare the error rate
    with md_eval_22_multifile.

    >>> import packaging.version
    >>> import numpy as np
    >>> if packaging.version.parse(np.__version__) >= packaging.version.parse('1.24'):
    ...     import pytest
    ...     pytest.skip(f'dscore fails with numpy >= 1.24. Current version: {np.__version__}')

    >>> from meeteval.io.rttm import RTTM
    >>> from meeteval.io.uem import UEM
    >>> reference = RTTM.parse('''
    ... SPEAKER S1 1 0.0 0.5 <NA> <NA> spk1 <NA>
    ... SPEAKER S1 1 0.5 0.5 <NA> <NA> spk2 <NA>
    ... SPEAKER S1 1 1.0 0.5 <NA> <NA> spk1 <NA>
    ... ''')
    >>> hypothesis = RTTM.parse('''
    ... SPEAKER S1 1 0.0 0.5 <NA> <NA> spk1 <NA>
    ... SPEAKER S1 1 0.5 0.5 <NA> <NA> spk2 <NA>
    ... SPEAKER S1 1 1.0 0.5 <NA> <NA> spk2 <NA>
    ... ''')
    >>> uem = UEM.parse('''
    ... S1 1 0.0 1.5
    ... ''')
    >>> import pprint
    >>> pprint.pprint(_dscore_multifile(reference, hypothesis, uem=uem))
    {'S1': Decimal('0.3333')}
    """
    from meeteval.der.md_eval import _FilenameEscaper
    escaper = _FilenameEscaper()

    from meeteval.io.rttm import RTTM
    reference = RTTM.new(reference)
    hypothesis = RTTM.new(hypothesis)

    reference = escaper.escape_rttm(reference)
    hypothesis = escaper.escape_rttm(hypothesis)

    score_py = Path(__file__).parent / 'dscore_repo' / 'score.py'
    if not score_py.exists():
        subprocess.run(['git', 'clone', 'https://github.com/nryant/dscore.git', score_py.parent])

    filtered = 0
    for line in reference:
        if line.duration == 0:
            filtered += 1
    if filtered:
        logging.info(f'Filtered {filtered} lines with zero duration in reference (dscore doesn\'t support zero duration)')
        reference = RTTM([line for line in reference if line.duration != 0])

    filtered = 0
    for line in hypothesis:
        if line.duration == 0:
            filtered += 1
    if filtered:
        logging.info(f'Filtered {filtered} lines with zero duration in hypothesis (dscore doesn\'t support zero duration)')
        hypothesis = RTTM([line for line in hypothesis if line.duration != 0])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        r_file = tmpdir / f'ref.rttm'
        h_file = tmpdir / f'hyp.rttm'
        reference.dump(r_file)
        hypothesis.dump(h_file)

        cmd = [
            sys.executable, str(score_py),
            '--collar', str(collar),
            # '--ignore_overlaps', regions,
            '-r', f'{r_file}',
            '-s', f'{h_file}',
            '--table_fmt', 'tsv'
        ]
        if uem:
            uem_file = tmpdir / 'uem.rttm'
            uem = escaper.escape_uem(uem)
            uem.dump(uem_file)
            cmd.extend(['-u', f'{uem_file}'])

        if regions == 'all':
            pass
        elif regions == 'nooverlap':
            cmd.append('--ignore_overlaps')

        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
            check=True, universal_newlines=True,
            cwd=score_py.parent
        )

        result = {}
        for line in p.stdout.strip().split('\n')[1:-1]:
            line_parts = line.split('\t')
            result[escaper.restore(line_parts[0].strip())] = decimal.Decimal(line_parts[1].strip()) / 100

        assert result, p.stdout
        return result


def _maybe_gen_uem(uem, reference, hypothesis):
    # Mirror the behavior of dscore
    if uem is None:
        from meeteval.io.uem import UEM, UEMLine
        uem_md_eval = UEM([
            UEMLine(
                filename=k, channel='1',
                begin_time=min(v.T['start_time']), end_time=max(v.T['end_time'])
            )
            for k, v in (reference + hypothesis).to_seglst().groupby('session_id').items()
        ])

        return uem_md_eval, None
    else:
        return uem, uem


def dscore_multifile(
        reference, hypothesis, collar=0, regions='all',
        uem=None, sanity_check=False,
):
    """
    Computes the Diarization Error Rate (DER) using md-eval-22.pl
    but create a uem if uem is None, as it is done in dscore [1].

    Additionally, compare the error rate with dscore [1], if sanity_check is True.

    Args:
        reference:
        hypothesis:
        collar:
        regions: 'all' or 'nooverlap'
        uem: If None, generate a uem from the reference and hypothesis.
             This is the default behavior of dscore, while md-eval-22,
             uses only the reference.
        sanity_check: Compare the result with dscore to ensure
                      the correctness of the implementation.
                      Requires the numpy < 1.24 (e.g. np.int),
                      because dscore fails with recent numpy versions.

    [1] https://github.com/nryant/dscore

    >>> from meeteval.io.rttm import RTTM
    >>> from meeteval.io.uem import UEM
    >>> reference = RTTM.parse('''
    ... SPEAKER rec 1 5.00 5.00 <NA> <NA> spk01 <NA>
    ... SPEAKER rec 1 10.00 10.00 <NA> <NA> spk00 <NA>
    ... ''')
    >>> hypothesis = RTTM.parse('''
    ... SPEAKER rec 1 0.00 10.00 <NA> <NA> spk01 <NA>
    ... SPEAKER rec 1 10.00 10.00 <NA> <NA> spk00 <NA>
    ... ''')
    >>> import pprint
    >>> pprint.pprint(dscore_multifile(reference, hypothesis))  # doctest: +NORMALIZE_WHITESPACE
    {'rec': DiaErrorRate(error_rate=Decimal('0.3333'),
                         scored_speaker_time=Decimal('15.000000'),
                         missed_speaker_time=Decimal('0.000000'),
                         falarm_speaker_time=Decimal('5.000000'),
                         speaker_error_time=Decimal('0.000000'))}

    >>> from meeteval.der.md_eval import md_eval_22_multifile
    >>> pprint.pprint(md_eval_22_multifile(reference, hypothesis))  # doctest: +NORMALIZE_WHITESPACE
    {'rec': DiaErrorRate(error_rate=Decimal('0.00'),
                         scored_speaker_time=Decimal('15.000000'),
                         missed_speaker_time=Decimal('0.000000'),
                         falarm_speaker_time=Decimal('0.000000'),
                         speaker_error_time=Decimal('0.000000'))}

    """
    from meeteval.der.md_eval import md_eval_22_multifile

    uem_md_eval, uem_dscore = _maybe_gen_uem(uem, reference, hypothesis)

    result = md_eval_22_multifile(
        reference, hypothesis, collar=collar, regions=regions, uem=uem_md_eval
    )
    if sanity_check:
        dscore_der = _dscore_multifile(reference, hypothesis, collar=collar, regions=regions, uem=uem_dscore)
        for key in result:
            assert key in dscore_der, (key, result, dscore_der)
            assert abs(dscore_der[key] - result[key].error_rate) <= decimal.Decimal('0.0001'), (key, dscore_der[key], result[key])

    return result


def dscore(reference, hypothesis, collar=0, regions='all', uem=None, sanity_check=False):
    """
    Computes the Diarization Error Rate (DER) using md-eval-22.pl
    but create a uem if uem is None, as it is done in dscore [1].

    Additionally, compare the error rate with dscore [1], if sanity_check is True.

    Args:
        reference:
        hypothesis:
        collar:
        regions: 'all' or 'nooverlap'
        uem: If None, generate a uem from the reference and hypothesis.
             This is the default behavior of dscore, while md-eval-22,
             uses only the reference.
        sanity_check: Compare the result with dscore to ensure
                      the correctness of the implementation.
                      Requires the numpy < 1.24 (e.g. np.int),
                      because dscore fails with recent numpy versions.

    [1] https://github.com/nryant/dscore

    >>> from meeteval.io.rttm import RTTM
    >>> from meeteval.io.uem import UEM
    >>> reference = RTTM.parse('''
    ... SPEAKER rec.a 1 5.00 5.00 <NA> <NA> spk01 <NA>
    ... SPEAKER rec.a 1 10.00 10.00 <NA> <NA> spk00 <NA>
    ... ''')
    >>> hypothesis = RTTM.parse('''
    ... SPEAKER rec.a 1 0.00 10.00 <NA> <NA> spk01 <NA>
    ... SPEAKER rec.a 1 10.00 10.00 <NA> <NA> spk00 <NA>
    ... ''')
    >>> uem = UEM.parse('''
    ... rec.a 1 0.00 15.00
    ... ''')
    >>> import pprint

    >>> pprint.pprint(dscore(reference, hypothesis))  # doctest: +NORMALIZE_WHITESPACE
    DiaErrorRate(error_rate=Decimal('0.3333'),
                 scored_speaker_time=Decimal('15.000000'),
                 missed_speaker_time=Decimal('0.000000'),
                 falarm_speaker_time=Decimal('5.000000'),
                 speaker_error_time=Decimal('0.000000'))
    >>> pprint.pprint(dscore(reference, hypothesis, uem=uem))  # doctest: +NORMALIZE_WHITESPACE
    DiaErrorRate(error_rate=Decimal('0.50'),
                 scored_speaker_time=Decimal('10.000000'),
                 missed_speaker_time=Decimal('0.000000'),
                 falarm_speaker_time=Decimal('5.000000'),
                 speaker_error_time=Decimal('0.000000'))

    # md_eval_22 ignores hyps before the first ref and after the last ref
    >>> from meeteval.der.md_eval import md_eval_22
    >>> pprint.pprint(md_eval_22(reference, hypothesis))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    DiaErrorRate(error_rate=Decimal('0.00'),
                 scored_speaker_time=Decimal('15.000000'),
                 missed_speaker_time=Decimal('0.000000'),
                 falarm_speaker_time=Decimal('0.000000'),
                 speaker_error_time=Decimal('0.000000'))
    """
    from meeteval.der.md_eval import md_eval_22
    from meeteval.io.rttm import RTTM

    reference = RTTM.new(reference, filename='dummy')
    hypothesis = RTTM.new(hypothesis, filename='dummy')

    uem_md_eval, uem_dscore = _maybe_gen_uem(uem, reference, hypothesis)

    result = md_eval_22(reference, hypothesis, collar=collar, regions=regions, uem=uem_md_eval)

    if sanity_check:
        dscore_der = _dscore_multifile(reference, hypothesis, collar=collar, regions=regions, uem=uem_dscore)
        assert list(dscore_der.values()) == [result.error_rate], (dscore_der, result.error_rate)

    return result
