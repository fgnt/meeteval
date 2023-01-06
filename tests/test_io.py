import tempfile
from pathlib import Path

from meeteval.io.stm import STM, STMLine
from meeteval.io.ctm import CTM, CTMLine

ctm_example = '''
;;  Example from https://github.com/usnistgov/SCTK/blob/master/doc/infmts.htm
;;  Comments follow ';;' 
;;
;;  The Blank lines are ignored

;;
7654 A 11.34 0.2  YES -6.763
7654 A 12.00 0.34 YOU -12.384530
7654 A 13.30 0.5  CAN 2.806418
7654 A 17.50 0.2  AS 0.537922
7654 B 1.34 0.2  I -6.763
7654 B 2.00 0.34 CAN -12.384530
7654 B 3.40 0.5  ADD 2.806418
7654 B 7.00 0.2  AS 0.537922
'''


def test_ctm_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        file = tmpdir / 'file.ctm'
        file.write_text(ctm_example)
        ctm = CTM.load(file)

        assert ctm.lines == [
            CTMLine(filename='7654', channel='A', begin_time=11.34, duration=0.2, word='YES', confidence='-6.763'),
            CTMLine(filename='7654', channel='A', begin_time=12.0, duration=0.34, word='YOU', confidence='-12.384530'),
            CTMLine(filename='7654', channel='A', begin_time=13.3, duration=0.5, word='CAN', confidence='2.806418'),
            CTMLine(filename='7654', channel='A', begin_time=17.5, duration=0.2, word='AS', confidence='0.537922'),
            CTMLine(filename='7654', channel='B', begin_time=1.34, duration=0.2, word='I', confidence='-6.763'),
            CTMLine(filename='7654', channel='B', begin_time=2.0, duration=0.34, word='CAN', confidence='-12.384530'),
            CTMLine(filename='7654', channel='B', begin_time=3.4, duration=0.5, word='ADD', confidence='2.806418'),
            CTMLine(filename='7654', channel='B', begin_time=7.0, duration=0.2, word='AS', confidence='0.537922')
        ]


stm_example = '''
;;  Example from https://github.com/usnistgov/SCTK/blob/master/doc/infmts.htm
;; comment
2345 A 2345-a 0.10 2.03 uh huh yes i thought
2345 A 2345-b 2.10 3.04 dog walking is a very
2345 A 2345-a 3.50 4.59 yes but it's worth it
'''


def test_stm_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        file = tmpdir / 'file.stm'
        file.write_text(stm_example)
        stm = STM.load(file)

        assert stm.lines == [
            STMLine(filename='2345', channel='A', speaker_id='2345-a', begin_time=0.1, end_time=2.03, transcript='uh huh yes i thought'),
            STMLine(filename='2345', channel='A', speaker_id='2345-b', begin_time=2.1, end_time=3.04, transcript='dog walking is a very'),
            STMLine(filename='2345', channel='A', speaker_id='2345-a', begin_time=3.5, end_time=4.59, transcript="yes but it's worth it")
        ]
