import tempfile
from pathlib import Path

from hypothesis import given, strategies as st, example
import decimal
import pytest

from meeteval.io import RTTM, UEM, UEMLine
from meeteval.io.keyed_text import KeyedText
from meeteval.io.stm import STM, STMLine
from meeteval.io.ctm import CTM, CTMLine

example_files = (Path(__file__).parent.parent / 'example_files').absolute()

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
            CTMLine(filename='7654', channel='A', begin_time=decimal.Decimal('11.34'), duration=decimal.Decimal('0.2'),
                    word='YES', confidence='-6.763'),
            CTMLine(filename='7654', channel='A', begin_time=decimal.Decimal('12.0'), duration=decimal.Decimal('0.34'),
                    word='YOU', confidence='-12.384530'),
            CTMLine(filename='7654', channel='A', begin_time=decimal.Decimal('13.3'), duration=decimal.Decimal('0.5'),
                    word='CAN', confidence='2.806418'),
            CTMLine(filename='7654', channel='A', begin_time=decimal.Decimal('17.5'), duration=decimal.Decimal('0.2'),
                    word='AS', confidence='0.537922'),
            CTMLine(filename='7654', channel='B', begin_time=decimal.Decimal('1.34'), duration=decimal.Decimal('0.2'),
                    word='I', confidence='-6.763'),
            CTMLine(filename='7654', channel='B', begin_time=decimal.Decimal('2.0'), duration=decimal.Decimal('0.34'),
                    word='CAN', confidence='-12.384530'),
            CTMLine(filename='7654', channel='B', begin_time=decimal.Decimal('3.4'), duration=decimal.Decimal('0.5'),
                    word='ADD', confidence='2.806418'),
            CTMLine(filename='7654', channel='B', begin_time=decimal.Decimal('7.0'), duration=decimal.Decimal('0.2'),
                    word='AS', confidence='0.537922')
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
            STMLine(filename='2345', channel='A', speaker_id='2345-a', begin_time=decimal.Decimal('0.1'),
                    end_time=decimal.Decimal('2.03'), transcript='uh huh yes i thought'),
            STMLine(filename='2345', channel='A', speaker_id='2345-b', begin_time=decimal.Decimal('2.1'),
                    end_time=decimal.Decimal('3.04'), transcript='dog walking is a very'),
            STMLine(filename='2345', channel='A', speaker_id='2345-a', begin_time=decimal.Decimal('3.5'),
                    end_time=decimal.Decimal('4.59'), transcript="yes but it's worth it")
        ]


uem_example = '''
recordingA 1 0 10
recordingB 1 0 0
recordingC 1 1.5 2.30
'''

def test_uem_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        file = tmpdir / 'file.uem'
        file.write_text(uem_example)
        uem = UEM.load(file)

        assert uem.lines == [
            UEMLine(filename='recordingA', channel=1, begin_time=decimal.Decimal('0'), end_time=decimal.Decimal('10')),
            UEMLine(filename='recordingB', channel=1, begin_time=decimal.Decimal('0'), end_time=decimal.Decimal('0')),
            UEMLine(filename='recordingC', channel=1, begin_time=decimal.Decimal('1.5'), end_time=decimal.Decimal('2.30')),
        ]


# Generate files
# The generated files don't contain comments since they cannot be reconstructed
filenames = st.text(st.characters(blacklist_categories=['Z', 'C'], blacklist_characters=';'), min_size=1)  # (no space, no control char, no comment char)
speaker_ids = st.text(st.characters(blacklist_categories=['Z', 'C']), min_size=1)  # (no space, no control char)
timestamps = st.decimals(allow_nan=False, allow_infinity=False)
durations = st.decimals(min_value=0, allow_nan=False, allow_infinity=False)
words = st.text(st.characters(blacklist_categories=['C', 'Z']), min_size=1)  # (no space, no control char)
transcripts = st.builds(' '.join, st.lists(words))

keyed_text_line = st.builds('{} {}\n'.format, filenames, transcripts)
keyed_text_str = st.builds(''.join, st.lists(keyed_text_line))

stm_line = st.builds('{} 1 {} {} {} {}\n'.format, filenames, speaker_ids, timestamps, timestamps, transcripts)
stm_str = st.builds(''.join, st.lists(stm_line))

ctm_line = st.builds('{} 1 {} {} {}\n'.format, filenames, timestamps, durations, words)
ctm_str = st.builds(''.join, st.lists(ctm_line))

# Only SPEAKER lines are supported
rttm_line = st.builds(
    'SPEAKER {} 1 {} {} <NA> <NA> {} <NA> <NA>\n'.format,
    filenames, timestamps, durations, speaker_ids
)
rttm_str = st.builds(''.join, st.lists(rttm_line))


@given(keyed_text_str)
def test_reconstruct_keyed_text(keyed_text_str):
    reconstructed = KeyedText.parse(keyed_text_str).dumps()
    assert reconstructed == keyed_text_str, (reconstructed, keyed_text_str)


@given(stm_str)
def test_reconstruct_stm(stm_str):
    reconstructed = STM.parse(stm_str).dumps()
    assert reconstructed == stm_str, (reconstructed, stm_str)


@given(ctm_str)
def test_reconstruct_ctm(ctm_str):
    reconstructed = CTM.parse(ctm_str.strip()).dumps()
    assert reconstructed == ctm_str, (reconstructed, ctm_str)


@given(rttm_str)
@example('SPEAKER CMU_20020319-1400_d01_NONE 1 130.430000 2.350 <NA> <NA> juliet <NA> <NA>\n')
def test_reconstruct_rttm(rttm_str):
    reconstructed = RTTM.parse(rttm_str).dumps()
    assert reconstructed == rttm_str, (reconstructed, rttm_str)



@pytest.mark.parametrize(
    'filename',
    [
        'hyp.stm',
        'text_hyp',
        'hyp.rttm',
        'hyp1.ctm',
        'hyp.seglst.json',
        ('hyp1.ctm', 'hyp2.ctm'),
        ('hyp.stm', 'ref.stm'),
    ]
)
def test_load_example_files(filename):
    """
    This function just checks whether the load function succeeds, the file contents are not checked.
    """
    if isinstance(filename, tuple):
        filename = tuple(example_files / f for f in filename)
    else:
        filename = example_files / filename
    import meeteval
    meeteval.io.load(filename)
