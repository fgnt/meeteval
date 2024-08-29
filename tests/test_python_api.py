import decimal
from pathlib import Path

import pytest

import meeteval
from meeteval.wer.wer import combine_error_rates, ErrorRate

example_files = (Path(__file__).parent.parent / 'example_files').absolute()


def check_result(avg, errors, length, msg=''):
    assert avg.errors == errors, (msg, errors, length, avg)
    assert avg.length == length, (msg, errors, length, avg)


def test_cpwer():
    from meeteval.wer import cpwer

    ref = example_files / 'ref.stm'
    hyp = example_files / 'hyp.stm'
    uem = example_files / 'uem.uem'

    details = cpwer(ref, hyp)
    avg = combine_error_rates(details)
    check_result(avg, 25, 92)

    details = cpwer([ref], [hyp])
    avg = combine_error_rates(details)
    check_result(avg, 25, 92)

    details = cpwer(meeteval.io.load(ref), meeteval.io.load(hyp))
    avg = combine_error_rates(details)
    check_result(avg, 25, 92)

    details = cpwer(meeteval.io.load(ref).to_seglst(), meeteval.io.load(hyp).to_seglst())
    avg = combine_error_rates(details)
    check_result(avg, 25, 92)

    details = cpwer(ref, hyp, uem=uem)
    avg = combine_error_rates(details)
    check_result(avg, 25, 61)

    details = cpwer(ref, hyp, uem=meeteval.io.UEM.load(uem))
    avg = combine_error_rates(details)
    check_result(avg, 25, 61)


def test_tcpwer():
    from meeteval.wer import tcpwer

    ref = example_files / 'ref.stm'
    hyp = example_files / 'hyp.stm'
    uem = example_files / 'uem.uem'

    details = tcpwer(ref, hyp, collar=5)
    avg = combine_error_rates(details)
    check_result(avg, 25, 92)

    details = tcpwer([ref], [hyp], collar=5)
    avg = combine_error_rates(details)
    check_result(avg, 25, 92)

    details = tcpwer(meeteval.io.load(ref), meeteval.io.load(hyp), collar=5)
    avg = combine_error_rates(details)
    check_result(avg, 25, 92)

    details = tcpwer(meeteval.io.load(ref).to_seglst(), meeteval.io.load(hyp).to_seglst(), collar=5)
    avg = combine_error_rates(details)
    check_result(avg, 25, 92)

    details = tcpwer(ref, hyp, collar=0)
    avg = combine_error_rates(details)
    check_result(avg, 88, 92)

    details = tcpwer(ref, hyp, uem=uem, collar=5)
    avg = combine_error_rates(details)
    check_result(avg, 25, 61)

    details = tcpwer(ref, hyp, uem=uem, collar=0)
    avg = combine_error_rates(details)
    check_result(avg, 88, 61)

    details = tcpwer(ref, hyp, uem=meeteval.io.UEM.load(uem), collar=5)
    avg = combine_error_rates(details)
    check_result(avg, 25, 61)


    # Try different pseudo_word_timing values.
    # There is no motivation for the selected combinations.
    # Hint: We recommend to use the defaults.
    for h_pwt, r_pwt, errors in [
        ['character_based_points', 'character_based', 42],
        ['equidistant_points', 'equidistant_intervals', 42],
        ['full_segment', 'full_segment', 25],
        ['character_based', 'character_based_points', 42],
        ['equidistant_intervals', 'equidistant_points', 40],
    ]:
        details = tcpwer(ref, hyp, collar=decimal.Decimal(0.5),
                         hyp_pseudo_word_timing=h_pwt,
                         ref_pseudo_word_timing=r_pwt, )
        avg = combine_error_rates(details)
        check_result(avg, errors, 92, msg=f'{h_pwt} {r_pwt}')


def test_orcwer():
    from meeteval.wer import orcwer

    ref = example_files / 'ref.stm'
    hyp = example_files / 'hyp.stm'

    details = orcwer(ref, hyp)
    avg = combine_error_rates(details)
    check_result(avg, 9, 92)

    details = orcwer([ref], [hyp])
    avg = combine_error_rates(details)
    check_result(avg, 9, 92)

    details = orcwer(meeteval.io.load(ref), meeteval.io.load(hyp))
    avg = combine_error_rates(details)
    check_result(avg, 9, 92)

    details = orcwer(meeteval.io.load(ref).to_seglst(), meeteval.io.load(hyp).to_seglst())
    avg = combine_error_rates(details)
    check_result(avg, 9, 92)


def test_mimower():
    from meeteval.wer import mimower

    ref = example_files / 'ref.stm'
    hyp = example_files / 'hyp.stm'

    details = mimower(ref, hyp)
    avg = combine_error_rates(details)
    check_result(avg, 9, 92)

    details = mimower([ref], [hyp])
    avg = combine_error_rates(details)
    check_result(avg, 9, 92)

    details = mimower(meeteval.io.load(ref), meeteval.io.load(hyp))
    avg = combine_error_rates(details)
    check_result(avg, 9, 92)

    details = mimower(meeteval.io.load(ref).to_seglst(), meeteval.io.load(hyp).to_seglst())
    avg = combine_error_rates(details)
    check_result(avg, 9, 92)


def test_di_cpwer():
    from meeteval.wer import greedy_dicpwer

    ref = example_files / 'ref.stm'
    hyp = example_files / 'hyp.stm'

    details = greedy_dicpwer(ref, hyp)
    avg = combine_error_rates(details)
    check_result(avg, 9, 92)

    details = greedy_dicpwer([ref], [hyp])
    avg = combine_error_rates(details)
    check_result(avg, 9, 92)

    details = greedy_dicpwer(meeteval.io.load(ref), meeteval.io.load(hyp))
    avg = combine_error_rates(details)
    check_result(avg, 9, 92)

    details = greedy_dicpwer(meeteval.io.load(ref).to_seglst(), meeteval.io.load(hyp).to_seglst())
    avg = combine_error_rates(details)
    check_result(avg, 9, 92)



def test_tcorcwer():
    from meeteval.wer import tcorcwer

    ref = example_files / 'ref.stm'
    hyp = example_files / 'hyp.stm'

    details = tcorcwer(ref, hyp, collar=5)
    avg = combine_error_rates(details)
    check_result(avg, 9, 92)

    details = tcorcwer([ref], [hyp], collar=5)
    avg = combine_error_rates(details)
    check_result(avg, 9, 92)

    details = tcorcwer(meeteval.io.load(ref), meeteval.io.load(hyp), collar=5)
    avg = combine_error_rates(details)
    check_result(avg, 9, 92)

    details = tcorcwer(meeteval.io.load(ref).to_seglst(), meeteval.io.load(hyp).to_seglst(), collar=5)
    avg = combine_error_rates(details)
    check_result(avg, 9, 92)


def test_md_eval_22():
    from meeteval.der import md_eval_22

    ref = example_files / 'ref.stm'
    hyp = example_files / 'hyp.stm'
    uem = example_files / 'uem.uem'

    details = md_eval_22(ref, hyp)
    avg = combine_error_rates(details)
    assert float(avg.error_rate) == 0.3
    assert avg.scored_speaker_time == 6
    assert float(avg.missed_speaker_time) == 0.8
    assert float(avg.falarm_speaker_time) == 0.6

    details = md_eval_22(ref, hyp, uem=uem)
    avg = combine_error_rates(details)
    assert float(avg.error_rate) == 0.36
    assert avg.scored_speaker_time == 5
    assert float(avg.missed_speaker_time) == 0.8
    assert float(avg.falarm_speaker_time) == 0.6


def test_multi_file():
    from meeteval.io.seglst import apply_multi_file

    class Mock:
        def __init__(self):
            self.session_ids = []

        def __call__(self, reference, hypothesis):
            self.session_ids.extend(
                reference.unique('session_id') | hypothesis.unique('session_id')
            )

    # Check example files
    ref = meeteval.io.load(example_files / 'ref.stm')
    hyp = meeteval.io.load(example_files / 'hyp.stm')
    m = Mock()
    apply_multi_file(m, ref, hyp)
    assert m.session_ids == ['recordingA', 'recordingB']

    # Check missing ref
    with pytest.raises(RuntimeError):
        m = Mock()
        apply_multi_file(m, ref.filter(lambda x: x.filename == 'recordingA'), hyp)
    m = Mock()
    apply_multi_file(m, ref.filter(lambda x: x.filename == 'recordingA'), hyp, partial=True)
    assert m.session_ids == ['recordingA']

    # Check missing hyp. Assumes that the missing hypothesis is empty
    m = Mock()
    apply_multi_file(m, ref, hyp.filter(lambda x: x.filename == 'recordingA'), allowed_empty_examples_ratio=0.5)
    assert m.session_ids == ['recordingA', 'recordingB']

    # Check missing hyp amount above ratio
    m = Mock()
    with pytest.raises(RuntimeError):
        apply_multi_file(m, ref, hyp.filter(lambda x: x.filename == 'recordingA'), allowed_empty_examples_ratio=0)
    assert m.session_ids == []

    # Check empty reference
    with pytest.raises(RuntimeError):
        m = Mock()
        apply_multi_file(m, meeteval.io.SegLST([]), hyp)
    with pytest.raises(RuntimeError):
        m = Mock()
        apply_multi_file(m, meeteval.io.SegLST([]), hyp, partial=True)

    # Check empty hypothesis
    m = Mock()
    apply_multi_file(m, ref, meeteval.io.SegLST([]), allowed_empty_examples_ratio=1)
    assert m.session_ids == ['recordingA', 'recordingB']
