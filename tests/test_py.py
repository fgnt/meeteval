from pathlib import Path

import meeteval
from meeteval.wer.wer import combine_error_rates, ErrorRate


example_files = (Path(__file__).parent.parent / 'example_files').absolute()


def check_result(avg, errors, length):
    assert avg.errors == errors, (errors, length, avg)
    assert avg.length == length, (errors, length, avg)


def test_cpwer():
    from meeteval.wer import cpwer

    ref = example_files / 'ref.stm'
    hyp = example_files / 'hyp.stm'

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


def test_tcpwer():
    from meeteval.wer import tcpwer

    ref = example_files / 'ref.stm'
    hyp = example_files / 'hyp.stm'

    details = tcpwer(ref, hyp)
    avg = combine_error_rates(details)
    check_result(avg, 88, 92)

    details = tcpwer([ref], [hyp])
    avg = combine_error_rates(details)
    check_result(avg, 88, 92)

    details = tcpwer(meeteval.io.load(ref), meeteval.io.load(hyp))
    avg = combine_error_rates(details)
    check_result(avg, 88, 92)

    details = tcpwer(meeteval.io.load(ref).to_seglst(), meeteval.io.load(hyp).to_seglst())
    avg = combine_error_rates(details)
    check_result(avg, 88, 92)


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
