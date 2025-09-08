from pathlib import Path

import pytest
from hypothesis import assume, settings, given, strategies as st, reproduce_failure

import meeteval.io
from meeteval.io import SegLST
from meeteval.wer import combine_error_rates


# Limit alphabet to ensure a few correct matches
@st.composite
def string(draw, max_length=100):
    return ' '.join(draw(st.text(alphabet='abcdefg', min_size=0, max_size=max_length)))


# Generate a random SegLST object
@st.composite
def seglst_segment(draw, max_speakers):
    start_time = draw(st.floats(min_value=0, max_value=90))
    end_time = draw(st.floats(min_value=start_time, max_value=100))
    return {
        'words': draw(string()),
        'session_id': 'test-session',
        'speaker': f'spk-{draw(st.integers(min_value=1, max_value=max_speakers))}',
        'start_time': start_time,
        'end_time': end_time,
    }


@st.composite
def seglst(draw, min_segments=0, max_segments=10, max_speakers=2):
    """
    Constraints:
        - end >= start
        - start values must be increasing
    """
    return SegLST(
        draw(st.lists(
            seglst_segment(max_speakers),
            min_size=min_segments,
            max_size=max_segments
        ))
    )


@given(
    seglst(max_speakers=1, min_segments=1),
    seglst(max_speakers=3, min_segments=1)
)
@settings(deadline=None)    # The tests take longer on the GitHub actions test servers
def test_tcorc_burn(reference, hypothesis):
    from meeteval.wer.wer.time_constrained_orc import time_constrained_orc_wer
    from meeteval.wer.wer.time_constrained import time_constrained_siso_word_error_rate

    tcorc = time_constrained_orc_wer(reference, hypothesis, collar=5, reference_sort=False, hypothesis_sort=False)

    assert len(tcorc.assignment) == len(reference)
    assert isinstance(tcorc.errors, int)
    assert tcorc.errors >= 0
    assigned_reference, assigned_hypothesis = tcorc.apply_assignment(reference, hypothesis)
    assigned_reference = assigned_reference.groupby('speaker')
    assigned_hypothesis = assigned_hypothesis.groupby('speaker')
    er = combine_error_rates(
        *[
            time_constrained_siso_word_error_rate(
                assigned_reference.get(k, meeteval.io.SegLST([])),
                assigned_hypothesis.get(k, meeteval.io.SegLST([])),
                reference_sort=False,
                hypothesis_sort=False,
                collar=5,
            )
            for k in set(assigned_reference.keys()) | set(assigned_hypothesis.keys())
        ]
    )
    assert er.errors == tcorc.errors
    assert er.length == tcorc.length
    assert er.insertions == tcorc.insertions
    assert er.deletions == tcorc.deletions
    assert er.substitutions == tcorc.substitutions


@given(
    seglst(max_speakers=1, min_segments=1),
    seglst(max_speakers=2, min_segments=1)
)
def test_tcorc_vs_orc(reference, hypothesis):
    """Tests that tcORC-WER is equal to ORC-WER when the collar is larger than
    the recording length
    """
    from meeteval.wer.wer.orc import orc_word_error_rate
    from meeteval.wer.wer.time_constrained_orc import time_constrained_orc_wer

    orc = orc_word_error_rate(reference, hypothesis, reference_sort=False, hypothesis_sort=False)

    # Without time constraint (collar is larger than the maximum length)
    # and without sorting because the low-level ORC-WER doesn't sort
    tcorc = time_constrained_orc_wer(reference, hypothesis, collar=1000, reference_sort=False, hypothesis_sort=False)
    assert orc.error_rate == tcorc.error_rate
    assert orc.errors == tcorc.errors
    # TODO: make sure that the following are equal
    # assert orc.insertions == tcorc.insertions
    # assert orc.deletions == tcorc.deletions
    # assert orc.substitutions == tcorc.substitutions


@given(
    seglst(max_speakers=1, min_segments=1),
    seglst(max_speakers=2, min_segments=1),
)
def test_orc_bound_by_tcorc(reference, hypothesis):
    """Tests that tcORC-WER is never lower than ORC-WER"""
    from meeteval.wer.wer.orc import orc_word_error_rate
    from meeteval.wer.wer.time_constrained_orc import time_constrained_orc_wer

    orc = orc_word_error_rate(reference, hypothesis, reference_sort=False, hypothesis_sort=False)
    tcorc = time_constrained_orc_wer(reference, hypothesis, collar=0.1, reference_sort=False, hypothesis_sort=False)

    # error_rate can be None when length is None
    assert orc.error_rate is None and tcorc.error_rate is None or orc.error_rate <= tcorc.error_rate


@given(
    seglst(max_speakers=1, min_segments=1),
    seglst(max_speakers=2, min_segments=1),
)
def test_tcorc_bound_by_tcp(reference, hypothesis):
    """Tests that tcORC-WER is never larger than tcpWER"""
    from meeteval.wer.wer.time_constrained import time_constrained_minimum_permutation_word_error_rate
    from meeteval.wer.wer.time_constrained_orc import time_constrained_orc_wer

    tcp = time_constrained_minimum_permutation_word_error_rate(reference, hypothesis, collar=0)  # use collar=5 for real data
    tcorc = time_constrained_orc_wer(reference, hypothesis, collar=0.1)

    # error_rate can be None when length is None
    assert tcp.error_rate is None and tcorc.error_rate is None or tcorc.error_rate <= tcp.error_rate


@given(
    seglst(max_speakers=1),
    seglst(max_speakers=1),
    st.floats(min_value=0, max_value=100)
)
def test_tcorc_vs_tcsiso(reference, hypothesis, collar):
    """Tests that the tcORC-WER is equal to the tcSISO-WER when
    only a single speaker is present"""
    from meeteval.wer.wer.time_constrained_orc import time_constrained_orc_wer
    from meeteval.wer.wer.time_constrained import time_constrained_siso_word_error_rate

    tcorc = time_constrained_orc_wer(reference, hypothesis, collar=collar)
    tcsiso = time_constrained_siso_word_error_rate(reference, hypothesis, collar=collar)

    assert tcorc.error_rate == tcsiso.error_rate


@given(
    seglst(max_speakers=3)
)
def test_tcorc_zero_vs_self(reference):
    """Tests that the tcORC-WER is zero when the hypothesis is equal to reference"""
    from meeteval.wer.wer.time_constrained_orc import time_constrained_orc_wer

    tcorc = time_constrained_orc_wer(reference, reference, collar=0.1)
    assert tcorc.error_rate == 0 or tcorc.error_rate is None
    assert tcorc.reference_self_overlap == tcorc.hypothesis_self_overlap


def test_examples_zero_self_overlap():
    """Tests that self-overlap is measured correctly (0) for the example files"""
    example_files = (Path(__file__).parent.parent / 'example_files').absolute()

    from meeteval.wer import tcorcwer
    wers = tcorcwer(example_files / 'ref.stm', example_files / 'hyp.stm', collar=5)
    for k, wer in wers.items():
        assert wer.reference_self_overlap.overlap_time == 0, (k, wer)
        assert wer.hypothesis_self_overlap.overlap_time == 0, (k, wer)


def test_assignment_keeps_order():
    """
    Tests that elements in the assignment correspond to the order in the input
    to the orc_wer function, not the sorted segments.
    """
    from meeteval.wer.wer.time_constrained_orc import time_constrained_orc_wer

    tcorc = time_constrained_orc_wer(
        SegLST([
            {'words': 'a1', 'session_id': 'a', 'speaker': 'A', 'start_time': 2, 'end_time': 3},
            {'words': '', 'session_id': 'a', 'speaker': 'A', 'start_time': 1, 'end_time': 2},
            {'words': 'a2', 'session_id': 'a', 'speaker': 'A', 'start_time': 0, 'end_time': 1}
        ]),
        SegLST([
            {'words': 'a1', 'session_id': 'a', 'speaker': 'A1', 'start_time': 2, 'end_time': 3},
            {'words': 'a2', 'session_id': 'a', 'speaker': 'A2', 'start_time': 0, 'end_time': 1}
        ]),
        reference_sort='segment',
        collar=5,
    )
    assert tcorc.assignment == ('A1', 'A1', 'A2'), tcorc.assignment


