import pytest
from hypothesis import assume, settings, given, strategies as st

import meeteval
from meeteval.io import SegLST
from meeteval.wer.wer.error_rate import combine_error_rates

# Limit alphabet to ensure a few correct matches
string = st.text(alphabet='abcdefg', min_size=1, max_size=100)


@st.composite
def string_with_timing(draw, max_length=100):
    """
    Constraints:
        - end >= start
        - start values must be increasing
    """
    s = draw(st.text(alphabet='abcdefg', min_size=1, max_size=max_length))
    t = []
    start = 0
    for _ in s:
        start = draw(st.integers(min_value=max(start - 2, 0), max_value=10))
        end = draw(st.integers(min_value=max(start, 0), max_value=start + 10))
        t.append((start, end))
    return s, t


@st.composite
def reference(draw, max_references, max_utterances, max_utterance_length=None):
    r = draw(st.lists(
        st.lists(string_with_timing(max_utterance_length), min_size=1, max_size=max_utterances),
        max_size=max_references,
        min_size=1,
    ))
    r_text = [[utterance for utterance, _ in speaker]for speaker in r]
    r_timing = [[utterance for _, utterance in speaker] for speaker in r]
    return r_text, r_timing


@st.composite
def hypothesis(draw, max_channels, max_channel_length=None):
    h = draw(st.lists(string_with_timing(max_channel_length), min_size=1, max_size=max_channels))
    h_text = [utterance for utterance, _ in h]
    h_timing = [utterance for _, utterance in h]
    return h_text, h_timing



# Limit alphabet to ensure a few correct matches
@st.composite
def string(draw, min_length=0, max_length=100):
    return ' '.join(draw(st.text(alphabet='abcdefg', min_size=min_length, max_size=max_length)))


# Generate a random SegLST object
@st.composite
def seglst_segment(draw, max_speakers, min_string_length=0):
    start_time = draw(st.floats(min_value=0, max_value=90))
    end_time = draw(st.floats(min_value=start_time, max_value=100))
    return {
        'words': draw(string(min_length=min_string_length)),
        'session_id': 'test-session',
        'speaker': f'spk-{draw(st.integers(min_value=1, max_value=max_speakers))}',
        'start_time': start_time,
        'end_time': end_time,
    }


@st.composite
def seglst(draw, min_segments=0, max_segments=10, max_speakers=2, min_string_length=0):
    """
    Constraints:
        - end >= start
        - start values must be increasing
    """
    return SegLST(
        draw(st.lists(
            seglst_segment(max_speakers, min_string_length=min_string_length),
            min_size=min_segments,
            max_size=max_segments
        ))
    )



@given(
    seglst(max_speakers=1),
    seglst(max_speakers=1),
    st.floats(min_value=0, max_value=100)
)
@settings(deadline=None)
def test_tmimo_vs_tcsiso(reference, hypothesis, collar):
    """Tests that the tcMIMO-WER is equal to the tcSISO-WER when
    only a single speaker is present"""
    from meeteval.wer.wer.time_constrained_mimo import time_constrained_mimo_word_error_rate
    from meeteval.wer.wer.time_constrained import time_constrained_siso_word_error_rate

    tcmimo = time_constrained_mimo_word_error_rate(reference, hypothesis, collar=collar)
    tcsiso = time_constrained_siso_word_error_rate(reference, hypothesis, collar=collar)

    assert tcmimo.error_rate == tcsiso.error_rate


@given(
    seglst(max_speakers=2, min_segments=1, max_segments=3, min_string_length=1),
    seglst(max_speakers=2, min_segments=1, max_segments=3, min_string_length=1),
)
@settings(deadline=None)
def test_tcmimo_against_mimo(reference, hypothesis):
    """
    We only test with non-empty segments because tcMIMO removes empty segments 
    to reduce the search space while MIMO does not. This has an impact on the 
    assignment, the alignment, and thus the detailed statistics.
    """
    from meeteval.wer.wer.mimo import mimo_word_error_rate
    from meeteval.wer.wer.time_constrained_mimo import time_constrained_mimo_word_error_rate

    mimo = mimo_word_error_rate(reference, hypothesis, reference_sort=False, hypothesis_sort=False)
    tcmimo = time_constrained_mimo_word_error_rate(
        reference, hypothesis, collar=10000, reference_sort=False, hypothesis_sort=False,
    )

    assert mimo.errors == tcmimo.errors, (mimo.errors, tcmimo.errors)
    assert mimo.length == tcmimo.length
    # Deactivated because we removed the assignment from the return value because it is not 
    # guaranteed to match the original order of segments in the reference
    # assert mimo.insertions == tcmimo.insertions
    # assert mimo.deletions == tcmimo.deletions
    # assert mimo.substitutions == tcmimo.substitutions
    # assert mimo.assignment == tcmimo.assignment   # The assignment is occasionally different for ambiguous edge-cases

# This example is intentionally kept small because the runtime can explode in unlucky cases
# Since this is randomly generated, it is not guaranteed that the runtime is reasonable
@given(
    seglst(max_speakers=2, min_segments=1, max_segments=3),
    seglst(max_speakers=2, min_segments=1, max_segments=3),
)
@settings(deadline=None)
def test_tcmimo_burn(reference, hypothesis):
    from meeteval.wer.wer.time_constrained_mimo import time_constrained_mimo_word_error_rate
    from meeteval.wer.wer.time_constrained import time_constrained_siso_word_error_rate

    tcmimo = time_constrained_mimo_word_error_rate(
        reference, hypothesis, collar=5, reference_sort=False, hypothesis_sort=False,

    )
    assert isinstance(tcmimo.errors, int)
    assert tcmimo.errors >= 0
    # Deactivated because we removed the assignment from the return value because it is not 
    # guaranteed to match the original order of segments in the reference
    # assert len(tcmimo.assignment) == len(reference)
    # 
    # assigned_reference, assigned_hypothesis = tcmimo.apply_assignment(reference, hypothesis)
    # assigned_reference = assigned_reference.groupby('speaker')
    # assigned_hypothesis = assigned_hypothesis.groupby('speaker')
    # er = combine_error_rates(
    #     *[
    #         time_constrained_siso_word_error_rate(
    #             assigned_reference.get(k, meeteval.io.SegLST([])),
    #             assigned_hypothesis.get(k, meeteval.io.SegLST([])),
    #             reference_sort=False,
    #             hypothesis_sort=False,
    #             collar=5,
    #         )
    #         for k in set(assigned_reference.keys()) | set(assigned_hypothesis.keys())
    #     ]
    # )
    # assert er.errors == tcmimo.errors, (er.errors, tcmimo.errors)
    # assert er.length == tcmimo.length
    # assert er.insertions == tcmimo.insertions
    # assert er.deletions == tcmimo.deletions
    # assert er.substitutions == tcmimo.substitutions


@given(
    seglst(max_speakers=1, min_segments=1),
    seglst(max_speakers=2, min_segments=1),
)
def test_mimo_bound_by_tcmimo(reference, hypothesis):
    """Tests that tcMIMO-WER is never lower than MIMO-WER"""
    from meeteval.wer.wer.mimo import mimo_word_error_rate
    from meeteval.wer.wer.time_constrained_mimo import time_constrained_mimo_word_error_rate

    mimo = mimo_word_error_rate(reference, hypothesis, reference_sort=False, hypothesis_sort=False)
    tcmimo = time_constrained_mimo_word_error_rate(reference, hypothesis, collar=0.1, reference_sort=False, hypothesis_sort=False)

    # error_rate can be None when length is None
    assert mimo.error_rate is None and tcmimo.error_rate is None or mimo.error_rate <= tcmimo.error_rate

@given(
    seglst(max_speakers=1, min_segments=1),
    seglst(max_speakers=2, min_segments=1),
)
def test_tcmimo_bound_by_tcorc(reference, hypothesis):
    """Tests that tcMIMO-WER is never larger than tcORC-WER"""
    from meeteval.wer.wer.time_constrained_orc import time_constrained_orc_wer
    from meeteval.wer.wer.time_constrained_mimo import time_constrained_mimo_word_error_rate

    tcorc = time_constrained_orc_wer(reference, hypothesis, collar=5, reference_sort=False, hypothesis_sort=False)
    tcmimo = time_constrained_mimo_word_error_rate(reference, hypothesis, collar=5, reference_sort=False, hypothesis_sort=False)

    assert tcmimo.error_rate is None or tcorc.error_rate is None or tcmimo.error_rate <= tcorc.error_rate


@given(
    seglst(max_speakers=2, max_segments=3)
)
def test_tcmimo_zero_vs_self(reference):
    """Tests that the tcMIMO-WER is zero when the hypothesis is equal to reference"""
    from meeteval.wer.wer.time_constrained_mimo import time_constrained_mimo_word_error_rate

    tcmimo = time_constrained_mimo_word_error_rate(reference, reference, collar=0.1)
    assert tcmimo.error_rate == 0 or tcmimo.error_rate is None
    assert tcmimo.reference_self_overlap == tcmimo.hypothesis_self_overlap
