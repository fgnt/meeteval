from hypothesis import given, strategies as st, assume, settings
import meeteval

seglst = st.builds(
    meeteval.io.SegLST,
    st.lists(
        st.builds(
            lambda **x: {
                'end_time': x['start_time'] + x.pop('duration'),
                **x,
            },
            speaker=st.sampled_from(['spkA', 'spkB', 'spkC', 'spkD', 'spkE', 'spkF', 'spkG', 'spkH']),
            words=st.text(),
            session_id=st.just('session1'),
            start_time=st.floats(min_value=0, max_value=100),
            duration=st.floats(min_value=0, max_value=10),
        ),
    )
)


@given(seglst, seglst)
@settings(deadline=None)  # The tests take longer on the GitHub actions test servers
def test_greedy_di_cp_bound_by_cp(ref, hyp):
    cp = meeteval.wer.wer.cp.cp_word_error_rate(ref, hyp)
    dicp = meeteval.wer.wer.di_cp.greedy_di_cp_word_error_rate(ref, hyp)

    assert cp.error_rate is None and dicp.error_rate is None or cp.error_rate >= dicp.error_rate


@given(seglst, seglst)
@settings(deadline=None)  # The tests take longer on the GitHub actions test servers
def test_greedy_di_cp_vs_greedy_orc(ref, hyp):
    """
    Test that the total distance of the greedy di-cp algorithm is equal to the
    distance computed by the greedy orc algorithm with swapped arguments
    """
    dicp = meeteval.wer.wer.di_cp.greedy_di_cp_word_error_rate(ref, hyp)
    orc = meeteval.wer.wer.orc.greedy_orc_word_error_rate(hyp, ref)

    assert dicp.errors == orc.errors
    assert dicp.substitutions == orc.substitutions
    assert dicp.insertions == orc.deletions
    assert dicp.deletions == orc.insertions


@given(seglst, seglst)
@settings(deadline=None)  # The tests take longer on the GitHub actions test servers
def test_greedy_di_tcp_bound_by_tcp(ref, hyp):
    cp = meeteval.wer.wer.time_constrained.tcp_word_error_rate(ref, hyp, collar=0)  # use collar=5 for real data
    dicp = meeteval.wer.wer.di_cp.greedy_di_tcp_word_error_rate(ref, hyp, collar=0)  # use collar=5 for real data

    assert cp.error_rate is None and dicp.error_rate is None or cp.error_rate >= dicp.error_rate


@given(seglst, seglst)
@settings(deadline=None)  # The tests take longer on the GitHub actions test servers
def test_greedy_di_tcp_vs_greedy_torc(ref, hyp):
    """
    Test that the total distance of the greedy di-tcp algorithm is equal to the
    distance computed by the greedy tcorc algorithm with swapped arguments
    """
    dicp = meeteval.wer.wer.di_cp.greedy_di_tcp_word_error_rate(ref, hyp, collar=0)  # use collar=5 for real data
    orc = meeteval.wer.wer.time_constrained_orc.greedy_time_constrained_orc_wer(
        hyp, ref,
        reference_pseudo_word_level_timing='character_based_points',
        hypothesis_pseudo_word_level_timing='character_based',
        collar=0,  # use collar=5 for real data
    )

    assert dicp.errors == orc.errors
    assert dicp.substitutions == orc.substitutions
    assert dicp.insertions == orc.deletions
    assert dicp.deletions == orc.insertions
