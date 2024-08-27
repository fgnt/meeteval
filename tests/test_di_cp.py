from hypothesis import given, strategies as st, assume, settings
import meeteval


seglst = st.builds(
    meeteval.io.SegLST,
    st.lists(
        st.builds(
            dict,
            speaker=st.sampled_from(['spkA', 'spkB', 'spkC', 'spkD', 'spkE', 'spkF', 'spkG', 'spkH']),
            words=st.text(),
        ),
        min_size=1,     # For the API wrapper to work, we need at least one element
    )
)


@given(seglst, seglst)
@settings(deadline=None)    # The tests take longer on the GitHub actions test servers
def test_greedy_di_cp_bound_by_cp(ref, hyp):
    cp = meeteval.wer.wer.cp.cp_word_error_rate(ref, hyp)
    dicp = meeteval.wer.wer.di_cp.greedy_di_cp_word_error_rate(ref, hyp)

    assert cp.error_rate is None and dicp.error_rate is None or cp.error_rate >= dicp.error_rate


@given(seglst, seglst)
@settings(deadline=None)    # The tests take longer on the GitHub actions test servers
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
