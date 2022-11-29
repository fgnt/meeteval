import pytest

from meeteval.wer.matching import orc_matching
from hypothesis import given, strategies as st, reproduce_failure, settings  # pip install hypothesis

# Set up hypothesis strategies
# Limit alphabet to ensure a few correct matches. Every character represents
# a word
utterance = st.text(alphabet='abcdefg', min_size=0, max_size=10)
reference = lambda max_size: st.lists(utterance, min_size=0, max_size=max_size)
hypothesis = lambda max_size: st.lists(utterance, min_size=1, max_size=max_size)


def _check_output(distance, assignment, ref, hyps):
    assert isinstance(distance, int)
    assert distance >= 0
    assert len(assignment) == len(ref)
    assert all(isinstance(a, int) for a in assignment)
    assert all(0 <= a < len(hyps) for a in assignment)
    d = orc_matching._levensthein_distance_for_assignment(ref, hyps, assignment)
    assert d == distance, (d, distance, assignment)


@given(reference(6), hypothesis(4))
@settings(deadline=None)    # The tests take longer on the GitHub actions test servers
def test_orc_matching_v1_burn(ref, hyps):
    """Burn-test. Brute-force is exponential in the number of reference
    utterances, so choose a small number."""
    distance, assignment = orc_matching.orc_matching_v1(ref, hyps)
    _check_output(distance, assignment, ref, hyps)


@given(reference(7), hypothesis(3))
@settings(deadline=None)    # The tests take longer on the GitHub actions test servers
def test_orc_matching_v2_burn(ref, hyps):
    """Burn-test. This algorithm is slow, so we can only test for small
    numbers of reference utterances/hypothesis"""
    distance, assignment = orc_matching.orc_matching_v2(ref, hyps)
    _check_output(distance, assignment, ref, hyps)


@given(reference(10), hypothesis(4))
@settings(deadline=None)    # The tests take longer on the GitHub actions test servers
def test_orc_matching_v3_burn(ref, hyps):
    """Burn-test."""
    distance, assignment = orc_matching.orc_matching_v3(ref, hyps)
    _check_output(distance, assignment, ref, hyps)


@pytest.mark.parametrize(
    'matching_function',
    (orc_matching.orc_matching_v2, orc_matching.orc_matching_v3)
)
@given(ref=reference(7), hyps=hypothesis(3))
@settings(deadline=None)    # The tests take longer on the GitHub actions test servers
def test_orclevenshtein_against_brute_force(matching_function, ref, hyps):
    """Test brute-force algorithm against DP algorithm for smaller examples"""
    distance, assignment = matching_function(ref, hyps)
    bf_distance, bf_assignment = orc_matching.orc_matching_v1(ref, hyps)

    assert distance == bf_distance

    # We cannot easily check the assignment because there might be multiple
    # assignments with the same distance. Check that the distance for them is
    # equal instead
    assert assignment == bf_assignment or (
        orc_matching._levensthein_distance_for_assignment(ref, hyps, assignment) ==
        orc_matching._levensthein_distance_for_assignment(ref, hyps, bf_assignment)
    )
