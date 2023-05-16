import pytest

from meeteval.wer.matching import orc_matching, mimo_matching
from hypothesis import given, strategies as st, settings  # pip install hypothesis

from meeteval.wer.wer.mimo import apply_mimo_assignment

# Set up hypothesis strategies
# Limit alphabet to ensure a few correct matches. Every character represents
# a word
utterance = st.text(alphabet='abcdefg', min_size=1, max_size=10)


def reference(max_references, max_utterances):
    return st.lists(
        st.lists(utterance, min_size=1, max_size=max_utterances),
        max_size=max_references,
        min_size=1,
    )


def orc_like_reference(max_utterances):
    return reference(max_references=1, max_utterances=max_utterances)


def hypothesis(max_channels):
    return st.lists(utterance, min_size=1, max_size=max_channels)


def _levensthein_distance_for_assignment(ref, hyps, assignment):
    import editdistance
    ref_, hyp_ = apply_mimo_assignment(assignment, ref, hyps)
    d = sum([editdistance.distance([r__ for r_ in h for r__ in r_], r) for h, r in zip(ref_, hyp_)])
    return d


def _check_output(distance, assignment, ref, hyps):
    assert isinstance(distance, int)
    assert distance >= 0
    assert len(assignment) == sum(len(r) for r in ref)
    assert all(isinstance(a, tuple) for a in assignment)
    assert all(0 <= r < len(ref) for r, h in assignment)
    assert all(0 <= h < len(hyps) for r, h in assignment)
    d = _levensthein_distance_for_assignment(ref, hyps, assignment)
    assert d == distance, (d, distance, assignment)


@given(reference(2, 2), hypothesis(3))
@settings(deadline=None)    # The tests take longer on the GitHub actions test servers
def test_mimo_matching_v2_burn(ref, hyps):
    """Burn-test. This algorithm is slow, so we can only test for small
    numbers of reference utterances/hypothesis"""
    distance, assignment = mimo_matching.mimo_matching_v2(ref, hyps)
    _check_output(distance, assignment, ref, hyps)


@given(reference(3, 8), hypothesis(3))
@settings(deadline=None)    # The tests take longer on the GitHub actions test servers
def test_mimo_matching_v3_burn(ref, hyps):
    """Burn-test."""
    distance, assignment = mimo_matching.mimo_matching_v3(ref, hyps)
    _check_output(distance, assignment, ref, hyps)


@given(reference(3, 8), hypothesis(3))
@settings(deadline=None)    # The tests take longer on the GitHub actions test servers
def test_mimo_matching_v4_burn(ref, hyps):
    """Burn-test."""
    distance, assignment = mimo_matching.mimo_matching_v4(ref, hyps)
    _check_output(distance, assignment, ref, hyps)


@given(ref=reference(2, 2), hyps=hypothesis(3))
@settings(deadline=None)    # The tests take longer on the GitHub actions test servers
def test_mimo_v2_against_v3(ref, hyps):
    """Test brute-force algorithm against DP algorithm for smaller examples"""
    distance_v2, assignment_v2 = mimo_matching.mimo_matching_v2(ref, hyps)
    distance_v3, assignment_v3 = mimo_matching.mimo_matching_v3(ref, hyps)

    assert distance_v2 == distance_v3

    # We cannot easily check the assignment because there might be multiple
    # assignments with the same distance. Check that the distance for them is
    # equal instead
    assert assignment_v2 == assignment_v3 or (
            _levensthein_distance_for_assignment(ref, hyps, assignment_v2) ==
            _levensthein_distance_for_assignment(ref, hyps, assignment_v3)
    )


@given(ref=orc_like_reference(5), hyps=hypothesis(3))
@settings(deadline=None)    # The tests take longer on the GitHub actions test servers
def test_against_orc(ref, hyps):
    mimo_distance, mimo_assignment = mimo_matching.mimo_matching_v3(ref, hyps)
    orc_distance, orc_assignment = orc_matching.orc_matching_v3(ref[0], hyps)

    assert mimo_distance == orc_distance
    assert tuple([x[1] for x in mimo_assignment]) == tuple(orc_assignment) or (
            _levensthein_distance_for_assignment(ref, hyps, mimo_assignment) ==
            orc_matching._levensthein_distance_for_assignment(ref[0], hyps, orc_assignment)
    )


@given(ref=reference(2, 2), hyps=hypothesis(3))
@settings(deadline=None)    # The tests take longer on the GitHub actions test servers
def test_mimo_v2_against_v4(ref, hyps):
    """Test brute-force algorithm against DP algorithm for smaller examples"""
    distance_v2, assignment_v2 = mimo_matching.mimo_matching_v2(ref, hyps)
    distance_v4, assignment_v4 = mimo_matching.mimo_matching_v4(ref, hyps)

    assert distance_v2 == distance_v4

    # We cannot easily check the assignment because there might be multiple
    # assignments with the same distance. Check that the distance for them is
    # equal instead
    assert assignment_v2 == assignment_v4 or (
            _levensthein_distance_for_assignment(ref, hyps, assignment_v2) ==
            _levensthein_distance_for_assignment(ref, hyps, assignment_v4)
    )
