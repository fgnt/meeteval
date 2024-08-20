import pytest
import time

from meeteval.wer.matching import greedy_combination_matching, orc_matching
import meeteval
from hypothesis import given, strategies as st, settings, assume

# Set up hypothesis strategies
# Limit alphabet to ensure a few correct matches. Every character represents
# a word
utterance = st.lists(st.integers(min_value=0, max_value=10), max_size=10)
segments = lambda max_size: st.lists(utterance, min_size=0, max_size=max_size)
streams = lambda max_size: st.lists(utterance, min_size=1, max_size=max_size)


def _check_output(distance, assignment, segments, streams):
    assert isinstance(distance, int)
    assert distance >= 0
    assert len(assignment) == len(segments)
    assert all(isinstance(a, int) for a in assignment)
    assert all(0 <= a < len(streams) for a in assignment)
    d = orc_matching._levensthein_distance_for_assignment(segments, streams, assignment)
    assert d == distance, (d, distance, assignment)


@given(segments(10), streams(4))
@settings(deadline=None)  # The tests take longer on the GitHub actions test servers
def test_greedy_combination_matching_burn(segments, streams):
    """Burn-test. Brute-force is exponential in the number of reference
    utterances, so choose a small number."""
    distance, assignment = greedy_combination_matching.greedy_combination_matching(
        segments, streams, [0] * len(segments)
    )
    _check_output(distance, assignment, segments, streams)


@given(segments(6), streams(4))
def test_greedy_bound_by_optimal(segments, streams):
    greedy_distance, _ = greedy_combination_matching.greedy_combination_matching(
        segments, streams, [0] * len(segments)
    )

    optimal_distance, _ = orc_matching.orc_matching_v3(segments, streams)

    assert optimal_distance <= greedy_distance
