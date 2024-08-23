from pathlib import Path

import pytest
import time

from meeteval.wer.matching import greedy_combination_matching, orc_matching
import meeteval
from hypothesis import given, strategies as st, settings, assume

example_files = (Path(__file__).parent.parent / 'example_files').absolute()

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


def test_example_files_orc_equals_greedy_orc():
    optimal_result = meeteval.wer.orcwer(example_files / 'ref.stm', example_files / 'hyp.stm')
    greedy_result = meeteval.wer.greedy_orcwer(example_files / 'ref.stm', example_files / 'hyp.stm')

    assert optimal_result == greedy_result


@given(segments(10), streams(4), st.sampled_from(['1', '21']))
def test_optimal_assignment_is_not_changed(segments, streams, distancetype):
    """
    Test that the optimal assignment is not changed.

    Only works with distancetype='1'. The optimal ORC assignment is not
    guaranteed to be optimal for other substitution costs than 1.
    For distanctype='21' we can at least guarantee that the greedily corrected
    assignment is not worse than the optimal assignment, but we cannot guarantee
    equality.
    """
    optimal_distance, optimal_assignment = orc_matching.orc_matching_v3(segments, streams)

    greedy_distance, greedy_assignment = greedy_combination_matching.greedy_combination_matching(
        segments, streams, optimal_assignment,
        distancetype=distancetype,
    )

    assert optimal_distance == greedy_distance
    if distancetype == '1':
        assert optimal_assignment == greedy_assignment
