import itertools
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
    optimal_result = meeteval.wer.orcwer(example_files / 'ref?.stm', example_files / 'hyp?.stm')
    greedy_result = meeteval.wer.greedy_orcwer(example_files / 'ref?.stm', example_files / 'hyp?.stm')

    assert optimal_result == greedy_result


@given(segments(10), streams(4))
def test_optimal_assignment_is_not_changed(segments, streams):
    """
    Test that the optimal assignment is not changed.

    Only works with distancetype='1'. The optimal ORC assignment is not
    guaranteed to be optimal for substitution costs other than 1.
    """
    optimal_distance, optimal_assignment = orc_matching.orc_matching_v3(segments, streams)

    greedy_distance, greedy_assignment = greedy_combination_matching.greedy_combination_matching(
        segments, streams, optimal_assignment,
        distancetype='1',
    )

    assert optimal_distance == greedy_distance
    assert optimal_assignment == greedy_assignment


# Limit alphabet to ensure a few correct matches
string = st.text(alphabet='abcdefg', min_size=0, max_size=100)


@st.composite
def string_with_timing(draw):
    """
    Constraints:
        - end >= start
        - start values must be increasing
    """
    s = draw(string)
    t = []
    start = 0
    for _ in s:
        start = draw(st.integers(min_value=start, max_value=10))
        end = draw(st.integers(min_value=start, max_value=start + 10))
        t.append((start, end))
    return s, t


@given(
    string_with_timing(),
    string_with_timing(),
)
def test_greedy_time_constrained_correct(a, b):
    """
    Tests the time-constrained cython matrix implementation against the
    time-constrained distance C++ implementation used in the time-constrained
    siso WER
    """
    from meeteval.wer.matching.cy_levenshtein import time_constrained_levenshtein_distance
    from meeteval.wer.matching.cy_greedy_combination_matching import cy_forward_col_time_constrained
    import numpy as np

    a, a_timing = a
    b, b_timing = b

    # cy_forward_col_time_constrained needs the sequences as integers
    import collections
    sym2int = collections.defaultdict(itertools.count().__next__)
    _ = sym2int['']  # Reserve 0 for the empty string
    a = [sym2int[c] for c in a]
    b = [sym2int[c] for c in b]

    siso_dist = time_constrained_levenshtein_distance(a, b, a_timing, b_timing)

    column = cy_forward_col_time_constrained(
        np.arange(len(a) + 1, dtype=np.uint),
        np.asarray(a, dtype=np.uint),
        np.asarray(b, dtype=np.uint),
        np.asarray([t[0] for t in a_timing], float),
        np.asarray([t[1] for t in a_timing], float),
        np.asarray([t[0] for t in b_timing], float),
        np.asarray([t[1] for t in b_timing], float),
    )

    assert siso_dist == column[-1]
