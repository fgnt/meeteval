from pathlib import Path

import pytest
from hypothesis import settings, given, strategies as st

from meeteval.io.ctm import CTMGroup, CTM
from meeteval.io.seglst import SegLST

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


@given(string, string)
@settings(deadline=None)
def test_time_constrained_against_levenshtein(a, b):
    """Check that the Levenshtein distance with time constraints is equal to computation without time constraints
    when all time intervals span the full length"""
    from meeteval.wer.matching.cy_levenshtein import levenshtein_distance, time_constrained_levenshtein_distance

    distance_time_constrained = time_constrained_levenshtein_distance(a, b, [(0, 1)] * len(a), [(0, 1)] * len(b))
    distance_unconstrained = levenshtein_distance(a, b)

    assert distance_time_constrained == distance_unconstrained


@given(string_with_timing(), string_with_timing())
@settings(deadline=None)
def test_time_constrained_levenshtein_distance_within_bounds(a, b):
    """Test that the distance is bound by the unconstrained Levenshtein distance (lower bound) and the max len
    of the inputs (upper bound)"""
    from meeteval.wer.matching.cy_levenshtein import levenshtein_distance, time_constrained_levenshtein_distance

    a, a_timing = a
    b, b_timing = b
    dist = time_constrained_levenshtein_distance(a, b, a_timing, b_timing)

    assert dist <= len(a) + len(b)
    assert dist >= levenshtein_distance(a, b)


@given(string_with_timing(), string_with_timing())
@settings(deadline=None)
def test_time_constrained_levenshtein_distance_optimized(a, b):
    """Check whether the pruning optimization always yields the correct result"""
    from meeteval.wer.matching.cy_levenshtein import time_constrained_levenshtein_distance

    a, a_timing = a
    b, b_timing = b

    unoptimized = time_constrained_levenshtein_distance(a, b, a_timing, b_timing, prune=False)
    optimized = time_constrained_levenshtein_distance(a, b, a_timing, b_timing, prune=True)

    assert optimized == unoptimized, (optimized, unoptimized)


@given(string_with_timing(), string_with_timing())
@settings(deadline=None)
def test_time_constrained_levenshtein_distance_vs_with_alignment(a, b):
    """Check that the alignment function gives the same distance as the version that does not return the alignment"""
    from meeteval.wer.matching.cy_levenshtein import time_constrained_levenshtein_distance, \
        time_constrained_levenshtein_distance_with_alignment

    a, a_timing = a
    b, b_timing = b

    ref_distance = time_constrained_levenshtein_distance(a, b, a_timing, b_timing)
    s = time_constrained_levenshtein_distance_with_alignment(a, b, a_timing, b_timing)
    distance = s['total']

    assert ref_distance == distance, (ref_distance, distance)
    assert s['insertions'] + s['deletions'] + s['substitutions'] == distance, s

    # Check that the alignment has the right format
    assert max(len(a), len(b)) <= len(s['alignment']) <= len(a) + len(b)
    assert [a[0] for a in s['alignment'] if a[0] is not None] == list(range(len(a)))
    assert [a[1] for a in s['alignment'] if a[1] is not None] == list(range(len(b)))
    assert len([a for a in s['alignment'] if a[0] is None or a[1] is None]) <= distance


@given(string, string)
@settings(deadline=None)
def test_time_constrained_levenshtein_distance_with_alignment_against_kaldialign(a, b):
    """Check that the returned alignment matches kaldialign then the time intervals span the full length"""
    from kaldialign import align as kaldi_align, edit_distance
    from meeteval.wer.wer.time_constrained import index_alignment_to_kaldi_alignment
    from meeteval.wer.matching import time_constrained_levenshtein_distance_with_alignment

    a_timing = [(0, 1)] * len(a)
    b_timing = [(0, 1)] * len(b)

    kaldialign_alignment = kaldi_align(a, b, '*')
    kaldialign_statistics = edit_distance(a, b)
    statistics = time_constrained_levenshtein_distance_with_alignment(a, b, a_timing, b_timing)
    alignment = statistics.pop('alignment')
    alignment = index_alignment_to_kaldi_alignment(alignment, a, b)
    assert alignment == kaldialign_alignment, (alignment, kaldialign_alignment)
    assert kaldialign_statistics['ins'] == statistics['insertions'], (kaldialign_statistics, statistics)
    assert kaldialign_statistics['sub'] == statistics['substitutions'], (kaldialign_statistics, statistics)
    assert kaldialign_statistics['total'] == statistics['total'], (kaldialign_statistics, statistics)
    assert kaldialign_statistics['del'] == statistics['deletions'], (kaldialign_statistics, statistics)


@given(
    st.lists(st.lists(string, min_size=2, max_size=10), min_size=2, max_size=10),
    st.lists(st.lists(string, min_size=2, max_size=10), min_size=2, max_size=10),
)
@settings(deadline=None)
def test_tcpwer_vs_cpwer(a, b):
    from meeteval.wer.wer.time_constrained import time_constrained_minimum_permutation_word_error_rate
    from meeteval.wer.wer.cp import cp_word_error_rate

    cp_statistics = cp_word_error_rate([' '.join(speaker) for speaker in a], [' '.join(speaker) for speaker in b])
    tcp_statistics = time_constrained_minimum_permutation_word_error_rate(
        SegLST([
            {'words': word, 'start_time': 0, 'end_time': 1, 'speaker': speaker_id}
            for speaker_id, speaker in enumerate(a) for word in speaker
        ]),
        SegLST([
            {'words': word, 'start_time': 0, 'end_time': 1, 'speaker': speaker_id}
            for speaker_id, speaker in enumerate(b) for word in speaker
        ]),
    )
    from dataclasses import replace
    tcp_statistics = replace(tcp_statistics, reference_self_overlap=None, hypothesis_self_overlap=None)
    assert cp_statistics == tcp_statistics, (cp_statistics, tcp_statistics)


def test_tcpwer_input_formats():
    from meeteval.wer.wer.time_constrained import time_constrained_minimum_permutation_word_error_rate
    from meeteval.io.stm import STM

    r1 = time_constrained_minimum_permutation_word_error_rate(
        SegLST([
            {'words': 'a', 'start_time': 0, 'end_time': 1, 'speaker': 'A'},
            {'words': 'b c', 'start_time': 1, 'end_time': 2, 'speaker': 'B'}
        ]),
        SegLST([
            {'words': 'a b', 'start_time': 0, 'end_time': 1, 'speaker': 'A'},
            {'words': 'c', 'start_time': 1, 'end_time': 2, 'speaker': 'B'}
        ]),
    )
    r2 = time_constrained_minimum_permutation_word_error_rate(
        STM.parse('dummy 1 A 0 1 a\ndummy 1 A 1 2 b c'),
        STM.parse('dummy 1 A 0 1 a b\ndummy 1 A 1 2 c'),
    )
    r3 = time_constrained_minimum_permutation_word_error_rate(
        CTMGroup({'A': CTM.parse("dummy 1 0 1 a\ndummy 1 1 0.5 b\ndummy 1 1.5 0.5 c")}),
        CTMGroup({0: CTM.parse("dummy 1 0 0.5 a\ndummy 1 0.5 0.5 b\ndummy 1 1 1 c")})
    )
    assert r1.error_rate == r2.error_rate
    assert r1.error_rate == r3.error_rate


def test_time_constrained_sorting_options():
    from meeteval.wer.wer.time_constrained import time_constrained_minimum_permutation_word_error_rate

    r1 = SegLST([
        {'words': 'a b', 'start_time': 0, 'end_time': 1, 'speaker': 'A'},
        {'words': 'b c', 'start_time': 0, 'end_time': 1, 'speaker': 'A'}
    ])

    # "True" checks whether word order matches the word-level timestamps.
    # Here, it doesn't match, so ValueError is raised.
    with pytest.raises(ValueError):
        time_constrained_minimum_permutation_word_error_rate(
            r1, r1, reference_sort=True, hypothesis_sort=True
        )

    er = time_constrained_minimum_permutation_word_error_rate(
        r1, r1, reference_sort='word', hypothesis_sort='word'
    )
    assert er.error_rate == 0

    r1 = SegLST([
        {'words': 'a b c d', 'start_time': 0, 'end_time': 4, 'speaker': 'A'},
        {'words': 'e f g h', 'start_time': 2, 'end_time': 6, 'speaker': 'A'},
    ])
    r2 = SegLST([
        {'words': 'a b c d e f g h', 'start_time': 0, 'end_time': 6, 'speaker': 'A'},
    ])
    er = time_constrained_minimum_permutation_word_error_rate(
        r1, r2, reference_sort='word'
    )
    assert er.error_rate == 0.75

    er = time_constrained_minimum_permutation_word_error_rate(
        r1, r2, reference_sort='segment'
    )
    assert er.error_rate == 0.75

    # With collar: "segment" keeps word order, so the error becomes 0
    er = time_constrained_minimum_permutation_word_error_rate(
        r1, r2, reference_sort='segment', collar=1
    )
    assert er.error_rate == 0

    # With collar: "word" does not keep word order, so the overlap gets penalized
    er = time_constrained_minimum_permutation_word_error_rate(
        r1, r2, reference_sort='word', collar=1
    )
    assert er.error_rate == 0.25

    # False means the user provides the sorting, so we can pass anything
    r1 = SegLST([
        {'words': 'e f g h', 'start_time': 4, 'end_time': 8, 'speaker': 'A'},
        {'words': 'a b c d', 'start_time': 0, 'end_time': 4, 'speaker': 'A'},
    ])
    r2 = SegLST([
        {'words': 'a b c d e f g h', 'start_time': 0, 'end_time': 8, 'speaker': 'A'},
    ])
    er = time_constrained_minimum_permutation_word_error_rate(
        r1, r2, reference_sort='segment',
    )
    assert er.error_rate == 0

    er = time_constrained_minimum_permutation_word_error_rate(
        r1, r2, reference_sort=False, hypothesis_sort=False,
    )
    assert er.error_rate == 1


def test_examples_zero_self_overlap():
    """Tests that self-overlap is measured correctly (0) for the example files"""
    example_files = (Path(__file__).parent.parent / 'example_files').absolute()

    from meeteval.wer import tcpwer
    wers = tcpwer(example_files / 'ref.stm', example_files / 'hyp.stm', collar=5)
    for k, wer in wers.items():
        assert wer.reference_self_overlap.overlap_time == 0, (k, wer)
        assert wer.hypothesis_self_overlap.overlap_time == 0, (k, wer)
