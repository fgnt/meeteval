from hypothesis import given, strategies as st

# Limit alphabet to ensure a few correct matches
string = st.text(alphabet='abcdefg', min_size=0, max_size=100)


@st.composite
def string_with_timing(draw):
    """
    Constraints:
        - end >= start
        - start values must be increasing
        - end values must be increasing
    """
    s = draw(string)
    t = []
    start = 0
    end = 0
    for _ in s:
        start = draw(st.integers(min_value=start, max_value=10))
        e = end if end > start else start
        end = draw(st.integers(min_value=e + 1, max_value=e + 10))
        t.append((start, end))
    return s, t


@given(string, string)
def test_time_constrained_against_levenshtein(a, b):
    """Check that the Levenshtein distance with time constraints is equal to computation without time constraints
    when all time intervals span the full length"""
    from meeteval.wer.matching.cy_levenshtein import levenshtein_distance, time_constrained_levenshtein_distance

    distance_time_constrained = time_constrained_levenshtein_distance(a, b, [(0, 1)] * len(a), [(0, 1)] * len(b))
    distance_unconstrained = levenshtein_distance(a, b)

    assert distance_time_constrained == distance_unconstrained


@given(string_with_timing(), string_with_timing())
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
def test_time_constrained_levenshtein_distance_optimized(a, b):
    """Check whether the pruning optimization always yields the correct result"""
    from meeteval.wer.matching.cy_levenshtein import time_constrained_levenshtein_distance, time_constrained_levenshtein_distance_unoptimized

    a, a_timing = a
    b, b_timing = b

    unoptimized = time_constrained_levenshtein_distance_unoptimized(a, b, a_timing, b_timing)
    optimized = time_constrained_levenshtein_distance(a, b, a_timing, b_timing)

    assert optimized == unoptimized, (optimized, unoptimized)


@given(string_with_timing(), string_with_timing())
def test_time_constrained_levenshtein_distance_vs_with_alignment(a, b):
    """Check that the alignment function gives the same distance as the version that does not return the alignment"""
    from meet_eval.levenshtein import time_aligned_levenshtein_distance, \
        time_aligned_levenshtein_distance_with_alignment

    a, a_timing = a
    b, b_timing = b

    ref_distance = time_aligned_levenshtein_distance(a, b, a_timing, b_timing)
    s = time_aligned_levenshtein_distance_with_alignment(a, b, a_timing, b_timing)
    distance = s['total']

    assert ref_distance == distance, (ref_distance, distance)

    # Check that the alignment has the right format
    assert max(len(a), len(b)) <= len(s['alignment']) <= len(a) + len(b)
    assert ''.join([a[0] for a in s['alignment'] if a[0] != '*']) == a
    assert ''.join([a[1] for a in s['alignment'] if a[1] != '*']) == b


@given(string, string)
def test_time_constrained_levenshtein_distance_with_alignment_against_kaldialign(a, b):
    """Check that the returned alignment matches kaldialign then the time intervals span the full length"""
    from kaldialign import align, edit_distance
    from meeteval.wer.matching import time_constrained_levenshtein_distance_with_alignment

    a_timing = [(0, 1)] * len(a)
    b_timing = [(0, 1)] * len(b)

    kaldialign_alignment = align(a, b, '*')
    kaldialign_statistics = edit_distance(a, b)
    statistics = time_constrained_levenshtein_distance_with_alignment(a, b, a_timing, b_timing)
    alignment = statistics.pop('alignment')
    assert alignment == kaldialign_alignment, (alignment, kaldialign_alignment)
    assert kaldialign_statistics['ins'] == statistics['insertions'], (kaldialign_statistics, statistics)
    assert kaldialign_statistics['sub'] == statistics['substitutions'], (kaldialign_statistics, statistics)
    assert kaldialign_statistics['total'] == statistics['total'], (kaldialign_statistics, statistics)
    assert kaldialign_statistics['del'] == statistics['deletions'], (kaldialign_statistics, statistics)
