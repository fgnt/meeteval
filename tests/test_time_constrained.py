from hypothesis import given, strategies as st

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
    from meeteval.wer.matching.cy_levenshtein import time_constrained_levenshtein_distance, \
        time_constrained_levenshtein_distance_unoptimized

    a, a_timing = a
    b, b_timing = b

    unoptimized = time_constrained_levenshtein_distance_unoptimized(a, b, a_timing, b_timing)
    optimized = time_constrained_levenshtein_distance(a, b, a_timing, b_timing)

    assert optimized == unoptimized, (optimized, unoptimized)


@given(string_with_timing(), string_with_timing())
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
    st.composite(lambda draw: [
        [
            draw(st.text(alphabet='abcdefg', min_size=1, max_size=3))
            for _ in range(draw(st.integers(min_value=2, max_value=10)))
        ]
        for _ in range(draw(st.integers(min_value=2, max_value=10)))
    ])(),
    st.composite(lambda draw: [
        [
            draw(st.text(alphabet='abcdefg', min_size=1, max_size=3))
            for _ in range(draw(st.integers(min_value=2, max_value=10)))
        ]
        for _ in range(draw(st.integers(min_value=2, max_value=10)))
    ])(),
)
def test_tcpwer_vs_cpwer(
        a, b
):
    from meeteval.wer.wer.time_constrained import time_constrained_minimum_permutation_word_error_rate
    from meeteval.wer.wer.cp import cp_word_error_rate

    cp_statistics = cp_word_error_rate([' '.join(speaker) for speaker in a], [' '.join(speaker) for speaker in b])
    tcp_statistics = time_constrained_minimum_permutation_word_error_rate(
        [[{'words': word, 'start_time': 0, 'end_time': 1} for word in speaker] for speaker in a],
        [[{'words': word, 'start_time': 0, 'end_time': 1} for word in speaker] for speaker in b],
    )
    assert cp_statistics == tcp_statistics, (cp_statistics, tcp_statistics)


def test_tcpwer_input_formats():
    from meeteval.wer.wer.time_constrained import time_constrained_minimum_permutation_word_error_rate, \
        TimeMarkedTranscript
    from meeteval.io.stm import STM, STMLine

    r1 = time_constrained_minimum_permutation_word_error_rate(
        [TimeMarkedTranscript(['a'], [(0, 1)]), TimeMarkedTranscript(['b c'], [(1, 2)])],
        [TimeMarkedTranscript(['a b'], [(0, 1)]), TimeMarkedTranscript(['c'], [(1, 2)])],
    )
    r2 = time_constrained_minimum_permutation_word_error_rate(
        [[{'words': 'a', 'start_time': 0, 'end_time': 1}], [{'words': 'b c', 'start_time': 1, 'end_time': 2}]],
        [[{'words': 'a b', 'start_time': 0, 'end_time': 1}], [{'words': 'c', 'start_time': 1, 'end_time': 2}]],
    )
    r3 = time_constrained_minimum_permutation_word_error_rate(
        [
            STM([STMLine('dummy', 0, 'A', 0, 1, 'a')]),
            STM([STMLine('dummy', 1, 'A', 1, 2, 'b c')])
        ],
        [
            STM([STMLine('dummy', 0, 'A', 0, 1, 'a b')]),
            STM([STMLine('dummy', 1, 'A', 1, 2, 'c')])
        ]
    )
    r4 = time_constrained_minimum_permutation_word_error_rate(
        {'A': TimeMarkedTranscript(['a'], [(0, 1)]), 'B': TimeMarkedTranscript(['b c'], [(1, 2)])},
        {'A': TimeMarkedTranscript(['a b'], [(0, 1)]), 'B': TimeMarkedTranscript(['c'], [(1, 2)])},
    )
    r5 = time_constrained_minimum_permutation_word_error_rate(
        {'A': [{'words': 'a', 'start_time': 0, 'end_time': 1}],
         'B': [{'words': 'b c', 'start_time': 1, 'end_time': 2}]},
        {'A': [{'words': 'a b', 'start_time': 0, 'end_time': 1}],
         'B': [{'words': 'c', 'start_time': 1, 'end_time': 2}]},
    )
    assert r1.error_rate == r2.error_rate
    assert r1.error_rate == r3.error_rate
    assert r1.error_rate == r4.error_rate
    assert r1.error_rate == r5.error_rate
