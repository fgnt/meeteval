from hypothesis import given, strategies as st
from meeteval.wer.matching.cy_levenshtein import levenshtein_distance

# Limit alphabet to ensure a few correct matches
string = st.text(alphabet='abcdefg', min_size=0, max_size=100)


@given(string, string)
def test_against_editdistance(a, b):
    """Test against the well-known `editdistance` package"""
    import editdistance

    ref_distance = editdistance.distance(a, b)
    distance = levenshtein_distance(a, b)

    assert ref_distance == distance, (ref_distance, distance)


@given(string, string)
def test_custom_cost_against_levenshtein_default(a, b):
    assert levenshtein_distance(a, b, 1, 1, 1, 0) == levenshtein_distance(a, b)
    assert levenshtein_distance(a, b, 2, 2, 2, 0) == 2 * levenshtein_distance(a, b)


@given(string, string)
def test_custom_cost(a, b):
    """
    Test a few weird special configurations
    """
    assert levenshtein_distance(a, b, 0, 0, 0, 0) == 0
    assert levenshtein_distance(a, b, 1, 1, 0, 0) == max(len(a), len(b)) - min(len(a), len(b))

    # Manhattan distance in levenshtein matrix
    assert levenshtein_distance(a, b, 1, 1, 2, 2) == len(a) + len(b)
