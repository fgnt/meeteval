import pytest

from meeteval.wer.wer.error_rate import ErrorRate
from meeteval.wer.wer.cp import CPErrorRate
from meeteval.wer.wer.orc import OrcErrorRate
from meeteval.wer.wer.mimo import MimoErrorRate
from meeteval.wer.wer.di_cp import DICPErrorRate
from meeteval.der.md_eval import DiaErrorRate
from hypothesis import given, strategies as st


all_word_error_rates = [
    ErrorRate,
    CPErrorRate,
    OrcErrorRate,
    MimoErrorRate,
    DICPErrorRate,
]

all_error_rates = all_word_error_rates + [DiaErrorRate]

@st.composite
def random_error_rate(draw, error_rate_cls):

    if isinstance(error_rate_cls, list):
        error_rate_cls = draw(st.sampled_from(error_rate_cls))

    if error_rate_cls is DiaErrorRate:
        # Extreme float values do not appear in practice and cause numerical 
        # issues in the serialization
        floats = st.floats(
            min_value=0, 
            max_value=1e5, 
            allow_infinity=False, 
            allow_nan=False,
            allow_subnormal=False
        )
        return DiaErrorRate(
            None,
            scored_speaker_time=draw(floats),
            missed_speaker_time=draw(floats),
            falarm_speaker_time=draw(floats),
            speaker_error_time=draw(floats),
        )
    else:
        integers = st.integers(min_value=0)
        insertions = draw(integers)
        deletions = draw(integers)
        substitutions = draw(integers)
        length = draw(st.integers(min_value=substitutions + deletions))

        if error_rate_cls in [
            OrcErrorRate,
            MimoErrorRate,
            DICPErrorRate,
        ]:
            assignment = ()
            return error_rate_cls(
                errors=insertions + deletions + substitutions,
                length=length,
                insertions=insertions,
                deletions=deletions,
                substitutions=substitutions,
                reference_self_overlap=None,
                hypothesis_self_overlap=None,
                assignment=assignment,
            )
        elif error_rate_cls is CPErrorRate:
            return CPErrorRate(
                errors=insertions + deletions + substitutions,
                length=length,
                insertions=insertions,
                deletions=deletions,
                substitutions=substitutions,
                reference_self_overlap=None,
                hypothesis_self_overlap=None,
                assignment={},
                falarm_speaker=draw(integers),
                missed_speaker=draw(integers),
                scored_speaker=draw(integers),
            )
        else:
            return ErrorRate(
                errors=insertions + deletions + substitutions,
                length=length,
                insertions=insertions,
                deletions=deletions,
                substitutions=substitutions,
                reference_self_overlap=None,
                hypothesis_self_overlap=None,
            )


@st.composite
def list_of_error_rates(draw, error_rate_cls=all_error_rates):
    error_rate_cls = draw(st.sampled_from(error_rate_cls))
    error_rate_list = draw(st.lists(random_error_rate(error_rate_cls), min_size=1))
    return error_rate_list

@given(list_of_error_rates(all_word_error_rates))
def test_sum_error_rates(list_of_error_rates):
    """Test that the sum of error rates works correctly"""
    total_errors = sum([er.errors for er in list_of_error_rates])
    total_insertions = sum([er.insertions for er in list_of_error_rates])
    total_deletions = sum([er.deletions for er in list_of_error_rates])
    total_substitutions = sum([er.substitutions for er in list_of_error_rates])
    summed = sum(list_of_error_rates)
    assert isinstance(summed, ErrorRate)
    assert summed.errors == total_errors
    assert summed.insertions == total_insertions
    assert summed.deletions == total_deletions
    assert summed.substitutions == total_substitutions

@pytest.mark.parametrize('cls', all_error_rates)
def test_zero(cls: ErrorRate):
    """Test that the zero function returns the right type and an error_rate of 0"""
    er = cls.zero()
    assert isinstance(er, cls)
    assert er.error_rate is None or er.error_rate == 0

@given(random_error_rate(all_error_rates))
def test_serialize(error_rate):
    serialized = error_rate.asdict()
    assert isinstance(serialized, dict)
    assert 'type' in serialized
    reconstructed = ErrorRate.from_dict(serialized)
    assert reconstructed == error_rate
