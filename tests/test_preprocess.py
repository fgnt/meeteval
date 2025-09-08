from pathlib import Path
import pytest
import meeteval
from meeteval.wer.preprocess import _preprocess_single

example_files = (Path(__file__).parent.parent / 'example_files').absolute()


@pytest.fixture
def example_seglst():
    return meeteval.io.load(example_files / 'hyp.seglst.json')


@pytest.mark.parametrize('keep_keys', [None])
@pytest.mark.parametrize('sort', ['word', 'segment', False])
@pytest.mark.parametrize('remove_empty_segments', ['True', 'False'])
@pytest.mark.parametrize('word_level_timing_strategy', [None, 'equidistant_points', 'character_based_points'])
@pytest.mark.parametrize('collar', [0, 5])
@pytest.mark.parametrize('segment_index', ['word', 'segment', False])
@pytest.mark.parametrize('segment_representation', ['segment', 'word', 'speaker'])
def test_preprocess_single(
        example_seglst,
        keep_keys,
        sort,
        remove_empty_segments,
        word_level_timing_strategy,
        collar,
        segment_index,
        segment_representation,
):
    # Skip tests with invalid configurations
    # TODO: can we do this in a more elegant way?
    if (
            segment_index == 'word' and segment_representation != 'word'
            or sort == 'word' and segment_index != 'word'
    ):
        pytest.skip()

    _preprocess_single(
        example_seglst,
        keep_keys=keep_keys,
        sort=sort,
        remove_empty_segments=remove_empty_segments,
        word_level_timing_strategy=word_level_timing_strategy,
        collar=collar,
        segment_index=segment_index,
        segment_representation=segment_representation,
    )


def test_preprocess_sort_false(example_seglst):
    processed, _ = _preprocess_single(
        example_seglst,
        sort=False,
        segment_representation='segment',
        collar=None,
    )
    assert processed.T['start_time'] == example_seglst.T['start_time']


def test_preprocess_sort_segment(example_seglst):
    processed, _ = _preprocess_single(
        example_seglst, sort='segment',
        segment_representation='segment',
        collar=None,

    )
    assert processed.T['start_time'] == sorted(example_seglst.T['start_time'])


def test_preprocess_sort_word(example_seglst):
    processed, _ = _preprocess_single(
        example_seglst, sort='word',
        segment_representation='word',
        collar=None,

    )
    assert processed.T['start_time'] == sorted(processed.T['start_time'])


def test_preprocess_segment_representation_word(example_seglst):
    processed, _ = _preprocess_single(
        example_seglst, segment_representation='word', collar=None,
    )
    assert not any(' ' in words for words in processed.T['words'])


def test_preprocess_segment_representation_segment(example_seglst):
    processed, _ = _preprocess_single(
        example_seglst, segment_representation='segment', collar=None,
    )
    assert len(example_seglst) == len(processed)


def test_preprocess_segment_representation_speaker(example_seglst):
    processed, _ = _preprocess_single(
        example_seglst, segment_representation='speaker', collar=None,
    )
    assert set(processed.T['speaker']) == set(example_seglst.T['speaker'])
    assert len(processed) == len(set(example_seglst.T['speaker']))


def test_preprocess_remove_empty_segments(example_seglst):
    processed, _ = _preprocess_single(
        example_seglst,
        remove_empty_segments=True,
        segment_representation='segment',
        collar=None,
    )
    # No empty segments in example
    assert len(example_seglst) == len(processed)

    # Insert empty examples
    example_seglst_empty = meeteval.io.SegLST.merge(
        example_seglst,
        meeteval.io.SegLST([{
            'start_time': 0,
            'end_time': 1,
            'words': '',
        }])
    )
    processed, _ = _preprocess_single(
        example_seglst_empty,
        remove_empty_segments=True,
        segment_representation='segment',
    )
    # No empty segments in example
    assert len(example_seglst) == len(processed)

def test_preprocess_remove_empty_segments(example_seglst):
    # Insert empty examples
    example_seglst_empty = meeteval.io.SegLST.merge(
        example_seglst,
        meeteval.io.SegLST([{
            'start_time': 0,
            'end_time': 1,
            'words': '',
        }])
    )
    processed, _ = _preprocess_single(
        example_seglst_empty,
        remove_empty_segments=False,
        segment_representation='segment',
        collar=None,
    )
    # No empty segments in example
    assert len(example_seglst_empty) == len(processed)

def test_preprocess_zero_collar_warn(example_seglst, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        _preprocess_single(
            example_seglst, sort='segment',
            segment_representation='word',
            collar=0,
        )
    assert 'Collar is set to 0' in caplog.text


def test_preprocess_small_collar_warn(example_seglst, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        _preprocess_single(
            example_seglst, sort='segment',
            segment_representation='word',
            collar=1,
        )
    assert 'The mean word length is 1.13 seconds, which is more than the collar length of 1 seconds.' in caplog.text, caplog.text
