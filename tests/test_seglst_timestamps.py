import decimal
import json

import pytest

from meeteval.io import SegLST
from meeteval.wer.preprocess import split_words


@pytest.mark.parametrize('parse_float', [decimal.Decimal, float])
@pytest.mark.parametrize('timestamps', [
    [0, 1], [0.125, 1.25], ['0.125', '1.25'], [0, '1.25'], [],
])
def test_parse_word_timestamp_lists(parse_float, timestamps):
    text = json.dumps([{
        'session_id': 'recording', 'words': ['a', 'b'][:len(timestamps)],
        'start_time': timestamps, 'end_time': timestamps,
    }])
    parsed = SegLST.parse(text, parse_float=parse_float)[0]
    expected = [parse_float(str(t)) for t in timestamps]
    assert parsed['start_time'] == expected
    assert parsed['end_time'] == expected
    assert all(isinstance(t, (int, float) if parse_float is float else decimal.Decimal)
               for t in parsed['start_time'] + parsed['end_time'])


@pytest.mark.parametrize('strategy', [
    'equidistant_intervals', 'equidistant_points', 'character_based',
    'character_based_points', 'full_segment',
])
def test_preprocessed_word_timestamps_round_trip(strategy, tmp_path):
    segments = SegLST([{
        'session_id': 'recording', 'speaker': 'Alice', 'words': 'a longer word',
        'start_time': decimal.Decimal('0.125'),
        'end_time': decimal.Decimal('1.625'),
    }])
    timed = split_words(segments, word_level_timing_strategy=strategy,
                        segment_representation='segment')
    assert SegLST.parse(timed.dumps()) == timed
    path = tmp_path / 'words.seglst.json'
    timed.dump(path)
    assert SegLST.load(path) == timed


def test_parse_mixed_scalar_and_word_timestamps():
    text = '''[
        {"session_id": "recording", "words": "a", "start_time": "0.125", "end_time": 1},
        {"session_id": "recording", "words": ["b"], "start_time": ["1.25"], "end_time": [2.5]}
    ]'''
    parsed = SegLST.parse(text)
    assert parsed[0]['start_time'] == decimal.Decimal('0.125')
    assert parsed[0]['end_time'] == decimal.Decimal('1')
    assert parsed[1]['start_time'] == [decimal.Decimal('1.25')]
    assert parsed[1]['end_time'] == [decimal.Decimal('2.5')]
