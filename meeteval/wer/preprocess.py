import itertools

import meeteval
from meeteval.io import SegLST
from meeteval.io.pbjson import zip_strict
from meeteval.io.seglst import seglst_map
from meeteval.wer.wer.utils import check_single_filename


@seglst_map()
def split_words(
        d: 'SegLST',
        *,
        keys=('words',)
):
    """
    Splits segments into words and copies all other entries.

    Uses a word-level timing strategy to convert the segment-level timings
    into word-level timings. If no strategy is given, the timings are
    copied as they are.

    >>> split_words(SegLST([{'words': 'a b c'}]))
    SegLST(segments=[{'words': 'a'}, {'words': 'b'}, {'words': 'c'}])

    >>> split_words(SegLST([{'words': 'a b c', 'start_time': 0, 'end_time': 1}]))
    SegLST(segments=[{'words': 'a', 'start_time': 0, 'end_time': 1}, {'words': 'b', 'start_time': 0, 'end_time': 1}, {'words': 'c', 'start_time': 0, 'end_time': 1}])

    >>> split_words(SegLST([{'words': 'a b c', 'word_timings': [(0, 1), (1, 2), (2, 3)]}]), keys=('words', 'word_timings'))
    SegLST(segments=[{'words': 'a', 'word_timings': (0, 1)}, {'words': 'b', 'word_timings': (1, 2)}, {'words': 'c', 'word_timings': (2, 3)}])
    """
    def split_entry(x):
        if isinstance(x, str):
            return x.split() or ['']
        elif isinstance(x, (list, tuple)):
            return x
        else:
            raise TypeError(x)

    return d.flatmap(
        lambda s: [
            {**s, **dict(zip_strict(keys, split))}
            for split in zip_strict(*(split_entry(s[key]) for key in keys))
        ])


def words_to_int(*d: SegLST):
    """
    Converts all words to ints. The mapping is created by iterating over all
    words in d and assigning an integer to each word.

    >>> words_to_int(SegLST([{'words': 'a b c'}]))
    [SegLST(segments=[{'words': 1}])]

    >>> words_to_int(SegLST([{'words': 'a'}, {'words': 'b'}]), SegLST([{'words': 'c'}, {'words': 'a'}]))
    [SegLST(segments=[{'words': 4}, {'words': 2}]), SegLST(segments=[{'words': 3}, {'words': 4}])]

    TODO: use cython code for speedup
    TODO: unify everything. This stuff is done in multiple places in the code base.
    """
    # Convert into integer representation to save some computation later.
    # `'words'` contains a single word only.
    sym2int = {v: i for i, v in enumerate([
        segment['words']
        for segment in itertools.chain(*d)
        if segment['words']
    ], start=1)}
    sym2int[''] = 0

    d = [d_.map(lambda s: {**s, 'words': sym2int[s['words']]}) for d_ in d]
    return d


def add_segment_index(d: SegLST):
    """
    Adds a segment index to the segments, if not already present.
    """
    if 'segment_index' not in d.T.keys():
        counter = itertools.count()
        d = d.map(lambda x: {**x, 'segment_index': next(counter)})
    return d


def preprocess(
        reference, hypothesis,
        segment_index='segment',
        convert_to_int=False,
        remove_empty_segments=True,
):
    """
    Preprocessing for non-time-constrained WERs.
    """
    if segment_index not in ('segment', 'word', False):
        raise ValueError(segment_index)

    # Convert before calling this function if special parameters are required
    reference = meeteval.io.asseglst(reference)
    hypothesis = meeteval.io.asseglst(hypothesis)

    check_single_filename(reference, hypothesis)

    if 'begin_time' in reference.T.keys() and 'end_time' in reference.T.keys():
        from meeteval.wer.wer.time_constrained import get_self_overlap
        reference_self_overlap = get_self_overlap(reference)
        if reference_self_overlap.total_time == 0:
            reference_self_overlap = None
    else:
        reference_self_overlap = None

    if 'begin_time' in hypothesis.T.keys() and 'end_time' in hypothesis.T.keys():
        from meeteval.wer.wer.time_constrained import get_self_overlap
        hypothesis_self_overlap = get_self_overlap(hypothesis)
        if hypothesis_self_overlap.total_time == 0:
            hypothesis_self_overlap = None
    else:
        hypothesis_self_overlap = None

    if segment_index == 'segment':
        reference = add_segment_index(reference)
        hypothesis = add_segment_index(hypothesis)

    reference = split_words(reference)
    hypothesis = split_words(hypothesis)

    if segment_index == 'word':
        reference = add_segment_index(reference)
        hypothesis = add_segment_index(hypothesis)

    if remove_empty_segments:
        reference = reference.filter(lambda s: s['words'])
        hypothesis = hypothesis.filter(lambda s: s['words'])

    if convert_to_int:
        reference, hypothesis = words_to_int(reference, hypothesis)

    return reference, hypothesis, reference_self_overlap, hypothesis_self_overlap
