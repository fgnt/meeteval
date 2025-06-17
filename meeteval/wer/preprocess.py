import itertools
import logging

import meeteval
from meeteval.io import SegLST
from meeteval.io.pbjson import zip_strict
from meeteval.io.seglst import seglst_map
from meeteval.wer.wer.utils import check_single_filename

logger = logging.getLogger('preprocess')


@seglst_map()
def split_words(
        d: 'SegLST',
        *,
        keys=('words',),
        word_level_timing_strategy=None,
        segment_representation='word',  # 'segment', 'word', 'speaker'
):
    """
    Splits segments into words and copies all other entries.

    If segment_representation is 'word', every segment in the output will
    contain a single word (or no word). If segment_representation is 'segment',
    every segment in the output will contain all words of the original segment
    as a list. If segment_representation is 'speaker', every segment in the
    output will contain all words of a single speaker as a list.

    >>> from paderbox.utils.pretty import pprint
    >>> pprint(split_words(SegLST([{'words': 'a b c'}])))
    SegLST([{'words': 'a'}, {'words': 'b'}, {'words': 'c'}])

    >>> pprint(split_words(SegLST([{'words': 'a b c', 'word_timings': [(0, 1), (1, 2), (2, 3)]}]), keys=('words', 'word_timings')))
    SegLST([{'words': 'a', 'word_timings': (0, 1)},
            {'words': 'b', 'word_timings': (1, 2)},
            {'words': 'c', 'word_timings': (2, 3)}])

    >>> pprint(split_words(SegLST([{'words': 'a b c', 'start_time': 0, 'end_time': 1}])))
    SegLST([{'words': 'a', 'start_time': 0, 'end_time': 1},
            {'words': 'b', 'start_time': 0, 'end_time': 1},
            {'words': 'c', 'start_time': 0, 'end_time': 1}])

    >>> pprint(split_words(SegLST([{'words': 'a b c', 'start_time': 0, 'end_time': 3}]), word_level_timing_strategy='character_based_points'))
    SegLST([{'words': 'a', 'start_time': 0.5, 'end_time': 0.5},
            {'words': 'b', 'start_time': 1.5, 'end_time': 1.5},
            {'words': 'c', 'start_time': 2.5, 'end_time': 2.5}])

    >>> pprint(split_words(SegLST([{'words': 'a b c', 'speaker': 'spkA'}, {'words': 'a b c', 'speaker': 'spkB'}, {'words': 'd e f', 'speaker': 'spkA'}]), segment_representation='speaker'))
    SegLST([{'words': ['a', 'b', 'c', 'd', 'e', 'f'], 'speaker': 'spkA'},
            {'words': ['a', 'b', 'c'], 'speaker': 'spkB'}])
    """
    assert 'words' in keys, keys

    def split_entry(x):
        if isinstance(x, str):
            return x.split()  # or ['']
        elif isinstance(x, (list, tuple)):
            return x
        else:
            raise TypeError(x)

    keys_ = keys
    if word_level_timing_strategy is not None:
        from meeteval.wer.wer.time_constrained import pseudo_word_level_strategies
        word_level_timing_strategy = pseudo_word_level_strategies[word_level_timing_strategy]
        keys_ = keys_ + ('start_time', 'end_time')

    def get_words(s):
        s = {
            **s,
            **{k: split_entry(s[k]) for k in keys}
        }

        if word_level_timing_strategy is not None:
            # Add a dummy word so that empty segments are not removed
            words = s['words'] or ['']
            timestamps = word_level_timing_strategy(
                (s['start_time'], s['end_time']),
                words
            )
            s['start_time'] = [s for s, _ in timestamps]
            s['end_time'] = [s for _, s in timestamps]

        if segment_representation == 'word':
            # Add an (empty) dummy word so that empty segments are not removed.
            if not s['words']:
                s['words'] = ['']

            s = [
                {
                    **s,
                    **dict(zip_strict(keys_, sp))
                }
                for sp in zip_strict(*[s[k] for k in keys_])
            ]
        else:
            s = [s]

        return s

    d = d.flatmap(get_words)

    if segment_representation == 'speaker':
        if 'speaker' not in d.T.keys():
            # Assume that all segments come from the same speaker
            d = merge_segments(d, (), keys_)
        else:
            d = merge_segments(d, 'speaker', keys_)

    return d


def merge_segments(
        d: SegLST,
        merge_by: 'str | tuple[str]',
        merge_keys: 'tuple[str]',
        ignore_keys: 'tuple[str]' = (),
        strict: 'bool' = False,
):
    """
    Group segments by `merge_by` and merge segments.

    >>> merge_segments(SegLST([{'a': 1, 'b': 2, 'c': 3}, {'a': 1, 'b': 3, 'c': 3}]), 'c', ('a', 'b'), ('c',))
    SegLST(segments=[{'a': [1, 1], 'b': [2, 3]}])
    >>> merge_segments(SegLST([{'a': 1, 'b': 2, 'c': 3}, {'a': 1, 'b': 3, 'c': 3}]), 'c', ('a', ), ('c',), strict=True)
    Traceback (most recent call last):
     ...
    ValueError: Expected all values to be the same, but found [2, 3] for key b
    >>> merge_segments(SegLST([{'a': [1, 2], 'b': 2, 'c': 3}, {'a': [3, 4], 'b': 3, 'c': 3}]), 'c', ('a', 'b'))
    SegLST(segments=[{'a': [1, 2, 3, 4], 'b': [2, 3], 'c': 3}])
    >>> merge_segments(SegLST([{'a': [1, 2], 'b': 2, 'c': 3}, {'a': [3, 4], 'b': 3, 'c': 3}]), ('b', 'c'), ('a', 'b'))
    SegLST(segments=[{'a': [1, 2], 'b': [2], 'c': 3}, {'a': [3, 4], 'b': [3], 'c': 3}])
    """

    def merge(segments) -> meeteval.io.seglst.SegLstSegment:
        """
        Merges a group of segments. Concatenates entries in merge_keys
        and drops keys in ignore_keys.

        If strict is True, it raises an exception if any other keys
        are not the same in all segments. Otherwise, drops any keys
        that are not unique.
        """
        assert len(segments) > 0

        # Only handle keys that are present in all segments
        all_keys = set.intersection(*[set(s.keys()) for s in segments]) - set(ignore_keys)
        all_keys = [k for k in segments[0] if k in all_keys]  # keep order

        # collate
        s = {
            k: [s[k] for s in segments]
            for k in all_keys
        }

        # Merge entries from merge_keys
        for k in merge_keys:
            # Concatenate iterables and keep a single list if
            # all values are not iterable
            try:
                s[k] = [w for v in s[k] for w in v]
            except TypeError:
                pass

        # Keep unique values, drop all others (or raise exception if strict=True)
        for k in all_keys:
            if k in merge_keys:
                continue
            if len(set(s[k])) == 1:
                s[k] = s[k][0]
            elif strict:
                raise ValueError(
                    f'Expected all values to be the same, but found '
                    f'{s[k]} for key {k}'
                )

        return s

    if isinstance(merge_by, str):
        merge_by = (merge_by,)

    groups = [d]
    for key in merge_by:
        groups = [
            new_group
            for group in groups
            for new_group in group.groupby(key).values()
        ]

    d = SegLST([merge(g) for g in groups])
    return d


def words_to_int(*d: 'SegLST'):
    """
    Converts all words to ints. The mapping is created by iterating over all
    words in d and assigning an integer to each word.

    >>> words_to_int(SegLST([{'words': 'a b c'}]))
    [SegLST(segments=[{'words': 1}])]

    >>> words_to_int(SegLST([{'words': 'a'}, {'words': 'b'}]), SegLST([{'words': 'c'}, {'words': 'a'}]))
    [SegLST(segments=[{'words': 1}, {'words': 2}]), SegLST(segments=[{'words': 3}, {'words': 1}])]

    TODO: use cython code for speedup
    TODO: unify everything. This stuff is done in multiple places in the code base.
    """
    # Convert into integer representation to save some computation later.
    # `'words'` contains a single word only.
    import collections
    sym2int = collections.defaultdict(itertools.count().__next__)
    _ = sym2int['']  # Reserve 0 for the empty string

    d = [
        d_.map(lambda s: {
            **s,
            'words': [
                sym2int[w]
                for w in s['words']]
                if isinstance(s['words'], list)
                else sym2int[s['words']]
        })
        for d_ in d
    ]
    return d


def add_segment_index(d: 'SegLST'):
    """
    Adds a segment index to the segments, if not already present.
    """
    if 'segment_index' not in d.T.keys():
        counter = itertools.count()
        d = d.map(lambda x: {**x, 'segment_index': next(counter)})
    return d


def check_timestamps_valid(seglst: 'SegLST', name=None):
    for s in seglst:
        if s['end_time'] < s['start_time']:
            raise ValueError(
                f'The end time of an interval must be larger than the start '
                f'time. Found {s} in {name}'
            )


def _select_keys(d: 'SegLST', keys=(), strict=True):
    if strict:
        return d.map(lambda x: {k: x[k] for k in keys})
    else:
        return d.map(lambda x: {k: x[k] for k in keys if k in x})


def _preprocess_single(
        segments: 'SegLST',
        *,
        collar,
        keep_keys=None,
        sort='segment',
        remove_empty_segments=True,
        word_level_timing_strategy=None,
        name=None,
        segment_index=False,  # 'segment', 'word', False
        segment_representation='word',  # 'segment', 'word', 'speaker'
):
    """
    >>> from paderbox.utils.pretty import pprint
    >>> segments = SegLST([{'words': 'c d', 'start_time': 1, 'end_time': 3}, {'words': 'a b', 'start_time': 0, 'end_time': 3}])
    >>> _preprocess_single(segments, sort=True, word_level_timing_strategy='character_based', name='test', collar=0)
    Traceback (most recent call last):
        ...
    ValueError: The order of word-level timings contradicts the segment-level order in test: 2 of 4 times.
    Consider setting sort to False or "segment" or "word".
    >>> pprint(_preprocess_single(segments, sort=False, word_level_timing_strategy='character_based', collar=0))
    (SegLST([{'words': 'c', 'start_time': 1.0, 'end_time': 2.0},
             {'words': 'd', 'start_time': 2.0, 'end_time': 3.0},
             {'words': 'a', 'start_time': 0.0, 'end_time': 1.5},
             {'words': 'b', 'start_time': 1.5, 'end_time': 3.0}]),
     SelfOverlap(overlap_rate=0.6666666666666666, overlap_time=2, total_time=3))
    >>> pprint(_preprocess_single(segments, sort='segment', word_level_timing_strategy='character_based', collar=0))
    (SegLST([{'words': 'a', 'start_time': 0.0, 'end_time': 1.5},
             {'words': 'b', 'start_time': 1.5, 'end_time': 3.0},
             {'words': 'c', 'start_time': 1.0, 'end_time': 2.0},
             {'words': 'd', 'start_time': 2.0, 'end_time': 3.0}]),
     SelfOverlap(overlap_rate=0.6666666666666666, overlap_time=2, total_time=3))
    >>> pprint(_preprocess_single(segments, sort='word', word_level_timing_strategy='character_based', collar=0))
    (SegLST([{'words': 'a', 'start_time': 0.0, 'end_time': 1.5},
             {'words': 'c', 'start_time': 1.0, 'end_time': 2.0},
             {'words': 'b', 'start_time': 1.5, 'end_time': 3.0},
             {'words': 'd', 'start_time': 2.0, 'end_time': 3.0}]),
     SelfOverlap(overlap_rate=0.6666666666666666, overlap_time=2, total_time=3))

    >>> pprint(_preprocess_single(segments, sort='word', keep_keys=('words',), collar=0))
    (SegLST([{'words': 'a'}, {'words': 'b'}, {'words': 'c'}, {'words': 'd'}]),
     SelfOverlap(overlap_rate=0.6666666666666666, overlap_time=2, total_time=3))
    """
    # Check if arguments are valid
    if segment_index not in ('segment', 'sorted_segment', 'word', False):
        raise ValueError(segment_index)

    contains_timestamps = (
            len(segments) == 0
            or 'start_time' in segments.T.keys()
            and 'end_time' in segments.T.keys()
    )
    if sort not in (True, False, 'segment', 'word', 'segment_if_available'):
        raise ValueError(
            f'Invalid value for sort: {sort}. Choose one of True, False, '
            f'"segment", "word"'
        )
    else:
        if sort == 'segment_if_available':
            if contains_timestamps:
                sort = 'segment'
            else:
                logger.warning(
                    f'Assuming sort=False because timestamps are '
                    f'missing in {name}.'
                )
                sort = False
        if sort == 'word' and segment_representation != 'word':
            raise ValueError(
                f'sort={sort} is only supported if segment_representation=\'word\'.'
            )
        if sort is not False and not contains_timestamps:
            raise ValueError(
                f'sort={sort} is only supported if the data contains timestamps.'
            )

    if segment_index == 'word' and segment_representation != 'word':
        raise ValueError(
            f'segment_index=\'word\' is only supported if segment_representation=\'word\'.'
        )
    if collar is not None and collar > 0:
        if not contains_timestamps:
            raise ValueError(
                f'collar={collar} is only supported if the data contains timestamps.'
            )
        if keep_keys is not None and 'begin_time' not in keep_keys and 'end_time' not in keep_keys:
            raise ValueError(
                f'collar={collar} but keep_keys does not contain "begin_time" and "end_time", '
                f'i.e. the timestamps are ignored'
            )
    if word_level_timing_strategy is not None and not contains_timestamps:
        raise ValueError(
            f'word_level_timing_strategy={word_level_timing_strategy} is only '
            f'supported if the data contains timestamps.'
        )

    # Check if data is valid
    if contains_timestamps:
        check_timestamps_valid(segments, name)

    # Compute self-overlap before anything else
    if 'start_time' in segments.T.keys() and 'end_time' in segments.T.keys():
        from meeteval.wer.wer.time_constrained import get_self_overlap
        if 'speaker' in segments.T.keys():
            self_overlap = sum(
                [get_self_overlap(s) for s in segments.groupby('speaker').values()],
                start=meeteval.wer.wer.time_constrained.SelfOverlap(0, 0)
            )
        else:
            self_overlap = get_self_overlap(segments)
    else:
        self_overlap = None

    # Add segment index
    if segment_index == 'segment':
        segments = add_segment_index(segments)

    # Sort, if requested and timestamps are available
    if sort in (True, 'segment', 'word') and contains_timestamps:
        segments = segments.sorted('start_time')

    if segment_index == 'sorted_segment':
        segments = add_segment_index(segments)

    # Remove keys that are not needed before duplicating them in the splitting
    # process
    if keep_keys is not None:
        keep_keys = set(keep_keys)
        if segment_index:
            keep_keys.add('segment_index')
        keep_keys1 = set(keep_keys)
        keep_keys2 = set(keep_keys)
        if contains_timestamps:
            # Timestamps are needed later for the check of the order of the
            # words matches
            keep_keys1.update({'start_time', 'end_time'})
        segments = _select_keys(segments, keep_keys1, strict=False)

    if collar is not None:
        if collar == 0:
            logger.warning(
                'Collar is set to 0, which means that no collar is applied.\n'
                'This is probably not what you want.\n' \
                'You may want to set the collar to 5 seconds.'
            )
        else:
            # words may be a list of words.
            # In that case, the start and end times are also lists.
            word_lengths = segments.flatmap(
                lambda s: (
                    [end - start for start, end in zip(
                        s['start_time'], s['end_time']
                        # strict=True,  # enable, once py310 is the minimum version
                    )] if isinstance(s['start_time'], (tuple, list)) else
                    [s['end_time'] - s['start_time']]
                )
            )
            if word_lengths:
                words = sum([
                    len(words.split()) if isinstance(words, str) else len(words)
                    for words in segments.T['words']
                ], start=0)
                mean_word_lengths = sum(word_lengths) / len(word_lengths)
                if mean_word_lengths > collar:
                    # Probably the unit of start and end times is not seconds.
                    # e.g., samples
                    logger.warning(
                        f'The mean word length is {mean_word_lengths:.2f} seconds, '
                        f'which is more than the collar length of {collar} seconds.'
                    )

    # Split into words. After this, the 'words' key contains a list of words
    # instead of a string
    words = split_words(
        segments,
        word_level_timing_strategy=word_level_timing_strategy,
        segment_representation=segment_representation
    )

    # Warn or raise an exception if the order of the words contradicts the
    # order of the segments.
    if contains_timestamps:

        # Check order for every speaker individually
        if 'speaker' in segments.T.keys():
            grouped_words = words.groupby('speaker')
        else:
            grouped_words = {None: words}

        if segment_representation != 'word':
            words_ = split_words(
                segments,
                word_level_timing_strategy=word_level_timing_strategy,
                segment_representation='word'
            )
            if 'speaker' in segments.T.keys():
                grouped_words_ = words_.groupby('speaker')
            else:
                grouped_words_ = {None: words_}
        else:
            grouped_words_ = grouped_words

        for speaker, spk_words in grouped_words.items():
            words_ = grouped_words_[speaker]

            words_sorted = words_.sorted('start_time')
            if words_sorted != words_:
                # This check should be fast because `sorted` doesn't change the identity
                # of the contained objects (so `words_sorted[0] is words[0] == True`
                # when they are sorted).
                contradictions = [a != b for a, b in zip(words_sorted, words_)]
                try:
                    session_ids = f' (session ids: {sorted(set(segments.T["session_id"]))})'
                except KeyError:
                    session_ids = ''
                if speaker is None:
                    speaker = ''
                else:
                    speaker = f' for speaker {speaker}'
                msg = (
                    f'The order of word-level timings contradicts the segment-level '
                    f'order in {name}{speaker}: '
                    f'{sum(contradictions)} of {len(contradictions)} times.'
                    f'{session_ids}'
                )
                if sort is not True:
                    logger.warning(msg)
                else:
                    raise ValueError(
                        f'{msg}\nConsider setting sort to False or "segment" or "word".'
                    )

        # Sort again here, this time across speakers
        if sort == 'word':
            words = words.sorted('start_time')

    if segment_index == 'word':
        words = add_segment_index(words)

    if remove_empty_segments:
        words = words.filter(lambda s: s['words'])

    if collar is not None and collar > 0:
        from meeteval.wer.wer.time_constrained import apply_collar
        words = apply_collar(words, collar)

    if keep_keys is not None and keep_keys1 != keep_keys2:
        words = _select_keys(words, keep_keys2)

    return words, self_overlap


def preprocess(
        reference, hypothesis,
        keep_keys=None,  # None or tuple[str]
        segment_index='segment',
        convert_to_int=False,
        remove_empty_segments=True,
        reference_sort='segment',
        hypothesis_sort='segment',
        collar=None,
        reference_pseudo_word_level_timing=None,
        hypothesis_pseudo_word_level_timing=None,
        segment_representation='segment',  # 'segment', 'word', 'speaker'
        ensure_single_session=True,
):
    """
    Preprocessing.
    """
    # Convert before calling this function if special parameters are required
    reference = meeteval.io.asseglst(reference)
    hypothesis = meeteval.io.asseglst(hypothesis)

    if ensure_single_session:
        check_single_filename(reference, hypothesis)

    reference, reference_self_overlap = _preprocess_single(
        reference,
        keep_keys=keep_keys,
        segment_index=segment_index,
        remove_empty_segments=remove_empty_segments,
        sort=reference_sort,
        name='reference',
        collar=None,   # collar is not applied to the reference
        word_level_timing_strategy=reference_pseudo_word_level_timing,
        segment_representation=segment_representation,
    )
    hypothesis, hypothesis_self_overlap = _preprocess_single(
        hypothesis,
        keep_keys=keep_keys,
        segment_index=segment_index,
        remove_empty_segments=remove_empty_segments,
        sort=hypothesis_sort,
        name='hypothesis',
        collar=collar,
        word_level_timing_strategy=hypothesis_pseudo_word_level_timing,
        segment_representation=segment_representation,
    )


    # Conversion to integer must be done across reference and hypothesis
    # for a consistent mapping.
    if convert_to_int:
        reference, hypothesis = words_to_int(reference, hypothesis)

    return reference, hypothesis, reference_self_overlap, hypothesis_self_overlap
