import typing
from typing import List, Hashable, Dict

from meeteval.wer.wer.error_rate import ErrorRate
from meeteval.io.tidy import tidy_args

if typing.TYPE_CHECKING:
    from meeteval.io.stm import STM
    from meeteval.io.tidy import Tidy

__all__ = ['siso_word_error_rate', 'siso_character_error_rate', 'siso_word_error_rate_keyed_text']


@tidy_args('words')
def siso_levenshtein_distance(reference: 'Tidy', hypothesis: 'Tidy') -> int:
    """
    Every element is treated as a single word.

    TODO: is this a good idea?
    """
    from meeteval.wer.matching.cy_levenshtein import levenshtein_distance

    reference = [s['words'] for s in reference if s['words']]
    hypothesis = [s['words'] for s in hypothesis if s['words']]

    return levenshtein_distance(reference, hypothesis)


def _siso_error_rate(reference: 'Tidy', hypothesis: 'Tidy') -> ErrorRate:
    import kaldialign

    reference = [s['words'] for s in reference if s['words']]
    hypothesis = [s['words'] for s in hypothesis if s['words']]

    try:
        result = kaldialign.edit_distance(reference, hypothesis)
    except TypeError:
        raise TypeError(type(reference), type(hypothesis), type(reference[0]), type(hypothesis[0]), reference[0],
                        hypothesis[0])

    return ErrorRate(
        result['total'],
        len(reference),
        insertions=result['ins'],
        deletions=result['del'],
        substitutions=result['sub'],
        reference_self_overlap=None,
        hypothesis_self_overlap=None,
    )


@tidy_args('words')
def siso_error_rate(reference: 'Tidy', hypothesis: 'Tidy') -> ErrorRate:
    if reference[0].get(keys.SESSION) != hypothesis[0].get(keys.SESSION):
        raise ValueError(
            f'Session ID must be identical, but found {reference[0].get(keys.SESSION)} for the reference '
            f'and {hypothesis[0].get(keys.SESSION)} for the hypothesis.'
        )
    return _siso_error_rate(reference, hypothesis)


@tidy_args('words')
def siso_word_error_rate(reference: 'Tidy', hypothesis: 'Tidy') -> ErrorRate:
    """
    The "standard" Single Input speaker, Single Output speaker (SISO) WER.

    This WER definition is what is generally called "WER". It just matches one
    word string against another word string.

    >>> siso_word_error_rate('a b c', 'a b c')
    ErrorRate(error_rate=0.0, errors=0, length=3, insertions=0, deletions=0, substitutions=0)
    >>> siso_word_error_rate('a b', 'c d')
    ErrorRate(error_rate=1.0, errors=2, length=2, insertions=0, deletions=0, substitutions=2)
    >>> siso_word_error_rate(reference='This is wikipedia', hypothesis='This wikipedia')  # Deletion example from https://en.wikipedia.org/wiki/Word_error_rate
    ErrorRate(error_rate=0.3333333333333333, errors=1, length=3, insertions=0, deletions=1, substitutions=0)
    """
    if len(reference) != 1:
        raise ValueError(f'Reference must contain exactly one line, but found {len(reference)} lines.')
    if len(hypothesis) != 1:
        raise ValueError(f'Hypothesis must contain exactly one line, but found {len(hypothesis)} lines.')

    def split_words(d):
        # TODO: only keep relevant keys?
        # TODO: move into tidy file?
        return [
            {**s, 'words': w}
            for s in d
            for w in (s['words'].split() if s['words'].strip() else [''])
        ]

    return siso_error_rate(split_words(reference), split_words(hypothesis))


def siso_word_error_rate_keyed_text(reference: 'STM | KeyedText',
                                    hypothesis: 'STM | KeyedText') -> 'Dict[str, ErrorRate]':
    """
    Computes the standard WER for each example in the reference and hypothesis files.

    To compute the overall WER, use `sum(siso_word_error_rate_keyed_text(r, h).values())`.
    """
    from meeteval.io.stm import apply_stm_multi_file
    return apply_stm_multi_file(siso_word_error_rate, reference, hypothesis, allowed_empty_examples_ratio=0)


@tidy_args('words')
def siso_character_error_rate(reference: 'Tidy', hypothesis: 'Tdiy') -> ErrorRate:
    """
    >>> siso_character_error_rate('abc', 'abc')
    ErrorRate(error_rate=0.0, errors=0, length=3, insertions=0, deletions=0, substitutions=0)
    """
    if len(reference) != 1:
        raise ValueError(f'Reference must contain exactly one line, but found {len(reference)} lines.')
    if len(hypothesis) != 1:
        raise ValueError(f'Hypothesis must contain exactly one line, but found {len(hypothesis)} lines.')

    def split_characters(d):
        # TODO: only keep relevant keys?
        # TODO: move into tidy file?
        return [
            {**s, 'words': c}
            for s in d
            for c in s['words'].strip()
        ]

    return siso_error_rate(reference.flatmap(split_characters), hypothesis.flatmap(split_characters))
