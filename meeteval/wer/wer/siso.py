from typing import List, Hashable, Dict

from meeteval.io.keyed_text import KeyedText
from meeteval.io.stm import STM
from meeteval.wer.wer.error_rate import ErrorRate

__all__ = ['siso_word_error_rate', 'siso_character_error_rate', 'siso_word_error_rate_keyed_text']


def _siso_error_rate(
        reference: List[Hashable],
        hypothesis: List[Hashable]
) -> ErrorRate:
    import kaldialign

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
    )


def siso_word_error_rate(
        reference: 'str | STM | KeyedText',
        hypothesis: 'str | STM | KeyedText',
) -> ErrorRate:
    """
    The "standard" Single Input speaker, Single Output speaker (SISO) WER.

    This WER definition is what is generally called "WER". It just matches one
    word string against another word string.

    >>> siso_word_error_rate('a b c', 'a b c')
    ErrorRate(errors=0, length=3, insertions=0, deletions=0, substitutions=0, error_rate=0.0)
    >>> siso_word_error_rate('a b', 'c d')
    ErrorRate(errors=2, length=2, insertions=0, deletions=0, substitutions=2, error_rate=1.0)
    >>> siso_word_error_rate(reference='This is wikipedia', hypothesis='This wikipedia')  # Deletion example from https://en.wikipedia.org/wiki/Word_error_rate
    ErrorRate(errors=1, length=3, insertions=0, deletions=1, substitutions=0, error_rate=0.3333333333333333)
    """
    if isinstance(reference, KeyedText) or isinstance(hypothesis, KeyedText):
        from meeteval.wer.wer.utils import _check_valid_input_files
        _check_valid_input_files(reference, hypothesis)
        if len(reference.lines) != 1:
            raise ValueError(
                f'Reference must contain exactly one line, but found {len(reference.lines)} lines in {reference}.'
            )
        if len(hypothesis.lines) != 1:
            raise ValueError(
                f'Hypothesis must contain exactly one line, but found {len(hypothesis.lines)} lines in {reference}.'
            )
        reference = reference.lines[0].transcript
        hypothesis = hypothesis.lines[0].transcript

    return _siso_error_rate(
        reference.split(),
        hypothesis.split()
    )


def siso_word_error_rate_keyed_text(reference: 'STM | KeyedText', hypothesis: 'STM | KeyedText') -> 'Dict[str, ErrorRate]':
    """
    Computes the standard WER for each example in the reference and hypothesis files.

    To compute the overall WER, use `sum(siso_word_error_rate_keyed_text(r, h).values())`.
    """
    from meeteval.io.stm import apply_stm_multi_file
    return apply_stm_multi_file(siso_word_error_rate, reference, hypothesis, allowed_empty_examples_ratio=0)


def siso_character_error_rate(
        reference: str,
        hypothesis: str,
) -> ErrorRate:
    """
    >>> siso_character_error_rate('abc', 'abc')
    ErrorRate(errors=0, length=3, insertions=0, deletions=0, substitutions=0, error_rate=0.0)
    """
    return _siso_error_rate(
        list(reference), list(hypothesis)
    )
