from typing import List, Hashable, Dict
from meeteval.wer.wer.error_rate import ErrorRate
from meeteval.io.stm import STM

__all__ = ['siso_word_error_rate', 'siso_character_error_rate']


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
        reference: 'str | STM',
        hypothesis: 'str | STM',
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
    if isinstance(reference, STM) or isinstance(hypothesis, STM):
        assert isinstance(hypothesis, STM) and isinstance(reference, STM)
        assert len(reference.filenames()) == 1, (len(reference.filenames()), reference.filenames(), reference)
        assert len(hypothesis.filenames()) == 1, (len(hypothesis.filenames()), hypothesis.filenames(), hypothesis)
        assert reference.filenames() == hypothesis.filenames(), (reference.filenames(), hypothesis.filenames())
        assert len(reference.lines) == 1, (len(reference.lines), reference.lines, reference)
        assert len(hypothesis.lines) == 1, (len(hypothesis.lines), hypothesis.lines, hypothesis)
        reference = reference.lines[0].transcript
        hypothesis = hypothesis.lines[0].transcript

    return _siso_error_rate(
        reference.split(),
        hypothesis.split()
    )


def mimo_word_error_rate_stm(reference_stm: 'STM', hypothesis_stm: 'STM') -> 'Dict[str, ErrorRate]':
    """
    TODO: doc
    """
    from meeteval.io.stm import apply_stm_multi_file
    return apply_stm_multi_file(siso_word_error_rate, reference_stm, hypothesis_stm)


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
