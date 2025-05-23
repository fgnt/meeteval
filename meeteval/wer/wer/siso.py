import typing
from typing import Hashable

from meeteval.io.py import NestedStructure
from meeteval.wer.preprocess import split_words
from meeteval.wer.wer.error_rate import ErrorRate
from meeteval.wer.wer.utils import kaldialign_edit_distance
from meeteval.io.seglst import asseglst

if typing.TYPE_CHECKING:
    from meeteval.io.stm import STM
    from meeteval.io.seglst import SegLST

__all__ = [
    'siso_word_error_rate',
    'siso_character_error_rate',
    'siso_word_error_rate_multifile'
]


def siso_levenshtein_distance(reference: 'SegLST', hypothesis: 'SegLST') -> int:
    """
    Every element is treated as a single word.
    """
    reference = asseglst(
        reference,
        required_keys=('words',),
        py_convert=lambda p: NestedStructure(p, ('segment',))
    )
    hypothesis = asseglst(
        hypothesis,
        required_keys=('words',),
        py_convert=lambda p: NestedStructure(p, ('segment',))
    )

    from meeteval.wer.matching.cy_levenshtein import levenshtein_distance

    reference = reference.T['words']
    if reference and isinstance(reference[0], list):
        reference = [w for r in reference for w in r]
    reference = [w for w in reference if w]
    hypothesis = hypothesis.T['words']
    if hypothesis and isinstance(hypothesis[0], list):
        hypothesis = [w for h in hypothesis for w in h]
    hypothesis = [w for w in hypothesis if w]

    return levenshtein_distance(reference, hypothesis)


def _siso_error_rate(
        reference: 'list[Hashable]', hypothesis: 'list[Hashable]'
) -> ErrorRate:
    try:
        result = kaldialign_edit_distance(reference, hypothesis)
    except TypeError:
        raise TypeError(
            type(reference), type(hypothesis), type(reference[0]),
            type(hypothesis[0]), reference[0], hypothesis[0]
        )

    return ErrorRate(
        result['total'],
        len(reference),
        insertions=result['ins'],
        deletions=result['del'],
        substitutions=result['sub'],
        reference_self_overlap=None,
        hypothesis_self_overlap=None,
    )


def _seglst_siso_error_rate(
        reference: 'SegLST',
        hypothesis: 'SegLST'
) -> ErrorRate:
    reference = [w for w in reference.T['words'] if w]
    hypothesis = [w for w in hypothesis.T['words'] if w]
    return _siso_error_rate(reference, hypothesis)


def siso_error_rate(reference: 'SegLST', hypothesis: 'SegLST') -> ErrorRate:
    reference = asseglst(
        reference,
        required_keys=('words',),
        py_convert=lambda p: NestedStructure(p, ('segment',))
    )
    hypothesis = asseglst(
        hypothesis,
        required_keys=('words',),
        py_convert=lambda p: NestedStructure(p, ('segment',))
    )
    if reference[0].get('session') != hypothesis[0].get('session'):
        raise ValueError(
            f'Session ID must be identical, but found '
            f'{reference[0].get("session")} for the reference '
            f'and {hypothesis[0].get("session")} for the hypothesis.'
        )
    return _seglst_siso_error_rate(reference, hypothesis)


def siso_word_error_rate(reference: 'SegLST', hypothesis: 'SegLST') -> ErrorRate:
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
    >>> from meeteval.io.stm import STM
    >>> siso_word_error_rate(STM.parse('X 1 Wikipedia 0 1 This is wikipedia'), STM.parse('X 1 Wikipedia 0 1 This wikipedia'))
    ErrorRate(error_rate=0.3333333333333333, errors=1, length=3, insertions=0, deletions=1, substitutions=0)
    """
    reference = asseglst(reference, required_keys=('words',), py_convert=lambda p: NestedStructure(p, ('segment',)))
    hypothesis = asseglst(hypothesis, required_keys=('words',), py_convert=lambda p: NestedStructure(p, ('segment',)))

    if len(reference) != 1:
        raise ValueError(f'Reference must contain exactly one line, but found {len(reference)} lines.')
    if len(hypothesis) != 1:
        raise ValueError(f'Hypothesis must contain exactly one line, but found {len(hypothesis)} lines.')

    return siso_error_rate(split_words(reference), split_words(hypothesis))


def siso_word_error_rate_multifile(
        reference,
        hypothesis,
        partial=False,
) -> 'dict[str, ErrorRate]':
    """
    Computes the standard WER for each example in the reference and hypothesis
    files.

    To compute the overall WER, use
    `sum(siso_word_error_rate_multifile(r, h).values())`.
    """
    from meeteval.io.seglst import apply_multi_file

    def groupby(r, h):
        """
        Checks whether the session IDs or (session ID, speaker) tuples are 
        unique in the reference and hypothesis and then groups by the 
        respective keys. `apply_multi_file` will then check whether the grouped
        keys match between reference and hypothesis.
        """
        if (
            len(r.unique('session_id')) == len(r)
            and len(h.unique('session_id')) == len(h)
        ):
            return r.groupby('session_id'), h.groupby('session_id')
        
        if 'speaker' not in r.T.keys() or 'speaker' not in h.T.keys():
            raise ValueError(
                f'The filenames / session ids are not unique and speaker IDs '
                f'are not present. The SISO-WER needs unique session IDs or '
                f'(session ID, speaker) pairs. '
            )
        
        if (
            len(r.unique(('session_id', 'speaker'))) == len(r)
            and len(h.unique(('session_id', 'speaker'))) == len(h)
        ):
            return r.groupby(('session_id', 'speaker')), h.groupby(('session_id', 'speaker'))
    
        raise ValueError(
            f'(session ID, speaker) pairs are not unique. The SISO-WER needs '
            f'unique session IDs or (session ID, speaker) pairs. '
        )

    return apply_multi_file(
        siso_word_error_rate, reference, hypothesis,
        allowed_empty_examples_ratio=0,
        partial=partial,
        _groupby=groupby,
    )


def siso_character_error_rate(
        reference: 'SegLST', hypothesis: 'SegLST'
) -> ErrorRate:
    """
    >>> siso_character_error_rate('abc', 'abc')
    ErrorRate(error_rate=0.0, errors=0, length=3, insertions=0, deletions=0, substitutions=0)
    """
    reference = asseglst(
        reference,
        required_keys=('words',),
        py_convert=lambda p: NestedStructure(p, ('segment',))
    )
    hypothesis = asseglst(
        hypothesis,
        required_keys=('words',),
        py_convert=lambda p: NestedStructure(p, ('segment',))
    )
    if len(reference) != 1:
        raise ValueError(
            f'Reference must contain exactly one line, but found '
            f'{len(reference)} lines.'
        )
    if len(hypothesis) != 1:
        raise ValueError(
            f'Hypothesis must contain exactly one line, but found '
            f'{len(hypothesis)} lines.'
        )

    def split_characters(s):
        return [
            {**s, 'words': c}
            for c in s['words'].strip()
            if c != ''
        ]

    return siso_error_rate(
        reference.flatmap(split_characters),
        hypothesis.flatmap(split_characters)
    )
