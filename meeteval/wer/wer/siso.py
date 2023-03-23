def _siso_error_rate(
        reference: List[Hashable],
        hypothesis: List[Hashable]
) -> ErrorRate:
    import kaldialign

    try:
        result = kaldialign.edit_distance(reference, hypothesis)
    except TypeError:
        raise TypeError(type(reference), type(hypothesis), type(reference[0]), type(hypothesis[0]), reference[0], hypothesis[0])

    return ErrorRate(
        result['total'],
        len(reference),
        insertions=result['ins'],
        deletions=result['del'],
        substitutions=result['sub'],
    )


def siso_word_error_rate(
        reference: str,
        hypothesis: str
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
    return _siso_error_rate(
        reference.split(),
        hypothesis.split()
    )


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
