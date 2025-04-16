
from meeteval.io.seglst import seglst_map
import re
import meeteval

__all__ = [
    'normalize',
]


class _Normalizers:
    """
    A normalizer that is applied to the transcript.
    Choices:
    - None: Do nothing (default)
    - lower,rm(.?!,): Lowercase the transcript and remove punctuations (.,?!).
    - lower,rm([^a-z0-9 ]): Lowercase the transcript and remove all characters that are not a-z, 0-9, or space.
    - chime6: Normalize the transcript according to the CHiME-6 challenge.
    - chime7: Normalize the transcript according to the CHiME-7 challenge.
              In contrast to chime6, words like "hmm" are normalized,
              e.g. "hm", "hmm" and "hmmm" get "hmmm".
    - chime8: Normalize the transcript according to the CHiME-8 challenge.
              Removes words like "hmm", converts integers to words,
              and much more.
    """
    # Note: The text above is used as CLI help text.
    #       So the None is not a real option for this class, but the default
    #       value for the normalizer argument.

    def keys(self):
        return [
            'lower,rm(.?!,)',
            'lower,rm([^a-z0-9 ])',
            'chime6',
            'chime7',
            'chime8',
        ]

    def __getitem__(self, normalizer):
        """
        >>> def test(normalizer):
        ...     seg = {'words': 'Hello, World! äèア😊 hm hmm hmmm [noise] [inaudible] wi-fi word-word 1.1 11 Mr. then,\" said'}
        ...     return _Normalizers()[normalizer](seg)['words']
        >>> test('lower,rm(.?!,)')
        'hello world äèア😊 hm hmm hmmm [noise] [inaudible] wi-fi word-word 11 11 mr then" said'
        >>> test('lower,rm([^a-z0-9 ])')
        'hello world hm hmm hmmm noise inaudible wifi wordword 11 11 mr then said'
        >>> test('chime8')
        'hello world aeア wifi word word 1.1 eleven mister then said'
        >>> test('chime7')
        'hello world äèア😊 hmmm hmmm hmmm wi-fi word-word 11 11 mr then said'
        >>> test('chime6')
        'hello world äèア😊 hm hmm hmmm wi-fi word-word 11 11 mr then said'
        """
        if normalizer == 'lower,rm(.?!,)':
            def normalizer(seg):
                seg['words'] = seg['words'].lower().replace('.', '').replace('?', '').replace('!', '').replace(',', '')
                return seg
        elif normalizer == 'lower,rm([^a-z0-9 ])':
            r = re.compile('[^a-z0-9 ]')
            r2 = re.compile(r'\s{2,}')
            def normalizer(seg):
                seg['words'] = r2.sub(' ', r.sub('', seg['words'].lower()).strip())
                return seg
        elif normalizer in ['chime8', 'chime7', 'chime6']:
            from chime_utils.text_norm import get_txt_norm  # pip install git+https://github.com/chimechallenge/chime-utils
            chime_utils_normalizer = get_txt_norm(normalizer)

            # Note:
            #     chime6 and chime7 remove all words, when [redacted] is
            #     present.

            def normalizer(seg):
                words = chime_utils_normalizer(seg['words'])
                if words != chime_utils_normalizer(seg['words']):
                    raise RuntimeError(
                        f'You discovered an idempotence bug in the chime_utils normalizer: {normalizer!r}.\n'
                        f'Original:         {seg["words"]}\n'
                        f'Normalized:       {words}\n'
                        f'Normalized again: {chime_utils_normalizer(words)}'
                    )
                seg['words'] = words
                return seg
        else:
            raise ValueError(f'Unknown normalizer: {normalizer}. Available normalizers: {self.keys()}')
        return normalizer


normalizers = _Normalizers()


@seglst_map(required_keys=('words',))
def normalize(input, *, normalizer: str):
    """
    Normalize the words in the transcript with the normalizer `normalizer`.
    Works with any object convertible to SegLST.

    Available normalizers are:
    - None: Do nothing (default)
    - lower,rm(.?!,): Lowercase the transcript and remove punctuations (.,?!).
    - lower,rm([^a-z0-9 ]): Lowercase the transcript and remove all characters that are not a-z, 0-9, or space.
    - chime6: Normalize the transcript according to the CHiME-6 challenge.
    - chime7: Normalize the transcript according to the CHiME-7 challenge.
              In contrast to chime6, words like "hmm" are normalized,
              e.g. "hm", "hmm" and "hmmm" get "hmmm".
    - chime8: Normalize the transcript according to the CHiME-8 challenge.
              Removes words like "hmm", converts integers to words,
              and much more.

    >>> normalize(meeteval.io.SegLST([{'words': 'Hello, World!'}]), normalizer='lower,rm(.?!,)')
    SegLST(segments=[{'words': 'hello world'}])
    >>> print(normalize(
    ...     meeteval.io.STM.parse('''recordingA 1 Alice 1 1 hello world äèア😊 hm hmm hmmm [noise] [inaudible] wi-fi word-word 1.1 11 mr then" said'''), 
    ...     normalizer='chime8'
    ... ).dumps())
    recordingA 1 Alice 1 1 hello world aeア wifi word word 1.1 eleven mister then said
    <BLANKLINE>
    """
    if normalizer is None:
        return input
    if isinstance(normalizer, str):
        normalizer = normalizers[normalizer]
    return input.map(normalizer)
