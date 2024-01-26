import glob
import logging
import re
from pathlib import Path

import meeteval.io
from meeteval.wer.wer import ErrorRate

__all__ = [
    'cpwer',
    'orcwer',
    'mimower',
    'tcpwer',
]


def _load_texts(
        reference_paths: 'list[str]',
        hypothesis_paths: 'list[str]',
        regex,
        reference_sort: 'bool | str' = False,
        hypothesis_sort: 'bool | str' = False,
        file_format=None,
) -> 'tuple[meeteval.io.SegLST, meeteval.io.SegLST]':
    """Load and validate reference and hypothesis texts.

    Validation checks that reference and hypothesis have the same example IDs.
    """

    # Normalize and glob (for backwards compatibility) the path input
    def _glob(pathname):
        match = list(glob.glob(pathname))
        # Forward pathname if not matched to get the correct error message
        return match or [pathname]

    def load(paths) -> meeteval.io.SegLST:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        if isinstance(paths, (tuple, list)):
            paths = [Path(file) for r in paths for file in _glob(str(r))]
            return meeteval.io.asseglst(
                meeteval.io.load(paths, format=file_format))
        else:
            try:
                return meeteval.io.asseglst(paths)
            except Exception as e:
                raise TypeError(type(paths), paths) from e

    # Load input files
    reference = load(reference_paths)
    hypothesis = load(hypothesis_paths)

    # Filter lines with regex based on filename
    if regex:
        r = re.compile(regex)

        def filter(s):
            filenames = s.T['session_id']
            filtered_filenames = [f for f in filenames if r.fullmatch(f)]
            assert filtered_filenames, (regex, filenames, 'Found nothing')
            return s.filter(lambda l: l['session_id'] in filtered_filenames)

        reference = filter(reference)
        hypothesis = filter(hypothesis)

    # Sort
    if reference_sort == 'segment':
        if 'start_time' in reference.T.keys():
            reference = reference.sorted('start_time')
        else:
            logging.warning(
                'Ignoring --reference-sort="segment" because no start_time is '
                'found in the reference'
            )
    elif not reference_sort:
        pass
    elif reference_sort in ('word', True):
        raise ValueError(
            f'reference_sort={reference_sort} is only supported for'
            f'time-constrained WERs.'
        )
    else:
        raise ValueError(
            f'Unknown choice for reference_sort: {reference_sort}'
        )
    if hypothesis_sort == 'segment':
        if 'start_time' in hypothesis.T.keys():
            hypothesis = hypothesis.sorted('start_time')
        else:
            logging.warning(
                'Ignoring --hypothesis-sort="segment" because no start_time is '
                'found in the hypothesis'
            )
    elif not hypothesis_sort:
        pass
    elif hypothesis_sort in ('word', True):
        raise ValueError(
            f'hypothesis_sort={hypothesis_sort} is only supported for'
            f'time-constrained WERs.'
        )
    else:
        raise ValueError(
            f'Unknown choice for hypothesis_sort: {hypothesis_sort}'
        )

    return reference, hypothesis


def orcwer(
        reference, hypothesis,
        regex=None,
        reference_sort='segment',
        hypothesis_sort='segment',
):
    """Computes the Optimal Reference Combination Word Error Rate (ORC WER)"""
    from meeteval.wer.wer.orc import orc_word_error_rate_multifile
    reference, hypothesis = _load_texts(
        reference, hypothesis, regex=regex,
        reference_sort=reference_sort, hypothesis_sort=hypothesis_sort
    )
    results = orc_word_error_rate_multifile(reference, hypothesis)
    return results


def cpwer(
        reference, hypothesis,
        regex=None,
        reference_sort='segment',
        hypothesis_sort='segment',
):
    """Computes the Concatenated minimum-Permutation Word Error Rate (cpWER)"""
    from meeteval.wer.wer.cp import cp_word_error_rate_multifile
    reference, hypothesis = _load_texts(
        reference, hypothesis, regex=regex,
        reference_sort=reference_sort, hypothesis_sort=hypothesis_sort
    )
    results = cp_word_error_rate_multifile(reference, hypothesis)
    return results


def mimower(
        reference, hypothesis,
        regex=None,
        reference_sort='segment',
        hypothesis_sort='segment',
):
    """Computes the MIMO WER"""
    from meeteval.wer.wer.mimo import mimo_word_error_rate_multifile
    reference, hypothesis = _load_texts(
        reference, hypothesis, regex=regex,
        reference_sort=reference_sort, hypothesis_sort=hypothesis_sort
    )
    results = mimo_word_error_rate_multifile(reference, hypothesis)
    return results


def tcpwer(
        reference, hypothesis,
        collar=0,
        hyp_pseudo_word_timing='character_based_points',
        ref_pseudo_word_timing='character_based',
        regex=None,
        reference_sort='segment',
        hypothesis_sort='segment',
):
    """Computes the time-constrained minimum permutation WER"""
    from meeteval.wer.wer.time_constrained import tcp_word_error_rate_multifile
    reference, hypothesis = _load_texts(reference, hypothesis, regex=regex)
    results = tcp_word_error_rate_multifile(
        reference, hypothesis,
        reference_pseudo_word_level_timing=ref_pseudo_word_timing,
        hypothesis_pseudo_word_level_timing=hyp_pseudo_word_timing,
        collar=collar,
        reference_sort=reference_sort,
        hypothesis_sort=hypothesis_sort,
    )
    average: ErrorRate = sum(results.values())
    average.hypothesis_self_overlap.warn('hypothesis')
    average.reference_self_overlap.warn('reference')
    return results
