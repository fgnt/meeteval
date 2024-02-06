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
    'tcorcwer',
]


def _glob(pathname):
    # Normalize and glob (for backwards compatibility) the path input
    match = list(glob.glob(pathname))
    # Forward pathname if not matched to get the correct error message
    return match or [pathname]


def _maybe_load(paths, file_format) -> meeteval.io.SegLST:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    if isinstance(paths, (tuple, list)) and len(paths) and isinstance(paths[0], (str, Path)):
        paths = [Path(file) for r in paths for file in _glob(str(r))]
        return meeteval.io.asseglst(
            meeteval.io.load(paths, format=file_format))
    else:
        try:
            return meeteval.io.asseglst(paths)
        except Exception as e:
            raise TypeError(type(paths), paths) from e


def _load_texts(
        reference_paths: 'list[str]',
        hypothesis_paths: 'list[str]',
        regex,
        reference_sort: 'bool | str' = False,
        hypothesis_sort: 'bool | str' = False,
        file_format=None,
        uem=None,
) -> 'tuple[meeteval.io.SegLST, meeteval.io.SegLST]':
    """Load and validate reference and hypothesis texts.

    Validation checks that reference and hypothesis have the same example IDs.
    """
    # Load input files
    reference = _maybe_load(reference_paths, file_format=file_format)
    hypothesis = _maybe_load(hypothesis_paths, file_format=file_format)

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

    # Filter by uem
    if uem is not None:
        from meeteval.io.uem import UEM
        if isinstance(uem, (str, Path, list, tuple)):
            uem = UEM.load(uem)
        if 'start_time' not in reference.T.keys() or 'end_time' not in reference.T.keys():
            required_keys = {'start_time', 'end_time'}
            raise ValueError(
                f'UEM is only supported when the data contains timestamps,'
                f'but the reference is missing {required_keys - reference.T.keys()}.'
            )
        if 'start_time' not in hypothesis.T.keys() or 'end_time' not in hypothesis.T.keys():
            required_keys = {'start_time', 'end_time'}
            raise ValueError(
                f'UEM is only supported when the data contains timestamps, but '
                f'the hypothesis is missing {required_keys - hypothesis.T.keys()}.'
            )
        reference = reference.filter_by_uem(uem)
        hypothesis = hypothesis.filter_by_uem(uem)

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
        uem=None,
):
    """Computes the Optimal Reference Combination Word Error Rate (ORC WER)"""
    from meeteval.wer.wer.orc import orc_word_error_rate_multifile
    reference, hypothesis = _load_texts(
        reference, hypothesis, regex=regex,
        reference_sort=reference_sort, hypothesis_sort=hypothesis_sort,
        uem=uem,
    )
    results = orc_word_error_rate_multifile(reference, hypothesis)
    return results


def cpwer(
        reference, hypothesis,
        regex=None,
        reference_sort='segment',
        hypothesis_sort='segment',
        uem=None,
):
    """Computes the Concatenated minimum-Permutation Word Error Rate (cpWER)"""
    from meeteval.wer.wer.cp import cp_word_error_rate_multifile
    reference, hypothesis = _load_texts(
        reference, hypothesis, regex=regex,
        reference_sort=reference_sort, hypothesis_sort=hypothesis_sort,
        uem=uem,
    )
    results = cp_word_error_rate_multifile(reference, hypothesis)
    return results


def mimower(
        reference, hypothesis,
        regex=None,
        reference_sort='segment',
        hypothesis_sort='segment',
        uem=None,
):
    """Computes the MIMO WER"""
    from meeteval.wer.wer.mimo import mimo_word_error_rate_multifile
    reference, hypothesis = _load_texts(
        reference, hypothesis, regex=regex,
        reference_sort=reference_sort, hypothesis_sort=hypothesis_sort,
        uem=uem,
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
        uem=None,
):
    """Computes the time-constrained minimum permutation WER"""
    from meeteval.wer.wer.time_constrained import tcp_word_error_rate_multifile
    reference, hypothesis = _load_texts(reference, hypothesis, regex=regex, uem=uem)
    results = tcp_word_error_rate_multifile(
        reference, hypothesis,
        reference_pseudo_word_level_timing=ref_pseudo_word_timing,
        hypothesis_pseudo_word_level_timing=hyp_pseudo_word_timing,
        collar=collar,
        reference_sort=reference_sort,
        hypothesis_sort=hypothesis_sort,
    )
    from meeteval.wer import combine_error_rates
    average: ErrorRate = combine_error_rates(results)
    if average.hypothesis_self_overlap is not None:
        average.hypothesis_self_overlap.warn('hypothesis')
    if average.reference_self_overlap is not None:
        average.reference_self_overlap.warn('reference')
    return results


def tcorcwer(
        reference, hypothesis,
        regex=None,
        collar=0,
        hyp_pseudo_word_timing='character_based_points',
        ref_pseudo_word_timing='character_based',
        hypothesis_sort='segment',
        reference_sort='segment',
        uem=None,
):
    """Computes the time-constrained ORC WER"""
    from meeteval.wer.wer.time_constrained_orc import time_constrained_orc_wer_multifile
    reference, hypothesis = _load_texts(reference, hypothesis, regex=regex, uem=uem)
    results = time_constrained_orc_wer_multifile(
        reference, hypothesis,
        reference_pseudo_word_level_timing=ref_pseudo_word_timing,
        hypothesis_pseudo_word_level_timing=hyp_pseudo_word_timing,
        collar=collar,
        hypothesis_sort=hypothesis_sort,
        reference_sort=reference_sort,
    )
    from meeteval.wer import combine_error_rates
    average: ErrorRate = combine_error_rates(results)
    if average.hypothesis_self_overlap is not None:
        average.hypothesis_self_overlap.warn('hypothesis')
    if average.reference_self_overlap is not None:
        average.reference_self_overlap.warn('reference')
    return results
