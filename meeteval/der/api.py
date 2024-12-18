from pathlib import Path

from meeteval.wer.api import _load_texts

__all__ = [
    'md_eval_22',
    'dscore',
]


def md_eval_22(
        reference,
        hypothesis,
        collar=0,
        regions='all',
        regex=None,
        uem=None,
):
    r, h = _load_texts(reference, hypothesis, regex)
    from meeteval.der.md_eval import md_eval_22_multifile
    if uem is not None:
        from meeteval.io.uem import UEM
        if isinstance(uem, (str, Path, list, tuple)):
            uem = UEM.load(uem)

    results = md_eval_22_multifile(
        r, h, collar, regions=regions, uem=uem
    )
    return results


def dscore(
        reference,
        hypothesis,
        collar=0,
        regions='all',
        regex=None,
        uem=None,
):
    r, h = _load_texts(reference, hypothesis, regex)
    from meeteval.der.nryant_dscore import dscore_multifile
    if uem is not None:
        from meeteval.io.uem import UEM
        if isinstance(uem, (str, Path, list, tuple)):
            uem = UEM.load(uem)

    results = dscore_multifile(
        r, h, collar, regions=regions, uem=uem
    )
    return results
