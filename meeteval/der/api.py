from pathlib import Path

from meeteval.wer.api import _load_texts

__all__ = [
    'md_eval_22',
]


def md_eval_22(
        reference,
        hypothesis,
        collar=0,
        regex=None,
        uem=None,
):
    r, h = _load_texts(reference, hypothesis, regex)
    from meeteval.der.md_eval import md_eval_22_multifile
    if uem is not None:
        from meeteval.io.uem import UEM
        if isinstance(uem, (str, Path, list, tuple)):
            uem = UEM.load(uem)

    results = md_eval_22_multifile(r, h, collar, uem=uem)
    return results
