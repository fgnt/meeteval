import pytest
import inspect

import meeteval.wer.api
import meeteval.wer.__main__


@pytest.mark.parametrize("name", meeteval.wer.api.__all__)
def test_compatible_signatures(name):

    py_fn = getattr(meeteval.wer.api, name)
    sh_fn = getattr(meeteval.wer.__main__, name)

    py_sig = inspect.signature(py_fn)
    sh_sig = inspect.signature(sh_fn)

    sh_sig = sh_sig.replace(parameters=[
        v
        for k, v in sh_sig.parameters.items()
        if k not in ['average_out', 'per_reco_out']
    ])

    assert py_sig == sh_sig
