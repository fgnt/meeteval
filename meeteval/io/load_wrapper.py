import decimal
import io
from pathlib import Path

__all__ = ['load']


def _guess_format(path: 'Path | io.TextIOBase'):
    """
    Guesses the file format from the `path`'s suffix.

    If the path is a pipe (e.g., `python -m ... <(cat hyp.stm)`), it is assumed to be an STM file.

    >>> _guess_format(Path('hyp.stm'))
    'stm'
    >>> _guess_format(Path('hyp.rttm'))
    'rttm'
    >>> _guess_format(Path('hyp.uem'))
    'uem'
    >>> _guess_format(Path('hyp.ctm'))
    'ctm'
    >>> _guess_format(Path('hyp.seglst.json'))
    'seglst'
    >>> _guess_format(Path('hyp'))
    'keyed_text'
    >>> _guess_format(Path('hyp.json'))
    'json'
    >>> _guess_format(Path('/dev/fd/63'))
    'stm'
    >>> _guess_format(Path('/proc/self/fd/63'))
    'stm'
    >>> from io import StringIO
    >>> example_files = (Path(__file__).parent.parent.parent / 'example_files').absolute()
    >>> with open(example_files / 'hyp.stm') as f:
    ...     print(_guess_format(f))
    stm
    """
    if isinstance(path, io.TextIOBase):
        path = Path(path.name)
    if str(path).startswith('/dev/fd/') or str(path).startswith('/proc/self/fd/'):
        # This is a pipe, i.e. python -m ... <(cat ...)
        # For now, assume it is an STM file
        # TODO: Support more formats that can be clearly distinguished just from the content, e.g., SegLST
        return 'stm'
    elif path.suffixes[-2:] == ['.seglst', '.json']:
        return 'seglst'
    elif path.suffix == '':
        # This is most likely a KeyedText file or a mistake
        return 'keyed_text'
    else:
        return path.suffix[1:].lower()


def load(path: 'Path | list[Path]', parse_float=decimal.Decimal, format: 'str | None' = None):
    """
    Load a file or multiple files as one of the formats in `meeteval.io`.

    Guesses the file format from the files suffix(es) by default. The format to use can be specified by the user by
    supplying `file_format`. This is especially useful when the files do not have a (correct) suffix, e.g., reading
    from STDIN.
    Available options are:
    - 'stm': NIST STM format
    - 'rttm': NIST RTTM format
    - 'uem': NIST UEM format
    - 'ctm': NIST CTM format
    - 'seglst': Chime7 JSON format (SegLST)
    - 'keyed_text': Kaldi KeyedText format

    Args:
        path: One or multiple paths to the file(s) to load. If multiple paths are provided, the file contents
            are merged. If multiple CTM files are supplied, they are merged into a single `CTMGroup` and file stems
            are interpreted as speakers.
        parse_float: Function to parse floats. Defaults to `decimal.Decimal`.
        format: Format of the file. If `None` or `'auto'`, guesses the format from the file suffix.

    If the suffix is `.json`, tries to load as Chime's `SegLST` format.
    """
    # Handle lists of path, when the user provides multiple files to load at once
    if isinstance(path, (list, tuple)):
        loaded = [load(p, parse_float=parse_float) for p in path]

        types = [type(l) for l in loaded]
        if len(set(types)) > 1:
            raise ValueError(
                f'All files must have the same format, but found {types} for {path}.'
            )
        return loaded[0].__class__.merge(*loaded)

    import meeteval

    if format in (None, 'none', 'auto'):
        format = _guess_format(Path(path))

    if format == 'stm':
        load_fn = meeteval.io.STM.load
    elif format == 'rttm':
        load_fn = meeteval.io.RTTM.load
    elif format == 'uem':
        load_fn = meeteval.io.UEM.load
    elif format == 'ctm':
        load_fn = meeteval.io.CTMGroup.load
    elif format == 'seglst':
        load_fn = meeteval.io.SegLST.load
    elif format == 'keyed_text':
        load_fn = meeteval.io.KeyedText.load
    elif format == 'json':
        # Guess the type from the file content. Only support Chime7 JSON / SegLST format.
        try:
            return meeteval.io.SegLST.load(path, parse_float=parse_float)
        except ValueError as e:
            # Catches simplejson's JSONDecodeError and our own ValueErrors
            raise ValueError(f'Unknown JSON format: {path}. Only SegLST format is supported.') from e
    else:
        raise ValueError(f'Unknown file type: {path}')

    return load_fn(path, parse_float=parse_float)
