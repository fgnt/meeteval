import os
import decimal
import io
from pathlib import Path


__all__ = ['load', 'dump']


def _get_format(format, path): 
    import meeteval
    format = {
        'stm': meeteval.io.STM,
        'rttm': meeteval.io.RTTM,
        'uem': meeteval.io.UEM,
        'ctm': meeteval.io.CTM,
        'seglst': meeteval.io.SegLST,
        'keyed_text': meeteval.io.KeyedText,
        'json': meeteval.io.SegLST,
    }.get(format)

    if format is None:
        raise ValueError(f'Unknown file type: {path}')
    
    return format


def _open(f, mode='r'):
    import contextlib
    if isinstance(f, io.TextIOBase):
        return contextlib.nullcontext(f)
    elif isinstance(f, str) and str(f).startswith('http'):
        # Web request
        import urllib.request, urllib.error
        try:
            resource = urllib.request.urlopen(str(f))
        except urllib.error.URLError as e:
            raise FileNotFoundError(f) from e
        # https://stackoverflow.com/a/19156107/5766934
        return contextlib.nullcontext(io.TextIOWrapper(
            resource, resource.headers.get_content_charset()))
    elif str(f) == '-':
        import sys
        if mode == 'r':
            return contextlib.nullcontext(sys.stdin)
        elif mode == 'w':
            return contextlib.nullcontext(sys.stdout)
        else:
            raise ValueError(
                f'Mode "{mode}" is not supported for "-" (stdin/stdout).'
            )
    elif isinstance(f, (str, os.PathLike)):
        return open(f, mode)
    else:
        raise TypeError(type(f), f)


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
    - 'seglst': SegLST (Chime7 JSON format)
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
        
        import meeteval

        if isinstance(loaded[0], meeteval.io.CTM):
            return meeteval.io.CTMGroup({p.stem: l for p, l in zip(path, loaded)})

        return loaded[0].__class__.merge(*loaded)

    if format in (None, 'none', 'auto'):
        format = _guess_format(Path(path))

    loader = _get_format(format, path)
  
    return loader.load(path, parse_float=parse_float)


def dump(obj, path, format: 'str | None'=None, force=True):
    """
    Dump a `meeteval.io` object to a file. Converts `obj` to the specified format.

    Guesses the file format from the files suffix by default. The format to use can be specified by the user by
    supplying `file_format`. This is especially useful when the files do not have a (correct) suffix, e.g., reading
    from STDIN.
    Available options are:
    - 'stm': NIST STM format
    - 'rttm': NIST RTTM format
    - 'uem': NIST UEM format
    - 'ctm': NIST CTM format
    - 'seglst': SegLST (Chime7 JSON format)
    - 'keyed_text': Kaldi KeyedText format

    
    Args:
        obj: Object to dump.
        path: File path to dump the object to. Also used to guess the file format if `format` is None, 'none' or 'auto'.
        force: Overwrite the file if it already exists.
    """
    if not force:
        if os.path.exists(path):
            raise FileExistsError(f'Output file "{path}" already exists.')
    
    format_guessed = format in (None, 'none', 'auto')
    if format_guessed:
        if str(path) == '-':
            format = None
        else:
            format = _guess_format(Path(path))

    if format is None:
        import meeteval
        dumper = obj.__class__
        assert issubclass(dumper, meeteval.io.base.BaseABC), dumper
        assert hasattr(dumper, 'dump'), dumper
    else:
        dumper = _get_format(format, path)

    # Convert if the dumper is not the same as the object class
    # Skip (idempotent) conversion if the object is already of the correct type
    # to save computation time
    if obj.__class__ != dumper:
        import meeteval
        obj = meeteval.io.asseglst(obj, py_convert=meeteval.io.SegLST)
        try:
            obj = dumper.new(obj)
        except Exception as e:
            raise ValueError(
                f'Failed to convert object to {format}' +
                (f' (format was guessed from path suffix {path}) ' if format_guessed else '')
            ) from e
    return obj.dump(path)
