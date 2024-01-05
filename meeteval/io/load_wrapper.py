import decimal
from pathlib import Path


def load(path, parse_float=decimal.Decimal):
    """
    Guesses the file format from the `path`'s suffix.

    If the suffix is `.json`, tries to load as Chime's `SegLST` format.
    """
    if isinstance(path, list):
        loaded = [load(p, parse_float=parse_float) for p in path]
        return loaded[0].__class__.merge(*loaded)

    import meeteval
    path = Path(path)

    if path.suffix == '.stm':
        load_fn = meeteval.io.STM.load
    elif path.suffix == '.rttm':
        load_fn = meeteval.io.RTTM.load
    elif path.suffix == '.uem':
        load_fn = meeteval.io.UEM.load
    elif path.suffix == '.ctm':
        load_fn = meeteval.io.CTM.load
    elif path.suffixes[-2:] == ['.seglst', '.json']:
        load_fn = meeteval.io.SegLST.load
    elif path.suffix == '':
        # This is most likely a KeyedText file or a mistake
        load_fn = meeteval.io.KeyedText.load
    elif path.suffix == '.json':
        # Guess the type from the file content. Only support Chime7 JSON / SegLST format.
        try:
            return meeteval.io.SegLST.load(path, parse_float=parse_float)
        except Exception as e:
            raise ValueError(f'Unknown JSON format: {path}. Only SegLST format is supported.') from e
    else:
        raise ValueError(f'Unknown file type: {path}')

    return load_fn(path, parse_float=parse_float)
