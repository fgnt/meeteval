def _items(obj: 'dict | tuple | list'):
    if isinstance(obj, dict):
        return list(obj.items())
    elif isinstance(obj, (tuple, list)):
        return list(enumerate(obj))
    else:
        raise TypeError(type(obj), obj)


def _keys(obj: 'dict | tuple | list'):
    if isinstance(obj, dict):
        return list(obj.keys())
    elif isinstance(obj, (tuple, list)):
        return list(range(len(obj)))
    else:
        raise TypeError(type(obj), obj)


def _values(obj: 'dict | tuple | list'):
    if isinstance(obj, dict):
        return list(obj.values())
    elif isinstance(obj, (tuple, list)):
        return obj
    else:
        raise TypeError(type(obj), obj)


def _map(fn, x):
    if isinstance(x, dict):
        return {k: fn(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return [fn(v) for v in x]
    else:
        raise TypeError(type(x))
