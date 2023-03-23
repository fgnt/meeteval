def _items(obj):
    if isinstance(obj, dict):
        return obj.items()
    elif isinstance(obj, (tuple, list)):
        return enumerate(obj)
    else:
        raise TypeError(type(obj), obj)

def _keys(obj):
    if isinstance(obj, dict):
        return list(obj.keys())
    else:
        return list(range(len(obj)))
