class ImmutableDict(dict):
    """Immutable version of python's dict type"""

    def __hash__(self):
        return id(self)

    def _immutable(self, *args, **kws):
        raise TypeError("object is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __setattr__ = _immutable

    clear = _immutable
    update = _immutable
    setdefault = _immutable
    pop = _immutable
    popitem = _immutable
