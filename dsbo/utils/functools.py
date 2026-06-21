from functools import wraps


def ensure_dict_return(fn):
    @wraps(fn)
    def _inner(*args, **kwargs):
        res = fn(*args, **kwargs)
        return {} if res is None else res
    return _inner


def auto_wrap(cls, method_name, *decorators):
    raw = cls.__dict__.get(method_name)
    if raw is None or getattr(raw, "_dsbo_wrapped", False):
        return
    wrapped = raw
    for decorator in decorators:
        wrapped = decorator(wrapped)
    setattr(wrapped, "_dsbo_wrapped", True)
    setattr(cls, method_name, wrapped)