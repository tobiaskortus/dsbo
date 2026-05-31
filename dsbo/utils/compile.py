import ast
from functools import wraps
import inspect
import textwrap
import contextlib
import contextvars
from typing import Callable, Any, Iterator
import warnings

_DRY_RUN = contextvars.ContextVar("dry_run", default=False)


def _get_ast_tree(obj):
    try:
        src = inspect.getsource(obj)
    except (OSError, TypeError, IOError):
        return None
    try:
        return ast.parse(textwrap.dedent(src))
    except Exception:
        return None


def _find_all_returns(tree) -> list[ast.Return]:
    if tree is None:
        return []
    return [n for n in ast.walk(tree) if isinstance(n, ast.Return)]

def _extract_dict_from_expr(expr, instance=None):
    if expr is None:
        return {}
    if isinstance(expr, ast.Dict):
        res = {}
        for k, v in zip(expr.keys, expr.values):
            if k is None:
                sub = _resolve_expr_to_dict(v, instance)
                if sub is None:
                    return None
                res.update(sub)
            else:
                if isinstance(k, ast.Constant):
                    res[k.value] = None
                elif isinstance(k, ast.Str):
                    res[k.s] = None
                else:
                    return None
        return res
    if isinstance(expr, ast.Attribute):
        if isinstance(expr.value, ast.Name) and expr.value.id == 'self' and instance is not None:
            member = getattr(instance, expr.attr, None)
            if callable(member):
                return _resolve_callable_return(member, instance)
        return None
    if isinstance(expr, ast.Call):
        func = expr.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == 'self' and instance is not None:
            method = getattr(instance, func.attr, None)
            if method is None:
                return None
            return _resolve_callable_return(method, instance)
        return None
    return None


def _resolve_expr_to_dict(expr, instance=None):
    try:
        val = ast.literal_eval(expr)
        if isinstance(val, dict):
            return {k: None for k in val.keys()}
        return None
    except Exception:
        return _extract_dict_from_expr(expr, instance)


def _resolve_callable_return(callable_obj, instance=None):
    t = _get_ast_tree(callable_obj)
    returns = _find_all_returns(t)
    schemas = []
    for r in returns:
        try:
            s = _resolve_expr_to_dict(r.value, instance)
        except Exception:
            s = None
        if s is not None:
            schemas.append(s)
    if schemas:
        keysets = [set(s.keys()) for s in schemas]
        union = set.union(*keysets) if keysets else set()
        if len({tuple(sorted(s)) for s in keysets}) > 1:
            try:
                name = getattr(callable_obj, '__name__', repr(callable_obj))
            except Exception:
                name = repr(callable_obj)
            warnings.warn(f"Multiple return dict schemas found in {name}: {[sorted(k) for k in keysets]}. Using union of keys.", UserWarning)
        return {k: None for k in union}
    return None


def dryrunnable(fn=None, *, warn_on_ambiguity: bool = True) -> Callable[..., dict | Any]:
    def wrap(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if is_dry_run():
                tree = _get_ast_tree(f)
                returns = _find_all_returns(tree)
                instance = args[0] if args else None

                schemas = []
                for r in returns:
                    try:
                        s = _resolve_expr_to_dict(r.value, instance)
                    except Exception:
                        s = None
                    if s is not None:
                        schemas.append(s)

                if not schemas:
                    return {}

                keysets = [set(s.keys()) for s in schemas]
                if schemas:
                    if len(keysets) == 1:
                        return {k: None for k in sorted(keysets[0])}
                        
                    union = set.union(*keysets)
                    return {k: None for k in union}


                raise ValueError(f"Could not resolve return dict for {f.__name__} in dry-run")
            return f(*args, **kwargs)
        return wrapper

    if fn is None:
        return wrap
    return wrap(fn)


@contextlib.contextmanager
def dry_run(enabled: bool = True) -> Iterator[None]:
    token = _DRY_RUN.set(enabled)
    try:
        from dsbo.worker.dask import WorkerContext

        with WorkerContext.use_worker(None):
            yield
    finally:
        _DRY_RUN.reset(token)


def is_dry_run() -> bool:
    return _DRY_RUN.get()