"""Compatibility wrapper for ReIFE pointwise evaluation callables."""

from __future__ import annotations

from typing import Any, Callable


class PointwiseEval:
    """Callable adapter for :func:`ReIFE` pointwise evaluation routines.

    The legacy pipeline code expects an object exposing an ``_eval_fn`` attribute
    for logging/introspection while remaining directly callable.  Earlier
    versions of the project provided this wrapper in ``evaluation_interfaces``
    but the refactor relocated the underlying implementation into the
    ``ReIFE`` package.  Re-introducing the adapter here keeps the public import
    stable without duplicating the actual evaluation logic.
    """

    def __init__(self) -> None:
        from ReIFE.ReIFE.methods.base_pointwise import pointwise_eval

        self._eval_fn: Callable[..., None] = pointwise_eval

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Proxy invocation to the wrapped evaluation function."""

        self._eval_fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(_eval_fn={self._eval_fn!r})"
