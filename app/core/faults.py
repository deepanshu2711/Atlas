import random
from contextlib import contextmanager
from contextvars import ContextVar

from app.core.config import settings

_faults: ContextVar[dict[str, float]] = ContextVar("faults", default={})


class InjectedFault(RuntimeError):
    pass


def _parse(header: str | None) -> dict[str, float]:
    """'SEARCH:0.3,LOOKUP_TABLE:0.5' -> {'SEARCH': 0.3, 'LOOKUP_TABLE': 0.5}"""
    out = {}
    for part in (header or "").split(","):
        name, _, p = part.partition(":")
        if name.strip() and p:
            out[name.strip().upper()] = float(p)
    return out


@contextmanager
def fault_scope(header: str | None):
    token = _faults.set(_parse(header=header)
                        if settings.allow_fault_injection else {})
    try:
        yield
    finally:
        _faults.reset(token)


def maybe_fail(tool: str, span):
    p = _faults.get().get(tool, 0.0)
    if p and random.random() < p:
        span.set_attribute("crucible.injected_fault", True)
        raise InjectedFault(f"injected failure in {tool}")
