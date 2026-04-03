from __future__ import annotations

import sys
from dataclasses import dataclass


def dataclass_slots(*args, **kwargs):
    if sys.version_info >= (3, 10):
        kwargs.setdefault("slots", True)
    else:
        kwargs.pop("slots", None)
    return dataclass(*args, **kwargs)


def strict_zip(*iterables):
    if sys.version_info >= (3, 10):
        return zip(*iterables, strict=True)
    return zip(*iterables)
