"""Path utilities for data loading.

Centralize the file name string formatting logic. This is for future
implementation of loading other types of datasets when the file names vary.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def expand_template(tpl: str, **kw) -> str:
    """Format template with date formatting and numeric padding."""
    return tpl.format(**kw)


def expand_many(tpl: str, seq: Iterable[Mapping]) -> list[str]:
    """Expand a template for each mapping in the sequence."""
    return [expand_template(tpl, **m) for m in seq]
