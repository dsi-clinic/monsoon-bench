# Centralize the file name string formatting logic
# This is for future implementation of loading other types of datasets when the
# file names vary
from __future__ import annotations
from typing import Iterable, Mapping
from datetime import datetime

def expand_template(tpl: str, **kw) -> str:
    """
    Supports date formatting like {init:%Y%m%d}, numeric padding {member:02d}, etc.
    """
    return tpl.format(**kw)

def expand_many(tpl: str, seq: Iterable[Mapping]) -> list[str]:
    return [expand_template(tpl, **m) for m in seq]
