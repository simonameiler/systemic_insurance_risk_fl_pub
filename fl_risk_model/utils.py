"""utils.py - county name normalization helper

Public API
----------
- norm_county_name(name): Normalize county display names to project standard.

Behavioral notes
----------------
- County names:
  * "St." -> "Saint", "St " -> "Saint " (space)
  * "Miami Dade" -> "Miami-Dade"
  * Hyphenated parts are Title-Cased individually (e.g., "miami-dade" -> "Miami-Dade").
- `norm_county_name(None or NaN)` returns None.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

__all__ = ["norm_county_name"]


def norm_county_name(name: Optional[object]) -> Optional[str]:
    """
    Normalize a county display name to the project standard.

    Parameters
    ----------
    name : object | None
        Typically a string; None/NaN are returned as None.

    Returns
    -------
    str | None
        Normalized county name (e.g., "Miami-Dade", "Saint Johns"), or None for NA.

    Notes
    -----
    This mirrors the exact behavior used elsewhere in the codebase; do not widen
    or tighten rules without checking all loaders/joins.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    s = str(name).strip()
    s = (
        # case-insensitive replacements
        re.sub(r"(?i)\bst\.", "Saint", s)
    )
    s = re.sub(r"(?i)\bst\s", "Saint ", s)
    s = re.sub(r"(?i)\bmiami dade\b", "Miami-Dade", s)
    s = "-".join([w.title() for w in s.split("-")])
    return s