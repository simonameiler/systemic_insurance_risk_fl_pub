"""
utils.py — county normalization & crosswalk helpers

Public API
----------
- make_xwalk_from_tiger(df): Convert TIGER/NA-style county table to ['County','county_fips'].
- norm_county_name(name): Normalize county display names to your project standard.

Behavioral notes (kept exactly as before)
-----------------------------------------
- County names:
  * "St." → "Saint", "St " → "Saint " (space)
  * "Miami Dade" → "Miami-Dade"
  * Hyphenated parts are Title-Cased individually (e.g., "miami-dade" → "Miami-Dade").
- `norm_county_name(None or NaN)` returns None.
- `make_xwalk_from_tiger` expects (case-insensitive) columns: STATEFP, COUNTYFP, COUNTYNAME,
  and returns unique rows of ['County','county_fips'] with county_fips = zfill2(STATEFP)+zfill3(COUNTYFP).
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

__all__ = ["make_xwalk_from_tiger", "norm_county_name"]


def make_xwalk_from_tiger(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a TIGER/NA-style county table to ['County','county_fips'].

    Parameters
    ----------
    df : pandas.DataFrame
        Must include (case-insensitive) columns:
          - STATEFP   : state FIPS (int/str)
          - COUNTYFP  : county FIPS (int/str)
          - COUNTYNAME: county name (str)

    Returns
    -------
    pandas.DataFrame
        Columns:
          - County      : normalized county display name (see notes above)
          - county_fips : zero-padded 5-character FIPS (str)

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    x = df.copy()

    # Normalize column names to upper for matching
    cols_up = {c: c.upper() for c in x.columns}
    x = x.rename(columns=cols_up)

    required = {"STATEFP", "COUNTYFP", "COUNTYNAME"}
    if not required.issubset(set(x.columns)):
        raise ValueError("Input must contain STATEFP, COUNTYFP, COUNTYNAME columns.")

    # Coerce to string and zero-pad; then concatenate to 5-char county FIPS
    x["STATEFP"] = (
        pd.to_numeric(x["STATEFP"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(2)
    )
    x["COUNTYFP"] = (
        pd.to_numeric(x["COUNTYFP"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(3)
    )
    x["county_fips"] = x["STATEFP"] + x["COUNTYFP"]

    # County name normalization (intentional, project-specific rules)
    County = (
        x["COUNTYNAME"]
        .astype(str)
        .str.strip()
        # case-insensitive replacements
        .str.replace(r"(?i)\bst\.", "Saint", regex=True)
        .str.replace(r"(?i)\bst\s", "Saint ", regex=True)
        .str.replace(r"(?i)\bmiami dade\b", "Miami-Dade", regex=True)
    )
    County = County.apply(lambda s: "-".join([w.title() for w in s.split("-")]))

    out = pd.DataFrame({"County": County, "county_fips": x["county_fips"]}).drop_duplicates()
    return out


def norm_county_name(name: Optional[object]) -> Optional[str]:
    """
    Normalize a county display name using the same rules as `make_xwalk_from_tiger`.

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