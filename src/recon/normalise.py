"""Turning a spreadsheet value into a comparable lookup key.

Two systems spell the same thing differently in harmless ways. One export has
a trailing space, another capitalises differently, a third has a double space
where a single space was intended. None of those are real differences, so
comparing raw strings would report breaks that are not breaks.

This module fixes only those harmless differences. It is deliberately narrow.

WHAT IT DOES
  - strips leading and trailing whitespace
  - collapses runs of internal whitespace to a single space
  - normalises non breaking spaces and similar invisible characters to a space
  - casefolds, so "L.P." and "l.p." are the same key

WHAT IT DELIBERATELY DOES NOT DO
  - fuzzy or approximate matching of any kind
  - removing or normalising punctuation ("L.P." stays different from "LP")
  - expanding abbreviations
  - stripping legal suffixes such as Ltd, SCSp, Sarl

The reason for the second list is that every item on it can silently join two
genuinely different records. "Chalbury HoldCo LP" and "Chalbury HoldCo L.P."
may well be the same entity, but they may equally be a fund and its general
partner, which are different legal entities with different books. A tool that
guesses produces a reconciliation that ties and is wrong, which is worse than
one that reports a gap. Anything punctuation deep is a mapping gap for a human
to decide, not a normalisation for this module to assume.
"""

from __future__ import annotations

import re

import pandas as pd

# Characters that render as a space but are not U+0020.
_INVISIBLE_SPACE = re.compile(r"[\u00a0\u2000-\u200b\u202f\u205f\u3000\t\r\n]")
_MULTI_SPACE = re.compile(r"\s+")

# Sentinel used inside composite keys. Chosen because it cannot occur in a
# normalised value: normalisation collapses whitespace, so a run of these
# characters is never produced by the cleanup rules above.
KEY_SEPARATOR = "\u241f"


def normalise_text(value: object) -> str | None:
    """Normalise a single value into a lookup key, or None if it is blank.

    None, NaN and strings that are empty or whitespace only all return None.
    A caller can then treat "there was no key here" differently from "there was
    a key and it did not match", which matters because the two have different
    causes and different fixes.
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None

    text = str(value)
    text = _INVISIBLE_SPACE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    if not text:
        return None
    return text.casefold()


def normalise_series(series: pd.Series) -> pd.Series:
    """Vectorised normalise_text. Blank values become pd.NA."""
    out = series.astype("string")
    out = out.str.replace(_INVISIBLE_SPACE, " ", regex=True)
    out = out.str.replace(_MULTI_SPACE, " ", regex=True)
    out = out.str.strip()
    out = out.mask(out.eq(""), pd.NA)
    return out.str.casefold()


def build_composite_key(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Join several normalised columns into one key.

    If any component is blank the whole key is blank, because a partial key
    would match the wrong record. A row missing its deal name should be
    reported as having no key, not matched on position alone.
    """
    if not columns:
        raise ValueError("build_composite_key requires at least one column")

    parts = [normalise_series(frame[col]) for col in columns]

    key = parts[0]
    for part in parts[1:]:
        key = key + KEY_SEPARATOR + part

    return key


def describe_key(key: str | None) -> str:
    """Render a composite key readably for a report."""
    if key is None or pd.isna(key):
        return "(blank)"
    return str(key).replace(KEY_SEPARATOR, " | ")
