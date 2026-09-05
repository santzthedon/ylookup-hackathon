"""A reusable exact match lookup that records misses instead of hiding them.

The four crosswalks in the workbook (legal entity, chart of accounts, investor,
deal) are all the same shape: some key columns on the left, some target system
columns on the right. So they are all built from this one class.

Three design rules, all of which exist because of what the fund manager in the
call transcript described:

  1. A row that does not match is never dropped. It flows through with its
     target columns empty and a status saying why. Money that vanishes silently
     between two systems is the failure mode the whole tool exists to catch.

  2. A key that appears twice in the mapping with two different answers is
     treated as unmatched, not as "take the first one". Picking arbitrarily
     produces a total that ties and is wrong.

  3. Every miss carries a reason, so the report can distinguish a source row
     with no key at all from one whose key is genuinely absent from the
     mapping table. Those have different fixes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from . import normalise

log = logging.getLogger(__name__)


class MatchStatus(str, Enum):
    """Why a row did or did not resolve."""

    MATCHED = "matched"
    NO_KEY = "no_key"
    NOT_IN_MAPPING = "not_in_mapping"
    AMBIGUOUS_MAPPING = "ambiguous_mapping"

    def __str__(self) -> str:
        return self.value


# Column names appended to any frame passed through a lookup.
STATUS_COL = "match_status"
REASON_COL = "match_reason"
KEY_COL = "match_key"

_REASON_TEXT = {
    MatchStatus.MATCHED: "resolved to a single mapping row",
    MatchStatus.NO_KEY: "source row has no value in one or more key columns",
    MatchStatus.NOT_IN_MAPPING: "key is not present in the mapping table",
    MatchStatus.AMBIGUOUS_MAPPING: (
        "key appears in the mapping table more than once with conflicting values"
    ),
}


@dataclass(frozen=True)
class LookupResult:
    """The outcome of a single key lookup."""

    status: MatchStatus
    values: dict[str, object] = field(default_factory=dict)
    key: str | None = None

    @property
    def matched(self) -> bool:
        return self.status is MatchStatus.MATCHED

    @property
    def reason(self) -> str:
        return _REASON_TEXT[self.status]


@dataclass
class LookupBuildReport:
    """What was found while building the lookup, before any row is looked up."""

    name: str
    mapping_rows: int
    usable_keys: int
    blank_key_rows: int
    duplicate_consistent_keys: int
    ambiguous_keys: list[str]

    def as_row(self) -> dict[str, object]:
        return {
            "lookup": self.name,
            "mapping_rows": self.mapping_rows,
            "usable_keys": self.usable_keys,
            "blank_key_rows": self.blank_key_rows,
            "duplicate_consistent_keys": self.duplicate_consistent_keys,
            "ambiguous_keys": len(self.ambiguous_keys),
        }


class Lookup:
    """Exact match crosswalk from one or more key columns to one or more values."""

    def __init__(
        self,
        name: str,
        mapping: pd.DataFrame,
        key_columns: list[str],
        value_columns: list[str],
        description: str = "",
    ) -> None:
        missing = [
            c for c in (*key_columns, *value_columns) if c not in mapping.columns
        ]
        if missing:
            raise KeyError(
                f"Lookup '{name}' cannot be built: mapping table is missing "
                f"column(s) {missing}. Present: {list(mapping.columns)}"
            )

        self.name = name
        self.key_columns = list(key_columns)
        self.value_columns = list(value_columns)
        self.description = description

        self._index, self._report = self._build_index(mapping)

    # -- construction ------------------------------------------------------

    def _build_index(
        self, mapping: pd.DataFrame
    ) -> tuple[dict[str, dict[str, object] | None], LookupBuildReport]:
        """Index the mapping table by normalised key.

        A key mapping to None in the index means "known but ambiguous", which
        is different from a key that is simply absent.
        """
        frame = mapping.copy()
        frame[KEY_COL] = normalise.build_composite_key(frame, self.key_columns)

        blank_key_rows = int(frame[KEY_COL].isna().sum())
        usable = frame.dropna(subset=[KEY_COL])

        index: dict[str, dict[str, object] | None] = {}
        ambiguous: list[str] = []
        duplicate_consistent = 0

        for key, group in usable.groupby(KEY_COL, sort=False):
            candidates = group[self.value_columns].drop_duplicates()
            if len(candidates) == 1:
                if len(group) > 1:
                    duplicate_consistent += 1
                index[str(key)] = candidates.iloc[0].to_dict()
            else:
                index[str(key)] = None
                ambiguous.append(str(key))

        report = LookupBuildReport(
            name=self.name,
            mapping_rows=len(mapping),
            usable_keys=len(index),
            blank_key_rows=blank_key_rows,
            duplicate_consistent_keys=duplicate_consistent,
            ambiguous_keys=ambiguous,
        )

        if blank_key_rows:
            log.info(
                "%s: %d mapping rows have no key and were excluded",
                self.name,
                blank_key_rows,
            )
        if ambiguous:
            log.warning(
                "%s: %d key(s) appear more than once with conflicting values "
                "and will be reported as ambiguous, not matched",
                self.name,
                len(ambiguous),
            )

        return index, report

    @property
    def build_report(self) -> LookupBuildReport:
        return self._report

    def __len__(self) -> int:
        return len(self._index)

    # -- single lookup -----------------------------------------------------

    def get(self, *key_parts: object) -> LookupResult:
        """Look up one key. Pass one argument per key column, in order."""
        if len(key_parts) != len(self.key_columns):
            raise ValueError(
                f"Lookup '{self.name}' expects {len(self.key_columns)} key "
                f"part(s) ({self.key_columns}), got {len(key_parts)}"
            )

        parts = [normalise.normalise_text(p) for p in key_parts]
        if any(p is None for p in parts):
            return LookupResult(MatchStatus.NO_KEY)

        key = normalise.KEY_SEPARATOR.join(parts)  # type: ignore[arg-type]

        if key not in self._index:
            return LookupResult(MatchStatus.NOT_IN_MAPPING, key=key)

        values = self._index[key]
        if values is None:
            return LookupResult(MatchStatus.AMBIGUOUS_MAPPING, key=key)

        return LookupResult(MatchStatus.MATCHED, values=dict(values), key=key)

    # -- frame lookup ------------------------------------------------------

    def apply(
        self,
        frame: pd.DataFrame,
        source_key_columns: list[str],
        prefix: str | None = None,
    ) -> pd.DataFrame:
        """Resolve every row of a frame, keeping unmatched rows.

        Returns a copy of the frame with the value columns appended (empty
        where unmatched), plus match_key, match_status and match_reason.
        """
        if len(source_key_columns) != len(self.key_columns):
            raise ValueError(
                f"Lookup '{self.name}' expects {len(self.key_columns)} key "
                f"column(s), got {len(source_key_columns)}"
            )
        missing = [c for c in source_key_columns if c not in frame.columns]
        if missing:
            raise KeyError(
                f"Lookup '{self.name}': frame is missing key column(s) {missing}"
            )

        out = frame.copy()
        out[KEY_COL] = normalise.build_composite_key(out, source_key_columns)

        prefix = prefix if prefix is not None else f"{self.name}_"
        renamed = {col: f"{prefix}{col}" for col in self.value_columns}

        # Split the index into resolved keys and ambiguous keys so both can be
        # handled with a vectorised join rather than a Python level loop.
        resolved = {k: v for k, v in self._index.items() if v is not None}
        ambiguous_keys = {k for k, v in self._index.items() if v is None}

        if resolved:
            values = pd.DataFrame.from_dict(resolved, orient="index")
            values.index.name = KEY_COL
            values = values.rename(columns=renamed).reset_index()
            out = out.merge(values, on=KEY_COL, how="left")
        else:
            for target in renamed.values():
                out[target] = pd.NA

        first_value_col = list(renamed.values())[0]

        no_key = out[KEY_COL].isna()
        is_ambiguous = out[KEY_COL].isin(ambiguous_keys)
        matched = out[first_value_col].notna() & ~is_ambiguous

        status = pd.Series(
            str(MatchStatus.NOT_IN_MAPPING), index=out.index, dtype="string"
        )
        status = status.mask(matched, str(MatchStatus.MATCHED))
        status = status.mask(is_ambiguous, str(MatchStatus.AMBIGUOUS_MAPPING))
        status = status.mask(no_key, str(MatchStatus.NO_KEY))

        out[STATUS_COL] = status
        out[REASON_COL] = status.map(
            {str(s): _REASON_TEXT[s] for s in MatchStatus}
        ).astype("string")

        # An ambiguous key may have picked up values from the resolved join if
        # the key also existed there; clear them so nothing downstream reads a
        # value the lookup refused to commit to.
        for target in renamed.values():
            out.loc[~matched, target] = pd.NA

        return out

    def unmatched(self, resolved_frame: pd.DataFrame) -> pd.DataFrame:
        """Every row of an applied frame that did not resolve."""
        return resolved_frame[
            resolved_frame[STATUS_COL] != str(MatchStatus.MATCHED)
        ].copy()

    def match_rate(self, resolved_frame: pd.DataFrame) -> float:
        if resolved_frame.empty:
            return 1.0
        matched = (resolved_frame[STATUS_COL] == str(MatchStatus.MATCHED)).sum()
        return float(matched) / len(resolved_frame)
