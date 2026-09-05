"""Turning unmatched rows into a report a human can act on.

An unmatched row on its own is not useful. Four hundred unmatched rows sharing
one cause are one decision, not four hundred. So gaps are aggregated by the key
that failed and the reason it failed, with a row count and a total amount
attached, which is the same shape as the workbook's own 'Mapping Gaps' sheet.

Materiality is the accounting idea that a difference small enough not to change
anyone's decision does not need chasing. It matters here because several of the
gaps in this dataset are floating point residue, amounts like 9.09e-13, which
are arithmetically not zero but are zero in substance. Grouping those with a
real four thousand pound gap would bury the one that matters. So each gap is
classified, and nothing is dropped either way.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config
from .lookup import REASON_COL, STATUS_COL, MatchStatus

C = config.COLS


@dataclass(frozen=True)
class GapSpec:
    """How to aggregate the unmatched rows of one resolved frame."""

    lookup_name: str
    key_columns: list[str]
    amount_column: str | None = None


def classify_materiality(
    amount: float | None, threshold: float = config.MATERIALITY_THRESHOLD
) -> str:
    """Label a gap by whether its amount is large enough to matter.

    'nets_to_zero' is the important category. It means real rows failed to map,
    so the mapping is genuinely incomplete, but the amounts cancel out, so the
    books still balance. Those need fixing before the next period and do not
    need chasing tonight.
    """
    if amount is None or pd.isna(amount):
        return "no_amount"
    if abs(float(amount)) < threshold:
        return "nets_to_zero"
    return "material"


def summarise_gaps(resolved: pd.DataFrame, spec: GapSpec) -> pd.DataFrame:
    """Aggregate the unmatched rows of one resolved frame into gap records."""
    unmatched = resolved[resolved[STATUS_COL] != str(MatchStatus.MATCHED)]

    columns = [
        "lookup",
        "key_values",
        "match_status",
        "match_reason",
        "row_count",
        "total_amount_entity_ccy",
        "materiality",
        "legal_entities_affected",
    ]

    if unmatched.empty:
        return pd.DataFrame(columns=columns)

    group_cols = list(spec.key_columns)
    grouped = unmatched.groupby(
        group_cols + [STATUS_COL, REASON_COL], dropna=False, sort=False
    )

    records: list[dict[str, object]] = []
    for keys, block in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        key_values = " | ".join(
            "(blank)" if pd.isna(k) else str(k) for k in keys[: len(group_cols)]
        )
        total = (
            float(block[spec.amount_column].sum())
            if spec.amount_column and spec.amount_column in block.columns
            else None
        )
        entities = (
            block[C.gl.legal_entity].nunique()
            if C.gl.legal_entity in block.columns
            else pd.NA
        )
        records.append(
            {
                "lookup": spec.lookup_name,
                "key_values": key_values,
                "match_status": keys[len(group_cols)],
                "match_reason": keys[len(group_cols) + 1],
                "row_count": len(block),
                "total_amount_entity_ccy": total,
                "materiality": classify_materiality(total),
                "legal_entities_affected": entities,
            }
        )

    gaps = pd.DataFrame(records, columns=columns)
    return gaps.sort_values(
        ["materiality", "row_count"], ascending=[True, False]
    ).reset_index(drop=True)


def combine_gaps(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack the gap reports from several lookups into one table."""
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return pd.DataFrame(
            columns=[
                "lookup",
                "key_values",
                "match_status",
                "match_reason",
                "row_count",
                "total_amount_entity_ccy",
                "materiality",
                "legal_entities_affected",
            ]
        )
    combined = pd.concat(non_empty, ignore_index=True)
    return combined.sort_values(
        ["lookup", "materiality", "row_count"], ascending=[True, True, False]
    ).reset_index(drop=True)


def gap_summary(gaps: pd.DataFrame) -> pd.DataFrame:
    """One row per lookup and reason: how many distinct gaps and rows."""
    if gaps.empty:
        return pd.DataFrame(
            columns=["lookup", "match_status", "materiality", "gaps", "rows", "amount"]
        )
    out = (
        gaps.groupby(["lookup", "match_status", "materiality"], dropna=False)
        .agg(
            gaps=("key_values", "size"),
            rows=("row_count", "sum"),
            amount=("total_amount_entity_ccy", "sum"),
        )
        .reset_index()
    )
    return out.sort_values(["lookup", "match_status"]).reset_index(drop=True)


def compare_to_published_gaps(
    computed: pd.DataFrame, published: pd.DataFrame
) -> pd.DataFrame:
    """Check our chart of accounts gaps against the workbook's own Mapping Gaps.

    This is the validation that the gap detection is right. The published sheet
    is the answer key: whatever it lists, we must also find.
    """
    pub = published.copy()
    pub["key_values"] = (
        pub[C.gaps.gl_account].astype(str).str.strip()
        + " | "
        + pub[C.gaps.trans_type].astype(str).str.strip()
    )
    pub = pub[["key_values", C.gaps.row_count, C.gaps.total_amount]].rename(
        columns={
            C.gaps.row_count: "published_row_count",
            C.gaps.total_amount: "published_amount",
        }
    )

    comp = computed[computed["lookup"] == "coa"][
        ["key_values", "row_count", "total_amount_entity_ccy"]
    ].rename(
        columns={
            "row_count": "computed_row_count",
            "total_amount_entity_ccy": "computed_amount",
        }
    )
    comp["key_values"] = comp["key_values"].str.strip()

    merged = pub.merge(comp, on="key_values", how="outer", indicator=True)
    merged["found_by_us"] = merged["_merge"].isin(["both", "right_only"])
    merged["in_published_sheet"] = merged["_merge"].isin(["both", "left_only"])
    merged["row_count_agrees"] = (
        merged["published_row_count"] == merged["computed_row_count"]
    )
    merged["amount_agrees"] = (
        (merged["published_amount"] - merged["computed_amount"]).abs()
        < config.AMOUNT_TOLERANCE
    )
    return merged.drop(columns=["_merge"])
