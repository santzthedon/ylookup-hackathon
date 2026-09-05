"""Turning technical gaps into questions a fund manager can answer.

This is the module the product exists for.

Ylookup told us this workbook took four iterations, and that the iterations
were about getting more information from fund managers. The fund manager in
call 1 said the same thing from the other side: the delay is not the
turnaround, it is the count of turns, and that is the drag on his time.

Four turns happen when questions are discovered one at a time. You build,
you hit an unmapped account, you email, you wait, you rebuild, you hit an
unresolved position, you email again. Every question was easy. Finding them
sequentially is what cost a month.

So this module takes everything the mapping layer found in a single pass and
frames each one as a decision: what is being asked, what it is worth, what
evidence sits behind it, and what the possible answers are. One consolidated
list, produced before anything is built.

The audience is explicitly a non technical fund manager. No decision below
uses the words crosswalk, lookup, join or key. If a question cannot be
answered by someone who has never opened the mapping workbook, it is not
framed well enough.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from . import config
from .load import Dataset
from .lookup import STATUS_COL, MatchStatus

log = logging.getLogger(__name__)

C = config.COLS

# How many example ledger rows to attach to a decision. Enough to see the
# pattern, few enough to read on one screen.
EVIDENCE_ROWS = 5


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------


class Priority:
    """Whether a decision holds the migration up.

    BLOCKING means real money is attached and nothing should be loaded until
    someone answers. DEFERRED means the gap is real but the amounts net to
    nothing, so it should be fixed before the next period and does not need
    chasing tonight. The split exists so a reviewer can triage; a list where
    everything looks equally urgent is a list nobody finishes.
    """

    BLOCKING = "blocking"
    DEFERRED = "deferred"


class DecisionKind:
    """The categories of question this pipeline can raise."""

    UNMAPPED_ACCOUNT = "unmapped_account"
    AMBIGUOUS_INVESTOR = "ambiguous_investor"
    UNRESOLVED_POSITION = "unresolved_position"
    UNMAPPED_ENTITY = "unmapped_entity"
    UNMAPPED_DEAL = "unmapped_deal"


class AnswerType:
    """What shape of answer the question needs, which drives the UI control."""

    CHOOSE_ACCOUNT = "choose_account"
    CHOOSE_ONE_OF = "choose_one_of"
    CHOOSE_APPROACH = "choose_approach"


# --------------------------------------------------------------------------
# The decision record
# --------------------------------------------------------------------------


@dataclass
class Decision:
    """One question that needs a human answer before the migration completes."""

    id: str
    kind: str
    priority: str

    question: str
    """One line, plain English, phrased as a question."""

    detail: str
    """Two or three sentences of context. What we found and why we stopped."""

    why_it_matters: str
    """What happens if this is answered wrongly or not at all."""

    amount: float | None
    row_count: int
    entities_affected: int
    materiality: str

    answer_type: str
    options: list[str] = field(default_factory=list)
    suggestion: str | None = None
    """Reserved for a proposed answer. Always requires approval, never applied
    automatically. Left unset until the suggestion layer is built."""

    evidence: list[dict] = field(default_factory=list)
    """Example ledger rows behind the question, so the answer can be given
    without opening the source file."""

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def amount_display(self) -> str:
        """User facing amount, always shown unsigned. Direction is not the
        fund manager's problem; `amount` keeps the sign for JSON consumers."""
        if self.amount is None or pd.isna(self.amount):
            return "n/a"
        return f"{abs(self.amount):,.2f}"


@dataclass
class DecisionSet:
    """Every decision found in one pass over the source data."""

    decisions: list[Decision]

    @property
    def blocking(self) -> list[Decision]:
        return [d for d in self.decisions if d.priority == Priority.BLOCKING]

    @property
    def deferred(self) -> list[Decision]:
        return [d for d in self.decisions if d.priority == Priority.DEFERRED]

    def summary(self) -> dict:
        """The headline figures. This is what the first screen shows."""
        blocking = self.blocking
        return {
            "decisions_found": len(self.decisions),
            "blocking": len(blocking),
            "deferred": len(self.deferred),
            "rows_affected": sum(d.row_count for d in self.decisions),
            "rows_blocked": sum(d.row_count for d in blocking),
            "value_blocked": sum(
                abs(d.amount) for d in blocking if d.amount is not None
            ),
            "entities_affected": max(
                (d.entities_affected for d in self.decisions), default=0
            ),
        }

    def to_frame(self) -> pd.DataFrame:
        """Tabular view, for CSV output and for tests."""
        if not self.decisions:
            return pd.DataFrame(
                columns=[
                    "id",
                    "priority",
                    "kind",
                    "question",
                    "row_count",
                    "amount",
                    "entities_affected",
                    "materiality",
                ]
            )
        return pd.DataFrame(
            [
                {
                    "id": d.id,
                    "priority": d.priority,
                    "kind": d.kind,
                    "question": d.question,
                    "row_count": d.row_count,
                    "amount": d.amount,
                    "entities_affected": d.entities_affected,
                    "materiality": d.materiality,
                }
                for d in self.decisions
            ]
        )

    def to_json(self, path: Path) -> None:
        payload = {
            "summary": self.summary(),
            "decisions": [d.as_dict() for d in self.decisions],
        }
        path.write_text(json.dumps(payload, indent=2, default=str))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _priority_for(amount: float | None) -> str:
    """A decision blocks only if real money is attached to it."""
    if amount is None or pd.isna(amount):
        return Priority.BLOCKING
    return (
        Priority.BLOCKING
        if abs(float(amount)) >= config.MATERIALITY_THRESHOLD
        else Priority.DEFERRED
    )


def _materiality_for(amount: float | None) -> str:
    if amount is None or pd.isna(amount):
        return "no_amount"
    return (
        "material"
        if abs(float(amount)) >= config.MATERIALITY_THRESHOLD
        else "nets_to_zero"
    )


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    word = singular if count == 1 else (plural or f"{singular}s")
    return f"{count:,} {word}"


def _money(amount: float | None) -> str:
    if amount is None or pd.isna(amount):
        return "an unknown amount"
    value = abs(float(amount))
    if value < config.MATERIALITY_THRESHOLD:
        return "a net amount of effectively zero"
    return f"{value:,.2f}"


def _evidence(block: pd.DataFrame, columns: list[str]) -> list[dict]:
    """A few readable example rows, using column names a reviewer recognises."""
    present = [c for c in columns if c in block.columns]
    sample = block[present].head(EVIDENCE_ROWS).copy()
    for col in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[col]):
            sample[col] = sample[col].dt.strftime("%d/%m/%Y")
    return sample.to_dict(orient="records")


EVIDENCE_COLUMNS = [
    C.gl.legal_entity,
    C.gl.gl_account,
    C.gl.trans_type,
    C.gl.deal_name,
    C.gl.position,
    C.gl.investor,
    C.gl.effective_date,
    C.gl.amount_entity,
]


def target_account_options(dataset: Dataset) -> list[str]:
    """Every GL account the new system will accept, as answer choices."""
    accounts = (
        dataset.corvus_coa[C.corvus_coa.gl_account].dropna().astype(str).str.strip()
    )
    return sorted(accounts.unique().tolist())


# --------------------------------------------------------------------------
# Builders, one per kind of question
# --------------------------------------------------------------------------


def unmapped_account_decisions(
    resolved_coa: pd.DataFrame, dataset: Dataset
) -> list[Decision]:
    """Old accounts with no home in the new chart of accounts."""
    unmatched = resolved_coa[resolved_coa[STATUS_COL] != str(MatchStatus.MATCHED)]
    if unmatched.empty:
        return []

    options = target_account_options(dataset)
    out: list[Decision] = []

    grouped = unmatched.groupby(
        [C.gl.gl_account, C.gl.trans_type], dropna=False, sort=False
    )
    for index, (keys, block) in enumerate(grouped, start=1):
        account, trans_type = keys
        amount = float(block[C.gl.amount_entity].sum())
        entities = int(block[C.gl.legal_entity].nunique())
        priority = _priority_for(amount)

        if priority == Priority.BLOCKING:
            question = (
                f"Where should {_plural(len(block), 'transaction')} worth "
                f"{_money(amount)} post in the new system?"
            )
        else:
            question = (
                f"{_plural(len(block), 'transaction')} have no account in the "
                f"new system. They net to zero, so where should they post "
                f"before next period?"
            )

        out.append(
            Decision(
                id=f"account-{index:02d}",
                kind=DecisionKind.UNMAPPED_ACCOUNT,
                priority=priority,
                question=question,
                detail=(
                    f"In the old system these sit in \u201c{account}\u201d and are "
                    f"labelled \u201c{trans_type}\u201d. The new system has no "
                    f"equivalent, so nobody has said where they belong. They "
                    f"affect {_plural(entities, 'fund')}."
                ),
                why_it_matters=(
                    "Until this is answered these transactions cannot be loaded. "
                    "Loading them against the wrong account would misstate that "
                    "account for every affected fund."
                ),
                amount=amount,
                row_count=len(block),
                entities_affected=entities,
                materiality=_materiality_for(amount),
                answer_type=AnswerType.CHOOSE_ACCOUNT,
                options=options,
                evidence=_evidence(block, EVIDENCE_COLUMNS),
            )
        )

    return out


def ambiguous_investor_decisions(
    resolved_investor: pd.DataFrame, dataset: Dataset
) -> list[Decision]:
    """One investor reference that points at more than one investor."""
    ambiguous = resolved_investor[
        resolved_investor[STATUS_COL] == str(MatchStatus.AMBIGUOUS_MAPPING)
    ]
    if ambiguous.empty:
        return []

    out: list[Decision] = []
    grouped = ambiguous.groupby(C.gl.rfx_id, dropna=False, sort=False)

    for index, (reference, block) in enumerate(grouped, start=1):
        amount = float(block[C.gl.amount_entity].sum())
        entities = int(block[C.gl.legal_entity].nunique())

        candidates = dataset.investor_mapping[
            dataset.investor_mapping[C.investor.external_ref].astype(str).str.strip()
            == str(reference).strip()
        ]
        options = [
            f"{row[C.investor.target_investor_name]} "
            f"(investor ID {row[C.investor.target_specific_id]})"
            for _, row in candidates.iterrows()
        ]

        out.append(
            Decision(
                id=f"investor-{index:02d}",
                kind=DecisionKind.AMBIGUOUS_INVESTOR,
                priority=Priority.BLOCKING,
                question=(
                    "One investor reference points at two different investors. "
                    "Which is correct?"
                ),
                detail=(
                    f"Reference \u201c{reference}\u201d appears more than once in the "
                    f"investor records, pointing at different investors. "
                    f"{_plural(len(block), 'transaction')} across "
                    f"{_plural(entities, 'fund')} use it."
                ),
                why_it_matters=(
                    "Picking the wrong one would post these transactions to "
                    "another investor's account. The totals would still balance, "
                    "so nothing downstream would flag it."
                ),
                amount=amount,
                row_count=len(block),
                entities_affected=entities,
                materiality=_materiality_for(amount),
                answer_type=AnswerType.CHOOSE_ONE_OF,
                options=options,
                evidence=_evidence(block, EVIDENCE_COLUMNS),
            )
        )

    return out


def unresolved_position_decisions(resolved_position: pd.DataFrame) -> list[Decision]:
    """Holdings referenced by the ledger that do not exist in the new system."""
    unmatched = resolved_position[
        resolved_position[STATUS_COL] != str(MatchStatus.MATCHED)
    ]
    if unmatched.empty:
        return []

    options = [
        "Create the holding in the new system and post against it",
        "Post at deal level without a specific holding",
        "Exclude these transactions from the migration",
    ]

    out: list[Decision] = []
    grouped = unmatched.groupby(
        [C.gl.deal_name, C.gl.position], dropna=False, sort=False
    )

    for index, (keys, block) in enumerate(grouped, start=1):
        deal, position = keys
        amount = float(block[C.gl.amount_entity].sum())
        entities = int(block[C.gl.legal_entity].nunique())
        priority = _priority_for(amount)

        if priority == Priority.BLOCKING:
            question = (
                f"{_plural(len(block), 'transaction')} covering {_money(amount)} "
                f"reference a holding that does not exist in the new system. "
                f"How should they be handled?"
            )
        else:
            question = (
                f"{_plural(len(block), 'transaction')} reference a holding that "
                f"does not exist in the new system. They net to zero. How should "
                f"they be handled?"
            )

        out.append(
            Decision(
                id=f"position-{index:02d}",
                kind=DecisionKind.UNRESOLVED_POSITION,
                priority=priority,
                question=question,
                detail=(
                    f"The deal \u201c{deal}\u201d exists in the new system, but the "
                    f"holding \u201c{position}\u201d within it does not. "
                    f"{_plural(entities, 'fund')} affected."
                ),
                why_it_matters=(
                    "Loading these without a holding leaves transactions that "
                    "belong to no identifiable investment, which breaks any "
                    "later reporting by holding."
                ),
                amount=amount,
                row_count=len(block),
                entities_affected=entities,
                materiality=_materiality_for(amount),
                answer_type=AnswerType.CHOOSE_APPROACH,
                options=options,
                evidence=_evidence(block, EVIDENCE_COLUMNS),
            )
        )

    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def build_decisions(
    dataset: Dataset, resolved: dict[str, pd.DataFrame]
) -> DecisionSet:
    """Every question raised by one pass over the source data.

    `resolved` is the output of applying each lookup, keyed by lookup name, as
    produced in the step 2 pipeline.
    """
    decisions: list[Decision] = []

    if "coa" in resolved:
        decisions += unmapped_account_decisions(resolved["coa"], dataset)
    if "investor" in resolved:
        decisions += ambiguous_investor_decisions(resolved["investor"], dataset)
    if "position" in resolved:
        decisions += unresolved_position_decisions(resolved["position"])

    # Blocking first, then by value, so the list reads in the order a reviewer
    # should work through it.
    decisions.sort(
        key=lambda d: (
            d.priority != Priority.BLOCKING,
            -abs(d.amount) if d.amount is not None else 0,
        )
    )

    log.info(
        "%d decisions found in one pass (%d blocking)",
        len(decisions),
        sum(1 for d in decisions if d.priority == Priority.BLOCKING),
    )
    return DecisionSet(decisions)
