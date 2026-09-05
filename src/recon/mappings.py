"""The four crosswalks, built from the workbook's mapping sheets.

Each is a Lookup from recon.lookup. What differs between them is only which
columns form the key and which are carried across.

Key choices below were made by testing match rates against the real data
rather than by guessing, and each is justified in a comment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from . import config
from .load import Dataset
from .lookup import Lookup, LookupBuildReport

log = logging.getLogger(__name__)

C = config.COLS


# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------


def in_scope_entities(dataset: Dataset) -> set[str]:
    """The legal entities this migration tranche actually covers.

    The source GL holds 79 legal entities but the verified loader holds 52,
    because only tranche 1 was migrated. Reconciling all 79 would report 27
    entities as missing from the loader, which is true but is a scoping fact,
    not a break. Everything downstream works on the tranche 1 subset.
    """
    return set(dataset.upload[C.upload.legal_entity].dropna().unique())


def scope_source_gl(dataset: Dataset) -> pd.DataFrame:
    """The source GL restricted to entities present in the loader."""
    entities = in_scope_entities(dataset)
    scoped = dataset.source_gl[
        dataset.source_gl[C.gl.legal_entity].isin(entities)
    ].copy()
    log.info(
        "scope: %d of %d source GL rows across %d of %d legal entities",
        len(scoped),
        len(dataset.source_gl),
        scoped[C.gl.legal_entity].nunique(),
        dataset.source_gl[C.gl.legal_entity].nunique(),
    )
    return scoped


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def build_legal_entity_lookup(dataset: Dataset) -> Lookup:
    """Old legal entity name to new name and numeric ID.

    Key: the source legal entity name alone. It is unique in the mapping sheet
    (zero duplicates) and resolves every in scope GL row.
    """
    return Lookup(
        name="legal_entity",
        mapping=dataset.le_mapping,
        key_columns=[C.le.source_legal_entity],
        value_columns=[C.le.target_legal_entity, C.le.target_legal_entity_id],
        description="Source legal entity name to target system entity and ID",
    )


def build_coa_lookup(dataset: Dataset) -> Lookup:
    """Old GL account plus transaction type to the new account and type.

    Key: both columns together, because the same old account splits into
    several new ones depending on the transaction type. '10010 - Cash' with
    'Cash Paid' and '10010 - Cash' with 'Cash Received' are different rows in
    the mapping. Keying on the account alone would make every cash account
    ambiguous.

    Six of the 105 mapping rows have no source account and are excluded by the
    blank key rule; they carry a transaction type only and belong to the
    transaction type lookup below.
    """
    return Lookup(
        name="coa",
        mapping=dataset.coa_mapping,
        key_columns=[C.coa.source_gl_account, C.coa.source_trans_type],
        value_columns=[
            C.coa.target_gl_account,
            C.coa.target_trans_type_default,
            C.coa.batch_type,
        ],
        description="Source GL account and transaction type to target account",
    )


def build_trans_type_account_lookup(dataset: Dataset) -> Lookup:
    """Target transaction type to target GL account.

    Needed because the loader records a transaction type but not a GL account,
    while the movements reconciliation is grouped by GL account. This lookup is
    the bridge between the two, and therefore what makes the footing check in
    step 3 possible at all.

    Built by stacking the default and debit transaction type columns, since a
    row can supply either. Verified against the data: all 45 transaction types
    used in the loader resolve, and none maps to more than one account.
    """
    coa = dataset.coa_mapping
    frames = []
    for col in (C.coa.target_trans_type_default, C.coa.target_trans_type_debit):
        part = coa[[col, C.coa.target_gl_account]].copy()
        part = part.rename(columns={col: "target_trans_type"})
        frames.append(part)

    stacked = pd.concat(frames, ignore_index=True)
    stacked = stacked.dropna(subset=["target_trans_type", C.coa.target_gl_account])
    stacked = stacked.drop_duplicates()

    return Lookup(
        name="trans_type_account",
        mapping=stacked,
        key_columns=["target_trans_type"],
        value_columns=[C.coa.target_gl_account],
        description="Target transaction type to target GL account",
    )


def build_investor_lookup(dataset: Dataset) -> Lookup:
    """Old investor record to the new system's investor IDs.

    Key: the external reference, which appears as 'RFX ID' in the GL and
    'Specific External Ref 1' in the mapping sheet. Tested against the
    alternative of legal entity plus investor name, which leaves 248 in scope
    rows unresolved; the external reference leaves none. Names are also the
    less safe key here, since several investors differ only by a suffix.

    The mapping contains repeated external references. Most repeat with
    identical target IDs and collapse harmlessly. Any that genuinely conflict
    are reported as ambiguous rather than resolved arbitrarily.
    """
    return Lookup(
        name="investor",
        mapping=dataset.investor_mapping,
        key_columns=[C.investor.external_ref],
        value_columns=[
            C.investor.target_investor_name,
            C.investor.target_common_id,
            C.investor.target_specific_id,
            C.investor.target_vehicle_id,
        ],
        description="Investor external reference to target system investor IDs",
    )


def build_deal_lookup(dataset: Dataset) -> Lookup:
    """Old deal name to the new deal name and ID.

    Key: the deal name alone. The mapping sheet holds deal only rows (63 of
    219) alongside deal and position rows; keying on the deal name and taking
    distinct target values collapses those safely, because every row for a
    given deal carries the same deal ID.
    """
    deals = dataset.deal_mapping[
        [
            C.deal.source_deal_name,
            C.deal.target_deal_name,
            C.deal.target_deal_id,
        ]
    ].drop_duplicates()

    return Lookup(
        name="deal",
        mapping=deals,
        key_columns=[C.deal.source_deal_name],
        value_columns=[C.deal.target_deal_name, C.deal.target_deal_id],
        description="Source deal name to target deal name and ID",
    )


def build_position_lookup(dataset: Dataset) -> Lookup:
    """Old deal and position together to the new position name and ID.

    Key: deal name plus position, because a position name is only meaningful
    inside its deal. Kept separate from the deal lookup so a row whose deal
    resolves but whose position does not is reported precisely that way,
    rather than failing wholesale.
    """
    positions = dataset.deal_mapping.dropna(subset=[C.deal.source_position])[
        [
            C.deal.source_deal_name,
            C.deal.source_position,
            C.deal.target_position_name,
            C.deal.target_position_id,
        ]
    ].drop_duplicates()

    return Lookup(
        name="position",
        mapping=positions,
        key_columns=[C.deal.source_deal_name, C.deal.source_position],
        value_columns=[C.deal.target_position_name, C.deal.target_position_id],
        description="Source deal and position to target position name and ID",
    )


# --------------------------------------------------------------------------
# Bundle
# --------------------------------------------------------------------------


@dataclass
class MappingSet:
    """All lookups the pipeline uses, built once."""

    legal_entity: Lookup
    coa: Lookup
    trans_type_account: Lookup
    investor: Lookup
    deal: Lookup
    position: Lookup

    def all_lookups(self) -> list[Lookup]:
        return [
            self.legal_entity,
            self.coa,
            self.trans_type_account,
            self.investor,
            self.deal,
            self.position,
        ]

    def build_reports(self) -> pd.DataFrame:
        """One row per lookup describing how it was constructed."""
        return pd.DataFrame([lk.build_report.as_row() for lk in self.all_lookups()])

    def ambiguous_keys(self) -> pd.DataFrame:
        """Every mapping key that appears more than once with conflicting values."""
        rows: list[dict[str, object]] = []
        for lk in self.all_lookups():
            report: LookupBuildReport = lk.build_report
            for key in report.ambiguous_keys:
                rows.append(
                    {
                        "lookup": lk.name,
                        "key": key.replace("\u241f", " | "),
                        "issue": "conflicting values for the same key in mapping",
                    }
                )
        return pd.DataFrame(
            rows, columns=["lookup", "key", "issue"]
        )


def build_all(dataset: Dataset) -> MappingSet:
    """Build every lookup from the loaded dataset."""
    return MappingSet(
        legal_entity=build_legal_entity_lookup(dataset),
        coa=build_coa_lookup(dataset),
        trans_type_account=build_trans_type_account_lookup(dataset),
        investor=build_investor_lookup(dataset),
        deal=build_deal_lookup(dataset),
        position=build_position_lookup(dataset),
    )
