"""Central configuration: where the files live and what the columns are called.

Everything that could change if Ylookup hands us a different quarter lives here,
so no other module contains a hard coded file path or spreadsheet column name.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Where the data lives
# --------------------------------------------------------------------------
# Default layout, relative to the repository root:
#
#   data/
#     02-investor-level-gl-to-loader/
#       source/Investor-Level GL - Q2 activity - all entities (anonymised).xlsx
#       output/Tranche 1 - reference and verified loader v4c (anonymised).xlsx
#
# Override with the YLOOKUP_DATA_DIR environment variable if the dataset sits
# somewhere else on your machine.

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("YLOOKUP_DATA_DIR", REPO_ROOT / "data"))
OUT_DIR = Path(os.environ.get("YLOOKUP_OUT_DIR", REPO_ROOT / "out"))

DATASET_DIR = DATA_DIR / "02-investor-level-gl-to-loader"

SOURCE_GL_PATH = (
    DATASET_DIR
    / "source"
    / "Investor-Level GL - Q2 activity - all entities (anonymised).xlsx"
)

LOADER_PATH = (
    DATASET_DIR
    / "output"
    / "Tranche 1 - reference and verified loader v4c (anonymised).xlsx"
)

# Parsed workbooks are cached here so we read the 34k row spreadsheet once
# rather than on every run. Excel parsing is the slowest part of the pipeline.
CACHE_DIR = Path(os.environ.get("YLOOKUP_CACHE_DIR", REPO_ROOT / ".cache"))


# --------------------------------------------------------------------------
# Sheet names
# --------------------------------------------------------------------------

SHEET_SOURCE_GL = "Investor-Level GL"

SHEET_UPLOAD = "Upload Template (VERIFIED v4c)"
SHEET_LE_MAPPING = "LE Mapping"
SHEET_COA_MAPPING = "CoA Mapping"
SHEET_INVESTOR_MAPPING = "Investor Mapping"
SHEET_DEAL_MAPPING = "Deal Mapping"
SHEET_ENTITY_LISTING = "Entity Listing"
SHEET_DEALS_LIST = "Deals List"
SHEET_INVESTORS_LIST = "Investors List"
SHEET_MAPPING_GAPS = "Mapping Gaps"
SHEET_MOVEMENTS_REC = "Movements Rec"
SHEET_BATCH_PREFERENCE = "Batch Preference"
SHEET_CORVUS_COA = "Corvus CoA"

# The LE Mapping sheet carries a banner row ("KESTREL Data" / "VERADO II DATA")
# above the real header, so the header is on the second row (index 1).
HEADER_ROW_OVERRIDES: dict[str, int] = {
    SHEET_LE_MAPPING: 1,
}


# --------------------------------------------------------------------------
# Column names
# --------------------------------------------------------------------------
# Grouped by the sheet they belong to. Referencing a column anywhere in the
# pipeline means referencing one of these constants, never a bare string, so a
# renamed column breaks in one obvious place instead of silently producing
# wrong totals.


@dataclass(frozen=True)
class SourceGLColumns:
    """Columns on the source general ledger export (the old system)."""

    fund_family: str = "Fund Family"
    legal_entity: str = "Legal Entity"
    vehicle: str = "Vehicle"
    deal_name: str = "Deal Name"
    deal_id: str = "Deal ID"
    position: str = "Position"
    position_id: str = "Position ID"
    batch_type: str = "Batch Type"
    batch_id: str = "Batch ID"
    je_index: str = "Journal Entry Index"
    trans_index: str = "Transaction Index"
    comments_batch: str = "Comments Batch"
    comments_transaction: str = "Comments transaction"
    gl_account: str = "GL Account"
    account_type: str = "Account Type"
    trans_type: str = "Trans Type"
    effective_date: str = "Effective Date"
    transaction_currency: str = "Transaction Currency"
    amount_local: str = "Amount (Local Currency)"
    debits_local: str = "Debits (Local Currency)"
    credits_local: str = "Credits (Local Currency)"
    entity_currency: str = "Legal Entity Currency"
    amount_entity: str = "Amount (Entity Currency)"
    debits_entity: str = "Debits (Entity Currency)"
    credits_entity: str = "Credits (Entity Currency)"
    allocation_rule: str = "Allocation Rule"
    investor: str = "Investor"
    rfx_id: str = "RFX ID"
    quantity: str = "Quantity"
    bank_account: str = "Bank Account"

    # Two columns are named "Static Date" and two are named "GL Date" in the
    # source workbook. pandas disambiguates duplicates by appending .1, so the
    # second occurrence of each is addressed through these names.
    static_date_from: str = "Static Date"
    static_date_to: str = "Static Date.1"
    gl_date: str = "GL Date"
    gl_date_alt: str = "GL Date.1"


@dataclass(frozen=True)
class UploadColumns:
    """Columns on the verified loader / upload template (the new system)."""

    batch_index: str = "Batch Index"
    je_index: str = "JE Index"
    trans_index: str = "Transaction Index"
    legal_entity: str = "Legal Entity"
    legal_entity_id: str = "Legal Entity ID"
    gl_date: str = "GL Date"
    effective_date: str = "Effective Date"
    deal_name: str = "Deal Name"
    deal_id: str = "Deal ID"
    position: str = "Position"
    position_id: str = "Position ID"
    trans_type: str = "Trans Type"
    transaction_currency: str = "Transaction Currency"
    amount_local: str = "Investor Amount (Local)"
    is_debit: str = "Is Debit"
    amount_entity: str = "Investor Amount (LE)"
    batch_type: str = "Batch Type"
    batch_comments: str = "Batch Comments"
    transaction_comments: str = "Transaction Comments"
    allocation_rule: str = "Allocation Rule"
    investor_account_id: str = "Investor Account ID"
    vehicle: str = "Vehicle"
    bank_account: str = "Bank Account"
    supplier: str = "Supplier"
    investor_quantity: str = "Investor Quantity"
    batch_ref: str = "Batch ref"

    # Values found in the is_debit column.
    debit_flag: str = "Y"
    credit_flag: str = "N"


@dataclass(frozen=True)
class CoAMappingColumns:
    """Chart of accounts crosswalk: old account and transaction type to new."""

    source_gl_account: str = "Helio GL Account"
    source_account_type: str = "Helio Account Type"
    source_trans_type: str = "Helio Trans Type"
    target_gl_code: str = "Verado II GL Account Code"
    target_gl_account: str = "Verado II GL Account"
    target_account_type: str = "Verado II Account Type"
    target_trans_type_default: str = "Verado II TransType (Default)"
    target_trans_type_debit: str = "Verado II TransType (Debit)"
    batch_type: str = "Batch Type"


@dataclass(frozen=True)
class LEMappingColumns:
    """Legal entity crosswalk."""

    fund_family: str = "Fund Family"
    source_legal_entity: str = "Legal Entity"
    source_currency: str = "Currency"
    target_legal_entity: str = "Corvus LE"
    target_legal_entity_id: str = "Corvus LE ID"
    target_currency: str = "Corvus Currency"


@dataclass(frozen=True)
class InvestorMappingColumns:
    """Investor crosswalk."""

    fund_family: str = "Fund Family"
    legal_entity: str = "Legal Entity"
    vehicle: str = "Vehicle"
    external_ref: str = "Specific External Ref 1"
    investor_name: str = "Investor Name"
    investor_lookup: str = "Investor Lookup"
    target_vehicle_name: str = "Corvus Veh Name"
    target_vehicle_id: str = "Corvus Veh ID"
    target_investor_name: str = "Corvus Investor Name"
    target_common_id: str = "Corvus Common ID"
    target_specific_id: str = "Corvus Specific Id"
    target_external_ref: str = "Corvus Ext Ref"
    ext_ref_check: str = "Ext Ref Check"


@dataclass(frozen=True)
class DealMappingColumns:
    """Deal and position crosswalk."""

    source_deal_name: str = "Deal Name"
    new_deal_name: str = "New Deal Name"
    source_position: str = "Position"
    new_position_name: str = "New Position Name"
    currency: str = "Currency"
    target_deal_name: str = "Corvus Deal Name"
    target_deal_id: str = "Corvus Deal ID"
    target_position_name: str = "Corvus Position Name"
    target_position_id: str = "Corvus Position ID"
    currency_check: str = "Curr Check"


@dataclass(frozen=True)
class MovementsRecColumns:
    """The published movements reconciliation we validate our own output against."""

    legal_entity: str = "Legal Entity"
    gl_account: str = "Verado II GL Account"
    sum_debits: str = "Sum Debits"
    sum_credits: str = "Sum Credits"
    net_movement: str = "Net Movement"


@dataclass(frozen=True)
class MappingGapsColumns:
    """The published list of accounts with no mapping, sent back for approval."""

    gl_account: str = "GL Account"
    trans_type: str = "Trans Type"
    row_count: str = "Row Count"
    total_amount: str = "Total Amount (Entity Currency)"
    proposed_account: str = "Proposed Verado II Account"
    proposed_trans_type: str = "Proposed Verado II TransType"
    approval: str = "Approval"


@dataclass(frozen=True)
class CorvusCoAColumns:
    """The target system's own chart of accounts.

    Every valid account and transaction type pair the new system will accept.
    This is the option set an unmapped account has to be resolved to, so it is
    what a decision offers the fund manager as choices.
    """

    account_type: str = "Account Type"
    account_code: str = "Account"
    gl_account: str = "GL Account"
    short_description: str = "Account Short Description"
    trans_type: str = "Trans Type"
    debit_credit: str = "Debit / Credit"


@dataclass(frozen=True)
class EntityListingColumns:
    """Which entities are in scope for migration, and in which tranche."""

    structure: str = "Structure"
    entity: str = "Entity"
    entity_type: str = "Entity Type"
    decision: str = "Decision"
    currency: str = "Currency"
    tranche: str = "Tranche"
    setup_status: str = "Corvus Set Up"
    to_complete: str = "Ylookup to complete (Y/N)"


@dataclass(frozen=True)
class Columns:
    """Single access point for every column group."""

    gl: SourceGLColumns = field(default_factory=SourceGLColumns)
    upload: UploadColumns = field(default_factory=UploadColumns)
    coa: CoAMappingColumns = field(default_factory=CoAMappingColumns)
    le: LEMappingColumns = field(default_factory=LEMappingColumns)
    investor: InvestorMappingColumns = field(default_factory=InvestorMappingColumns)
    deal: DealMappingColumns = field(default_factory=DealMappingColumns)
    movements: MovementsRecColumns = field(default_factory=MovementsRecColumns)
    gaps: MappingGapsColumns = field(default_factory=MappingGapsColumns)
    entities: EntityListingColumns = field(default_factory=EntityListingColumns)
    corvus_coa: CorvusCoAColumns = field(default_factory=CorvusCoAColumns)


COLS = Columns()


# --------------------------------------------------------------------------
# Tolerances
# --------------------------------------------------------------------------
# Two figures that differ by less than this are treated as equal. Floating
# point sums over 19k rows of currency do not land on exactly the same value,
# so a hard equality test would report thousands of false breaks.

AMOUNT_TOLERANCE = 0.01


# Gaps whose total amount is smaller than this in absolute terms are labelled
# "nets_to_zero" rather than "material". Several genuine mapping gaps in this
# dataset total to floating point residue such as 9.09e-13; those are real gaps
# in the mapping but not real differences in the books, and mixing them in with
# a four thousand pound gap would bury the one that matters.
MATERIALITY_THRESHOLD = 1.00
