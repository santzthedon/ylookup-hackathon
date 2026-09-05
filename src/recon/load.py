"""Read the two workbooks into dataframes, once, with validation.

Every sheet the pipeline needs is loaded through one function here. Two reasons
for that. First, Excel parsing is slow and we do it repeatedly during
development, so results are cached on disk. Second, a sheet that is missing or
has been renamed should fail loudly at load time with a readable message,
rather than three modules later as an empty dataframe that quietly produces a
reconciliation of zero.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Cached Excel reading
# --------------------------------------------------------------------------


def _cache_key(path: Path, sheet: str, header: int) -> Path:
    """Build a cache filename from the workbook's path, size and mtime.

    If the underlying file is replaced the key changes, so a stale cache can
    never be served for a file that has been edited.
    """
    stat = path.stat()
    raw = f"{path.resolve()}|{sheet}|{header}|{stat.st_size}|{stat.st_mtime_ns}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    safe_sheet = "".join(c if c.isalnum() else "_" for c in sheet)[:40]
    return config.CACHE_DIR / f"{path.stem[:30]}__{safe_sheet}__{digest}.pkl"


def read_sheet(path: Path, sheet: str, use_cache: bool = True) -> pd.DataFrame:
    """Read one sheet from one workbook.

    The header row defaults to the first row, unless the sheet appears in
    config.HEADER_ROW_OVERRIDES (LE Mapping has a banner row above its header).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Workbook not found: {path}\n"
            f"Expected the dataset under {config.DATA_DIR}. "
            f"Set YLOOKUP_DATA_DIR if it lives elsewhere."
        )

    header = config.HEADER_ROW_OVERRIDES.get(sheet, 0)
    cache_path = _cache_key(path, sheet, header)

    if use_cache and cache_path.exists():
        log.debug("cache hit: %s / %s", path.name, sheet)
        return pd.read_pickle(cache_path)

    log.info("reading %s / %s", path.name, sheet)
    try:
        frame = pd.read_excel(path, sheet_name=sheet, header=header)
    except ValueError as exc:
        available = pd.ExcelFile(path).sheet_names
        raise ValueError(
            f"Sheet '{sheet}' not found in {path.name}.\n"
            f"Available sheets: {available}"
        ) from exc

    frame = _drop_unnamed_columns(frame)

    if use_cache:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(cache_path)

    return frame


def _drop_unnamed_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove the blank spacer columns Excel exports leave behind.

    Only drops a column if it is both unnamed and entirely empty, so a real
    column that merely lost its header is kept and will surface later as a
    missing column error rather than disappearing silently.
    """
    to_drop = [
        col
        for col in frame.columns
        if str(col).startswith("Unnamed:") and frame[col].isna().all()
    ]
    if to_drop:
        log.debug("dropping %d empty unnamed columns", len(to_drop))
    return frame.drop(columns=to_drop)


def require_columns(frame: pd.DataFrame, expected: list[str], label: str) -> None:
    """Fail with a readable message if the sheet is missing a column we need."""
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        raise KeyError(
            f"{label} is missing expected column(s): {missing}\n"
            f"Columns actually present: {list(frame.columns)}"
        )


# --------------------------------------------------------------------------
# The loaded dataset
# --------------------------------------------------------------------------


@dataclass
class Dataset:
    """Every sheet the pipeline needs, loaded and validated."""

    source_gl: pd.DataFrame
    upload: pd.DataFrame
    le_mapping: pd.DataFrame
    coa_mapping: pd.DataFrame
    investor_mapping: pd.DataFrame
    deal_mapping: pd.DataFrame
    entity_listing: pd.DataFrame
    movements_rec: pd.DataFrame
    mapping_gaps: pd.DataFrame
    corvus_coa: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        """One row per sheet: how many rows and columns came back."""
        rows = [
            ("source_gl", self.source_gl),
            ("upload", self.upload),
            ("le_mapping", self.le_mapping),
            ("coa_mapping", self.coa_mapping),
            ("investor_mapping", self.investor_mapping),
            ("deal_mapping", self.deal_mapping),
            ("entity_listing", self.entity_listing),
            ("movements_rec", self.movements_rec),
            ("mapping_gaps", self.mapping_gaps),
            ("corvus_coa", self.corvus_coa),
        ]
        return pd.DataFrame(
            [
                {"sheet": name, "rows": len(df), "columns": df.shape[1]}
                for name, df in rows
            ]
        )


def load_dataset(
    use_cache: bool = True,
    source_path: Path | None = None,
    loader_path: Path | None = None,
) -> Dataset:
    """Load and validate every sheet the pipeline depends on.

    Defaults to the configured dataset location. Pass explicit paths (for
    example, an uploaded workbook saved to a temporary file) to run the same
    validated load against a different pair of workbooks without touching
    global configuration.
    """
    src = source_path if source_path is not None else config.SOURCE_GL_PATH
    out = loader_path if loader_path is not None else config.LOADER_PATH

    source_gl = read_sheet(src, config.SHEET_SOURCE_GL, use_cache)
    upload = read_sheet(out, config.SHEET_UPLOAD, use_cache)
    le_mapping = read_sheet(out, config.SHEET_LE_MAPPING, use_cache)
    coa_mapping = read_sheet(out, config.SHEET_COA_MAPPING, use_cache)
    investor_mapping = read_sheet(out, config.SHEET_INVESTOR_MAPPING, use_cache)
    deal_mapping = read_sheet(out, config.SHEET_DEAL_MAPPING, use_cache)
    entity_listing = read_sheet(out, config.SHEET_ENTITY_LISTING, use_cache)
    movements_rec = read_sheet(out, config.SHEET_MOVEMENTS_REC, use_cache)
    mapping_gaps = read_sheet(out, config.SHEET_MAPPING_GAPS, use_cache)
    corvus_coa = read_sheet(out, config.SHEET_CORVUS_COA, use_cache)

    c = config.COLS

    require_columns(
        source_gl,
        [c.gl.legal_entity, c.gl.gl_account, c.gl.trans_type, c.gl.amount_entity],
        "Source GL",
    )
    require_columns(
        upload,
        [
            c.upload.legal_entity,
            c.upload.trans_type,
            c.upload.is_debit,
            c.upload.amount_entity,
        ],
        "Upload template",
    )
    require_columns(
        coa_mapping,
        [
            c.coa.target_gl_account,
            c.coa.target_trans_type_default,
            c.coa.target_trans_type_debit,
        ],
        "CoA mapping",
    )
    require_columns(
        le_mapping,
        [c.le.source_legal_entity, c.le.target_legal_entity, c.le.target_legal_entity_id],
        "LE mapping",
    )
    require_columns(
        movements_rec,
        [
            c.movements.legal_entity,
            c.movements.gl_account,
            c.movements.sum_debits,
            c.movements.sum_credits,
        ],
        "Movements Rec",
    )

    require_columns(
        corvus_coa,
        [c.corvus_coa.gl_account, c.corvus_coa.trans_type, c.corvus_coa.account_type],
        "Corvus CoA",
    )

    _validate_debit_flags(upload)

    return Dataset(
        source_gl=source_gl,
        upload=upload,
        le_mapping=le_mapping,
        coa_mapping=coa_mapping,
        investor_mapping=investor_mapping,
        deal_mapping=deal_mapping,
        entity_listing=entity_listing,
        movements_rec=movements_rec,
        mapping_gaps=mapping_gaps,
        corvus_coa=corvus_coa,
    )


def _validate_debit_flags(upload: pd.DataFrame) -> None:
    """The debit flag must only ever be Y or N.

    Every debit and credit total in the pipeline is split on this one column,
    so an unexpected third value would silently drop rows out of both sides of
    the reconciliation.
    """
    c = config.COLS.upload
    allowed = {c.debit_flag, c.credit_flag}
    found = set(upload[c.is_debit].dropna().unique())
    unexpected = found - allowed
    if unexpected:
        raise ValueError(
            f"Unexpected values in '{c.is_debit}': {sorted(unexpected)}. "
            f"Expected only {sorted(allowed)}."
        )
    null_count = int(upload[c.is_debit].isna().sum())
    if null_count:
        raise ValueError(
            f"{null_count} upload rows have no value in '{c.is_debit}'. "
            f"Each row must be either a debit or a credit."
        )
