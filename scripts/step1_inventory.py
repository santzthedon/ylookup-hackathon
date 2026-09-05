"""Step 1 check: prove every sheet loads and every expected column is present.

Run this before anything else. If it passes, the pipeline is reading the right
files and the column constants in config.py match reality.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from recon import config, load  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s"
    )

    print("=" * 78)
    print("STEP 1 — DATA INVENTORY")
    print("=" * 78)

    print("\nFile paths")
    print("-" * 78)
    for label, path in [
        ("source GL", config.SOURCE_GL_PATH),
        ("verified loader", config.LOADER_PATH),
    ]:
        state = "found" if path.exists() else "MISSING"
        size = f"{path.stat().st_size / 1_048_576:.1f} MB" if path.exists() else "-"
        print(f"  {label:<18} {state:<8} {size:>9}  {path}")

    print("\nLoading sheets")
    print("-" * 78)
    data = load.load_dataset()

    print("\nSheet inventory")
    print("-" * 78)
    print(data.summary().to_string(index=False))

    c = config.COLS

    print("\nKey figures")
    print("-" * 78)
    src_entities = data.source_gl[c.gl.legal_entity].nunique()
    upl_entities = data.upload[c.upload.legal_entity].nunique()
    print(f"  legal entities in source GL      : {src_entities}")
    print(f"  legal entities in upload template: {upl_entities}")
    print(f"  legal entities in movements rec   : "
          f"{data.movements_rec[c.movements.legal_entity].nunique()}")
    print(f"  distinct target transaction types : "
          f"{data.upload[c.upload.trans_type].nunique()}")
    print(f"  published movements rec rows      : {len(data.movements_rec)}")
    print(f"  published mapping gap rows        : {len(data.mapping_gaps)}")

    debits = (data.upload[c.upload.is_debit] == c.upload.debit_flag).sum()
    credits = (data.upload[c.upload.is_debit] == c.upload.credit_flag).sum()
    print(f"  upload debit rows / credit rows   : {debits} / {credits}")

    debit_total = data.upload.loc[
        data.upload[c.upload.is_debit] == c.upload.debit_flag, c.upload.amount_entity
    ].sum()
    credit_total = data.upload.loc[
        data.upload[c.upload.is_debit] == c.upload.credit_flag, c.upload.amount_entity
    ].sum()
    print(f"  total debits (entity currency)    : {debit_total:>22,.2f}")
    print(f"  total credits (entity currency)   : {credit_total:>22,.2f}")
    print(f"  difference                        : "
          f"{debit_total - credit_total:>22,.2f}")

    print("\nScope note")
    print("-" * 78)
    tranche_counts = (
        data.entity_listing.groupby(c.entities.tranche)[c.entities.entity]
        .nunique()
        .to_dict()
    )
    print(f"  entities per tranche in entity listing: {tranche_counts}")
    print("  the upload template covers tranche 1 only, which is why it holds")
    print("  fewer entities than the source GL")

    print("\n" + "=" * 78)
    print("STEP 1 PASSED — all sheets loaded, all expected columns present")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    raise SystemExit(main())
