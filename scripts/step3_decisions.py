"""Step 3: the consolidated decision report.

One pass over the source data, every question a human has to answer, framed
in language a fund manager can act on without opening the mapping workbook.
"""

from __future__ import annotations

import logging
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from recon import config, decisions as dec, load, mappings  # noqa: E402

C = config.COLS


def _wrap(text: str, indent: str = "     ") -> str:
    return textwrap.fill(
        text, width=78, initial_indent=indent, subsequent_indent=indent
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s"
    )
    pd.set_option("display.width", 140)

    data = load.load_dataset()
    maps = mappings.build_all(data)
    scoped = mappings.scope_source_gl(data)

    resolved = {
        "coa": maps.coa.apply(scoped, [C.gl.gl_account, C.gl.trans_type]),
        "investor": maps.investor.apply(scoped, [C.gl.rfx_id]),
        "position": maps.position.apply(scoped, [C.gl.deal_name, C.gl.position]),
    }

    decision_set = dec.build_decisions(data, resolved)
    summary = decision_set.summary()

    print("\n" + "=" * 80)
    print("  MIGRATION REVIEW — every decision found in one pass")
    print("=" * 80)
    print(
        f"\n  {summary['blocking']} decisions blocking, "
        f"{summary['deferred']} can wait"
    )
    print(
        f"  {summary['rows_blocked']:,} transactions held up, "
        f"worth {summary['value_blocked']:,.2f}"
    )
    print(f"  Across up to {summary['entities_affected']} funds")

    print("\n" + "-" * 80)
    print("  BLOCKING — answer these before anything is loaded")
    print("-" * 80)
    for d in decision_set.blocking:
        print(f"\n  [{d.id}]  {d.amount_display}   {d.row_count:,} transactions")
        print(_wrap(d.question, "     "))
        print(_wrap(d.detail, "       "))
        print(_wrap(f"Why it matters: {d.why_it_matters}", "       "))
        if d.answer_type == dec.AnswerType.CHOOSE_ACCOUNT:
            print(f"       Answer: choose one of {len(d.options):,} accounts")
        else:
            for opt in d.options:
                print(f"       - {opt}")

    print("\n" + "-" * 80)
    print(f"  CAN WAIT — {len(decision_set.deferred)} gaps that net to zero")
    print("-" * 80)
    for d in decision_set.deferred[:5]:
        print(f"  [{d.id}]  {d.row_count:>6,} transactions   {d.detail[:70]}...")
    if len(decision_set.deferred) > 5:
        print(f"  ... and {len(decision_set.deferred) - 5} more")

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    decision_set.to_json(config.OUT_DIR / "step3_decisions.json")
    decision_set.to_frame().to_csv(
        config.OUT_DIR / "step3_decisions.csv", index=False
    )

    print("\n" + "=" * 80)
    print("  wrote out/step3_decisions.json and out/step3_decisions.csv")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
