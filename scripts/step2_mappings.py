"""Step 2: resolve every source row through the four crosswalks.

Applies each lookup to the in scope general ledger, keeps every unmatched row,
validates the chart of accounts gaps against the workbook's own Mapping Gaps
sheet, and writes three reports to out/.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from recon import config, gaps, load, mappings  # noqa: E402
from recon.lookup import STATUS_COL, MatchStatus  # noqa: E402

C = config.COLS


def _rule(title: str = "") -> None:
    print("\n" + "=" * 82)
    if title:
        print(title)
        print("=" * 82)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s"
    )
    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", 60)

    _rule("STEP 2 — MAPPING RESOLUTION")

    data = load.load_dataset()
    maps = mappings.build_all(data)
    scoped = mappings.scope_source_gl(data)

    # ------------------------------------------------------------------
    # How each lookup was built
    # ------------------------------------------------------------------
    _rule("Lookup construction")
    print(maps.build_reports().to_string(index=False))

    ambiguous = maps.ambiguous_keys()
    print(f"\nAmbiguous mapping keys found: {len(ambiguous)}")
    if not ambiguous.empty:
        print(ambiguous.to_string(index=False))
        print(
            "\nThese keys appear in a mapping sheet more than once with different\n"
            "target values. The lookup refuses to choose, so rows using them are\n"
            "reported as gaps rather than resolved arbitrarily."
        )

    # ------------------------------------------------------------------
    # Apply every lookup
    # ------------------------------------------------------------------
    _rule("Resolution against the in scope general ledger")

    plans = [
        (maps.legal_entity, [C.gl.legal_entity]),
        (maps.coa, [C.gl.gl_account, C.gl.trans_type]),
        (maps.investor, [C.gl.rfx_id]),
        (maps.deal, [C.gl.deal_name]),
        (maps.position, [C.gl.deal_name, C.gl.position]),
    ]

    resolved: dict[str, pd.DataFrame] = {}
    rows = []
    for lookup, key_cols in plans:
        frame = lookup.apply(scoped, key_cols)
        resolved[lookup.name] = frame
        counts = frame[STATUS_COL].value_counts()
        rows.append(
            {
                "lookup": lookup.name,
                "source_rows": len(frame),
                "matched": int(counts.get(str(MatchStatus.MATCHED), 0)),
                "not_in_mapping": int(counts.get(str(MatchStatus.NOT_IN_MAPPING), 0)),
                "no_key": int(counts.get(str(MatchStatus.NO_KEY), 0)),
                "ambiguous": int(counts.get(str(MatchStatus.AMBIGUOUS_MAPPING), 0)),
                "match_rate": round(lookup.match_rate(frame) * 100, 2),
            }
        )

    resolution = pd.DataFrame(rows)
    print(resolution.to_string(index=False))

    # The loader's transaction types resolve to GL accounts; checked separately
    # because it runs against the upload template, not the source GL.
    tt = maps.trans_type_account.apply(data.upload, [C.upload.trans_type])
    tt_rate = maps.trans_type_account.match_rate(tt)
    print(
        f"\ntrans_type_account against the loader: "
        f"{tt_rate * 100:.2f}% of {len(tt)} rows resolve to a GL account"
    )

    # ------------------------------------------------------------------
    # Gap reports
    # ------------------------------------------------------------------
    _rule("Gap report")

    specs = [
        gaps.GapSpec("legal_entity", [C.gl.legal_entity], C.gl.amount_entity),
        gaps.GapSpec("coa", [C.gl.gl_account, C.gl.trans_type], C.gl.amount_entity),
        gaps.GapSpec("investor", [C.gl.investor, C.gl.rfx_id], C.gl.amount_entity),
        gaps.GapSpec("deal", [C.gl.deal_name], C.gl.amount_entity),
        gaps.GapSpec("position", [C.gl.deal_name, C.gl.position], C.gl.amount_entity),
    ]

    all_gaps = gaps.combine_gaps(
        [gaps.summarise_gaps(resolved[s.lookup_name], s) for s in specs]
    )

    print(gaps.gap_summary(all_gaps).to_string(index=False))
    print(f"\nDistinct gaps found: {len(all_gaps)}")

    material = all_gaps[all_gaps["materiality"] == "material"]
    print(f"Material gaps (amount at or above "
          f"{config.MATERIALITY_THRESHOLD:.2f}): {len(material)}")
    if not material.empty:
        print(
            material[
                [
                    "lookup",
                    "key_values",
                    "match_status",
                    "row_count",
                    "total_amount_entity_ccy",
                ]
            ].to_string(index=False)
        )

    # ------------------------------------------------------------------
    # Validate against the published Mapping Gaps sheet
    # ------------------------------------------------------------------
    _rule("Validation against the published Mapping Gaps sheet")

    comparison = gaps.compare_to_published_gaps(all_gaps, data.mapping_gaps)
    published = comparison[comparison["in_published_sheet"]]
    caught = published[published["found_by_us"]]

    print(f"Gaps listed in the published sheet : {len(published)}")
    print(f"Of those, found by this pipeline    : {len(caught)}")
    print(
        f"Row counts agreeing                 : "
        f"{int(caught['row_count_agrees'].sum())} of {len(caught)}"
    )
    print(
        f"Amounts agreeing within "
        f"{config.AMOUNT_TOLERANCE}        : "
        f"{int(caught['amount_agrees'].sum())} of {len(caught)}"
    )

    extra = comparison[~comparison["in_published_sheet"]]
    print(f"\nAdditional gaps we found that the sheet does not list: {len(extra)}")
    if not extra.empty:
        print(
            extra[["key_values", "computed_row_count", "computed_amount"]]
            .head(25)
            .to_string(index=False)
        )

    missed = published[~published["found_by_us"]]
    if not missed.empty:
        print("\nFAILURE: published gaps this pipeline did not detect:")
        print(missed[["key_values", "published_row_count"]].to_string(index=False))

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    _rule("Outputs")
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    unmatched_rows = []
    for spec in specs:
        frame = resolved[spec.lookup_name]
        block = frame[frame[STATUS_COL] != str(MatchStatus.MATCHED)].copy()
        if block.empty:
            continue
        keep = [
            C.gl.legal_entity,
            C.gl.gl_account,
            C.gl.trans_type,
            C.gl.deal_name,
            C.gl.position,
            C.gl.investor,
            C.gl.rfx_id,
            C.gl.batch_id,
            C.gl.amount_entity,
            STATUS_COL,
            "match_reason",
        ]
        block = block[[c for c in keep if c in block.columns]]
        block.insert(0, "lookup", spec.lookup_name)
        unmatched_rows.append(block)

    unmatched = (
        pd.concat(unmatched_rows, ignore_index=True)
        if unmatched_rows
        else pd.DataFrame()
    )

    written = [
        (config.OUT_DIR / "step2_gaps.csv", all_gaps),
        (config.OUT_DIR / "step2_unmatched_rows.csv", unmatched),
        (config.OUT_DIR / "step2_resolution_summary.csv", resolution),
        (config.OUT_DIR / "step2_published_gap_validation.csv", comparison),
    ]
    for path, frame in written:
        frame.to_csv(path, index=False)
        print(f"  wrote {path.name:<38} {len(frame):>6} rows")

    ok = missed.empty
    _rule("STEP 2 PASSED" if ok else "STEP 2 FAILED — see missed gaps above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
