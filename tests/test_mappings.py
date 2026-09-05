"""Integration tests against the real workbook.

Slower than the unit tests because they parse the spreadsheets, but they are
the ones that prove the pipeline agrees with the verified answer key. The
published Mapping Gaps sheet is treated as ground truth: anything on it, we
must find.
"""

from __future__ import annotations

import pandas as pd
import pytest

from recon import config, gaps, load, mappings
from recon.lookup import STATUS_COL, MatchStatus

C = config.COLS


@pytest.fixture(scope="module")
def dataset():
    if not config.SOURCE_GL_PATH.exists():
        pytest.skip(f"dataset not present at {config.DATA_DIR}")
    return load.load_dataset()


@pytest.fixture(scope="module")
def maps(dataset):
    return mappings.build_all(dataset)


@pytest.fixture(scope="module")
def scoped(dataset):
    return mappings.scope_source_gl(dataset)


# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------


class TestScope:
    def test_scope_is_the_loader_entities(self, dataset, scoped):
        entities = mappings.in_scope_entities(dataset)
        assert scoped[C.gl.legal_entity].nunique() == len(entities)

    def test_scope_is_smaller_than_the_full_ledger(self, dataset, scoped):
        assert len(scoped) < len(dataset.source_gl)

    def test_scope_covers_every_loader_entity(self, dataset, scoped):
        loader_entities = set(dataset.upload[C.upload.legal_entity].dropna())
        assert loader_entities <= set(scoped[C.gl.legal_entity])


# --------------------------------------------------------------------------
# Lookup construction against real data
# --------------------------------------------------------------------------


class TestRealMappings:
    def test_legal_entity_mapping_has_no_duplicate_keys(self, maps):
        assert maps.legal_entity.build_report.ambiguous_keys == []

    def test_coa_key_needs_both_columns(self, maps, dataset):
        """Keying the chart of accounts on the account alone would be ambiguous.

        The same old account splits into several new ones by transaction type,
        so this asserts the two column key is necessary, not incidental.
        """
        from recon.lookup import Lookup

        account_only = Lookup(
            "coa_account_only",
            dataset.coa_mapping,
            [C.coa.source_gl_account],
            [C.coa.target_gl_account, C.coa.target_trans_type_default],
        )
        assert account_only.build_report.ambiguous_keys, (
            "expected the account only key to be ambiguous, which is why the "
            "real lookup keys on account plus transaction type"
        )

    def test_trans_type_account_lookup_is_unambiguous(self, maps):
        assert maps.trans_type_account.build_report.ambiguous_keys == []


# --------------------------------------------------------------------------
# Resolution rates
# --------------------------------------------------------------------------


class TestResolution:
    def test_every_legal_entity_resolves(self, maps, scoped):
        out = maps.legal_entity.apply(scoped, [C.gl.legal_entity])
        assert maps.legal_entity.match_rate(out) == 1.0

    def test_every_investor_resolves_by_external_reference(self, maps, scoped):
        out = maps.investor.apply(scoped, [C.gl.rfx_id])
        assert maps.investor.match_rate(out) == 1.0

    def test_every_deal_resolves(self, maps, scoped):
        out = maps.deal.apply(scoped, [C.gl.deal_name])
        assert maps.deal.match_rate(out) == 1.0

    def test_every_loader_trans_type_resolves_to_an_account(self, maps, dataset):
        """Step 3's footing check depends on this being complete."""
        out = maps.trans_type_account.apply(dataset.upload, [C.upload.trans_type])
        assert maps.trans_type_account.match_rate(out) == 1.0

    def test_no_rows_are_lost_by_any_lookup(self, maps, scoped):
        for lookup, keys in [
            (maps.legal_entity, [C.gl.legal_entity]),
            (maps.coa, [C.gl.gl_account, C.gl.trans_type]),
            (maps.investor, [C.gl.rfx_id]),
            (maps.deal, [C.gl.deal_name]),
            (maps.position, [C.gl.deal_name, C.gl.position]),
        ]:
            out = lookup.apply(scoped, keys)
            assert len(out) == len(scoped), f"{lookup.name} changed the row count"

    def test_unmatched_rows_retain_their_amounts(self, maps, scoped):
        """A gap must not lose the money attached to it."""
        out = maps.coa.apply(scoped, [C.gl.gl_account, C.gl.trans_type])
        unmatched = maps.coa.unmatched(out)
        assert unmatched[C.gl.amount_entity].notna().all()
        assert out[C.gl.amount_entity].sum() == pytest.approx(
            scoped[C.gl.amount_entity].sum()
        )


# --------------------------------------------------------------------------
# Gap detection validated against the published sheet
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def coa_gaps(maps, scoped):
    resolved = maps.coa.apply(scoped, [C.gl.gl_account, C.gl.trans_type])
    spec = gaps.GapSpec("coa", [C.gl.gl_account, C.gl.trans_type], C.gl.amount_entity)
    return gaps.summarise_gaps(resolved, spec)


class TestPublishedGapValidation:
    def test_every_published_gap_is_detected(self, coa_gaps, dataset):
        comparison = gaps.compare_to_published_gaps(coa_gaps, dataset.mapping_gaps)
        published = comparison[comparison["in_published_sheet"]]
        assert len(published) == len(dataset.mapping_gaps)
        assert published["found_by_us"].all(), (
            "the pipeline missed a gap the verified workbook lists"
        )

    def test_published_row_counts_agree(self, coa_gaps, dataset):
        comparison = gaps.compare_to_published_gaps(coa_gaps, dataset.mapping_gaps)
        both = comparison[comparison["in_published_sheet"] & comparison["found_by_us"]]
        assert both["row_count_agrees"].all()

    def test_published_amounts_agree(self, coa_gaps, dataset):
        comparison = gaps.compare_to_published_gaps(coa_gaps, dataset.mapping_gaps)
        both = comparison[comparison["in_published_sheet"] & comparison["found_by_us"]]
        assert both["amount_agrees"].all()

    def test_the_material_published_gap_is_flagged_material(self, coa_gaps):
        """The interest income gap is roughly 4,867 and must not be filed as noise."""
        material = coa_gaps[coa_gaps["materiality"] == "material"]
        assert len(material) >= 1
        assert material["total_amount_entity_ccy"].abs().max() > 1000


# --------------------------------------------------------------------------
# Materiality
# --------------------------------------------------------------------------


class TestMateriality:
    def test_floating_point_residue_is_not_material(self):
        assert gaps.classify_materiality(9.094947e-13) == "nets_to_zero"

    def test_exact_zero_is_not_material(self):
        assert gaps.classify_materiality(0.0) == "nets_to_zero"

    def test_real_amount_is_material(self):
        assert gaps.classify_materiality(4867.16) == "material"

    def test_negative_amounts_are_judged_on_magnitude(self):
        assert gaps.classify_materiality(-4867.16) == "material"

    def test_missing_amount_is_labelled_separately(self):
        assert gaps.classify_materiality(None) == "no_amount"
        assert gaps.classify_materiality(pd.NA) == "no_amount"


# --------------------------------------------------------------------------
# Gap report shape
# --------------------------------------------------------------------------


class TestGapReport:
    def test_gap_rows_sum_to_the_unmatched_row_count(self, maps, scoped, coa_gaps):
        resolved = maps.coa.apply(scoped, [C.gl.gl_account, C.gl.trans_type])
        unmatched = (resolved[STATUS_COL] != str(MatchStatus.MATCHED)).sum()
        assert coa_gaps["row_count"].sum() == unmatched

    def test_every_gap_has_a_reason(self, coa_gaps):
        assert coa_gaps["match_reason"].notna().all()

    def test_empty_input_produces_an_empty_report_not_an_error(self, maps):
        empty = pd.DataFrame(
            columns=[C.gl.gl_account, C.gl.trans_type, C.gl.amount_entity,
                     STATUS_COL, "match_reason"]
        )
        spec = gaps.GapSpec(
            "coa", [C.gl.gl_account, C.gl.trans_type], C.gl.amount_entity
        )
        assert gaps.summarise_gaps(empty, spec).empty
