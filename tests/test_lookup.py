"""Unit tests for normalisation and the generic Lookup.

These use small synthetic frames rather than the real workbook, so they run in
milliseconds and each failure points at exactly one behaviour.
"""

from __future__ import annotations

import pandas as pd
import pytest

from recon import normalise
from recon.lookup import REASON_COL, STATUS_COL, Lookup, MatchStatus


# --------------------------------------------------------------------------
# normalise
# --------------------------------------------------------------------------


class TestNormaliseText:
    def test_trims_surrounding_whitespace(self):
        assert normalise.normalise_text("  Chalbury  ") == "chalbury"

    def test_collapses_internal_whitespace(self):
        assert normalise.normalise_text("Chalbury   HoldCo") == "chalbury holdco"

    def test_casefolds(self):
        assert normalise.normalise_text("CHALBURY") == normalise.normalise_text(
            "chalbury"
        )

    def test_handles_non_breaking_space(self):
        assert normalise.normalise_text("Chalbury\u00a0HoldCo") == "chalbury holdco"

    def test_handles_tabs_and_newlines(self):
        assert normalise.normalise_text("Chalbury\tHoldCo\n") == "chalbury holdco"

    @pytest.mark.parametrize("blank", [None, "", "   ", "\t", float("nan"), pd.NA])
    def test_blank_values_return_none(self, blank):
        assert normalise.normalise_text(blank) is None

    def test_preserves_punctuation(self):
        """L.P. and LP must stay different keys. Merging them is a human decision."""
        assert normalise.normalise_text("Chalbury L.P.") != normalise.normalise_text(
            "Chalbury LP"
        )

    def test_preserves_legal_suffix(self):
        """A fund and its general partner often differ only by a suffix."""
        assert normalise.normalise_text("Ellwold GP LLC") != normalise.normalise_text(
            "Ellwold LLC"
        )

    def test_numbers_are_stringified_consistently(self):
        assert normalise.normalise_text(7144) == "7144"


class TestCompositeKey:
    def test_joins_components(self):
        frame = pd.DataFrame({"a": ["Deal One"], "b": ["Position A"]})
        key = normalise.build_composite_key(frame, ["a", "b"])
        assert key.iloc[0] == f"deal one{normalise.KEY_SEPARATOR}position a"

    def test_blank_component_blanks_the_whole_key(self):
        """A partial key would match the wrong record, so it must not be built."""
        frame = pd.DataFrame({"a": ["Deal One"], "b": [None]})
        key = normalise.build_composite_key(frame, ["a", "b"])
        assert pd.isna(key.iloc[0])

    def test_requires_at_least_one_column(self):
        with pytest.raises(ValueError):
            normalise.build_composite_key(pd.DataFrame({"a": [1]}), [])


# --------------------------------------------------------------------------
# Lookup
# --------------------------------------------------------------------------


@pytest.fixture
def simple_mapping() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "src": ["Alpha Fund", "Beta Fund ", "GAMMA FUND"],
            "target": ["Alpha Ltd", "Beta Ltd", "Gamma Ltd"],
            "target_id": [1, 2, 3],
        }
    )


@pytest.fixture
def simple_lookup(simple_mapping) -> Lookup:
    return Lookup(
        name="test",
        mapping=simple_mapping,
        key_columns=["src"],
        value_columns=["target", "target_id"],
    )


class TestLookupConstruction:
    def test_rejects_missing_key_column(self, simple_mapping):
        with pytest.raises(KeyError, match="missing column"):
            Lookup("test", simple_mapping, ["nope"], ["target"])

    def test_rejects_missing_value_column(self, simple_mapping):
        with pytest.raises(KeyError, match="missing column"):
            Lookup("test", simple_mapping, ["src"], ["nope"])

    def test_indexes_every_usable_key(self, simple_lookup):
        assert len(simple_lookup) == 3

    def test_blank_keys_are_excluded_and_counted(self):
        mapping = pd.DataFrame(
            {"src": ["Alpha", None, "  "], "target": ["A", "B", "C"]}
        )
        lookup = Lookup("test", mapping, ["src"], ["target"])
        assert len(lookup) == 1
        assert lookup.build_report.blank_key_rows == 2

    def test_duplicate_rows_with_identical_values_collapse(self):
        mapping = pd.DataFrame(
            {"src": ["Alpha", "alpha", " ALPHA "], "target": ["A", "A", "A"]}
        )
        lookup = Lookup("test", mapping, ["src"], ["target"])
        assert len(lookup) == 1
        assert lookup.build_report.duplicate_consistent_keys == 1
        assert lookup.get("Alpha").matched

    def test_duplicate_rows_with_conflicting_values_are_ambiguous(self):
        mapping = pd.DataFrame({"src": ["Alpha", "Alpha"], "target": ["A", "B"]})
        lookup = Lookup("test", mapping, ["src"], ["target"])
        assert lookup.build_report.ambiguous_keys == ["alpha"]
        assert lookup.get("Alpha").status is MatchStatus.AMBIGUOUS_MAPPING


class TestLookupGet:
    def test_exact_match(self, simple_lookup):
        result = simple_lookup.get("Alpha Fund")
        assert result.matched
        assert result.values["target"] == "Alpha Ltd"
        assert result.values["target_id"] == 1

    def test_match_is_case_insensitive(self, simple_lookup):
        assert simple_lookup.get("alpha fund").matched

    def test_match_ignores_surrounding_whitespace(self, simple_lookup):
        assert simple_lookup.get("  Beta Fund  ").matched

    def test_absent_key_reports_not_in_mapping(self, simple_lookup):
        result = simple_lookup.get("Delta Fund")
        assert result.status is MatchStatus.NOT_IN_MAPPING
        assert not result.matched

    def test_blank_key_reports_no_key(self, simple_lookup):
        assert simple_lookup.get(None).status is MatchStatus.NO_KEY
        assert simple_lookup.get("   ").status is MatchStatus.NO_KEY

    def test_no_fuzzy_matching(self, simple_lookup):
        """A near miss must miss. Guessing corrupts totals silently."""
        assert not simple_lookup.get("Alpha Fnd").matched
        assert not simple_lookup.get("Alpha").matched
        assert not simple_lookup.get("Alpha Fund LP").matched

    def test_wrong_number_of_key_parts_raises(self, simple_lookup):
        with pytest.raises(ValueError, match="expects 1 key"):
            simple_lookup.get("Alpha Fund", "extra")

    def test_every_result_carries_a_reason(self, simple_lookup):
        for key in ["Alpha Fund", "Delta Fund", None]:
            assert simple_lookup.get(key).reason


class TestLookupApply:
    def test_no_rows_are_dropped(self, simple_lookup):
        frame = pd.DataFrame({"src": ["Alpha Fund", "Delta Fund", None, "beta fund"]})
        out = simple_lookup.apply(frame, ["src"])
        assert len(out) == len(frame)

    def test_status_per_row(self, simple_lookup):
        frame = pd.DataFrame({"src": ["Alpha Fund", "Delta Fund", None]})
        out = simple_lookup.apply(frame, ["src"])
        assert list(out[STATUS_COL]) == [
            str(MatchStatus.MATCHED),
            str(MatchStatus.NOT_IN_MAPPING),
            str(MatchStatus.NO_KEY),
        ]

    def test_unmatched_rows_have_empty_target_values(self, simple_lookup):
        frame = pd.DataFrame({"src": ["Delta Fund"]})
        out = simple_lookup.apply(frame, ["src"])
        assert pd.isna(out["test_target"].iloc[0])

    def test_matched_rows_carry_target_values(self, simple_lookup):
        frame = pd.DataFrame({"src": ["GAMMA fund"]})
        out = simple_lookup.apply(frame, ["src"])
        assert out["test_target"].iloc[0] == "Gamma Ltd"

    def test_reason_column_is_populated(self, simple_lookup):
        frame = pd.DataFrame({"src": ["Alpha Fund", "Delta Fund"]})
        out = simple_lookup.apply(frame, ["src"])
        assert out[REASON_COL].notna().all()

    def test_ambiguous_rows_get_no_values(self):
        mapping = pd.DataFrame({"src": ["Alpha", "Alpha"], "target": ["A", "B"]})
        lookup = Lookup("test", mapping, ["src"], ["target"])
        out = lookup.apply(pd.DataFrame({"src": ["Alpha"]}), ["src"])
        assert out[STATUS_COL].iloc[0] == str(MatchStatus.AMBIGUOUS_MAPPING)
        assert pd.isna(out["test_target"].iloc[0])

    def test_composite_key_apply(self):
        mapping = pd.DataFrame(
            {
                "deal": ["Deal One", "Deal One"],
                "pos": ["Shares", "Loan"],
                "pos_id": [10, 20],
            }
        )
        lookup = Lookup("pos", mapping, ["deal", "pos"], ["pos_id"])
        frame = pd.DataFrame(
            {"deal": ["Deal One", "Deal One"], "pos": ["Loan", "Warrant"]}
        )
        out = lookup.apply(frame, ["deal", "pos"])
        assert out["pos_pos_id"].iloc[0] == 20
        assert pd.isna(out["pos_pos_id"].iloc[1])

    def test_mismatched_key_column_count_raises(self, simple_lookup):
        frame = pd.DataFrame({"a": ["x"], "b": ["y"]})
        with pytest.raises(ValueError, match="expects 1 key column"):
            simple_lookup.apply(frame, ["a", "b"])

    def test_missing_source_column_raises(self, simple_lookup):
        with pytest.raises(KeyError, match="missing key column"):
            simple_lookup.apply(pd.DataFrame({"other": ["x"]}), ["src"])

    def test_match_rate(self, simple_lookup):
        frame = pd.DataFrame({"src": ["Alpha Fund", "Delta Fund"]})
        out = simple_lookup.apply(frame, ["src"])
        assert simple_lookup.match_rate(out) == 0.5

    def test_unmatched_helper(self, simple_lookup):
        frame = pd.DataFrame({"src": ["Alpha Fund", "Delta Fund", None]})
        out = simple_lookup.apply(frame, ["src"])
        assert len(simple_lookup.unmatched(out)) == 2
