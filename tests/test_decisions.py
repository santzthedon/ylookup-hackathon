"""Tests for decision framing.

Two kinds here. Unit tests on synthetic frames check the framing rules. One
set checks something unusual for a test suite: that the language stays
readable. The audience is a non technical fund manager, so a decision that
leaks the word "crosswalk" has failed at its actual job, and a test is the
only thing that keeps that true as the code changes.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from recon import config, decisions as dec, load, mappings
from recon.lookup import REASON_COL, STATUS_COL, MatchStatus

C = config.COLS


# --------------------------------------------------------------------------
# Synthetic fixtures
# --------------------------------------------------------------------------


def _resolved_frame(rows: list[dict], status: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame[STATUS_COL] = status
    frame[REASON_COL] = "test"
    return frame


@pytest.fixture
def unmapped_account_frame() -> pd.DataFrame:
    return _resolved_frame(
        [
            {
                C.gl.legal_entity: "Alpha Fund LP",
                C.gl.gl_account: "40070 - Interest Income",
                C.gl.trans_type: "Expense: Administration Fees",
                C.gl.deal_name: "Alpha Deal",
                C.gl.position: "Alpha Shares",
                C.gl.investor: "Investor One",
                C.gl.amount_entity: 4000.00,
            },
            {
                C.gl.legal_entity: "Beta Fund LP",
                C.gl.gl_account: "40070 - Interest Income",
                C.gl.trans_type: "Expense: Administration Fees",
                C.gl.deal_name: "Beta Deal",
                C.gl.position: "Beta Shares",
                C.gl.investor: "Investor Two",
                C.gl.amount_entity: 867.16,
            },
        ],
        str(MatchStatus.NOT_IN_MAPPING),
    )


@pytest.fixture
def immaterial_account_frame() -> pd.DataFrame:
    return _resolved_frame(
        [
            {
                C.gl.legal_entity: "Alpha Fund LP",
                C.gl.gl_account: "30050 - Partner Transfers",
                C.gl.trans_type: "Expense: Bank Charges",
                C.gl.deal_name: "Alpha Deal",
                C.gl.position: "Alpha Shares",
                C.gl.investor: "Investor One",
                C.gl.amount_entity: 1e-12,
            }
        ],
        str(MatchStatus.NOT_IN_MAPPING),
    )


@pytest.fixture
def negative_amount_frame() -> pd.DataFrame:
    return _resolved_frame(
        [
            {
                C.gl.legal_entity: "Alpha Fund LP",
                C.gl.gl_account: "40070 - Interest Income",
                C.gl.trans_type: "Expense: Administration Fees",
                C.gl.deal_name: "Alpha Deal",
                C.gl.position: "Alpha Shares",
                C.gl.investor: "Investor One",
                C.gl.amount_entity: -221_592_500.00,
            }
        ],
        str(MatchStatus.NOT_IN_MAPPING),
    )


@pytest.fixture
def fake_dataset(unmapped_account_frame):
    """A Dataset stub carrying only what the account builder reads."""

    class Stub:
        corvus_coa = pd.DataFrame(
            {
                C.corvus_coa.gl_account: [
                    "50080 - Administration fees",
                    "10000 - Cash",
                    "50080 - Administration fees",
                ],
                C.corvus_coa.trans_type: ["a", "b", "c"],
                C.corvus_coa.account_type: ["Expense", "Assets", "Expense"],
            }
        )
        investor_mapping = pd.DataFrame(
            {
                C.investor.external_ref: ["2020_13450", "2020_13450"],
                C.investor.target_investor_name: ["Investor A", "Investor B"],
                C.investor.target_specific_id: [111, 222],
            }
        )

    return Stub()


# --------------------------------------------------------------------------
# Priority and materiality
# --------------------------------------------------------------------------


class TestPriority:
    def test_real_money_blocks(self):
        assert dec._priority_for(4867.16) == dec.Priority.BLOCKING

    def test_negative_amounts_judged_on_magnitude(self):
        assert dec._priority_for(-221_592_500.0) == dec.Priority.BLOCKING

    def test_residue_is_deferred(self):
        assert dec._priority_for(1e-12) == dec.Priority.DEFERRED

    def test_zero_is_deferred(self):
        assert dec._priority_for(0.0) == dec.Priority.DEFERRED

    def test_unknown_amount_blocks(self):
        """If we cannot value it, a human decides rather than the code."""
        assert dec._priority_for(None) == dec.Priority.BLOCKING


class TestPhrasingHelpers:
    def test_plural_singular(self):
        assert dec._plural(1, "transaction") == "1 transaction"

    def test_plural_many_has_thousands_separator(self):
        assert dec._plural(15530, "transaction") == "15,530 transactions"

    def test_plural_custom_form(self):
        assert dec._plural(2, "fund") == "2 funds"

    def test_money_formats_with_separators(self):
        assert dec._money(4867.16) == "4,867.16"

    def test_money_describes_residue_in_words(self):
        assert "zero" in dec._money(1e-12)

    def test_money_handles_missing(self):
        assert dec._money(None) == "an unknown amount"


# --------------------------------------------------------------------------
# Building decisions
# --------------------------------------------------------------------------


class TestUnmappedAccountDecisions:
    def test_one_decision_per_account_and_type(
        self, unmapped_account_frame, fake_dataset
    ):
        out = dec.unmapped_account_decisions(unmapped_account_frame, fake_dataset)
        assert len(out) == 1

    def test_amount_is_the_sum_of_affected_rows(
        self, unmapped_account_frame, fake_dataset
    ):
        out = dec.unmapped_account_decisions(unmapped_account_frame, fake_dataset)
        assert out[0].amount == pytest.approx(4867.16)

    def test_counts_rows_and_entities(self, unmapped_account_frame, fake_dataset):
        d = dec.unmapped_account_decisions(unmapped_account_frame, fake_dataset)[0]
        assert d.row_count == 2
        assert d.entities_affected == 2

    def test_options_are_deduplicated_target_accounts(
        self, unmapped_account_frame, fake_dataset
    ):
        d = dec.unmapped_account_decisions(unmapped_account_frame, fake_dataset)[0]
        assert d.options == ["10000 - Cash", "50080 - Administration fees"]

    def test_evidence_rows_are_attached(self, unmapped_account_frame, fake_dataset):
        d = dec.unmapped_account_decisions(unmapped_account_frame, fake_dataset)[0]
        assert len(d.evidence) == 2
        assert C.gl.legal_entity in d.evidence[0]

    def test_immaterial_gap_is_deferred(self, immaterial_account_frame, fake_dataset):
        d = dec.unmapped_account_decisions(immaterial_account_frame, fake_dataset)[0]
        assert d.priority == dec.Priority.DEFERRED

    def test_no_unmatched_rows_produces_no_decisions(self, fake_dataset):
        matched = _resolved_frame(
            [
                {
                    C.gl.legal_entity: "Alpha",
                    C.gl.gl_account: "x",
                    C.gl.trans_type: "y",
                    C.gl.amount_entity: 1.0,
                }
            ],
            str(MatchStatus.MATCHED),
        )
        assert dec.unmapped_account_decisions(matched, fake_dataset) == []

    def test_suggestion_is_empty_until_that_layer_exists(
        self, unmapped_account_frame, fake_dataset
    ):
        d = dec.unmapped_account_decisions(unmapped_account_frame, fake_dataset)[0]
        assert d.suggestion is None


class TestSignHandling:
    """Amounts are stored signed but must never read as signed to a fund manager."""

    def test_no_minus_sign_in_user_facing_text(
        self, negative_amount_frame, fake_dataset
    ):
        d = dec.unmapped_account_decisions(negative_amount_frame, fake_dataset)[0]
        assert d.amount == pytest.approx(-221_592_500.00)
        assert d.priority == dec.Priority.BLOCKING

        # The bug this guards: amount_display used to render the signed
        # value while question/detail (via _money()) already showed it
        # unsigned, so the same figure appeared with and without a minus
        # sign depending on where it was read.
        negative_form = "-221,592,500.00"
        positive_form = "221,592,500.00"
        for text in (d.question, d.detail, d.amount_display):
            assert negative_form not in text, text
        assert positive_form in d.question
        assert positive_form in d.amount_display


class TestAmbiguousInvestorDecisions:
    def test_ambiguous_reference_produces_a_decision(self, fake_dataset):
        frame = _resolved_frame(
            [
                {
                    C.gl.legal_entity: "Alpha Fund LP",
                    C.gl.rfx_id: "2020_13450",
                    C.gl.amount_entity: 500.0,
                    C.gl.investor: "Someone",
                }
            ],
            str(MatchStatus.AMBIGUOUS_MAPPING),
        )
        out = dec.ambiguous_investor_decisions(frame, fake_dataset)
        assert len(out) == 1
        assert out[0].priority == dec.Priority.BLOCKING

    def test_candidate_investors_become_the_options(self, fake_dataset):
        frame = _resolved_frame(
            [
                {
                    C.gl.legal_entity: "Alpha Fund LP",
                    C.gl.rfx_id: "2020_13450",
                    C.gl.amount_entity: 500.0,
                    C.gl.investor: "Someone",
                }
            ],
            str(MatchStatus.AMBIGUOUS_MAPPING),
        )
        d = dec.ambiguous_investor_decisions(frame, fake_dataset)[0]
        assert len(d.options) == 2
        assert "Investor A" in d.options[0]

    def test_rows_that_merely_failed_to_match_are_not_ambiguity(self, fake_dataset):
        """Only a self contradicting record raises this question."""
        frame = _resolved_frame(
            [
                {
                    C.gl.legal_entity: "Alpha",
                    C.gl.rfx_id: "9999_00000",
                    C.gl.amount_entity: 1.0,
                    C.gl.investor: "Someone",
                }
            ],
            str(MatchStatus.NOT_IN_MAPPING),
        )
        assert dec.ambiguous_investor_decisions(frame, fake_dataset) == []


class TestUnresolvedPositionDecisions:
    @pytest.fixture
    def position_frame(self):
        return _resolved_frame(
            [
                {
                    C.gl.legal_entity: "Alpha Fund LP",
                    C.gl.deal_name: "Alpha Deal",
                    C.gl.position: "Unknown Holding",
                    C.gl.amount_entity: 1_000_000.0,
                    C.gl.investor: "Investor One",
                }
            ],
            str(MatchStatus.NOT_IN_MAPPING),
        )

    def test_offers_three_ways_forward(self, position_frame):
        d = dec.unresolved_position_decisions(position_frame)[0]
        assert len(d.options) == 3

    def test_names_both_the_deal_and_the_holding(self, position_frame):
        d = dec.unresolved_position_decisions(position_frame)[0]
        assert "Alpha Deal" in d.detail
        assert "Unknown Holding" in d.detail


# --------------------------------------------------------------------------
# The language itself
# --------------------------------------------------------------------------

JARGON = [
    "crosswalk",
    "lookup",
    "join",
    "dataframe",
    "null",
    "nan",
    "key column",
    "mapping table",
    "normalis",
    "match_status",
    "pipeline",
]


class TestReadability:
    """A fund manager has to answer these without a glossary."""

    @pytest.fixture
    def all_decisions(self, unmapped_account_frame, fake_dataset):
        position = _resolved_frame(
            [
                {
                    C.gl.legal_entity: "Alpha Fund LP",
                    C.gl.deal_name: "Alpha Deal",
                    C.gl.position: "Unknown Holding",
                    C.gl.amount_entity: 1_000_000.0,
                    C.gl.investor: "Investor One",
                }
            ],
            str(MatchStatus.NOT_IN_MAPPING),
        )
        return dec.unmapped_account_decisions(
            unmapped_account_frame, fake_dataset
        ) + dec.unresolved_position_decisions(position)

    def test_no_engineering_jargon_in_user_facing_text(self, all_decisions):
        for d in all_decisions:
            text = f"{d.question} {d.detail} {d.why_it_matters}".lower()
            for word in JARGON:
                assert word not in text, f"{d.id} leaks jargon: {word}"

    def test_every_question_is_a_question(self, all_decisions):
        for d in all_decisions:
            assert d.question.strip().endswith("?"), d.id

    def test_every_decision_explains_the_consequence(self, all_decisions):
        for d in all_decisions:
            assert len(d.why_it_matters) > 30, d.id

    def test_every_decision_offers_a_way_to_answer(self, all_decisions):
        for d in all_decisions:
            assert d.answer_type
            assert d.options, d.id


# --------------------------------------------------------------------------
# The set
# --------------------------------------------------------------------------


class TestDecisionSet:
    @pytest.fixture
    def mixed_set(self, unmapped_account_frame, immaterial_account_frame, fake_dataset):
        blocking = dec.unmapped_account_decisions(unmapped_account_frame, fake_dataset)
        deferred = dec.unmapped_account_decisions(
            immaterial_account_frame, fake_dataset
        )
        combined = dec.DecisionSet(deferred + blocking)
        combined.decisions.sort(
            key=lambda d: (
                d.priority != dec.Priority.BLOCKING,
                -abs(d.amount) if d.amount is not None else 0,
            )
        )
        return combined

    def test_blocking_and_deferred_are_separated(self, mixed_set):
        assert len(mixed_set.blocking) == 1
        assert len(mixed_set.deferred) == 1

    def test_blocking_sorts_first(self, mixed_set):
        assert mixed_set.decisions[0].priority == dec.Priority.BLOCKING

    def test_summary_counts_only_blocking_value(self, mixed_set):
        summary = mixed_set.summary()
        assert summary["decisions_found"] == 2
        assert summary["blocking"] == 1
        assert summary["value_blocked"] == pytest.approx(4867.16)

    def test_json_round_trips(self, mixed_set, tmp_path):
        path = tmp_path / "decisions.json"
        mixed_set.to_json(path)
        payload = json.loads(path.read_text())
        assert payload["summary"]["blocking"] == 1
        assert len(payload["decisions"]) == 2

    def test_empty_set_is_safe(self):
        empty = dec.DecisionSet([])
        assert empty.summary()["decisions_found"] == 0
        assert empty.to_frame().empty


# --------------------------------------------------------------------------
# Against the real workbook
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_decisions():
    if not config.SOURCE_GL_PATH.exists():
        pytest.skip("dataset not present")
    data = load.load_dataset()
    maps = mappings.build_all(data)
    scoped = mappings.scope_source_gl(data)
    resolved = {
        "coa": maps.coa.apply(scoped, [C.gl.gl_account, C.gl.trans_type]),
        "investor": maps.investor.apply(scoped, [C.gl.rfx_id]),
        "position": maps.position.apply(scoped, [C.gl.deal_name, C.gl.position]),
    }
    return dec.build_decisions(data, resolved)


class TestAgainstRealData:
    def test_the_published_gap_appears_as_a_blocking_decision(self, real_decisions):
        """The 4,867.16 interest income gap must reach the fund manager."""
        blocking = real_decisions.blocking
        amounts = [abs(d.amount) for d in blocking if d.amount is not None]
        assert any(abs(a - 4867.16) < 0.01 for a in amounts)

    def test_blocking_list_is_short_enough_to_action(self, real_decisions):
        """A list nobody finishes is not a product. Ten is the outer limit."""
        assert 0 < len(real_decisions.blocking) <= 10

    def test_high_volume_zero_value_gaps_do_not_block(self, real_decisions):
        for d in real_decisions.blocking:
            assert d.materiality == "material", d.id

    def test_every_decision_has_evidence(self, real_decisions):
        for d in real_decisions.decisions:
            assert d.evidence, d.id

    def test_decisions_are_serialisable(self, real_decisions, tmp_path):
        real_decisions.to_json(tmp_path / "d.json")
        assert (tmp_path / "d.json").stat().st_size > 0
