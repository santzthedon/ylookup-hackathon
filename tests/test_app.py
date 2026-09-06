"""Tests for the FastAPI front end in src/web/app.py.

Nothing here touches recon/**; the pipeline itself is exercised the same way
scripts/step3_decisions.py and tests/test_mappings.py already do. These guard
bugs found by two independent audits of the running web layer:

1. The evidence table was stripping the sign off real ledger amounts (reusing
   the "always unsigned" convention meant only for headline totals), so a
   reviewer comparing a transaction to the source file would see the wrong
   number.
2. "Back to results" and the final "Save & next decision" both linked to the
   upload form (`/`), not to any page showing the decisions -- leaving a
   decision page had no way back to the list except the browser's own Back
   button.
3. A corrupt or unreadable uploaded workbook could bubble up as a bare 500
   with no explanation (an .xlsx is a zip container, and pandas/openpyxl
   raise zipfile.BadZipFile for a truncated one -- not ValueError/KeyError/
   FileNotFoundError, so it escaped the app's own error handling entirely),
   and a request FastAPI itself rejects (a missing file field) rendered raw
   JSON instead of the app's own error page.
4. An unbounded upload had no size limit, reading the whole file into memory
   before any validation ran.
5. A script-block JSON-escaping gap: `tojson` is deliberately Markup-wrapped
   so Jinja's autoescape leaves it alone inside <script> (entities are never
   decoded there), which means it has to do its own escaping. Workbook
   content (an account or investor name) containing "</script>" could close
   the surrounding tag early and let injected markup/script run.
6. `money`/`signed_money` formatted a NaN evidence value (a genuinely missing
   amount, or a blank date that became NaT then NaN) as the literal text
   "nan" instead of "n/a".
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from recon import config
from recon.decisions import AnswerType, Decision, DecisionSet, Priority
from web import app as web_app

C = config.COLS


@pytest.fixture()
def client() -> TestClient:
    return TestClient(web_app.app)


@pytest.fixture(autouse=True)
def _reset_last_scan():
    """`_last_scan` is a module-level global; each test starts and ends clean,
    regardless of what an earlier test in this module (or run order) left
    behind."""
    web_app._last_scan = None
    yield
    web_app._last_scan = None


def _xlsx_bytes() -> bytes:
    """A minimal, genuinely valid .xlsx file, built without needing a sheet
    with real pipeline columns — good enough to prove a *well-formed* upload
    is not what the corruption test below is about."""
    import openpyxl

    buf = io.BytesIO()
    wb = openpyxl.Workbook()
    wb.active.title = "Sheet1"
    wb.active["A1"] = "hello"
    wb.save(buf)
    return buf.getvalue()


def _sample_decision_set() -> DecisionSet:
    d = Decision(
        id="account-01",
        kind="unmapped_account",
        priority=Priority.BLOCKING,
        question="Where should 1 transaction post in the new system?",
        detail="Test decision.",
        why_it_matters="Test.",
        amount=100.0,
        row_count=1,
        entities_affected=1,
        materiality="material",
        answer_type=AnswerType.CHOOSE_ACCOUNT,
        options=["10000 - Cash"],
        evidence=[{"Amount": -50.0}],
    )
    return DecisionSet([d])


# --------------------------------------------------------------------------
# Basic routing
# --------------------------------------------------------------------------


def test_upload_page_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Upload workbooks" in resp.text


def test_decision_before_any_scan_is_a_styled_404(client):
    resp = client.get("/decision/account-01")
    assert resp.status_code == 404
    # Rendered through error.html, not FastAPI's default {"detail": ...} JSON.
    assert "application/json" not in resp.headers["content-type"]
    assert "No scan has been run yet" in resp.text


# --------------------------------------------------------------------------
# /results: both navigation dead ends now land on it, not on the upload form.
# --------------------------------------------------------------------------


def test_results_without_a_scan_is_a_styled_404_not_a_blank_page(client):
    resp = client.get("/results")
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("text/html")
    assert "No scan has been run yet" in resp.text


def test_results_with_a_scan_shows_the_decision(client):
    web_app._last_scan = _sample_decision_set()
    resp = client.get("/results")
    assert resp.status_code == 200
    assert "Where should 1 transaction post in the new system?" in resp.text


def test_upload_page_offers_a_way_back_to_an_in_progress_scan(client):
    web_app._last_scan = _sample_decision_set()
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Continue reviewing" in resp.text
    assert 'href="/results"' in resp.text


def test_upload_page_has_no_stale_continue_link_without_a_scan(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Continue reviewing" not in resp.text


# --------------------------------------------------------------------------
# Malformed / rejected upload handling — never a bare 500 or raw JSON.
# --------------------------------------------------------------------------


def test_scan_missing_a_file_field_is_a_styled_error_not_raw_json(client):
    resp = client.post(
        "/scan",
        files={"source_gl": ("a.xlsx", b"placeholder", "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert resp.headers["content-type"].startswith("text/html")
    assert "Both workbooks are required" in resp.text


def test_corrupt_workbook_upload_is_a_styled_400_not_an_unhandled_500(client):
    # Real bytes that were once a valid xlsx, cut in half: same failure mode
    # as an interrupted upload. Deliberately not just garbage bytes, since a
    # totally-unrecognisable file already raised ValueError before this fix
    # (pandas: "Excel file format cannot be determined") -- the gap this
    # test targets is specifically the zip-container-shaped corruption,
    # which raises zipfile.BadZipFile instead.
    good = _xlsx_bytes()
    truncated = good[: len(good) // 2]

    resp = client.post(
        "/scan",
        files={
            "source_gl": ("source.xlsx", truncated, "application/octet-stream"),
            "loader": ("loader.xlsx", truncated, "application/octet-stream"),
        },
    )

    assert resp.status_code == 400
    assert "application/json" not in resp.headers["content-type"]
    assert "valid .xlsx workbook" in resp.text


def test_scan_with_a_plain_text_file_gives_a_friendly_message(client):
    text = b"this is not an excel file at all, just plain text"
    resp = client.post(
        "/scan",
        files={
            "source_gl": ("source.xlsx", text, "application/octet-stream"),
            "loader": ("loader.xlsx", text, "application/octet-stream"),
        },
    )
    assert resp.status_code == 400
    assert resp.headers["content-type"].startswith("text/html")
    assert "valid .xlsx workbook" in resp.text
    # The raw pandas/openpyxl message must not leak through to the user.
    assert "engine manually" not in resp.text


def test_oversized_upload_is_rejected_before_being_fully_buffered(client, monkeypatch):
    monkeypatch.setattr(web_app, "MAX_UPLOAD_BYTES", 10)
    monkeypatch.setattr(web_app, "_UPLOAD_CHUNK_SIZE", 4)

    resp = client.post(
        "/scan",
        files={
            "source_gl": ("source.xlsx", b"x" * 100, "application/octet-stream"),
            "loader": ("loader.xlsx", b"y" * 10, "application/octet-stream"),
        },
    )

    assert resp.status_code == 413
    assert "source_gl" in resp.text
    assert "limit" in resp.text


# --------------------------------------------------------------------------
# money / signed_money: sign convention and NaN safety.
#
# money is for figures that are deliberately unsigned by convention
# (headline totals, Decision.amount_display). signed_money is for individual
# evidence rows, which are real ledger entries where the sign is part of the
# transaction. Both must render a NaN value (a genuinely missing amount) as
# "n/a", not the literal text "nan".
# --------------------------------------------------------------------------


def test_money_filter_strips_sign_for_headline_totals():
    money = web_app.templates.env.filters["money"]
    assert money(-1234.5) == "1,234.50"
    assert money(1234.5) == "1,234.50"


def test_signed_money_filter_preserves_sign_for_evidence_rows():
    signed_money = web_app.templates.env.filters["signed_money"]
    assert signed_money(-1234.5) == "-1,234.50"
    assert signed_money(1234.5) == "1,234.50"


def test_money_filter_renders_nan_as_na_not_the_literal_string_nan():
    assert web_app._money(float("nan")) == "n/a"


def test_signed_money_filter_renders_nan_as_na_not_the_literal_string_nan():
    assert web_app._signed_money(float("nan")) == "n/a"


def test_money_filter_unaffected_for_real_numbers():
    assert web_app._money(1234.5) == "1,234.50"
    assert web_app._money(-42) == "42.00"
    assert web_app._money(0) == "0.00"
    assert web_app._money(None) == "0.00"


# --------------------------------------------------------------------------
# tojson: safe embedding of workbook-derived strings inside a <script> block.
#
# json_attr (used for HTML-attribute contexts) is autoescaped by Jinja, so an
# apostrophe or quote in an account name can't break out of the surrounding
# attribute. tojson is Markup-wrapped so autoescape leaves it alone inside
# <script> -- which means it has to do its own escaping, or an
# account/investor/deal name containing the literal substring "</script>"
# closes the surrounding script tag early and whatever follows in the page
# runs as markup/script. json.dumps alone does not do this escaping.
# --------------------------------------------------------------------------


def test_tojson_escapes_script_closing_sequence():
    value = "Partners' capital: </script><script>alert(1)</script>"
    rendered = str(web_app._tojson(value))

    assert "</script>" not in rendered
    assert "<script>" not in rendered
    # Must still be the original string once actually parsed as JS/JSON.
    import json

    assert json.loads(rendered) == value


def test_tojson_escapes_html_significant_characters():
    value = "Tom & Jerry <b>bold</b>"
    rendered = str(web_app._tojson(value))
    assert "<" not in rendered
    assert ">" not in rendered
    assert "&" not in rendered

    import json

    assert json.loads(rendered) == value


def test_tojson_escapes_js_line_terminators():
    # U+2028/U+2029 are valid JSON string characters but are treated as line
    # terminators by a JS parser reading `var x = "...";` directly (not
    # through JSON.parse), which is exactly how this filter's output is used
    # in decision.html. Left unescaped, an account name containing one would
    # break the surrounding script with a syntax error.
    value = "line one line two line three"
    rendered = str(web_app._tojson(value))
    assert " " not in rendered
    assert " " not in rendered

    import json

    assert json.loads(rendered) == value


def test_tojson_still_produces_valid_json_for_plain_values():
    import json

    assert json.loads(str(web_app._tojson("blocking"))) == "blocking"
    assert json.loads(str(web_app._tojson(None))) is None
    assert json.loads(str(web_app._tojson(42))) == 42


# --------------------------------------------------------------------------
# End to end against the real dataset, if available.
#
# Exercises the same /scan path a real upload takes, including running the
# pipeline via run_in_threadpool (moved off the event loop as part of this
# audit) -- proves that change didn't alter behaviour.
# --------------------------------------------------------------------------


@pytest.fixture()
def real_workbooks():
    if not config.SOURCE_GL_PATH.exists() or not config.LOADER_PATH.exists():
        pytest.skip(f"dataset not present at {config.DATA_DIR}")
    return (
        config.SOURCE_GL_PATH.read_bytes(),
        config.LOADER_PATH.read_bytes(),
    )


def test_scan_end_to_end_with_real_dataset(client, real_workbooks):
    source_bytes, loader_bytes = real_workbooks

    resp = client.post(
        "/scan",
        files={
            "source_gl": (
                "source.xlsx",
                source_bytes,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
            "loader": (
                "loader.xlsx",
                loader_bytes,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        },
    )

    assert resp.status_code == 200
    assert "Scan results" in resp.text
    assert web_app._last_scan is not None
    assert len(web_app._last_scan.decisions) > 0

    # And the decision detail page for the first blocking decision resolves,
    # proving /decision/{id} still reads from the freshly-computed scan.
    first_id = web_app._last_scan.decisions[0].id
    detail_resp = client.get(f"/decision/{first_id}")
    assert detail_resp.status_code == 200

    # /results re-renders the same scan without re-running the pipeline.
    results_resp = client.get("/results")
    assert results_resp.status_code == 200
    assert "Scan results" in results_resp.text
