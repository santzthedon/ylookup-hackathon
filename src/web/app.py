"""FastAPI front end for the migration decision review pipeline.

Runs the same pipeline as scripts/step3_decisions.py — load, mappings,
build_decisions — directly against the two uploaded workbooks. This never
reads out/step3_decisions.json: the decision set shown here is always freshly
computed in process from whatever files were uploaded in the current scan.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from recon import config, decisions as dec, load, mappings
from recon.decisions import DecisionSet

C = config.COLS
log = logging.getLogger(__name__)

app = FastAPI(title="Migration decision review")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _money(value: object) -> str:
    """Unsigned, two decimals, comma thousands separators — for figures that
    are deliberately unsigned by convention: headline totals and
    Decision.amount_display. Never for a raw ledger row, where the sign is
    part of the transaction (see signed_money below).

    A blank evidence cell (a genuinely missing amount, or a blank date that
    became NaT and then NaN once formatted) can reach a money filter as a
    Python float NaN, not None. Jinja's `is number` test is true for NaN, so
    the evidence table always routes it through a money filter rather than
    the else-branch that would print it as-is. Without this guard,
    `float(nan or 0)` still evaluates to nan and `f"{nan:,.2f}"` formats to
    the literal text "nan" -- illegible to a non-technical fund manager.
    Decision.amount_display already guards the same case for the headline
    amount; this keeps every other figure consistent with it.
    """
    try:
        amount = float(value or 0)
    except (TypeError, ValueError):
        return "n/a"
    if amount != amount:  # NaN != NaN is the portable, import-free NaN check.
        return "n/a"
    return f"{abs(amount):,.2f}"


def _signed_money(value: object) -> str:
    """Same convention as _money, but keeps the sign.

    Individual evidence rows are real ledger entries: debits and credits
    carry opposite signs, and a reviewer checking evidence against the
    source workbook needs to see the same number that is actually there --
    stripping the sign there would silently show the wrong figure. Headline
    totals and Decision.amount_display are deliberately unsigned by
    convention and must keep using _money instead.
    """
    try:
        amount = float(value or 0)
    except (TypeError, ValueError):
        return "n/a"
    if amount != amount:
        return "n/a"
    return f"{amount:,.2f}"


templates.env.filters["money"] = _money
templates.env.filters["signed_money"] = _signed_money
templates.env.filters["count"] = lambda value: f"{int(value):,}"
# Plain Jinja2 (unlike Flask) has no tojson filter. Two variants, because a
# JSON literal needs opposite escaping depending on where it lands:
#
# - Inside a <script> block, entities are never decoded (it's raw text, not
#   HTML), so escaping a quote there is a silent syntax error. `tojson` is
#   Markup-wrapped, like Flask's own tojson, so autoescape leaves it alone.
# - Inside an HTML attribute (e.g. a data-* attribute holding JSON), normal
#   escaping is exactly what's needed: an apostrophe in the data must not be
#   allowed to close a single-quoted attribute early. `json_attr` is a plain
#   string, so autoescape still applies to it.


def _tojson(value: object) -> Markup:
    """JSON literal safe to inline inside a <script> block.

    json.dumps on its own is not enough here: it happily emits a literal
    "</script>" if a string value (an account name, an investor name, ...
    anything sourced from an uploaded workbook) contains that substring,
    which closes the surrounding <script> tag early and lets whatever
    follows run as markup/script — the same class of escaping bug as the
    single-quoted `data-options` attribute this filter was split off from,
    just landing in a different context. U+2028/U+2029 are valid in a JSON
    string but are line terminators to a JS parser, so they are escaped too,
    or an account name containing one would break the script with a syntax
    error rather than an injection.
    """
    encoded = json.dumps(value)
    encoded = (
        encoded.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
    )
    return Markup(encoded)


templates.env.filters["tojson"] = _tojson
templates.env.filters["json_attr"] = json.dumps


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> HTMLResponse:
    """Render errors as a page in the app's own design, not FastAPI's raw JSON.

    A fund manager who mis-uploads a file or follows a stale link should see
    something legible, not a {"detail": "..."} blob.
    """
    return templates.TemplateResponse(
        request,
        "error.html",
        {"status_code": exc.status_code, "detail": exc.detail},
        status_code=exc.status_code,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> HTMLResponse:
    """Same styled error page for a request FastAPI itself rejects.

    Without this, submitting the scan form with a file field missing (for
    example, a browser that ignored the `required` attribute) would show
    FastAPI's raw `{"detail": [...]}` JSON instead of the app's own error
    page -- exactly the illegible blob the HTTPException handler above exists
    to avoid.
    """
    return templates.TemplateResponse(
        request,
        "error.html",
        {
            "status_code": 400,
            "detail": "Both workbooks are required. Choose a file for each before scanning.",
        },
        status_code=400,
    )


def _unreadable_workbook_message(exc: Exception) -> str:
    """Translate a low-level file-format error into something a fund manager
    who has never seen a stack trace can act on.

    pandas/openpyxl raise several different exception shapes for "this is not
    a real Excel file" (a bad zip signature, an undetectable engine, and so
    on). Anything recognisable as that class of problem gets one plain
    message; anything else (a missing sheet, a missing column) already comes
    with a specific, actionable message from recon.load and is passed through
    unchanged.
    """
    if isinstance(exc, zipfile.BadZipFile):
        return (
            "That file doesn't look like a valid .xlsx workbook (it may be "
            "corrupted or an incomplete upload). Please re-export it and "
            "try again."
        )
    text = str(exc)
    lower = text.lower()
    if "engine" in lower or "zip" in lower:
        return (
            "That file doesn't look like a valid .xlsx workbook. Please "
            "upload the original Excel file, not a renamed or converted copy."
        )
    return text


# Single in-memory scan result. This is a one-operator review tool, not a
# multi-user service, so there is deliberately no session or database layer:
# /decision/{id} and /results always look at the most recently run scan.
_last_scan: DecisionSet | None = None

# Generous but bounded: real workbooks here run to tens of thousands of rows
# and a few MB, so this only exists to stop an accidental (or hostile) upload
# from being read fully into memory with no limit at all.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB
_UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB


async def _save_upload(upload: UploadFile, dest: Path, field_name: str) -> None:
    """Stream an uploaded file to disk without ever holding the whole thing
    in memory at once, and refuse anything past MAX_UPLOAD_BYTES.

    `UploadFile.read()` with no size returns the entire file as one `bytes`
    object regardless of how large it is, so a single huge (or hostile)
    upload could exhaust memory before validation ever runs. Reading in
    bounded chunks and stopping early avoids that.
    """
    size = 0
    with dest.open("wb") as out:
        while chunk := await upload.read(_UPLOAD_CHUNK_SIZE):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"'{field_name}' is larger than the "
                        f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
                    ),
                )
            out.write(chunk)


def _run_pipeline(source_path: Path, loader_path: Path) -> DecisionSet:
    """The same three steps scripts/step3_decisions.py runs, on given paths."""
    data = load.load_dataset(
        use_cache=False, source_path=source_path, loader_path=loader_path
    )
    maps = mappings.build_all(data)
    scoped = mappings.scope_source_gl(data)
    resolved = {
        "coa": maps.coa.apply(scoped, [C.gl.gl_account, C.gl.trans_type]),
        "investor": maps.investor.apply(scoped, [C.gl.rfx_id]),
        "position": maps.position.apply(scoped, [C.gl.deal_name, C.gl.position]),
    }
    return dec.build_decisions(data, resolved)


def _results_context(decision_set: DecisionSet) -> dict:
    """Template context shared by the just-scanned response and /results."""
    return {
        "summary": decision_set.summary(),
        "blocking": decision_set.blocking,
        "deferred": decision_set.deferred,
    }


@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request) -> HTMLResponse:
    # A scan already sitting in memory means someone left mid-review (closed
    # the tab, came back later) rather than wanting to start over. Surfacing
    # a way back to it here is the difference between "revisit your
    # decisions" and "re-upload both workbooks and wait another ten seconds
    # to get back where you were" -- exactly the kind of extra round trip
    # this tool exists to remove.
    return templates.TemplateResponse(
        request, "upload.html", {"has_previous_scan": _last_scan is not None}
    )


@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request) -> HTMLResponse:
    """Re-render the most recent scan's results, without re-scanning.

    This is what "Back to results" (from a decision page) and the final
    "Save & next decision" on the last decision in a group link to, so
    leaving a decision page never dead-ends back at a blank upload form.
    """
    if _last_scan is None:
        raise HTTPException(
            status_code=404,
            detail="No scan has been run yet. Upload the two workbooks to start one.",
        )
    return templates.TemplateResponse(
        request, "results.html", _results_context(_last_scan)
    )


@app.post("/scan", response_class=HTMLResponse)
async def scan(
    request: Request,
    source_gl: UploadFile = File(...),
    loader: UploadFile = File(...),
) -> HTMLResponse:
    global _last_scan

    tmp_dir = Path(tempfile.mkdtemp(prefix="ylookup_scan_"))
    try:
        # Fixed names, not the uploaded filenames: read_sheet only cares
        # about sheet names inside the workbook, and this avoids trusting
        # user supplied filenames for a filesystem path.
        source_path = tmp_dir / "source.xlsx"
        loader_path = tmp_dir / "loader.xlsx"
        await _save_upload(source_gl, source_path, "source_gl")
        await _save_upload(loader, loader_path, "loader")

        try:
            # Parsing two real workbooks and running the full reconciliation
            # is a CPU/IO-bound, synchronous pandas call that can take
            # seconds on a large ledger. Running it directly in this async
            # route would block the single event loop for that whole time,
            # freezing every other request (including the static asset
            # requests for this same page). Offload it to a worker thread.
            decision_set = await run_in_threadpool(
                _run_pipeline, source_path, loader_path
            )
        except (FileNotFoundError, KeyError, ValueError, zipfile.BadZipFile) as exc:
            raise HTTPException(
                status_code=400, detail=_unreadable_workbook_message(exc)
            ) from exc
        except Exception as exc:  # noqa: BLE001 - last-resort safety net, see below
            # Anything else is an unanticipated way to fail to parse a
            # user-supplied file (openpyxl and pandas raise a wide variety of
            # exception types for corrupt or unexpected workbook content, and
            # this endpoint's whole job is handling files it does not
            # control). The alternative is a bare 500 with no explanation --
            # exactly the illegible failure the app's own error page exists
            # to avoid. The full traceback still goes to the server log for
            # debugging; only the user-facing message is generic.
            log.exception("Unhandled error scanning uploaded workbooks")
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not read one of the uploaded workbooks. Double-check "
                    "both files are the correct, unmodified .xlsx exports and "
                    "try again."
                ),
            ) from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    _last_scan = decision_set

    return templates.TemplateResponse(
        request, "results.html", _results_context(decision_set)
    )


@app.get("/decision/{decision_id}", response_class=HTMLResponse)
async def decision_detail(request: Request, decision_id: str) -> HTMLResponse:
    if _last_scan is None:
        raise HTTPException(status_code=404, detail="No scan has been run yet")

    match = next((d for d in _last_scan.decisions if d.id == decision_id), None)
    if match is None:
        raise HTTPException(
            status_code=404, detail=f"No decision '{decision_id}' in the current scan"
        )

    # Prev/next stay within the same priority group: walking the 5 blocking
    # decisions in order is the actual workflow; jumping into deferred mid
    # sequence would be a surprising, not a helpful, shortcut.
    group = (
        _last_scan.blocking if match.priority == dec.Priority.BLOCKING else _last_scan.deferred
    )
    position = group.index(match) + 1
    prev_id = group[position - 2].id if position > 1 else None
    next_id = group[position].id if position < len(group) else None

    return templates.TemplateResponse(
        request,
        "decision.html",
        {
            "d": match,
            "position": position,
            "group_total": len(group),
            "prev_id": prev_id,
            "next_id": next_id,
        },
    )
