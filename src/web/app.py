"""FastAPI front end for the migration decision review pipeline.

Runs the same pipeline as scripts/step3_decisions.py — load, mappings,
build_decisions — directly against the two uploaded workbooks. This never
reads out/step3_decisions.json: the decision set shown here is always freshly
computed in process from whatever files were uploaded in the current scan.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from recon import config, decisions as dec, load, mappings
from recon.decisions import DecisionSet

C = config.COLS

app = FastAPI(title="Migration decision review")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
# Same convention as Decision.amount_display: unsigned, two decimals, comma
# thousands separators. Kept as filters so templates never format a number
# themselves.
templates.env.filters["money"] = lambda value: f"{abs(float(value or 0)):,.2f}"
templates.env.filters["count"] = lambda value: f"{int(value):,}"
# Plain Jinja2 (unlike Flask) has no tojson filter; needed to hand the
# account option list to the searchable combobox's JS as a JSON literal.
templates.env.filters["tojson"] = json.dumps

# Single in-memory scan result. This is a one-operator review tool, not a
# multi-user service, so there is deliberately no session or database layer:
# /decision/{id} always looks the decision up in the most recently run scan.
_last_scan: DecisionSet | None = None


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


@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "upload.html", {})


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
        source_path.write_bytes(await source_gl.read())
        loader_path.write_bytes(await loader.read())

        try:
            decision_set = _run_pipeline(source_path, loader_path)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    _last_scan = decision_set

    return templates.TemplateResponse(
        request,
        "results.html",
        {
            "summary": decision_set.summary(),
            "blocking": decision_set.blocking,
            "deferred": decision_set.deferred,
        },
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

    return templates.TemplateResponse(request, "decision.html", {"d": match})
