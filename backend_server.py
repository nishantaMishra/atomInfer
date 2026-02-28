#!/usr/bin/env python3
"""
AtomInfer WebUI Backend
=======================
FastAPI + WebSocket streaming server.
The frontend connects via WebSocket, sends a job spec,
and receives streaming JSON events as the agent runs.

Run:
  pip install fastapi uvicorn python-multipart
  export GROQ_API_KEY=...
  export MP_API_KEY=...
  uvicorn backend_server:app --reload --host 0.0.0.0 --port 8000
"""

import os, json, asyncio, uuid, traceback
from pathlib import Path
from typing import Optional

# Load .env if present (before anything else reads env vars)
try:
    _env = Path(__file__).parent / '.env'
    if _env.exists():
        for _line in _env.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                os.environ.setdefault(_k.strip(), _v.strip())
except Exception:
    pass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

app = FastAPI(title="AtomInfer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Output directory for generated structures and plots
OUTPUT_DIR = Path("./atomInfer_output")
OUTPUT_DIR.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Upload directory
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory run store: run_id → queue of events
runs: dict[str, asyncio.Queue] = {}


# ── Upload endpoint ────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Save uploaded file, return saved path."""
    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    dest = UPLOAD_DIR / safe_name
    content = await file.read()
    dest.write_bytes(content)
    return {"filename": safe_name, "path": str(dest), "size": len(content)}


# ── Start a run ────────────────────────────────────────────────────────────────
@app.post("/api/runs")
async def create_run(body: dict):
    """
    body: {
      "xrd_path": str,        # required
      "sheet_name": str,      # optional
      "raman_path": str,      # optional
      "afm_path": str,        # optional
    }
    Returns { "run_id": str }
    """
    run_id = uuid.uuid4().hex
    q: asyncio.Queue = asyncio.Queue()
    runs[run_id] = q

    async def run_pipeline():
        try:
            await q.put({"type": "status", "message": "Starting AtomInfer pipeline..."})
            
            # Import agent tools (from atomInfer_v2.py in same directory)
            import sys
            sys.path.insert(0, ".")
            
            # We'll emit events manually as each step completes
            # This replaces the CLI verbose output with structured events
            
            xrd_path   = body.get("xrd_path")
            sheet_name = body.get("sheet_name")
            raman_path = body.get("raman_path")
            afm_path   = body.get("afm_path")

            if not xrd_path or not Path(xrd_path).exists():
                await q.put({"type": "error", "message": f"XRD file not found: {xrd_path}"})
                await q.put({"type": "done"})
                return

            # ── STEP 1: Parse XRD ────────────────────────────────────────────
            await q.put({"type": "step", "step": 1, "label": "Parsing XRD data...",
                         "icon": "📂"})
            from atomInfer_v2 import parse_xrd_file, analyze_xrd_peaks
            from atomInfer_v2 import query_materials_project, build_doped_supercell
            from atomInfer_v2 import compute_xrd_r_factor

            xrd_data = parse_xrd_file(xrd_path, sheet_name)
            if "error" in xrd_data:
                await q.put({"type": "error", "message": xrd_data["error"]})
                await q.put({"type": "done"})
                return

            await q.put({
                "type": "event",
                "label": "XRD file parsed",
                "data": {
                    "n_points":      xrd_data["n_points"],
                    "2theta_range":  xrd_data["two_theta_range"],
                    "wavelength_A":  xrd_data["wavelength_A"],
                }
            })

            # ── STEP 2: Analyze XRD ──────────────────────────────────────────
            await q.put({"type": "step", "step": 2, "label": "Analyzing peaks (Nelson-Riley + Scherrer)...",
                         "icon": "🔬"})
            await asyncio.sleep(0.05)

            xrd_analysis = analyze_xrd_peaks(
                xrd_data["two_theta"],
                xrd_data["intensity"],
                xrd_data["wavelength_A"]
            )
            if "error" in xrd_analysis:
                await q.put({"type": "error", "message": xrd_analysis["error"]})
                await q.put({"type": "done"})
                return

            lattice_a = xrd_analysis["lattice_parameter_A"]
            d_nm      = xrd_analysis["crystallite_size_nm"]

            await q.put({
                "type": "event",
                "label": "XRD peaks assigned",
                "data": {
                    "phase":           xrd_analysis["phase_identified"],
                    "lattice_a_A":     lattice_a,
                    "n_peaks":         xrd_analysis["n_peaks_assigned"],
                    "crystallite_nm":  d_nm,
                    "ceo2_secondary":  xrd_analysis["ceo2_secondary_phase"],
                    "delta_a_ICDD":    xrd_analysis["delta_a_vs_ICDD"],
                }
            })

            # Estimate Ce fraction from lattice using Vegard's law approximation
            # Calibrated from Nikhil's data: a increases ~0.035 Å per 1% Ce (from 1-2% Ce)
            # This is a rough estimate — the agent will refine it
            a_pristine = 8.2480  # ICDD reference
            ce_frac = max(0.0, min(0.10, (lattice_a - a_pristine) / 0.35))
            await q.put({
                "type": "event",
                "label": "Ce fraction estimated from Vegard's law",
                "data": {
                    "ce_fraction": round(ce_frac, 4),
                    "ce_pct":      round(ce_frac*100, 2),
                    "note":        "Rough estimate from lattice parameter; structure will refine this"
                }
            })

            # ── STEP 3: Materials Project query ──────────────────────────────
            await q.put({"type": "step", "step": 3,
                         "label": "Fetching base structure from Materials Project (mp-19017)...",
                         "icon": "🌐"})
            await asyncio.sleep(0.05)

            mp_result = query_materials_project()
            if "error" in mp_result:
                await q.put({
                    "type": "warning",
                    "message": f"MP API failed: {mp_result['error']}. Using fallback structure."
                })
                # Fallback: try to use pymatgen built-in
                cif_path = None
            else:
                cif_path = mp_result["cif_saved"]
                await q.put({
                    "type": "event",
                    "label": "MP structure fetched",
                    "data": {
                        "mp_id":       mp_result["mp_id"],
                        "formula":     mp_result["formula"],
                        "space_group": mp_result["space_group"],
                        "lattice_MP":  mp_result["lattice_a_MP"],
                        "n_atoms":     mp_result["n_atoms"],
                    }
                })

            # ── STEP 4: Build doped supercell ─────────────────────────────────
            await q.put({"type": "step", "step": 4,
                         "label": f"Building Ce-doped supercell (Ce={ce_frac*100:.1f}%, a={lattice_a:.4f} Å)...",
                         "icon": "⚛️"})
            await asyncio.sleep(0.05)

            if cif_path:
                build_result = build_doped_supercell(
                    base_cif_path=cif_path,
                    measured_lattice_a=lattice_a,
                    Ce_fraction=ce_frac,
                    crystallite_size_nm=d_nm,
                )
            else:
                build_result = {"error": "No base CIF available. MP API key required."}

            if "error" in build_result:
                await q.put({"type": "error", "message": build_result["error"]})
                await q.put({"type": "done"})
                return

            await q.put({
                "type": "event",
                "label": "Atomistic model built",
                "data": {
                    "supercell":     build_result["supercell_dims"],
                    "n_atoms":       build_result["n_atoms_total"],
                    "n_Li":          build_result["n_Li"],
                    "n_Mn":          build_result["n_Mn"],
                    "n_Ce":          build_result["n_Ce"],
                    "n_O":           build_result["n_O"],
                    "ce_actual_pct": build_result["actual_Ce_pct"],
                    "lattice_a_A":   build_result["lattice_a_used_A"],
                    "cif_file":      build_result["output_cif"],
                    "vasp_file":     build_result["output_vasp"],
                }
            })

            # ── STEP 5: Validate with R-factor ─────────────────────────────────
            await q.put({"type": "step", "step": 5,
                         "label": "Validating structure (XRD R-factor)...",
                         "icon": "✅"})
            await asyncio.sleep(0.05)

            r_result = compute_xrd_r_factor(
                structure_path=build_result["output_vasp"],
                two_theta_exp=xrd_data["two_theta"],
                intensity_exp=xrd_data["intensity"],
                wavelength_A=xrd_data["wavelength_A"],
            )

            if "error" in r_result:
                await q.put({"type": "warning", "message": f"R-factor failed: {r_result['error']}"})
            else:
                await q.put({
                    "type": "event",
                    "label": "R-factor computed",
                    "data": {
                        "R_factor":     r_result["R_factor"],
                        "R_pct":        r_result["R_factor_pct"],
                        "quality":      r_result["quality"],
                        "top_peaks":    r_result["top_sim_peaks"],
                    }
                })

            # ── DONE ──────────────────────────────────────────────────────────
            await q.put({
                "type": "done",
                "summary": {
                    "phase":           xrd_analysis["phase_identified"],
                    "lattice_a_A":     lattice_a,
                    "ce_pct":          round(ce_frac*100, 2),
                    "crystallite_nm":  d_nm,
                    "n_atoms":         build_result["n_atoms_total"],
                    "R_factor":        r_result.get("R_factor") if "error" not in r_result else None,
                    "output_vasp":     build_result["output_vasp"],
                    "output_cif":      build_result["output_cif"],
                }
            })

        except Exception as e:
            tb = traceback.format_exc()
            await q.put({"type": "error", "message": str(e), "traceback": tb[:500]})
            await q.put({"type": "done"})

    # Launch pipeline in background
    asyncio.create_task(run_pipeline())
    return {"run_id": run_id}


# ── WebSocket stream ───────────────────────────────────────────────────────────
@app.websocket("/api/runs/{run_id}/stream")
async def stream_run(websocket: WebSocket, run_id: str):
    await websocket.accept()
    if run_id not in runs:
        await websocket.send_json({"type": "error", "message": "Run not found"})
        await websocket.close()
        return

    q = runs[run_id]
    try:
        while True:
            event = await asyncio.wait_for(q.get(), timeout=120.0)
            await websocket.send_json(event)
            if event.get("type") == "done":
                break
    except asyncio.TimeoutError:
        await websocket.send_json({"type": "error", "message": "Pipeline timed out"})
    except WebSocketDisconnect:
        pass
    finally:
        runs.pop(run_id, None)
        try:
            await websocket.close()
        except:
            pass


# ── Serve the frontend HTML ───────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_ui():
    html_path = Path(__file__).parent / 'atomInfer_ui.html'
    if not html_path.exists():
        return JSONResponse({'error': 'UI file not found'}, status_code=404)
    return FileResponse(str(html_path), media_type='text/html')


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "groq_key": bool(os.environ.get("GROQ_API_KEY")),
        "mp_key":   bool(os.environ.get("MP_API_KEY")),
    }


# ── Materials Project structure fetch (for 3-D viewer) ──────────────────────
@app.get("/api/mp_structure/{mp_id}")
async def get_mp_structure(mp_id: str):
    """Return full site list + lattice for a given MP id so the browser can
    render it with Three.js.  The `structure` field from the summary endpoint
    is a pymatgen-serialised JSON object that includes lattice matrix and sites
    with Cartesian (xyz) and fractional (abc) coordinates."""
    import requests
    key = os.environ.get("MP_API_KEY", "")
    if not key:
        return JSONResponse({"error": "MP_API_KEY not set"}, status_code=503)
    headers = {"X-API-KEY": key, "accept": "application/json"}
    base    = "https://api.materialsproject.org"
    try:
        resp = requests.get(
            f"{base}/materials/summary/",
            params={
                "material_ids": mp_id,
                "_fields": "material_id,formula_pretty,symmetry,nsites,structure,lattice",
            },
            headers=headers, timeout=15,
        )
        resp.raise_for_status()
        raw   = resp.json()
        items = raw if isinstance(raw, list) else raw.get("data", [])
        if not items:
            return JSONResponse({"error": f"{mp_id} not found"}, status_code=404)
        item   = items[0]
        struct = item.get("structure") or {}
        lat    = struct.get("lattice") or item.get("lattice") or {}
        sites  = struct.get("sites") or []
        return {
            "material_id": item.get("material_id", mp_id),
            "formula":     item.get("formula_pretty", ""),
            "symmetry":    item.get("symmetry", {}),
            "lattice":     lat,
            "sites":       sites,
        }
    except requests.exceptions.HTTPError as e:
        return JSONResponse({"error": f"MP API {e.response.status_code}: {e.response.text[:200]}"}, status_code=502)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Materials Project search (frontend helper) ───────────────────────────────
@app.post("/api/mp_search")
async def mp_search(body: dict):
    """Search Materials Project for a provided formula string.
    Returns a small set of results (mock if MP_API_KEY not configured).
    """
    formula = (body or {}).get("formula", "")
    if not formula:
        return JSONResponse({"error": "No formula provided"}, status_code=400)
    try:
        from atomInfer_v2 import search_materials_project
        res = search_materials_project(formula)
        return res
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_server:app", host="0.0.0.0", port=8000, reload=True)
