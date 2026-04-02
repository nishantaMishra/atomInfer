#!/usr/bin/env python3
"""
AtomInfer WebUI Backend
=======================
FastAPI + WebSocket streaming server.
The frontend connects via WebSocket, sends a job spec,
and receives streaming JSON events as the agent runs.

Run:
  python backend_server.py
  # or: uvicorn backend_server:app --reload --host 0.0.0.0 --port 8000

All settings are read from config.toml (see config.default.toml).
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

# Load configuration and model registry
from config_loader import cfg
from model_registry import registry
from materials import get_active_material

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

# Directories from config
OUTPUT_DIR = Path(cfg.directories.output)
OUTPUT_DIR.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

UPLOAD_DIR = Path(cfg.directories.uploads)
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

            # Estimate Ce fraction from lattice using Vegard's law from config
            mat = get_active_material()
            ce_frac = mat.estimate_dopant_fraction(lattice_a)
            await q.put({
                "type": "event",
                "label": "Dopant fraction estimated from Vegard's law",
                "data": {
                    "ce_fraction": round(ce_frac, 4),
                    "ce_pct":      round(ce_frac*100, 2),
                    "note":        "Rough estimate from lattice parameter; structure will refine this"
                }
            })

            # ── STEP 3: Materials Project query ──────────────────────────────
            mp_id = cfg.material.mp_id
            await q.put({"type": "step", "step": 3,
                         "label": f"Fetching base structure from Materials Project ({mp_id})...",
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
    registry.refresh_availability()
    return {
        "status": "ok",
        "groq_key": bool(cfg.get_api_key("groq")),
        "mp_key":   bool(cfg.get_api_key("mp")),
        "config":   cfg.to_summary_dict(),
        "models":   registry.get_status_summary(),
    }


# ── Materials Project structure fetch (for 3-D viewer) ──────────────────────
@app.get("/api/mp_structure/{mp_id}")
async def get_mp_structure(mp_id: str):
    """Return full site list + lattice for a given MP id so the browser can
    render it with Three.js."""
    import requests
    key = cfg.get_api_key("mp") or ""
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


# ── HRMC PNG Frame Generator ──────────────────────────────────────────────────

def _generate_hrmc_png_frames(cif_path: str, ce_pct: float, n_frames: int = 5) -> list:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from pymatgen.core import Structure

    CPK   = {'Li':('#cc80ff',80),'Mn':('#9c7ac7',130),'O':('#ff2200',55),'Ce':('#ffff88',180)}
    s     = Structure.from_file(cif_path)
    mn_idx = [i for i,site in enumerate(s) if str(site.specie)=='Mn']
    n_ce_target = max(1, int((ce_pct/100.0)*len(mn_idx)))

    def proj(v): return v[0]+0.45*v[2], v[1]+0.35*v[2]

    paths = []
    for fi in range(n_frames):
        sc = s.copy()
        t  = fi / max(n_frames-1, 1)
        # Progressive Ce substitution
        np.random.seed(42)
        n_ce_now = round(n_ce_target * t)
        chosen   = np.random.choice(mn_idx, min(n_ce_now, len(mn_idx)), replace=False)
        for idx in chosen: sc[int(idx)] = 'Ce'
        # Progressive Mn/Ce displacements
        np.random.seed(fi*13+7)
        for i,site in enumerate(sc):
            if str(site.specie) in ('Mn','Ce'):
                sc.translate_sites([i], np.random.uniform(-0.5,0.5,3)*t, frac_coords=False)

        fig, ax = plt.subplots(figsize=(5,5), facecolor='#070a0e')
        ax.set_facecolor('#070a0e'); ax.set_aspect('equal'); ax.axis('off')
        # Unit cell edges
        m = sc.lattice.matrix
        o = np.zeros(3)
        corners = [o,m[0],m[1],m[2],m[0]+m[1],m[0]+m[2],m[1]+m[2],m[0]+m[1]+m[2]]
        for a,b in [(0,1),(0,2),(0,3),(1,4),(1,5),(2,4),(2,6),(3,5),(3,6),(4,7),(5,7),(6,7)]:
            p1,p2=proj(corners[a]),proj(corners[b])
            ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color='#3d7eff',alpha=0.45,lw=0.8)
        # Atoms back→front
        for site in sorted(sc, key=lambda s: s.coords[2]):
            el = str(site.specie)
            col,sz = CPK.get(el,('#ff69b4',80))
            px,py  = proj(site.coords)
            ax.scatter(px,py,c=col,s=sz,alpha=0.92,edgecolors='#ffffff',linewidths=0.35,zorder=3)
        ce_n = sum(1 for site in sc if str(site.specie)=='Ce')
        ax.set_title(f'MC Step {fi+1}/{n_frames}  ·  Ce: {ce_n} atoms substituted',
                     color='#00e5c8',fontsize=10,pad=6,fontfamily='monospace')
        plt.tight_layout(pad=0.2)
        fpath = str(OUTPUT_DIR / f'hrmc_frame_{fi:02d}.png')
        plt.savefig(fpath, dpi=130, facecolor='#070a0e', bbox_inches='tight')
        plt.close()
        paths.append(f'/outputs/hrmc_frame_{fi:02d}.png')
    return paths


@app.post("/api/hrmc/frames")
async def hrmc_frames_endpoint(body: dict):
    import concurrent.futures
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        frames = await loop.run_in_executor(
            pool, lambda: _generate_hrmc_png_frames(
                body.get("cif_path","./LiMn2O4.cif"),
                float(body.get("ce_pct",5.0)),
                int(body.get("n_frames",5))
            ))
    return {"frames": frames}


# ── HRMC Integration ──────────────────────────────────────────────────────────

def structure_to_viewer_json(structure) -> dict:
    """Convert a pymatgen Structure to the Three.js viewer format used by /api/mp_structure/."""
    lat = structure.lattice
    mat = cfg.material
    return {
        "material_id": "hrmc",
        "formula": structure.formula,
        "nsites": len(structure),
        "symmetry": {"symbol": mat.space_group, "crystal_system": mat.crystal_system},
        "lattice": {
            "matrix": lat.matrix.tolist(),
            "a": float(lat.a), "b": float(lat.b), "c": float(lat.c),
            "alpha": float(lat.alpha), "beta": float(lat.beta), "gamma": float(lat.gamma),
            "volume": float(lat.volume),
        },
        "sites": [
            {
                "species": [{"element": str(site.specie), "occu": 1.0}],
                "label": str(site.specie),
                "xyz": site.coords.tolist(),
                "abc": site.frac_coords.tolist(),
            }
            for site in structure
        ],
    }


@app.get("/api/cif_structure")
async def get_cif_structure(path: str = "./LiMn2O4.cif"):
    """Load a local CIF file and return a Three.js-compatible structure dict."""
    try:
        import sys; sys.path.insert(0, str(Path(__file__).parent))
        from pymatgen.core import Structure
        s = Structure.from_file(path)
        return structure_to_viewer_json(s)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# HRMC job queues: run_id → asyncio.Queue of events
hrmc_jobs: dict[str, asyncio.Queue] = {}


@app.post("/api/hrmc/start")
async def hrmc_start(body: dict):
    """
    Start an HRMC refinement run.
    body: { cif_path, ce_pct, n_steps, supercell? }
    Returns { run_id }
    """
    run_id = uuid.uuid4().hex
    q: asyncio.Queue = asyncio.Queue()
    hrmc_jobs[run_id] = q
    loop = asyncio.get_running_loop()

    async def _run():
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        import numpy as np
        from pymatgen.core import Structure, Lattice
        from pymatgen.analysis.diffraction.xrd import XRDCalculator
        from meam_calculator import MEAMCalculator

        cif_path  = body.get("cif_path", "./LiMn2O4.cif")
        ce_pct    = float(body.get("ce_pct", 5.0))
        n_steps   = int(body.get("n_steps", cfg.hrmc.default_steps))
        supercell = body.get("supercell", [1, 1, 1])
        x_ce      = ce_pct / 100.0

        # Configuration from config.toml
        XRD_INTERVAL   = cfg.hrmc.xrd_interval
        TARGET_FPS     = cfg.hrmc.target_fps
        FRAME_DELAY    = 1.0 / TARGET_FPS

        async def emit(data):
            await q.put(data)
            await asyncio.sleep(0.001)  # tiny yield to ensure flush

        try:
            # ── STEP 1: Load Structure ──────────────────────────────────────
            await emit({"type": "status", "message": "Loading structure from CIF..."})
            
            structure = await asyncio.to_thread(Structure.from_file, cif_path)
            
            if supercell != [1, 1, 1]:
                await emit({"type": "status", "message": f"Building supercell {supercell}..."})
                await asyncio.to_thread(structure.make_supercell, supercell)

            sp_list = [str(s.specie) for s in structure]
            host_el = cfg.material.doping.host_element
            mn_idx  = [i for i, sp in enumerate(sp_list) if sp == host_el]
            n_ce_init = max(1, int(x_ce * len(mn_idx))) if x_ce > 0 and mn_idx else 0
            
            dopant_el = cfg.material.doping.allowed_dopants[0] if cfg.material.doping.allowed_dopants else "Ce"
            if n_ce_init:
                chosen = np.random.choice(mn_idx, n_ce_init, replace=False)
                for idx in chosen:
                    structure[int(idx)] = dopant_el

            await emit({"type": "status",
                  "message": f"Loaded {structure.formula} ({len(structure)} atoms)"})

            # ── STEP 2: Initialize MEAM Calculator ──────────────────────────
            await emit({"type": "status", "message": "Initializing MEAM calculator..."})
            meam = MEAMCalculator(rcut=cfg.potentials.cutoff_radius_A)

            # ── STEP 3: Build synthetic target XRD ──────────────────────────
            await emit({"type": "status", "message": "Building target XRD pattern..."})
            
            a_tgt = 8.248 + 0.04 * x_ce
            lat_syn = Lattice.cubic(a_tgt)
            li_s = [[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0],
                    [.25,.25,.25],[.25,.75,.75],[.75,.25,.75],[.75,.75,.25]]
            mn_s = [[.625,.625,.625],[.625,.875,.875],[.875,.625,.875],[.875,.875,.625],
                    [.125,.125,.625],[.125,.375,.875],[.375,.125,.875],[.375,.375,.625],
                    [.125,.625,.125],[.125,.875,.375],[.375,.625,.375],[.375,.875,.125],
                    [.625,.125,.125],[.625,.375,.375],[.875,.125,.375],[.875,.375,.125]]
            o_s  = [[.386,.386,.386],[.386,.614,.614],[.614,.386,.614],[.614,.614,.386],
                    [.136,.136,.136],[.136,.364,.364],[.364,.136,.364],[.364,.364,.136]]
            n_ce_syn = max(1, round(x_ce * len(mn_s))) if x_ce > 0 else 0
            sp_syn   = (["Li"]*len(li_s) + ["Ce"]*n_ce_syn +
                        ["Mn"]*(len(mn_s)-n_ce_syn) + ["O"]*len(o_s))
            syn_struct  = Structure(lat_syn, sp_syn, li_s + mn_s + o_s)
            
            xrd_calc    = XRDCalculator(wavelength="CuKa")
            patt_syn    = await asyncio.to_thread(xrd_calc.get_pattern, syn_struct, two_theta_range=(10, 80))
            grid        = np.linspace(10, 80, 300)
            I_exp       = np.interp(grid, patt_syn.x, patt_syn.y, left=0, right=0)
            I_exp_norm  = (100 * I_exp / (I_exp.max() + 1e-10)).tolist()
            grid_list   = grid.tolist()

            # ── Helpers (run in thread to not block) ────────────────────────
            def calc_xrd(s):
                """Return (R_factor, sim_y_list) using shared grid."""
                p     = xrd_calc.get_pattern(s, two_theta_range=(10, 80))
                I_sim = np.interp(grid, p.x, p.y, left=0, right=0)
                sim_n = 100 * I_sim / (I_sim.max() + 1e-10)
                exp_n = 100 * I_exp / (I_exp.max() + 1e-10)
                R     = float(np.sum(np.abs(exp_n - sim_n)) /
                              (np.sum(np.abs(exp_n)) + 1e-10))
                return R, sim_n.tolist()

            def calc_meam_cost(s, R_cached):
                """Cost using R-factor + composition constraint. MEAM is computed for display only."""
                E   = meam.compute_energy(s) / len(s)
                sp_ = [str(st.specie) for st in s]
                nCe = sp_.count(dopant_el); nMn = sp_.count(host_el)
                xc  = nCe / (nCe + nMn) if (nCe + nMn) > 0 else 0
                # Use only R-factor + composition constraint for MC acceptance
                F   = w_xrd*R_cached + w_ce*(xc - x_ce)**2
                return {"F": F, "R_factor": R_cached, "E_per_atom": E, "x_Ce": xc}

            def metropolis(dF, T):
                return dF <= 0 or np.random.random() < np.exp(-dF / max(T, 1e-9))

            # ── STEP 4: Compute Initial State ───────────────────────────────
            await emit({"type": "status", "message": "Computing initial XRD and energy..."})
            
            current       = structure.copy()
            R_cur, sim_y  = await asyncio.to_thread(calc_xrd, current)
            cost          = await asyncio.to_thread(calc_meam_cost, current, R_cur)

            await emit({"type": "frame", "step": 0,
                  "cost": cost,
                  "structure": structure_to_viewer_json(current),
                  "pattern": {"x": grid_list, "sim_y": sim_y, "exp_y": I_exp_norm},
                  "n_steps": n_steps})

            # ── STEP 5: MC Loop ─────────────────────────────────────────────
            await emit({"type": "status", "message": f"Starting MC loop ({n_steps} steps)..."})
            
            T_start = cfg.hrmc.temperature.start
            T_end   = cfg.hrmc.temperature.end
            disp_w  = cfg.hrmc.moves.displacement_weight
            max_disp = cfg.hrmc.moves.max_displacement_A
            w_xrd   = cfg.hrmc.cost_weights.xrd
            w_ce    = cfg.hrmc.cost_weights.ce_target
            last_emit_time = asyncio.get_event_loop().time()
            
            for step in range(n_steps):
                T    = T_start + (T_end - T_start) * step / max(n_steps - 1, 1)
                roll = np.random.random()
                trial = None

                # Displacement move
                if roll < disp_w:
                    sp_ = [str(st.specie) for st in current]
                    cands = [i for i, sp in enumerate(sp_) if sp != "Li"]
                    if cands:
                        trial = current.copy()
                        idx   = int(np.random.choice(cands))
                        disp  = np.random.uniform(-max_disp, max_disp, 3)
                        trial.translate_sites([idx], disp, frac_coords=False)
                # Swap move: dopant<->host substitution
                else:
                    sp_ = [str(st.specie) for st in current]
                    nCe = sp_.count(dopant_el); nMn = sp_.count(host_el)
                    x_cur = nCe / (nCe + nMn) if (nCe + nMn) > 0 else 0
                    trial = current.copy()
                    if x_cur < x_ce:
                        mn_cands = [i for i, sp in enumerate(sp_) if sp == host_el]
                        if mn_cands:
                            trial[int(np.random.choice(mn_cands))] = dopant_el
                        else:
                            trial = None
                    else:
                        ce_cands = [i for i, sp in enumerate(sp_) if sp == dopant_el]
                        if ce_cands:
                            trial[int(np.random.choice(ce_cands))] = host_el
                        else:
                            trial = None

                # Evaluate and accept/reject
                if trial is not None:
                    trial_cost = await asyncio.to_thread(calc_meam_cost, trial, R_cur)
                    if metropolis(trial_cost["F"] - cost["F"], T):
                        current = trial
                        cost    = trial_cost

                # Refresh XRD every XRD_INTERVAL steps
                if (step + 1) % XRD_INTERVAL == 0:
                    R_cur, sim_y = await asyncio.to_thread(calc_xrd, current)
                    cost = await asyncio.to_thread(calc_meam_cost, current, R_cur)

                # Emit frame every step with a small sleep so browser can render
                include_pattern = ((step + 1) % XRD_INTERVAL == 0 or step == n_steps - 1)
                await emit({"type": "frame", "step": step + 1,
                      "cost": cost,
                      "structure": structure_to_viewer_json(current),
                      "pattern": ({"x": grid_list, "sim_y": sim_y, "exp_y": I_exp_norm}
                                  if include_pattern else None),
                      "n_steps": n_steps})
                await asyncio.sleep(0.04)  # 40ms/step → ~25fps max, browser can render each frame

            await emit({"type": "done", "n_steps": n_steps, "final_cost": cost})

        except Exception as exc:
            import traceback as tb
            await emit({"type": "error", "message": str(exc),
                  "traceback": tb.format_exc()[:800]})
            await emit({"type": "done"})

    asyncio.create_task(_run())
    return {"run_id": run_id}


@app.websocket("/api/hrmc/{run_id}/stream")
async def hrmc_stream(websocket: WebSocket, run_id: str):
    await websocket.accept()
    if run_id not in hrmc_jobs:
        await websocket.send_json({"type": "error", "message": "HRMC run not found"})
        await websocket.close()
        return
    q = hrmc_jobs[run_id]
    try:
        while True:
            event = await asyncio.wait_for(q.get(), timeout=300.0)
            await websocket.send_json(event)
            if event.get("type") == "done":
                break
    except asyncio.TimeoutError:
        await websocket.send_json({"type": "error", "message": "HRMC run timed out"})
    except WebSocketDisconnect:
        pass
    finally:
        hrmc_jobs.pop(run_id, None)
        try:
            await websocket.close()
        except Exception:
            pass


# ── Configuration & Material endpoints ─────────────────────────────────────────

@app.get("/api/config")
async def get_config():
    """Return active configuration summary for the frontend UI."""
    registry.refresh_availability()
    return {
        "config": cfg.to_summary_dict(),
        "models": registry.get_status_summary(),
    }


@app.get("/api/materials/list")
async def list_materials_endpoint():
    """Return all configured material profiles."""
    from materials import list_materials
    return {"materials": list_materials(), "active": cfg.material.name}


if __name__ == "__main__":
    import uvicorn, webbrowser, threading

    def _open_browser():
        """Wait briefly for the server to start, then open the UI."""
        import time, urllib.request
        url = f"http://localhost:{cfg.server.port}"
        for _ in range(20):
            time.sleep(0.5)
            try:
                urllib.request.urlopen(f"{url}/health")
                break
            except Exception:
                continue
        webbrowser.open(url)

    if cfg.server.open_browser:
        threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run("backend_server:app", host=cfg.server.host,
                port=cfg.server.port, reload=True)
