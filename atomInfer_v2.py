#!/usr/bin/env python3
"""
AtomInfer v2 - Experimental Characterization → Atomistic Model Agent
=====================================================================
Architecture: Raw tool-calling loop with configurable LLM providers.
Supports: Ollama (local), Groq, OpenAI, Anthropic, vLLM, LM Studio.
Configuration: config.toml (see config.default.toml for reference).

Tools:
  XRD   : parse_xrd_file, analyze_xrd_peaks
  Raman : parse_raman_file, analyze_raman_peaks
  AFM   : parse_afm_file, analyze_afm_data
  Model : query_materials_project, build_doped_supercell, compute_xrd_r_factor

Usage:
  # Configure your models in config.toml, then:
  python atomInfer_v2.py --xrd XRD_Data.xlsx --sheet "2% Ce-doped"

  # Multi-technique
  python atomInfer_v2.py --xrd data.xlsx --sheet "2% Ce" --raman raman.txt

  # Full demo (all sheets)
  python atomInfer_v2.py --demo --xrd XRD_Data.xlsx
"""

import os, json, argparse, sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Configuration & model registry ───────────────────────────────────────────
from config_loader import cfg
from model_registry import registry

def get_llm_client(task: str = "general_reasoning"):
    """Get an LLM client for the given task type.

    Uses model_registry to select the best available model based on
    config.toml [llm.task_assignments] and model availability.
    Returns (client, model_name) compatible with the agent loop.
    """
    llm_client, model_name = registry.get_client_for_task(task)
    return llm_client, model_name

# ── pymatgen ─────────────────────────────────────────────────────────────────
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# =============================================================================
# SYSTEM PROMPT
# Generated dynamically from the active material profile in config.toml.
# The agent must derive all numbers from the data via tools.
# =============================================================================

SYSTEM_PROMPT = cfg.generate_system_prompt()


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

def parse_xrd_file(filepath: str, sheet_name: str = None) -> dict:
    """
    Parse XRD data file. Supports:
    - Rigaku Excel (.xlsx): columns Angle, I(o), I(b) — net = I(o)-I(b)
    - Plain text (.txt, .csv, .xy): two columns 2theta, intensity
    """
    fp = Path(filepath)
    if not fp.exists():
        return {"error": f"File not found: {filepath}"}

    try:
        if fp.suffix.lower() in ['.xlsx', '.xls']:
            if sheet_name is None:
                xl = pd.ExcelFile(filepath)
                sheet_name = xl.sheet_names[0]
                print(f"  [parse_xrd] No sheet specified, using: {sheet_name}")

            df = pd.read_excel(filepath, sheet_name=sheet_name, header=None)

            # Find header row with 'Angle'
            header_row = None
            for i, row in df.iterrows():
                if str(row.iloc[0]).strip() == 'Angle':
                    header_row = i
                    break
            if header_row is None:
                return {"error": "Could not find 'Angle' column header in sheet"}

            # Extract metadata
            meta = str(df.iloc[1, 0]) if len(df) > 1 else ""

            data = df.iloc[header_row+1:, :3].copy()
            data.columns = ['two_theta', 'I_obs', 'I_bg']
            data = data.dropna()
            data = data[pd.to_numeric(data['two_theta'], errors='coerce').notna()]
            data = data.astype(float)
            data['I_net'] = data['I_obs'] - data['I_bg']

        else:
            # Plain text: 2 columns
            data = pd.read_csv(filepath, sep=None, engine='python',
                               comment='#', header=None,
                               names=['two_theta', 'I_net'])
            data = data.dropna().astype(float)
            meta = f"Text file: {filepath}"

        # Filter to useful angular range
        ang_lo, ang_hi = cfg.xrd.angular_range_deg
        data = data[(data['two_theta'] >= ang_lo) & (data['two_theta'] <= ang_hi)]

        return {
            "status": "success",
            "sheet": sheet_name,
            "n_points": len(data),
            "two_theta_range": [round(float(data['two_theta'].min()), 2),
                                 round(float(data['two_theta'].max()), 2)],
            "wavelength_A": cfg.xrd.wavelength_A,
            "two_theta": data['two_theta'].tolist(),
            "intensity":  data['I_net'].tolist(),
            "metadata": meta[:120],
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_xrd_peaks(two_theta: list, intensity: list,
                      wavelength_A: float = None) -> dict:
    """
    Peak finding, hkl assignment, Nelson-Riley lattice parameter extraction,
    Scherrer crystallite size from first peak.
    Also checks for secondary phase peaks.
    All parameters driven by config.toml.
    """
    from scipy.signal import find_peaks

    tth = np.array(two_theta)
    I   = np.array(intensity)
    lam = wavelength_A if wavelength_A is not None else cfg.xrd.wavelength_A

    # Reflection windows from config
    hkl_windows = cfg.get_hkl_windows_tuples()

    nr_x, nr_a = [], []
    assigned_peaks = {}

    for (h,k,l), (lo,hi) in hkl_windows.items():
        mask = (tth >= lo) & (tth <= hi)
        if mask.sum() < 5:
            continue
        w_tth = tth[mask]
        w_I   = I[mask]
        pk    = np.argmax(w_I)

        # Parabolic sub-pixel refinement
        if 0 < pk < len(w_tth)-1:
            y0,y1,y2 = w_I[pk-1], w_I[pk], w_I[pk+1]
            x0,x1,x2 = w_tth[pk-1], w_tth[pk], w_tth[pk+1]
            denom = y0 - 2*y1 + y2
            tth_pk = x1 - 0.5*(x2-x0)*(y2-y0)/(2*denom) if abs(denom)>1 else x1
            if not (lo < tth_pk < hi):
                tth_pk = x1
        else:
            tth_pk = w_tth[pk]

        theta  = np.radians(tth_pk/2)
        d      = lam / (2*np.sin(theta))
        a      = d * np.sqrt(h**2 + k**2 + l**2)
        nr_fn  = (np.cos(theta)**2/np.sin(theta) + np.cos(theta)**2/theta) / 2
        nr_x.append(nr_fn)
        nr_a.append(a)

        assigned_peaks[f"({h}{k}{l})"] = {
            "2theta_obs":   round(float(tth_pk), 4),
            "intensity":    round(float(w_I[pk]), 1),
            "d_spacing_A":  round(float(d), 4),
            "a_from_peak":  round(float(a), 4),
        }

    # Nelson-Riley lattice parameter
    if len(nr_x) >= 3:
        coeffs = np.polyfit(nr_x, nr_a, 1)
        a0 = float(coeffs[1])
        nr_quality = "good" if len(nr_x) >= 4 else "fair"
    elif len(nr_x) >= 1:
        a0 = float(np.mean(nr_a))
        nr_quality = "poor (few peaks)"
    else:
        return {"error": "No LiMn2O4 spinel peaks found in XRD data"}

    # Scherrer from first hkl window (typically (111))
    first_hkl = next(iter(hkl_windows), None)
    if first_hkl:
        flo, fhi = hkl_windows[first_hkl]
        mask_first = (tth >= flo - 0.5) & (tth <= fhi + 0.5)
    else:
        mask_first = (tth >= 17.5) & (tth <= 21.0)
    w_tth_first = tth[mask_first]
    w_I_first   = I[mask_first]
    K = cfg.xrd.scherrer_constant
    if len(w_I_first) > 5:
        pk_first = np.argmax(w_I_first)
        tth_111 = w_tth_first[pk_first]
        half_max = w_I_first[pk_first] / 2
        above = w_tth_first[w_I_first >= half_max]
        fwhm_deg = float(above[-1]-above[0]) if len(above)>=2 else 0.15
        fwhm_rad = np.radians(fwhm_deg)
        theta_111 = np.radians(tth_111/2)
        D_nm = float(K * lam / (fwhm_rad * np.cos(theta_111)) / 10)
    else:
        fwhm_deg, D_nm, tth_111 = None, None, None

    # Check for secondary phases from config
    ceo2_detected = []
    for phase_name, phase_cfg in cfg.xrd.secondary_phases.items():
        for pos in phase_cfg.peak_positions_deg:
            mask_c = (tth >= pos-0.5) & (tth <= pos+0.5)
            if mask_c.sum() > 0:
                local_I = I[mask_c]
                if local_I.max() > I.max() * phase_cfg.detection_threshold:
                    ceo2_detected.append({
                        "phase": phase_name,
                        "2theta": pos,
                        "relative_intensity": round(float(local_I.max()/I.max()), 3)
                })

    mat = cfg.material
    return {
        "phase_identified":       f"{mat.formula} {mat.structure_type} ({mat.space_group})",
        "n_peaks_assigned":       len(assigned_peaks),
        "assigned_peaks":         assigned_peaks,
        "lattice_parameter_A":    round(a0, 4),
        "nr_quality":             nr_quality,
        "crystallite_size_nm":    round(D_nm, 1) if D_nm else None,
        "fwhm_111_deg":           round(fwhm_deg, 4) if fwhm_deg else None,
        "tth_111_deg":            round(float(tth_111), 4) if tth_111 is not None else None,
        "ceo2_secondary_phase":   ceo2_detected if ceo2_detected else "not detected",
        "delta_a_vs_ICDD":        round(a0 - mat.reference_lattice_A, 4),
        "note": f"Negative delta_a common in nanoparticle {mat.formula}; systematic zero-error possible"
    }


def parse_raman_file(filepath: str, sheet_name: str = None) -> dict:
    """
    Parse Raman spectrum. Supports .txt, .csv, .xlsx.
    Expects columns: wavenumber (cm-1), intensity.
    """
    fp = Path(filepath)
    if not fp.exists():
        return {"error": f"File not found: {filepath}"}
    try:
        if fp.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath, sheet_name=sheet_name or 0, header=None)
            # Find numeric start
            start = 0
            for i, row in df.iterrows():
                if pd.to_numeric(row.iloc[0], errors='coerce') is not np.nan:
                    try:
                        float(row.iloc[0])
                        start = i
                        break
                    except:
                        pass
            data = df.iloc[start:, :2].copy()
        else:
            data = pd.read_csv(filepath, sep=None, engine='python',
                               comment='#', header=None)
            data = data.iloc[:, :2]

        data.columns = ['wavenumber', 'intensity']
        data = data.dropna()
        data = data[pd.to_numeric(data['wavenumber'], errors='coerce').notna()]
        data = data.astype(float)
        data = data[(data['wavenumber'] >= 100) & (data['wavenumber'] <= 1000)]
        data = data.sort_values('wavenumber')

        return {
            "status": "success",
            "n_points": len(data),
            "wavenumber_range": [round(float(data['wavenumber'].min()), 1),
                                  round(float(data['wavenumber'].max()), 1)],
            "wavenumber": data['wavenumber'].tolist(),
            "intensity":  data['intensity'].tolist(),
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_raman_peaks(wavenumber: list, intensity: list) -> dict:
    """
    Find and assign Raman peaks for LiMn2O4 spinel.
    Extracts A1g position, FWHM, disorder indicators, CeO2 check.
    """
    from scipy.signal import find_peaks
    from scipy.optimize import curve_fit

    wn = np.array(wavenumber)
    I  = np.array(intensity)

    def lorentzian(x, x0, gamma, A):
        return A * gamma**2 / ((x-x0)**2 + gamma**2)

    # Raman mode windows from config
    mode_windows = {}
    for mode, (lo, hi) in cfg.raman.mode_windows.items():
        mode_windows[mode] = (lo, hi)

    results = {}
    for mode, (lo, hi) in mode_windows.items():
        mask = (wn >= lo) & (wn <= hi)
        if mask.sum() < 5:
            continue
        w_wn = wn[mask]
        w_I  = I[mask]
        pk   = np.argmax(w_I)
        wn_pk = w_wn[pk]

        # FWHM
        half_max = w_I[pk] / 2
        above = w_wn[w_I >= half_max]
        fwhm = float(above[-1]-above[0]) if len(above)>=2 else None

        results[mode] = {
            "peak_position_cm1": round(float(wn_pk), 1),
            "intensity":         round(float(w_I[pk]), 1),
            "fwhm_cm1":          round(fwhm, 1) if fwhm else None,
            "relative_intensity": round(float(w_I[pk]/I.max()), 3),
        }

    # Disorder metric: I(577)/I(625) ratio
    i577 = results.get("A1g_Mn3O", {}).get("intensity", 0)
    i625 = results.get("A1g_Mn4O", {}).get("intensity", 1)
    disorder_ratio = round(i577/i625, 3) if i625 > 0 else None

    # A1g shift from pristine reference
    a1g_pos = results.get("A1g_Mn4O", {}).get("peak_position_cm1")
    a1g_shift = round(a1g_pos - cfg.raman.reference_A1g_cm1, 1) if a1g_pos else None

    # CeO2 check
    ceo2 = results.get("CeO2", {})
    ceo2_present = ceo2.get("relative_intensity", 0) > 0.05

    return {
        "assigned_modes":    results,
        "A1g_position_cm1":  a1g_pos,
        "A1g_shift_cm1":     a1g_shift,
        "A1g_FWHM_cm1":      results.get("A1g_Mn4O", {}).get("fwhm_cm1"),
        "disorder_ratio_I577_I625": disorder_ratio,
        "CeO2_secondary_detected": ceo2_present,
        "interpretation": {
            "A1g_shift": "negative shift = lattice softening from dopant" if a1g_shift and a1g_shift < 0 else "positive shift = lattice stiffening",
            "disorder":  "high Mn3+ character" if disorder_ratio and disorder_ratio > 0.3 else "low Mn3+ content",
        }
    }


def parse_afm_file(filepath: str) -> dict:
    """
    Parse AFM data. Supports:
    - .txt with height matrix
    - .csv with x, y, z columns
    - .ibw, .spm (Asylum/Bruker): read as binary if possible
    Basic roughness and grain statistics from height data.
    """
    fp = Path(filepath)
    if not fp.exists():
        return {"error": f"File not found: {filepath}"}

    try:
        if fp.suffix.lower() in ['.txt', '.csv']:
            data = pd.read_csv(filepath, sep=None, engine='python',
                               comment='#', header=None)
            z_data = data.values.astype(float)
        else:
            return {"error": f"AFM format {fp.suffix} not yet supported. "
                             f"Export as .txt height matrix and retry."}

        return {
            "status": "success",
            "data_shape": list(z_data.shape),
            "z_matrix": z_data.flatten().tolist()[:500],  # first 500 for analysis
            "z_min_nm": round(float(z_data.min()), 3),
            "z_max_nm": round(float(z_data.max()), 3),
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_afm_data(z_matrix: list, pixel_size_nm: float = 1.0) -> dict:
    """
    Compute surface roughness (Ra, Rq) and estimate grain size from AFM height data.
    pixel_size_nm: physical size of each pixel in nm (from instrument settings).
    """
    z = np.array(z_matrix)
    z = z - z.mean()  # level

    Ra = float(np.mean(np.abs(z)))    # average roughness
    Rq = float(np.sqrt(np.mean(z**2)))  # RMS roughness

    # Simple grain size estimate from autocorrelation length
    # Correlation length ≈ distance where autocorrelation drops to 1/e
    if len(z) > 10:
        autocorr = np.correlate(z, z, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr /= autocorr[0]
        try:
            corr_len_px = np.where(autocorr < 1/np.e)[0][0]
            grain_size_nm = round(corr_len_px * pixel_size_nm * 2, 1)
        except:
            grain_size_nm = None
    else:
        grain_size_nm = None

    return {
        "Ra_nm":         round(Ra, 3),
        "Rq_nm":         round(Rq, 3),
        "z_range_nm":    round(float(np.ptp(z)), 3),
        "grain_size_nm": grain_size_nm,
        "note": "Grain size from AFM ≥ Scherrer size (AFM sees agglomerates)"
    }


def query_materials_project(formula: str = None,
                             mp_id: str = None,
                             api_key: str = None) -> dict:
    """
    Fetch structure from Materials Project.
    Defaults to the active material profile from config.toml.
    Returns structure summary + saves CIF to disk.
    api_key: from env MP_API_KEY if not provided.
    """
    formula = formula or cfg.material.formula
    mp_id = mp_id or cfg.material.mp_id
    key = api_key or cfg.get_api_key("mp") or ""
    if not key:
        return {"error": "MP_API_KEY not set. Set env variable or pass api_key parameter."}

    try:
        from mp_api.client import MPRester
        with MPRester(key) as mpr:
            # Get structure
            structure = mpr.get_structure_by_material_id(mp_id, conventional_unit_cell=True)

        out_dir = Path(cfg.directories.output)
        out_dir.mkdir(exist_ok=True)
        cif_path = str(out_dir / f"base_{mp_id}.cif")
        structure.to(filename=cif_path, fmt="cif")

        species_count = {}
        for site in structure:
            sp = str(site.specie)
            species_count[sp] = species_count.get(sp, 0) + 1

        return {
            "status": "success",
            "mp_id": mp_id,
            "formula": str(structure.composition.reduced_formula),
            "space_group": str(structure.get_space_group_info()),
            "lattice_a_MP": round(structure.lattice.a, 4),
            "lattice_b_MP": round(structure.lattice.b, 4),
            "lattice_c_MP": round(structure.lattice.c, 4),
            "n_atoms": len(structure),
            "species_count": species_count,
            "cif_saved": cif_path,
            "note": "Use lattice_a_MP as reference only. Scale to your measured value."
        }
    except Exception as e:
        return {"error": str(e)}


def search_materials_project(formula: str, max_results: int = 6, api_key: str = None) -> dict:
    """
    Search Materials Project v3 REST API for structures matching `formula`.
    Uses requests to call https://api.materialsproject.org directly — no
    mp_api client required.  Falls back to a small mock set when no API key
    is configured, so the UI stays functional during development.

    Returns: { status, results: [ {mp_id, formula, space_group, lattice_a, n_atoms} ] }
    """
    import requests

    key = api_key or cfg.get_api_key("mp") or ""

    # ── No API key: return sensible mock results ──────────────────────────────
    if not key:
        mat = cfg.material
        mock = []
        # Provide a mock entry based on the active material profile
        mock = [
            {"mp_id": mat.mp_id, "formula": mat.formula,
             "space_group": mat.space_group, "lattice_a": mat.reference_lattice_A, "n_atoms": 56},
        ]
        return {"status": "mock", "results": mock, "note": "MP_API_KEY not set — showing demo results"}

    # ── Live search via MP REST API v3 ────────────────────────────────────────
    base = "https://api.materialsproject.org"
    headers = {"X-API-KEY": key, "accept": "application/json"}

    try:
        # Use the summary endpoint with formula filter
        params = {
            "formula": formula,
            "_fields": "material_id,formula_pretty,symmetry,nsites,lattice",
            "_limit": max_results,
        }
        resp = requests.get(f"{base}/materials/summary/", params=params,
                            headers=headers, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
        items = raw if isinstance(raw, list) else raw.get("data", [])

        results = []
        for it in items:
            # Extract lattice 'a' from nested structure if present
            lattice_a = None
            lat = it.get("lattice") or {}
            if isinstance(lat, dict):
                lattice_a = lat.get("a") or lat.get("matrix", [[None]])[0][0]
            # Space group symbol
            sym = it.get("symmetry") or {}
            sg = sym.get("symbol") or sym.get("point_group") or "—"
            results.append({
                "mp_id":       it.get("material_id", ""),
                "formula":     it.get("formula_pretty", ""),
                "space_group": sg,
                "lattice_a":   round(float(lattice_a), 4) if lattice_a else None,
                "n_atoms":     it.get("nsites"),
            })

        return {"status": "success", "results": results}

    except requests.exceptions.HTTPError as e:
        return {"error": f"MP API HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}


def build_doped_supercell(base_cif_path: str,
                           measured_lattice_a: float,
                           Ce_fraction: float,
                           crystallite_size_nm: float = None,
                           supercell_hint: list = None) -> dict:
    """
    Build Ce-doped LiMn2O4 supercell:
    1. Load base structure from CIF (from MP)
    2. Scale lattice to measured value
    3. Make supercell — size based on crystallite_size_nm if provided
    4. Randomly substitute Ce_fraction of Mn (16d) sites with Ce
    5. Save as CIF and POSCAR

    Ce_fraction: fraction of Mn sites to substitute (e.g. 0.02 for 2%)
    crystallite_size_nm: if provided, supercell edge ≈ crystallite size
    supercell_hint: override supercell dimensions [nx, ny, nz]
    """
    fp = Path(base_cif_path)
    if not fp.exists():
        return {"error": f"CIF file not found: {base_cif_path}"}

    try:
        structure = Structure.from_file(str(fp))

        # Scale lattice to measured value
        scale = measured_lattice_a / structure.lattice.a
        new_lattice = Lattice.cubic(measured_lattice_a)

        # Rebuild with scaled lattice (preserve fractional coords)
        structure = Structure(new_lattice,
                              [str(s.specie) for s in structure],
                              [s.frac_coords for s in structure])

        # Determine supercell size
        max_sc = cfg.structure.max_supercell_dim
        if supercell_hint:
            sc = supercell_hint
        elif crystallite_size_nm:
            # Edge length in Angstroms
            target_A = crystallite_size_nm * 10
            n = max(1, round(target_A / measured_lattice_a))
            n = min(n, max_sc)
            sc = [n, n, n]
        else:
            sc = list(cfg.structure.default_supercell)

        structure.make_supercell(sc)
        n_total = len(structure)

        # Find host element sites for doping
        host_el = cfg.material.doping.host_element
        mn_indices = [i for i, site in enumerate(structure)
                      if str(site.specie) == host_el]
        n_mn = len(mn_indices)
        n_ce = max(1, round(Ce_fraction * n_mn)) if Ce_fraction > 0 else 0
        n_ce = min(n_ce, n_mn)

        # Random substitution (deterministic seed for reproducibility)
        rng = np.random.default_rng(seed=cfg.material.doping.seed)
        ce_indices = rng.choice(mn_indices, size=n_ce, replace=False)

        # Apply substitution — use first allowed dopant
        dopant = cfg.material.doping.allowed_dopants[0] if cfg.material.doping.allowed_dopants else "Ce"
        species = [str(s.specie) for s in structure]
        for idx in ce_indices:
            species[idx] = dopant

        new_structure = Structure(structure.lattice,
                                   species,
                                   [s.frac_coords for s in structure])

        # Save outputs
        out_dir = Path(cfg.directories.output)
        out_dir.mkdir(exist_ok=True)
        ce_label = f"Ce{int(Ce_fraction*100)}pct"
        sc_label  = f"{sc[0]}x{sc[1]}x{sc[2]}"
        cif_path  = str(out_dir / f"LMO_{ce_label}_{sc_label}.cif")
        vasp_path = str(out_dir / f"LMO_{ce_label}_{sc_label}.vasp")
        new_structure.to(filename=cif_path,  fmt="cif")
        new_structure.to(filename=vasp_path, fmt="poscar")

        actual_ce_frac = n_ce / n_mn if n_mn > 0 else 0

        return {
            "status":              "success",
            "supercell_dims":      sc,
            "n_atoms_total":       n_total,
            "n_Li":  sum(1 for s in new_structure if str(s.specie)=="Li"),
            "n_Mn":  sum(1 for s in new_structure if str(s.specie)=="Mn"),
            "n_Ce":  n_ce,
            "n_O":   sum(1 for s in new_structure if str(s.specie)=="O"),
            "actual_Ce_fraction":  round(actual_ce_frac, 4),
            "actual_Ce_pct":       round(actual_ce_frac*100, 2),
            "target_Ce_fraction":  Ce_fraction,
            "lattice_a_used_A":    round(measured_lattice_a, 4),
            "output_cif":          cif_path,
            "output_vasp":         vasp_path,
            "note": f"Random seed=42 for reproducibility. Ce at 16d Mn sites only."
        }
    except Exception as e:
        return {"error": str(e)}


def compute_xrd_r_factor(structure_path: str,
                          two_theta_exp: list,
                          intensity_exp: list,
                          wavelength_A: float = None) -> dict:
    """
    Compute XRD R-factor between built structure and experimental pattern.
    R = Σ|I_obs - I_calc| / Σ|I_obs|
    Also returns simulated peak positions for comparison.
    """
    fp = Path(structure_path)
    if not fp.exists():
        return {"error": f"Structure file not found: {structure_path}"}

    lam = wavelength_A if wavelength_A is not None else cfg.xrd.wavelength_A
    ang_lo, ang_hi = cfg.xrd.angular_range_deg

    try:
        structure = Structure.from_file(str(fp))
        calc = XRDCalculator(wavelength="CuKa")
        pattern_sim = calc.get_pattern(structure, two_theta_range=(ang_lo + 5, ang_hi - 10))

        tth_exp = np.array(two_theta_exp)
        I_exp   = np.array(intensity_exp)

        I_sim = np.interp(tth_exp, pattern_sim.x, pattern_sim.y, left=0, right=0)

        # Normalize to max=100
        I_sim_n = 100 * I_sim / (I_sim.max() + 1e-10)
        I_exp_n = 100 * I_exp / (I_exp.max() + 1e-10)

        R = float(np.sum(np.abs(I_exp_n - I_sim_n)) / (np.sum(np.abs(I_exp_n)) + 1e-10))

        # Top simulated peaks
        top_idx = np.argsort(pattern_sim.y)[::-1][:6]
        top_peaks = [{"2theta": round(float(pattern_sim.x[i]),3),
                      "hkl": str(pattern_sim.hkls[i]),
                      "intensity": round(float(pattern_sim.y[i]),1)}
                     for i in top_idx]

        return {
            "R_factor":      round(R, 4),
            "R_factor_pct":  round(R*100, 2),
            "quality":       cfg.xrd.r_factor.quality_label(R),
            "n_atoms":       len(structure),
            "top_sim_peaks": top_peaks,
            "interpretation": "R-factor measures agreement between simulated and experimental XRD"
        }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# TOOL REGISTRY + JSON SCHEMAS
# =============================================================================

TOOL_REGISTRY = {
    "parse_xrd_file":               parse_xrd_file,
    "analyze_xrd_peaks":            analyze_xrd_peaks,
    "parse_raman_file":             parse_raman_file,
    "analyze_raman_peaks":          analyze_raman_peaks,
    "parse_afm_file":               parse_afm_file,
    "analyze_afm_data":             analyze_afm_data,
    "query_materials_project":      query_materials_project,
    "build_doped_supercell":        build_doped_supercell,
    "compute_xrd_r_factor":         compute_xrd_r_factor,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "parse_xrd_file",
            "description": "Parse XRD data file (Rigaku Excel or plain text). Call this first for any XRD input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath":   {"type": "string"},
                    "sheet_name": {"type": "string", "description": "Sheet name for Excel files"}
                },
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_xrd_peaks",
            "description": "Assign hkl indices, extract lattice parameter (Nelson-Riley), crystallite size (Scherrer), check for CeO2 secondary phase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "two_theta":    {"type": "array", "items": {"type": "number"}},
                    "intensity":    {"type": "array", "items": {"type": "number"}},
                    "wavelength_A": {"type": "number", "default": 1.54059}
                },
                "required": ["two_theta", "intensity"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "parse_raman_file",
            "description": "Parse Raman spectrum file (.txt, .csv, .xlsx). Returns wavenumber and intensity arrays.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath":   {"type": "string"},
                    "sheet_name": {"type": "string"}
                },
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_raman_peaks",
            "description": "Assign LiMn2O4 Raman modes, extract A1g position and FWHM, compute disorder ratio, check for CeO2.",
            "parameters": {
                "type": "object",
                "properties": {
                    "wavenumber": {"type": "array", "items": {"type": "number"}},
                    "intensity":  {"type": "array", "items": {"type": "number"}}
                },
                "required": ["wavenumber", "intensity"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "parse_afm_file",
            "description": "Parse AFM height data file (.txt or .csv height matrix).",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"}
                },
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_afm_data",
            "description": "Compute surface roughness (Ra, Rq) and grain size from AFM height data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "z_matrix":       {"type": "array", "items": {"type": "number"}},
                    "pixel_size_nm":  {"type": "number", "description": "nm per pixel"}
                },
                "required": ["z_matrix"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_materials_project",
            "description": "Fetch base crystal structure from Materials Project for the configured material.",
            "parameters": {
                "type": "object",
                "properties": {
                    "formula":  {"type": "string", "description": "Chemical formula (default: from config)"},
                    "mp_id":    {"type": "string", "description": "Materials Project ID (default: from config)"},
                    "api_key":  {"type": "string", "description": "MP API key (optional if env set)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "build_doped_supercell",
            "description": "Build doped supercell from MP structure. Scales lattice to measured value, substitutes dopant at host sites.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_cif_path":       {"type": "string", "description": "CIF from query_materials_project"},
                    "measured_lattice_a":  {"type": "number", "description": "Your measured lattice parameter in Angstroms"},
                    "Ce_fraction":         {"type": "number", "description": "Ce fraction of Mn sites (e.g. 0.02)"},
                    "crystallite_size_nm": {"type": "number", "description": "From Scherrer - sets supercell size"},
                    "supercell_hint":      {"type": "array",  "items": {"type": "integer"}}
                },
                "required": ["base_cif_path", "measured_lattice_a", "Ce_fraction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compute_xrd_r_factor",
            "description": "Validate built structure against experimental XRD by computing R-factor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "structure_path":  {"type": "string"},
                    "two_theta_exp":   {"type": "array", "items": {"type": "number"}},
                    "intensity_exp":   {"type": "array", "items": {"type": "number"}},
                    "wavelength_A":    {"type": "number", "default": 1.54059}
                },
                "required": ["structure_path", "two_theta_exp", "intensity_exp"]
            }
        }
    }
]


# =============================================================================
# AGENTIC LOOP (raw tool calling — no LangChain)
# =============================================================================

def run_agent(user_request: str,
              client=None,
              model: str = None,
              verbose: bool = True,
              max_iterations: int = None,
              task: str = "general_reasoning") -> str:

    # Use model registry if client not provided
    if client is None:
        llm_client, model = registry.get_client_for_task(task)
    else:
        llm_client = client

    max_iter = max_iterations or cfg.llm.max_iterations

    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_request}
    ]

    for iteration in range(max_iter):
        # Use unified client interface if available, else raw client
        if hasattr(llm_client, 'chat_completion'):
            response = llm_client.chat_completion(
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        else:
            response = llm_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=cfg.llm.temperature,
                max_tokens=cfg.llm.max_tokens,
            )

        msg = response.choices[0].message

        # Build serializable assistant message
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if msg.content and verbose:
            print(f"\n[Agent] {msg.content[:400]}")

        # No tool calls = final answer
        if not msg.tool_calls:
            return msg.content

        # Execute all tool calls
        for tc in msg.tool_calls:
            fname = tc.function.name
            try:
                fargs = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fargs = {}

            if verbose:
                # Print args without large arrays
                safe_args = {k: (f"[array len={len(v)}]" if isinstance(v, list) and len(v) > 10 else v)
                             for k, v in fargs.items()}
                print(f"\n>>> {fname}({json.dumps(safe_args, indent=2)[:300]})")

            if fname in TOOL_REGISTRY:
                try:
                    result = TOOL_REGISTRY[fname](**fargs)
                except Exception as e:
                    result = {"error": str(e), "tool": fname}
            else:
                result = {"error": f"Unknown tool: {fname}"}

            # Truncate large arrays before sending back to LLM
            result_for_llm = {}
            for k, v in result.items():
                if isinstance(v, list) and len(v) > 15:
                    result_for_llm[k] = v[:5] + ["...(truncated)..."] + v[-3:]
                else:
                    result_for_llm[k] = v

            if verbose:
                print(f"    => {json.dumps(result_for_llm, indent=2)[:500]}")

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      json.dumps(result_for_llm)
            })

    return "Max iterations reached."


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AtomInfer v2 - XRD/Raman/AFM → Atomistic Model")
    parser.add_argument("--xrd",      type=str, help="XRD file path")
    parser.add_argument("--sheet",    type=str, default=None, help="Excel sheet name")
    parser.add_argument("--raman",    type=str, default=None, help="Raman file path")
    parser.add_argument("--afm",      type=str, default=None, help="AFM file path")
    parser.add_argument("--demo",     action="store_true", help="Run all sheets in demo mode")
    parser.add_argument("--config",   type=str, default=None, help="Path to config.toml")
    parser.add_argument("--verbose",  action="store_true", default=True)
    args = parser.parse_args()

    # Refresh model availability and select
    registry.refresh_availability()
    client, model = get_llm_client(task="general_reasoning")

    if args.demo and args.xrd:
        xl = pd.ExcelFile(args.xrd)
        sheets = xl.sheet_names
        print(f"Demo mode: processing sheets {sheets}")
        for sheet in sheets:
            print(f"\n{'='*60}\nSheet: {sheet}\n{'='*60}")
            request = (
                f"Analyze the XRD data in file '{args.xrd}', sheet '{sheet}'. "
                f"Identify the phase, extract all structural parameters, fetch the "
                f"base structure from Materials Project, build a Ce-doped atomistic "
                f"model, and validate it with the R-factor."
            )
            result = run_agent(request, client, model, verbose=args.verbose)
            print(f"\nFINAL OUTPUT:\n{result}\n")

    elif args.xrd:
        # Build request from provided files
        parts = [f"Analyze the XRD data in file '{args.xrd}'"]
        if args.sheet:
            parts[0] += f", sheet '{args.sheet}'"
        if args.raman:
            parts.append(f"Raman data in '{args.raman}'")
        if args.afm:
            parts.append(f"AFM data in '{args.afm}'")

        techniques = " and ".join(parts)
        request = (
            f"{techniques}. "
            f"Identify the phase, extract all structural parameters from each technique, "
            f"cross-check consistency between techniques, fetch the base structure from "
            f"Materials Project (mp-19017), build a Ce-doped atomistic model using your "
            f"measured lattice parameter and estimated Ce fraction, and validate with XRD R-factor. "
            f"Produce output files ready for molecular dynamics simulation."
        )
        result = run_agent(request, client, model, verbose=args.verbose)
        print(f"\n{'='*60}\nFINAL OUTPUT:\n{result}")

    else:
        print("Usage: python atomInfer_v2.py --xrd YOUR_FILE.xlsx --sheet 'Sheet Name'")
        print("       python atomInfer_v2.py --demo --xrd YOUR_FILE.xlsx")
        print("       python atomInfer_v2.py --xrd file.xlsx --raman raman.txt --afm afm.txt")
        print("\nConfiguration:")
        print("  Edit config.toml (copy from config.default.toml)")
        print("  Set LLM models, material profiles, and API keys")
        print("\nEnvironment variables (optional, override config.toml):")
        print("  GROQ_API_KEY    — from console.groq.com")
        print("  OPENAI_API_KEY  — from platform.openai.com")
        print("  MP_API_KEY      — from materialsproject.org")
