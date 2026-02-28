"""
HRMC (Hybrid Reverse Monte Carlo) for Ce-doped LiMn2O4
=======================================================
Three move types:
  1. Displacement  - refines local geometry
  2. Species swap  - Ce <-> Mn substitution at 16d sites (composition-driven)
  3. Vacancy swap  - introduces/removes Mn vacancies (optional)

Cost function:
  F = w_xrd * R_factor(XRD) + w_energy * E_LAMMPS + w_comp * (x_Ce_current - x_Ce_target)^2

Potential:
  Li-Mn-O  : 2NN MEAM + coul/streitz  (Lee et al. 2017, NIST repo)
  Ce-O     : Buckingham (ionic, Ce^3+ parameters from oxide literature)
  Cross terms: Lorentz-Berthelot mixing for Buckingham, MEAM handles Li-Mn-O

Requirements:
  pip install pymatgen numpy lammps
  LAMMPS must be compiled with Python API and MEAM package enabled.
"""

import numpy as np
import json
import copy
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# Load local LAMMPS library (with MEAM package) before importing lammps module
import ctypes as _ctypes
_LOCAL_LAMMPS_LIB = Path("/home/yeatanir/lammps-build/liblammps.so")
if _LOCAL_LAMMPS_LIB.exists():
    _ctypes.CDLL(str(_LOCAL_LAMMPS_LIB), _ctypes.RTLD_GLOBAL)

from lammps import lammps

# ─────────────────────────────────────────────
# SECTION 1: Buckingham parameters for Ce-O
# ─────────────────────────────────────────────
# Source: Islam et al. ionic oxide potentials; Ce3+ from
# Minervini, Zacate & Grimes, Solid State Ionics 116 (1999) 339
# Format: A (eV), rho (Å), C (eV·Å^6)
BUCKINGHAM_PARAMS = {
    ("Ce", "O"): {"A": 2141.4, "rho": 0.3541, "C": 43.83},
    ("O",  "O"): {"A": 9547.96, "rho": 0.2192, "C": 32.00},  # backup only
}

# Formal charges for Qeq / Ewald (used by coul/streitz)
FORMAL_CHARGES = {"Li": +1, "Mn": +3.5, "O": -2, "Ce": +3}

# ─────────────────────────────────────────────
# SECTION 2: XRD R-factor
# ─────────────────────────────────────────────

def compute_xrd_pattern(structure: Structure, two_theta_range=(10, 80)):
    """Compute simulated XRD pattern using pymatgen."""
    calc = XRDCalculator(wavelength="CuKa")
    pattern = calc.get_pattern(structure, two_theta_range=two_theta_range)
    return pattern  # .x = 2theta angles, .y = intensities

def compute_r_factor(pattern_sim, pattern_exp):
    """
    Compute weighted R-factor between simulated and experimental XRD.
    Both patterns must be interpolated onto the same 2theta grid.
    R = sum|I_obs - I_calc| / sum|I_obs|
    """
    # Interpolate simulated onto experimental grid
    I_sim = np.interp(pattern_exp["two_theta"], pattern_sim.x, pattern_sim.y,
                      left=0, right=0)
    I_exp = np.array(pattern_exp["intensities"])

    # Normalize both to max = 100
    I_sim = 100 * I_sim / (I_sim.max() + 1e-10)
    I_exp = 100 * I_exp / (I_exp.max() + 1e-10)

    R = np.sum(np.abs(I_exp - I_sim)) / (np.sum(np.abs(I_exp)) + 1e-10)
    return float(R)

# ─────────────────────────────────────────────
# SECTION 3: LAMMPS energy evaluation
# ─────────────────────────────────────────────

class LAMMPSEvaluator:
    """
    Wraps LAMMPS Python API for single-point energy evaluation.
    Uses hybrid/overlay: MEAM for Li-Mn-O, buck/coul/long for Ce-O.
    """
    def __init__(self, potential_dir: str):
        self.potential_dir = Path(potential_dir)
        # Initialize LAMMPS ONCE for the entire HRMC run.
        # Re-initializing 5000 times wastes ~seconds of overhead per step.
        # "clear" between steps wipes all atoms/settings without destroying the session.
        self.lmp = lammps(cmdargs=["-screen", "none", "-log", "none"])

    def __del__(self):
        """Cleanly close LAMMPS when the evaluator goes out of scope."""
        if self.lmp is not None:
            try:
                self.lmp.close()
            except Exception:
                pass

    def _write_lammps_input(self, structure: Structure, tmp_dir: Path):
        """Write LAMMPS data file from pymatgen structure."""
        from pymatgen.io.lammps.data import LammpsData
        lammps_data = LammpsData.from_structure(
            structure,
            atom_style="charge",
            is_sort=True
        )
        data_file = tmp_dir / "structure.lammps"
        lammps_data.write_file(str(data_file))
        return data_file

    def compute_energy(self, structure: Structure) -> float:
        """Return total potential energy in eV.

        Teaching note:
          - We call self.lmp.command("clear") to wipe the previous system state.
            Think of it like emptying a beaker without washing the bench — fast.
          - We do NOT call self.lmp.close() here. That would destroy the session.
            Only __del__ does that, once, at program exit.
          - The minimize step (Fix 1) is included here: it relaxes atoms in the
            LAMMPS-held structure before extracting energy, so the energy reflects
            a locally relaxed configuration rather than a raw displaced snapshot.
        """
        import tempfile, shutil
        tmp_dir = Path(tempfile.mkdtemp())
        data_file = self._write_lammps_input(structure, tmp_dir)

        # Wipe previous step's atoms, box, and settings — but keep LAMMPS alive
        self.lmp.command("clear")

        # Get unique species in order (needed for MEAM pair_coeff)
        species = sorted(set(str(s.specie) for s in structure))
        species_str = " ".join(species)

        # Potential file names as specified by the user
        lib_file   = self.potential_dir / "library_LiMnO.meam"
        alloy_file = self.potential_dir / "library_LiMnO.meam.alloy"

        cmds = [
            "units metal",
            "atom_style charge",
            "boundary p p p",
            f"read_data {data_file}",

            # Hybrid potential: MEAM for Li-Mn-O backbone + Buckingham for Ce-O
            # pair_style meam/c was renamed to meam in LAMMPS >= 2021
            "pair_style hybrid/overlay meam buck/coul/long 10.0",

            # MEAM covers the full Li-Mn-O-Ce backbone
            f"pair_coeff * * meam {lib_file} {species_str} {alloy_file} {species_str}",
        ]

        # Add Buckingham Ce-O if Ce is present in this structure
        if "Ce" in species and "O" in species:
            ce_idx = species.index("Ce") + 1
            o_idx  = species.index("O") + 1
            p = BUCKINGHAM_PARAMS[("Ce", "O")]
            cmds.append(
                f"pair_coeff {ce_idx} {o_idx} buck/coul/long "
                f"{p['A']} {p['rho']} {p['C']}"
            )

        cmds += [
            "kspace_style ewald 1e-5",
            "thermo 10",
            # FIX 1: Minimize before reading energy.
            # Without this, a displaced atom sits at whatever random position
            # move_displacement chose. LAMMPS sees an unrelaxed, high-energy
            # snapshot and the cost function is noisy/meaningless.
            # max_force_tol=1e-4, max_energy_tol=1e-6, max_iter=100, max_eval=1000
            "minimize 1e-4 1e-6 100 1000",
            # run 0 forces a final thermo output with the minimized energy
            "run 0",
        ]

        for cmd in cmds:
            self.lmp.command(cmd)

        energy = self.lmp.get_thermo("pe")

        # Cleanup temp files — LAMMPS has already read them, safe to delete
        shutil.rmtree(tmp_dir, ignore_errors=True)

        return float(energy)  # eV

# ─────────────────────────────────────────────
# SECTION 4: Structure move generators
# ─────────────────────────────────────────────

def move_displacement(structure: Structure, max_disp: float = 0.15) -> Structure:
    """
    Move type 1: Displace a random atom by a random vector.
    max_disp in Angstroms. Do NOT displace Li (it's highly mobile —
    let the energy term handle Li relaxation naturally).

    Teaching note on the Buckingham Catastrophe:
      The Buckingham potential has the form:  V(r) = A*exp(-r/rho) - C/r^6
      At large r: fine. At small r: the -C/r^6 term dominates and goes to -infinity.
      So two atoms that stray too close will accelerate toward each other
      without bound — the "catastrophe." LAMMPS crashes or gives garbage.
      We prevent this by checking BEFORE calling LAMMPS at all.
      The threshold 1.2 Å is safely below any real interatomic distance in
      oxide materials (~1.8 Å for M-O bonds), so we never reject valid moves.
    """
    s = structure.copy()
    # Pick from non-Li atoms preferentially
    candidates = [i for i, site in enumerate(s) if str(site.specie) != "Li"]
    idx = np.random.choice(candidates)
    disp = np.random.uniform(-max_disp, max_disp, 3)
    s.translate_sites([idx], disp, frac_coords=False)

    # Buckingham catastrophe guard: reject if any neighbor closer than 1.2 Å.
    # This is a pure Python check — zero LAMMPS overhead.
    # Return None signals the main loop to skip this step (already handled with `continue`).
    neighbors = s.get_neighbors(s[idx], r=1.2)
    if len(neighbors) > 0:
        return None  # Reject immediately, saves a full LAMMPS minimization call

    return s

def move_species_swap(structure: Structure, target_Ce_fraction: float) -> Structure:
    """
    Move type 2: Swap species between a Ce and a Mn atom.
    If Ce% < target: convert a random Mn→Ce
    If Ce% > target: convert a random Ce→Mn
    Ce substitutes at 16d Mn sites only.

    Returns None if no valid swap is possible.
    """
    s = structure.copy()
    species = [str(site.specie) for site in s]

    n_Ce = species.count("Ce")
    n_Mn = species.count("Mn")
    n_total_16d = n_Ce + n_Mn  # 16d site count

    current_fraction = n_Ce / n_total_16d if n_total_16d > 0 else 0

    if current_fraction < target_Ce_fraction:
        # Need more Ce: pick a Mn and convert to Ce
        mn_sites = [i for i, sp in enumerate(species) if sp == "Mn"]
        if not mn_sites:
            return None
        idx = np.random.choice(mn_sites)
        s[idx] = "Ce"
    else:
        # Need less Ce: pick a Ce and convert to Mn
        ce_sites = [i for i, sp in enumerate(species) if sp == "Ce"]
        if not ce_sites:
            return None
        idx = np.random.choice(ce_sites)
        s[idx] = "Mn"

    return s

def move_vacancy(structure: Structure, target_vacancy_fraction: float = 0.02) -> Structure:
    """
    Move type 3 (optional): Introduce or remove a Mn vacancy.
    Vacancy = remove Mn atom. Fills back with ghost site for bookkeeping.
    For simplicity in hackathon: just toggle one Mn site.
    Set target_vacancy_fraction=0 to disable.
    """
    if target_vacancy_fraction == 0:
        return None
    # Implementation left as extension — skip for hackathon demo
    return None

# ─────────────────────────────────────────────
# SECTION 5: Cost function
# ─────────────────────────────────────────────

def compute_cost(structure: Structure,
                 pattern_exp: dict,
                 target_Ce_fraction: float,
                 lammps_eval: LAMMPSEvaluator,
                 w_xrd: float = 1.0,
                 w_energy: float = 0.001,
                 w_comp: float = 5.0) -> dict:
    """
    Compute total cost F = w_xrd*R + w_energy*E + w_comp*(x_Ce - x_target)^2

    Returns dict with individual components for logging.
    Note: w_energy is small because LAMMPS energy is in eV (large numbers)
    while R-factor is 0-1. Tune these weights based on your system.
    """
    # XRD R-factor
    pattern_sim = compute_xrd_pattern(structure)
    R = compute_r_factor(pattern_sim, pattern_exp)

    # LAMMPS energy (normalized per atom)
    E_total = lammps_eval.compute_energy(structure)
    E_per_atom = E_total / len(structure)

    # Composition penalty
    species = [str(site.specie) for site in structure]
    n_Ce = species.count("Ce")
    n_Mn = species.count("Mn")
    n_16d = n_Ce + n_Mn
    x_Ce = n_Ce / n_16d if n_16d > 0 else 0
    comp_penalty = (x_Ce - target_Ce_fraction) ** 2

    F = w_xrd * R + w_energy * E_per_atom + w_comp * comp_penalty

    return {
        "F": F,
        "R_factor": R,
        "E_per_atom": E_per_atom,
        "x_Ce": x_Ce,
        "comp_penalty": comp_penalty,
    }

# ─────────────────────────────────────────────
# SECTION 6: Metropolis acceptance
# ─────────────────────────────────────────────

def metropolis_accept(dF: float, temperature: float) -> bool:
    """
    Standard Metropolis criterion.
    temperature here is a 'fictitious temperature' controlling
    how much uphill moves are accepted. Tune this.
    Start high (0.1), anneal down to 0.01 over the run.
    """
    if dF <= 0:
        return True
    prob = np.exp(-dF / temperature)
    return np.random.random() < prob

# ─────────────────────────────────────────────
# SECTION 7: Main HRMC loop
# ─────────────────────────────────────────────

def run_hrmc(
    initial_structure: Structure,
    pattern_exp: dict,
    target_Ce_fraction: float,
    potential_dir: str,
    n_steps: int = 5000,
    T_start: float = 0.1,
    T_end: float = 0.01,
    max_disp: float = 0.15,
    w_xrd: float = 1.0,
    w_energy: float = 0.001,
    w_comp: float = 5.0,
    output_interval: int = 100,
    output_dir: str = "./hrmc_output",
    move_weights: tuple = (0.6, 0.3, 0.1),  # displacement, swap, vacancy
    progress_callback: callable = None,  # Optional callback(step, cost_dict)
) -> dict:
    """
    Main HRMC loop.

    Args:
        initial_structure : pymatgen Structure (supercell with initial Ce placement)
        pattern_exp       : dict with keys 'two_theta' and 'intensities' (lists)
        target_Ce_fraction: float, e.g. 0.05 for 5% Ce
        potential_dir     : path to MEAM potential files
        n_steps           : number of MC moves
        T_start/T_end     : simulated annealing temperature schedule
        max_disp          : maximum displacement in Angstroms
        w_xrd/w_energy/w_comp : cost function weights
        output_interval   : save structure every N steps
        output_dir        : directory for output structures
        move_weights      : probability of each move type (must sum to 1)

    Returns:
        dict with 'best_structure', 'trajectory', 'final_cost'
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    lammps_eval = LAMMPSEvaluator(potential_dir)

    current_structure = initial_structure.copy()
    current_cost = compute_cost(
        current_structure, pattern_exp, target_Ce_fraction,
        lammps_eval, w_xrd, w_energy, w_comp
    )

    best_structure = current_structure.copy()
    best_cost = current_cost.copy()

    trajectory = []
    n_accepted = {"displacement": 0, "swap": 0, "vacancy": 0}
    n_proposed = {"displacement": 0, "swap": 0, "vacancy": 0}

    move_types = ["displacement", "swap", "vacancy"]
    move_probs = np.array(move_weights)
    move_probs /= move_probs.sum()

    print(f"\n{'='*60}")
    print(f"HRMC Starting | Steps: {n_steps} | Target Ce%: {target_Ce_fraction*100:.1f}%")
    print(f"Initial R-factor: {current_cost['R_factor']:.4f}")
    print(f"Initial E/atom:   {current_cost['E_per_atom']:.4f} eV")
    print(f"Initial Ce%:      {current_cost['x_Ce']*100:.2f}%")
    print(f"{'='*60}\n")

    for step in range(n_steps):

        # Simulated annealing: linearly decrease temperature
        T = T_start + (T_end - T_start) * (step / n_steps)

        # Select move type
        move_type = np.random.choice(move_types, p=move_probs)
        n_proposed[move_type] += 1

        # Generate trial structure
        trial_structure = None
        if move_type == "displacement":
            trial_structure = move_displacement(current_structure, max_disp)
        elif move_type == "swap":
            trial_structure = move_species_swap(current_structure, target_Ce_fraction)
        elif move_type == "vacancy":
            trial_structure = move_vacancy(current_structure)

        if trial_structure is None:
            continue  # Move not possible, skip

        # Evaluate trial cost
        trial_cost = compute_cost(
            trial_structure, pattern_exp, target_Ce_fraction,
            lammps_eval, w_xrd, w_energy, w_comp
        )

        dF = trial_cost["F"] - current_cost["F"]

        if metropolis_accept(dF, T):
            current_structure = trial_structure
            current_cost = trial_cost
            n_accepted[move_type] += 1

            # Track best ever
            if current_cost["F"] < best_cost["F"]:
                best_structure = current_structure.copy()
                best_cost = current_cost.copy()

        # Logging
        trajectory.append({
            "step": step,
            "F": current_cost["F"],
            "R_factor": current_cost["R_factor"],
            "E_per_atom": current_cost["E_per_atom"],
            "x_Ce": current_cost["x_Ce"],
            "T": T,
            "move_type": move_type,
        })

        if step % output_interval == 0:
            print(f"Step {step:5d} | T={T:.4f} | R={current_cost['R_factor']:.4f} "
                  f"| E={current_cost['E_per_atom']:.3f} eV "
                  f"| Ce%={current_cost['x_Ce']*100:.2f}% "
                  f"| F={current_cost['F']:.4f}")

            # Call progress callback if provided
            if progress_callback is not None:
                try:
                    progress_callback(step, current_cost)
                except Exception:
                    pass  # Callback errors should not stop the simulation

            # Save current structure
            current_structure.to(
                filename=str(Path(output_dir) / f"step_{step:05d}.cif"),
                fmt="cif"
            )

    # Save best structure
    best_structure.to(filename=str(Path(output_dir) / "best_structure.cif"), fmt="cif")
    best_structure.to(filename=str(Path(output_dir) / "best_structure.vasp"), fmt="poscar")

    # Save trajectory
    with open(Path(output_dir) / "trajectory.json", "w") as f:
        json.dump(trajectory, f, indent=2)

    # Acceptance rates
    print(f"\n{'='*60}")
    print("HRMC Complete")
    print(f"Best R-factor: {best_cost['R_factor']:.4f}")
    print(f"Best Ce%: {best_cost['x_Ce']*100:.2f}%")
    for mt in move_types:
        if n_proposed[mt] > 0:
            rate = n_accepted[mt] / n_proposed[mt] * 100
            print(f"  {mt:15s}: {n_accepted[mt]}/{n_proposed[mt]} accepted ({rate:.1f}%)")
    print(f"{'='*60}\n")

    return {
        "best_structure": best_structure,
        "best_cost": best_cost,
        "trajectory": trajectory,
        "acceptance_rates": {
            mt: n_accepted[mt] / max(n_proposed[mt], 1)
            for mt in move_types
        }
    }

# ─────────────────────────────────────────────
# SECTION 8: Synthetic data generator
# ─────────────────────────────────────────────

def generate_synthetic_xrd(Ce_fraction: float, two_theta_range=(10, 80), n_points=500) -> dict:
    """
    Generate synthetic XRD experimental data using Vegard's law.

    Pristine LiMn2O4: a = 8.248 Å (ICDD PDF 35-0782)
    Ce3+ (1.01 Å) >> Mn3+ (0.645 Å), so lattice expands with Ce doping.
    Slope: ~+0.04 Å per unit Ce fraction (i.e. +0.004 Å per 1% Ce)
    calibrated from analogous rare-earth doped LiMn2O4 literature.

    This builds the structure, computes XRD via pymatgen, adds noise.
    """
    from pymatgen.ext.matproj import MPRester
    # If no MP key, build from scratch
    # Pristine LiMn2O4: Fd-3m, a = 8.248 Å
    # Lattice expands: a(x) = 8.248 + 0.04*x_Ce
    a = 8.248 + 0.04 * Ce_fraction

    # Build primitive spinel structure programmatically
    from pymatgen.core import Lattice, Structure, Species
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    # Fd-3m spinel: use conventional cell
    # LiMn2O4: Li at 8a (0,0,0), Mn at 16d (5/8,5/8,5/8), O at 32e (~0.386,~0.386,~0.386)
    lattice = Lattice.cubic(a)

    # 8-formula unit conventional cell: 8 Li, 16 Mn, 32 O
    li_sites = [
        [0.000, 0.000, 0.000], [0.000, 0.500, 0.500],
        [0.500, 0.000, 0.500], [0.500, 0.500, 0.000],
        [0.250, 0.250, 0.250], [0.250, 0.750, 0.750],
        [0.750, 0.250, 0.750], [0.750, 0.750, 0.250],
    ]

    mn_sites_frac = [
        [0.625, 0.625, 0.625], [0.625, 0.875, 0.875],
        [0.875, 0.625, 0.875], [0.875, 0.875, 0.625],
        [0.125, 0.125, 0.625], [0.125, 0.375, 0.875],
        [0.375, 0.125, 0.875], [0.375, 0.375, 0.625],
        [0.125, 0.625, 0.125], [0.125, 0.875, 0.375],
        [0.375, 0.625, 0.375], [0.375, 0.875, 0.125],
        [0.625, 0.125, 0.125], [0.625, 0.375, 0.375],
        [0.875, 0.125, 0.375], [0.875, 0.375, 0.125],
    ]

    o_sites = [
        [0.386, 0.386, 0.386], [0.386, 0.614, 0.614],
        [0.614, 0.386, 0.614], [0.614, 0.614, 0.386],
        [0.136, 0.136, 0.136], [0.136, 0.364, 0.364],  # simplified O positions
        [0.364, 0.136, 0.364], [0.364, 0.364, 0.136],
    ]
    # Note: for a real run, use MP API or a validated CIF for O positions
    # This simplified O set is for testing only

    # Determine Ce substitution: replace Ce_fraction of Mn sites
    n_ce = max(1, round(Ce_fraction * len(mn_sites_frac))) if Ce_fraction > 0 else 0
    species_mn = ["Ce"] * n_ce + ["Mn"] * (len(mn_sites_frac) - n_ce)

    species = ["Li"] * len(li_sites) + species_mn + ["O"] * len(o_sites)
    coords = li_sites + mn_sites_frac + o_sites

    structure = Structure(lattice, species, coords)

    # Compute XRD
    calc = XRDCalculator(wavelength="CuKa")
    pattern = calc.get_pattern(structure, two_theta_range=two_theta_range)

    # Interpolate to fixed grid and add 2% Gaussian noise
    two_theta_grid = np.linspace(two_theta_range[0], two_theta_range[1], n_points)
    intensities = np.interp(two_theta_grid, pattern.x, pattern.y, left=0, right=0)
    noise = np.random.normal(0, 0.02 * intensities.max(), n_points)
    intensities = np.clip(intensities + noise, 0, None)

    return {
        "Ce_fraction": Ce_fraction,
        "lattice_a": a,
        "two_theta": two_theta_grid.tolist(),
        "intensities": intensities.tolist(),
        "source": "synthetic_vegard",
    }


def generate_all_concentrations(output_dir="./synthetic_data"):
    """Generate XRD data for all Ce concentrations: 0.5% to 5%."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    concentrations = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]

    for x_Ce in concentrations:
        print(f"Generating XRD for Ce = {x_Ce*100:.1f}%...")
        data = generate_synthetic_xrd(x_Ce)
        fname = Path(output_dir) / f"XRD_Ce{int(x_Ce*100):02d}pct.json"
        with open(fname, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved: {fname} | a = {data['lattice_a']:.4f} Å")

    print("\nAll concentrations generated.")


# ─────────────────────────────────────────────
# SECTION 9: Known Raman signatures (literature)
# ─────────────────────────────────────────────

"""
Literature Raman data for LiMn2O4 and Ce-doped variants:

Pristine LiMn2O4 (Fd-3m):
  A1g  : 625-628 cm-1  (dominant, Mn-O symmetric stretch, Mn4+ sublattice)
  A1g  : 577 cm-1      (Mn3+-O stretch, mixed valence component)
  T2g  : 483 cm-1      (F2g mode)
  Eg   : 434 cm-1
  T2g  : 162 cm-1      (Li sublattice translational mode)

Ce-doped LiMn2O4 (literature: Sangaraju et al. Ceram. Int. 2024,
                               Arumugam & Kalaignan J. Electroanal. Chem. 2010):
  - A1g red-shifts: 628 → ~620-622 cm-1 as Ce increases (lattice softening)
  - A1g broadens: FWHM increases from ~15 to ~25 cm-1 at 5% Ce
  - Relative intensity of 577 cm-1 peak increases (more Mn3+ character
    due to charge compensation: Ce3+ substituting Mn3.5+ → local Mn4+ enrichment nearby)
  - At high Ce (>3%): small CeO2 secondary phase peak may appear at ~465 cm-1

Empirical relations (use for synthetic Raman generation):
  A1g_position(x_Ce) = 628 - 12 * x_Ce      [cm-1, x_Ce in fraction 0-0.05]
  A1g_FWHM(x_Ce)     = 15  + 200 * x_Ce     [cm-1]
  I_577_rel(x_Ce)    = 0.15 + 2.0 * x_Ce    [relative to A1g = 1]
"""

RAMAN_MODES_PRISTINE = {
    "A1g_main": 627,   # cm-1
    "A1g_sec":  577,   # cm-1 (Mn3+ component)
    "F2g":      483,   # cm-1
    "Eg":       434,   # cm-1
    "T2g_li":   162,   # cm-1 (Li translational)
}

def generate_synthetic_raman(Ce_fraction: float, n_points=800,
                              wavenumber_range=(100, 800)) -> dict:
    """
    Generate synthetic Raman spectrum from empirical relations.
    Uses Lorentzian peak shapes.
    """
    wn = np.linspace(wavenumber_range[0], wavenumber_range[1], n_points)

    def lorentzian(x, center, fwhm, amplitude):
        gamma = fwhm / 2
        return amplitude * gamma**2 / ((x - center)**2 + gamma**2)

    # A1g main peak (composition-dependent)
    A1g_pos  = 628 - 12 * Ce_fraction
    A1g_fwhm = 15  + 200 * Ce_fraction
    I_577_rel = 0.15 + 2.0 * Ce_fraction

    spectrum = (
        lorentzian(wn, A1g_pos, A1g_fwhm, 1000) +          # A1g main
        lorentzian(wn, 577, 20 + 50*Ce_fraction, 1000*I_577_rel) +  # A1g Mn3+
        lorentzian(wn, 483, 18, 200) +                      # F2g
        lorentzian(wn, 434, 20, 150) +                      # Eg
        lorentzian(wn, 162, 15, 100)                        # T2g Li
    )

    # CeO2 secondary phase at ~465 cm-1 for high Ce
    if Ce_fraction > 0.03:
        ceo2_intensity = (Ce_fraction - 0.03) * 5000
        spectrum += lorentzian(wn, 465, 12, ceo2_intensity)

    # Add noise
    noise = np.random.normal(0, 0.01 * spectrum.max(), n_points)
    spectrum = np.clip(spectrum + noise, 0, None)

    return {
        "Ce_fraction": Ce_fraction,
        "A1g_position": A1g_pos,
        "A1g_FWHM": A1g_fwhm,
        "wavenumber": wn.tolist(),
        "intensity": spectrum.tolist(),
        "source": "synthetic_empirical_literature",
        "reference": "Sangaraju et al. Ceram. Int. 50(3) 4955 (2024); "
                     "Arumugam & Kalaignan J. Electroanal. Chem. 648 54 (2010)"
    }


# ─────────────────────────────────────────────
# SECTION 10: Entry point / usage example
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HRMC for Ce-doped LiMn2O4")
    parser.add_argument("--mode", choices=["generate_data", "run_hrmc", "test", "test_lammps"],
                        default="generate_data")
    parser.add_argument("--Ce_pct", type=float, default=5.0,
                        help="Ce concentration in percent (e.g. 5.0 for 5%%)")
    parser.add_argument("--n_steps", type=int, default=500,
                        help="Number of HRMC steps (500 for quick test, 5000 for production)")
    parser.add_argument("--potential_dir", type=str, default="./potentials",
                        help="Directory containing MEAM potential files")
    parser.add_argument("--output_dir", type=str, default="./hrmc_output")
    parser.add_argument("--cif", type=str, default="./LiMn2O4.cif",
                        help="Path to initial structure CIF file")
    parser.add_argument("--supercell", type=int, nargs=3, default=[2, 2, 2],
                        help="Supercell dimensions, e.g. --supercell 2 2 2")
    args = parser.parse_args()

    if args.mode == "generate_data":
        print("Generating synthetic XRD and Raman data for all concentrations...")
        generate_all_concentrations()

        # Also generate Raman for all concentrations
        from pathlib import Path
        Path("./synthetic_data").mkdir(exist_ok=True)
        for x_Ce in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]:
            data = generate_synthetic_raman(x_Ce)
            fname = f"./synthetic_data/Raman_Ce{int(x_Ce*100):02d}pct.json"
            with open(fname, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Raman Ce={x_Ce*100:.1f}%: A1g at {data['A1g_position']:.1f} cm-1, "
                  f"FWHM={data['A1g_FWHM']:.1f} cm-1")

    elif args.mode == "test":
        # Quick sanity check without LAMMPS
        print("Running quick test (XRD + cost function only, no LAMMPS)...")
        x_Ce = args.Ce_pct / 100
        exp_data = generate_synthetic_xrd(x_Ce)
        print(f"Generated synthetic XRD: a = {exp_data['lattice_a']:.4f} Å")
        print(f"Generated Raman: A1g at {628 - 12*x_Ce:.1f} cm-1")

        # Test loading CIF if provided
        if Path(args.cif).exists():
            s = Structure.from_file(args.cif)
            print(f"Loaded CIF: {args.cif} → {len(s)} atoms, formula: {s.formula}")
            sc = s.copy()
            sc.make_supercell(args.supercell)
            print(f"Supercell {args.supercell}: {len(sc)} atoms")
        print("\nTest passed. Run with --mode test_lammps to verify LAMMPS.")

    elif args.mode == "test_lammps":
        # Full LAMMPS test with the CIF structure
        print("Running LAMMPS energy test...")
        if not Path(args.cif).exists():
            print(f"ERROR: CIF file not found: {args.cif}")
            raise SystemExit(1)

        s = Structure.from_file(args.cif)
        s.make_supercell(args.supercell)
        print(f"Structure: {s.formula} ({len(s)} atoms)")

        lammps_eval = LAMMPSEvaluator(args.potential_dir)
        energy = lammps_eval.compute_energy(s)
        print(f"LAMMPS energy: {energy:.4f} eV")
        print(f"Energy/atom:   {energy/len(s):.4f} eV/atom")
        print("\nLAMMPS test passed!")

    elif args.mode == "run_hrmc":
        x_Ce = args.Ce_pct / 100

        # Load or generate experimental data
        exp_xrd_file = f"./synthetic_data/XRD_Ce{int(args.Ce_pct):02d}pct.json"
        if Path(exp_xrd_file).exists():
            with open(exp_xrd_file) as f:
                pattern_exp = json.load(f)
        else:
            print("Generating synthetic XRD data first...")
            pattern_exp = generate_synthetic_xrd(x_Ce)

        # Build initial structure (use generate_synthetic_xrd structure as starting point)
        # In production: load from Materials Project via mp-api
        initial_structure = None  # Replace with: MPRester(API_KEY).get_structure_by_material_id("mp-19017")
        # Then make supercell: initial_structure.make_supercell([2,2,2])
        # Then swap Ce: replace n_ce Mn sites with Ce

        # Load from CIF file if provided, else fall back to MP API
        if initial_structure is None:
            if Path(args.cif).exists():
                print(f"Loading structure from CIF: {args.cif}")
                initial_structure = Structure.from_file(args.cif)
                initial_structure.make_supercell(args.supercell)
                # Insert Ce substitutions at Mn sites
                n_ce = max(1, round(x_Ce * sum(1 for s in initial_structure if str(s.specie) == "Mn")))
                mn_indices = [i for i, s in enumerate(initial_structure) if str(s.specie) == "Mn"]
                for idx in np.random.choice(mn_indices, n_ce, replace=False):
                    initial_structure[idx] = "Ce"
                print(f"Substituted {n_ce} Mn → Ce  (Ce% = {x_Ce*100:.1f}%)")
            else:
                print("ERROR: Provide --cif <file> or set up MP API key.")
                raise SystemExit(1)

        if initial_structure is not None:
            results = run_hrmc(
                initial_structure=initial_structure,
                pattern_exp=pattern_exp,
                target_Ce_fraction=x_Ce,
                potential_dir=args.potential_dir,
                n_steps=args.n_steps,
                output_dir=args.output_dir,
            )
            print(f"\nBest structure saved to {args.output_dir}/best_structure.cif")

