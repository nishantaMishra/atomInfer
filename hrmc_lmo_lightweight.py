"""
HRMC (Hybrid Reverse Monte Carlo) for Ce-doped LiMn2O4 - LIGHTWEIGHT VERSION
============================================================================
No LAMMPS dependency! Uses pure Python MEAM calculator.

Three move types:
  1. Displacement  - refines local geometry
  2. Species swap  - Ce <-> Mn substitution at 16d sites
  3. Vacancy swap  - introduces/removes Mn vacancies (optional)

Cost function:
  F = w_xrd * R_factor(XRD) + w_energy * E_MEAM + w_comp * (x_Ce_current - x_Ce_target)^2

Potential:
  Li-Mn-O  : 2NN MEAM (Lee et al. 2017)
  Ce-O     : Buckingham + MEAM (Ce approximated as Mn)
  Coulomb  : Screened Coulomb for long-range interactions

Usage:
  # With CIF file
  from pymatgen.core import Structure
  structure = Structure.from_file("LiMn2O4.cif")
  
  # Or from Materials Project
  from mp_api.client import MPRester
  with MPRester(API_KEY) as mpr:
      structure = mpr.get_structure_by_material_id("mp-19017")
  
  # Run HRMC
  python hrmc_lmo_lightweight.py --cif LiMn2O4.cif --Ce_pct 5.0 --n_steps 1000
"""

import numpy as np
import json
import copy
import argparse
from pathlib import Path
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# Import our lightweight calculator
from meam_calculator import MEAMCalculator, minimize_structure, LightweightLAMMPSEvaluator

# ─────────────────────────────────────────────
# XRD R-factor computation
# ─────────────────────────────────────────────

def compute_xrd_pattern(structure: Structure, two_theta_range=(10, 80)):
    """Compute simulated XRD pattern using pymatgen."""
    calc = XRDCalculator(wavelength="CuKa")
    pattern = calc.get_pattern(structure, two_theta_range=two_theta_range)
    return pattern

def compute_r_factor(pattern_sim, pattern_exp):
    """
    Compute weighted R-factor between simulated and experimental XRD.
    R = sum|I_obs - I_calc| / sum|I_obs|
    """
    I_sim = np.interp(pattern_exp["two_theta"], pattern_sim.x, pattern_sim.y, left=0, right=0)
    I_exp = np.array(pattern_exp["intensities"])
    
    I_sim = 100 * I_sim / (I_sim.max() + 1e-10)
    I_exp = 100 * I_exp / (I_exp.max() + 1e-10)
    
    R = np.sum(np.abs(I_exp - I_sim)) / (np.sum(np.abs(I_exp)) + 1e-10)
    return float(R)

# ─────────────────────────────────────────────
# Structure moves
# ─────────────────────────────────────────────

def move_displacement(structure: Structure, max_disp: float = 0.15) -> Structure:
    """Move type 1: Displace a random non-Li atom."""
    s = structure.copy()
    candidates = [i for i, site in enumerate(s) if str(site.specie) != "Li"]
    if not candidates:
        return None
    idx = np.random.choice(candidates)
    disp = np.random.uniform(-max_disp, max_disp, 3)
    s.translate_sites([idx], disp, frac_coords=False)
    
    # Buckingham catastrophe guard
    neighbors = s.get_neighbors(s[idx], r=1.2)
    if len(neighbors) > 0:
        return None
    
    return s

def move_species_swap(structure: Structure, target_Ce_fraction: float) -> Structure:
    """Move type 2: Swap Ce <-> Mn to match target composition."""
    s = structure.copy()
    species = [str(site.specie) for site in s]
    
    n_Ce = species.count("Ce")
    n_Mn = species.count("Mn")
    n_total_16d = n_Ce + n_Mn
    current_fraction = n_Ce / n_total_16d if n_total_16d > 0 else 0
    
    if current_fraction < target_Ce_fraction:
        mn_sites = [i for i, sp in enumerate(species) if sp == "Mn"]
        if not mn_sites:
            return None
        idx = np.random.choice(mn_sites)
        s[idx] = "Ce"
    else:
        ce_sites = [i for i, sp in enumerate(species) if sp == "Ce"]
        if not ce_sites:
            return None
        idx = np.random.choice(ce_sites)
        s[idx] = "Mn"
    
    return s

def move_vacancy(structure: Structure, target_vacancy_fraction: float = 0.02) -> Structure:
    """Move type 3: Introduce/remove Mn vacancy."""
    if target_vacancy_fraction == 0:
        return None
    # Simplified: not implemented for hackathon
    return None

# ─────────────────────────────────────────────
# Cost function
# ─────────────────────────────────────────────

def compute_cost(structure: Structure,
                 pattern_exp: dict,
                 target_Ce_fraction: float,
                 energy_evaluator,
                 w_xrd: float = 1.0,
                 w_energy: float = 0.001,
                 w_comp: float = 5.0) -> dict:
    """
    Compute total cost F = w_xrd*R + w_energy*E + w_comp*(x_Ce - x_target)^2
    """
    # XRD R-factor
    pattern_sim = compute_xrd_pattern(structure)
    R = compute_r_factor(pattern_sim, pattern_exp)
    
    # Energy (with quick minimization)
    E_total = energy_evaluator.compute_energy(structure)
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
# Metropolis acceptance
# ─────────────────────────────────────────────

def metropolis_accept(dF: float, temperature: float) -> bool:
    """Standard Metropolis criterion."""
    if dF <= 0:
        return True
    prob = np.exp(-dF / temperature)
    return np.random.random() < prob

# ─────────────────────────────────────────────
# Main HRMC loop
# ─────────────────────────────────────────────

def run_hrmc(
    initial_structure: Structure,
    pattern_exp: dict,
    target_Ce_fraction: float,
    n_steps: int = 5000,
    T_start: float = 0.1,
    T_end: float = 0.01,
    max_disp: float = 0.15,
    w_xrd: float = 1.0,
    w_energy: float = 0.001,
    w_comp: float = 5.0,
    output_interval: int = 100,
    output_dir: str = "./hrmc_output",
    move_weights: tuple = (0.6, 0.3, 0.1),
) -> dict:
    """
    Main HRMC loop using lightweight MEAM calculator.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize lightweight evaluator (no LAMMPS!)
    energy_evaluator = LightweightLAMMPSEvaluator()
    
    current_structure = initial_structure.copy()
    current_cost = compute_cost(
        current_structure, pattern_exp, target_Ce_fraction,
        energy_evaluator, w_xrd, w_energy, w_comp
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
    print(f"HRMC Starting (Lightweight MEAM) | Steps: {n_steps}")
    print(f"Target Ce%: {target_Ce_fraction*100:.1f}%")
    print(f"Initial R-factor: {current_cost['R_factor']:.4f}")
    print(f"Initial E/atom:   {current_cost['E_per_atom']:.4f} eV")
    print(f"Initial Ce%:      {current_cost['x_Ce']*100:.2f}%")
    print(f"{'='*60}\n")
    
    for step in range(n_steps):
        T = T_start + (T_end - T_start) * (step / n_steps)
        move_type = np.random.choice(move_types, p=move_probs)
        n_proposed[move_type] += 1
        
        trial_structure = None
        if move_type == "displacement":
            trial_structure = move_displacement(current_structure, max_disp)
        elif move_type == "swap":
            trial_structure = move_species_swap(current_structure, target_Ce_fraction)
        elif move_type == "vacancy":
            trial_structure = move_vacancy(current_structure)
        
        if trial_structure is None:
            continue
        
        trial_cost = compute_cost(
            trial_structure, pattern_exp, target_Ce_fraction,
            energy_evaluator, w_xrd, w_energy, w_comp
        )
        
        dF = trial_cost["F"] - current_cost["F"]
        
        if metropolis_accept(dF, T):
            current_structure = trial_structure
            current_cost = trial_cost
            n_accepted[move_type] += 1
            
            if current_cost["F"] < best_cost["F"]:
                best_structure = current_structure.copy()
                best_cost = current_cost.copy()
        
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
            
            current_structure.to(
                filename=str(Path(output_dir) / f"step_{step:05d}.cif"),
                fmt="cif"
            )
    
    best_structure.to(filename=str(Path(output_dir) / "best_structure.cif"), fmt="cif")
    best_structure.to(filename=str(Path(output_dir) / "best_structure.vasp"), fmt="poscar")
    
    with open(Path(output_dir) / "trajectory.json", "w") as f:
        json.dump(trajectory, f, indent=2)
    
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
    }

# ─────────────────────────────────────────────
# Synthetic data generator
# ─────────────────────────────────────────────

def generate_synthetic_xrd(Ce_fraction: float, two_theta_range=(10, 80), n_points=500) -> dict:
    """Generate synthetic XRD using Vegard's law."""
    a = 8.248 + 0.04 * Ce_fraction
    
    lattice = Lattice.cubic(a)
    
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
        [0.136, 0.136, 0.136], [0.136, 0.364, 0.364],
        [0.364, 0.136, 0.364], [0.364, 0.364, 0.136],
    ]
    
    n_ce = max(1, round(Ce_fraction * len(mn_sites_frac))) if Ce_fraction > 0 else 0
    species_mn = ["Ce"] * n_ce + ["Mn"] * (len(mn_sites_frac) - n_ce)
    
    species = ["Li"] * len(li_sites) + species_mn + ["O"] * len(o_sites)
    coords = li_sites + mn_sites_frac + o_sites
    
    structure = Structure(lattice, species, coords)
    
    calc = XRDCalculator(wavelength="CuKa")
    pattern = calc.get_pattern(structure, two_theta_range=two_theta_range)
    
    two_theta_grid = np.linspace(two_theta_range[0], two_theta_range[1], n_points)
    intensities = np.interp(two_theta_grid, pattern.x, pattern.y, left=0, right=0)
    noise = np.random.normal(0, 0.02 * intensities.max(), n_points)
    intensities = np.clip(intensities + noise, 0, None)
    
    return {
        "Ce_fraction": Ce_fraction,
        "lattice_a": a,
        "two_theta": two_theta_grid.tolist(),
        "intensities": intensities.tolist(),
    }

def load_structure_from_cif(cif_path: str, supercell: tuple = (2, 2, 2), 
                            target_Ce_fraction: float = 0.05) -> Structure:
    """
    Load structure from CIF and prepare for HRMC.
    
    Args:
        cif_path: Path to CIF file
        supercell: Supercell dimensions
        target_Ce_fraction: Target Ce doping fraction
    """
    structure = Structure.from_file(cif_path)
    
    # Make supercell
    structure.make_supercell(supercell)
    
    # Add Ce doping (replace some Mn with Ce)
    species = [str(site.specie) for site in structure]
    mn_indices = [i for i, sp in enumerate(species) if sp == "Mn"]
    
    n_mn = len(mn_indices)
    n_ce = int(target_Ce_fraction * n_mn)
    
    if n_ce > 0 and mn_indices:
        ce_indices = np.random.choice(mn_indices, size=n_ce, replace=False)
        for idx in ce_indices:
            structure[int(idx)] = "Ce"
    
    print(f"Loaded structure from {cif_path}")
    print(f"  Formula: {structure.composition.reduced_formula}")
    print(f"  Supercell: {supercell}")
    print(f"  Total atoms: {len(structure)}")
    print(f"  Initial Ce fraction: {n_ce/n_mn*100:.2f}%")
    
    return structure

# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HRMC for Ce-doped LiMn2O4 (Lightweight)")
    parser.add_argument("--mode", choices=["generate_data", "run_hrmc", "test"],
                        default="test")
    parser.add_argument("--cif", type=str, help="Path to CIF file")
    parser.add_argument("--Ce_pct", type=float, default=5.0,
                        help="Ce concentration in percent")
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="./hrmc_output")
    parser.add_argument("--supercell", type=int, nargs=3, default=[2, 2, 2],
                        help="Supercell dimensions (e.g., 2 2 2)")
    args = parser.parse_args()
    
    if args.mode == "test":
        print("Running quick test...")
        x_Ce = args.Ce_pct / 100
        
        # Generate synthetic data
        exp_data = generate_synthetic_xrd(x_Ce)
        print(f"Generated XRD: a = {exp_data['lattice_a']:.4f} Å")
        
        # Test with evaluator
        evaluator = LightweightLAMMPSEvaluator()
        
        # Build simple test structure
        lattice = Lattice.cubic(exp_data['lattice_a'])
        s = Structure(lattice, ["Li", "Mn", "Mn", "O", "O", "O", "O"],
                      [[0, 0, 0], [0.5, 0.5, 0.5], [0, 0.5, 0],
                       [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
                       [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]])
        
        e = evaluator.compute_energy(s)
        print(f"Test energy: {e:.4f} eV")
        print("\n✓ Test passed! Run with --mode run_hrmc to start full HRMC.")
    
    elif args.mode == "generate_data":
        print("Generating synthetic XRD data...")
        Path("./synthetic_data").mkdir(exist_ok=True)
        
        for x_Ce in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]:
            data = generate_synthetic_xrd(x_Ce)
            fname = f"./synthetic_data/XRD_Ce{int(x_Ce*100):02d}pct.json"
            with open(fname, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  Ce={x_Ce*100:.1f}%: a = {data['lattice_a']:.4f} Å")
    
    elif args.mode == "run_hrmc":
        x_Ce = args.Ce_pct / 100
        
        # Load experimental data
        exp_file = f"./synthetic_data/XRD_Ce{int(args.Ce_pct):02d}pct.json"
        if Path(exp_file).exists():
            with open(exp_file) as f:
                pattern_exp = json.load(f)
        else:
            print(f"Generating synthetic data for Ce={args.Ce_pct}%...")
            pattern_exp = generate_synthetic_xrd(x_Ce)
        
        # Load or build initial structure
        if args.cif and Path(args.cif).exists():
            initial_structure = load_structure_from_cif(
                args.cif, 
                supercell=tuple(args.supercell),
                target_Ce_fraction=x_Ce
            )
        else:
            print("No CIF provided, building structure from scratch...")
            initial_structure = generate_synthetic_xrd(x_Ce)  # Returns dict, need structure
            # Build from the synthetic data parameters
            lattice = Lattice.cubic(8.248 + 0.04 * x_Ce)
            # Simple spinel structure
            s = Structure(lattice, ["Li", "Mn", "Mn", "O", "O", "O", "O"],
                          [[0, 0, 0], [0.5, 0.5, 0.5], [0, 0.5, 0],
                           [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
                           [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]])
            s.make_supercell([2, 2, 2])
            # Add Ce
            mn_sites = [i for i, site in enumerate(s) if str(site.specie) == "Mn"]
            n_ce = int(x_Ce * len(mn_sites))
            for idx in np.random.choice(mn_sites, n_ce, replace=False):
                s[int(idx)] = "Ce"
            initial_structure = s
        
        # Run HRMC
        results = run_hrmc(
            initial_structure=initial_structure,
            pattern_exp=pattern_exp,
            target_Ce_fraction=x_Ce,
            n_steps=args.n_steps,
            output_dir=args.output_dir,
        )
        
        print(f"\n✓ HRMC complete! Best structure: {args.output_dir}/best_structure.cif")
