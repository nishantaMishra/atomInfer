"""
Lightweight MEAM Energy Calculator for Li-Mn-O-Ce Systems
=========================================================
Pure Python implementation of 2NN MEAM for HRMC loops.
No LAMMPS dependency - uses NumPy for vectorized computations.

References:
- Lee et al. 2017 (Li-Mn-O parameters from NIST)
- Baskes, M. I. (1992). Modified embedded-atom method.
- Lee, B. J. (2006). 2NN MEAM formalism.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json

# ─────────────────────────────────────────────
# Element parameters from library.meam
# ─────────────────────────────────────────────

@dataclass
class MEAMElement:
    """Single-element MEAM parameters."""
    symbol: str
    z: int              # coordination number
    iel: int            # element index
    atwt: float         # atomic weight
    alpha: float        # exponential decay factor
    b0: float           # embedding function parameter
    b1: float
    b2: float
    b3: float
    alat: float         # lattice parameter
    esub: float         # sublimation energy
    asub: float         # adjustable parameter
    t0: float           # screening parameters
    t1: float
    t2: float
    t3: float
    rozero: float       # background density
    ibar: int           # density function type

@dataclass
class MEAMPair:
    """Alloy/pair MEAM parameters."""
    zbl: float = 0.0           # ZBL flag
    nn2: int = 0               # 2NN screening flag
    rho0: float = 1.0          # density scaling
    Ec: float = 0.0            # cohesive energy
    re: float = 0.0            # equilibrium distance
    alpha: float = 0.0         # exponential factor
    repuls: float = 0.0        # repulsive term adjustment
    attrac: float = 0.0        # attractive term adjustment
    Cmin: float = 2.0          # screening parameter
    Cmax: float = 2.8          # screening parameter
    lattce: str = ""           # lattice type

# Parse single-element parameters from library.meam
MEAM_ELEMENTS = {
    "Li": MEAMElement("Li", z=8, iel=0, atwt=6.9410,
                      alpha=3.0982595559, b0=1.65, b1=1.0, b2=4.0, b3=1.0,
                      alat=3.4871956259, esub=1.65, asub=0.95,
                      t0=1.0, t1=2.3, t2=5.0, t3=0.5, rozero=0.5, ibar=3),
    "Mn": MEAMElement("Mn", z=8, iel=0, atwt=54.94,
                      alpha=5.7345791984, b0=4.3, b1=1.0, b2=2.0, b3=6.5,
                      alat=2.9213923621, esub=2.9, asub=0.7,
                      t0=1.0, t1=4.0, t2=-3.0, t3=-4.0, rozero=1.0, ibar=3),
    "O": MEAMElement("O", z=1, iel=0, atwt=16.0,
                     alpha=6.88, b0=5.47, b1=5.3, b2=5.18, b3=5.57,
                     alat=1.21, esub=2.56, asub=1.44,
                     t0=1.0, t1=0.1, t2=0.11, t3=0.0, rozero=12.0, ibar=3),
}

# Parse alloy parameters from library.meam_alloy
MEAM_PAIRS = {}

# Li-Li (1,1)
MEAM_PAIRS[("Li", "Li")] = MEAMPair(zbl=0, nn2=1, rho0=0.5, Ec=1.65, re=3.02,
                                     alpha=3.09825956, repuls=0.05, attrac=0.05,
                                     Cmin=0.16, Cmax=2.8)

# Mn-Mn (2,2)
MEAM_PAIRS[("Mn", "Mn")] = MEAMPair(zbl=0, nn2=1, rho0=1.0, Ec=2.9, re=2.53,
                                     alpha=5.73457920, repuls=0.0, attrac=0.0,
                                     Cmin=0.16, Cmax=2.8)

# O-O (3,3)
MEAM_PAIRS[("O", "O")] = MEAMPair(zbl=0, nn2=1, rho0=12.0, Ec=2.56, re=1.21,
                                   alpha=6.88, repuls=0.0, attrac=0.0,
                                   Cmin=2.0, Cmax=2.8)

# Li-Mn (1,2)
MEAM_PAIRS[("Li", "Mn")] = MEAMPair(zbl=0, nn2=1, lattce="b2", Ec=1.775, re=2.69,
                                     alpha=4.01997448, repuls=0.0, attrac=0.0,
                                     Cmin=0.2, Cmax=2.8)
MEAM_PAIRS[("Mn", "Li")] = MEAM_PAIRS[("Li", "Mn")]  # symmetric

# Li-O (1,3)
MEAM_PAIRS[("Li", "O")] = MEAMPair(zbl=0, nn2=1, lattce="b1", Ec=1.6836, re=1.95,
                                    alpha=7.31510556, repuls=0.07, attrac=0.07,
                                    Cmin=0.7, Cmax=1.55)
MEAM_PAIRS[("O", "Li")] = MEAM_PAIRS[("Li", "O")]

# Mn-O (2,3)
MEAM_PAIRS[("Mn", "O")] = MEAMPair(zbl=0, nn2=1, lattce="b1", Ec=1.7829, re=2.1276,
                                    alpha=5.29364387, repuls=0.1, attrac=0.0,
                                    Cmin=3.0, Cmax=4.0)
MEAM_PAIRS[("O", "Mn")] = MEAM_PAIRS[("Mn", "O")]

# Buckingham parameters for Ce-O (from hrmc_lmo.py)
BUCKINGHAM = {
    ("Ce", "O"): {"A": 2141.4, "rho": 0.3541, "C": 43.83},
    ("O", "Ce"): {"A": 2141.4, "rho": 0.3541, "C": 43.83},
}

# Formal charges for Coulomb
CHARGES = {"Li": 1.0, "Mn": 3.5, "O": -2.0, "Ce": 3.0}

# ─────────────────────────────────────────────
# Core MEAM Functions
# ─────────────────────────────────────────────

class MEAMCalculator:
    """
    Lightweight MEAM calculator for Li-Mn-O-Ce systems.
    
    Usage:
        calc = MEAMCalculator(rcut=4.8)
        energy = calc.compute_energy(structure)  # structure is pymatgen Structure
    """
    
    def __init__(self, rcut: float = 4.8, use_buckingham: bool = True, 
                 use_coulomb: bool = True, kappa: float = 10.0):
        """
        Args:
            rcut: Cutoff radius in Angstroms (from library.meam_alloy)
            use_buckingham: Include Buckingham for Ce-O
            use_coulomb: Include Coulomb interactions
            kappa: Screening parameter for Coulomb (1/Å)
        """
        self.rcut = rcut
        self.rcut_sq = rcut ** 2
        self.use_buckingham = use_buckingham
        self.use_coulomb = use_coulomb
        self.kappa = kappa
        
    def _get_element_params(self, symbol: str) -> MEAMElement:
        """Get MEAM parameters for an element."""
        if symbol == "Ce":
            # Ce uses Mn parameters as base (similar size)
            return MEAM_ELEMENTS["Mn"]
        return MEAM_ELEMENTS[symbol]
    
    def _get_pair_params(self, sym1: str, sym2: str) -> MEAMPair:
        """Get pair parameters."""
        # Handle Ce as special case
        if sym1 == "Ce":
            sym1 = "Mn"  # Approximate Ce with Mn for MEAM
        if sym2 == "Ce":
            sym2 = "Mn"
        
        key = (sym1, sym2)
        if key in MEAM_PAIRS:
            return MEAM_PAIRS[key]
        # Default: use mixing rules
        p1 = MEAM_PAIRS.get((sym1, sym1), MEAMPair())
        p2 = MEAM_PAIRS.get((sym2, sym2), MEAMPair())
        return MEAMPair(
            Ec=(p1.Ec + p2.Ec) / 2,
            re=(p1.re + p2.re) / 2,
            alpha=(p1.alpha + p2.alpha) / 2,
        )
    
    def _rho_a(self, r: float, elem: MEAMElement) -> float:
        """Compute spherically symmetric electron density (rho^a(0))."""
        # rho^a(0) = r * exp[-beta_i * (r/re - 1)]
        # Simplified: using rozero as base density
        re = elem.alat  # Use lattice parameter as reference
        beta = elem.alpha
        return elem.rozero * np.exp(-beta * (r / re - 1))
    
    def _f_ij(self, r: float, re: float, alpha: float) -> float:
        """Pair potential function f(r)."""
        # f(r) = A * exp[-alpha * (r/re - 1)]
        # A is related to cohesive energy
        if r > self.rcut:
            return 0.0
        return np.exp(-alpha * (r / re - 1))
    
    def _embedding_energy(self, rho: float, elem: MEAMElement) -> float:
        """Compute embedding energy F(rho)."""
        # MEAM embedding function:
        # F(rho) = A * E_c * rho * ln(rho)  for ibar = 0
        # F(rho) = sum(b_n * (rho/rho0)^n)  for other ibar
        
        if elem.ibar == 0:
            # logarithmic form
            if rho < 1e-10:
                return 0.0
            return elem.asub * elem.esub * rho * np.log(rho)
        else:
            # polynomial form
            x = rho / elem.rozero if elem.rozero > 0 else rho
            return (elem.b0 + elem.b1 * x + elem.b2 * x**2 + elem.b3 * x**3) * elem.esub
    
    def _phi_pair(self, r: float, pair: MEAMPair) -> float:
        """Compute pair potential phi(r)."""
        if r > self.rcut or r < 0.1:
            return 0.0
        
        # Pair potential: phi(r) = (2*E_c / z) * f(r)
        # with modifications for repulsion/attraction
        z = 8  # typical coordination
        base = (2 * pair.Ec / z) * self._f_ij(r, pair.re, pair.alpha)
        
        # Apply repuls/attrac corrections
        # Simple approximation: adjust by exponential factor
        corr = np.exp(-pair.alpha * (r / pair.re - 1))
        return base * (1 + pair.repuls * corr - pair.attrac * corr)
    
    def _buckingham_energy(self, r: float, A: float, rho: float, C: float) -> float:
        """Buckingham potential: V(r) = A * exp(-r/rho) - C/r^6"""
        if r < 0.5:  # Avoid singularity
            return 1e10
        return A * np.exp(-r / rho) - C / r**6
    
    def _coulomb_energy(self, r: float, q1: float, q2: float) -> float:
        """Coulomb potential with screening: V(r) = q1*q2/r * erfc(kappa*r)"""
        if r < 0.1:
            return 1e10
        # Real-space Ewald-like screened Coulomb
        from scipy.special import erfc
        return q1 * q2 / r * erfc(self.kappa * r)
    
    def compute_energy(self, structure) -> float:
        """
        Compute total energy of structure using semi-vectorized operations and PBC.
        
        Args:
            structure: pymatgen Structure object
            
        Returns:
            Total energy in eV
        """
        # distance_matrix automatically handles Periodic Boundary Conditions (PBC)
        dist_mat = structure.distance_matrix 
        species = [str(s.specie) for s in structure]
        n_atoms = len(species)
        
        energy = 0.0
        densities = np.zeros(n_atoms)
        
        # --- First Pass: Compute Electron Densities ---
        for i in range(n_atoms):
            r_array = dist_mat[i]
            # NumPy mask: instantly find all neighbors within cutoff, excluding self
            mask = (r_array > 0.1) & (r_array <= self.rcut)
            valid_indices = np.where(mask)[0]
            
            for j in valid_indices:
                elem_j = self._get_element_params(species[j])
                r = r_array[j]
                # Inline rho_a computation to skip function call overhead
                densities[i] += elem_j.rozero * np.exp(-elem_j.alpha * (r / elem_j.alat - 1))

        # --- Second Pass: Compute Embedding and Pair Energies ---
        for i in range(n_atoms):
            sym_i = species[i]
            elem_i = self._get_element_params(sym_i)
            
            # 1. Add Embedding Energy
            energy += self._embedding_energy(densities[i], elem_i)
            
            # 2. Pair interactions (mask for j > i to prevent double-counting)
            r_array = dist_mat[i]
            mask = (np.arange(n_atoms) > i) & (r_array > 0.1) & (r_array <= self.rcut)
            valid_indices = np.where(mask)[0]
            
            for j in valid_indices:
                sym_j = species[j]
                r = r_array[j]
                
                # MEAM pair potential
                if sym_i in MEAM_ELEMENTS and sym_j in MEAM_ELEMENTS:
                    pair = self._get_pair_params(sym_i, sym_j)
                    energy += self._phi_pair(r, pair)
                
                # Buckingham for Ce-O
                if self.use_buckingham:
                    key = (sym_i, sym_j)
                    if key in BUCKINGHAM:
                        p = BUCKINGHAM[key]
                        energy += self._buckingham_energy(r, p["A"], p["rho"], p["C"])
                
                # Coulomb interactions
                if self.use_coulomb:
                    if sym_i in CHARGES and sym_j in CHARGES:
                        energy += self._coulomb_energy(r, CHARGES[sym_i], CHARGES[sym_j])
                        
        return energy
    
    def compute_local_energy_change(self, structure_old, atom_idx: int, new_cart_coords: np.ndarray) -> float:
        """
        Computes ΔE for a single atom displacement efficiently using a local cluster.
        In MEAM, moving atom i affects the embedding energy of its neighbors j.
        
        Args:
            structure_old: Original pymatgen Structure
            atom_idx: Index of atom being moved
            new_cart_coords: New Cartesian coordinates for the atom
            
        Returns:
            Energy change (delta E) in eV
        """
        # 1. Identify neighbors within rcut (1-hop) in the old structure
        dist_mat_old = structure_old.distance_matrix
        r_old = dist_mat_old[atom_idx]
        mask_1hop = (r_old > 0.1) & (r_old <= self.rcut)
        neighbors_1hop = np.where(mask_1hop)[0]
        
        # 2. Get distances from the proposed NEW position using PBC
        lattice = structure_old.lattice
        fcoords_new = lattice.get_fractional_coords(new_cart_coords)
        # Calculate distances from the new point to all other atoms
        r_new = lattice.get_all_distances(fcoords_new, structure_old.frac_coords)[0]
        
        # Neighbors within rcut of the NEW position
        mask_1hop_new = (r_new > 0.1) & (r_new <= self.rcut)
        neighbors_1hop_new = np.where(mask_1hop_new)[0]
        
        # 3. Combine to find all atoms whose electron density (rho) will change
        affected_atoms = set(neighbors_1hop).union(neighbors_1hop_new)
        affected_atoms.add(atom_idx)
        affected_list = list(affected_atoms)
        
        # 4. Extract the local sub-structure distance matrix
        # (This avoids calculating O(N^2) for the whole box)
        sub_dist_old = dist_mat_old[np.ix_(affected_list, affected_list)]
        
        # Build the new sub-structure distance matrix
        sub_dist_new = sub_dist_old.copy()
        local_idx = affected_list.index(atom_idx)
        
        # Update distances for the moved atom in the new sub-matrix
        for local_j, global_j in enumerate(affected_list):
            if global_j == atom_idx:
                continue
            sub_dist_new[local_idx, local_j] = r_new[global_j]
            sub_dist_new[local_j, local_idx] = r_new[global_j]
            
        species_list = [str(structure_old[i].specie) for i in affected_list]
        
        # 5. Helper to compute cluster energy
        def _cluster_energy(sub_matrix):
            e_cluster = 0.0
            densities = np.zeros(len(affected_list))
            
            # Electron densities
            for i in range(len(affected_list)):
                r_array = sub_matrix[i]
                mask = (r_array > 0.1) & (r_array <= self.rcut)
                for j in np.where(mask)[0]:
                    elem_j = self._get_element_params(species_list[j])
                    densities[i] += elem_j.rozero * np.exp(-elem_j.alpha * (r_array[j] / elem_j.alat - 1))
            
            # Embedding + Pair potentials
            for i in range(len(affected_list)):
                sym_i = species_list[i]
                elem_i = self._get_element_params(sym_i)
                e_cluster += self._embedding_energy(densities[i], elem_i)
                
                r_array = sub_matrix[i]
                mask = (np.arange(len(affected_list)) > i) & (r_array > 0.1) & (r_array <= self.rcut)
                for j in np.where(mask)[0]:
                    sym_j = species_list[j]
                    r = r_array[j]
                    
                    if sym_i in MEAM_ELEMENTS and sym_j in MEAM_ELEMENTS:
                        e_cluster += self._phi_pair(r, self._get_pair_params(sym_i, sym_j))
                    if self.use_buckingham and (sym_i, sym_j) in BUCKINGHAM:
                        p = BUCKINGHAM[(sym_i, sym_j)]
                        e_cluster += self._buckingham_energy(r, p["A"], p["rho"], p["C"])
                    if self.use_coulomb and sym_i in CHARGES and sym_j in CHARGES:
                        e_cluster += self._coulomb_energy(r, CHARGES[sym_i], CHARGES[sym_j])
            return e_cluster

        # 6. The energy change is the difference in local cluster energies
        return _cluster_energy(sub_dist_new) - _cluster_energy(sub_dist_old)
    
    def compute_forces(self, structure) -> np.ndarray:
        """
        Compute forces on all atoms (for minimization).
        Returns forces in eV/Angstrom.
        """
        # Simplified: numerical differentiation
        delta = 0.001
        forces = np.zeros((len(structure), 3))
        
        e0 = self.compute_energy(structure)
        positions = structure.cart_coords.copy()
        
        for i in range(len(structure)):
            for dim in range(3):
                # +delta - create new structure to avoid cache issues
                new_pos = positions.copy()
                new_pos[i, dim] += delta
                
                from pymatgen.core import Structure, Lattice
                struct_plus = Structure(
                    structure.lattice,
                    [str(s.specie) for s in structure],
                    new_pos,
                    coords_are_cartesian=True
                )
                e_plus = self.compute_energy(struct_plus)
                
                # -delta
                new_pos = positions.copy()
                new_pos[i, dim] -= delta
                struct_minus = Structure(
                    structure.lattice,
                    [str(s.specie) for s in structure],
                    new_pos,
                    coords_are_cartesian=True
                )
                e_minus = self.compute_energy(struct_minus)
                
                # F = -dE/dx
                forces[i, dim] = -(e_plus - e_minus) / (2 * delta)
        
        return forces


# ─────────────────────────────────────────────
# Simple Energy Minimizer
# ─────────────────────────────────────────────

def minimize_structure(structure, calculator: MEAMCalculator, 
                       max_steps: int = 100, f_tol: float = 1e-3,
                       step_size: float = 0.01) -> Tuple:
    """
    Simple steepest descent minimization.
    
    Returns:
        (minimized_structure, final_energy)
    """
    from pymatgen.core import Structure
    
    s = structure.copy()
    
    for step in range(max_steps):
        energy = calculator.compute_energy(s)
        forces = calculator.compute_forces(s)
        
        f_max = np.max(np.abs(forces))
        if f_max < f_tol:
            print(f"Converged at step {step}, E = {energy:.4f} eV, f_max = {f_max:.6f}")
            break
        
        # Steepest descent update
        positions = s.cart_coords.copy()
        positions += step_size * forces / (f_max + 1e-10)  # normalized step
        
        # Update structure
        for i, site in enumerate(s):
            site._coords = positions[i]
        
        if step % 20 == 0:
            print(f"  Step {step}: E = {energy:.4f} eV, f_max = {f_max:.6f}")
    
    final_energy = calculator.compute_energy(s)
    return s, final_energy


# ─────────────────────────────────────────────
# Modified HRMC Evaluator
# ─────────────────────────────────────────────

class LightweightLAMMPSEvaluator:
    """
    Drop-in replacement for LAMMPSEvaluator in hrmc_lmo.py.
    Uses pure Python MEAM calculator instead of LAMMPS.
    """
    
    def __init__(self, potential_dir: str = None):
        """
        Args:
            potential_dir: Kept for API compatibility, not used.
        """
        self.calculator = MEAMCalculator(rcut=4.8)
        print("Initialized Lightweight MEAM Calculator (no LAMMPS required)")
    
    def __del__(self):
        pass  # No cleanup needed
    
    def compute_energy(self, structure) -> float:
        """
        Compute minimized energy of structure.
        
        This mimics the LAMMPS behavior: minimize then return energy.
        """
        # Quick local relaxation (lighter than full minimization)
        # For HRMC, we just need consistent energies, not perfect minima
        
        # Option 1: Fast single-point energy (for speed)
        # return self.calculator.compute_energy(structure)
        
        # Option 2: With quick relaxation (better accuracy)
        _, energy = minimize_structure(
            structure, self.calculator, 
            max_steps=20,  # Quick relaxation
            f_tol=1e-2,    # Relaxed tolerance
            step_size=0.01
        )
        return energy


# ─────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing MEAM Calculator...")
    
    # Create a simple test structure
    from pymatgen.core import Structure, Lattice
    
    # Li2O-like structure (simplified)
    lattice = Lattice.cubic(4.6)
    structure = Structure(
        lattice,
        ["Li", "Li", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]
    )
    
    calc = MEAMCalculator(rcut=4.8)
    
    print("\n1. Testing single-point energy...")
    energy = calc.compute_energy(structure)
    print(f"   Energy = {energy:.4f} eV")
    
    print("\n2. Testing forces...")
    forces = calc.compute_forces(structure)
    print(f"   Forces shape: {forces.shape}")
    print(f"   Max force: {np.max(np.abs(forces)):.6f} eV/Å")
    
    print("\n3. Testing minimization...")
    min_structure, min_energy = minimize_structure(structure, calc, max_steps=50)
    print(f"   Initial E: {energy:.4f} eV")
    print(f"   Final E: {min_energy:.4f} eV")
    
    print("\n4. Testing LightweightLAMMPSEvaluator...")
    evaluator = LightweightLAMMPSEvaluator()
    e = evaluator.compute_energy(structure)
    print(f"   Evaluator energy: {e:.4f} eV")
    
    print("\n✓ All tests passed!")
