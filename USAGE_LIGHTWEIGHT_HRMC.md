# Lightweight HRMC for Ce-doped LiMn₂O₄

**No LAMMPS required!** This version uses a pure Python MEAM calculator.

## 📁 Files

| File | Description |
|------|-------------|
| `meam_calculator.py` | Lightweight MEAM energy calculator |
| `hrmc_lmo_lightweight.py` | HRMC script (no LAMMPS dependency) |
| `potentials/library.meam` | MEAM single-element parameters |
| `potentials/library.meam_alloy` | MEAM pair parameters |

## 🚀 Quick Start

### 1. Test the calculator
```bash
python hrmc_lmo_lightweight.py --mode test --Ce_pct 5.0
```

### 2. Generate synthetic XRD data
```bash
python hrmc_lmo_lightweight.py --mode generate_data
```

### 3. Run HRMC with a CIF file
```bash
python hrmc_lmo_lightweight.py --mode run_hrmc \
    --cif LiMn2O4.cif \
    --Ce_pct 5.0 \
    --n_steps 1000 \
    --supercell 2 2 2 \
    --output_dir ./hrmc_output
```

## 📥 Getting a CIF File

### Option A: From Materials Project (Recommended)
```python
from mp_api.client import MPRester

with MPRester("YOUR_API_KEY") as mpr:
    # LiMn2O4 spinel structure
    structure = mpr.get_structure_by_material_id("mp-19017")
    structure.to(filename="LiMn2O4.cif")
```

### Option B: From Online Databases
- **Materials Project**: https://materialsproject.org
- **AFLOW**: http://aflowlib.org
- **COD**: http://crystallography.net

Search for "LiMn2O4" or "lithium manganese oxide spinel"

### Option C: Create Manually
```python
from pymatgen.core import Structure, Lattice

# Spinel LiMn2O4 (Fd-3m, a = 8.248 Å)
lattice = Lattice.cubic(8.248)
structure = Structure(
    lattice,
    ["Li", "Mn", "Mn", "O", "O", "O", "O"],
    [[0, 0, 0], 
     [0.625, 0.625, 0.625], 
     [0.375, 0.375, 0.625],
     [0.386, 0.386, 0.386],
     [0.614, 0.614, 0.386],
     [0.614, 0.386, 0.614],
     [0.386, 0.614, 0.614]]
)
structure.to(filename="LiMn2O4.cif")
```

## 🔬 How It Works

### Energy Calculation (No LAMMPS!)
```python
from meam_calculator import LightweightLAMMPSEvaluator

evaluator = LightweightLAMMPSEvaluator()  # No LAMMPS needed!
energy = evaluator.compute_energy(structure)
```

### The MEAM Calculator Implements:
1. **Electron density** computation (ρ_i)
2. **Embedding energy** F(ρ_i)
3. **Pair potential** φ(r_ij)
4. **Screening functions** (2NN MEAM)
5. **Buckingham** for Ce-O
6. **Coulomb** with Ewald screening

### HRMC Loop
```
For each step:
  1. Propose move (displacement / swap / vacancy)
  2. Compute XRD R-factor
  3. Compute MEAM energy (with lightweight calculator)
  4. Metropolis acceptance
  5. Save best structure
```

## ⚡ Performance

| Operation | Time (100 atoms) |
|-----------|-----------------|
| Single energy | ~0.05 s |
| Energy + minimization | ~1-2 s |
| 1 HRMC step | ~2-3 s |
| 1000 steps | ~30-50 min |

**Speed tips:**
- Use `--n_steps 500` for testing
- Use smaller supercell: `--supercell 1 1 1`
- Disable Coulomb if not needed (edit code)

## 📊 Output Files

```
hrmc_output/
├── step_00000.cif      # Structure every N steps
├── step_00100.cif
├── ...
├── best_structure.cif  # Final best structure
├── best_structure.vasp # VASP POSCAR format
└── trajectory.json     # Energy/R-factor history
```

## 🔧 Troubleshooting

### "ImportError: No module named 'pymatgen'"
```bash
pip install pymatgen numpy scipy
```

### Energy too high / unrealistic
- Check CIF file has correct species (Li, Mn, O)
- Make sure structure is not overlapping
- Try initial minimization before HRMC

### R-factor not improving
- Increase `w_xrd` weight
- Check experimental XRD pattern format
- Try more steps: `--n_steps 5000`

## 📚 Citation

The MEAM potential parameters are from:
> Lee, E., Lee, K.-R., & Lee, B.-J. (2017). "Interatomic Potential of Li–Mn–O and Molecular Dynamics Simulations on Li Diffusion in Spinel Li1–xMn2O4", *J. Phys. Chem. C* 121(24), 13008-13017.

Available at NIST Interatomic Potentials Repository:
https://www.ctcms.nist.gov/potentials/system/Li-Mn-O/
