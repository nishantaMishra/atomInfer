"""
AtomInfer Configuration Loader
==============================
Loads config.toml (user) or config.default.toml (fallback), validates
structure, and exposes typed accessors for every subsystem.

Usage:
    from config_loader import cfg

    cfg.server.port          # 8000
    cfg.llm.temperature      # 0.05
    cfg.material.formula     # "LiMn2O4"
    cfg.xrd.wavelength_A     # 1.54059
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# Python 3.11+ ships tomllib; older versions need tomli
try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        raise ImportError(
            "TOML support requires Python 3.11+ or the 'tomli' package.\n"
            "  pip install tomli"
        )

_ROOT = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════════════════════
# Data classes — typed views into the raw TOML dict
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    open_browser: bool = True


@dataclass
class DirectoryConfig:
    output: str = "./atomInfer_output"
    uploads: str = "./uploads"
    models: str = "./models"


@dataclass
class ModelProfile:
    name: str = ""
    provider: str = "ollama"
    model: str = ""
    endpoint: str = ""
    api_key_env: str = ""

    def get_api_key(self) -> Optional[str]:
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


@dataclass
class LLMConfig:
    temperature: float = 0.05
    max_tokens: int = 4096
    max_iterations: int = 15
    models: Dict[str, ModelProfile] = field(default_factory=dict)
    task_assignments: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class DopingConfig:
    host_site: str = "16d"
    host_element: str = "Mn"
    allowed_dopants: List[str] = field(default_factory=lambda: ["Ce"])
    max_fraction: float = 0.10
    seed: int = 42
    ionic_radii_A: Dict[str, float] = field(default_factory=dict)
    vegard_a_pristine_A: float = 8.2480
    vegard_coefficient: float = 0.35
    vegard_min_fraction: float = 0.0
    vegard_max_fraction: float = 0.10


@dataclass
class MaterialConfig:
    name: str = "LiMn2O4"
    formula: str = "LiMn2O4"
    mp_id: str = "mp-19017"
    space_group: str = "Fd-3m"
    space_group_number: int = 227
    crystal_system: str = "cubic"
    structure_type: str = "spinel"
    reference_lattice_A: float = 8.2480
    wyckoff: Dict[str, str] = field(default_factory=dict)
    doping: DopingConfig = field(default_factory=DopingConfig)


@dataclass
class RFactorConfig:
    excellent: float = 0.10
    good: float = 0.20
    acceptable: float = 0.30

    def quality_label(self, r: float) -> str:
        if r < self.excellent:
            return "excellent"
        if r < self.good:
            return "good"
        if r < self.acceptable:
            return "acceptable"
        return "poor - consider HRMC refinement"


@dataclass
class SecondaryPhase:
    peak_positions_deg: List[float] = field(default_factory=list)
    detection_threshold: float = 0.02


@dataclass
class XRDConfig:
    wavelength_A: float = 1.54059
    scherrer_constant: float = 0.9
    angular_range_deg: List[float] = field(default_factory=lambda: [10.0, 80.0])
    r_factor: RFactorConfig = field(default_factory=RFactorConfig)
    hkl_windows: Dict[str, List[float]] = field(default_factory=dict)
    secondary_phases: Dict[str, SecondaryPhase] = field(default_factory=dict)


@dataclass
class RamanConfig:
    wavenumber_range_cm1: List[float] = field(default_factory=lambda: [100.0, 1000.0])
    reference_A1g_cm1: float = 628.0
    mode_windows: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class HRMCTemperature:
    start: float = 0.1
    end: float = 0.01


@dataclass
class HRMCMoves:
    max_displacement_A: float = 0.5
    displacement_weight: float = 0.7
    swap_weight: float = 0.3


@dataclass
class HRMCCostWeights:
    xrd: float = 1.0
    ce_target: float = 5.0


@dataclass
class HRMCConfig:
    default_steps: int = 100
    xrd_interval: int = 5
    target_fps: int = 15
    temperature: HRMCTemperature = field(default_factory=HRMCTemperature)
    moves: HRMCMoves = field(default_factory=HRMCMoves)
    cost_weights: HRMCCostWeights = field(default_factory=HRMCCostWeights)


@dataclass
class BuckinghamParams:
    A_eV: float = 0.0
    rho_A: float = 0.0
    C_eVA6: float = 0.0


@dataclass
class PotentialConfig:
    cutoff_radius_A: float = 4.8
    coulomb_screening_kappa: float = 10.0
    use_buckingham: bool = True
    use_coulomb: bool = True
    library_path: str = "./potentials/library.meam"
    alloy_path: str = "./potentials/library.meam_alloy"
    buckingham: Dict[str, BuckinghamParams] = field(default_factory=dict)
    charges: Dict[str, float] = field(default_factory=dict)


@dataclass
class StructureConfig:
    max_supercell_dim: int = 4
    default_supercell: List[int] = field(default_factory=lambda: [2, 2, 2])


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level configuration container
# ═══════════════════════════════════════════════════════════════════════════════

class AtomInferConfig:
    """Singleton-style configuration loaded from TOML."""

    def __init__(self, raw: dict):
        self._raw = raw
        self.server = self._load_server(raw.get("server", {}))
        self.directories = self._load_directories(raw.get("directories", {}))
        self.llm = self._load_llm(raw.get("llm", {}))
        self.material = self._load_material(raw)
        self.xrd = self._load_xrd(raw.get("analysis", {}).get("xrd", {}))
        self.raman = self._load_raman(raw.get("analysis", {}).get("raman", {}))
        self.hrmc = self._load_hrmc(raw.get("analysis", {}).get("hrmc", {}))
        self.potentials = self._load_potentials(raw.get("potentials", {}))
        self.structure = self._load_structure(raw.get("structure", {}))
        self._ensure_directories()

    # ── Helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _g(d: dict, key: str, default=None):
        """Safe nested get."""
        return d.get(key, default) if d else default

    def _ensure_directories(self):
        for d in [self.directories.output, self.directories.uploads]:
            Path(d).mkdir(parents=True, exist_ok=True)

    # ── Loaders ────────────────────────────────────────────────────────────
    def _load_server(self, d: dict) -> ServerConfig:
        return ServerConfig(
            host=d.get("host", "0.0.0.0"),
            port=d.get("port", 8000),
            open_browser=d.get("open_browser", True),
        )

    def _load_directories(self, d: dict) -> DirectoryConfig:
        return DirectoryConfig(
            output=d.get("output", "./atomInfer_output"),
            uploads=d.get("uploads", "./uploads"),
            models=d.get("models", "./models"),
        )

    def _load_llm(self, d: dict) -> LLMConfig:
        models = {}
        for name, md in d.get("models", {}).items():
            if not isinstance(md, dict):
                continue
            models[name] = ModelProfile(
                name=name,
                provider=md.get("provider", "ollama"),
                model=md.get("model", ""),
                endpoint=md.get("endpoint", ""),
                api_key_env=md.get("api_key_env", ""),
            )
        return LLMConfig(
            temperature=d.get("temperature", 0.05),
            max_tokens=d.get("max_tokens", 4096),
            max_iterations=d.get("max_iterations", 15),
            models=models,
            task_assignments=d.get("task_assignments", {}),
        )

    def _load_material(self, raw: dict) -> MaterialConfig:
        mat_section = raw.get("materials", {})
        active_name = mat_section.get("active", "LiMn2O4")
        md = mat_section.get(active_name, {})

        doping_d = md.get("doping", {})
        vegard_d = doping_d.get("vegard", {})
        doping = DopingConfig(
            host_site=doping_d.get("host_site", "16d"),
            host_element=doping_d.get("host_element", "Mn"),
            allowed_dopants=doping_d.get("allowed_dopants", ["Ce"]),
            max_fraction=doping_d.get("max_fraction", 0.10),
            seed=doping_d.get("seed", 42),
            ionic_radii_A=doping_d.get("ionic_radii_A", {}),
            vegard_a_pristine_A=vegard_d.get("a_pristine_A", 8.2480),
            vegard_coefficient=vegard_d.get("coefficient", 0.35),
            vegard_min_fraction=vegard_d.get("min_fraction", 0.0),
            vegard_max_fraction=vegard_d.get("max_fraction", 0.10),
        )

        return MaterialConfig(
            name=active_name,
            formula=md.get("formula", active_name),
            mp_id=md.get("mp_id", ""),
            space_group=md.get("space_group", ""),
            space_group_number=md.get("space_group_number", 0),
            crystal_system=md.get("crystal_system", ""),
            structure_type=md.get("structure_type", ""),
            reference_lattice_A=md.get("reference_lattice_A", 0.0),
            wyckoff=md.get("wyckoff", {}),
            doping=doping,
        )

    def _load_xrd(self, d: dict) -> XRDConfig:
        rf = d.get("r_factor", {})
        phases = {}
        for pname, pd in d.get("secondary_phases", {}).items():
            if isinstance(pd, dict):
                phases[pname] = SecondaryPhase(
                    peak_positions_deg=pd.get("peak_positions_deg", []),
                    detection_threshold=pd.get("detection_threshold", 0.02),
                )
        return XRDConfig(
            wavelength_A=d.get("wavelength_A", 1.54059),
            scherrer_constant=d.get("scherrer_constant", 0.9),
            angular_range_deg=d.get("angular_range_deg", [10.0, 80.0]),
            r_factor=RFactorConfig(
                excellent=rf.get("excellent", 0.10),
                good=rf.get("good", 0.20),
                acceptable=rf.get("acceptable", 0.30),
            ),
            hkl_windows=d.get("hkl_windows", {}),
            secondary_phases=phases,
        )

    def _load_raman(self, d: dict) -> RamanConfig:
        return RamanConfig(
            wavenumber_range_cm1=d.get("wavenumber_range_cm1", [100.0, 1000.0]),
            reference_A1g_cm1=d.get("reference_A1g_cm1", 628.0),
            mode_windows=d.get("mode_windows", {}),
        )

    def _load_hrmc(self, d: dict) -> HRMCConfig:
        td = d.get("temperature", {})
        md = d.get("moves", {})
        cd = d.get("cost_weights", {})
        return HRMCConfig(
            default_steps=d.get("default_steps", 100),
            xrd_interval=d.get("xrd_interval", 5),
            target_fps=d.get("target_fps", 15),
            temperature=HRMCTemperature(
                start=td.get("start", 0.1),
                end=td.get("end", 0.01),
            ),
            moves=HRMCMoves(
                max_displacement_A=md.get("max_displacement_A", 0.5),
                displacement_weight=md.get("displacement_weight", 0.7),
                swap_weight=md.get("swap_weight", 0.3),
            ),
            cost_weights=HRMCCostWeights(
                xrd=cd.get("xrd", 1.0),
                ce_target=cd.get("ce_target", 5.0),
            ),
        )

    def _load_potentials(self, d: dict) -> PotentialConfig:
        meam_d = d.get("meam", {})
        buck = {}
        for pname, pd in d.get("buckingham", {}).items():
            if isinstance(pd, dict):
                buck[pname] = BuckinghamParams(
                    A_eV=pd.get("A_eV", 0.0),
                    rho_A=pd.get("rho_A", 0.0),
                    C_eVA6=pd.get("C_eVA6", 0.0),
                )
        return PotentialConfig(
            cutoff_radius_A=meam_d.get("cutoff_radius_A", 4.8),
            coulomb_screening_kappa=meam_d.get("coulomb_screening_kappa", 10.0),
            use_buckingham=meam_d.get("use_buckingham", True),
            use_coulomb=meam_d.get("use_coulomb", True),
            library_path=meam_d.get("library_path", "./potentials/library.meam"),
            alloy_path=meam_d.get("alloy_path", "./potentials/library.meam_alloy"),
            buckingham=buck,
            charges=d.get("charges", {}),
        )

    def _load_structure(self, d: dict) -> StructureConfig:
        return StructureConfig(
            max_supercell_dim=d.get("max_supercell_dim", 4),
            default_supercell=d.get("default_supercell", [2, 2, 2]),
        )

    # ── Convenience ────────────────────────────────────────────────────────
    def get_api_key(self, name: str) -> Optional[str]:
        """Get an API key, checking environment first, then config fallback."""
        env_map = {
            "groq":      "GROQ_API_KEY",
            "openai":    "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "mp":        "MP_API_KEY",
        }
        env_var = env_map.get(name, f"{name.upper()}_API_KEY")
        val = os.environ.get(env_var)
        if val:
            return val
        return self._raw.get("api_keys", {}).get(f"{name}_api_key")

    def get_hkl_windows_tuples(self) -> dict:
        """Convert string hkl keys "111" to tuple keys (1,1,1) for analysis code."""
        result = {}
        for hkl_str, (lo, hi) in self.xrd.hkl_windows.items():
            digits = [int(c) for c in hkl_str]
            if len(digits) == 3:
                result[tuple(digits)] = (lo, hi)
        return result

    def generate_system_prompt(self) -> str:
        """Generate an LLM system prompt from the active material profile."""
        mat = self.material
        dop = mat.doping
        xrd = self.xrd

        # Build peak reference string
        peak_ref_lines = []
        for hkl_str, (lo, hi) in xrd.hkl_windows.items():
            mid = (lo + hi) / 2
            peak_ref_lines.append(f"  ({hkl_str})~{mid:.1f}°")

        # Build ionic radii string
        radii_lines = []
        for ion, r in dop.ionic_radii_A.items():
            radii_lines.append(f"  {ion}: {r} Å")

        # Build Raman modes string
        raman_lines = []
        for mode, (lo, hi) in self.raman.mode_windows.items():
            mid = (lo + hi) / 2
            raman_lines.append(f"  {mode}: ~{mid:.0f} cm⁻¹ ({lo}-{hi})")

        dopant_list = ", ".join(dop.allowed_dopants)
        dopant = dop.allowed_dopants[0] if dop.allowed_dopants else "X"

        return f"""You are AtomInfer, an expert AI agent for computational materials science.
Your mission: analyze experimental characterization data and produce atomistic models
ready for molecular dynamics or DFT simulation.

## DOMAIN KNOWLEDGE

### Material system
You are working with {mat.formula} and its doped variants.
- Space group: {mat.space_group} ({mat.crystal_system} {mat.structure_type}, #{mat.space_group_number})
- Wyckoff positions: {', '.join(f'{el} at {site}' for site, el in mat.wyckoff.items())}
- {dopant} substitutes at the {dop.host_site} {dop.host_element} site only
- Pristine reference: a ≈ {mat.reference_lattice_A} Å (ICDD) — treat as literature reference only
- Materials Project ID: {mat.mp_id}
- Allowed dopants: {dopant_list}

### Ionic radii
{chr(10).join(radii_lines)}

### XRD analysis (λ = {xrd.wavelength_A} Å)
- Assign peaks using {mat.formula} {mat.space_group} reflection rules
- Key reflections:
{chr(10).join(peak_ref_lines)}
- Extract lattice parameter via Nelson-Riley extrapolation (use multiple peaks)
- Scherrer equation: D = Kλ / (β cosθ), K={xrd.scherrer_constant}, β=FWHM in radians

### Raman spectroscopy
{chr(10).join(raman_lines)}

### Structure building rules
1. Always fetch base structure from Materials Project ({mat.mp_id} for {mat.formula})
2. Scale lattice parameter to match your measured value — do not use MP lattice directly
3. {dopant} substitution: randomly replace x fraction of {dop.host_element} ({dop.host_site}) sites with {dopant}
4. For Scherrer size D nm: build supercell with edge length ≈ D nm (a × n ≈ D×10 Å)
5. Validate with XRD R-factor — R < {xrd.r_factor.acceptable} is acceptable for initial model

## WORKFLOW

You MUST follow this sequence. Do not skip steps.

STEP 1 — Parse all provided input files using parse_* tools
STEP 2 — Analyze each technique using analyze_* tools to extract parameters
STEP 3 — Cross-check parameters across techniques
STEP 4 — Query Materials Project for base structure (use {mat.mp_id} for {mat.formula})
STEP 5 — Build doped supercell using measured lattice parameter and dopant fraction
STEP 6 — Validate structure with XRD R-factor
STEP 7 — Report: phase, space group, lattice parameter, dopant %, crystallite size,
          output file paths, R-factor, and any secondary phases detected

## REASONING STYLE
- Think step by step. State what you expect before calling a tool.
- After each tool result, interpret it in 1-2 sentences before proceeding.
- If a tool returns an error, diagnose and try with corrected parameters.
- Never hardcode lattice parameters or dopant fractions — always derive from data.
- Be quantitatively precise. Report numbers with appropriate significant figures.
- Flag any physically unusual results and explain using domain knowledge."""

    def to_summary_dict(self) -> dict:
        """Return a JSON-serializable summary for the /health endpoint and UI."""
        return {
            "active_material": self.material.name,
            "material_formula": self.material.formula,
            "space_group": self.material.space_group,
            "mp_id": self.material.mp_id,
            "reference_lattice_A": self.material.reference_lattice_A,
            "llm_models": {
                name: {"provider": m.provider, "model": m.model}
                for name, m in self.llm.models.items()
            },
            "task_assignments": self.llm.task_assignments,
            "xrd_wavelength_A": self.xrd.wavelength_A,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(path: Optional[str] = None) -> AtomInferConfig:
    """Load configuration from TOML file.

    Priority:
      1. Explicit *path* argument
      2. ATOMINFER_CONFIG environment variable
      3. config.toml  in project root
      4. config.default.toml  in project root (always present)
    """
    if path:
        cfg_path = Path(path)
    elif os.environ.get("ATOMINFER_CONFIG"):
        cfg_path = Path(os.environ["ATOMINFER_CONFIG"])
    elif (_ROOT / "config.toml").exists():
        cfg_path = _ROOT / "config.toml"
    else:
        cfg_path = _ROOT / "config.default.toml"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    with open(cfg_path, "rb") as f:
        raw = tomllib.load(f)

    print(f"[config] Loaded {cfg_path.name}")
    return AtomInferConfig(raw)


# Global instance — import and use: from config_loader import cfg
cfg = load_config()
