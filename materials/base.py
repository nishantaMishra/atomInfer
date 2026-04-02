"""
Material Profile Base
=====================
Abstract base class for material profiles. The default implementation
reads everything from config.toml (via config_loader). Users can subclass
to add custom analysis logic for new material systems.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional


class MaterialProfile:
    """Base material profile backed by AtomInferConfig.material."""

    def __init__(self, material_cfg):
        """
        Args:
            material_cfg: A config_loader.MaterialConfig instance.
        """
        self._cfg = material_cfg

    # ── Identity ──────────────────────────────────────────────────────────
    @property
    def name(self) -> str:
        return self._cfg.name

    @property
    def formula(self) -> str:
        return self._cfg.formula

    @property
    def mp_id(self) -> str:
        return self._cfg.mp_id

    @property
    def space_group(self) -> str:
        return self._cfg.space_group

    @property
    def space_group_number(self) -> int:
        return self._cfg.space_group_number

    @property
    def crystal_system(self) -> str:
        return self._cfg.crystal_system

    @property
    def structure_type(self) -> str:
        return self._cfg.structure_type

    @property
    def reference_lattice_A(self) -> float:
        return self._cfg.reference_lattice_A

    # ── Wyckoff sites ─────────────────────────────────────────────────────
    @property
    def wyckoff(self) -> Dict[str, str]:
        return self._cfg.wyckoff

    # ── Doping ────────────────────────────────────────────────────────────
    @property
    def doping(self):
        return self._cfg.doping

    @property
    def host_element(self) -> str:
        return self._cfg.doping.host_element

    @property
    def host_site(self) -> str:
        return self._cfg.doping.host_site

    @property
    def allowed_dopants(self) -> List[str]:
        return self._cfg.doping.allowed_dopants

    @property
    def default_dopant(self) -> str:
        return self._cfg.doping.allowed_dopants[0] if self._cfg.doping.allowed_dopants else ""

    @property
    def ionic_radii_A(self) -> Dict[str, float]:
        return self._cfg.doping.ionic_radii_A

    # ── Vegard's law ──────────────────────────────────────────────────────
    def estimate_dopant_fraction(self, measured_lattice_A: float) -> float:
        """Estimate dopant fraction from measured lattice using Vegard's law."""
        d = self._cfg.doping
        raw = (measured_lattice_A - d.vegard_a_pristine_A) / d.vegard_coefficient
        return max(d.vegard_min_fraction, min(d.vegard_max_fraction, raw))

    # ── Display ───────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return f"MaterialProfile({self.name}: {self.formula}, {self.space_group})"
