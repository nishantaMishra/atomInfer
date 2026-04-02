"""
Material Profile Registry
=========================
Loads the active material profile from configuration.
"""

from __future__ import annotations
from typing import Dict
from materials.base import MaterialProfile
from config_loader import cfg


def get_active_material() -> MaterialProfile:
    """Return the currently active MaterialProfile based on config."""
    return MaterialProfile(cfg.material)


def list_materials() -> Dict[str, str]:
    """List all material profiles defined in config.

    Returns dict of {name: formula}.
    """
    mat_section = cfg._raw.get("materials", {})
    result = {}
    for key, val in mat_section.items():
        if isinstance(val, dict) and "formula" in val:
            result[key] = val["formula"]
    return result
