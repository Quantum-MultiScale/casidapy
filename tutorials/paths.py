"""Shared paths for CasidaPy tutorial notebooks.

Pseudos and ``setup_env.sh`` live in the tutorials root. Notebooks may sit in
subfolders (``tddft/``, ``xas/``, ``qed/``, ``soc/``).
"""
from __future__ import annotations

from pathlib import Path

# Absolute root of the tutorials package (this file's directory)
ROOT = Path(__file__).resolve().parent

# Norm-conserving ONCV PP files used by H₂O / XAS demos
O_UPF = ROOT / "O_ONCV_PBE-1.2.upf"
H_UPF = ROOT / "H_ONCV_PBE-1.2.upf"
SETUP_ENV = ROOT / "setup_env.sh"


def require_oncv_pseudos() -> Path:
    """Return tutorials root after asserting O/H ONCV UPFs exist."""
    if not O_UPF.is_file() or not H_UPF.is_file():
        raise FileNotFoundError(
            f"Missing ONCV UPFs under {ROOT} "
            f"(need {O_UPF.name} and {H_UPF.name})"
        )
    return ROOT
