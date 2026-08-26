"""Plot / summarize XAS stick spectra from :class:`~casidapy.casida_api.CasidaResults`.

Typical use::

    from casidapy.xas import plot_sticks, summarize_xas, omega_ev
    print(summarize_xas(res, core))
    plot_sticks(res, normalize=True, broaden_sigma_ev=0.3)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from casidapy.xas.cvs import CoreOrbitals

HA_TO_EV = 27.211386245988


def omega_ev(results) -> np.ndarray:
    """Excitation energies in eV from a :class:`~casidapy.casida_api.CasidaResults`."""
    return np.asarray(results.omega, dtype=float) * HA_TO_EV


def stick_spectrum(results, *, max_states: Optional[int] = None, normalize: bool = False):
    """Return ``(omega_ev, f)`` arrays, optionally truncated / ∑f-normalized."""
    w = omega_ev(results)
    f = np.asarray(getattr(results, "f", None), dtype=float)
    if f.size == 0:
        f = np.ones_like(w)
    if max_states is not None:
        n = int(max_states)
        w, f = w[:n], f[:n]
    if normalize:
        s = float(np.sum(f))
        if s > 0.0:
            f = f / s
    return w, f


def plot_sticks(
    results,
    *,
    ax=None,
    label: Optional[str] = None,
    color=None,
    max_states: Optional[int] = None,
    broaden_sigma_ev: float = 0.0,
    energy_shift_ev: float = 0.0,
    normalize: bool = False,
):
    """Plot stick (and optional Gaussian-broadened) spectrum.

    If ``normalize`` is True, oscillator strengths are scaled so ``∑f = 1``
    (useful when overlaying GTO vs PW hosts with different absolute scales).

    Returns ``(fig, ax)``. Requires matplotlib.
    """
    import matplotlib.pyplot as plt

    w, f = stick_spectrum(results, max_states=max_states, normalize=normalize)
    w = w + float(energy_shift_ev)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
    else:
        fig = ax.figure

    ymin = 0.0
    first = True
    for wi, fi in zip(w, f):
        ax.vlines(
            wi,
            ymin,
            fi,
            colors=color,
            linewidth=1.2,
            label=label if first else None,
        )
        first = False

    if broaden_sigma_ev and broaden_sigma_ev > 0:
        grid = np.linspace(float(w.min()) - 2, float(w.max()) + 2, 800)
        sig = float(broaden_sigma_ev)
        spectrum = np.zeros_like(grid)
        for wi, fi in zip(w, f):
            spectrum += fi * np.exp(-0.5 * ((grid - wi) / sig) ** 2)
        ax.plot(grid, spectrum, color=color, alpha=0.7)

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Oscillator strength" + (" (normalized)" if normalize else ""))
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=8)
    return fig, ax


def summarize_xas(results, core: Optional[CoreOrbitals] = None, *, n_print: int = 8) -> str:
    """One-line + table string for notebook display."""
    w = omega_ev(results)
    lines = [
        f"n_states={w.size}  ω₀={w[0]:.2f} eV  ω_last={w[-1]:.2f} eV",
    ]
    if core is not None:
        eps = float(np.min(core.energies))
        lines.append(
            f"edge={core.edge} shell={core.shell}  n_core={core.energies.size}  "
            f"ε_core={eps:.3f} Ha ({eps * HA_TO_EV:.1f} eV)"
        )
    lines.append("first roots (eV): " + ", ".join(f"{x:.2f}" for x in w[:n_print]))
    return "\n".join(lines)


__all__ = [
    "HA_TO_EV",
    "omega_ev",
    "stick_spectrum",
    "plot_sticks",
    "summarize_xas",
]
