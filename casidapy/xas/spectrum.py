"""Plot / summarize XAS stick spectra from :class:`~casidapy.casida_api.CasidaResults`.

Typical use::

    from casidapy.xas import plot_sticks, summarize_xas, omega_ev
    print(summarize_xas(res, core))
    plot_sticks(res, normalize=True, broaden_sigma_ev=0.3)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from casidapy.xas.chemical_data import DEFAULTS
from casidapy.xas.cvs import CoreOrbitals

HA_TO_EV = DEFAULTS["ha_to_ev"]


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


def stick_cartesian(results, *, max_states: Optional[int] = None):
    """``(omega_ev, f, fx, fy, fz)`` from ``d_mode``.

    Isotropic ``f = (2/3) ω |μ|²``. Polarized ``f_α = 2 ω μ_α²`` so
    ``(fx + fy + fz) / 3 = f``. Graphene/graphite: ``fz`` is E ⊥ plane,
    ``(fx+fy)/2`` is E ∥ plane (one in-plane Cartesian).
    """
    w = omega_ev(results)
    w_ha = np.asarray(results.omega, dtype=float).ravel()
    f = np.asarray(getattr(results, "f", None), dtype=float).ravel()
    if f.size != w.size:
        f = np.zeros_like(w)
    d_mode = getattr(results, "d_mode", None)
    if d_mode is None:
        z = np.zeros_like(w)
        return w, f, z, z, z
    d = np.asarray(d_mode, dtype=float)
    if d.ndim != 2:
        z = np.zeros_like(w)
        return w, f, z, z, z
    if d.shape[0] == 3:
        mx, my, mz = d[0], d[1], d[2]
    elif d.shape[1] == 3:
        mx, my, mz = d[:, 0], d[:, 1], d[:, 2]
    else:
        z = np.zeros_like(w)
        return w, f, z, z, z
    fx = 2.0 * w_ha * np.square(np.asarray(mx, float).ravel())
    fy = 2.0 * w_ha * np.square(np.asarray(my, float).ravel())
    fz = 2.0 * w_ha * np.square(np.asarray(mz, float).ravel())
    if max_states is not None:
        n = int(max_states)
        w, f, fx, fy, fz = w[:n], f[:n], fx[:n], fy[:n], fz[:n]
    return w, f, fx, fy, fz


def sum_d2(results) -> float:
    """Σ_n |d_n|² from ``d_mode``.

    For a symmetric TDA ``A`` with a complete orthonormal eigenvector set
    this equals Σ_ia |μ_ia|² and is independent of ``A`` (off / fft / bare
    / screened, any ``c_x``).  Use it as an implementation gate, not a
    spectral judgement.  Incomplete LOBPCG roots will not satisfy it.
    """
    d_mode = getattr(results, "d_mode", None)
    if d_mode is None:
        raise ValueError("sum_d2 requires results.d_mode")
    d = np.asarray(d_mode, dtype=float)
    if d.ndim != 2:
        raise ValueError(f"d_mode must be 2-D, got shape {d.shape}")
    if d.shape[0] == 3:
        return float(np.sum(d * d))
    if d.shape[1] == 3:
        return float(np.sum(d * d))
    raise ValueError(f"d_mode shape {d.shape} is not (3,n) or (n,3)")



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


def first_peak_shift_ev(reference, results, *, index: int = 0) -> float:
    """Rigid excitation-energy shift (eV) aligning ``results`` to ``reference``.

    Returns ``omega_ref[index] - omega[index]`` so plotting with
    ``energy_shift_ev=first_peak_shift_ev(...)`` overlays the leading stick.
    """
    w_ref = omega_ev(reference)
    w = omega_ev(results)
    i = int(index)
    if i < 0 or i >= w_ref.size or i >= w.size:
        raise IndexError(
            f"index={i} out of range for reference ({w_ref.size}) / results ({w.size})"
        )
    return float(w_ref[i] - w[i])


def _abs_dipole_au(results) -> np.ndarray:
    """Length-gauge |μ_n| (a.u.) from ``d_mode`` or inverted from ``f``."""
    w_ha = np.asarray(results.omega, dtype=float).ravel()
    f = np.asarray(getattr(results, "f", None), dtype=float).ravel()
    if f.size != w_ha.size:
        f = np.zeros_like(w_ha)
    d_mode = getattr(results, "d_mode", None)
    if d_mode is not None:
        d = np.asarray(d_mode, dtype=float)
        if d.ndim == 2 and d.shape[0] == 3:
            return np.linalg.norm(d, axis=0)
        if d.ndim == 2 and d.shape[1] == 3:
            return np.linalg.norm(d, axis=1)
    # f = (2/3) ω |μ|²  →  |μ| = sqrt(3f / 2ω)
    out = np.zeros_like(w_ha)
    ok = (w_ha > 0.0) & (f >= 0.0)
    out[ok] = np.sqrt(1.5 * f[ok] / w_ha[ok])
    return out


def oscillator_table(results, *, n_print: int = 12) -> str:
    """ω, raw oscillator strength, and |μ| for the leading sticks."""
    w = omega_ev(results)
    f = np.asarray(getattr(results, "f", None), dtype=float).ravel()
    if f.size != w.size:
        f = np.zeros_like(w)
    mu = _abs_dipole_au(results)
    n = min(int(n_print), w.size)
    lines = ["   i     omega_eV            f         |μ| (a.u.)"]
    for i in range(n):
        lines.append(f"{i+1:4d}  {w[i]:12.4f}  {f[i]:.6e}  {mu[i]:.6e}")
    if w.size > n:
        lines.append(f"  ... {w.size - n} more roots  ω_last={w[-1]:.4f} eV")
    lines.append(
        f"∑f (shown {n}) = {float(f[:n].sum()):.6e}   "
        f"∑f (all {f.size}) = {float(f.sum()):.6e}"
    )
    return "\n".join(lines)


def brightest_peak_shift_ev(reference, results) -> float:
    """Rigid shift aligning the brightest stick of ``results`` to ``reference``."""
    w_ref = omega_ev(reference)
    w = omega_ev(results)
    f_ref = np.asarray(getattr(reference, "f", None), dtype=float).ravel()
    f = np.asarray(getattr(results, "f", None), dtype=float).ravel()
    if f_ref.size != w_ref.size:
        f_ref = np.ones_like(w_ref)
    if f.size != w.size:
        f = np.ones_like(w)
    return float(w_ref[int(np.argmax(f_ref))] - w[int(np.argmax(f))])


def summarize_xas(results, core: Optional[CoreOrbitals] = None, *, n_print: int = 8) -> str:
    """One-line + oscillator-strength table for notebook display."""
    w = omega_ev(results)
    f = np.asarray(getattr(results, "f", None), dtype=float).ravel()
    lines = [
        f"n_states={w.size}  ω₀={w[0]:.2f} eV  ω_last={w[-1]:.2f} eV",
    ]
    if f.size == w.size and w.size:
        lines[0] += f"  ∑f={float(f.sum()):.4e}  f_max={float(f.max()):.4e}"
    if core is not None:
        eps = float(np.min(core.energies))
        lines.append(
            f"edge={core.edge} shell={core.shell}  n_core={core.energies.size}  "
            f"ε_core={eps:.3f} Ha ({eps * HA_TO_EV:.1f} eV)"
        )
    lines.append(oscillator_table(results, n_print=n_print))
    return "\n".join(lines)


__all__ = [
    "HA_TO_EV",
    "omega_ev",
    "stick_spectrum",
    "stick_cartesian",
    "sum_d2",
    "plot_sticks",
    "first_peak_shift_ev",
    "brightest_peak_shift_ev",
    "oscillator_table",
    "summarize_xas",
]
