#!/usr/bin/env python3
"""Stick + Gaussian-broadened Casida absorption spectrum plots."""
from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence, Union

import numpy as np

_HARTREE_TO_EV = 27.211386246


def gauss(x, x0, sigma):
    """Peak-normalized Gaussian (max = 1 at x = x0)."""
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def read_casida_results(path):
    energies = []
    osc = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                float(parts[0])
                e = float(parts[1])
                fosc = float(parts[2])
                energies.append(e)
                osc.append(fosc)
            except ValueError:
                continue
    if not energies:
        raise ValueError(f"No spectrum data found in {path}")
    return np.array(energies), np.array(osc)


def write_casida_spectrum_txt(
    path: str,
    energies_ev: np.ndarray,
    fosc: np.ndarray,
    is_coupled: Optional[np.ndarray] = None,
) -> str:
    """Write ``state energy_eV oscillator_strength [tag]`` lines for :func:`read_casida_results`."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    energies_ev = np.asarray(energies_ev, dtype=float).ravel()
    fosc = np.asarray(fosc, dtype=float).ravel()
    with open(path, "w") as f:
        f.write("# state  energy_eV  oscillator_strength  tag\n")
        for i, (e, fv) in enumerate(zip(energies_ev, fosc)):
            tag = ""
            if is_coupled is not None and i < len(is_coupled):
                tag = "coupled" if bool(is_coupled[i]) else "uncoupled"
            f.write(f"{i:5d}  {e:14.8f}  {fv:14.8f}  {tag}\n")
    return path


def plot_casida_spectrum(
    energies_ev: Union[np.ndarray, Sequence[float]],
    fosc: Union[np.ndarray, Sequence[float]],
    out: str = "casida_merged_spectrum.png",
    *,
    sigma: float = 0.08,
    emin: Optional[float] = None,
    emax: Optional[float] = None,
    npts: int = 4000,
    is_coupled: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    dpi: int = 160,
) -> str:
    """Save stick + broadened spectrum PNG. Energies must be in eV."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    energies_ev = np.asarray(energies_ev, dtype=float).ravel()
    fosc = np.asarray(fosc, dtype=float).ravel()
    if energies_ev.size != fosc.size:
        raise ValueError(
            f"len(energies)={energies_ev.size} != len(fosc)={fosc.size}",
        )
    if energies_ev.size == 0:
        raise ValueError("empty spectrum")

    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    emin = emin if emin is not None else max(0.0, float(energies_ev.min()) - 2.0 * sigma)
    emax = emax if emax is not None else float(energies_ev.max()) + 2.0 * sigma
    grid = np.linspace(emin, emax, npts)

    spectrum = np.zeros_like(grid)
    for e, f in zip(energies_ev, fosc):
        spectrum += f * gauss(grid, e, sigma)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    if is_coupled is not None and len(is_coupled) == len(energies_ev):
        ic = np.asarray(is_coupled, dtype=bool)
        if np.any(ic):
            ax1.stem(
                energies_ev[ic],
                fosc[ic],
                basefmt=" ",
                markerfmt="C0o",
                linefmt="C0-",
                label="coupled",
            )
        if np.any(~ic):
            ax1.stem(
                energies_ev[~ic],
                fosc[~ic],
                basefmt=" ",
                markerfmt="C3o",
                linefmt="C3-",
                label="uncoupled",
            )
        ax1.legend(loc="best")
    else:
        ax1.stem(energies_ev, fosc, basefmt=" ", markerfmt="ro", linefmt="r-")

    ax1.set_ylabel("Oscillator strength")
    ax1.set_title(title or "Casida stick spectrum")
    ax1.grid(alpha=0.3)

    ax2.plot(grid, spectrum, "b-", lw=1.8)
    ax2.fill_between(grid, spectrum, alpha=0.25)
    ax2.set_xlabel("Energy (eV)")
    ax2.set_ylabel("Absorption (arb. units)")
    ax2.set_title(f"Gaussian broadened (sigma = {sigma} eV)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_casida_spectrum_from_ha(
    omega_ha: Union[np.ndarray, Sequence[float]],
    fosc: Union[np.ndarray, Sequence[float]],
    out: str = "casida_merged_spectrum.png",
    **kwargs,
) -> str:
    """Like :func:`plot_casida_spectrum` with energies in Hartree."""
    omega_ha = np.asarray(omega_ha, dtype=float).ravel()
    return plot_casida_spectrum(omega_ha * _HARTREE_TO_EV, fosc, out=out, **kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_txt", help="casida_parallel_results.txt")
    p.add_argument("--sigma", type=float, default=0.8, help="Gaussian broadening in eV")
    p.add_argument("--emin", type=float, default=None)
    p.add_argument("--emax", type=float, default=None)
    p.add_argument("--npts", type=int, default=4000)
    p.add_argument("--out", default="casida_spectrum_sigma0.8eV.png")
    args = p.parse_args()

    energies, fosc = read_casida_results(args.results_txt)
    print(f"Loaded {len(energies)} states from {args.results_txt}")
    print(f"Sum of oscillator strengths: {fosc.sum():.6f}")

    out = plot_casida_spectrum(
        energies,
        fosc,
        out=args.out,
        sigma=args.sigma,
        emin=args.emin,
        emax=args.emax,
        npts=args.npts,
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
