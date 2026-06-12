#!/usr/bin/env python3
"""Plot uncoupled fragment Casida states from ``casida_uncoupled_spectrum.txt``.

Example::

    python plot_uncoupled_spectrum.py casida_uncoupled_spectrum.txt \\
        --sigma 0.08 --out casida_uncoupled_spectrum.png
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np

from casidapy.plot_casida_spectrum import gauss, plot_casida_spectrum


def read_uncoupled_spectrum_txt(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read eDFTpy ``write_uncoupled_excluded_txt`` output.

    Returns ``(subsystem, state_index, omega_ev, f)``.
    """
    subsystem: List[int] = []
    state_index: List[int] = []
    omega_ev: List[float] = []
    fosc: List[float] = []

    with open(path, "r") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 5:
                continue
            try:
                subsystem.append(int(parts[0]))
                state_index.append(int(parts[1]))
                omega_ev.append(float(parts[3]))
                fosc.append(float(parts[4]))
            except ValueError:
                continue

    if not omega_ev:
        raise ValueError(f"No uncoupled states found in {path}")

    return (
        np.asarray(subsystem, dtype=int),
        np.asarray(state_index, dtype=int),
        np.asarray(omega_ev, dtype=float),
        np.asarray(fosc, dtype=float),
    )


def plot_uncoupled_by_subsystem(
    omega_ev: np.ndarray,
    fosc: np.ndarray,
    subsystem: np.ndarray,
    out: str,
    *,
    sigma: float = 0.08,
    emin: Optional[float] = None,
    emax: Optional[float] = None,
    npts: int = 4000,
    dpi: int = 160,
) -> str:
    """Stick plot colored by subsystem + Gaussian-broadened curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    omega_ev = np.asarray(omega_ev, dtype=float).ravel()
    fosc = np.asarray(fosc, dtype=float).ravel()
    subsystem = np.asarray(subsystem, dtype=int).ravel()

    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    emin = emin if emin is not None else max(0.0, float(omega_ev.min()) - 2.0 * sigma)
    emax = emax if emax is not None else float(omega_ev.max()) + 2.0 * sigma
    grid = np.linspace(emin, emax, npts)
    spectrum = np.zeros_like(grid)
    for e, f in zip(omega_ev, fosc):
        spectrum += f * gauss(grid, e, sigma)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    frags = np.unique(subsystem)
    cmap = plt.cm.tab10
    for k, frag in enumerate(frags):
        mask = subsystem == frag
        ax1.stem(
            omega_ev[mask],
            fosc[mask],
            basefmt=" ",
            markerfmt=f"C{k % 10}o",
            linefmt=f"C{k % 10}-",
            label=f"subsystem {frag}",
        )
    ax1.set_ylabel("Oscillator strength")
    ax1.set_title("Uncoupled excluded Casida states (fragment-local)")
    ax1.legend(loc="best", fontsize=8, ncol=2)
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


def main():
    p = argparse.ArgumentParser(
        description="Plot uncoupled Casida states from casida_uncoupled_spectrum.txt",
    )
    p.add_argument(
        "uncoupled_txt",
        nargs="?",
        default="casida_uncoupled_spectrum.txt",
        help="Output of eDFTpy write_uncoupled_excluded_txt",
    )
    p.add_argument("--sigma", type=float, default=0.08, help="Gaussian width (eV)")
    p.add_argument("--emin", type=float, default=None)
    p.add_argument("--emax", type=float, default=None)
    p.add_argument("--npts", type=int, default=4000)
    p.add_argument(
        "--out",
        default=None,
        help="PNG path (default: <txt_basename>.png)",
    )
    p.add_argument(
        "--simple",
        action="store_true",
        help="Single-color plot via plot_casida_spectrum (no per-subsystem colors)",
    )
    args = p.parse_args()

    sub, st, omega_ev, fosc = read_uncoupled_spectrum_txt(args.uncoupled_txt)
    out = args.out
    if out is None:
        base = os.path.splitext(os.path.basename(args.uncoupled_txt))[0]
        out = base + ".png"

    print(f"Loaded {len(omega_ev)} uncoupled states from {args.uncoupled_txt}")
    print(f"Sum of oscillator strengths: {fosc.sum():.6f}")

    if args.simple:
        plot_casida_spectrum(
            omega_ev,
            fosc,
            out=out,
            sigma=args.sigma,
            emin=args.emin,
            emax=args.emax,
            npts=args.npts,
            title="Uncoupled excluded Casida states",
        )
    else:
        plot_uncoupled_by_subsystem(
            omega_ev,
            fosc,
            sub,
            out,
            sigma=args.sigma,
            emin=args.emin,
            emax=args.emax,
            npts=args.npts,
        )
    print(f"Saved: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
