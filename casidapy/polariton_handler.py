"""
Polariton postprocessing module.

This module provides a unified interface for polaritonic postprocessing of
Casida TDDFT results, including cavity-exciton coupling and polariton spectrum
computation.
"""

import os
import numpy as np

# Constants from polariton_postprocess.py
HA_TO_EV = 27.2114
EPS0_SI = 8.8541878128e-12       # F/m
HBAR_SI = 1.054571817e-34        # J*s
E_CHARGE_SI = 1.602176634e-19    # C
AU_DIPOLE_TO_C_M = 8.4783536255e-30  # 1 (e a0) in C*m


def lorentzian(x, x0, gamma):
    """Lorentzian lineshape function."""
    gamma = max(float(gamma), 1.0e-12)
    return (gamma / np.pi) / ((x - x0) ** 2 + gamma ** 2)


def build_couplings(
    fvals,
    model,
    g0,
    g_file,
    omega_ev,
    mu_transition,
    mu_units,
    wc_ev,
    mode_volume_m3,
    eps_r,
    orient_factor,
):
    """
    Build exciton-photon coupling strengths.
    
    This function computes the coupling strengths g_i between excitons and
    cavity photons using various models.
    
    Parameters
    ----------
    fvals : array
        Oscillator strengths
    model : str
        Coupling model: 'sqrt-f', 'uniform', 'file', or 'dipole-vac'
    g0 : float
        Global coupling scale (eV)
    g_file : str or None
        File containing coupling values (for 'file' model)
    omega_ev : array
        Excitation energies (eV)
    mu_transition : array or None
        Transition dipole moments (required for 'dipole-vac')
    mu_units : str
        Units of mu_transition: 'au' or 'si'
    wc_ev : float
        Cavity photon energy (eV)
    mode_volume_m3 : float or None
        Cavity mode volume in m³ (required for 'dipole-vac')
    eps_r : float
        Relative permittivity
    orient_factor : float
        Dipole-cavity orientation factor
        
    Returns
    -------
    g : array
        Coupling strengths (eV)
    """
    n = len(fvals)
    if n == 0:
        raise ValueError("No excitations were selected.")

    if model == "sqrt-f":
        fclip = np.clip(fvals, 0.0, None)
        s = np.sum(fclip)
        if s <= 0.0:
            return np.full(n, float(g0) / np.sqrt(n))
        return float(g0) * np.sqrt(fclip / s)

    if model == "uniform":
        return np.full(n, float(g0) / np.sqrt(n))

    if model == "file":
        if not g_file:
            raise ValueError("--g-file is required when --coupling-model=file")
        g = np.loadtxt(g_file, ndmin=1)
        g = np.asarray(g, dtype=float).reshape(-1)
        if len(g) != n:
            raise ValueError(f"Coupling file length {len(g)} != selected states {n}")
        return g

    if model == "dipole-vac":
        if mu_transition is None:
            raise ValueError("--mu-transition is required when --coupling-model=dipole-vac")
        if mode_volume_m3 is None:
            raise ValueError("--mode-volume-m3 is required when --coupling-model=dipole-vac")
        if eps_r <= 0.0:
            raise ValueError("--eps-r must be positive")
        if mode_volume_m3 <= 0.0:
            raise ValueError("--mode-volume-m3 must be positive")

        mu_arr = np.asarray(mu_transition)
        if mu_arr.ndim == 2 and mu_arr.shape[1] == 3:
            mu_mag = np.linalg.norm(mu_arr[:n], axis=1)
        else:
            mu_mag = mu_arr[:n].reshape(-1)
        if len(mu_mag) != n:
            raise ValueError(f"mu_transition length {len(mu_mag)} != selected states {n}")

        # Convert transition dipole magnitudes to SI
        if mu_units == "au":
            mu_si = mu_mag * AU_DIPOLE_TO_C_M
        elif mu_units == "si":
            mu_si = mu_mag
        else:
            raise ValueError(f"Unknown mu units: {mu_units}")

        # E_vac = sqrt(hbar*omega_c / (2 eps0 eps_r Veff)), omega_c in rad/s
        omega_c_rad_s = (wc_ev * E_CHARGE_SI) / HBAR_SI
        e_vac = np.sqrt((HBAR_SI * omega_c_rad_s) / (2.0 * EPS0_SI * eps_r * mode_volume_m3))

        # Coupling energy used in Hamiltonian (eV): g_i = mu_i * E_vac / e
        g_ev = np.abs(orient_factor) * (mu_si * e_vac / E_CHARGE_SI)
        return g_ev

    raise ValueError(f"Unknown coupling model: {model}")


def setup_polariton(args, omega, f, mu_transition=None, comm=None):
    """
    Setup polariton postprocessing.
    
    This function builds the cavity-exciton Hamiltonian, diagonalizes it,
    and computes polariton properties.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments. Must include:
        - polariton : bool
            Whether to enable polariton postprocessing
        - wc : float
            Cavity photon energy (eV)
        - g0 : float
            Global coupling scale (eV)
        - coupling_model : str
            Coupling model ('sqrt-f', 'uniform', 'file', 'dipole-vac')
        - g_file : str or None
            File with coupling values
        - mu_units : str
            Units of transition dipoles
        - mode_volume_m3 : float or None
            Cavity mode volume
        - eps_r : float
            Relative permittivity
        - orient_factor : float
            Orientation factor
        - max_states : int or None
            Maximum number of states to include
    omega : array
        Excitation energies (Hartree)
    f : array
        Oscillator strengths
    mu_transition : array or None
        Transition dipole moments
    comm : MPI communicator, optional
        
    Returns
    -------
    polariton_data : dict or None
        Dictionary containing:
        - energies : array
            Polariton energies (eV)
        - eigenvectors : array
            Polariton eigenvectors
        - photonic_fraction : array
            Photonic fraction of each polariton
        - excitonic_fraction : array
            Excitonic fraction of each polariton
        - oscillator_strength : array
            Effective oscillator strengths
        - couplings : array
            Coupling strengths (eV)
        Returns None if polariton is not enabled
    """
    rank = 0 if comm is None else comm.Get_rank()
    
    # Check if polariton is enabled
    if not hasattr(args, 'polariton') or not args.polariton:
        return None
    
    # Convert energies to eV
    omega_ev = omega * HA_TO_EV
    
    # Limit number of states
    n = min(len(omega), len(f))
    max_states = getattr(args, 'max_states', None)
    if max_states is not None:
        n = min(n, max_states)
    omega_ev = omega_ev[:n]
    f_sel = f[:n]
    
    # Build couplings
    g = build_couplings(
        fvals=f_sel,
        model=getattr(args, 'coupling_model', 'sqrt-f'),
        g0=getattr(args, 'g0', 0.10),
        g_file=getattr(args, 'g_file', None),
        omega_ev=omega_ev,
        mu_transition=mu_transition,
        mu_units=getattr(args, 'mu_units', 'au'),
        wc_ev=args.wc,
        mode_volume_m3=getattr(args, 'mode_volume_m3', None),
        eps_r=getattr(args, 'eps_r', 1.0),
        orient_factor=getattr(args, 'orient_factor', 1.0),
    )
    
    # Build Hamiltonian: one photon + N excitons
    hdim = n + 1
    H = np.zeros((hdim, hdim), dtype=float)
    H[0, 0] = args.wc
    H[1:, 1:] = np.diag(omega_ev)
    H[0, 1:] = g
    H[1:, 0] = g
    
    # Diagonalize
    e_pol, U = np.linalg.eigh(H)  # columns are eigenvectors
    
    # Compute fractions
    phot_frac = np.abs(U[0, :]) ** 2
    exc_frac = 1.0 - phot_frac
    
    # Effective oscillator strength
    f_pol = np.sum((np.abs(U[1:, :]) ** 2) * f_sel[:, None], axis=0)
    
    # Create frequency grid for broadened spectrum
    energy_min = np.min(e_pol)
    energy_max = np.max(e_pol)
    energy_range = energy_max - energy_min
    padding = max(energy_range * 0.05, 0.5)  # Min 0.5 eV padding
    x_min = max(0.0, energy_min - padding)
    x_max = energy_max + padding
    
    npts = getattr(args, 'npts', 2000)
    npts = max(npts, int((x_max - x_min) / 0.01))  # Ensure sufficient points
    grid_ev = np.linspace(x_min, x_max, npts)
    
    # Compute broadened spectra
    gamma_exc = getattr(args, 'gamma_exc', 0.05)
    gamma_pol = getattr(args, 'gamma_pol', 0.05)
    
    bare_spectrum = np.zeros_like(grid_ev)
    pol_spectrum = np.zeros_like(grid_ev)
    for i in range(n):
        bare_spectrum += f_sel[i] * lorentzian(grid_ev, omega_ev[i], gamma_exc)
    for m in range(hdim):
        pol_spectrum += f_pol[m] * lorentzian(grid_ev, e_pol[m], gamma_pol)
    
    return {
        'energies': e_pol,
        'eigenvectors': U,
        'photonic_fraction': phot_frac,
        'excitonic_fraction': exc_frac,
        'oscillator_strength': f_pol,
        'couplings': g,
        'n_states': n,
        'spectrum_grid_ev': grid_ev,
        'spectrum_bare': bare_spectrum,
        'spectrum_polariton': pol_spectrum,
    }


def save_polariton_results(polariton_data, output_base, args, comm=None):
    """
    Save polariton postprocessing results to files.
    
    Parameters
    ----------
    polariton_data : dict
        Polariton data from setup_polariton()
    output_base : str
        Base output filename (without extension)
    args : argparse.Namespace
        Command-line arguments
    comm : MPI communicator, optional
        Only rank 0 saves files
    """
    rank = 0 if comm is None else comm.Get_rank()
    
    if rank != 0:
        return
    
    if polariton_data is None:
        return
    
    # Save arrays
    np.save(f"{output_base}_polariton_energies_ev.npy", polariton_data['energies'])
    np.save(f"{output_base}_polariton_eigenvectors.npy", polariton_data['eigenvectors'])
    np.save(f"{output_base}_polariton_photonic_fraction.npy", polariton_data['photonic_fraction'])
    np.save(f"{output_base}_polariton_excitonic_fraction.npy", polariton_data['excitonic_fraction'])
    np.save(f"{output_base}_polariton_osc_strength.npy", polariton_data['oscillator_strength'])
    
    if 'spectrum_grid_ev' in polariton_data:
        np.save(f"{output_base}_polariton_spectrum_grid_ev.npy", polariton_data['spectrum_grid_ev'])
        np.save(f"{output_base}_polariton_spectrum_bare.npy", polariton_data['spectrum_bare'])
        np.save(f"{output_base}_polariton_spectrum_polariton.npy", polariton_data['spectrum_polariton'])
    
    # Save summary
    with open(f"{output_base}_polariton_summary.txt", "w") as fout:
        fout.write("# Polaritonic Postprocessing Summary\n")
        fout.write(f"# selected_states: {polariton_data['n_states']}\n")
        fout.write(f"# cavity_energy_wc_eV: {args.wc:.6f}\n")
        fout.write(f"# coupling_model: {getattr(args, 'coupling_model', 'sqrt-f')}\n")
        fout.write(f"# g0_eV: {getattr(args, 'g0', 0.10):.6f}\n")
        if getattr(args, 'coupling_model', 'sqrt-f') == "dipole-vac":
            fout.write(f"# mu_units: {getattr(args, 'mu_units', 'au')}\n")
            fout.write(f"# mode_volume_m3: {getattr(args, 'mode_volume_m3', None)}\n")
            fout.write(f"# eps_r: {getattr(args, 'eps_r', 1.0)}\n")
            fout.write(f"# orient_factor: {getattr(args, 'orient_factor', 1.0)}\n")
        fout.write(f"# g_min_eV: {np.min(polariton_data['couplings']):.6e}\n")
        fout.write(f"# g_max_eV: {np.max(polariton_data['couplings']):.6e}\n")
        fout.write(f"# gamma_exc_eV: {getattr(args, 'gamma_exc', 0.05):.6f}\n")
        fout.write(f"# gamma_pol_eV: {getattr(args, 'gamma_pol', 0.05):.6f}\n")
        fout.write("#\n")
        fout.write("# mode  energy_eV  photon_frac  exciton_frac  eff_osc_strength\n")
        for i, e in enumerate(polariton_data['energies']):
            fout.write(f"{i:4d}  {e:9.5f}  {polariton_data['photonic_fraction'][i]:11.6f}  "
                      f"{polariton_data['excitonic_fraction'][i]:11.6f}  "
                      f"{polariton_data['oscillator_strength'][i]:14.6f}\n")
    
    # Plot if requested
    if getattr(args, 'plot', False):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            if 'spectrum_grid_ev' in polariton_data:
                ax[0].plot(polariton_data['spectrum_grid_ev'], polariton_data['spectrum_bare'], 
                          label="Bare excitonic", linewidth=1.8)
                ax[0].plot(polariton_data['spectrum_grid_ev'], polariton_data['spectrum_polariton'], 
                          label="Polaritonic", linewidth=1.8)
            else:
                # Fallback: plot stick spectrum
                ax[0].stem(polariton_data['energies'], polariton_data['oscillator_strength'], 
                          basefmt=" ", markerfmt="ro", linefmt="r-", label="Polaritons")
            
            ax[0].set_ylabel("Intensity (arb.)")
            ax[0].legend()
            ax[0].grid(alpha=0.3)
            ax[0].set_title("Bare vs Polaritonic Spectrum")
            
            marker_sizes = 20 + 140 * polariton_data['photonic_fraction'] / max(np.max(polariton_data['photonic_fraction']), 1.0e-12)
            ax[1].scatter(polariton_data['energies'], polariton_data['oscillator_strength'], 
                         s=marker_sizes, alpha=0.8, label="Polaritons")
            ax[1].axvline(args.wc, linestyle="--", linewidth=1.0, label="Cavity mode")
            ax[1].set_xlabel("Energy (eV)")
            ax[1].set_ylabel("Eff. osc. strength")
            ax[1].grid(alpha=0.3)
            ax[1].legend()
            
            fig.tight_layout()
            fig.savefig(f"{output_base}_polariton_comparison.png", dpi=150)
        except ImportError:
            pass  # matplotlib not available, skip plot
