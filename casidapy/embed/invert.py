"""ZMP density inversion and the OEP embedding potential built from it.

Two functions:

``invert_xc_potential_qepy``
    The inversion. Finds ``v`` (QE external XC slot) whose KS density matches
    a target ``ρ_ref`` by the Anderson-accelerated ZMP fixed point::

        v ← v + α v_H[ρ[v] − ρ_ref]

``build_invert_embedding``
    Hirshfeld-peel the absorber density ``ρ_A``, invert it, and assemble::

        V_env = V_ionic + v_H[ρ_env] + v_corr

Settings are fixed constants below — change them here, not at call sites.
"""
from __future__ import annotations

import copy
import logging
import os
from collections import deque
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from casidapy.embed.potential import (
    DEFAULT_GAUGE_ALIGN,
    DEFAULT_VLOC_SOURCE,
    ensure_charge_grid,
    ionic_vloc_residual,
    ry_to_ha,
    shift_v_env_gauge_at,
)

logger = logging.getLogger(__name__)

MAX_ITER = 40
INNER_MAXITER = 100  # QE's internal SCF niter cap for each ρ[v] solve
INNER_TR2 = 1.0e-8  # tight QE SCF density threshold, used near outer convergence
INNER_TR2_LOOSE = 1.0e-4  # loose inner threshold while the outer residual is large
TOL = 1.0e-3  # production peels plateau ~5e-4; CO still reaches this easily
ALPHA0 = 2.5  # ZMP mixing; grows back toward this after a good step
ALPHA_MIN = 0.1  # floor for the α backtracking on a rejected step
# Consecutive rejects at α = ALPHA_MIN → stop and return best (α-floor stall).
STALL_REJECTS = 4
# Stop when best residual fails to improve by ≥5% for this many outer iters.
PLATEAU_ITERS = 8
PLATEAU_MIN_IMPROVE = 0.05
NBND_EXTRA = 8  # empty bands above the subsystem's occupied states
ANDERSON_M = 5  # residual-history depth for Anderson mixing (0 → plain steps)
ANDERSON_REG = 1e-10  # Tikhonov ridge on the Anderson least-squares solve

_V_CORR_CACHE_NAME = "v_corr_ry.npy"
_V_CORR_META_NAME = "v_corr_meta.json"


def _inner_mode() -> str:
    """``INVERT_INNER`` env: ``scf`` (default) or ``diag`` (single diagonalize)."""
    m = str(os.environ.get("INVERT_INNER", "scf")).lower().strip()
    if m in ("diag", "diagonalize", "diag_only", "once"):
        return "diag"
    return "scf"


def _unquote_qe(val) -> str:
    s = str(val).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    return s


def _cache_dir(workdir: Optional[str | Path] = None) -> Path:
    """Persistent invert cache directory (``INVERT_CACHE_DIR`` or invert workdir)."""
    env = os.environ.get("INVERT_CACHE_DIR", "").strip()
    if env:
        p = Path(env)
    elif workdir is not None:
        p = Path(workdir)
    else:
        p = Path(os.environ.get("TMPDIR") or ".") / "qepy_invert"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_v_corr_checkpoint(v_corr_ry, meta: dict, workdir: Optional[str | Path] = None) -> Path:
    """Write ``v_corr_ry.npy`` + JSON meta so a later job can skip re-invert."""
    import json

    cdir = _cache_dir(workdir)
    v = np.asarray(v_corr_ry, dtype=float).ravel()
    np.save(cdir / _V_CORR_CACHE_NAME, v)
    payload = {
        "nnr": int(v.size),
        "nelec": meta.get("nelec"),
        "err": meta.get("err"),
        "tol": meta.get("tol"),
        "converged": meta.get("converged"),
        "stalled": meta.get("stalled"),
        "usable": meta.get("usable"),
        "inner": meta.get("inner"),
        "n_iter": meta.get("n_iter"),
    }
    (cdir / _V_CORR_META_NAME).write_text(json.dumps(payload, indent=2))
    print(f"[invert] checkpoint → {cdir / _V_CORR_CACHE_NAME}", flush=True)
    return cdir


def load_v_corr_checkpoint(
    *,
    nnr: int,
    nelec: float,
    workdir: Optional[str | Path] = None,
):
    """Load cached ``v_corr`` if shape/nelec match; else ``(None, None)``.

    Set ``INVERT_FORCE=1`` to ignore an existing cache.
    """
    import json

    if str(os.environ.get("INVERT_FORCE", "")).strip().lower() in ("1", "true", "yes"):
        return None, None
    cdir = _cache_dir(workdir)
    npy = cdir / _V_CORR_CACHE_NAME
    meta_path = cdir / _V_CORR_META_NAME
    if not npy.is_file():
        return None, None
    try:
        v = np.asarray(np.load(npy), dtype=float).ravel()
        info = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    except Exception as exc:
        print(f"[invert] cache unreadable ({exc}); recomputing", flush=True)
        return None, None
    if int(v.size) != int(nnr):
        print(
            f"[invert] cache nnr={v.size} ≠ {nnr}; recomputing",
            flush=True,
        )
        return None, None
    cached_nelec = info.get("nelec")
    if cached_nelec is not None and abs(float(cached_nelec) - float(nelec)) > 0.05:
        print(
            f"[invert] cache nelec={cached_nelec} ≠ {nelec:.3f}; recomputing",
            flush=True,
        )
        return None, None
    meta = {
        "converged": bool(info.get("converged", False)),
        "stalled": bool(info.get("stalled", False)),
        "usable": bool(info.get("usable", True)),
        "inner": info.get("inner", "cache"),
        "n_iter": int(info.get("n_iter") or 0),
        "err": float(info["err"]) if info.get("err") is not None else float("nan"),
        "tol": float(info["tol"]) if info.get("tol") is not None else TOL,
        "nelec": float(nelec),
        "from_cache": True,
        "cache_dir": str(cdir),
    }
    print(
        f"[invert] loaded cache  E_H[Δρ]={meta['err']:.6e}  "
        f"converged={meta['converged']}  dir={cdir}",
        flush=True,
    )
    return np.asfortranarray(v), meta


def revive_host_driver(host_driver, *, comm=None, logfile=None):
    """Rebuild a live host ``Driver`` from its on-disk SCF save.

    Invert's per-residual ``Driver.stop()`` wipes process-global QE Fortran
    state, leaving the caller's host object with NULL ``vltot``. Restart SCF
    from the existing ``outdir``/``prefix`` (usually a few iterations) restores
    potentials and wavefunctions for vacuum align + PW Casida.
    """
    from qepy.driver import Driver

    opts = copy.deepcopy(getattr(host_driver, "qe_options", None) or {})
    if not opts:
        raise RuntimeError("revive_host_driver needs host_driver.qe_options")
    ctrl = opts.setdefault("&control", {})
    ctrl["restart_mode"] = "'restart'"
    ctrl.setdefault("calculation", "'scf'")
    prefix = _unquote_qe(ctrl.get("prefix", "pwscf"))
    outdir = Path(_unquote_qe(ctrl.get("outdir", "./out"))).resolve()
    workdir = outdir.parent if outdir.name == "out" else outdir
    save = outdir / f"{prefix}.save"
    if not save.is_dir():
        raise FileNotFoundError(f"host SCF save missing: {save}")

    try:
        host_driver.stop(what="no")
    except Exception:
        pass

    log = logfile or str(workdir / "host_revive.log")
    cwd = os.getcwd()
    print(f"[invert] revive host from {save} …", flush=True)
    try:
        os.chdir(workdir)
        d = Driver(
            qe_options=opts,
            comm=comm,
            iterative=False,
            logfile=log,
        )
        d.scf()
        d.qe_options = opts
    finally:
        os.chdir(cwd)
    print("[invert] host revived", flush=True)
    return d


def _col(rho) -> np.ndarray:
    """QE density/potential as a Fortran-ordered ``(nnr, 1)`` column."""
    a = np.asarray(rho, dtype=float)
    if a.ndim == 2:
        a = a[:, 0]
    a = np.asfortranarray(a.ravel())
    return a.reshape(a.size, 1, order="F")


def _anderson_step(v_cur, f_cur, hist, beta: float):
    """One Anderson-accelerated step of the fixed point ``v ← v + β F(v)``.

    ``F`` is the residual ``v_H[Δρ]`` (``f_cur`` here), which vanishes at the
    solution. ``hist`` holds the previous ``(v, F)`` pairs (oldest→newest,
    current excluded). With no history this is the plain damped step
    ``v + β F``; otherwise it solves the Walker–Ni least-squares problem

        γ = argmin ‖F_cur − ΔF γ‖   over the residual differences ΔF,

    and returns ``v_cur + β F_cur − (ΔX + β ΔF) γ``. A small Tikhonov ridge
    keeps the solve well-posed when the ΔF columns are nearly collinear.
    """
    if not hist:
        return np.asfortranarray(v_cur + beta * f_cur)
    Vs = [p[0] for p in hist] + [v_cur]
    Fs = [p[1] for p in hist] + [f_cur]
    dF = np.column_stack([(Fs[i + 1] - Fs[i]).ravel() for i in range(len(Fs) - 1)])
    dX = np.column_stack([(Vs[i + 1] - Vs[i]).ravel() for i in range(len(Vs) - 1)])
    fc = f_cur.ravel()
    A = dF.T @ dF
    A[np.diag_indices_from(A)] += ANDERSON_REG * (np.trace(A) / A.shape[0] + 1e-30)
    gamma = np.linalg.solve(A, dF.T @ fc)
    v_new = v_cur.ravel() + beta * fc - (dX + beta * dF) @ gamma
    return np.asfortranarray(v_new.reshape(v_cur.shape))


def _invert_qe_options(qe_options: dict, nelec: float, workdir: Path, prefix: str, host_driver):
    """Host ``qe_options`` → isolated invert run: own outdir, ``nbnd``, ``tot_charge``.

    ``tot_charge`` makes QE's ``nelec`` match the subsystem (``nelec = ΣZ_val −
    tot_charge``); ``nbnd`` drops the host's XAS virtual pool, which the
    inversion does not need.
    """
    from casidapy.embed.potential import read_upf_local_potential, resolve_upf_path

    opts = copy.deepcopy(qe_options)
    outdir = workdir / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    control = dict(opts.get("&control") or {})
    control["calculation"] = "'scf'"
    control["prefix"] = f"'{prefix}'"
    control["outdir"] = f"'{outdir}/'"
    opts["&control"] = control

    zval_sum = 0.0
    for sym in host_driver.get_ase_atoms().get_chemical_symbols():
        _, _, zv = read_upf_local_potential(resolve_upf_path(host_driver, sym, None))
        zval_sum += float(zv)

    system = dict(opts.get("&system") or {})
    system["tot_charge"] = float(zval_sum - nelec)
    system["nbnd"] = max(8, int(np.ceil(nelec / 2.0)) + NBND_EXTRA)
    opts["&system"] = system
    return opts


def invert_xc_potential_qepy(
    rho_ref,
    qe_options: dict,
    *,
    dV: float,
    host_driver,
    workdir: Optional[str | Path] = None,
    prefix: str = "invert",
    comm=None,
):
    """Invert ``rho_ref``: return ``(v_corr_ry, meta)`` for the QE XC slot.

    ZMP fixed point ``v ← v + α v_H[Δρ]`` with Anderson acceleration and α
    backtracking. Each ``ρ[v]`` uses a **fresh QE Driver** (torn down with
    ``stop(what='no')``) so repeated external-potential SCFs cannot rot global
    MPI/SCF state. Inner solve mode is selected by env ``INVERT_INNER``:

    * ``scf`` (default) — full ``scf(maxiter=…)`` under frozen ``v``
    * ``diag`` — one ``diagonalize()`` only (fast; ``ρ[v]`` may period-2)

    When α sits at ``ALPHA_MIN`` and ``STALL_REJECTS`` consecutive trials are
    rejected, or the best residual fails to improve by ``PLATEAU_MIN_IMPROVE``
    for ``PLATEAU_ITERS`` outer steps, the loop stops and returns the best ``v``.
    A matching ``v_corr`` checkpoint under ``workdir`` / ``INVERT_CACHE_DIR``
    skips the solve (``INVERT_FORCE=1`` to recompute).
    """
    from dftpy.functional import Hartree
    from qepy.driver import Driver

    inner = _inner_mode()
    rho_ref = _col(np.asarray(rho_ref, dtype=float).ravel())
    nelec = float(rho_ref.sum() * float(dV))
    workdir = Path(workdir or os.environ.get("TMPDIR") or ".") / "qepy_invert"
    workdir.mkdir(parents=True, exist_ok=True)

    cached_v, cached_meta = load_v_corr_checkpoint(
        nnr=int(rho_ref.size), nelec=nelec, workdir=workdir
    )
    if cached_v is not None:
        return cached_v, cached_meta

    opts = _invert_qe_options(qe_options, nelec, workdir, prefix, host_driver)
    logpath = str(workdir / "qe_inv.log")

    print(
        f"[invert] nelec≈{nelec:.3f}  nbnd={opts['&system']['nbnd']}  "
        f"inner={inner}  α0={ALPHA0}  anderson_m={ANDERSON_M}  "
        f"max_iter={MAX_ITER}  tol={TOL:.1e}  fresh_driver=each  "
        f"stall_rejects={STALL_REJECTS}  plateau_iters={PLATEAU_ITERS}",
        flush=True,
    )

    alpha = float(ALPHA0)
    history = []
    cwd = os.getcwd()
    n_floor_reject = 0
    n_no_improve = 0
    stalled = False

    def _stop(d):
        if d is None:
            return
        try:
            d.stop(what="no")
        except Exception:
            pass

    def _residual(v_col, tr2):
        """Fresh Driver → ρ[v] → ``(E_H[Δρ], ‖Δρ‖, v_H[Δρ], n_scf, conv)``."""
        d = None
        try:
            os.chdir(workdir)
            # diag needs iterative=True so diagonalize() is exposed; scf uses
            # non-iterative so one call runs QE's full mixing loop.
            d = Driver(
                qe_options=opts,
                comm=comm,
                iterative=(inner == "diag"),
                logfile=logpath,
            )
            flags = d.qepy_modules.control_flags
            d.set_external_potential(np.asfortranarray(v_col), exttype=("xc",))
            if inner == "diag":
                d.diagonalize()
                conv = False
                n_scf = 1
            else:
                flags.set_tr2(float(tr2))
                d.scf(maxiter=INNER_MAXITER)
                conv = bool(flags.get_conv_elec())
                n_scf = int(flags.get_n_scf_steps())
            drho = _col(d.get_density()) - rho_ref
            hout = Hartree()(d.data2field(drho), calcType=["E", "V"])
            direction = np.asfortranarray(2.0 * d.field2data(hout.potential))
            return (
                float(hout.energy),
                float(np.sqrt(np.mean(drho**2))),
                direction,
                n_scf,
                conv,
            )
        finally:
            _stop(d)
            os.chdir(cwd)

    def _log(it, err, rms, note):
        print(f"[invert] it={it:<3d} E_H[Δρ]={err:.6e}  ‖Δρ‖={rms:.3e}  {note}", flush=True)
        history.append({"iter": it, "err": err, "drho_rms": rms, "alpha": alpha, "note": note})

    # Baseline ρ[v=0].
    v_best = np.zeros_like(rho_ref, order="F")
    err_best, rms, dir_best, n_scf, conv = _residual(v_best, INNER_TR2_LOOSE)
    hist = deque(maxlen=ANDERSON_M)
    _log(
        0,
        err_best,
        rms,
        f"start  inner={inner}  n_scf={n_scf}{'' if (conv or inner == 'diag') else ' NOCONV'}",
    )

    for it in range(1, MAX_ITER + 1):
        if err_best < TOL:
            break
        tr2 = INNER_TR2 if err_best < 100.0 * TOL else INNER_TR2_LOOSE
        note = "anderson" if hist else "damped"
        v_try = _anderson_step(v_best, dir_best, hist, alpha)
        err, rms, dir_try, n_scf, conv = _residual(v_try, tr2)
        if err < err_best:
            improved = err < err_best * (1.0 - PLATEAU_MIN_IMPROVE)
            hist.append((v_best, dir_best))
            v_best, err_best, dir_best = v_try, err, dir_try
            alpha = min(1.2 * alpha, ALPHA0)
            n_floor_reject = 0
            n_no_improve = 0 if improved else (n_no_improve + 1)
        else:
            hist.clear()
            alpha = max(0.5 * alpha, ALPHA_MIN)
            note = "reject"
            n_no_improve += 1
            if alpha <= ALPHA_MIN * (1.0 + 1e-12):
                n_floor_reject += 1
            else:
                n_floor_reject = 0
        _log(
            it,
            err,
            rms,
            f"{note}  α={alpha:.2f}  n_scf={n_scf}  tr2={tr2:.0e}"
            f"{'' if conv else ' NOCONV'}",
        )
        if n_floor_reject >= STALL_REJECTS:
            stalled = True
            print(
                f"[invert] stall: {STALL_REJECTS} rejects at α={ALPHA_MIN}  "
                f"returning best E_H[Δρ]={err_best:.6e}",
                flush=True,
            )
            break
        if n_no_improve >= PLATEAU_ITERS and err_best >= TOL:
            stalled = True
            print(
                f"[invert] plateau: no ≥{PLATEAU_MIN_IMPROVE:.0%} improve "
                f"for {PLATEAU_ITERS} iters  returning best E_H[Δρ]={err_best:.6e}",
                flush=True,
            )
            break

    converged = err_best < TOL
    # Plateau near TOL is usable for embedding; treat as success.
    usable = converged or stalled
    print(
        f"[invert] {'converged' if converged else ('stalled' if stalled else 'NOT converged')}  "
        f"E_H[Δρ]={err_best:.6e}  α={alpha:.3g}  n_iter={len(history)}",
        flush=True,
    )
    meta = {
        "converged": bool(converged),
        "stalled": bool(stalled),
        "usable": bool(usable),
        "inner": inner,
        "from_cache": False,
        "n_iter": len(history),
        "err": err_best,
        "tol": TOL,
        "alpha": alpha,
        "anderson_m": ANDERSON_M,
        "nelec": nelec,
        "nbnd": int(opts["&system"]["nbnd"]),
        "tot_charge": float(opts["&system"]["tot_charge"]),
        "logfile": logpath,
        "history": history,
    }
    save_v_corr_checkpoint(v_best, meta, workdir=workdir)
    return np.asarray(v_best, dtype=float).ravel(), meta


def build_invert_embedding(
    driver,
    edge_atom,
    *,
    upf_path=None,
    grid=None,
    verbose: bool = False,
    gauge_align=DEFAULT_GAUGE_ALIGN,
    gauge_tol: float = 1e-6,
    vloc_source=DEFAULT_VLOC_SOURCE,
    use_gpu: bool = False,
    comm=None,
    active_atoms: Optional[Sequence[int]] = None,
    hirshfeld_reference: str = "neutral",
    oxidation_state=None,
    oxidation_states=None,
    formal_ion_basis: str = "aug-cc-pvdz",
    formal_ion_xc: str = "pbe",
    formal_ion_valence_only: bool = True,
    density_partition: str = "hirshfeld",
    invert_workdir=None,
):
    """Invert the absorber peel ``ρ_A`` → ``V_env = V_ionic + v_H[ρ_env] + v_corr``."""
    from dftpy.field import DirectField

    from casidapy.adapter.qepy import build_uspp_map_from_driver
    from casidapy.embed.hirshfeld import (
        _normalize_density_partition,
        _normalize_hirshfeld_reference,
        _partition_active_env,
        hartree_potential_ha,
        sample_field_at_point,
        shift_v_env_gauge_hirshfeld,
        shift_v_env_gauge_min_density,
    )
    from casidapy.embed.potential import _edge_atom_info
    from casidapy.utils.accel import array_module, asnumpy

    grid = grid if grid is not None else ensure_charge_grid(driver)
    edge_atom, sym, z_atomic, upf_a, z_val, center_bohr = _edge_atom_info(
        driver, edge_atom, upf_path=upf_path
    )
    active = [int(i) for i in (active_atoms if active_atoms is not None else [edge_atom])]
    part_mode = _normalize_density_partition(density_partition)
    ref_mode = _normalize_hirshfeld_reference(hirshfeld_reference)

    v_loc_a, v_ionic = ionic_vloc_residual(
        driver,
        edge_atom,
        grid,
        upf_path=upf_a,
        vloc_source=vloc_source,
        atom_indices=active,
    )

    upf_map = dict(build_uspp_map_from_driver(driver) or {})
    upf_map[str(sym).strip().capitalize()] = upf_a
    rho_tot, rho_ta, _rho_tsum, _w_a, rho_act, rho_env, part_extra = _partition_active_env(
        driver,
        edge_atom,
        grid,
        density_partition=part_mode,
        upf_map=upf_map,
        use_gpu=use_gpu,
        comm=comm,
        active_atoms=active,
        hirshfeld_reference=ref_mode,
        oxidation_state=oxidation_state,
        oxidation_states=oxidation_states,
        formal_ion_basis=formal_ion_basis,
        formal_ion_xc=formal_ion_xc,
        formal_ion_valence_only=formal_ion_valence_only,
    )

    qe_opts = getattr(driver, "qe_options", None)
    if not qe_opts:
        raise RuntimeError("build_invert_embedding needs driver.qe_options")

    # QEpy is process-global: the invert Driver's stop() tears down Fortran
    # state the host still needs, so snapshot host quantities first.
    v_h_env = hartree_potential_ha(driver, rho_env)
    gmode = str(gauge_align).lower().strip()
    if gmode in ("far_field", "farfield", "vacuum", "default", ""):
        gmode = "hirshfeld"
    fermi_pre_ha = None
    if gmode in ("fermi", "ef", "efermi", "e_fermi"):
        from casidapy.adapter.qepy import fermi_energy_ha

        fermi_pre_ha = float(fermi_energy_ha(driver))

    v_corr_ry, inv_meta = invert_xc_potential_qepy(
        rho_act,
        qe_opts,
        dV=float(grid.dV),
        host_driver=driver,
        workdir=invert_workdir,
        prefix=f"inv_{sym}",
        comm=comm,
    )
    v_corr_ha = ry_to_ha(v_corr_ry)

    nr = tuple(int(x) for x in grid.nr)
    xp = array_module(use_gpu)
    v_env = DirectField(
        grid=grid,
        rank=1,
        griddata_3d=asnumpy(
            xp.asarray(v_ionic, dtype=float).reshape(nr, order="F")
            + xp.asarray(v_h_env, dtype=float).reshape(nr, order="F")
            + xp.asarray(v_corr_ha, dtype=float).reshape(nr, order="F")
        ),
    )

    fermi_ref_ha = None
    if gmode in ("hirshfeld", "stockholder", "weight", "auto"):
        v_env, gauge_shift = shift_v_env_gauge_hirshfeld(
            v_env, grid, rho_ta, rho_tot, tol=gauge_tol
        )
        gmode = "hirshfeld"
    elif gmode in ("true_vacuum", "min_density", "vacuum_global"):
        v_env, gauge_shift = shift_v_env_gauge_min_density(v_env, grid, rho_tot)
        gmode = "true_vacuum"
    elif gmode in ("fermi", "ef", "efermi", "e_fermi"):
        v_env, gauge_shift = shift_v_env_gauge_at(
            v_env, grid, center_bohr, mode="fermi", fermi_ha=fermi_pre_ha
        )
        gmode = "fermi"
        fermi_ref_ha = fermi_pre_ha
    else:
        v_env, gauge_shift = shift_v_env_gauge_at(v_env, grid, center_bohr, mode=gmode)

    dV = float(grid.dV)
    meta = {
        "edge_atom": edge_atom,
        "active_atoms": list(active),
        "symbol": sym,
        "z_atomic": z_atomic,
        "upf_path": str(upf_a),
        "z_valence": float(z_val),
        "vloc_source": str(vloc_source).lower().strip(),
        "embed_mode": "invert",
        "invert_assemble": "V_ionic + v_H[rho_env] + v_corr(OEP)",
        "density_partition": part_mode,
        "hirshfeld_reference": ref_mode,
        "kedf": None,
        "kedf_enabled": False,
        "gauge_align": gmode,
        "gauge_shift_ha": float(gauge_shift),
        "fermi_energy_ha": fermi_ref_ha,
        "rho_tot_charge": float(rho_tot.sum() * dV),
        "rho_active_charge": float(rho_act.sum() * dV),
        "rho_env_charge": float(rho_env.sum() * dV),
        "v_h_env_mean_ha": float(np.mean(v_h_env)),
        "v_corr_mean_ha": float(np.mean(v_corr_ha)),
        "v_corr_absmax_ha": float(np.max(np.abs(v_corr_ha))),
        "v_ionic_mean_ha": float(np.mean(np.asarray(v_ionic))),
        "v_loc_a_mean_ha": float(np.mean(np.asarray(v_loc_a))),
        "v_env_mean_ha": float(np.mean(np.asarray(v_env))),
        "v_env_at_nuc_ha": float(sample_field_at_point(v_env, grid, center_bohr)),
        "invert": inv_meta,
        **{f"part_{k}": v for k, v in (part_extra or {}).items()},
    }
    if verbose:
        logger.info(
            "[embed-invert] %s atom=%d converged=%s err=%.3e",
            sym,
            edge_atom,
            inv_meta.get("converged"),
            inv_meta.get("err", float("nan")),
        )
    return v_env, grid, meta
