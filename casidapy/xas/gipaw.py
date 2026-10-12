"""GIPAW restoration of pseudo-virtuals inside ``r_c`` (Taillefumier / XSpectra).

This is **not** QE ``is_paw=.true.`` DFT and not CasidaPy USPP ``Q_ij``
(that augments ``ρ_ia``).  The host SCF must have used the same GIPAW UPF
that supplies the AE/PS partial waves; ONCV cannot be reconstructed.

    ψ_n^{AE}(r) = ψ̃_n(r) + Σ_i [φ_i^{AE}(r) − φ_i^{PS}(r)] c_{in}
    c = S^{+} ⟨φ^{PS}|ψ̃⟩_{r < r_c}
    S_{ij} = ⟨φ_i^{PS}|φ_j^{PS}⟩_{r < r_c}

``φ`` are the GIPAW valence orbitals (2S/2P/… including ``m``), not the
Kleinman–Bylander ``PP_BETA`` set (often a single ``s`` channel).

UPF stores χ(r) = r R(r).  The 3-D orbital is ``R(r) Y_lm``.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from scipy.interpolate import CubicSpline

from casidapy.utils.uspp import _ylm_on_grid

_Y00 = 1.0 / np.sqrt(4.0 * np.pi)
_GIPAW_RCOND = 1e-8


class GipawUpfError(ValueError):
    """UPF has no usable GIPAW reconstruction data."""


@dataclass
class GipawCoreOrbital:
    """Atomic core radial function from ``PP_GIPAW_CORE_ORBITAL``."""

    label: str
    n: int
    l: int
    chi: np.ndarray


@dataclass
class GipawValenceOrbital:
    """One GIPAW reconstruction channel (radial AE + PS, all ``m``)."""

    label: str
    l: int
    cutoff_radius: float
    chi_ae: np.ndarray
    chi_ps: np.ndarray


@dataclass
class GipawChannel:
    """One ``(n, l, m)`` reconstruction channel."""

    label: str
    iorb: int
    l: int
    m: int
    cutoff_radius: float


@dataclass
class GipawData:
    """Parsed GIPAW reconstruction table for one absorber species."""

    path: str
    element: str
    r: np.ndarray
    rab: np.ndarray
    core_orbitals: list
    valence_orbitals: list
    z_valence: float = 0.0
    format: str = "xml"
    _R_ae: list = field(default_factory=list, repr=False)
    _R_ps: list = field(default_factory=list, repr=False)
    _R_core: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        self.r = np.asarray(self.r, dtype=float)
        self.rab = np.asarray(self.rab, dtype=float)
        self._R_ae = [_radial_R_spline(self.r, o.chi_ae) for o in self.valence_orbitals]
        self._R_ps = [_radial_R_spline(self.r, o.chi_ps) for o in self.valence_orbitals]
        self._R_core = [_radial_R_spline(self.r, o.chi) for o in self.core_orbitals]

    @property
    def r_c(self) -> float:
        cuts = [float(o.cutoff_radius) for o in self.valence_orbitals if o.cutoff_radius > 0.0]
        return float(max(cuts)) if cuts else float(self.r[-1])

    def channels(self) -> list:
        out = []
        for i, orb in enumerate(self.valence_orbitals):
            for m in range(-int(orb.l), int(orb.l) + 1):
                out.append(
                    GipawChannel(
                        label=orb.label,
                        iorb=i,
                        l=int(orb.l),
                        m=int(m),
                        cutoff_radius=float(orb.cutoff_radius),
                    )
                )
        return out

    def core_orbital(self, label: str = "1S") -> GipawCoreOrbital:
        want = str(label).upper().replace(" ", "")
        for o in self.core_orbitals:
            if str(o.label).upper().replace(" ", "") == want:
                return o
        if self.core_orbitals and want in ("1S", "1S0"):
            for o in self.core_orbitals:
                if int(o.n) == 1 and int(o.l) == 0:
                    return o
        raise GipawUpfError(
            f"{Path(self.path).name}: no GIPAW core orbital {label!r} "
            f"(have {[o.label for o in self.core_orbitals]})"
        )


def _radial_R_spline(r, chi):
    r = np.asarray(r, dtype=float)
    chi = np.asarray(chi, dtype=float)
    R = np.empty_like(r)
    R[0] = chi[0] / r[0]
    R[1:] = chi[1:] / r[1:]
    return CubicSpline(r, R, extrapolate=False)


def _eval_R(spline, r_mesh, dist, r_cut=None):
    dist = np.asarray(dist, dtype=float)
    out = np.zeros_like(dist, dtype=float)
    r0 = float(r_mesh[0])
    r1 = float(r_mesh[-1])
    inside = (dist >= 0.0) & (dist <= r1)
    if r_cut is not None:
        inside = inside & (dist <= float(r_cut) + 1e-12)
    if not np.any(inside):
        return out
    q = np.clip(dist[inside], r0, r1)
    out[inside] = spline(q)
    return out


def _fromstring(text) -> np.ndarray:
    if text is None:
        return np.zeros(0)
    return np.fromstring(str(text).replace("D", "E").replace("d", "e"), sep=" ")


def _regex_block(text: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.S | re.I)
    return None if m is None else m.group(1)


def _load_radial_mesh(root, text: str):
    r = rab = None
    if root is not None:
        r_el = root.find(".//PP_MESH/PP_R")
        rab_el = root.find(".//PP_MESH/PP_RAB")
        if r_el is not None and r_el.text:
            r = _fromstring(r_el.text)
        if rab_el is not None and rab_el.text:
            rab = _fromstring(rab_el.text)
    if r is None:
        body = _regex_block(text, "PP_R")
        if not body:
            raise GipawUpfError("UPF has no PP_R mesh")
        r = _fromstring(body)
    if rab is None:
        body = _regex_block(text, "PP_RAB")
        rab = _fromstring(body) if body else np.gradient(r)
    n = int(min(r.size, rab.size))
    if n < 8:
        raise GipawUpfError(f"UPF radial mesh too short ({n})")
    return r[:n], rab[:n]


def _header_attr(root, text: str, name: str, default: str = "") -> str:
    if root is not None:
        hdr = root.find(".//PP_HEADER")
        if hdr is not None and hdr.get(name) is not None:
            return str(hdr.get(name))
    m = re.search(rf'\b{name}="([^"]*)"', text, re.I)
    if m:
        return m.group(1)
    m = re.search(rf"\b{name}='([^']*)'", text, re.I)
    if m:
        return m.group(1)
    # Old UPF (non-XML) PP_HEADER: "   O                   Element"
    aliases = {
        "element": r"^\s+(\S+)\s+Element\s*$",
        "z_valence": r"^\s+([+-]?\d+(?:\.\d*)?)\s+Z valence\s*$",
    }
    pat = aliases.get(str(name).lower())
    if pat:
        m = re.search(pat, text, re.M | re.I)
        if m:
            return m.group(1)
    return default


def _trim_chi(vals: np.ndarray, nmesh: int) -> np.ndarray:
    vals = np.asarray(vals, dtype=float).ravel()
    if vals.size == nmesh:
        return vals
    if vals.size > nmesh:
        return vals[-nmesh:]
    out = np.zeros(nmesh, dtype=float)
    out[: vals.size] = vals
    return out


def _infer_cutoff(r, chi_ae, chi_ps, stated: float) -> float:
    if float(stated) > 1e-8:
        return float(stated)
    d = np.abs(np.asarray(chi_ae) - np.asarray(chi_ps))
    scale = float(np.max(np.abs(chi_ae))) + 1e-30
    hit = np.where(d > 1e-6 * scale)[0]
    if hit.size == 0:
        return float(r[min(r.size - 1, 8)])
    return float(r[min(int(hit[-1]) + 1, r.size - 1)])


def _parse_nl_label(label: str, l_fallback: int = 0):
    lab = str(label).strip()
    m = re.match(r"\s*(\d+)\s*([SPDFGspdfg])", lab)
    if m:
        n = int(m.group(1))
        l = "SPDFG".index(m.group(2).upper())
        return lab.upper().replace(" ", ""), n, l
    return lab.upper().replace(" ", "") or "?", 0, int(l_fallback)


def parse_gipaw_upf(path) -> GipawData:
    """Load AE/PS partial waves, core 1s, and ``r_c`` from a GIPAW UPF."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(errors="ignore")
    root = None
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        root = None

    has = (
        'has_gipaw="true"' in text
        or "has_gipaw='true'" in text
        or "<PP_GIPAW" in text
        or "PP_GIPAW_RECONSTRUCTION_DATA" in text
        or "<PP_GIPAW_WFS_AE" in text
        or "<PP_GIPAW_AE_ORBITAL" in text
    )
    if not has:
        raise GipawUpfError(
            f"{path.name} has no GIPAW reconstruction data. ONCV / NC "
            "pseudos without PP_GIPAW cannot restore nodes inside r_c; "
            "the absorber SCF must use a GIPAW UPF (has_gipaw=true)."
        )

    r, rab = _load_radial_mesh(root, text)
    nmesh = int(r.size)
    element = _header_attr(root, text, "element", "?").strip()
    try:
        z_val = float(_header_attr(root, text, "z_valence", "0") or 0.0)
    except ValueError:
        z_val = 0.0

    cores, valence, fmt = [], [], "legacy"
    if root is not None:
        cores, valence = _parse_gipaw_xml(root, r, nmesh)
        if valence:
            fmt = "xml"

    if not valence:
        cores, valence = _parse_gipaw_legacy(text, r, nmesh)
        fmt = "legacy"

    if not valence:
        raise GipawUpfError(
            f"{path.name}: GIPAW flag set but no AE/PS valence orbitals parsed"
        )
    if not cores:
        raise GipawUpfError(f"{path.name}: no PP_GIPAW_CORE_ORBITAL (need atomic 1s)")

    data = GipawData(
        path=str(path),
        element=element,
        r=r,
        rab=rab,
        core_orbitals=cores,
        valence_orbitals=valence,
        z_valence=z_val,
        format=fmt,
    )
    assert_gipaw_ae_ps_ok(data)
    return data


def _parse_gipaw_xml(root, r, nmesh: int):
    cores, valence = [], []
    core_parent = root.find(".//PP_GIPAW_CORE_ORBITALS")
    if core_parent is not None:
        for child in list(core_parent):
            if not str(child.tag).upper().startswith("PP_GIPAW_CORE_ORBITAL"):
                continue
            if child.text is None or not child.text.strip():
                continue
            lab = child.get("label") or "1S"
            n_qn = int(float(child.get("n", "1") or 1))
            l_qn = int(float(child.get("l", "0") or 0))
            lab, n_qn, l_qn = _parse_nl_label(lab, l_qn)
            cores.append(
                GipawCoreOrbital(
                    label=lab, n=n_qn, l=l_qn, chi=_trim_chi(_fromstring(child.text), nmesh)
                )
            )

    orb_parent = root.find(".//PP_GIPAW_ORBITALS")
    if orb_parent is None:
        return cores, valence
    for child in list(orb_parent):
        tag = str(child.tag).upper()
        if not tag.startswith("PP_GIPAW_ORBITAL"):
            continue
        ae_el = child.find("PP_GIPAW_WFS_AE")
        ps_el = child.find("PP_GIPAW_WFS_PS")
        if ae_el is None or ps_el is None:
            continue
        lab = child.get("label") or "?"
        l_qn = int(float(child.get("l", "0") or 0))
        lab, _, l_qn = _parse_nl_label(lab, l_qn)
        chi_ae = _trim_chi(_fromstring(ae_el.text), nmesh)
        chi_ps = _trim_chi(_fromstring(ps_el.text), nmesh)
        rc = _infer_cutoff(
            r,
            chi_ae,
            chi_ps,
            float(child.get("cutoff_radius", "0") or 0.0),
        )
        valence.append(
            GipawValenceOrbital(
                label=lab, l=l_qn, cutoff_radius=rc, chi_ae=chi_ae, chi_ps=chi_ps
            )
        )
    return cores, valence


def _split_header_data(body: str):
    lines = [ln for ln in str(body).splitlines() if ln.strip()]
    if not lines:
        return "", ""
    return lines[0], "\n".join(lines[1:])


def _parse_gipaw_legacy(text: str, r, nmesh: int):
    cores, valence = [], []
    for block in re.finditer(
        r"<PP_GIPAW_CORE_ORBITAL>\s*(.*?)\s*</PP_GIPAW_CORE_ORBITAL>",
        text,
        re.S | re.I,
    ):
        head, data = _split_header_data(block.group(1))
        toks = head.split()
        n_qn = int(float(toks[0])) if toks else 1
        l_qn = int(float(toks[1])) if len(toks) > 1 else 0
        lab = f"{n_qn}{'SPDFG'[l_qn] if l_qn < 5 else '?'}"
        cores.append(
            GipawCoreOrbital(
                label=lab, n=n_qn, l=l_qn, chi=_trim_chi(_fromstring(data), nmesh)
            )
        )

    ae_blocks = re.findall(
        r"<PP_GIPAW_AE_ORBITAL>\s*(.*?)\s*</PP_GIPAW_AE_ORBITAL>",
        text,
        re.S | re.I,
    )
    ps_blocks = re.findall(
        r"<PP_GIPAW_PS_ORBITAL>\s*(.*?)\s*</PP_GIPAW_PS_ORBITAL>",
        text,
        re.S | re.I,
    )
    n_pair = min(len(ae_blocks), len(ps_blocks))
    for i in range(n_pair):
        ae_head, ae_data = _split_header_data(ae_blocks[i])
        ps_head, ps_data = _split_header_data(ps_blocks[i])
        ae_toks = ae_head.split()
        lab = ae_toks[0] if ae_toks else f"ORB{i+1}"
        l_qn = 0
        if len(ae_toks) > 1:
            try:
                l_qn = int(float(ae_toks[1]))
            except ValueError:
                l_qn = 0
        lab, _, l_qn = _parse_nl_label(lab, l_qn)
        ps_toks = ps_head.split()
        stated = 0.0
        try:
            if ps_toks:
                stated = float(ps_toks[0])
        except ValueError:
            stated = 0.0
        chi_ae = _trim_chi(_fromstring(ae_data), nmesh)
        chi_ps = _trim_chi(_fromstring(ps_data), nmesh)
        rc = _infer_cutoff(r, chi_ae, chi_ps, stated)
        valence.append(
            GipawValenceOrbital(
                label=lab, l=l_qn, cutoff_radius=rc, chi_ae=chi_ae, chi_ps=chi_ps
            )
        )
    return cores, valence


def core_chi_norm(data: GipawData, label: str = "1S") -> float:
    """``∫ χ² dr`` for a core orbital (1 if UPF stores normalized ``r R``)."""
    orb = data.core_orbital(label)
    return float(np.sum(orb.chi * orb.chi * data.rab))


def ae_ps_match_at_rc(data: GipawData, *, atol: float = 1e-4) -> list:
    """``|χ_AE − χ_PS|`` at each channel ``r_c`` (should vanish by construction)."""
    rows = []
    for orb in data.valence_orbitals:
        irc = int(np.argmin(np.abs(data.r - orb.cutoff_radius)))
        d = abs(float(orb.chi_ae[irc] - orb.chi_ps[irc]))
        scale = max(abs(float(orb.chi_ae[irc])), 1e-30)
        rows.append(
            {
                "label": orb.label,
                "l": int(orb.l),
                "r_c": float(orb.cutoff_radius),
                "abs_diff": d,
                "rel_diff": d / scale,
                "ok": bool(d < float(atol) or d / scale < 1e-3),
            }
        )
    return rows


def assert_gipaw_ae_ps_ok(
    data: GipawData,
    *,
    hard_rel: float = 0.05,
    p_hard_rel: float = 0.01,
    warn_rel: float = 1e-3,
) -> list:
    """Reject UPFs whose AE/PS reconstruction channels disagree at ``r_c``.

    The broken Mar/Jul-era ``B.pbe-tm-gipaw`` / ``N.pbe-tm-gipaw`` **3P** pairs
    sit at ``rel_diff ~ 0.02–1``, which poisons dual-projector ``c`` and
    K-edge oscillator strength.  Mild s-channel mismatches (``~1e-2`` on some
    ``tm-new-dc`` files) only warn.
    """
    rows = ae_ps_match_at_rc(data, atol=5e-4)

    def _hard(r) -> bool:
        rel = float(r["rel_diff"])
        if rel >= float(hard_rel):
            return True
        # p-channels are the known poison (old B/N 3P).
        return int(r["l"]) == 1 and rel >= float(p_hard_rel)

    hard = [r for r in rows if _hard(r)]
    if hard:
        detail = ", ".join(
            f"{r['label']}(rel={r['rel_diff']:.2e})" for r in hard
        )
        raise GipawUpfError(
            f"{Path(data.path).name}: AE/PS mismatch at r_c for {detail}; "
            "use a tm-new-gipaw-dc (or equivalent) reconstruction UPF"
        )
    for r in rows:
        if r["ok"]:
            continue
        if float(r["rel_diff"]) >= float(warn_rel):
            print(
                f"  [gipaw] warn {Path(data.path).name} {r['label']}: "
                f"|AE−PS|(rc)/|AE|={r['rel_diff']:.2e}",
                flush=True,
            )
    return rows


def eval_core_at_coords(data: GipawData, coords, center, *, label: str = "1S"):
    """Atomic GIPAW core orbital on Cartesian points (Bohr)."""
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    R0 = np.asarray(center, dtype=float).reshape(3)
    d = coords - R0
    dist = np.sqrt(np.sum(d * d, axis=1))
    orb = data.core_orbital(label)
    idx = data.core_orbitals.index(orb)
    rad = _eval_R(data._R_core[idx], data.r, dist, r_cut=None)
    if int(orb.l) == 0:
        return rad * _Y00
    y = _ylm_on_grid(int(orb.l), 0, d[:, 0], d[:, 1], d[:, 2], dist)
    return rad * y


def partial_waves_at_coords(data: GipawData, coords, center):
    """AE, PS, and Δφ on Cartesian points.  Arrays ``(n_chan, n_pts)``."""
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    R0 = np.asarray(center, dtype=float).reshape(3)
    d = coords - R0
    dist = np.sqrt(np.sum(d * d, axis=1))
    channels = data.channels()
    n_pts = int(coords.shape[0])
    n_chan = int(len(channels))
    ae = np.zeros((n_chan, n_pts), dtype=float)
    ps = np.zeros((n_chan, n_pts), dtype=float)
    for k, ch in enumerate(channels):
        y = _ylm_on_grid(ch.l, ch.m, d[:, 0], d[:, 1], d[:, 2], dist)
        rae = _eval_R(data._R_ae[ch.iorb], data.r, dist, r_cut=ch.cutoff_radius)
        rps = _eval_R(data._R_ps[ch.iorb], data.r, dist, r_cut=ch.cutoff_radius)
        ae[k] = rae * y
        ps[k] = rps * y
    return {"ae": ae, "ps": ps, "delta": ae - ps, "channels": channels, "dist": dist}


def dual_projector_coeffs(ps_waves, psi, weights, *, rcond: float = _GIPAW_RCOND):
    """``c = S^{+} ⟨φ^{PS}|ψ̃⟩`` with ``S = ⟨φ^{PS}|φ^{PS}⟩`` on the sphere.

    ``ps_waves``: ``(n_chan, n_pts)``.  ``psi``: ``(n_pts,)`` or ``(n_pts, n_virt)``.
    """
    ps = np.asarray(ps_waves, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    psi = np.asarray(psi, dtype=float)
    one = psi.ndim == 1
    if one:
        psi = psi.reshape(-1, 1)
    psw = ps * w[None, :]
    S = psw @ ps.T
    S = 0.5 * (S + S.T)
    ov = psw @ psi
    c = np.linalg.pinv(S, rcond=float(rcond)) @ ov
    return c[:, 0] if one else c


def reconstruct_at_coords(psi_tilde, delta_waves, coeffs):
    """``ψ_AE = ψ̃ + Δφᵀ c`` on the same points as ``psi_tilde``."""
    psi = np.asarray(psi_tilde, dtype=float)
    delta = np.asarray(delta_waves, dtype=float)
    c = np.asarray(coeffs, dtype=float)
    if psi.ndim == 1:
        return psi + delta.T @ c.reshape(-1)
    return psi + delta.T @ c


def _mic_cart(coords, center, lattice):
    """Minimum-image Cartesian displacements ``coords - center`` (Bohr)."""
    lat = np.asarray(lattice, dtype=float).reshape(3, 3)
    d = np.asarray(coords, dtype=float).reshape(-1, 3) - np.asarray(
        center, dtype=float
    ).reshape(3)
    frac = d @ np.linalg.inv(lat)
    frac -= np.round(frac)
    return frac @ lat


def _wrap_field(grid, data, like):
    arr = np.asarray(data, dtype=float).reshape(tuple(int(n) for n in grid.nr))
    if hasattr(like, "grid"):
        from dftpy.field import DirectField

        return DirectField(grid=grid, rank=1, griddata_3d=arr)
    return arr


def reconstruct_pw_virtuals(
    virtual_orbitals,
    host_grid,
    gipaw: Union[GipawData, str, Path],
    origin,
    *,
    rcond: float = _GIPAW_RCOND,
    psi_mode: str = "spline3",
    mapping: str = "log",
    n_rad: Optional[int] = None,
    n_ang: Optional[int] = None,
):
    """Restore ``ψ̃_a`` inside ``r_c`` on the host FFT mesh.

    Dual-projector coefficients use the same dense PAW-sphere quadrature as
    :func:`core_virtual_dipole_matrix_paw_gipaw` (so ``μ`` and ``K`` share
    ``c``).  ``Δφ`` is then added only on FFT voxels inside ``r_c`` (MIC).
    Orbitals are **not** re-normalized over the cell — Taillefumier form.

    Returns ``(reconstructed, meta)``.  ``meta['tilde']`` is the input list
    (unmodified).  Callers that rebuild ``μ`` with ``paw_gipaw`` must sample
    the tilde orbitals, not the reconstructed ones, or ``Δφ`` is applied twice.
    """
    from casidapy.xas.dipole import (
        LOCAL_N_ANG,
        LOCAL_N_RAD,
        absorber_radial_grid,
        sample_host_orbitals_at_coords,
    )
    from casidapy.xas.xas_utils import grid_coords_bohr

    if not isinstance(gipaw, GipawData):
        gipaw = parse_gipaw_upf(gipaw)

    virts = list(virtual_orbitals)
    n_virt = len(virts)
    R0 = np.asarray(origin, dtype=float).reshape(3)
    r_c = float(gipaw.r_c)
    if n_virt == 0:
        return [], {
            "n_virt": 0,
            "n_inside": 0,
            "r_c": r_c,
            "tilde": [],
            "mean_abs_c": 0.0,
        }

    coords_loc, wgt = absorber_radial_grid(
        R0,
        r_max=r_c,
        n_rad=int(LOCAL_N_RAD if n_rad is None else n_rad),
        n_ang=int(LOCAL_N_ANG if n_ang is None else n_ang),
        mapping=mapping,
    )
    waves_loc = partial_waves_at_coords(gipaw, coords_loc, R0)
    psi_t, _ = sample_host_orbitals_at_coords(
        host_grid, virts, coords_loc, psi_mode=psi_mode
    )
    coeff = dual_projector_coeffs(
        waves_loc["ps"], psi_t, wgt, rcond=rcond
    )
    if coeff.ndim == 1:
        coeff = coeff.reshape(-1, 1)

    coords = grid_coords_bohr(host_grid)
    d_mic = _mic_cart(coords, R0, host_grid.lattice)
    dist = np.sqrt(np.sum(d_mic * d_mic, axis=1))
    inside = dist < r_c
    n_inside = int(np.count_nonzero(inside))
    out = []
    if n_inside == 0:
        for psi in virts:
            arr = np.asarray(psi, dtype=float).ravel().copy()
            out.append(_wrap_field(host_grid, arr, psi))
    else:
        waves_g = partial_waves_at_coords(gipaw, R0 + d_mic[inside], R0)
        delta = waves_g["delta"]
        for a, psi in enumerate(virts):
            arr = np.asarray(psi, dtype=float).ravel().copy()
            arr[inside] = reconstruct_at_coords(arr[inside], delta, coeff[:, a])
            out.append(_wrap_field(host_grid, arr, psi))

    abs_c = np.abs(coeff)
    meta = {
        "n_virt": int(n_virt),
        "n_inside": n_inside,
        "r_c": r_c,
        "tilde": virts,
        "mean_abs_c": float(np.mean(abs_c)) if abs_c.size else 0.0,
        "max_abs_c": float(np.max(abs_c)) if abs_c.size else 0.0,
        "n_chan": int(coeff.shape[0]),
    }
    return out, meta


def apply_gipaw_kernel_virtuals(kernel, gipaw, origin, **kwargs):
    """Replace ``kernel._psi_unocc`` with reconstructed virtuals; keep tilde.

    Must run after ``set_active_orbitals`` / ``inject_core_orbitals`` and
    before ``setup()``.  ``paw_gipaw`` dipoles read ``_psi_unocc_tilde``.
    """
    tilde = list(kernel._psi_unocc)
    recon, meta = reconstruct_pw_virtuals(
        tilde, kernel.grid, gipaw, origin, **kwargs
    )
    kernel.set_active_orbitals(
        kernel._occ_e, kernel._unocc_e, kernel._psi_occ, recon
    )
    kernel._psi_unocc_tilde = tilde
    kernel._gipaw_k_meta = meta
    return meta


def _eval_ae_core_on_coords(core, coords) -> np.ndarray:
    from casidapy.xas.dipole import _eval_core_ao_on_coords

    return _eval_core_ao_on_coords(core, coords)


def core_virtual_dipole_matrix_paw_gipaw(
    core,
    virtual_orbitals,
    host_grid,
    gipaw: Union[GipawData, str, Path],
    *,
    origin=None,
    r_max: Optional[float] = None,
    n_rad: Optional[int] = None,
    n_ang: Optional[int] = None,
    mapping: str = "log",
    psi_mode: str = "spline3",
    core_wfc: str = "ae",
    gs_pass: Optional[int] = None,
    rcond: float = _GIPAW_RCOND,
    gauge: str = "length",
):
    """Length-gauge ``⟨c|r−R₀|ψ_recon⟩`` on the PAW sphere (shape ``(3, n_c, n_v)``).

    Host virtuals are sampled as ``ψ̃``, then restored with GIPAW Δφ inside
    ``r_c``.  The first (pseudo) term is the usual tiny ``⟨1s|r|ψ̃⟩``; the
    second term is the node restoration that carries K-edge strength.

    ``core_wfc``
        ``'ae'`` — fragment GTO 1s (``CoreOrbitals``; production embedding).
        ``'gipaw'`` — atomic ``PP_GIPAW_CORE_ORBITAL`` 1s (isolated-atom check).
    """
    from casidapy.xas.dipole import (
        LOCAL_N_ANG,
        LOCAL_N_RAD,
        absorber_origin_bohr,
        absorber_radial_grid,
        core_charge_centroid_bohr,
        core_dipole_radius_bohr,
        sample_host_orbitals_at_coords,
    )

    if not isinstance(gipaw, GipawData):
        gipaw = parse_gipaw_upf(gipaw)

    gmode = str(gauge).lower().strip()
    if gmode not in ("length", "r", "position"):
        raise ValueError(
            f"paw_gipaw supports length gauge only (got {gauge!r}); "
            "velocity would need reconstructed ∇ψ"
        )

    virts = list(virtual_orbitals)
    n_virt = len(virts)
    src = str(core_wfc).lower().strip()
    if src in ("gipaw", "upf", "atomic"):
        n_core = 1
        R0 = np.asarray(
            origin if origin is not None else absorber_origin_bohr(core, fallback=(0, 0, 0)),
            dtype=float,
        ).reshape(3)
        gs = 0 if gs_pass is None else int(gs_pass)
    else:
        C = np.asarray(core.mo_coeff, dtype=float)
        if C.ndim == 1:
            C = C.reshape(-1, 1)
        n_core = int(C.shape[1])
        R0 = np.asarray(
            origin
            if origin is not None
            else core_charge_centroid_bohr(core, host_grid),
            dtype=float,
        ).reshape(3)
        gs = 2 if gs_pass is None else int(gs_pass)

    if n_virt == 0:
        return np.zeros((3, n_core, 0), dtype=float)

    r_use = float(gipaw.r_c)
    if src not in ("gipaw", "upf", "atomic"):
        r_1s = core_dipole_radius_bohr(core, host_grid, R0)
        r_use = max(r_use, float(r_1s))
    if r_max is not None:
        r_use = max(r_use, float(r_max)) if float(r_max) > 0.0 else r_use

    coords, wgt = absorber_radial_grid(
        R0,
        r_max=r_use,
        n_rad=int(LOCAL_N_RAD if n_rad is None else n_rad),
        n_ang=int(LOCAL_N_ANG if n_ang is None else n_ang),
        mapping=mapping,
    )
    waves = partial_waves_at_coords(gipaw, coords, R0)
    psi_t, _ = sample_host_orbitals_at_coords(
        host_grid, virts, coords, psi_mode=psi_mode
    )
    coeff = dual_projector_coeffs(waves["ps"], psi_t, wgt, rcond=rcond)
    psi_v = reconstruct_at_coords(psi_t, waves["delta"], coeff)

    if src in ("gipaw", "upf", "atomic"):
        psi_c = eval_core_at_coords(gipaw, coords, R0).reshape(-1, 1)
    else:
        psi_c = _eval_ae_core_on_coords(core, coords)

    if gs > 0:
        vw = psi_v * wgt[:, None]
        G = psi_v.T @ vw
        G = 0.5 * (G + G.T)
        for _ in range(gs):
            mix = np.linalg.lstsq(G, vw.T @ psi_c, rcond=1e-10)[0]
            psi_c = psi_c - psi_v @ mix
        norms = np.sqrt(np.einsum("pc,p,pc->c", psi_c, wgt, psi_c))
        psi_c = psi_c / np.maximum(norms, 1e-30)

    rel = coords - R0
    out = np.empty((3, n_core, n_virt), dtype=float)
    for alpha in range(3):
        out[alpha] = psi_c.T @ ((rel[:, alpha] * wgt)[:, None] * psi_v)
    return out


def load_gipaw_data(path) -> GipawData:
    """Alias of :func:`parse_gipaw_upf`."""
    return parse_gipaw_upf(path)
