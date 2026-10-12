"""
Unified Ultrasoft Pseudopotential (USPP) module for CasidaPy.

Consolidates all USPP functionality:
  - UPF file parsing (old Vanderbilt + UPF v2 XML)
  - Real spherical harmonics and Gaunt coefficients
  - 3D grid projection of beta projectors and Q_nm augmentation
  - USPP/NC setup helpers for Casida workflows
  - S-overlap normalization

Usage
-----
    from casidapy.utils.uspp import load_uspp_data, setup_uspp, normalize_uspp_wavefunctions

    beta_projectors, qij_augmentation, core_density = load_uspp_data(
        upf_files={"Ag": "ag_pbe_v1.4.uspp.F.UPF"},
        grid=grid,
        ions=ions,
    )
"""

import os
import re
import sys
import numpy as np
from scipy.interpolate import interp1d

try:
    from dftpy.field import DirectField
    from dftpy.functional import LocalPseudo
except ImportError:
    DirectField = None
    LocalPseudo = None


# =========================================================================
# Real spherical harmonics
# =========================================================================

def _real_spherical_harmonic(l, m, theta, phi):
    """
    Compute the real spherical harmonic Y_l^m(theta, phi).

    Uses the convention:
        m > 0 : ~ cos(m*phi)
        m = 0 : standard Y_l^0
        m < 0 : ~ sin(|m|*phi)
    """
    # scipy.special.sph_harm is deprecated in 1.15 and removed in 1.17; the
    # replacement swaps the argument order to (l, m, theta, phi).
    try:
        from scipy.special import sph_harm_y as _sph_harm_y

        def _y(mm):
            return _sph_harm_y(l, mm, theta, phi)
    except ImportError:  # scipy < 1.15
        from scipy.special import sph_harm as _sph_harm

        def _y(mm):
            return _sph_harm(mm, l, phi, theta)

    if m > 0:
        result = np.sqrt(2.0) * (-1)**m * _y(m).real
    elif m < 0:
        result = np.sqrt(2.0) * (-1)**(-m + 1) * _y(-m).imag
    else:
        result = _y(0).real

    return result


def _ylm_on_grid(l, m, dx, dy, dz, dist):
    """
    Evaluate real spherical harmonic Y_l^m for displacement vectors (dx, dy, dz).
    Handles the r=0 case gracefully.
    """
    safe_dist = np.where(dist > 1e-30, dist, 1.0)
    cos_theta = np.where(dist > 1e-30, dz / safe_dist, 0.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    phi = np.arctan2(dy, dx)

    return _real_spherical_harmonic(l, m, theta, phi)


def _real_gaunt(l1, m1, l2, m2, l3, m3):
    """
    Compute the real Gaunt coefficient:
        G = integral Y_{l1}^{m1} * Y_{l2}^{m2} * Y_{l3}^{m3} dOmega

    Uses numerical integration over angular coordinates.
    """
    if abs(m1) > l1 or abs(m2) > l2 or abs(m3) > l3:
        return 0.0
    if l3 < abs(l1 - l2) or l3 > l1 + l2:
        return 0.0
    if (l1 + l2 + l3) % 2 != 0:
        return 0.0

    n_theta = 50
    n_phi = 100
    theta = np.linspace(0, np.pi, n_theta, endpoint=False) + np.pi / (2 * n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')

    Y1 = _real_spherical_harmonic(l1, m1, THETA, PHI)
    Y2 = _real_spherical_harmonic(l2, m2, THETA, PHI)
    Y3 = _real_spherical_harmonic(l3, m3, THETA, PHI)

    dtheta = np.pi / n_theta
    dphi = 2 * np.pi / n_phi
    integrand = Y1 * Y2 * Y3 * np.sin(THETA)
    result = np.sum(integrand) * dtheta * dphi

    return float(result)


# =========================================================================
# UPF file parsing
# =========================================================================

def _floats_from_text(text):
    """Parse whitespace-separated floats, skipping labels (``2S``, ``Beta``, …).

    Old Troullier–Martins / Dal Corso UPFs put ``nl`` tags and cutoff radii
    after the ``PP_BETA`` radial table.  ``float(tok)`` on ``2S`` used to
    abort the whole parse, including callers that only needed ``has_nlcc``.
    """
    out = []
    for tok in str(text).split():
        tok = tok.replace("D", "E").replace("d", "e")
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return np.asarray(out, dtype=float)


def _inner_text(lines, tag):
    """Text between the first ``<tag>`` / ``</tag>`` pair, or ``None``."""
    open_t, close_t = f"<{tag}>", f"</{tag}>"
    start = None
    prefix = ""
    for i, line in enumerate(lines):
        if open_t in line and start is None:
            after = line.split(open_t, 1)[1]
            if close_t in after:
                return after.split(close_t, 1)[0]
            start = i + 1
            prefix = after
            continue
        if start is not None and close_t in line:
            before = line.split(close_t, 1)[0]
            return prefix + " " + " ".join(lines[start:i]) + " " + before
    return None


def _dij_from_numbers(nums, n_proj):
    """Old UPF ``PP_DIJ`` is sparse ``n_nz`` then ``i j D_ij`` (1-indexed, Ry)."""
    nums = np.asarray(nums, float).ravel()
    if nums.size == 0:
        return None
    n_side = max(int(n_proj), 0)

    def _sparse(flat, n0):
        if flat.size < 3 or flat.size % 3 != 0:
            return None
        triples = flat.reshape(-1, 3)
        n = max(n0, 1)
        dij = np.zeros((n, n), float)
        for i_idx, j_idx, dval in triples:
            ii = int(round(i_idx)) - 1
            jj = int(round(j_idx)) - 1
            if ii < 0 or jj < 0:
                return None
            need = max(n, ii + 1, jj + 1)
            if need > n:
                tmp = np.zeros((need, need), float)
                tmp[:n, :n] = dij
                dij, n = tmp, need
            dij[ii, jj] = float(dval)
            dij[jj, ii] = float(dval)
        return dij

    n_nz = int(round(nums[0]))
    rest = nums[1:]
    if n_nz > 0 and rest.size == 3 * n_nz:
        out = _sparse(rest, n_side)
        if out is not None:
            return out
    if n_side > 0 and rest.size >= n_side * n_side:
        return rest[: n_side * n_side].reshape(n_side, n_side)
    if n_side > 0 and nums.size >= n_side * n_side:
        return nums[: n_side * n_side].reshape(n_side, n_side)
    if rest.size >= 3 and rest.size % 3 == 0:
        return _sparse(rest, n_side)
    return None


def detect_upf_format(filepath):
    """
    Detect whether UPF file is old-format or new XML (v2) format.

    Returns 'v2' for UPF v2 XML format, 'old' for old Vanderbilt format.
    """
    with open(filepath, 'r') as f:
        first_lines = f.read(2000)
    if '<UPF version=' in first_lines:
        return 'v2'
    return 'old'


def _parse_upf_v2(filepath):
    """
    Parse a UPF v2 (XML) pseudopotential file.

    Returns dict with keys: 'r', 'rab', 'n_proj', 'beta_l', 'beta_r',
    'qij', 'dij_ry', 'vloc_rad_ry', 'is_uspp', 'z_valence', 'has_nlcc',
    'rho_core_rad'.  ``dij_ry`` and ``vloc_rad_ry`` keep the UPF's Rydberg
    units; converters live in :mod:`casidapy.embed.potential`.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(filepath)
    root = tree.getroot()

    header = root.find('.//PP_HEADER')
    is_uspp = header.get('is_ultrasoft', 'false').lower() == 'true'
    z_val = float(header.get('z_valence', '0'))
    n_proj = int(header.get('number_of_proj', '0'))
    has_nlcc = header.get('core_correction', 'false').lower() == 'true'

    r_text = root.find('.//PP_MESH/PP_R').text
    r_rad = np.array([float(x) for x in r_text.split()])
    rab_text = root.find('.//PP_MESH/PP_RAB').text
    rab = np.array([float(x) for x in rab_text.split()])

    beta_l = []
    beta_r = []
    nl = root.find('.//PP_NONLOCAL')

    for idx in range(1, n_proj + 1):
        tag = nl.find(f'PP_BETA.{idx}')
        if tag is None:
            break
        l_val = int(tag.get('angular_momentum'))
        beta_l.append(l_val)
        vals = np.array([float(x) for x in tag.text.split()])
        beta_full = np.zeros(len(r_rad))
        n_copy = min(len(vals), len(r_rad))
        beta_full[:n_copy] = vals[:n_copy]
        # UPF v2 stores beta(r) * r
        beta_of_r = np.zeros_like(r_rad)
        beta_of_r[1:] = beta_full[1:] / r_rad[1:]
        beta_of_r[0] = beta_of_r[1] if len(beta_of_r) > 1 else 0.0
        beta_r.append(beta_of_r)

    qij = {}
    aug = nl.find('PP_AUGMENTATION')
    if aug is not None:
        q_with_l = aug.get('q_with_l', 'false').lower() == 'true'

        if q_with_l:
            for child in aug:
                tag_name = child.tag
                if not tag_name.startswith('PP_QIJL'):
                    continue
                parts = tag_name.split('.')
                i_idx = int(parts[1]) - 1
                j_idx = int(parts[2]) - 1
                l_aug = int(parts[3])
                vals = np.array([float(x) for x in child.text.split()])
                q_full = np.zeros(len(r_rad))
                n_copy = min(len(vals), len(r_rad))
                q_full[:n_copy] = vals[:n_copy]
                key = (i_idx, j_idx)
                if key not in qij:
                    qij[key] = []
                qij[key].append((l_aug, q_full))
                if i_idx != j_idx:
                    key_sym = (j_idx, i_idx)
                    if key_sym not in qij:
                        qij[key_sym] = []
                    qij[key_sym].append((l_aug, q_full))
        else:
            for child in aug:
                tag_name = child.tag
                if not tag_name.startswith('PP_QIJ'):
                    continue
                parts = tag_name.split('.')
                if len(parts) < 3:
                    continue
                i_idx = int(parts[1]) - 1
                j_idx = int(parts[2]) - 1
                vals = np.array([float(x) for x in child.text.split()])
                q_full = np.zeros(len(r_rad))
                n_copy = min(len(vals), len(r_rad))
                q_full[:n_copy] = vals[:n_copy]
                key = (i_idx, j_idx)
                qij[key] = [(0, q_full)]
                if i_idx != j_idx:
                    qij[(j_idx, i_idx)] = [(0, q_full)]

    # Nonlocal coefficient matrix D_ij, stored in Ry.  Required to build
    # V_NL = Σ_ij D_ij |β_i><β_j|: for a norm-conserving PP the local channel
    # alone is strongly attractive and not a usable Hamiltonian, since the
    # repulsive projectors are what enforce orthogonality to the absent core.
    dij_ry = None
    dij_tag = nl.find('PP_DIJ') if nl is not None else None
    if dij_tag is not None and dij_tag.text:
        vals = _floats_from_text(dij_tag.text)
        if n_proj > 0 and vals.size >= n_proj * n_proj:
            dij_ry = vals[: n_proj * n_proj].reshape(n_proj, n_proj)
        elif vals.size:
            n_side = int(round(np.sqrt(vals.size)))
            if n_side * n_side == vals.size:
                dij_ry = vals.reshape(n_side, n_side)
            else:
                dij_ry = _dij_from_numbers(vals, n_proj)

    # Local channel v_loc(r), stored in Ry.  Needed to subtract one atom's own
    # local PP out of the host's total potential.
    vloc_rad_ry = None
    vloc_tag = root.find('.//PP_LOCAL')
    if vloc_tag is not None and vloc_tag.text:
        vals = _floats_from_text(vloc_tag.text)
        vloc_rad_ry = np.zeros(len(r_rad))
        n_copy = min(len(vals), len(r_rad))
        vloc_rad_ry[:n_copy] = vals[:n_copy]

    rho_core_rad = None
    if has_nlcc:
        nlcc_tag = root.find('.//PP_NLCC')
        if nlcc_tag is not None and nlcc_tag.text:
            rho_core_rad = np.array([float(x) for x in nlcc_tag.text.split()])
            if len(rho_core_rad) < len(r_rad):
                tmp = np.zeros(len(r_rad))
                tmp[:len(rho_core_rad)] = rho_core_rad
                rho_core_rad = tmp
            elif len(rho_core_rad) > len(r_rad):
                rho_core_rad = rho_core_rad[:len(r_rad)]

    return {
        'r': r_rad,
        'rab': rab,
        'n_proj': n_proj,
        'beta_l': beta_l,
        'beta_r': beta_r,
        'qij': qij,
        'dij_ry': dij_ry,
        'vloc_rad_ry': vloc_rad_ry,
        'is_uspp': is_uspp,
        'z_valence': z_val,
        'has_nlcc': has_nlcc,
        'rho_core_rad': rho_core_rad,
    }


def _parse_upf_old(filepath):
    """
    Parse an old-format (Vanderbilt) UPF pseudopotential file.
    Returns same dict structure as _parse_upf_v2.
    """
    with open(filepath, 'r') as f:
        lines = f.read().split('\n')

    # Radial mesh
    r_start = r_end = None
    for i, line in enumerate(lines):
        if '<PP_R>' in line:
            r_start = i + 1
        if '</PP_R>' in line:
            r_end = i
            break
    r_text = ' '.join(lines[r_start:r_end])
    r_rad = _floats_from_text(r_text)

    rab_start = rab_end = None
    for i, line in enumerate(lines):
        if '<PP_RAB>' in line:
            rab_start = i + 1
        if '</PP_RAB>' in line:
            rab_end = i
            break
    rab_text = ' '.join(lines[rab_start:rab_end])
    rab = _floats_from_text(rab_text)

    # Header info
    n_proj = 0
    z_val = 0.0
    is_uspp = False
    has_nlcc = False
    for line in lines:
        if 'Number of Wavefunctions, Number of Projectors' in line:
            parts = line.split()
            n_proj = int(parts[1])
        if 'Z valence' in line:
            z_val = float(line.split()[0])
        if 'US' in line and 'Ultrasoft' in line:
            is_uspp = True
        if 'Nonlinear Core Correction' in line:
            has_nlcc = line.strip().startswith('T')

    # Beta projectors
    beta_l = []
    beta_r = []
    beta_blocks = []
    in_beta = False
    current_block = []
    for line in lines:
        if '<PP_BETA>' in line:
            in_beta = True
            current_block = []
            continue
        if '</PP_BETA>' in line:
            in_beta = False
            beta_blocks.append(current_block)
            continue
        if in_beta:
            current_block.append(line)

    for block in beta_blocks:
        header_parts = block[0].split()
        l_val = int(header_parts[1])
        beta_l.append(l_val)
        n_points = int(block[1].split()[0])
        data_text = ' '.join(block[2:])
        vals = _floats_from_text(data_text)
        beta_full = np.zeros(len(r_rad))
        n_copy = min(len(vals), len(r_rad), n_points)
        beta_full[:n_copy] = vals[:n_copy]
        # Old format stores beta(r) * r
        beta_of_r = np.zeros_like(r_rad)
        beta_of_r[1:] = beta_full[1:] / r_rad[1:]
        beta_of_r[0] = beta_of_r[1] if len(beta_of_r) > 1 else 0.0
        beta_r.append(beta_of_r)

    if beta_l:
        n_proj = len(beta_l)

    dij_ry = None
    dij_text = _inner_text(lines, "PP_DIJ")
    if dij_text is not None:
        dij_ry = _dij_from_numbers(_floats_from_text(dij_text), n_proj)

    vloc_rad_ry = None
    vloc_text = _inner_text(lines, "PP_LOCAL")
    if vloc_text is not None:
        vloc_vals = _floats_from_text(vloc_text)
        vloc_rad_ry = np.zeros(len(r_rad))
        n_copy = min(len(vloc_vals), len(r_rad))
        vloc_rad_ry[:n_copy] = vloc_vals[:n_copy]

    # Q_ij augmentation
    qij = {}
    qij_start = qij_end = None
    for i, line in enumerate(lines):
        if '<PP_QIJ>' in line:
            qij_start = i + 1
        if '</PP_QIJ>' in line:
            qij_end = i
            break

    if qij_start is not None and qij_end is not None:
        qij_lines = lines[qij_start:qij_end]
        nqf_line = qij_lines[0].strip()
        nqf = int(nqf_line.split()[0])

        rinner = {}
        in_rinner = False
        line_idx = 1
        while line_idx < len(qij_lines):
            line = qij_lines[line_idx].strip()
            if '<PP_RINNER>' in line:
                in_rinner = True
                line_idx += 1
                continue
            if '</PP_RINNER>' in line:
                in_rinner = False
                line_idx += 1
                continue
            if in_rinner:
                parts = line.split()
                l_idx = int(parts[0]) - 1
                r_inner = float(parts[1])
                rinner[l_idx] = r_inner
                line_idx += 1
                continue

            match = re.match(r'\s*(\d+)\s+(\d+)\s+(\d+)\s+i\s+j\s+\(l\(j\)\)', line)
            if match:
                i_idx = int(match.group(1)) - 1
                j_idx = int(match.group(2)) - 1
                l_qij = int(match.group(3))
                line_idx += 1
                # Skip Q_int line
                line_idx += 1

                data_lines = []
                while line_idx < len(qij_lines):
                    test_line = qij_lines[line_idx].strip()
                    if '<PP_QFCOEF>' in test_line:
                        break
                    if re.match(r'\s*\d+\s+\d+\s+\d+\s+i\s+j', test_line):
                        break
                    data_lines.append(test_line)
                    line_idx += 1

                data_text = ' '.join(data_lines)
                vals = _floats_from_text(data_text)
                q_full = np.zeros(len(r_rad))
                n_copy = min(len(vals), len(r_rad))
                q_full[:n_copy] = vals[:n_copy]

                qfcoef = None
                if line_idx < len(qij_lines) and '<PP_QFCOEF>' in qij_lines[line_idx]:
                    line_idx += 1
                    coef_lines = []
                    while line_idx < len(qij_lines):
                        if '</PP_QFCOEF>' in qij_lines[line_idx]:
                            line_idx += 1
                            break
                        coef_lines.append(qij_lines[line_idx].strip())
                        line_idx += 1
                    coef_text = ' '.join(coef_lines)
                    qfcoef = _floats_from_text(coef_text)

                if nqf > 0 and qfcoef is not None and l_qij in rinner:
                    r_in = rinner[l_qij]
                    mask = r_rad < r_in
                    li = beta_l[i_idx]
                    lj = beta_l[j_idx]
                    n_ltot = (li + lj - abs(li - lj)) // 2 + 1
                    for lt_idx in range(n_ltot):
                        ltot = abs(li - lj) + 2 * lt_idx
                        coef_start = lt_idx * nqf
                        coef_end = coef_start + nqf
                        if coef_end <= len(qfcoef):
                            coefs = qfcoef[coef_start:coef_end]
                            for k, c in enumerate(coefs):
                                q_full[mask] += c * r_rad[mask]**(2 * ltot + 2 * k)

                key = (i_idx, j_idx)
                if key not in qij:
                    qij[key] = []
                qij[key].append((l_qij, q_full))
                if i_idx != j_idx:
                    key_sym = (j_idx, i_idx)
                    if key_sym not in qij:
                        qij[key_sym] = []
                    qij[key_sym].append((l_qij, q_full))
                continue

            line_idx += 1

    # NLCC core charge density
    rho_core_rad = None
    if has_nlcc:
        nlcc_start = nlcc_end = None
        for i, line in enumerate(lines):
            if '<PP_NLCC>' in line:
                nlcc_start = i + 1
            if '</PP_NLCC>' in line:
                nlcc_end = i
                break
        if nlcc_start is not None and nlcc_end is not None:
            nlcc_text = ' '.join(lines[nlcc_start:nlcc_end])
            rho_core_rad = _floats_from_text(nlcc_text)
            if len(rho_core_rad) < len(r_rad):
                tmp = np.zeros(len(r_rad))
                tmp[:len(rho_core_rad)] = rho_core_rad
                rho_core_rad = tmp
            elif len(rho_core_rad) > len(r_rad):
                rho_core_rad = rho_core_rad[:len(r_rad)]

    return {
        'r': r_rad,
        'rab': rab,
        'n_proj': n_proj,
        'beta_l': beta_l,
        'beta_r': beta_r,
        'qij': qij,
        'dij_ry': dij_ry,
        'vloc_rad_ry': vloc_rad_ry,
        'is_uspp': is_uspp,
        'z_valence': z_val,
        'has_nlcc': has_nlcc,
        'rho_core_rad': rho_core_rad,
    }


def parse_upf(filepath):
    """
    Parse a UPF pseudopotential file (auto-detects format).

    Returns dict with keys: 'r', 'rab', 'n_proj', 'beta_l', 'beta_r',
    'qij', 'is_uspp', 'z_valence', 'has_nlcc', 'rho_core_rad'
    """
    fmt = detect_upf_format(filepath)
    if fmt == 'v2':
        return _parse_upf_v2(filepath)
    return _parse_upf_old(filepath)


def print_upf_info(filepath):
    """Print summary information about a UPF pseudopotential file."""
    data = parse_upf(filepath)
    fmt = detect_upf_format(filepath)
    print(f"File: {filepath}")
    print(f"  Format: {'UPF v2 (XML)' if fmt == 'v2' else 'Old Vanderbilt'}")
    print(f"  Ultrasoft: {data['is_uspp']}")
    print(f"  Z valence: {data['z_valence']}")
    print(f"  NLCC: {data.get('has_nlcc', False)}")
    if data.get('rho_core_rad') is not None:
        print(f"  Core density: {len(data['rho_core_rad'])} radial points")
    print(f"  Radial mesh: {len(data['r'])} points, "
          f"r_max = {data['r'][-1]:.4f} Bohr")
    print(f"  Projectors: {data['n_proj']}")
    for i, l in enumerate(data['beta_l']):
        print(f"    beta_{i+1}: l = {l}  ({['s','p','d','f','g'][l]})")
    print(f"  Q_ij pairs: {len(data['qij'])}")
    for (i, j), entries in sorted(data['qij'].items()):
        l_list = [l for l, _ in entries]
        print(f"    Q({i+1},{j+1}): L channels = {l_list}")
    n_channels = sum(2 * l + 1 for l in data['beta_l'])
    print(f"  Total (l,m) channels per atom: {n_channels}")


# =========================================================================
# 3D grid projection
# =========================================================================

class SparseGridField:
    """Atom-centered field: nonzero only on a compact set of flat grid indices.

    USPP β and Q are supported within ~1–2 Bohr of each atom. Storing them as
    full FFT grids (``n_proj × N_FFT``) OOMs on dense charge grids; this keeps
    only the nonzero samples (typically ~10⁴–10⁵ pts/atom).
    """

    __slots__ = ("indices", "values", "shape")

    def __init__(self, indices, values, shape):
        self.indices = np.ascontiguousarray(indices, dtype=np.int64)
        self.values = np.ascontiguousarray(values, dtype=np.float64)
        self.shape = tuple(int(x) for x in shape)

    def __array__(self, dtype=None):
        """Dense fallback (avoid in hot paths — allocates full grid)."""
        out = np.zeros(self.shape, dtype=np.float64)
        if self.indices.size:
            out.ravel(order="C")[self.indices] = self.values
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return out

    @property
    def nnz(self) -> int:
        return int(self.indices.size)


def _minimum_image_displacement(pos, point, cell):
    """
    Compute minimum-image displacement vector from atom at `pos` to `point`,
    applying periodic boundary conditions.
    """
    inv_cell = np.linalg.inv(cell)

    delta = point - pos[..., np.newaxis, np.newaxis, np.newaxis] if point.ndim == 4 else point - pos
    if delta.ndim == 4:
        s = np.einsum('ij,jxyz->ixyz', inv_cell, delta)
    else:
        s = inv_cell @ delta

    s = s - np.round(s)

    if s.ndim == 4:
        d = np.einsum('ij,jxyz->ixyz', cell, s)
    else:
        d = cell @ s

    return d[0], d[1], d[2]


def _beta_support_radius(r_rad, f_rad, floor: float = 1e-12) -> float:
    """Radial cutoff where |f| last exceeds ``floor`` (plus small pad)."""
    r = np.asarray(r_rad, dtype=float)
    f = np.asarray(f_rad, dtype=float)
    nz = np.where(np.abs(f) > floor)[0]
    if nz.size == 0:
        return float(r[0]) if r.size else 0.0
    return float(r[nz[-1]] * 1.05)


def _project_radial_to_3d(r_rad, f_rad, l, grid, atom_pos, cell, r_cutoff=None):
    """
    Project a radial function f(r) * Y_l^m onto the 3D grid for all m values.

    Returns list of (2l+1) dense 3D arrays, one per m (m = -l, ..., +l).
    Prefer :func:`_project_radial_to_3d_sparse` for large grids.
    """
    rx = np.asarray(grid.r[0])
    ry = np.asarray(grid.r[1])
    rz = np.asarray(grid.r[2])
    grid_pts = np.stack([rx, ry, rz], axis=0)

    dx, dy, dz = _minimum_image_displacement(atom_pos, grid_pts, cell)
    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    if r_cutoff is None:
        r_cutoff = r_rad[-1]
    interp_func = interp1d(r_rad, f_rad, kind='cubic',
                           fill_value=0.0, bounds_error=False)
    f_3d = interp_func(dist)
    f_3d[dist > r_cutoff] = 0.0

    fields_lm = []
    for m in range(-l, l + 1):
        ylm = _ylm_on_grid(l, m, dx, dy, dz, dist)
        fields_lm.append(f_3d * ylm)

    return fields_lm


def _samples_within_cutoff(grid, atom_pos, cell, r_cutoff):
    """Minimum-image samples inside ``r_cutoff``, without a full-grid temporary.

    Crystal coordinates match :attr:`DirectGrid.r` (``s = index / nr``, endpoint
    excluded). The displacement matches :func:`_minimum_image_displacement`.
    Returns flat C-order indices and ``dx, dy, dz, dist``.
    """
    lattice = np.asarray(grid.lattice, dtype=np.float64).reshape(3, 3)
    cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    nr = np.asarray(grid.nrR, dtype=np.int64).reshape(3)
    pos = np.asarray(atom_pos, dtype=np.float64).reshape(3)
    r_cutoff = float(r_cutoff)
    empty_i = np.zeros(0, dtype=np.int64)
    empty_d = np.zeros(0, dtype=np.float64)
    if r_cutoff <= 0.0 or np.any(nr < 1):
        return empty_i, empty_d, empty_d, empty_d, empty_d

    lengths = np.linalg.norm(lattice, axis=1)
    spacing = float(np.min(lengths / np.maximum(nr, 1)))
    reach = int(np.ceil(r_cutoff / max(spacing, 1e-12))) + 2
    s_atom = np.linalg.solve(lattice.T, pos)
    centers = np.rint(s_atom * nr).astype(np.int64)
    axes = [
        np.arange(int(centers[d]) - reach, int(centers[d]) + reach + 1, dtype=np.int64)
        for d in range(3)
    ]
    # Chunk so a cutoff that spans the cell cannot allocate the whole mesh at once.
    nj = int(axes[1].size)
    nk = int(axes[2].size)
    step = max(1, 500_000 // max(nj * nk, 1))
    inv_cell = np.linalg.inv(cell)
    idx_parts = []
    disp_parts = []
    dist_parts = []
    for start in range(0, int(axes[0].size), step):
        ii, jj, kk = np.meshgrid(
            axes[0][start:start + step], axes[1], axes[2], indexing="ij"
        )
        iw = np.mod(ii, nr[0]).ravel()
        jw = np.mod(jj, nr[1]).ravel()
        kw = np.mod(kk, nr[2]).ravel()
        flat = (iw * nr[1] + jw) * nr[2] + kw
        order = np.argsort(flat, kind="mergesort")
        flat_s = flat[order]
        first = np.empty(flat_s.size, dtype=bool)
        first[0] = True
        if flat_s.size > 1:
            first[1:] = flat_s[1:] != flat_s[:-1]
        sel = order[first]
        iw = iw[sel]
        jw = jw[sel]
        kw = kw[sel]
        flat = flat[sel]
        s = np.stack((iw / nr[0], jw / nr[1], kw / nr[2]), axis=0)
        point = np.einsum("j...,jk->k...", s, lattice)
        frac = inv_cell @ (point - pos.reshape(3, 1))
        frac = frac - np.round(frac)
        disp = cell @ frac
        dist = np.sqrt(np.sum(disp * disp, axis=0))
        keep = dist <= r_cutoff
        if not np.any(keep):
            continue
        idx_parts.append(flat[keep])
        disp_parts.append(disp[:, keep])
        dist_parts.append(dist[keep])
    if not idx_parts:
        return empty_i, empty_d, empty_d, empty_d, empty_d
    idx = np.concatenate(idx_parts)
    disp = np.concatenate(disp_parts, axis=1)
    dist = np.concatenate(dist_parts)
    # Chunks can repeat a wrapped index. Keep the first copy.
    order = np.argsort(idx, kind="mergesort")
    idx = idx[order]
    disp = disp[:, order]
    dist = dist[order]
    first = np.empty(idx.size, dtype=bool)
    first[0] = True
    if idx.size > 1:
        first[1:] = idx[1:] != idx[:-1]
    idx = np.ascontiguousarray(idx[first], dtype=np.int64)
    disp = disp[:, first]
    dist = np.ascontiguousarray(dist[first], dtype=np.float64)
    return (
        idx,
        np.ascontiguousarray(disp[0], dtype=np.float64),
        np.ascontiguousarray(disp[1], dtype=np.float64),
        np.ascontiguousarray(disp[2], dtype=np.float64),
        dist,
    )


def _project_radial_to_3d_sparse(r_rad, f_rad, l, grid, atom_pos, cell, r_cutoff=None):
    """Like :func:`_project_radial_to_3d` but returns ``SparseGridField`` list."""
    if r_cutoff is None:
        r_cutoff = _beta_support_radius(r_rad, f_rad)
    shape = tuple(int(x) for x in np.asarray(grid.nrR).reshape(3))
    idx, dx_m, dy_m, dz_m, dist_m = _samples_within_cutoff(
        grid, atom_pos, cell, r_cutoff
    )
    if idx.size == 0:
        empty = SparseGridField(
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            shape,
        )
        return [empty for _ in range(2 * l + 1)]

    interp_func = interp1d(r_rad, f_rad, kind="cubic", fill_value=0.0, bounds_error=False)
    f_m = np.asarray(interp_func(dist_m), dtype=np.float64)

    fields_lm = []
    for m in range(-l, l + 1):
        ylm = _ylm_on_grid(l, m, dx_m, dy_m, dz_m, dist_m)
        fields_lm.append(SparseGridField(idx, f_m * np.asarray(ylm, dtype=np.float64), shape))
    return fields_lm


def load_uspp_data(upf_files, grid, ions, *, sparse: bool = True):
    """
    Parse UPF files and build beta projectors and Q_nm augmentation charges
    on the 3D real-space grid, including full (l, m) angular channels.

    Parameters
    ----------
    upf_files : dict
        Mapping element symbol -> UPF file path.
    grid : DirectGrid
        3D real-space grid.
    ions : Ions
        DFTpy Ions object with atomic positions and symbols.
    sparse : bool
        If True (default), store β / Q as :class:`SparseGridField` near each
        atom. Full dense grids OOMs for ~50M-point charge grids.

    Returns
    -------
    beta_projectors : list of DirectField or SparseGridField
        All |beta_{n,l,m}> projectors on the grid.
    qij_augmentation : list of lists
        qij_augmentation[I][J] is Q_{IJ}(r) (or ``None`` if identically zero).
    core_density : DirectField or None
        3D NLCC core charge density (sum over all atoms), or None.
    """
    if DirectField is None:
        raise ImportError("DFTpy is required for load_uspp_data")

    cell = np.array(ions.cell)
    symbols = list(ions.symbols) if hasattr(ions, 'symbols') else [str(s) for s in ions.labels]

    upf_data = {}
    for elem, path in upf_files.items():
        upf_data[elem] = parse_upf(path)
        if not upf_data[elem]['is_uspp']:
            print(f"WARNING: {path} is not an ultrasoft PP. "
                  f"USPP corrections may be zero/invalid.")

    # Pass 1: Determine global projector layout
    proj_map = []
    for atom_idx, sym in enumerate(symbols):
        if sym not in upf_data:
            continue
        data = upf_data[sym]
        for p_idx, l_val in enumerate(data['beta_l']):
            for m_val in range(-l_val, l_val + 1):
                proj_map.append((atom_idx, sym, p_idx, l_val, m_val))

    n_proj_total = len(proj_map)
    print(f"USPP: {n_proj_total} total projectors "
          f"({len(set(s for _, s, _, _, _ in proj_map))} elements, "
          f"{len(set(a for a, _, _, _, _ in proj_map))} atoms)"
          f"{' [sparse]' if sparse else ''}")

    # Pass 2: Build beta projectors on 3D grid
    beta_projectors = []
    beta_3d_cache = {}

    for glob_idx, (atom_idx, sym, p_idx, l_val, m_val) in enumerate(proj_map):
        cache_key = (atom_idx, p_idx)
        if cache_key not in beta_3d_cache:
            data = upf_data[sym]
            atom_pos = np.array(ions.positions[atom_idx])
            rcut = _beta_support_radius(data['r'], data['beta_r'][p_idx])
            if sparse:
                fields_lm = _project_radial_to_3d_sparse(
                    data['r'], data['beta_r'][p_idx], l_val,
                    grid, atom_pos, cell, r_cutoff=rcut,
                )
            else:
                fields_lm = _project_radial_to_3d(
                    data['r'], data['beta_r'][p_idx], l_val,
                    grid, atom_pos, cell, r_cutoff=rcut,
                )
            beta_3d_cache[cache_key] = fields_lm

        m_idx = m_val + l_val
        beta_3d = beta_3d_cache[cache_key][m_idx]
        if sparse:
            beta_projectors.append(beta_3d)
        else:
            beta_projectors.append(
                DirectField(grid, rank=1, griddata_3d=beta_3d)
            )

    if sparse and beta_projectors:
        nnz = sum(b.nnz for b in beta_projectors)
        print(f"USPP: sparse betas nnz={nnz:,}  (~{nnz * 8 / 1e6:.1f} MB values)")

    # Pass 3: Build Q_nm augmentation on 3D grid
    qij_augmentation = [[None] * n_proj_total for _ in range(n_proj_total)]

    atom_proj_ranges = {}
    for glob_idx, (atom_idx, sym, p_idx, l_val, m_val) in enumerate(proj_map):
        if atom_idx not in atom_proj_ranges:
            atom_proj_ranges[atom_idx] = []
        atom_proj_ranges[atom_idx].append((glob_idx, p_idx, l_val, m_val))

    n_q_stored = 0
    for atom_idx, proj_list in atom_proj_ranges.items():
        sym = symbols[atom_idx]
        data = upf_data[sym]
        atom_pos = np.array(ions.positions[atom_idx])

        # Q support ≈ beta support; use max beta rcut for this species.
        rcut_atom = max(
            (_beta_support_radius(data['r'], data['beta_r'][p]) for p in range(len(data['beta_l']))),
            default=float(data['r'][-1]),
        )
        if sparse:
            idx, dx_m, dy_m, dz_m, dist_m = _samples_within_cutoff(
                grid, atom_pos, cell, rcut_atom
            )
            if idx.size == 0:
                continue
        else:
            rx = np.asarray(grid.r[0])
            ry = np.asarray(grid.r[1])
            rz = np.asarray(grid.r[2])
            grid_pts = np.stack([rx, ry, rz], axis=0)
            dx, dy, dz = _minimum_image_displacement(atom_pos, grid_pts, cell)
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            idx = dist_m = dx_m = dy_m = dz_m = None

        for I_glob, I_ploc, I_l, I_m in proj_list:
            for J_glob, J_ploc, J_l, J_m in proj_list:
                key = (I_ploc, J_ploc)
                if key not in data['qij']:
                    continue

                q_entries = data['qij'][key]
                if sparse:
                    q_vals = np.zeros(idx.size, dtype=np.float64)
                else:
                    q_3d = np.zeros(grid.nr)

                for l_aug, q_rad in q_entries:
                    interp_func = interp1d(data['r'], q_rad, kind='cubic',
                                           fill_value=0.0, bounds_error=False)
                    if sparse:
                        q_radial = np.asarray(interp_func(dist_m), dtype=np.float64)
                        q_radial[dist_m > data['r'][-1]] = 0.0
                        if l_aug == 0:
                            if I_l == J_l and I_m == J_m:
                                q_vals += q_radial / np.sqrt(4.0 * np.pi)
                        else:
                            for M in range(-l_aug, l_aug + 1):
                                gaunt = _real_gaunt(I_l, I_m, J_l, J_m, l_aug, M)
                                if abs(gaunt) > 1e-15:
                                    ylm = _ylm_on_grid(l_aug, M, dx_m, dy_m, dz_m, dist_m)
                                    q_vals += gaunt * q_radial * np.asarray(ylm, dtype=np.float64)
                    else:
                        q_radial_3d = interp_func(dist)
                        q_radial_3d[dist > data['r'][-1]] = 0.0
                        if l_aug == 0:
                            if I_l == J_l and I_m == J_m:
                                q_3d += q_radial_3d / np.sqrt(4.0 * np.pi)
                        else:
                            for M in range(-l_aug, l_aug + 1):
                                gaunt = _real_gaunt(I_l, I_m, J_l, J_m, l_aug, M)
                                if abs(gaunt) > 1e-15:
                                    ylm = _ylm_on_grid(l_aug, M, dx, dy, dz, dist)
                                    q_3d += gaunt * q_radial_3d * ylm

                if sparse:
                    if np.any(np.abs(q_vals) > 1e-18):
                        qij_augmentation[I_glob][J_glob] = SparseGridField(
                            idx, q_vals, tuple(int(x) for x in np.asarray(grid.nrR).reshape(3))
                        )
                        n_q_stored += 1
                else:
                    qij_augmentation[I_glob][J_glob] = DirectField(
                        grid, rank=1, griddata_3d=q_3d
                    )
                    n_q_stored += 1

    if not sparse:
        # Dense path: fill missing entries with explicit zeros (legacy).
        for I in range(n_proj_total):
            for J in range(n_proj_total):
                if qij_augmentation[I][J] is None:
                    qij_augmentation[I][J] = DirectField(
                        grid, rank=1, griddata_3d=np.zeros(grid.nr)
                    )
    else:
        print(f"USPP: sparse Q_ij stored pairs={n_q_stored} "
              f"(None = identically zero; no full-grid zero fill)")

    # Pass 4: Build NLCC core charge density on 3D grid
    core_density_3d = None
    any_nlcc = any(upf_data[sym].get('has_nlcc', False) for sym in upf_data)
    if any_nlcc:
        core_arr = np.zeros(grid.nr)
        for atom_idx, sym in enumerate(symbols):
            if sym not in upf_data:
                continue
            data = upf_data[sym]
            if not data.get('has_nlcc', False) or data.get('rho_core_rad') is None:
                continue

            atom_pos = np.array(ions.positions[atom_idx])
            rcut_core = _beta_support_radius(data['r'], data['rho_core_rad'])
            idx, _dx, _dy, _dz, dist = _samples_within_cutoff(
                grid, atom_pos, cell, rcut_core
            )
            if idx.size == 0:
                continue
            rho_core_interp = interp1d(data['r'], data['rho_core_rad'],
                                       kind='cubic', fill_value=0.0,
                                       bounds_error=False)
            rho_c = np.asarray(rho_core_interp(dist), dtype=np.float64)
            core_arr.ravel(order="C")[idx] += rho_c

        core_density_3d = DirectField(grid, rank=1, griddata_3d=core_arr)
        dv = grid.dV
        q_core = np.sum(core_arr) * dv
        print(f"USPP: NLCC core charge density built on grid, "
              f"integrated core charge = {q_core:.4f}")

    return beta_projectors, qij_augmentation, core_density_3d


# =========================================================================
# Setup helpers (CLI workflow)
# =========================================================================

def parse_pseudo_map(pseudo_map_str):
    """
    Parse pseudopotential map string into dictionary.
    Format: "Element1:file1.upf,Element2:file2.upf"
    """
    pp_dict = {}
    if not pseudo_map_str:
        return pp_dict
    for pair in pseudo_map_str.split(","):
        pair = pair.strip()
        if ":" in pair:
            el, fn = pair.split(":", 1)
            pp_dict[el.strip()] = fn.strip()
    return pp_dict


def setup_uspp(args, grid, ions, atoms, resolved_workdir, comm=None):
    """
    Setup ultrasoft pseudopotential corrections from CLI args.

    Returns (beta_projectors, qij_augmentation, core_density, use_uspp_flag, pp_list).
    """
    rank = 0 if comm is None else comm.Get_rank()
    use_uspp_flag = False
    beta_projectors = None
    qij_augmentation = None
    core_density = None
    pp_list = None

    if not args.use_uspp:
        return beta_projectors, qij_augmentation, core_density, use_uspp_flag, pp_list

    if args.uspp_map is None:
        if rank == 0:
            print("    FATAL: --use-uspp requires --uspp-map to specify USPP files.", flush=True)
            print('    Example: --uspp-map "Ag:ag_pbe_v1.4.uspp.F.UPF"', flush=True)
        if comm is not None:
            comm.Abort(1)
        else:
            raise ValueError("--use-uspp requires --uspp-map")

    uspp_files = parse_pseudo_map(args.uspp_map)
    uspp_paths = {}
    for el, fn in uspp_files.items():
        p = fn if os.path.isabs(fn) else os.path.join(resolved_workdir, fn)
        if not os.path.exists(p):
            if rank == 0:
                print(f"    FATAL: USPP file not found: {p}", flush=True)
            if comm is not None:
                comm.Abort(1)
            else:
                raise FileNotFoundError(f"USPP file not found: {p}")
        uspp_paths[el] = p

    if rank == 0:
        for el, p in uspp_paths.items():
            print(f"    USPP file for {el}: {p}")
        sys.stdout.flush()

    try:
        beta_projectors, qij_augmentation, uspp_core_density = load_uspp_data(
            upf_files=uspp_paths, grid=grid, ions=ions
        )
        use_uspp_flag = True
        core_density = uspp_core_density
        if rank == 0:
            if core_density is not None:
                print(
                    f"    NLCC core density from USPP (integral: "
                    f"{np.sum(np.asarray(core_density)) * grid.dV:.4f})"
                )
            else:
                print("    No NLCC in USPP file(s), core_density=None")
            print(f"    Loaded {len(beta_projectors)} beta projectors")
            print(f"    Q_ij augmentation: {len(qij_augmentation)}x{len(qij_augmentation[0])}")
            sys.stdout.flush()
    except Exception as e:
        if rank == 0:
            print(f"    ERROR loading USPP data: {e}", flush=True)
            import traceback
            traceback.print_exc()
        use_uspp_flag = False
        if comm is None:
            raise RuntimeError(f"Failed to load USPP data: {e}")

    return beta_projectors, qij_augmentation, core_density, use_uspp_flag, pp_list


def setup_nc_pseudos(args, grid, ions, atoms, resolved_workdir, core_density, use_uspp_flag, comm=None):
    """
    Setup norm-conserving pseudopotentials using LocalPseudo.

    Returns (core_density, pseudo).
    """
    rank = 0 if comm is None else comm.Get_rank()
    core_density_new = core_density
    pseudo = None

    if not args.pseudo_map:
        if not use_uspp_flag:
            if rank == 0:
                print("    FATAL: Must provide --pseudo-map (NC PPs) or --use-uspp with --uspp-map", flush=True)
            if comm is not None:
                comm.Abort(1)
            else:
                raise ValueError("Must provide --pseudo-map or --use-uspp")
        return core_density_new, pseudo

    if LocalPseudo is None:
        if rank == 0:
            print("    ERROR: DFTpy not available for LocalPseudo", flush=True)
        if comm is not None:
            comm.Abort(1)
        else:
            raise ImportError("DFTpy is required for norm-conserving pseudopotentials")

    pp_list = parse_pseudo_map(args.pseudo_map)
    elems = set(atoms.get_chemical_symbols())
    if not elems.issubset(set(pp_list.keys())):
        missing = sorted(elems - set(pp_list.keys()))
        if rank == 0:
            print(f"    FATAL: Missing pseudopotentials for elements: {missing}", flush=True)
        if comm is not None:
            comm.Abort(1)
        else:
            raise ValueError(f"Missing pseudopotentials for elements: {missing}")

    for el, pp_file in pp_list.items():
        pp_path = pp_file if os.path.isabs(pp_file) else os.path.join(resolved_workdir, pp_file)
        if os.path.exists(pp_path):
            try:
                chk = parse_upf(pp_path)
                if chk["is_uspp"]:
                    if rank == 0:
                        print(f"    WARNING: {pp_file} for {el} is ultrasoft!", flush=True)
                        print("    LocalPseudo requires norm-conserving PPs.", flush=True)
                    if comm is not None:
                        comm.Abort(1)
                    else:
                        raise ValueError(f"{pp_file} for {el} is ultrasoft")
            except (ValueError, FileNotFoundError):
                raise
            except Exception:
                pass

    try:
        pseudo = LocalPseudo(grid=grid, ions=ions, PP_list=pp_list)
        if core_density_new is None:
            core_density_new = pseudo.core_density
            if rank == 0 and core_density_new is not None:
                print("    Core density from LocalPseudo (NC PP)")
                sys.stdout.flush()
    except Exception as e:
        if rank == 0:
            print(f"    WARNING: LocalPseudo failed: {e}", flush=True)
            if not use_uspp_flag:
                print("    FATAL: No USPP path and LocalPseudo failed.", flush=True)
                import traceback
                traceback.print_exc()
        if not use_uspp_flag:
            if comm is not None:
                comm.Abort(1)
            else:
                raise RuntimeError(f"LocalPseudo failed: {e}")

    return core_density_new, pseudo


# =========================================================================
# S-overlap normalization
# =========================================================================

def normalize_uspp_wavefunction(psi, beta_projectors, qij_augmentation, grid):
    """
    Normalize a USPP wavefunction using the S-overlap operator.
    Normalization condition: <psi|S|psi> = 1
    where S = 1 + sum_{nm} |beta_n> Q_nm <beta_m|
    """
    n_proj = len(beta_projectors)

    norm_l2 = (psi.conj() * psi).integral().real

    proj_overlaps = np.zeros(n_proj, dtype=complex)
    for n in range(n_proj):
        proj_overlaps[n] = (beta_projectors[n].conj() * psi).integral()

    norm_aug = 0.0
    for n in range(n_proj):
        for m in range(n_proj):
            q_integral = qij_augmentation[n][m].integral().real
            norm_aug += (proj_overlaps[n].conj() * q_integral * proj_overlaps[m]).real

    norm_s = norm_l2 + norm_aug
    if norm_s <= 0:
        raise ValueError(f"S-overlap norm is non-positive: {norm_s}")

    return psi / np.sqrt(norm_s)


def normalize_uspp_wavefunctions(psi_list, beta_projectors, qij_augmentation, grid):
    """Normalize a list of USPP wavefunctions using S-overlap."""
    return [
        normalize_uspp_wavefunction(psi, beta_projectors, qij_augmentation, grid)
        for psi in psi_list
    ]


def check_uspp_normalization(psi_list, beta_projectors, qij_augmentation, grid):
    """Compute <psi|S|psi> norms for diagnostics. Should be ~1.0 for each."""
    n_proj = len(beta_projectors)
    norms = []
    for psi in psi_list:
        norm_l2 = (psi.conj() * psi).integral().real
        proj_overlaps = np.zeros(n_proj, dtype=complex)
        for n in range(n_proj):
            proj_overlaps[n] = (beta_projectors[n].conj() * psi).integral()
        norm_aug = 0.0
        for n in range(n_proj):
            for m in range(n_proj):
                q_integral = qij_augmentation[n][m].integral().real
                norm_aug += (proj_overlaps[n].conj() * q_integral * proj_overlaps[m]).real
        norms.append(norm_l2 + norm_aug)
    return norms


# =========================================================================
# Public API
# =========================================================================

__all__ = [
    "parse_upf",
    "print_upf_info",
    "detect_upf_format",
    "load_uspp_data",
    "parse_pseudo_map",
    "setup_uspp",
    "setup_nc_pseudos",
    "normalize_uspp_wavefunction",
    "normalize_uspp_wavefunctions",
    "check_uspp_normalization",
]
