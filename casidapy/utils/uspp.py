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
    from scipy.special import sph_harm as _sph_harm

    if m > 0:
        ylm_pos = _sph_harm(m, l, phi, theta)
        result = np.sqrt(2.0) * (-1)**m * ylm_pos.real
    elif m < 0:
        ylm_pos = _sph_harm(-m, l, phi, theta)
        result = np.sqrt(2.0) * (-1)**(-m + 1) * ylm_pos.imag
    else:
        result = _sph_harm(0, l, phi, theta).real

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
    'qij', 'is_uspp', 'z_valence', 'has_nlcc', 'rho_core_rad'
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
    r_rad = np.array([float(x) for x in r_text.split()])

    rab_start = rab_end = None
    for i, line in enumerate(lines):
        if '<PP_RAB>' in line:
            rab_start = i + 1
        if '</PP_RAB>' in line:
            rab_end = i
            break
    rab_text = ' '.join(lines[rab_start:rab_end])
    rab = np.array([float(x) for x in rab_text.split()])

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
        n_points = int(block[1].strip())
        data_text = ' '.join(block[2:])
        vals = np.array([float(x) for x in data_text.split()])
        beta_full = np.zeros(len(r_rad))
        n_copy = min(len(vals), len(r_rad), n_points)
        beta_full[:n_copy] = vals[:n_copy]
        # Old format stores beta(r) * r
        beta_of_r = np.zeros_like(r_rad)
        beta_of_r[1:] = beta_full[1:] / r_rad[1:]
        beta_of_r[0] = beta_of_r[1] if len(beta_of_r) > 1 else 0.0
        beta_r.append(beta_of_r)

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
                vals = np.array([float(x) for x in data_text.split()])
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
                    qfcoef = np.array([float(x) for x in coef_text.split()])

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
            rho_core_rad = np.array([float(x) for x in nlcc_text.split()])
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


def _project_radial_to_3d(r_rad, f_rad, l, grid, atom_pos, cell, r_cutoff=None):
    """
    Project a radial function f(r) * Y_l^m onto the 3D grid for all m values.

    Returns list of (2l+1) 3D arrays, one per m value (m = -l, ..., +l).
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


def load_uspp_data(upf_files, grid, ions):
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

    Returns
    -------
    beta_projectors : list of DirectField
        All |beta_{n,l,m}> projectors on the grid.
    qij_augmentation : list of lists of DirectField
        qij_augmentation[I][J] is Q_{IJ}(r) on the grid.
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
          f"{len(set(a for a, _, _, _, _ in proj_map))} atoms)")

    # Pass 2: Build beta projectors on 3D grid
    beta_projectors = []
    beta_3d_cache = {}

    for glob_idx, (atom_idx, sym, p_idx, l_val, m_val) in enumerate(proj_map):
        cache_key = (atom_idx, p_idx)
        if cache_key not in beta_3d_cache:
            data = upf_data[sym]
            atom_pos = np.array(ions.positions[atom_idx])
            fields_lm = _project_radial_to_3d(
                data['r'], data['beta_r'][p_idx], l_val,
                grid, atom_pos, cell,
                r_cutoff=data['r'][-1]
            )
            beta_3d_cache[cache_key] = fields_lm

        m_idx = m_val + l_val
        beta_3d = beta_3d_cache[cache_key][m_idx]
        beta_projectors.append(
            DirectField(grid, rank=1, griddata_3d=beta_3d)
        )

    # Pass 3: Build Q_nm augmentation on 3D grid
    qij_augmentation = [[None] * n_proj_total for _ in range(n_proj_total)]

    atom_proj_ranges = {}
    for glob_idx, (atom_idx, sym, p_idx, l_val, m_val) in enumerate(proj_map):
        if atom_idx not in atom_proj_ranges:
            atom_proj_ranges[atom_idx] = []
        atom_proj_ranges[atom_idx].append((glob_idx, p_idx, l_val, m_val))

    for atom_idx, proj_list in atom_proj_ranges.items():
        sym = symbols[atom_idx]
        data = upf_data[sym]
        atom_pos = np.array(ions.positions[atom_idx])

        rx = np.asarray(grid.r[0])
        ry = np.asarray(grid.r[1])
        rz = np.asarray(grid.r[2])
        grid_pts = np.stack([rx, ry, rz], axis=0)
        dx, dy, dz = _minimum_image_displacement(atom_pos, grid_pts, cell)
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        for I_glob, I_ploc, I_l, I_m in proj_list:
            for J_glob, J_ploc, J_l, J_m in proj_list:
                key = (I_ploc, J_ploc)
                if key not in data['qij']:
                    qij_augmentation[I_glob][J_glob] = DirectField(
                        grid, rank=1, griddata_3d=np.zeros(grid.nr)
                    )
                    continue

                q_entries = data['qij'][key]
                q_3d = np.zeros(grid.nr)

                for l_aug, q_rad in q_entries:
                    interp_func = interp1d(data['r'], q_rad, kind='cubic',
                                           fill_value=0.0, bounds_error=False)
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

                qij_augmentation[I_glob][J_glob] = DirectField(
                    grid, rank=1, griddata_3d=q_3d
                )

    # Fill remaining None entries with zero
    for I in range(n_proj_total):
        for J in range(n_proj_total):
            if qij_augmentation[I][J] is None:
                qij_augmentation[I][J] = DirectField(
                    grid, rank=1, griddata_3d=np.zeros(grid.nr)
                )

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
            rx = np.asarray(grid.r[0])
            ry = np.asarray(grid.r[1])
            rz = np.asarray(grid.r[2])
            grid_pts = np.stack([rx, ry, rz], axis=0)
            dx, dy, dz = _minimum_image_displacement(atom_pos, grid_pts, cell)
            dist = np.sqrt(dx**2 + dy**2 + dz**2)

            rho_core_interp = interp1d(data['r'], data['rho_core_rad'],
                                       kind='cubic', fill_value=0.0,
                                       bounds_error=False)
            rho_c = rho_core_interp(dist)
            rho_c[dist > data['r'][-1]] = 0.0
            core_arr += rho_c

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
