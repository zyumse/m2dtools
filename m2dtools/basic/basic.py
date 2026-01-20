"""Basic analysis utilities for molecular dynamics trajectories.

This module gathers small helper routines for coordination numbers, angle
distributions, compressibility, and force autocorrelations.
"""
import numpy as np
from numba import njit
from scipy.spatial import cKDTree

def CN(box, coors, cutoff):
    """Compute coordination numbers within a cutoff.

    Parameters
    ----------
    box : np.ndarray
        Simulation cell vectors as a 3x3 matrix.
    coors : np.ndarray
        Atomic coordinates with shape ``(n_atoms, 3)``.
    cutoff : float
        Distance cutoff for neighbor identification.

    Returns
    -------
    tuple
        A tuple ``(CN, CN_idx, CN_dist, diff)`` containing coordination
        counts, neighbor indices, neighbor distances, and displacement
        vectors under periodic boundary conditions.
    """

    rcoors = np.dot(coors, np.linalg.inv(box))

    r1 = rcoors[:, np.newaxis, :]
    r2 = rcoors[np.newaxis, :, :]

    rdis = r1-r2

    while np.sum((rdis < -0.5) | (rdis > 0.5)) > 0:
        rdis[rdis < -0.5] = rdis[rdis < -0.5]+1
        rdis[rdis > 0.5] = rdis[rdis > 0.5]-1

    diff = np.dot(rdis, box)

    dis = np.sqrt(np.sum(np.square(diff), axis=2))

    CN_idx = []
    CN_dist = []
    CN = np.zeros(coors.shape[0])
    for i in range(coors.shape[0]):
        tmp = np.argwhere((dis[i, :] < cutoff) & (dis[i, :] > 0))
        CN[i] = tmp.shape[0]
        CN_idx.append(tmp)
        CN_dist.append(dis[i, (dis[i, :] < cutoff) & (dis[i, :] > 0)])
    return CN, CN_idx, CN_dist, diff


def CN_large(box, coors, cutoff):
    """Compute coordination numbers for large systems.

    Parameters
    ----------
    box : np.ndarray
        Simulation cell vectors as a 3x3 matrix.
    coors : np.ndarray
        Atomic coordinates with shape ``(n_atoms, 3)``.
    cutoff : float
        Distance cutoff for neighbor identification.

    Returns
    -------
    tuple
        A tuple ``(CN, CN_idx, CN_dist)`` with coordination counts,
        neighbor indices, and neighbor distances for each atom.
        Note that CN does include the atom itself.
    """
    CN = []
    CN_idx = []
    CN_dist = []
    for i in range(coors.shape[0]):
        # find atom in the cubic at the center of coord[1], within the cutoff
        coord0 = coors[i]
        diff_coord = coors - coord0
        # periodic boundary condition
        diff_coord = diff_coord - np.round(diff_coord @ np.linalg.inv(box) ) @ box
        idx_interest = np.argwhere((diff_coord[:, 0] >= -cutoff)*(diff_coord[:, 0] <= cutoff)*(diff_coord[:, 1] >= -cutoff)*(diff_coord[:, 1] <= cutoff)*(diff_coord[:, 2] >= -cutoff)*(diff_coord[:, 2] <= cutoff)).flatten()
        dist_tmp = np.linalg.norm(diff_coord[idx_interest,:], axis=1)
        idx_CN_tmp = np.argwhere(dist_tmp<=cutoff).flatten()
        CN.append(idx_CN_tmp.shape[0])
        CN_idx.append(idx_interest[idx_CN_tmp])
        CN_dist.append(dist_tmp[idx_CN_tmp])
    return CN, CN_idx, CN_dist


def CN_kdtree(box, coors, cutoff):
    """
    Coordination number using KD-tree with orthorhombic PBC.

    Parameters
    ----------
    box : (3,3) ndarray
        Orthorhombic simulation cell (diagonal matrix).
    coors : (N,3) ndarray
        Atomic coordinates.
    cutoff : float
        Distance cutoff.

    Returns
    -------
    CN : (N,) ndarray
        Coordination number (excluding self).
    CN_idx : list of ndarray
        Neighbor indices for each atom.
    CN_dist : list of ndarray
        Neighbor distances for each atom.
    """

    box = np.asarray(box, dtype=float)
    coors = np.asarray(coors, dtype=float)

    # --- orthorhombic check ---
    if not np.allclose(box, np.diag(np.diag(box))):
        raise ValueError("CN_kdtree_ortho requires an orthorhombic box")

    box_lengths = np.diag(box)
    N = coors.shape[0]

    # --- KD-tree with periodicity ---
    tree = cKDTree(coors, boxsize=box_lengths)

    CN = np.zeros(N, dtype=int)
    CN_idx = []
    CN_dist = []

    # --- neighbor search ---
    for i in range(N):
        js = tree.query_ball_point(coors[i], cutoff)

        idx_i = []
        dist_i = []

        for j in js:
            if j == i:
                continue

            # minimum-image displacement (orthorhombic)
            d = coors[j] - coors[i]
            d -= box_lengths * np.rint(d / box_lengths)

            r = np.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
            if r < cutoff:
                idx_i.append(j)
                dist_i.append(r)

        CN[i] = len(idx_i)
        CN_idx.append(np.asarray(idx_i, dtype=int))
        CN_dist.append(np.asarray(dist_i, dtype=float))

    return CN, CN_idx, CN_dist


def calc_compressibility(V, T=300):
    """Calculate the isothermal compressibility from volume fluctuations.

    Parameters
    ----------
    V : np.ndarray
        Array of volume samples in nm^3.
    T : float, optional
        Temperature in Kelvin. Default is 300 K.

    Returns
    -------
    float
        Isothermal compressibility in ``1/Pa``.
    """
    kB = 1.380649e-23  # J/K
    V = V * 1e-27  # Convert from nm^3 to m^3
    kappa_T = np.var(V) / (V.mean() * kB * T)
    return kappa_T * 1e9 # to 1/GPa


def compute_autocorrelation(forces, max_lag):
    """Compute the force autocorrelation function (FAF).

    Parameters
    ----------
    forces : np.ndarray
        Array of shape ``(n_frames, n_atoms, 3)`` containing forces per
        atom.
    max_lag : int
        Maximum lag time (in frames) for computing the autocorrelation.

    Returns
    -------
    np.ndarray
        Force autocorrelation values with length ``max_lag``.
    """
    n_frames, n_atoms, _ = forces.shape

    # Initialize autocorrelation array
    faf = np.zeros(max_lag)

    # Loop over lag times
    for lag in range(max_lag):
        # Compute dot product of forces separated by lag
        dot_products = np.sum(forces[:n_frames - lag] * forces[lag:], axis=(1, 2))
        faf[lag] = np.mean(dot_products)

    # Normalize by the zero-lag correlation
    faf /= faf[0]

    return faf



def build_cell_list(box, coors, cutoff):
    box = np.asarray(box, dtype=np.float64)
    coors = np.asarray(coors, dtype=np.float64)

    inv_box = np.linalg.inv(box)

    # fractional coords in [0,1)
    frac = coors @ inv_box
    frac -= np.floor(frac)

    # number of cells along each direction
    box_len = np.linalg.norm(box, axis=1)
    ncell = np.maximum((box_len / cutoff).astype(np.int64), 1)

    # cell index per atom
    cell_id = np.floor(frac * ncell).astype(np.int64) % ncell

    # build linked list
    head = -np.ones(np.prod(ncell), dtype=np.int64)
    next_atom = -np.ones(coors.shape[0], dtype=np.int64)

    def flatten(cid):
        return (cid[0] * ncell[1] + cid[1]) * ncell[2] + cid[2]

    for i, cid in enumerate(cell_id):
        f = flatten(cid)
        next_atom[i] = head[f]
        head[f] = i

    return inv_box, cell_id, head, next_atom, ncell


@njit
def CN_count(inv_box, box, coors, cell_id, head, next_atom, ncell, cutoff):
    N = coors.shape[0]
    CN = np.zeros(N, dtype=np.int64)
    rc2 = cutoff * cutoff

    for i in range(N):
        ci = cell_id[i]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cj0 = (ci[0] + dx) % ncell[0]
                    cj1 = (ci[1] + dy) % ncell[1]
                    cj2 = (ci[2] + dz) % ncell[2]
                    cflat = (cj0*ncell[1] + cj1)*ncell[2] + cj2

                    j = head[cflat]
                    while j != -1:
                        if j != i:
                            diff = coors[j] - coors[i]
                            frac = diff @ inv_box
                            frac -= np.round(frac)
                            diff = frac @ box
                            if diff[0]**2 + diff[1]**2 + diff[2]**2 < rc2:
                                CN[i] += 1
                        j = next_atom[j]
    return CN

@njit
def CN_store(inv_box, box, coors, cell_id, head, next_atom, ncell, cutoff, CN):
    N = coors.shape[0]
    maxCN = np.max(CN)
    rc2 = cutoff * cutoff

    CN_idx = -np.ones((N, maxCN), dtype=np.int64)
    CN_dist = np.zeros((N, maxCN), dtype=np.float64)
    counter = np.zeros(N, dtype=np.int64)

    for i in range(N):
        ci = cell_id[i]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cj0 = (ci[0] + dx) % ncell[0]
                    cj1 = (ci[1] + dy) % ncell[1]
                    cj2 = (ci[2] + dz) % ncell[2]
                    cflat = (cj0*ncell[1] + cj1)*ncell[2] + cj2

                    j = head[cflat]
                    while j != -1:
                        if j != i:
                            diff = coors[j] - coors[i]
                            frac = diff @ inv_box
                            frac -= np.round(frac)
                            diff = frac @ box
                            d2 = diff[0]**2 + diff[1]**2 + diff[2]**2
                            if d2 < rc2:
                                k = counter[i]
                                CN_idx[i, k] = j
                                CN_dist[i, k] = np.sqrt(d2)
                                counter[i] += 1
                        j = next_atom[j]

    return CN_idx, CN_dist

def CN_fast_full(box, coors, cutoff):
    inv_box, cell_id, head, next_atom, ncell = build_cell_list(box, coors, cutoff)
    CN = CN_count(inv_box, box, coors, cell_id, head, next_atom, ncell, cutoff)
    CN_idx, CN_dist = CN_store(inv_box, box, coors, cell_id, head, next_atom, ncell, cutoff, CN)
    return CN, CN_idx, CN_dist