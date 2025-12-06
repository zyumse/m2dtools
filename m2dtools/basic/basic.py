"""Basic analysis utilities for molecular dynamics trajectories.

This module gathers small helper routines for coordination numbers, angle
distributions, compressibility, and force autocorrelations.
"""
import numpy as np

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
