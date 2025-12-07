"""Dynamic analysis routines for trajectory data."""

import numpy as np


def compute_msd_pbc(positions, box_lengths, lag_array):
    """Compute mean-squared displacement with periodic boundaries.

    Parameters
    ----------
    positions : np.ndarray
        Atom positions with shape ``(n_frames, n_atoms, 3)``.
    box_lengths : np.ndarray or list
        Simulation box lengths per frame with shape ``(n_frames, 3)``.
    lag_array : np.ndarray or list
        Lag times (in frames) at which to evaluate the MSD.

    Returns
    -------
    np.ndarray
        Mean-squared displacement values for each lag time.
    """
    n_frames, n_atoms, _ = positions.shape

    # Unwrap positions considering periodic boundary conditions
    unwrapped_pos = np.zeros_like(positions)
    unwrapped_pos[0] = positions[0]

    for t in range(1, n_frames):
        box_length_tmp = (box_lengths[t] + box_lengths[t - 1]) / 2
        delta = positions[t] - unwrapped_pos[t - 1]
        delta -= box_length_tmp * np.round(delta / box_length_tmp)
        unwrapped_pos[t] = unwrapped_pos[t - 1] + delta

    msd = np.zeros(len(lag_array))

    for i, lag in enumerate(lag_array):
        displacements = unwrapped_pos[lag:] - unwrapped_pos[:-lag]
        squared_displacements = np.sum(displacements**2, axis=2)
        msd[i] = np.mean(squared_displacements)

    return msd


def compute_D(time, msd, fit_from=0, dim=3):
    """Compute the diffusion coefficient from an MSD curve.

    Parameters
    ----------
    time : np.ndarray
        Time values corresponding to the MSD entries.
    msd : np.ndarray
        Mean-squared displacement values.
    fit_from : int, optional
        Index at which to start the linear regression. Default is ``0``.
    dim : int, optional
        Dimensionality of the system (e.g., ``3`` for 3D). Default is ``3``.

    Returns
    -------
    float
        Estimated diffusion coefficient in units of ``msd/time``.
    """
    coeffs = np.polyfit(time[fit_from:], msd[fit_from:], 1)
    D = coeffs[0] / 2 / dim # in unit of msd / time
    return D


def random_vectors(length, num_vectors):
    phi = np.random.uniform(0, 2*np.pi, size=num_vectors)
    costheta = np.random.uniform(-1, 1, size=num_vectors)
    theta = np.arccos(costheta)

    x = length * np.sin(theta) * np.cos(phi)
    y = length * np.sin(theta) * np.sin(phi)
    z = length * np.cos(theta)

    return np.column_stack((x, y, z))   # shape = (num_vectors, 3)



#compute self-intermediate scattering function
def compute_SISF(positions, box_lengths, lag_array, k, num_vectors=100, n_repeat=100):
    """Compute the self-intermediate scattering function (SISF).

    Parameters
    ----------
    positions : np.ndarray
        Atom positions with shape ``(n_frames, n_atoms, 3)``.
    box_lengths : np.ndarray or list
        Simulation box lengths per frame with shape ``(n_frames, 3)``.
    lag_array : np.ndarray or list
        Lag times (in frames) at which to evaluate the SISF.
    k : float
        Magnitude of the wave vector ``|k|``.
    num_vectors : int, optional
        Number of random ``k``-vectors to sample. Default is ``100``.
    n_repeat : int, optional
        Number of time origins to average over. Default is ``100``.

    Returns
    -------
    np.ndarray
        Self-intermediate scattering values for each lag time.
    """
    n_frames, n_atoms, _ = positions.shape

    # Unwrap positions considering periodic boundary conditions
    unwrapped_pos = np.zeros_like(positions)
    unwrapped_pos[0] = positions[0]

    for t in range(1, n_frames):
        box_length_tmp = (box_lengths[t] + box_lengths[t - 1]) / 2
        delta = positions[t] - unwrapped_pos[t - 1]
        delta -= box_length_tmp * np.round(delta / box_length_tmp)
        unwrapped_pos[t] = unwrapped_pos[t - 1] + delta

    sisf = np.zeros(len(lag_array))
    vectors=random_vectors(k, num_vectors)

    for i, lag in enumerate(lag_array):
        if len(positions)-lag < n_repeat:
            displacements = unwrapped_pos[lag:] - unwrapped_pos[:-lag]
            #compute cos(k*r) for each random vector and average
            cos_kr = np.cos(np.einsum('ij,tkj->tki', vectors, displacements))
        else:
            random_indices = np.random.choice(len(positions)-lag, n_repeat)
            displacements = unwrapped_pos[lag + random_indices] - unwrapped_pos[random_indices]
            cos_kr = np.cos(np.einsum('ij,tkj->tki', vectors, displacements))
        sisf[i] = np.mean(cos_kr)
    return sisf

def random_vectors_2D(k, num_vectors):
        phi = np.random.uniform(0, 2*np.pi, size=num_vectors)
        x = k * np.cos(phi)
        y = k * np.sin(phi)
        z = np.zeros(num_vectors)
        return np.column_stack((x, y, z))   # shape = (num_vectors, 3)

def compute_SISF_zresolved(positions, box_lengths, lag_array, k,
                            z_ranges,
                            num_vectors=100, n_repeat=100) -> np.ndarray:
    """
    Layer-resolved SISF with z-bin assignment performed at every time origin t0.

    Parameters
    ----------
    z_ranges : list of tuple
        List of (zmin, zmax) tuples defining the layers along z-axis.
    Other parameters are the same as in `compute_SISF`.

    Returns
    -------
    np.ndarray
        Self-intermediate scattering values for each layer and lag time.
    """

    n_frames, n_atoms, _ = positions.shape
    n_layers = len(z_ranges)

    # ---- unwrap ----
    unwrapped = np.zeros_like(positions)
    unwrapped[0] = positions[0]

    for t in range(1, n_frames):
        box_tmp = 0.5 * (box_lengths[t] + box_lengths[t-1])
        delta = positions[t] - unwrapped[t-1]
        delta -= box_tmp * np.round(delta / box_tmp)
        unwrapped[t] = unwrapped[t-1] + delta

    # ---- random k vectors ----
    vectors = random_vectors_2D(k, num_vectors)

    # ---- output ----
    sisf_layers = np.zeros((n_layers, len(lag_array)))

    # ---- compute for each lag ----
    for i, lag in enumerate(lag_array):

        max_origin = n_frames - lag
        if max_origin < n_repeat:
            origins = np.arange(max_origin)
        else:
            origins = np.random.choice(max_origin, n_repeat, replace=False)

        # accumulator for each layer
        accum = [ [] for _ in range(n_layers) ]

        for t0 in origins:
            z_now = unwrapped[t0, :, 2]   # z positions at time origin

            # assign atoms at this time origin
            atom_layers = []
            for (zmin, zmax) in z_ranges:
                atom_layers.append(np.where((z_now >= zmin) & (z_now < zmax))[0])

            # compute displacements from t0 to t0+lag
            disp = unwrapped[t0 + lag] - unwrapped[t0]   # (N,3)

            for L, atoms_L in enumerate(atom_layers):
                if len(atoms_L) == 0:
                    continue

                d = disp[atoms_L]           # (n_atoms_L, 3)

                cos_kr = np.cos(np.einsum("ij,nj->ni", vectors, d))
                accum[L].append(np.mean(cos_kr))

        # average for each layer
        for L in range(n_layers):
            if len(accum[L]) > 0:
                sisf_layers[L, i] = np.mean(accum[L])

    return sisf_layers
