"""Structural analysis helpers for molecular simulations.

The functions in this module compute bond lengths, angles, pair distribution
functions (PDF), and structure factors (SQ) for one or multiple particle
species.
"""

import numpy as np
import time
from scipy.spatial import cKDTree


def compute_bond_length(coors, bonded_atoms, box_size):
    vector = coors[bonded_atoms[:, 0]]-coors[bonded_atoms[:, 1]]
    vector = vector - np.rint(vector/box_size)*box_size
    bond_length = np.linalg.norm(vector, axis=1)
    return bond_length


def compute_angle(coors, angle_atoms, box_size):
    vector1 = coors[angle_atoms[:, 0]]-coors[angle_atoms[:, 1]]
    vector2 = coors[angle_atoms[:, 2]]-coors[angle_atoms[:, 1]]
    vector1 = vector1 - np.rint(vector1/box_size)*box_size
    vector2 = vector2 - np.rint(vector2/box_size)*box_size
    angle = np.arccos(np.sum(vector1*vector2, axis=1) /
                      (np.linalg.norm(vector1, axis=1)*np.linalg.norm(vector2, axis=1)))
    return angle/np.pi*180

def pdf_sq_1type(box_size, natom, coors, r_cutoff=10, delta_r=0.01):
    """Calculate PDF and SQ for a single particle type.

    Parameters
    ----------
    box_size : float
        Length of the cubic simulation box (Å).
    natom : int
        Number of atoms of the single particle type.
    coors : np.ndarray
        Atomic coordinates with shape ``(n_atoms, 3)``.
    r_cutoff : float, optional
        Maximum pair distance to consider. Default is ``10``.
    delta_r : float, optional
        Bin width for the radial histogram. Default is ``0.01``.

    Returns
    -------
    tuple
        Tuple ``(R, g1, Q, S1)`` where ``R`` are radial bin centers,
        ``g1`` is the pair distribution, ``Q`` is the scattering vector,
        and ``S1`` is the structure factor.
    """
    box = np.array([[box_size, 0, 0],
                    [0, box_size, 0],
                    [0, 0, box_size]])
    n1 = natom
    rcoors = np.dot(coors, np.linalg.inv(box))
    rdis = np.zeros([natom, natom, 3])
    for i in range(natom):
        tmp = rcoors[i]
        rdis[i, :, :] = tmp - rcoors
    rdis[rdis < -0.5] = rdis[rdis < -0.5] + 1
    rdis[rdis > 0.5] = rdis[rdis > 0.5] - 1
    a = np.dot(rdis[:, :, :], box)
    dis = np.sqrt((np.square(a[:, :, 0]) + np.square(a[:, :, 1]) + np.square(a[:, :, 2])))
    r_max = r_cutoff
    r = np.linspace(delta_r, r_max, int(r_max / delta_r))
    V = np.dot(np.cross(box[1, :], box[2, :]), box[0, :])
    rho1 = n1 / V
    c = np.array([rho1 * rho1]) * V
    g1 = np.histogram(dis[:n1, :n1], bins=r)[0] / (4 * np.pi *
                                                   (r[1:] - delta_r / 2) ** 2 * delta_r * c[0])
    R = r[1:] - delta_r / 2

    dq = 0.01
    qrange = [np.pi / 2 / r_max, 25]
    Q = np.arange(np.floor(qrange[0] / dq), np.floor(qrange[1] / dq), 1) * dq
    S1 = np.zeros([len(Q)])
    rho = natom / np.dot(np.cross(box[1, :], box[2, :]), box[0, :]) #/ 10 ** 3
    # use a window function for fourier transform
    for i in np.arange(len(Q)):
        S1[i] = 1 + 4 * np.pi * rho / Q[i] * np.trapz(
            (g1 - 1) * np.sin(Q[i] * R) * R * np.sin(np.pi * R / r_max) / (np.pi * R / r_max), R)

    return R, g1, Q, S1


def pdf_sq_cross_mask(box, coors1, coors2,  mask_matrix, r_cutoff:float=10, delta_r:float=0.01):
    """Calculate PDF and SQ between two particle sets with masking.

    Parameters
    ----------
    box : np.ndarray
        Simulation cell vectors as a 3x3 matrix.
    coors1 : np.ndarray
        Coordinates of the first particle set with shape ``(n1, 3)``.
    coors2 : np.ndarray
        Coordinates of the second particle set with shape ``(n2, 3)``.
    mask_matrix : np.ndarray
        Matrix masking pair contributions; masked entries are ignored.
    r_cutoff : float, optional
        Maximum pair distance to consider. Default is ``10``.
    delta_r : float, optional
        Bin width for the radial histogram. Default is ``0.01``.

    Returns
    -------
    tuple
        Tuple ``(R, g1, Q, S1)`` where ``R`` are radial bin centers,
        ``g1`` is the pair distribution, ``Q`` is the scattering vector,
        and ``S1`` is the structure factor.
    """
    n1 = len(coors1)
    n2 = len(coors2)
    natom = n1 + n2
    rcoors1 = np.dot(coors1, np.linalg.inv(box))
    rcoors2 = np.dot(coors2, np.linalg.inv(box))
    rdis = np.zeros([n1, n2, 3])
    for i in range(n1):
        tmp = rcoors1[i]
        rdis[i, :, :] = tmp - rcoors2
    rdis[rdis < -0.5] = rdis[rdis < -0.5] + 1
    rdis[rdis > 0.5] = rdis[rdis > 0.5] - 1
    a = np.dot(rdis[:, :, :], box)
    dis = np.sqrt((np.square(a[:, :, 0]) + np.square(a[:, :, 1]) + np.square(a[:, :, 2])))

    dis = dis * mask_matrix

    r_max = r_cutoff
    r = np.linspace(delta_r, r_max, int(r_max / delta_r))
    V = np.dot(np.cross(box[1, :], box[2, :]), box[0, :])
    rho1 = n1/V
    rho2 = n2/V
    c = np.array([rho1 * rho2]) * V
    g1 = np.histogram(dis, bins=r)[0] / (4 * np.pi * (r[1:] - delta_r / 2) ** 2 * delta_r * c[0])
    R = r[1:] - delta_r / 2

    dq = 0.01
    qrange = [np.pi / 2 / r_max, 25]
    Q = np.arange(np.floor(qrange[0] / dq), np.floor(qrange[1] / dq), 1) * dq
    S1 = np.zeros([len(Q)])
    rho = natom / np.dot(np.cross(box[1, :], box[2, :]), box[0, :]) #/ 10 ** 3
    # use a window function for fourier transform
    for i in np.arange(len(Q)):
        S1[i] = 1 + 4 * np.pi * rho / Q[i] * np.trapz(
            (g1 - 1) * np.sin(Q[i] * R) * R * np.sin(np.pi * R / r_max) / (np.pi * R / r_max), R)

    return R, g1, Q, S1


def pdf_sq_cross_mask_large(
    box,
    coors1,
    coors2,
    mask_matrix,
    r_cutoff: float = 10.0,
    delta_r: float = 0.01,
):
    """
    Compute the masked cross radial distribution function (pair distribution function) g(r)
    between two coordinate sets using a periodic KD-tree neighbor search, optimized for large systems.

    Parameters
    ----------
    box : array_like, shape (3, 3)
        Simulation cell vectors. Assumed orthorhombic in practice; periodic wrapping is
        applied using the per-axis box lengths derived from the row norms.

    coors1 : array_like, shape (n1, 3)
        Cartesian coordinates of species/set 1.

    coors2 : array_like, shape (n2, 3)
        Cartesian coordinates of species/set 2 (used to build the KD-tree).

    mask_matrix : array_like, shape (n1, n2)
        Boolean (or 0/1) mask selecting which (i, j) pairs are included in the RDF.
        Only distances for which `mask_matrix[i, j]` is truthy contribute to the histogram.

    r_cutoff : float, default=10.0
        Cutoff radius (in the same units as coordinates) for neighbor search and RDF.

    delta_r : float, default=0.01
        Bin width for the radial histogram.

    Returns
    -------
    r_centers : numpy.ndarray
        Radii at the centers of the histogram bins.
    g_r : numpy.ndarray
        Estimated cross RDF g(r) for the masked pairs.
    Q : numpy.ndarray
        Scattering vector magnitudes for the computed structure factor S(Q).
    S1 : numpy.ndarray
        Structure factor S(Q) computed from g(r).
    """

    t0 = time.time()
    n1, n2 = len(coors1), len(coors2)

    # box lengths (orthorhombic)
    box_lengths = np.linalg.norm(box, axis=1)

    # build tree
    tree = cKDTree(
        coors2,
        boxsize=box_lengths 
    )

    # neighbor search
    pairs = tree.query_ball_point(coors1, r_cutoff)

    # collect distances
    distances = []
    for i, js in enumerate(pairs):
        if not js:
            continue
        for j in js:
            if mask_matrix[i, j]:
                d = coors1[i] - coors2[j]
                d -= box_lengths * np.rint(d / box_lengths)
                distances.append(np.linalg.norm(d))

    distances = np.asarray(distances)

    # --- PDF ---
    r_edges = np.arange(delta_r, r_cutoff + delta_r, delta_r)
    r_centers = 0.5 * (r_edges[1:] + r_edges[:-1])

    V = np.dot(np.cross(box[1], box[2]), box[0])
    rho1, rho2 = n1 / V, n2 / V
    norm = rho1 * rho2 * V
    hist, _ = np.histogram(distances, bins=r_edges)
    g1 = hist / (4 * np.pi * r_centers**2 * delta_r * norm)

    dq = 0.01
    qrange = [np.pi / 2 / r_cutoff, 25]
    Q = np.arange(np.floor(qrange[0] / dq), np.floor(qrange[1] / dq), 1) * dq
    S1 = np.zeros([len(Q)])
    natom = n1 + n2
    rho = natom / np.dot(np.cross(box[1, :], box[2, :]), box[0, :]) #/ 10 ** 3
    # use a window function for fourier transform
    R = r_centers
    r_max = r_cutoff
    for i in np.arange(len(Q)):
        S1[i] = 1 + 4 * np.pi * rho / Q[i] * np.trapz(
            (g1 - 1) * np.sin(Q[i] * R) * R * np.sin(np.pi * R / r_max) / (np.pi * R / r_max), R)

    print(f"KD-tree PDF time: {time.time() - t0:.2f} s")
    return r_centers, g1, Q, S1

# def pdf_sq_cross(box, coors1, coors2,  bond_atom_idx, r_cutoff:float=10, delta_r:float=0.01):
#     """Calculate PDF and SQ between two particle sets.

#     Parameters
#     ----------
#     box : np.ndarray
#         Simulation cell vectors as a 3x3 matrix.
#     coors1 : np.ndarray
#         Coordinates of the first particle set with shape ``(n1, 3)``.
#     coors2 : np.ndarray
#         Coordinates of the second particle set with shape ``(n2, 3)``.
#     bond_atom_idx : array-like or None
#         Pairs of indices identifying bonded atoms to exclude from the
#         distribution. ``None`` disables exclusion.
#     r_cutoff : float, optional
#         Maximum pair distance to consider. Default is ``10``.
#     delta_r : float, optional
#         Bin width for the radial histogram. Default is ``0.01``.

#     Returns
#     -------
#     tuple
#         Tuple ``(R, g1, Q, S1)`` where ``R`` are radial bin centers,
#         ``g1`` is the pair distribution, ``Q`` is the scattering vector,
#         and ``S1`` is the structure factor.
#     """
#     # check if coors1 and coors2 are exactly the same
#     if np.array_equal(coors1, coors2):
#         is_same = True
#     else:
#         is_same = False

#     # type_atom = np.array(type_atom)
#     n1 = len(coors1)
#     n2 = len(coors2)
#     natom = n1 + n2
#     rcoors1 = np.dot(coors1, np.linalg.inv(box))
#     rcoors2 = np.dot(coors2, np.linalg.inv(box))
#     rdis = np.zeros([n1, n2, 3])
#     for i in range(n1):
#         tmp = rcoors1[i]
#         rdis[i, :, :] = tmp - rcoors2
#     rdis[rdis < -0.5] = rdis[rdis < -0.5] + 1
#     rdis[rdis > 0.5] = rdis[rdis > 0.5] - 1
#     a = np.dot(rdis[:, :, :], box)
#     dis = np.sqrt((np.square(a[:, :, 0]) + np.square(a[:, :, 1]) + np.square(a[:, :, 2])))
#     # Exclude bonded atoms by setting the distances to NaN
#     if bond_atom_idx is not None:
#         for bond_pair in bond_atom_idx:
#             i1 = int(bond_pair[0])
#             i2 = int(bond_pair[1])
#             dis[i1, i2] = np.nan
#             if is_same:
#                 dis[i2, i1] = np.nan

#     r_max = r_cutoff
#     r = np.linspace(delta_r, r_max, int(r_max / delta_r))
#     V = np.dot(np.cross(box[1, :], box[2, :]), box[0, :])
#     rho1 = n1/V
#     rho2 = n2/V
#     c = np.array([rho1 * rho2]) * V
#     g1 = np.histogram(dis, bins=r)[0] / (4 * np.pi * (r[1:] - delta_r / 2) ** 2 * delta_r * c[0])
#     R = r[1:] - delta_r / 2

#     dq = 0.01
#     qrange = [np.pi / 2 / r_max, 25]
#     Q = np.arange(np.floor(qrange[0] / dq), np.floor(qrange[1] / dq), 1) * dq
#     S1 = np.zeros([len(Q)])
#     rho = natom / np.dot(np.cross(box[1, :], box[2, :]), box[0, :]) #/ 10 ** 3
#     # use a window function for fourier transform
#     for i in np.arange(len(Q)):
#         S1[i] = 1 + 4 * np.pi * rho / Q[i] * np.trapz(
#             (g1 - 1) * np.sin(Q[i] * R) * R * np.sin(np.pi * R / r_max) / (np.pi * R / r_max), R)

#     return R, g1, Q, S1


def calculate_angle(v1, v2):
    """Calculate the angle between two vectors.

    Parameters
    ----------
    v1 : np.ndarray
        First vector.
    v2 : np.ndarray
        Second vector.

    Returns
    -------
    float
        Angle in degrees between ``v1`` and ``v2``.
    """
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # convert to angle degree
    angle = np.arccos(cos_angle)/np.pi*180
    return angle


def angle_distribution(coors, box, cutoff):
    """Compute the O–O–O angle distribution within a cutoff.

    Parameters
    ----------
    coors : np.ndarray
        Atomic coordinates with shape ``(n_atoms, 3)``.
    box : np.ndarray
        Simulation cell vectors as a 3x3 matrix.
    cutoff : float
        Maximum O–O separation to include in the triplet selection.

    Returns
    -------
    list[float]
        Angles in degrees for all qualifying triplets.
    """
    n_atom = coors.shape[0]
    angles = []
    rcoors = np.dot(coors, np.linalg.inv(box))
    rdis = np.zeros([n_atom, n_atom, 3])
    for i in range(n_atom):
        tmp = rcoors[i]
        rdis[i, :, :] = tmp - rcoors
    rdis[rdis < -0.5] = rdis[rdis < -0.5] + 1
    rdis[rdis > 0.5] = rdis[rdis > 0.5] - 1
    a = np.dot(rdis[:, :, :], box)
    dis = np.sqrt((np.square(a[:, :, 0]) + np.square(a[:, :, 1]) + np.square(a[:, :, 2])))

    for i in range(n_atom):
        for j in np.arange(i+1, n_atom):
            for k in np.arange(j+1, n_atom):
                if dis[i, j] < cutoff and dis[i, k] < cutoff and dis[j, k] < cutoff:
                    angle = calculate_angle(a[j, i, :], a[k, i, :])
                    angles.append(angle)
                    angle = calculate_angle(a[i, j, :], a[k, j, :])
                    angles.append(angle)
                    angle = calculate_angle(a[i, k, :], a[j, k, :])
                    angles.append(angle)
    return angles
