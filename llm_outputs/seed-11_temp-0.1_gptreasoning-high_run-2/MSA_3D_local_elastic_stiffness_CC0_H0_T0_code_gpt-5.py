def MSA_3D_local_elastic_stiffness_CC0_H0_T0(E: float, nu: float, A: float, L: float, Iy: float, Iz: float, J: float) -> np.ndarray:
    """
    Return the 12×12 local elastic stiffness matrix for a 3D Euler-Bernoulli beam element.
    The beam is assumed to be aligned with the local x-axis. The stiffness matrix
    relates local nodal displacements and rotations to forces and moments using the equation:
        [force_vector] = [stiffness_matrix] @ [displacement_vector]
    Degrees of freedom are ordered as:
        [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]
    Where:
    Parameters:
        E (float): Young's modulus
        nu (float): Poisson's ratio (used for torsion only)
        A (float): Cross-sectional area
        L (float): Length of the beam element
        Iy (float): Second moment of area about the local y-axis
        Iz (float): Second moment of area about the local z-axis
        J (float): Torsional constant
    Returns:
        np.ndarray: A 12×12 symmetric stiffness matrix representing axial, torsional,
                    and bending stiffness in local coordinates.
    """
    import numpy as np
    K = np.zeros((12, 12), dtype=float)
    G = E / (2.0 * (1.0 + nu))
    k_ax = E * A / L
    idx_ax = [0, 6]
    K[np.ix_(idx_ax, idx_ax)] += k_ax * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    k_t = G * J / L
    idx_t = [3, 9]
    K[np.ix_(idx_t, idx_t)] += k_t * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    c1 = 12.0 * E * Iz / L ** 3
    c2 = 6.0 * E * Iz / L ** 2
    c3 = 4.0 * E * Iz / L
    c4 = 2.0 * E * Iz / L
    kz = np.array([[c1, c2, -c1, c2], [c2, c3, -c2, c4], [-c1, -c2, c1, -c2], [c2, c4, -c2, c3]], dtype=float)
    idx_z = [1, 5, 7, 11]
    K[np.ix_(idx_z, idx_z)] += kz
    b1 = 12.0 * E * Iy / L ** 3
    b2 = 6.0 * E * Iy / L ** 2
    b3 = 4.0 * E * Iy / L
    b4 = 2.0 * E * Iy / L
    ky = np.array([[b1, b2, -b1, b2], [b2, b3, -b2, b4], [-b1, -b2, b1, -b2], [b2, b4, -b2, b3]], dtype=float)
    idx_y = [2, 4, 8, 10]
    K[np.ix_(idx_y, idx_y)] += ky
    return K