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
    k = np.zeros((12, 12))
    G = E / (2 * (1 + nu))
    k_axial = E * A / L
    k[0, 0] = k_axial
    k[0, 6] = -k_axial
    k[6, 0] = -k_axial
    k[6, 6] = k_axial
    k_tor = G * J / L
    k[3, 3] = k_tor
    k[3, 9] = -k_tor
    k[9, 3] = -k_tor
    k[9, 9] = k_tor
    bz1 = 12 * E * Iz / L ** 3
    bz2 = 6 * E * Iz / L ** 2
    bz3 = 4 * E * Iz / L
    bz4 = 2 * E * Iz / L
    k[1, 1] = bz1
    k[1, 5] = bz2
    k[1, 7] = -bz1
    k[1, 11] = bz2
    k[5, 1] = bz2
    k[5, 5] = bz3
    k[5, 7] = -bz2
    k[5, 11] = bz4
    k[7, 1] = -bz1
    k[7, 5] = -bz2
    k[7, 7] = bz1
    k[7, 11] = -bz2
    k[11, 1] = bz2
    k[11, 5] = bz4
    k[11, 7] = -bz2
    k[11, 11] = bz3
    by1 = 12 * E * Iy / L ** 3
    by2 = 6 * E * Iy / L ** 2
    by3 = 4 * E * Iy / L
    by4 = 2 * E * Iy / L
    k[2, 2] = by1
    k[2, 4] = -by2
    k[2, 8] = -by1
    k[2, 10] = -by2
    k[4, 2] = -by2
    k[4, 4] = by3
    k[4, 8] = by2
    k[4, 10] = by4
    k[8, 2] = -by1
    k[8, 4] = by2
    k[8, 8] = by1
    k[8, 10] = by2
    k[10, 2] = -by2
    k[10, 4] = by4
    k[10, 8] = by2
    k[10, 10] = by3
    return k