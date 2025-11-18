def local_elastic_stiffness_matrix_3D_beam(E: float, nu: float, A: float, L: float, Iy: float, Iz: float, J: float) -> np.ndarray:
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
    EA_over_L = E * A / L
    EIz_over_L3 = E * Iz / L ** 3
    EIy_over_L3 = E * Iy / L ** 3
    GJ_over_L = G * J / L
    k[0, 0] = EA_over_L
    k[0, 6] = -EA_over_L
    k[6, 0] = -EA_over_L
    k[6, 6] = EA_over_L
    k[1, 1] = 12 * EIy_over_L3
    k[1, 5] = 6 * EIy_over_L2
    k[1, 7] = -12 * EIy_over_L3
    k[1, 11] = 6 * EIy_over_L2
    k[5, 1] = 6 * EIy_over_L2
    k[5, 5] = 4 * EIy_over_L
    k[5, 7] = -6 * EIy_over_L2
    k[5, 11] = 2 * EIy_over_L
    k[7, 1] = -12 * EIy_over_L3
    k[7, 5] = -6 * EIy_over_L2
    k[7, 7] = 12 * EIy_over_L3
    k[7, 11] = -6 * EIy_over_L2
    k[11, 1] = 6 * EIy_over_L2
    k[11, 5] = 2 * EIy_over_L
    k[11, 7] = -6 * EIy_over_L2
    k[11, 11] = 4 * EIy_over_L
    k[2, 2] = 12 * EIz_over_L3
    k[2, 4] = -6 * EIz_over_L2
    k[2, 8] = -12 * EIz_over_L3
    k[2, 10] = -6 * EIz_over_L2
    k[4, 2] = -6 * EIz_over_L2
    k[4, 4] = 4 * EIz_over_L
    k[4, 8] = 6 * EIz_over_L2
    k[4, 10] = 2 * EIz_over_L
    k[8, 2] = -12 * EIz_over_L3
    k[8, 4] = 6 * EIz_over_L2
    k[8, 8] = 12 * EIz_over_L3
    k[8, 10] = 6 * EIz_over_L2
    k[10, 2] = -6 * EIz_over_L2
    k[10, 4] = 2 * EIz_over_L
    k[10, 8] = 6 * EIz_over_L2
    k[10, 10] = 4 * EIz_over_L
    k[3, 3] = GJ_over_L
    k[3, 9] = -GJ_over_L
    k[9, 3] = -GJ_over_L
    k[9, 9] = GJ_over_L
    return k