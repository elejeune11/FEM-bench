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
    K = np.zeros((12, 12))
    EA_L = E * A / L
    K[0, 0] = K[6, 6] = EA_L
    K[0, 6] = K[6, 0] = -EA_L
    GJ_L = E * J / (2 * (1 + nu) * L)
    K[3, 3] = K[9, 9] = GJ_L
    K[3, 9] = K[9, 3] = -GJ_L
    EIz_L = E * Iz / L
    EIz_L2 = EIz_L / L
    EIz_L3 = EIz_L2 / L
    K[1, 1] = K[7, 7] = 12 * EIz_L3
    K[5, 5] = K[11, 11] = 4 * EIz_L
    K[1, 5] = K[5, 1] = 6 * EIz_L2
    K[1, 7] = K[7, 1] = -12 * EIz_L3
    K[1, 11] = K[11, 1] = 6 * EIz_L2
    K[5, 7] = K[7, 5] = -6 * EIz_L2
    K[5, 11] = K[11, 5] = 2 * EIz_L
    K[7, 11] = K[11, 7] = -6 * EIz_L2
    EIy_L = E * Iy / L
    EIy_L2 = EIy_L / L
    EIy_L3 = EIy_L2 / L
    K[2, 2] = K[8, 8] = 12 * EIy_L3
    K[4, 4] = K[10, 10] = 4 * EIy_L
    K[2, 4] = K[4, 2] = -6 * EIy_L2
    K[2, 8] = K[8, 2] = -12 * EIy_L3
    K[2, 10] = K[10, 2] = -6 * EIy_L2
    K[4, 8] = K[8, 4] = 6 * EIy_L2
    K[4, 10] = K[10, 4] = 2 * EIy_L
    K[8, 10] = K[10, 8] = 6 * EIy_L2
    return K