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
    EA_over_L = E * A / L
    EIz_over_L3 = E * Iz / L ** 3
    EIy_over_L3 = E * Iy / L ** 3
    GJ_over_L = G * J / L
    K = np.array([[EA_over_L, 0, 0, 0, 0, 0, -EA_over_L, 0, 0, 0, 0, 0], [0, 12 * EIy_over_L3, 0, 0, 0, 6 * EIy_over_L3, 0, -12 * EIy_over_L3, 0, 0, 0, 6 * EIy_over_L3], [0, 0, 12 * EIz_over_L3, 0, -6 * EIz_over_L3, 0, 0, 0, -12 * EIz_over_L3, 0, 6 * EIz_over_L3, 0], [0, 0, 0, GJ_over_L, 0, 0, 0, 0, 0, -GJ_over_L, 0, 0], [0, 0, -6 * EIz_over_L3, 0, 4 * EIz_over_L3, 0, 0, 0, 6 * EIz_over_L3, 0, 2 * EIz_over_L3, 0], [0, 6 * EIy_over_L3, 0, 0, 0, 4 * EIy_over_L3, 0, -6 * EIy_over_L3, 0, 0, 0, 2 * EIy_over_L3], [-EA_over_L, 0, 0, 0, 0, 0, EA_over_L, 0, 0, 0, 0, 0], [0, -12 * EIy_over_L3, 0, 0, 0, -6 * EIy_over_L3, 0, 12 * EIy_over_L3, 0, 0, 0, -6 * EIy_over_L3], [0, 0, -12 * EIz_over_L3, 0, 6 * EIz_over_L3, 0, 0, 0, 12 * EIz_over_L3, 0, -6 * EIz_over_L3, 0], [0, 0, 0, -GJ_over_L, 0, 0, 0, 0, 0, GJ_over_L, 0, 0], [0, 0, 6 * EIz_over_L3, 0, 2 * EIz_over_L3, 0, 0, 0, -6 * EIz_over_L3, 0, 4 * EIz_over_L3, 0], [0, 6 * EIy_over_L3, 0, 0, 0, 2 * EIy_over_L3, 0, -6 * EIy_over_L3, 0, 0, 0, 4 * EIy_over_L3]])
    G = E / (2 * (1 + nu))
    return K