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
    G = E / (2 * (1 + nu))
    K = np.zeros((12, 12))
    K[0, 0] = E * A / L
    K[0, 6] = -E * A / L
    K[1, 1] = 12 * E * Iz / L ** 3
    K[1, 5] = 6 * E * Iz / L ** 2
    K[1, 7] = -12 * E * Iz / L ** 3
    K[1, 11] = 6 * E * Iz / L ** 2
    K[2, 2] = 12 * E * Iy / L ** 3
    K[2, 4] = -6 * E * Iy / L ** 2
    K[2, 8] = -12 * E * Iy / L ** 3
    K[2, 10] = -6 * E * Iy / L ** 2
    K[3, 3] = G * J / L
    K[3, 9] = -G * J / L
    K[4, 2] = -6 * E * Iy / L ** 2
    K[4, 4] = 4 * E * Iy / L
    K[4, 8] = 6 * E * Iy / L ** 2
    K[4, 10] = 2 * E * Iy / L
    K[5, 1] = 6 * E * Iz / L ** 2
    K[5, 5] = 4 * E * Iz / L
    K[5, 7] = -6 * E * Iz / L ** 2
    K[5, 11] = 2 * E * Iz / L
    K[6, 0] = -E * A / L
    K[6, 6] = E * A / L
    K[7, 1] = -12 * E * Iz / L ** 3
    K[7, 5] = -6 * E * Iz / L ** 2
    K[7, 7] = 12 * E * Iz / L ** 3
    K[7, 11] = -6 * E * Iz / L ** 2
    K[8, 2] = -12 * E * Iy / L ** 3
    K[8, 4] = 6 * E * Iy / L ** 2
    K[8, 8] = 12 * E * Iy / L ** 3
    K[8, 10] = 6 * E * Iy / L ** 2
    K[9, 3] = -G * J / L
    K[9, 9] = G * J / L
    K[10, 2] = -6 * E * Iy / L ** 2
    K[10, 4] = 2 * E * Iy / L
    K[10, 8] = 6 * E * Iy / L ** 2
    K[10, 10] = 4 * E * Iy / L
    K[11, 1] = 6 * E * Iz / L ** 2
    K[11, 5] = 2 * E * Iz / L
    K[11, 7] = -6 * E * Iz / L ** 2
    K[11, 11] = 4 * E * Iz / L
    return K