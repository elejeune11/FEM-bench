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
    import numpy as np
    K = np.zeros((12, 12), dtype=float)
    EA_L = E * A / L
    G = E / (2.0 * (1.0 + nu))
    GJ_L = G * J / L
    c1z = 12.0 * E * Iz / L ** 3
    c2z = 6.0 * E * Iz / L ** 2
    c3z = 4.0 * E * Iz / L
    c4z = 2.0 * E * Iz / L
    c1y = 12.0 * E * Iy / L ** 3
    c2y = 6.0 * E * Iy / L ** 2
    c3y = 4.0 * E * Iy / L
    c4y = 2.0 * E * Iy / L
    K[0, 0] = EA_L
    K[0, 6] = -EA_L
    K[6, 0] = -EA_L
    K[6, 6] = EA_L
    K[3, 3] = GJ_L
    K[3, 9] = -GJ_L
    K[9, 3] = -GJ_L
    K[9, 9] = GJ_L
    K[1, 1] = c1z
    K[1, 5] = c2z
    K[1, 7] = -c1z
    K[1, 11] = c2z
    K[5, 1] = c2z
    K[5, 5] = c3z
    K[5, 7] = -c2z
    K[5, 11] = c4z
    K[7, 1] = -c1z
    K[7, 5] = -c2z
    K[7, 7] = c1z
    K[7, 11] = -c2z
    K[11, 1] = c2z
    K[11, 5] = c4z
    K[11, 7] = -c2z
    K[11, 11] = c3z
    K[2, 2] = c1y
    K[2, 4] = -c2y
    K[2, 8] = -c1y
    K[2, 10] = -c2y
    K[4, 2] = -c2y
    K[4, 4] = c3y
    K[4, 8] = c2y
    K[4, 10] = c4y
    K[8, 2] = -c1y
    K[8, 4] = c2y
    K[8, 8] = c1y
    K[8, 10] = c2y
    K[10, 2] = -c2y
    K[10, 4] = c4y
    K[10, 8] = c2y
    K[10, 10] = c3y
    return K