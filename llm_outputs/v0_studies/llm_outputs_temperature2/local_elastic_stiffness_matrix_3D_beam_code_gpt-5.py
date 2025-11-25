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
    k = np.zeros((12, 12), dtype=float)
    EA_L = E * A / L
    G = E / (2.0 * (1.0 + nu))
    GJ_L = G * J / L
    b = 12.0 * E * Iz / L ** 3
    c = 6.0 * E * Iz / L ** 2
    d = 4.0 * E * Iz / L
    e = 2.0 * E * Iz / L
    f = 12.0 * E * Iy / L ** 3
    g = 6.0 * E * Iy / L ** 2
    h = 4.0 * E * Iy / L
    i = 2.0 * E * Iy / L
    k[0, 0] = EA_L
    k[0, 6] = -EA_L
    k[6, 0] = -EA_L
    k[6, 6] = EA_L
    k[3, 3] = GJ_L
    k[3, 9] = -GJ_L
    k[9, 3] = -GJ_L
    k[9, 9] = GJ_L
    k[1, 1] = b
    k[1, 5] = c
    k[1, 7] = -b
    k[1, 11] = c
    k[5, 1] = c
    k[5, 5] = d
    k[5, 7] = -c
    k[5, 11] = e
    k[7, 1] = -b
    k[7, 5] = -c
    k[7, 7] = b
    k[7, 11] = -c
    k[11, 1] = c
    k[11, 5] = e
    k[11, 7] = -c
    k[11, 11] = d
    k[2, 2] = f
    k[2, 4] = -g
    k[2, 8] = -f
    k[2, 10] = -g
    k[4, 2] = -g
    k[4, 4] = h
    k[4, 8] = g
    k[4, 10] = i
    k[8, 2] = -f
    k[8, 4] = g
    k[8, 8] = f
    k[8, 10] = g
    k[10, 2] = -g
    k[10, 4] = i
    k[10, 8] = g
    k[10, 10] = h
    return k