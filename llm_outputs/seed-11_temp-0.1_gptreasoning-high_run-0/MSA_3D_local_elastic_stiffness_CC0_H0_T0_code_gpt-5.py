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
    k = np.zeros((12, 12), dtype=float)
    if L != 0.0:
        EA_L = E * A / L
    else:
        EA_L = np.inf
    k[0, 0] += EA_L
    k[0, 6] -= EA_L
    k[6, 0] -= EA_L
    k[6, 6] += EA_L
    G = E / (2.0 * (1.0 + nu)) if 1.0 + nu != 0.0 else np.inf
    GJ_L = G * J / L if L != 0.0 else np.inf
    k[3, 3] += GJ_L
    k[3, 9] -= GJ_L
    k[9, 3] -= GJ_L
    k[9, 9] += GJ_L
    if L != 0.0:
        c_z = E * Iz / L ** 3
        Lc = L * c_z
        L2c = L ** 2 * c_z
    else:
        c_z = Lc = L2c = np.inf
    (v1, tz1, v2, tz2) = (1, 5, 7, 11)
    k[v1, v1] += 12.0 * c_z
    k[v1, tz1] += 6.0 * Lc
    k[v1, v2] += -12.0 * c_z
    k[v1, tz2] += 6.0 * Lc
    k[tz1, v1] += 6.0 * Lc
    k[tz1, tz1] += 4.0 * L2c
    k[tz1, v2] += -6.0 * Lc
    k[tz1, tz2] += 2.0 * L2c
    k[v2, v1] += -12.0 * c_z
    k[v2, tz1] += -6.0 * Lc
    k[v2, v2] += 12.0 * c_z
    k[v2, tz2] += -6.0 * Lc
    k[tz2, v1] += 6.0 * Lc
    k[tz2, tz1] += 2.0 * L2c
    k[tz2, v2] += -6.0 * Lc
    k[tz2, tz2] += 4.0 * L2c
    if L != 0.0:
        c_y = E * Iy / L ** 3
        Lc = L * c_y
        L2c = L ** 2 * c_y
    else:
        c_y = Lc = L2c = np.inf
    (w1, ty1, w2, ty2) = (2, 4, 8, 10)
    k[w1, w1] += 12.0 * c_y
    k[w1, ty1] += 6.0 * Lc
    k[w1, w2] += -12.0 * c_y
    k[w1, ty2] += 6.0 * Lc
    k[ty1, w1] += 6.0 * Lc
    k[ty1, ty1] += 4.0 * L2c
    k[ty1, w2] += -6.0 * Lc
    k[ty1, ty2] += 2.0 * L2c
    k[w2, w1] += -12.0 * c_y
    k[w2, ty1] += -6.0 * Lc
    k[w2, w2] += 12.0 * c_y
    k[w2, ty2] += -6.0 * Lc
    k[ty2, w1] += 6.0 * Lc
    k[ty2, ty1] += 2.0 * L2c
    k[ty2, w2] += -6.0 * Lc
    k[ty2, ty2] += 4.0 * L2c
    return k