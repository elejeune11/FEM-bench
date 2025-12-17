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
    K[0, 0] = k_ax
    K[0, 6] = -k_ax
    K[6, 0] = -k_ax
    K[6, 6] = k_ax
    k_tor = G * J / L
    K[3, 3] = k_tor
    K[3, 9] = -k_tor
    K[9, 3] = -k_tor
    K[9, 9] = k_tor
    EIz = E * Iz
    c1 = 12.0 * EIz / L ** 3
    c2 = 6.0 * EIz / L ** 2
    c3 = 4.0 * EIz / L
    c4 = 2.0 * EIz / L
    v1, tz1, v2, tz2 = (1, 5, 7, 11)
    K[v1, v1] += c1
    K[v1, tz1] += c2
    K[v1, v2] += -c1
    K[v1, tz2] += c2
    K[tz1, v1] += c2
    K[tz1, tz1] += c3
    K[tz1, v2] += -c2
    K[tz1, tz2] += c4
    K[v2, v1] += -c1
    K[v2, tz1] += -c2
    K[v2, v2] += c1
    K[v2, tz2] += -c2
    K[tz2, v1] += c2
    K[tz2, tz1] += c4
    K[tz2, v2] += -c2
    K[tz2, tz2] += c3
    EIy = E * Iy
    d1 = 12.0 * EIy / L ** 3
    d2 = 6.0 * EIy / L ** 2
    d3 = 4.0 * EIy / L
    d4 = 2.0 * EIy / L
    w1, ty1, w2, ty2 = (2, 4, 8, 10)
    K[w1, w1] += d1
    K[w1, ty1] += -d2
    K[w1, w2] += -d1
    K[w1, ty2] += -d2
    K[ty1, w1] += -d2
    K[ty1, ty1] += d3
    K[ty1, w2] += d2
    K[ty1, ty2] += d4
    K[w2, w1] += -d1
    K[w2, ty1] += d2
    K[w2, w2] += d1
    K[w2, ty2] += d2
    K[ty2, w1] += -d2
    K[ty2, ty1] += d4
    K[ty2, w2] += d2
    K[ty2, ty2] += d3
    return K