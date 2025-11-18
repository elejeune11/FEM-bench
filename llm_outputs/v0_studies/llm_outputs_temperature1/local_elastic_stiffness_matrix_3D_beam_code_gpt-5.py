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
    import pytest
    if L <= 0:
        raise ValueError('Beam length L must be positive.')
    G = E / (2.0 * (1.0 + nu))
    k = np.zeros((12, 12), dtype=float)
    ka = E * A / L
    k[0, 0] = ka
    k[0, 6] = -ka
    k[6, 0] = -ka
    k[6, 6] = ka
    kt = G * J / L
    k[3, 3] = kt
    k[3, 9] = -kt
    k[9, 3] = -kt
    k[9, 9] = kt
    EIz = E * Iz
    kv = 12.0 * EIz / L ** 3
    kvt = 6.0 * EIz / L ** 2
    kzz4 = 4.0 * EIz / L
    kzz2 = 2.0 * EIz / L
    (v1, tz1, v2, tz2) = (1, 5, 7, 11)
    k[v1, v1] = kv
    k[v1, tz1] = kvt
    k[v1, v2] = -kv
    k[v1, tz2] = kvt
    k[tz1, v1] = kvt
    k[tz1, tz1] = kzz4
    k[tz1, v2] = -kvt
    k[tz1, tz2] = kzz2
    k[v2, v1] = -kv
    k[v2, tz1] = -kvt
    k[v2, v2] = kv
    k[v2, tz2] = -kvt
    k[tz2, v1] = kvt
    k[tz2, tz1] = kzz2
    k[tz2, v2] = -kvt
    k[tz2, tz2] = kzz4
    EIy = E * Iy
    kw = 12.0 * EIy / L ** 3
    kwt = 6.0 * EIy / L ** 2
    kyy4 = 4.0 * EIy / L
    kyy2 = 2.0 * EIy / L
    (w1, ty1, w2, ty2) = (2, 4, 8, 10)
    k[w1, w1] = kw
    k[w1, ty1] = -kwt
    k[w1, w2] = -kw
    k[w1, ty2] = -kwt
    k[ty1, w1] = -kwt
    k[ty1, ty1] = kyy4
    k[ty1, w2] = kwt
    k[ty1, ty2] = kyy2
    k[w2, w1] = -kw
    k[w2, ty1] = kwt
    k[w2, w2] = kw
    k[w2, ty2] = kwt
    k[ty2, w1] = -kwt
    k[ty2, ty1] = kyy2
    k[ty2, w2] = kwt
    k[ty2, ty2] = kyy4
    return k