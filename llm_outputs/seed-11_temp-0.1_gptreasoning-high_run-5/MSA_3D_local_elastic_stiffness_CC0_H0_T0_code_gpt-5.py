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
    EA = E * A
    G = E / (2.0 * (1.0 + nu))
    GJ = G * J
    k_ax = EA / L
    K[0, 0] = k_ax
    K[0, 6] = -k_ax
    K[6, 0] = -k_ax
    K[6, 6] = k_ax
    k_tor = GJ / L
    K[3, 3] = k_tor
    K[3, 9] = -k_tor
    K[9, 3] = -k_tor
    K[9, 9] = k_tor
    EIz = E * Iz
    a = 12.0 * EIz / L ** 3
    b = 6.0 * EIz / L ** 2
    c = 4.0 * EIz / L
    d = 2.0 * EIz / L
    K[1, 1] += a
    K[1, 5] += b
    K[1, 7] += -a
    K[1, 11] += b
    K[5, 1] = K[1, 5]
    K[5, 5] += c
    K[5, 7] += -b
    K[5, 11] += d
    K[7, 1] = K[1, 7]
    K[7, 5] = K[5, 7]
    K[7, 7] += a
    K[7, 11] += -b
    K[11, 1] = K[1, 11]
    K[11, 5] = K[5, 11]
    K[11, 7] = K[7, 11]
    K[11, 11] += c
    EIy = E * Iy
    a2 = 12.0 * EIy / L ** 3
    b2 = 6.0 * EIy / L ** 2
    c2 = 4.0 * EIy / L
    d2 = 2.0 * EIy / L
    K[2, 2] += a2
    K[2, 4] += -b2
    K[2, 8] += -a2
    K[2, 10] += -b2
    K[4, 2] = K[2, 4]
    K[4, 4] += c2
    K[4, 8] += b2
    K[4, 10] += d2
    K[8, 2] = K[2, 8]
    K[8, 4] = K[4, 8]
    K[8, 8] += a2
    K[8, 10] += b2
    K[10, 2] = K[2, 10]
    K[10, 4] = K[4, 10]
    K[10, 8] = K[8, 10]
    K[10, 10] += c2
    return K