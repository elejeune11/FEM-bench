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
    K = np.zeros((12, 12), dtype=float)
    G = E / (2.0 * (1.0 + nu))
    k_ax = E * A / L
    K[0, 0] += k_ax
    K[0, 6] += -k_ax
    K[6, 0] += -k_ax
    K[6, 6] += k_ax
    k_tx = G * J / L
    K[3, 3] += k_tx
    K[3, 9] += -k_tx
    K[9, 3] += -k_tx
    K[9, 9] += k_tx
    EIz = E * Iz
    c1z = 12.0 * EIz / L ** 3
    c2z = 6.0 * EIz / L ** 2
    c3z = 4.0 * EIz / L
    c4z = 2.0 * EIz / L
    (v1, rz1, v2, rz2) = (1, 5, 7, 11)
    K[v1, v1] += c1z
    K[v1, rz1] += c2z
    K[v1, v2] += -c1z
    K[v1, rz2] += c2z
    K[rz1, v1] += c2z
    K[rz1, rz1] += c3z
    K[rz1, v2] += -c2z
    K[rz1, rz2] += c4z
    K[v2, v1] += -c1z
    K[v2, rz1] += -c2z
    K[v2, v2] += c1z
    K[v2, rz2] += -c2z
    K[rz2, v1] += c2z
    K[rz2, rz1] += c4z
    K[rz2, v2] += -c2z
    K[rz2, rz2] += c3z
    EIy = E * Iy
    c1y = 12.0 * EIy / L ** 3
    c2y = 6.0 * EIy / L ** 2
    c3y = 4.0 * EIy / L
    c4y = 2.0 * EIy / L
    (w1, ry1, w2, ry2) = (2, 4, 8, 10)
    K[w1, w1] += c1y
    K[w1, ry1] += -c2y
    K[w1, w2] += -c1y
    K[w1, ry2] += -c2y
    K[ry1, w1] += -c2y
    K[ry1, ry1] += c3y
    K[ry1, w2] += c2y
    K[ry1, ry2] += c4y
    K[w2, w1] += -c1y
    K[w2, ry1] += c2y
    K[w2, w2] += c1y
    K[w2, ry2] += c2y
    K[ry2, w1] += -c2y
    K[ry2, ry1] += c4y
    K[ry2, w2] += c2y
    K[ry2, ry2] += c3y
    return K