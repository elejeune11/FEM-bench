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
    if L == 0:
        raise ZeroDivisionError('Element length L must be non-zero.')
    k_axial = E * A / L
    K[0, 0] = k_axial
    K[0, 6] = -k_axial
    K[6, 0] = -k_axial
    K[6, 6] = k_axial
    G = E / (2.0 * (1.0 + nu))
    k_torsion = G * J / L
    K[3, 3] = k_torsion
    K[3, 9] = -k_torsion
    K[9, 3] = -k_torsion
    K[9, 9] = k_torsion
    L2 = L * L
    L3 = L2 * L
    bending_template = np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]], dtype=float)
    kb_z = E * Iz / L3
    indices_v = [1, 5, 7, 11]
    for i in range(4):
        for j in range(4):
            K[indices_v[i], indices_v[j]] += kb_z * bending_template[i, j]
    kb_y = E * Iy / L3
    indices_w = [2, 4, 8, 10]
    for i in range(4):
        for j in range(4):
            K[indices_w[i], indices_w[j]] += kb_y * bending_template[i, j]
    return K