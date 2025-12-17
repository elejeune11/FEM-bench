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
    k = np.zeros((12, 12), dtype=float)
    EA_L = E * A / L
    G = E / (2.0 * (1.0 + nu))
    GJ_L = G * J / L
    EIy = E * Iy
    EIz = E * Iz
    L2 = L * L
    L3 = L2 * L
    k[0, 0] = EA_L
    k[0, 6] = -EA_L
    k[6, 0] = -EA_L
    k[6, 6] = EA_L
    k[3, 3] = GJ_L
    k[3, 9] = -GJ_L
    k[9, 3] = -GJ_L
    k[9, 9] = GJ_L
    kz = EIz / L3 * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]], dtype=float)
    idx_z = [1, 5, 7, 11]
    for i in range(4):
        for j in range(4):
            k[idx_z[i], idx_z[j]] += kz[i, j]
    ky = EIy / L3 * np.array([[12.0, -6.0 * L, -12.0, -6.0 * L], [-6.0 * L, 4.0 * L2, 6.0 * L, 2.0 * L2], [-12.0, 6.0 * L, 12.0, 6.0 * L], [-6.0 * L, 2.0 * L2, 6.0 * L, 4.0 * L2]], dtype=float)
    idx_y = [2, 4, 8, 10]
    for i in range(4):
        for j in range(4):
            k[idx_y[i], idx_y[j]] += ky[i, j]
    return k