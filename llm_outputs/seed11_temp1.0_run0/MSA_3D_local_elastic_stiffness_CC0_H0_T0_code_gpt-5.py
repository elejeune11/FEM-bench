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
    k[0, 0] = EA_L
    k[0, 6] = -EA_L
    k[6, 0] = -EA_L
    k[6, 6] = EA_L
    G = E / (2.0 * (1.0 + nu))
    GJ_L = G * J / L
    k[3, 3] = GJ_L
    k[3, 9] = -GJ_L
    k[9, 3] = -GJ_L
    k[9, 9] = GJ_L
    EIz = E * Iz
    c = EIz / L ** 3
    kbz = np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L ** 2, -6.0 * L, 2.0 * L ** 2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L ** 2, -6.0 * L, 4.0 * L ** 2]], dtype=float) * c
    map_bz = [1, 5, 7, 11]
    for i in range(4):
        for j in range(4):
            k[map_bz[i], map_bz[j]] = kbz[i, j]
    EIy = E * Iy
    c = EIy / L ** 3
    kby = np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L ** 2, -6.0 * L, 2.0 * L ** 2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L ** 2, -6.0 * L, 4.0 * L ** 2]], dtype=float) * c
    map_by = [2, 4, 8, 10]
    for i in range(4):
        for j in range(4):
            k[map_by[i], map_by[j]] = kby[i, j]
    return k