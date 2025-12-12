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
    L2 = L * L
    L3 = L * L * L
    k_axial = E * A / L
    G = E / (2 * (1 + nu))
    k_torsion = G * J / L
    ky1 = 12 * E * Iy / L3
    ky2 = 6 * E * Iy / L2
    ky3 = 4 * E * Iy / L
    ky4 = 2 * E * Iy / L
    kz1 = 12 * E * Iz / L3
    kz2 = 6 * E * Iz / L2
    kz3 = 4 * E * Iz / L
    kz4 = 2 * E * Iz / L
    K = np.array([[k_axial, 0, 0, 0, 0, 0, -k_axial, 0, 0, 0, 0, 0], [0, kz1, 0, 0, 0, kz2, 0, -kz1, 0, 0, 0, kz2], [0, 0, ky1, 0, -ky2, 0, 0, 0, -ky1, 0, -ky2, 0], [0, 0, 0, k_torsion, 0, 0, 0, 0, 0, -k_torsion, 0, 0], [0, 0, -ky2, 0, ky3, 0, 0, 0, ky2, 0, ky4, 0], [0, kz2, 0, 0, 0, kz3, 0, -kz2, 0, 0, 0, kz4], [-k_axial, 0, 0, 0, 0, 0, k_axial, 0, 0, 0, 0, 0], [0, -kz1, 0, 0, 0, -kz2, 0, kz1, 0, 0, 0, -kz2], [0, 0, -ky1, 0, ky2, 0, 0, 0, ky1, 0, ky2, 0], [0, 0, 0, -k_torsion, 0, 0, 0, 0, 0, k_torsion, 0, 0], [0, 0, -ky2, 0, ky4, 0, 0, 0, ky2, 0, ky3, 0], [0, kz2, 0, 0, 0, kz4, 0, -kz2, 0, 0, 0, kz3]], dtype=float)
    return K