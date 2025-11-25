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
    L2 = L * L
    L3 = L * L * L
    G = E / (2 * (1 + nu))
    K = np.zeros((12, 12))
    k_ax = E * A / L
    K[np.ix_([0, 6], [0, 6])] = k_ax * np.array([[1, -1], [-1, 1]])
    k_tor = G * J / L
    K[np.ix_([3, 9], [3, 9])] = k_tor * np.array([[1, -1], [-1, 1]])
    c1z = 12 * E * Iz / L3
    c2z = 6 * E * Iz / L2
    c3z = 4 * E * Iz / L
    c4z = 2 * E * Iz / L
    idx_v_thz = np.array([1, 5, 7, 11])
    K_b_z = np.array([[c1z, c2z, -c1z, c2z], [c2z, c3z, -c2z, c4z], [-c1z, -c2z, c1z, -c2z], [c2z, c4z, -c2z, c3z]])
    K[np.ix_(idx_v_thz, idx_v_thz)] = K_b_z
    c1y = 12 * E * Iy / L3
    c2y = 6 * E * Iy / L2
    c3y = 4 * E * Iy / L
    c4y = 2 * E * Iy / L
    idx_w_thy = np.array([2, 4, 8, 10])
    K_b_y = np.array([[c1y, -c2y, -c1y, -c2y], [-c2y, c3y, c2y, c4y], [-c1y, c2y, c1y, c2y], [-c2y, c4y, c2y, c3y]])
    K[np.ix_(idx_w_thy, idx_w_thy)] = K_b_y
    return K