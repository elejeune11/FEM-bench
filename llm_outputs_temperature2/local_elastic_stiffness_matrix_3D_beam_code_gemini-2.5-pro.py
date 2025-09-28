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
    K = np.zeros((12, 12))
    G = E / (2 * (1 + nu))
    L2 = L * L
    L3 = L * L * L
    k_axial_val = E * A / L
    indices_axial = np.ix_([0, 6], [0, 6])
    K[indices_axial] = k_axial_val * np.array([[1, -1], [-1, 1]])
    k_torsion_val = G * J / L
    indices_torsion = np.ix_([3, 9], [3, 9])
    K[indices_torsion] = k_torsion_val * np.array([[1, -1], [-1, 1]])
    k_bend_z_submatrix = E * Iz / L3 * np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L2, -6 * L, 2 * L2], [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L2, -6 * L, 4 * L2]])
    indices_bend_z = np.ix_([1, 5, 7, 11], [1, 5, 7, 11])
    K[indices_bend_z] = k_bend_z_submatrix
    k_bend_y_submatrix = E * Iy / L3 * np.array([[12, -6 * L, -12, -6 * L], [-6 * L, 4 * L2, 6 * L, 2 * L2], [-12, 6 * L, 12, 6 * L], [-6 * L, 2 * L2, 6 * L, 4 * L2]])
    indices_bend_y = np.ix_([2, 4, 8, 10], [2, 4, 8, 10])
    K[indices_bend_y] = k_bend_y_submatrix
    return K