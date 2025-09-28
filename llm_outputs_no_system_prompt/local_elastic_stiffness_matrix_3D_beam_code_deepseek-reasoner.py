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
    axial_term = E * A / L
    K[0, 0] = axial_term
    K[0, 6] = -axial_term
    K[6, 0] = -axial_term
    K[6, 6] = axial_term
    G = E / (2 * (1 + nu))
    torsional_term = G * J / L
    K[3, 3] = torsional_term
    K[3, 9] = -torsional_term
    K[9, 3] = -torsional_term
    K[9, 9] = torsional_term
    bending_z_terms = E * Iz / L ** 3 * np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L ** 2, -6 * L, 2 * L ** 2], [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L ** 2, -6 * L, 4 * L ** 2]])
    dofs_z = [1, 5, 7, 11]
    for (i, dof_i) in enumerate(dofs_z):
        for (j, dof_j) in enumerate(dofs_z):
            K[dof_i, dof_j] = bending_z_terms[i, j]
    bending_y_terms = E * Iy / L ** 3 * np.array([[12, -6 * L, -12, -6 * L], [-6 * L, 4 * L ** 2, 6 * L, 2 * L ** 2], [-12, 6 * L, 12, 6 * L], [-6 * L, 2 * L ** 2, 6 * L, 4 * L ** 2]])
    dofs_y = [2, 4, 8, 10]
    for (i, dof_i) in enumerate(dofs_y):
        for (j, dof_j) in enumerate(dofs_y):
            K[dof_i, dof_j] = bending_y_terms[i, j]
    return K