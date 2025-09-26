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
    axial = E * A / L
    K[0, 0] = axial
    K[0, 6] = -axial
    K[6, 0] = -axial
    K[6, 6] = axial
    G = E / (2 * (1 + nu))
    torsional = G * J / L
    K[3, 3] = torsional
    K[3, 9] = -torsional
    K[9, 3] = -torsional
    K[9, 9] = torsional
    bending_y_1 = 12 * E * Iz / L ** 3
    bending_y_2 = 6 * E * Iz / L ** 2
    bending_y_3 = 4 * E * Iz / L
    bending_y_4 = 2 * E * Iz / L
    K[2, 2] = bending_y_1
    K[2, 5] = bending_y_2
    K[2, 8] = -bending_y_1
    K[2, 11] = bending_y_2
    K[5, 2] = bending_y_2
    K[5, 5] = bending_y_3
    K[5, 8] = -bending_y_2
    K[5, 11] = bending_y_4
    K[8, 2] = -bending_y_1
    K[8, 5] = -bending_y_2
    K[8, 8] = bending_y_1
    K[8, 11] = -bending_y_2
    K[11, 2] = bending_y_2
    K[11, 5] = bending_y_4
    K[11, 8] = -bending_y_2
    K[11, 11] = bending_y_3
    bending_z_1 = 12 * E * Iy / L ** 3
    bending_z_2 = 6 * E * Iy / L ** 2
    bending_z_3 = 4 * E * Iy / L
    bending_z_4 = 2 * E * Iy / L
    K[1, 1] = bending_z_1
    K[1, 4] = -bending_z_2
    K[1, 7] = -bending_z_1
    K[1, 10] = -bending_z_2
    K[4, 1] = -bending_z_2
    K[4, 4] = bending_z_3
    K[4, 7] = bending_z_2
    K[4, 10] = bending_z_4
    K[7, 1] = -bending_z_1
    K[7, 4] = bending_z_2
    K[7, 7] = bending_z_1
    K[7, 10] = bending_z_2
    K[10, 1] = -bending_z_2
    K[10, 4] = bending_z_4
    K[10, 7] = bending_z_2
    K[10, 10] = bending_z_3
    return K