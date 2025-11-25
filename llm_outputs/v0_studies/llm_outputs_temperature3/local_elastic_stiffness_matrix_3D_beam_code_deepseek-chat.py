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
    K[6, 6] = axial
    K[0, 6] = -axial
    K[6, 0] = -axial
    G = E / (2 * (1 + nu))
    torsional = G * J / L
    K[3, 3] = torsional
    K[9, 9] = torsional
    K[3, 9] = -torsional
    K[9, 3] = -torsional
    bending_z = E * Iz / L
    K[1, 1] = 12 * bending_z / (L * L)
    K[7, 7] = 12 * bending_z / (L * L)
    K[1, 7] = -12 * bending_z / (L * L)
    K[7, 1] = -12 * bending_z / (L * L)
    K[5, 5] = 4 * bending_z
    K[11, 11] = 4 * bending_z
    K[5, 11] = 2 * bending_z
    K[11, 5] = 2 * bending_z
    K[1, 5] = 6 * bending_z / L
    K[5, 1] = 6 * bending_z / L
    K[1, 11] = 6 * bending_z / L
    K[11, 1] = 6 * bending_z / L
    K[7, 5] = -6 * bending_z / L
    K[5, 7] = -6 * bending_z / L
    K[7, 11] = -6 * bending_z / L
    K[11, 7] = -6 * bending_z / L
    bending_y = E * Iy / L
    K[2, 2] = 12 * bending_y / (L * L)
    K[8, 8] = 12 * bending_y / (L * L)
    K[2, 8] = -12 * bending_y / (L * L)
    K[8, 2] = -12 * bending_y / (L * L)
    K[4, 4] = 4 * bending_y
    K[10, 10] = 4 * bending_y
    K[4, 10] = 2 * bending_y
    K[10, 4] = 2 * bending_y
    K[2, 4] = -6 * bending_y / L
    K[4, 2] = -6 * bending_y / L
    K[2, 10] = -6 * bending_y / L
    K[10, 2] = -6 * bending_y / L
    K[8, 4] = 6 * bending_y / L
    K[4, 8] = 6 * bending_y / L
    K[8, 10] = 6 * bending_y / L
    K[10, 8] = 6 * bending_y / L
    return K