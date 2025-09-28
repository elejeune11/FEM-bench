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
    G = E / (2 * (1 + nu))
    k = np.zeros((12, 12))
    axial = E * A / L
    k[0, 0] = axial
    k[0, 6] = -axial
    k[6, 0] = -axial
    k[6, 6] = axial
    torsional = G * J / L
    k[3, 3] = torsional
    k[3, 9] = -torsional
    k[9, 3] = -torsional
    k[9, 9] = torsional
    bending_z = E * Iz / L
    k[1, 1] = 12 * bending_z / L ** 2
    k[1, 5] = 6 * bending_z / L
    k[1, 7] = -12 * bending_z / L ** 2
    k[1, 11] = 6 * bending_z / L
    k[5, 1] = 6 * bending_z / L
    k[5, 5] = 4 * bending_z
    k[5, 7] = -6 * bending_z / L
    k[5, 11] = 2 * bending_z
    k[7, 1] = -12 * bending_z / L ** 2
    k[7, 5] = -6 * bending_z / L
    k[7, 7] = 12 * bending_z / L ** 2
    k[7, 11] = -6 * bending_z / L
    k[11, 1] = 6 * bending_z / L
    k[11, 5] = 2 * bending_z
    k[11, 7] = -6 * bending_z / L
    k[11, 11] = 4 * bending_z
    bending_y = E * Iy / L
    k[2, 2] = 12 * bending_y / L ** 2
    k[2, 4] = -6 * bending_y / L
    k[2, 8] = -12 * bending_y / L ** 2
    k[2, 10] = -6 * bending_y / L
    k[4, 2] = -6 * bending_y / L
    k[4, 4] = 4 * bending_y
    k[4, 8] = 6 * bending_y / L
    k[4, 10] = 2 * bending_y
    k[8, 2] = -12 * bending_y / L ** 2
    k[8, 4] = 6 * bending_y / L
    k[8, 8] = 12 * bending_y / L ** 2
    k[8, 10] = 6 * bending_y / L
    k[10, 2] = -6 * bending_y / L
    k[10, 4] = 2 * bending_y
    k[10, 8] = 6 * bending_y / L
    k[10, 10] = 4 * bending_y
    return k