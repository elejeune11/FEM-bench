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
    K = np.zeros((12, 12))
    k_axial = E * A / L
    K[0, 0] = k_axial
    K[0, 6] = -k_axial
    K[6, 0] = -k_axial
    K[6, 6] = k_axial
    G = E / (2 * (1 + nu))
    k_torsion = G * J / L
    K[3, 3] = k_torsion
    K[3, 9] = -k_torsion
    K[9, 3] = -k_torsion
    K[9, 9] = k_torsion
    k_v = 12 * E * Iz / L ** 3
    k_v_m = 6 * E * Iz / L ** 2
    k_v_mm = 4 * E * Iz / L
    K[1, 1] = k_v
    K[1, 5] = k_v_m
    K[1, 7] = -k_v
    K[1, 11] = k_v_m
    K[5, 1] = k_v_m
    K[5, 5] = k_v_mm
    K[5, 7] = -k_v_m
    K[5, 11] = 2 * E * Iz / L
    K[7, 1] = -k_v
    K[7, 5] = -k_v_m
    K[7, 7] = k_v
    K[7, 11] = -k_v_m
    K[11, 1] = k_v_m
    K[11, 5] = 2 * E * Iz / L
    K[11, 7] = -k_v_m
    K[11, 11] = k_v_mm
    k_w = 12 * E * Iy / L ** 3
    k_w_m = 6 * E * Iy / L ** 2
    k_w_mm = 4 * E * Iy / L
    K[2, 2] = k_w
    K[2, 4] = -k_w_m
    K[2, 8] = -k_w
    K[2, 10] = -k_w_m
    K[4, 2] = -k_w_m
    K[4, 4] = k_w_mm
    K[4, 8] = k_w_m
    K[4, 10] = 2 * E * Iy / L
    K[8, 2] = -k_w
    K[8, 4] = k_w_m
    K[8, 8] = k_w
    K[8, 10] = k_w_m
    K[10, 2] = -k_w_m
    K[10, 4] = 2 * E * Iy / L
    K[10, 8] = k_w_m
    K[10, 10] = k_w_mm
    return K