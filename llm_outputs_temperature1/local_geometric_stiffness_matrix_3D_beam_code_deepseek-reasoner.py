def local_geometric_stiffness_matrix_3D_beam(L: float, A: float, I_rho: float, Fx2: float, Mx2: float, My1: float, Mz1: float, My2: float, Mz2: float) -> np.ndarray:
    """
    Return the 12x12 local geometric stiffness matrix with torsion-bending coupling for a 3D Euler-Bernoulli beam element.
    The beam is assumed to be aligned with the local x-axis. The geometric stiffness matrix is used in conjunction with the elastic stiffness matrix for nonlinear structural analysis.
    Degrees of freedom are ordered as:
        [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]
    Where:
    Parameters:
        L (float): Length of the beam element [length units]
        A (float): Cross-sectional area [length² units]
        I_rho (float): Polar moment of inertia about the x-axis [length⁴ units]
        Fx2 (float): Internal axial force in the element (positive = tension), evaluated at node 2 [force units]
        Mx2 (float): Torsional moment at node 2 about x-axis [forcexlength units]
        My1 (float): Bending moment at node 1 about y-axis [forcexlength units]
        Mz1 (float): Bending moment at node 1 about z-axis [forcexlength units]
        My2 (float): Bending moment at node 2 about y-axis [forcexlength units]
        Mz2 (float): Bending moment at node 2 about z-axis [forcexlength units]
    Returns:
        np.ndarray: A 12x12 symmetric geometric stiffness matrix in local coordinates.
                    Positive axial force (tension) contributes to element stiffness;
                    negative axial force (compression) can lead to instability.
    Notes:
    Effects captured
        + **Tension (+Fx2)** increases lateral/torsional stiffness.
        + Compression (-Fx2) decreases it and may trigger buckling when K_e + K_g becomes singular.
    Implementation details
    """
    K_geo = np.zeros((12, 12))
    P = Fx2
    a = P / L
    K_geo[1, 1] = 6 / 5 * a
    K_geo[1, 5] = -a / 10
    K_geo[1, 7] = -6 / 5 * a
    K_geo[1, 11] = -a / 10
    K_geo[2, 2] = 6 / 5 * a
    K_geo[2, 4] = a / 10
    K_geo[2, 8] = -6 / 5 * a
    K_geo[2, 10] = a / 10
    K_geo[4, 4] = 2 * L * a / 15
    K_geo[4, 8] = -a / 10
    K_geo[4, 10] = -L * a / 30
    K_geo[5, 5] = 2 * L * a / 15
    K_geo[5, 7] = a / 10
    K_geo[5, 11] = -L * a / 30
    K_geo[7, 7] = 6 / 5 * a
    K_geo[7, 11] = a / 10
    K_geo[8, 8] = 6 / 5 * a
    K_geo[8, 10] = -a / 10
    K_geo[10, 10] = 2 * L * a / 15
    K_geo[11, 11] = 2 * L * a / 15
    T = Mx2
    b = T / (2 * L)
    K_geo[1, 3] = b
    K_geo[2, 3] = -b
    K_geo[3, 7] = -b
    K_geo[3, 8] = b
    K_geo[3, 4] = -L * b / 3
    K_geo[3, 5] = -L * b / 3
    K_geo[3, 10] = L * b / 6
    K_geo[3, 11] = L * b / 6
    c_y1 = My1 / L
    c_y2 = My2 / L
    c_z1 = Mz1 / L
    c_z2 = Mz2 / L
    K_geo[1, 4] += c_z1 / 2
    K_geo[1, 10] += c_z2 / 2
    K_geo[2, 5] += -c_y1 / 2
    K_geo[2, 11] += -c_y2 / 2
    K_geo[4, 4] += L * (c_y1 + c_y2) / 3
    K_geo[5, 5] += L * (c_z1 + c_z2) / 3
    K_geo[4, 10] += L * c_y2 / 6
    K_geo[5, 11] += L * c_z2 / 6
    for i in range(12):
        for j in range(i + 1, 12):
            K_geo[j, i] = K_geo[i, j]
    return K_geo