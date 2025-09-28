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
    K = np.zeros((12, 12))
    P = Fx2
    tau = Mx2
    phi = P * I_rho / (A * L)
    a_y = P * L ** 2 / np.pi ** 2 if P != 0 else 0
    K[1, 1] = 6 / (5 * L) * P
    K[1, 5] = P / 10
    K[1, 7] = -6 / (5 * L) * P
    K[1, 11] = P / 10
    K[5, 1] = P / 10
    K[5, 5] = 2 * L * P / 15
    K[5, 7] = -P / 10
    K[5, 11] = -L * P / 30
    K[7, 1] = -6 / (5 * L) * P
    K[7, 5] = -P / 10
    K[7, 7] = 6 / (5 * L) * P
    K[7, 11] = -P / 10
    K[11, 1] = P / 10
    K[11, 5] = -L * P / 30
    K[11, 7] = -P / 10
    K[11, 11] = 2 * L * P / 15
    K[2, 2] = 6 / (5 * L) * P
    K[2, 4] = -P / 10
    K[2, 8] = -6 / (5 * L) * P
    K[2, 10] = -P / 10
    K[4, 2] = -P / 10
    K[4, 4] = 2 * L * P / 15
    K[4, 8] = P / 10
    K[4, 10] = -L * P / 30
    K[8, 2] = -6 / (5 * L) * P
    K[8, 4] = P / 10
    K[8, 8] = 6 / (5 * L) * P
    K[8, 10] = P / 10
    K[10, 2] = -P / 10
    K[10, 4] = -L * P / 30
    K[10, 8] = P / 10
    K[10, 10] = 2 * L * P / 15
    K[3, 1] = -tau / (2 * L)
    K[3, 7] = tau / (2 * L)
    K[1, 3] = -tau / (2 * L)
    K[7, 3] = tau / (2 * L)
    K[3, 2] = tau / (2 * L)
    K[3, 8] = -tau / (2 * L)
    K[2, 3] = tau / (2 * L)
    K[8, 3] = -tau / (2 * L)
    K[3, 3] = phi
    K[3, 9] = -phi
    K[9, 3] = -phi
    K[9, 9] = phi