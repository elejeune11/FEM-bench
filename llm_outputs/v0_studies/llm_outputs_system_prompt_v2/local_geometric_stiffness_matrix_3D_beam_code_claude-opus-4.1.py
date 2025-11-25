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
    Kg = np.zeros((12, 12))
    P = -Fx2
    T = Mx2
    c1 = 6.0 / (5.0 * L)
    c2 = 1.0 / 10.0
    c3 = 2.0 * L / 15.0
    c4 = L / 30.0
    Kg[1, 1] = c1 * P
    Kg[1, 5] = c2 * P
    Kg[1, 7] = -c1 * P
    Kg[1, 11] = c2 * P
    Kg[2, 2] = c1 * P
    Kg[2, 4] = -c2 * P
    Kg[2, 8] = -c1 * P
    Kg[2, 10] = -c2 * P
    Kg[3, 3] = I_rho * P / (A * L)
    Kg[3, 9] = -I_rho * P / (A * L)
    Kg[4, 2] = -c2 * P
    Kg[4, 4] = c3 * P
    Kg[4, 8] = c2 * P
    Kg[4, 10] = -c4 * P
    Kg[5, 1] = c2 * P
    Kg[5, 5] = c3 * P
    Kg[5, 7] = -c2 * P
    Kg[5, 11] = -c4 * P
    Kg[7, 1] = -c1 * P
    Kg[7, 5] = -c2 * P
    Kg[7, 7] = c1 * P
    Kg[7, 11] = -c2 * P
    Kg[8, 2] = -c1 * P
    Kg[8, 4] = c2 * P
    Kg[8, 8] = c1 * P
    Kg[8, 10] = c2 * P
    Kg[9, 3] = -I_rho * P / (A * L)
    Kg[9, 9] = I_rho * P / (A * L)
    Kg[10, 2] = -c2 * P
    Kg[10, 4] = -c4 * P
    Kg[10, 8] = c2 * P
    Kg[10, 10] = c3 * P
    Kg[11, 1] = c2 * P
    Kg[11, 5] = -c4 * P
    Kg[11, 7] = -c2 * P
    Kg[11, 11] = c3 * P
    Kg[2, 3] = -T / L
    Kg[3, 2] = -T / L
    Kg[1, 3] = T / L
    Kg[3, 1] = T / L
    Kg[8, 3] = T / L
    Kg[3, 8] = T / L
    Kg[7, 3] = -T / L
    Kg[3, 7] = -T / L
    Kg[2, 9] = T / L
    Kg[9, 2] = T / L
    Kg[1, 9] = -T / L
    Kg[9, 1] = -T / L
    Kg[8, 9] = -T / L
    Kg[9, 8] = -T / L
    Kg[7, 9] = T / L
    Kg[9, 7] = T / L
    Kg[1, 0] = (My2 - My1) / L ** 2
    Kg[0, 1] = (My2 - My1) / L ** 2
    Kg[7, 0] = -(My2 - My1) / L ** 2
    Kg[0, 7] = -(My2 - My1) / L ** 2
    Kg[1, 6] = -(My2 - My1) / L ** 2
    Kg[6, 1] = -(My2 - My1) / L ** 2
    Kg[7, 6] = (My2 - My1) / L ** 2
    Kg[6, 7] = (My2 - My1) / L ** 2
    Kg[2, 0] = -(Mz2 - Mz1) / L ** 2
    Kg[0, 2] = -(Mz2 - Mz1) / L ** 2
    Kg[8, 0] = (Mz2 - Mz1) / L ** 2
    Kg[0, 8] = (Mz2 - Mz1) / L ** 2
    Kg[2, 6] = (Mz2 - Mz1) / L ** 2
    Kg[6, 2] = (Mz2 - Mz1) / L ** 2
    Kg[8, 6] = -(Mz2 - Mz1) / L ** 2
    Kg[6, 8] = -(Mz2 - Mz1) / L ** 2
    Kg[5, 0] = -My1 / L
    Kg[0, 5] = -My1 / L
    Kg[11, 0] = -My2 / L
    Kg[0, 11] = -My2 / L
    Kg[5, 6] = My1 / L
    Kg[6, 5] = My1 / L
    Kg[11, 6] = My2 / L
    Kg[6, 11] = My2 / L
    Kg[4, 0] = Mz1 / L
    Kg[0, 4] = Mz1 / L
    Kg[10, 0] = Mz2 / L
    Kg[0, 10] = Mz2 / L
    Kg[4, 6] = -Mz1 / L
    Kg[6, 4] = -Mz1 / L
    Kg[10, 6] = -Mz2 / L
    Kg[6, 10] = -Mz2 / L
    return Kg