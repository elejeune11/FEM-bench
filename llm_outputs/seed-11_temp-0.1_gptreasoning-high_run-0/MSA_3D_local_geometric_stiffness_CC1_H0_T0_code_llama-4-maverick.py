def MSA_3D_local_geometric_stiffness_CC1_H0_T0(L: float, A: float, I_rho: float, Fx2: float, Mx2: float, My1: float, Mz1: float, My2: float, Mz2: float) -> np.ndarray:
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
    K_g = np.zeros((12, 12))
    K_g[0, 0] = Fx2 / L
    K_g[0, 6] = -Fx2 / L
    K_g[1, 1] = (6 * Fx2 * L ** 2 + 24 * My1 * L + 24 * My2 * L) / (5 * L ** 3)
    K_g[1, 2] = 0
    K_g[1, 4] = (Fx2 * L + 4 * My1 + 6 * My2) / (10 * L ** 2)
    K_g[1, 5] = (-6 * Mz1 - 6 * Mz2) / (5 * L ** 2)
    K_g[1, 7] = -(6 * Fx2 * L ** 2 + 24 * My1 * L + 24 * My2 * L) / (5 * L ** 3)
    K_g[1, 8] = 0
    K_g[1, 10] = (4 * Fx2 * L + 6 * My1 + 4 * My2) / (10 * L ** 2)
    K_g[1, 11] = (-6 * Mz1 - 6 * Mz2) / (5 * L ** 2)
    K_g[2, 2] = (6 * Fx2 * L ** 2 + 24 * Mz1 * L + 24 * Mz2 * L) / (5 * L ** 3)
    K_g[2, 4] = (6 * My1 + 6 * My2) / (5 * L ** 2)
    K_g[2, 5] = (Fx2 * L + 4 * Mz1 + 6 * Mz2) / (10 * L ** 2)
    K_g[2, 7] = 0
    K_g[2, 8] = -(6 * Fx2 * L ** 2 + 24 * Mz1 * L + 24 * Mz2 * L) / (5 * L ** 3)
    K_g[2, 10] = (6 * My1 + 6 * My2) / (5 * L ** 2)
    K_g[2, 11] = (6 * Mz1 + 4 * Mz2 + Fx2 * L) / (10 * L ** 2)
    K_g[3, 3] = I_rho * Fx2 / (A * L)
    K_g[3, 4] = Mz2 / L
    K_g[3, 5] = -My2 / L
    K_g[3, 9] = -I_rho * Fx2 / (A * L)
    K_g[3, 10] = -Mz2 / L
    K_g[3, 11] = My2 / L
    K_g[4, 4] = (2 * Fx2 * L ** 2 + 8 * My1 * L + 12 * My2 * L) / (15 * L)
    K_g[4, 5] = -Mx2 / (2 * L)
    K_g[4, 7] = -(Fx2 * L + 4 * My1 + 6 * My2) / (10 * L ** 2)
    K_g[4, 8] = -(6 * My1 + 6 * My2) / (5 * L ** 2)
    K_g[4, 10] = (Fx2 * L + My1 + My2) / 30
    K_g[4, 11] = Mx2 / (2 * L)
    K_g[5, 5] = (2 * Fx2 * L ** 2 + 8 * Mz1 * L + 12 * Mz2 * L) / (15 * L)
    K_g[5, 7] = (6 * Mz1 + 6 * Mz2) / (5 * L ** 2)
    K_g[5, 8] = -(Fx2 * L + 4 * Mz1 + 6 * Mz2) / (10 * L ** 2)
    K_g[5, 10] = -Mx2 / (2 * L)
    K_g[5, 11] = (Fx2 * L + Mz1 + Mz2) / 30
    K_g[6, 6] = Fx2 / L
    K_g[7, 7] = (6 * Fx2 * L ** 2 + 24 * My1 * L + 24 * My2 * L) / (5 * L ** 3)
    K_g[7, 8] = 0
    K_g[7, 10] = -(4 * Fx2 * L + 6 * My1 + 4 * My2) / (10 * L ** 2)
    K_g[7, 11] = (6 * Mz1 + 6 * Mz2) / (5 * L ** 2)
    K_g[8, 8] = (6 * Fx2 * L ** 2 + 24 * Mz1 * L + 24 * Mz2 * L) / (5 * L ** 3)
    K_g[8, 10] = -(6 * My1 + 6 * My2) / (5 * L ** 2)
    K_g[8, 11] = -(6 * Mz1 + 4 * Mz2 + Fx2 * L) / (10 * L ** 2)
    K_g[9, 9] = I_rho * Fx2 / (A * L)
    K_g[9, 10] = Mz2 / L
    K_g[9, 11] = -My2 / L
    K_g[10, 10] = (2 * L * Fx2 + 8 * My2 + 4 * My1) / 15
    K_g[10, 11] = Mx2 / (2 * L)
    K_g[11, 11] = (2 * L * Fx2 + 8 * Mz2 + 4 * Mz1) / 15
    K_g = (K_g + K_g.T) / 2
    return K_g