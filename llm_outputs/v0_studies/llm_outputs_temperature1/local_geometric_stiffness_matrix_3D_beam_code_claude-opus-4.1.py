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
    import numpy as np
    Kg = np.zeros((12, 12))
    P = Fx2
    Kg[1, 1] = 6 * P / (5 * L)
    Kg[1, 5] = P / 10
    Kg[1, 7] = -6 * P / (5 * L)
    Kg[1, 11] = P / 10
    Kg[2, 2] = 6 * P / (5 * L)
    Kg[2, 4] = -P / 10
    Kg[2, 8] = -6 * P / (5 * L)
    Kg[2, 10] = -P / 10
    Kg[4, 2] = -P / 10
    Kg[4, 4] = 2 * P * L / 15
    Kg[4, 8] = P / 10
    Kg[4, 10] = -P * L / 30
    Kg[5, 1] = P / 10
    Kg[5, 5] = 2 * P * L / 15
    Kg[5, 7] = -P / 10
    Kg[5, 11] = -P * L / 30
    Kg[7, 1] = -6 * P / (5 * L)
    Kg[7, 5] = -P / 10
    Kg[7, 7] = 6 * P / (5 * L)
    Kg[7, 11] = -P / 10
    Kg[8, 2] = -6 * P / (5 * L)
    Kg[8, 4] = P / 10
    Kg[8, 8] = 6 * P / (5 * L)
    Kg[8, 10] = P / 10
    Kg[10, 2] = -P / 10
    Kg[10, 4] = -P * L / 30
    Kg[10, 8] = P / 10
    Kg[10, 10] = 2 * P * L / 15
    Kg[11, 1] = P / 10
    Kg[11, 5] = -P * L / 30
    Kg[11, 7] = -P / 10
    Kg[11, 11] = 2 * P * L / 15
    T = Mx2
    if I_rho != 0:
        factor = T / I_rho
        Kg[1, 4] = factor
        Kg[2, 5] = -factor
        Kg[4, 1] = factor
        Kg[5, 2] = -factor
        Kg[7, 10] = factor
        Kg[8, 11] = -factor
        Kg[10, 7] = factor
        Kg[11, 8] = -factor
    Kg[2, 0] = -(My2 - My1) / (L * L)
    Kg[2, 6] = (My2 - My1) / (L * L)
    Kg[0, 2] = -(My2 - My1) / (L * L)
    Kg[6, 2] = (My2 - My1) / (L * L)
    Kg[8, 0] = (My2 - My1) / (L * L)
    Kg[8, 6] = -(My2 - My1) / (L * L)
    Kg[0, 8] = (My2 - My1) / (L * L)
    Kg[6, 8] = -(My2 - My1) / (L * L)
    Kg[4, 0] = (2 * My1 + My2) / (L * L)
    Kg[4, 6] = -(2 * My1 + My2) / (L * L)
    Kg[0, 4] = (2 * My1 + My2) / (L * L)
    Kg[6, 4] = -(2 * My1 + My2) / (L * L)
    Kg[10, 0] = (My1 + 2 * My2) / (L * L)
    Kg[10, 6] = -(My1 + 2 * My2) / (L * L)
    Kg[0, 10] = (My1 + 2 * My2) / (L * L)
    Kg[6, 10] = -(My1 + 2 * My2) / (L * L)
    Kg[1, 0] = (Mz2 - Mz1) / (L * L)
    Kg[1, 6] = -(Mz2 - Mz1) / (L * L)
    Kg[0, 1] = (Mz2 - Mz1) / (L * L)
    Kg[6, 1] = -(Mz2 - Mz1) / (L * L)
    Kg[7, 0] = -(Mz2 - Mz1) / (L * L)
    Kg[7, 6] = (Mz2 - Mz1) / (L * L)
    Kg[0, 7] = -(Mz2 - Mz1) / (L * L)
    Kg[6, 7] = (Mz2 - Mz1) / (L * L)
    Kg[5, 0] = (2 * Mz1 + Mz2) / (L * L)
    Kg[5, 6] = -(2 * Mz1 + Mz2) / (L * L)
    Kg[0, 5] = (2 * Mz1 + Mz2) / (L * L)
    Kg[6, 5] = -(2 * Mz1 + Mz2) / (L * L)
    Kg[11, 0] = (Mz1 + 2 * Mz2) / (L * L)
    Kg[11, 6] = -(Mz1 + 2 * Mz2) / (L * L)
    Kg[0, 11] = (Mz1 + 2 * Mz2) / (L * L)
    Kg[6, 11] = -(Mz1 + 2 * Mz2) / (L * L)
    return Kg