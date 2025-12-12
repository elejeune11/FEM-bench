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
    Kg = np.zeros((12, 12))
    c1 = Fx2 / (30.0 * L)
    c2 = Fx2 * L / 30.0
    c3 = (My1 + My2) / (30.0 * L)
    c4 = (Mz1 + Mz2) / (30.0 * L)
    c5 = (My2 - My1) / 30.0
    c6 = (Mz2 - Mz1) / 30.0
    c7 = Mx2 / (30.0 * L)
    Kg[0, 0] = 0.0
    Kg[1, 1] = 6.0 * c1 + 12.0 * c3
    Kg[1, 4] = 4.0 * L * c1 + 2.0 * L * c3 + 0.5 * My1
    Kg[1, 7] = -6.0 * c1 - 12.0 * c3
    Kg[1, 10] = 2.0 * L * c1 + 2.0 * L * c3 + 0.5 * My2
    Kg[2, 2] = 6.0 * c1 + 12.0 * c4
    Kg[2, 5] = -(4.0 * L * c1 + 2.0 * L * c4 + 0.5 * Mz1)
    Kg[2, 8] = -6.0 * c1 - 12.0 * c4
    Kg[2, 11] = -(2.0 * L * c1 + 2.0 * L * c4 + 0.5 * Mz2)
    Kg[3, 3] = 0.0
    Kg[4, 1] = Kg[1, 4]
    Kg[4, 4] = 4.0 * L * c1 + 8.0 * L * c3 / 3.0
    Kg[4, 7] = -2.0 * L * c1 - 2.0 * L * c3
    Kg[4, 10] = 2.0 * L * c1 + 2.0 * L * c3 / 3.0
    Kg[5, 2] = Kg[2, 5]
    Kg[5, 5] = 4.0 * L * c1 + 8.0 * L * c4 / 3.0
    Kg[5, 8] = 2.0 * L * c1 + 2.0 * L * c4
    Kg[5, 11] = 2.0 * L * c1 + 2.0 * L * c4 / 3.0
    Kg[6, 6] = 0.0
    Kg[7, 1] = Kg[1, 7]
    Kg[7, 4] = -2.0 * L * c1 - 2.0 * L * c3
    Kg[7, 7] = 6.0 * c1 + 12.0 * c3
    Kg[7, 10] = -(4.0 * L * c1 + 2.0 * L * c3 + 0.5 * My2)
    Kg[8, 2] = Kg[2, 8]
    Kg[8, 5] = 2.0 * L * c1 + 2.0 * L * c4
    Kg[8, 8] = 6.0 * c1 + 12.0 * c4
    Kg[8, 11] = 4.0 * L * c1 + 2.0 * L * c4 + 0.5 * Mz2
    Kg[9, 9] = 0.0
    Kg[10, 1] = Kg[1, 10]
    Kg[10, 4] = Kg[4, 10]
    Kg[10, 7] = Kg[7, 10]
    Kg[10, 10] = 4.0 * L * c1 + 8.0 * L * c3 / 3.0
    Kg[11, 2] = Kg[2, 11]
    Kg[11, 5] = Kg[5, 11]
    Kg[11, 8] = Kg[8, 11]
    Kg[11, 11] = 4.0 * L * c1 + 8.0 * L * c4 / 3.0
    for i in range(12):
        for j in range(i + 1, 12):
            Kg[j, i] = Kg[i, j]
    return Kg