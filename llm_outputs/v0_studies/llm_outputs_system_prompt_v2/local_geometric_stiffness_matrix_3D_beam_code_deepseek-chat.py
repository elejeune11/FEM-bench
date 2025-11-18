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
    Kg[1, 1] = Fx2 * 6.0 / (5.0 * L)
    Kg[1, 5] = Fx2 / 10.0
    Kg[1, 7] = -Fx2 * 6.0 / (5.0 * L)
    Kg[1, 11] = Fx2 / 10.0
    Kg[5, 1] = Fx2 / 10.0
    Kg[5, 5] = Fx2 * (2.0 * L) / 15.0
    Kg[5, 7] = -Fx2 / 10.0
    Kg[5, 11] = -Fx2 * L / 30.0
    Kg[7, 1] = -Fx2 * 6.0 / (5.0 * L)
    Kg[7, 5] = -Fx2 / 10.0
    Kg[7, 7] = Fx2 * 6.0 / (5.0 * L)
    Kg[7, 11] = -Fx2 / 10.0
    Kg[11, 1] = Fx2 / 10.0
    Kg[11, 5] = -Fx2 * L / 30.0
    Kg[11, 7] = -Fx2 / 10.0
    Kg[11, 11] = Fx2 * (2.0 * L) / 15.0
    Kg[2, 2] = Fx2 * 6.0 / (5.0 * L)
    Kg[2, 4] = -Fx2 / 10.0
    Kg[2, 8] = -Fx2 * 6.0 / (5.0 * L)
    Kg[2, 10] = -Fx2 / 10.0
    Kg[4, 2] = -Fx2 / 10.0
    Kg[4, 4] = Fx2 * (2.0 * L) / 15.0
    Kg[4, 8] = Fx2 / 10.0
    Kg[4, 10] = -Fx2 * L / 30.0
    Kg[8, 2] = -Fx2 * 6.0 / (5.0 * L)
    Kg[8, 4] = Fx2 / 10.0
    Kg[8, 8] = Fx2 * 6.0 / (5.0 * L)
    Kg[8, 10] = Fx2 / 10.0
    Kg[10, 2] = -Fx2 / 10.0
    Kg[10, 4] = -Fx2 * L / 30.0
    Kg[10, 8] = Fx2 / 10.0
    Kg[10, 10] = Fx2 * (2.0 * L) / 15.0
    Kg[3, 5] = -Mx2 / (2.0 * L)
    Kg[3, 11] = -Mx2 / (2.0 * L)
    Kg[5, 3] = -Mx2 / (2.0 * L)
    Kg[11, 3] = -Mx2 / (2.0 * L)
    Kg[3, 4] = Mx2 / (2.0 * L)
    Kg[3, 10] = Mx2 / (2.0 * L)
    Kg[4, 3] = Mx2 / (2.0 * L)
    Kg[10, 3] = Mx2 / (2.0 * L)
    Kg[1, 3] = My1 / (2.0 * L)
    Kg[3, 1] = My1 / (2.0 * L)
    Kg[5, 3] += My1 / 2.0
    Kg[3, 5] += My1 / 2.0
    Kg[7, 3] = -My1 / (2.0 * L)
    Kg[3, 7] = -My1 / (2.0 * L)
    Kg[11, 3] += My1 / 2.0
    Kg[3, 11] += My1 / 2.0
    Kg[2, 3] = -Mz1 / (2.0 * L)
    Kg[3, 2] = -Mz1 / (2.0 * L)
    Kg[4, 3] += -Mz1 / 2.0
    Kg[3, 4] += -Mz1 / 2.0
    Kg[8, 3] = Mz1 / (2.0 * L)
    Kg[3, 8] = Mz1 / (2.0 * L)
    Kg[10, 3] += -Mz1 / 2.0
    Kg[3, 10] += -Mz1 / 2.0
    Kg[1, 9] = My2 / (2.0 * L)
    Kg[9, 1] = My2 / (2.0 * L)
    Kg[5, 9] += My2 / 2.0
    Kg[9, 5] += My2 / 2.0
    Kg[7, 9] = -My2 / (2.0 * L)
    Kg[9, 7] = -My2 / (2.0 * L)
    Kg[11, 9] += My2 / 2.0
    Kg[9, 11] += My2 / 2.0
    Kg[2, 9] = -Mz2 / (2.0 * L)
    Kg[9, 2] = -Mz2 / (2.0 * L)
    Kg[4, 9] += -Mz2 / 2.0
    Kg[9, 4] += -Mz2 / 2.0
    Kg[8, 9] = Mz2 / (2.0 * L)
    Kg[9, 8] = Mz2 / (2.0 * L)
    Kg[10, 9] += -Mz2 / 2.0
    Kg[9, 10] += -Mz2 / 2.0
    for i in range(12):
        for j in range(i + 1, 12):
            Kg[i, j] = Kg[j, i]
    return Kg