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
    P = Fx2
    if L > 0:
        sub_P = P / (30.0 * L) * np.array([[36.0, 3.0 * L, -36.0, 3.0 * L], [3.0 * L, 4.0 * L ** 2, -3.0 * L, -L ** 2], [-36.0, -3.0 * L, 36.0, -3.0 * L], [3.0 * L, -L ** 2, -3.0 * L, 4.0 * L ** 2]])
        dofs_yz = np.array([1, 5, 7, 11])
        Kg[np.ix_(dofs_yz, dofs_yz)] += sub_P
        dofs_xz = np.array([2, 4, 8, 10])
        Kg[np.ix_(dofs_xz, dofs_xz)] += sub_P
        if A > 0:
            term_P_torsion = P * I_rho / (A * L)
            Kg[3, 3] += term_P_torsion
            Kg[9, 9] += term_P_torsion
            Kg[3, 9] -= term_P_torsion
            Kg[9, 3] -= term_P_torsion
    T = Mx2
    term_T = T / 2.0
    Kg[1, 10] -= term_T
    Kg[10, 1] -= term_T
    Kg[2, 11] += term_T
    Kg[11, 2] += term_T
    Kg[4, 7] += term_T
    Kg[7, 4] += term_T
    Kg[5, 8] -= term_T
    Kg[8, 5] -= term_T
    if L > 0:
        My_avg_L = (My1 + My2) / (2.0 * L)
        Kg[0, 2] += My_avg_L
        Kg[2, 0] += My_avg_L
        Kg[0, 8] -= My_avg_L
        Kg[8, 0] -= My_avg_L
        Kg[2, 6] -= My_avg_L
        Kg[6, 2] -= My_avg_L
        Kg[6, 8] += My_avg_L
        Kg[8, 6] += My_avg_L
    term_My1 = My1 / 2.0
    term_My2 = My2 / 2.0
    Kg[2, 3] -= term_My1
    Kg[3, 2] -= term_My1
    Kg[2, 9] -= term_My2
    Kg[9, 2] -= term_My2
    Kg[3, 8] += term_My1
    Kg[8, 3] += term_My1
    Kg[8, 9] += term_My2
    Kg[9, 8] += term_My2
    if L > 0:
        Mz_avg_L = (Mz1 + Mz2) / (2.0 * L)
        Kg[0, 1] -= Mz_avg_L
        Kg[1, 0] -= Mz_avg_L
        Kg[0, 7] += Mz_avg_L
        Kg[7, 0] += Mz_avg_L
        Kg[1, 6] += Mz_avg_L
        Kg[6, 1] += Mz_avg_L
        Kg[6, 7] -= Mz_avg_L
        Kg[7, 6] -= Mz_avg_L
    term_Mz1 = Mz1 / 2.0
    term_Mz2 = Mz2 / 2.0
    Kg[1, 3] -= term_Mz1
    Kg[3, 1] -= term_Mz1
    Kg[1, 9] -= term_Mz2
    Kg[9, 1] -= term_Mz2
    Kg[3, 7] += term_Mz1
    Kg[7, 3] += term_Mz1
    Kg[7, 9] += term_Mz2
    Kg[9, 7] += term_Mz2
    return Kg