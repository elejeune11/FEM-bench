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
    T = Mx2
    C1 = P / (30.0 * L)
    mat_P_bend = C1 * np.array([[36, 3 * L, -36, 3 * L], [3 * L, 4 * L ** 2, -3 * L, -L ** 2], [-36, -3 * L, 36, -3 * L], [3 * L, -L ** 2, -3 * L, 4 * L ** 2]])
    dofs_vy = np.array([1, 5, 7, 11])
    Kg[np.ix_(dofs_vy, dofs_vy)] += mat_P_bend
    dofs_wz = np.array([2, 4, 8, 10])
    Kg[np.ix_(dofs_wz, dofs_wz)] += mat_P_bend
    C2 = P * I_rho / (A * L)
    Kg[3, 3] += C2
    Kg[9, 9] += C2
    Kg[3, 9] -= C2
    Kg[9, 3] -= C2
    My_avg = (My1 + My2) / 2.0
    Mz_avg = (Mz1 + Mz2) / 2.0
    C_Mz = Mz_avg / L
    Kg[1, 3] += C_Mz
    Kg[3, 1] += C_Mz
    Kg[1, 9] -= C_Mz
    Kg[9, 1] -= C_Mz
    Kg[3, 7] -= C_Mz
    Kg[7, 3] -= C_Mz
    Kg[7, 9] += C_Mz
    Kg[9, 7] += C_Mz
    C_My = My_avg / L
    Kg[2, 3] -= C_My
    Kg[3, 2] -= C_My
    Kg[2, 9] += C_My
    Kg[9, 2] += C_My
    Kg[3, 8] += C_My
    Kg[8, 3] += C_My
    Kg[8, 9] -= C_My
    Kg[9, 8] -= C_My
    C_T = T / 6.0
    Kg[4, 5] += C_T
    Kg[5, 4] += C_T
    Kg[4, 11] -= C_T
    Kg[11, 4] -= C_T
    Kg[5, 10] -= C_T
    Kg[10, 5] -= C_T
    Kg[10, 11] += C_T
    Kg[11, 10] += C_T
    return Kg