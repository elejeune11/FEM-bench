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
    c_P_tor = P * I_rho / (A * L)
    Kg[3, 3] += c_P_tor
    Kg[3, 9] -= c_P_tor
    Kg[9, 3] -= c_P_tor
    Kg[9, 9] += c_P_tor
    c_P_bend = P / (30.0 * L)
    K_base_axial = c_P_bend * np.array([[36.0, 3.0 * L, -36.0, 3.0 * L], [3.0 * L, 4.0 * L ** 2, -3.0 * L, -L ** 2], [-36.0, -3.0 * L, 36.0, -3.0 * L], [3.0 * L, -L ** 2, -3.0 * L, 4.0 * L ** 2]])
    idx_v_thz = np.ix_([1, 5, 7, 11], [1, 5, 7, 11])
    Kg[idx_v_thz] += K_base_axial
    idx_w_thy = np.ix_([2, 4, 8, 10], [2, 4, 8, 10])
    Kg[idx_w_thy] += K_base_axial
    c_T = T / 2.0
    Kg[1, 10] -= c_T
    Kg[10, 1] -= c_T
    Kg[2, 11] += c_T
    Kg[11, 2] += c_T
    Kg[4, 7] -= c_T
    Kg[7, 4] -= c_T
    Kg[5, 8] += c_T
    Kg[8, 5] += c_T
    My_sum = My1 + My2
    c_my1 = My_sum / (2.0 * L)
    c_my2 = My1 / 2.0
    c_my3 = My2 / 2.0
    c_my4 = My_sum / 2.0
    Kg[0, 2] -= c_my1
    Kg[2, 0] -= c_my1
    Kg[0, 8] += c_my1
    Kg[8, 0] += c_my1
    Kg[0, 4] -= c_my2
    Kg[4, 0] -= c_my2
    Kg[0, 10] -= c_my3
    Kg[10, 0] -= c_my3
    Kg[6, 2] += c_my1
    Kg[2, 6] += c_my1
    Kg[6, 8] -= c_my1
    Kg[8, 6] -= c_my1
    Kg[6, 4] += c_my2
    Kg[4, 6] += c_my2
    Kg[6, 10] += c_my3
    Kg[10, 6] += c_my3
    Kg[3, 5] -= c_my4
    Kg[5, 3] -= c_my4
    Kg[3, 11] += c_my4
    Kg[11, 3] += c_my4
    Kg[9, 5] += c_my4
    Kg[5, 9] += c_my4
    Kg[9, 11] -= c_my4
    Kg[11, 9] -= c_my4
    Mz_sum = Mz1 + Mz2
    c_mz1 = Mz_sum / (2.0 * L)
    c_mz2 = Mz1 / 2.0
    c_mz3 = Mz2 / 2.0
    c_mz4 = Mz_sum / 2.0
    Kg[0, 1] += c_mz1
    Kg[1, 0] += c_mz1
    Kg[0, 7] -= c_mz1
    Kg[7, 0] -= c_mz1
    Kg[0, 5] -= c_mz2
    Kg[5, 0] -= c_mz2
    Kg[0, 11] -= c_mz3
    Kg[11, 0] -= c_mz3
    Kg[6, 1] -= c_mz1
    Kg[1, 6] -= c_mz1
    Kg[6, 7] += c_mz1
    Kg[7, 6] += c_mz1
    Kg[6, 5] += c_mz2
    Kg[5, 6] += c_mz2
    Kg[6, 11] += c_mz3
    Kg[11, 6] += c_mz3
    Kg[3, 4] += c