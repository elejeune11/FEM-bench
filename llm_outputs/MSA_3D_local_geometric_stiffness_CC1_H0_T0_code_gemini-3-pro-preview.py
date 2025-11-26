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
    kg = np.zeros((12, 12))
    P = Fx2
    T = Mx2
    rho = I_rho / A
    k_v_thz = np.array([[6 * P / (5 * L), P / 10, -6 * P / (5 * L), P / 10], [P / 10, 2 * P * L / 15, -P / 10, -P * L / 30], [-6 * P / (5 * L), -P / 10, 6 * P / (5 * L), -P / 10], [P / 10, -P * L / 30, -P / 10, 2 * P * L / 15]])
    indices_v = [1, 5, 7, 11]
    for i in range(4):
        for j in range(4):
            kg[indices_v[i], indices_v[j]] += k_v_thz[i, j]
    k_w_thy = np.array([[6 * P / (5 * L), -P / 10, -6 * P / (5 * L), -P / 10], [-P / 10, 2 * P * L / 15, P / 10, -P * L / 30], [-6 * P / (5 * L), P / 10, 6 * P / (5 * L), P / 10], [-P / 10, -P * L / 30, P / 10, 2 * P * L / 15]])
    indices_w = [2, 4, 8, 10]
    for i in range(4):
        for j in range(4):
            kg[indices_w[i], indices_w[j]] += k_w_thy[i, j]
    k_tor_P = P * (I_rho / A) / L
    kg[3, 3] += k_tor_P
    kg[3, 9] -= k_tor_P
    kg[9, 3] -= k_tor_P
    kg[9, 9] += k_tor_P
    kg[1, 2] -= T / L
    kg[2, 1] -= T / L
    kg[1, 8] += T / L
    kg[8, 1] += T / L
    kg[7, 2] += T / L
    kg[2, 7] += T / L
    kg[7, 8] -= T / L
    kg[8, 7] -= T / L
    kg[4, 5] -= T * L / 6
    kg[5, 4] -= T * L / 6
    kg[4, 11] += T * L / 6
    kg[11, 4] += T * L / 6
    kg[10, 5] += T * L / 6
    kg[5, 10] += T * L / 6
    kg[10, 11] -= T * L / 6
    kg[11, 10] -= T * L / 6
    Vy = -(Mz1 + Mz2) / L
    Vz = (My1 + My2) / L
    term_v_thx = Vz / 10.0
    kg[1, 3] += term_v_thx
    kg[3, 1] += term_v_thx
    kg[1, 9] += term_v_thx
    kg[9, 1] += term_v_thx
    kg[7, 3] -= term_v_thx
    kg[3, 7] -= term_v_thx
    kg[7, 9] -= term_v_thx
    kg[9, 7] -= term_v_thx
    term_w_thx = Vy / 10.0
    kg[2, 3] -= term_w_thx
    kg[3, 2] -= term_w_thx
    kg[2, 9] -= term_w_thx
    kg[9, 2] -= term_w_thx
    kg[8, 3] += term_w_thx
    kg[3, 8] += term_w_thx
    kg[8, 9] += term_w_thx
    kg[9, 8] += term_w_thx
    return kg