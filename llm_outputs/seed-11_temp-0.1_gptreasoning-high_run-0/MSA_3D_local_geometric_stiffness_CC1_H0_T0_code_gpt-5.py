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
    if not np.isfinite(L) or L == 0.0:
        return np.zeros((12, 12), dtype=float)
    K = np.zeros((12, 12), dtype=float)
    (u1, v1, w1, thx1, thy1, thz1, u2, v2, w2, thx2, thy2, thz2) = range(12)
    L1 = L
    L2 = L * L
    G4 = np.array([[36.0, 3.0 * L1, -36.0, 3.0 * L1], [3.0 * L1, 4.0 * L2, -3.0 * L1, -1.0 * L2], [-36.0, -3.0 * L1, 36.0, -3.0 * L1], [3.0 * L1, -1.0 * L2, -3.0 * L1, 4.0 * L2]], dtype=float)
    p_fac = Fx2 / (30.0 * L1)
    My_avg = 0.5 * (My1 + My2)
    Mz_avg = 0.5 * (Mz1 + Mz2)
    my_fac = My_avg / (30.0 * L2)
    mz_fac = Mz_avg / (30.0 * L2)
    idx_w = [w1, thy1, w2, thy2]
    K[np.ix_(idx_w, idx_w)] += (p_fac + my_fac) * G4
    idx_v = [v1, thz1, v2, thz2]
    K[np.ix_(idx_v, idx_v)] += (p_fac + mz_fac) * G4
    Li = 1.0 / L1
    T = Mx2
    B_base = np.array([[6.0 * Li, 1.0, -6.0 * Li, 1.0], [-6.0 * Li, -1.0, 6.0 * Li, -1.0]], dtype=float)
    t_fac = T / 10.0
    rows_thx = [thx1, thx2]
    cols_v = idx_v
    Bv = t_fac * B_base
    K[np.ix_(rows_thx, cols_v)] += Bv
    K[np.ix_(cols_v, rows_thx)] += Bv.T
    cols_w = idx_w
    Bw = t_fac * B_base
    K[np.ix_(rows_thx, cols_w)] += Bw
    K[np.ix_(cols_w, rows_thx)] += Bw.T
    K = 0.5 * (K + K.T)
    return K