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
    K = np.zeros((12, 12), dtype=float)
    if not np.isfinite(L) or L <= 0.0:
        return K

    def add_sub(KM, idx, S):
        for ii in range(4):
            i = idx[ii]
            for jj in range(4):
                j = idx[jj]
                KM[i, j] += S[ii, jj]
    P = Fx2
    factor_axial = P / (30.0 * L)
    L2 = L * L
    Kg_2D = np.array([[36.0, 3.0 * L, -36.0, 3.0 * L], [3.0 * L, 4.0 * L2, -3.0 * L, -1.0 * L2], [-36.0, -3.0 * L, 36.0, -3.0 * L], [3.0 * L, -1.0 * L2, -3.0 * L, 4.0 * L2]], dtype=float)
    add_sub(K, [1, 5, 7, 11], factor_axial * Kg_2D)
    add_sub(K, [2, 4, 8, 10], factor_axial * Kg_2D)
    if np.isfinite(Mx2) and np.isfinite(I_rho) and (I_rho > 0.0) and (Mx2 != 0.0):
        kT = Mx2 / I_rho * (L2 / 30.0)
        T6 = np.array([[0.0, 1.0, 1.0, 0.0, -1.0, -1.0], [1.0, 0.0, 0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, -1.0, 0.0, 0.0], [0.0, -1.0, -1.0, 0.0, 1.0, 1.0], [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=float) * kT
        rot_idx = [3, 4, 5, 9, 10, 11]
        for a in range(6):
            ia = rot_idx[a]
            for b in range(6):
                ib = rot_idx[b]
                K[ia, ib] += T6[a, b]
    gamma_m = 1.0 / 15.0
    my_avg = 0.5 * (My1 + My2) if np.isfinite(My1) and np.isfinite(My2) else 0.0
    mz_avg = 0.5 * (Mz1 + Mz2) if np.isfinite(Mz1) and np.isfinite(Mz2) else 0.0
    if L > 0.0:
        kMy = gamma_m * my_avg / (L2 if L2 != 0.0 else 1.0)
        kMz = gamma_m * mz_avg / (L2 if L2 != 0.0 else 1.0)
        B4 = np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]], dtype=float)
        if kMy != 0.0:
            add_sub(K, [2, 4, 8, 10], kMy * B4)
        if kMz != 0.0:
            add_sub(K, [1, 5, 7, 11], kMz * B4)
    K = 0.5 * (K + K.T)
    return K