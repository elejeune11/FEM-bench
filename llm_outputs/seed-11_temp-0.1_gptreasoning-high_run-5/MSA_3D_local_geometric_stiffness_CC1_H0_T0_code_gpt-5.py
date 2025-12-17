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
    K = np.zeros((12, 12), dtype=float)
    if not np.isfinite(L) or L <= 0.0:
        return K
    P = Fx2
    fac = P / (30.0 * L)
    L2 = L * L
    sub = fac * np.array([[36.0, 3.0 * L, -36.0, 3.0 * L], [3.0 * L, 4.0 * L2, -3.0 * L, -1.0 * L2], [-36.0, -3.0 * L, 36.0, -3.0 * L], [3.0 * L, -1.0 * L2, -3.0 * L, 4.0 * L2]], dtype=float)
    idx_y = [1, 5, 7, 11]
    for a in range(4):
        for b in range(4):
            K[idx_y[a], idx_y[b]] += sub[a, b]
    idx_z = [2, 4, 8, 10]
    for a in range(4):
        for b in range(4):
            K[idx_z[a], idx_z[b]] += sub[a, b]
    K = 0.5 * (K + K.T)
    return K