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
    import numpy as np
    K = np.zeros((12, 12), dtype=float)
    if L != 0.0:
        k_axial = Fx2 / L
    else:
        k_axial = 0.0
    K[np.ix_([0, 6], [0, 6])] += k_axial * np.array([[1.0, -1.0], [-1.0, 1.0]])
    if L != 0.0:
        P = Fx2
        factor = P / (30.0 * L)
        base4 = factor * np.array([[36.0, 3.0 * L, -36.0, 3.0 * L], [3.0 * L, 4.0 * L * L, -3.0 * L, -1.0 * L * L], [-36.0, -3.0 * L, 36.0, -3.0 * L], [3.0 * L, -1.0 * L * L, -3.0 * L, 4.0 * L * L]])
    else:
        base4 = np.zeros((4, 4), dtype=float)
    idx_v_plane = [1, 5, 7, 11]
    idx_w_plane = [2, 4, 8, 10]
    K[np.ix_(idx_v_plane, idx_v_plane)] += base4
    K[np.ix_(idx_w_plane, idx_w_plane)] += base4
    if L != 0.0:
        Mz_factor = 1.0 / (2.0 * L)
        My_factor = 1.0 / (2.0 * L)
        Mv = np.array([[0.0, Mz1, 0.0, Mz2], [Mz1, 0.0, Mz2, 0.0], [0.0, Mz2, 0.0, Mz1], [Mz2, 0.0, Mz1, 0.0]]) * Mz_factor
        K[np.ix_(idx_v_plane, idx_v_plane)] += Mv
        Mw = np.array([[0.0, My1, 0.0, My2], [My1, 0.0, My2, 0.0], [0.0, My2, 0.0, My1], [My2, 0.0, My1, 0.0]]) * My_factor
        K[np.ix_(idx_w_plane, idx_w_plane)] += Mw
    idx_tx = [3, 9]
    if L != 0.0:
        K_torsion = Mx2 / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    else:
        K_torsion = np.zeros((2, 2), dtype=float)
    K[np.ix_(idx_tx, idx_tx)] += K_torsion
    if L != 0.0:
        coupling_factor = Mx2 / (2.0 * L)
        idx_ty = [4, 10]
        idx_tz = [5, 11]
        base_coup = coupling_factor * np.array([[1.0, -1.0], [-1.0, 1.0]])
        K[np.ix_(idx_tx, idx_ty)] += base_coup
        K[np.ix_(idx_ty, idx_tx)] += base_coup.T
        K[np.ix_(idx_tx, idx_tz)] += base_coup
        K[np.ix_(idx_tz, idx_tx)] += base_coup.T
    K = 0.5 * (K + K.T)
    return K