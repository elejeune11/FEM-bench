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
    if not np.isfinite(L) or L <= 0.0:
        return K
    id_v_plane = [1, 5, 7, 11]
    id_w_plane = [2, 4, 8, 10]
    id_thx = [3, 9]
    id_thy = [4, 10]
    id_thz = [5, 11]
    L2 = L * L
    base = np.array([[36.0, 3.0 * L, -36.0, 3.0 * L], [3.0 * L, 4.0 * L2, -3.0 * L, -1.0 * L2], [-36.0, -3.0 * L, 36.0, -3.0 * L], [3.0 * L, -1.0 * L2, -3.0 * L, 4.0 * L2]], dtype=float)
    kg_flex = Fx2 / (30.0 * L) * base
    K[np.ix_(id_v_plane, id_v_plane)] += kg_flex
    K[np.ix_(id_w_plane, id_w_plane)] += kg_flex
    My_avg = 0.5 * (My1 + My2)
    Mz_avg = 0.5 * (Mz1 + Mz2)
    if My_avg != 0.0:
        km_y = My_avg / (30.0 * L2) * base
        K[np.ix_(id_w_plane, id_w_plane)] += km_y
    if Mz_avg != 0.0:
        km_z = Mz_avg / (30.0 * L2) * base
        K[np.ix_(id_v_plane, id_v_plane)] += km_z
    if Mx2 != 0.0:
        S2 = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=float)
        a = Mx2 / (15.0 * L)
        C_thx_thy = a * S2
        C_thx_thz = a * S2
        K[np.ix_(id_thx, id_thy)] += C_thx_thy
        K[np.ix_(id_thy, id_thx)] += C_thx_thy
        K[np.ix_(id_thx, id_thz)] += C_thx_thz
        K[np.ix_(id_thz, id_thx)] += C_thx_thz
    K = 0.5 * (K + K.T)
    return K