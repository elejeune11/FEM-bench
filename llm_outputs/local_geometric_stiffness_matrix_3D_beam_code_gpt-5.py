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
    if L <= 0.0:
        raise ValueError('Beam length L must be positive.')
    K = np.zeros((12, 12), dtype=float)
    P = float(Fx2)
    L2 = L * L
    M4 = np.array([[36.0, 3.0 * L, -36.0, 3.0 * L], [3.0 * L, 4.0 * L2, -3.0 * L, -1.0 * L2], [-36.0, -3.0 * L, 36.0, -3.0 * L], [3.0 * L, -1.0 * L2, -3.0 * L, 4.0 * L2]], dtype=float)
    coef_P = P / (30.0 * L)
    Kg_plane = coef_P * M4
    idx_v = [1, 5, 7, 11]
    idx_w = [2, 4, 8, 10]

    def add_block(Kmat: np.ndarray, idx: list, block: np.ndarray):
        for a in range(4):
            ia = idx[a]
            for b in range(4):
                ib = idx[b]
                Kmat[ia, ib] += block[a, b]
    add_block(K, idx_v, Kg_plane)
    add_block(K, idx_w, Kg_plane)
    My_avg = 0.5 * (float(My1) + float(My2))
    Mz_avg = 0.5 * (float(Mz1) + float(Mz2))
    if L2 > 0.0:
        coef_My = My_avg / (30.0 * L2)
        coef_Mz = Mz_avg / (30.0 * L2)
        Kg_My = coef_My * M4
        Kg_Mz = coef_Mz * M4
        add_block(K, idx_w, Kg_My)
        add_block(K, idx_v, Kg_Mz)
    T4 = np.array([[0.0, 1.0, 0.0, -1.0], [1.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [-1.0, 0.0, 1.0, 0.0]], dtype=float)
    c_t = float(Mx2) / 20.0
    idx_t_y = [3, 4, 9, 10]
    idx_t_z = [3, 5, 9, 11]
    add_block(K, idx_t_y, c_t * T4)
    add_block(K, idx_t_z, c_t * T4)
    K = 0.5 * (K + K.T)
    return K