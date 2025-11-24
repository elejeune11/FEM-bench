def local_geometric_stiffness_matrix_3D_beam(L: float, A: float, I_rho: float, Fx2: float, Mx2: float, My1: float, Mz1: float, My2: float, Mz2: float) -> np.ndarray:
    K_g = np.zeros((12, 12))
    K_g[1, 1] = K_g[7, 7] = 6 * Fx2 / (5 * L)
    K_g[1, 7] = K_g[7, 1] = -6 * Fx2 / (5 * L)
    K_g[2, 2] = K_g[8, 8] = 6 * Fx2 / (5 * L)
    K_g[2, 8] = K_g[8, 2] = -6 * Fx2 / (5 * L)
    K_g[1, 5] = K_g[5, 1] = K_g[1, 11] = K_g[11, 1] = -Fx2 / 10
    K_g[2, 4] = K_g[4, 2] = K_g[2, 10] = K_g[10, 2] = Fx2 / 10
    K_g[4, 8] = K_g[8, 4] = K_g[10, 8] = K_g[8, 10] = -Fx2 / 10
    K_g[5, 7] = K_g[7, 5] = K_g[11, 7] = K_g[7, 11] = -Fx2 / 10
    K_g[4, 5] = K_g[5, 4] = K_g[4, 11] = K_g[11, 4] = K_g[5, 10] = K_g[10, 5] = K_g[10, 11] = K_g[11, 10] = Fx2 * L / 30
    K_g[3, 3] = K_g[9, 9] = Mx2 / (2 * A * L)
    K_g[3, 9] = K_g[9, 3] = -Mx2 / (2 * A * L)
    K_g[1, 2] = K_g[2, 1] = (My1 + My2) / (2 * L)
    K_g[1, 8] = K_g[8, 1] = (My1 - My2) / (2 * L)
    K_g[2, 7] = K_g[7, 2] = (My2 - My1) / (2 * L)
    K_g[7, 8] = K_g[8, 7] = -(My1 + My2) / (2 * L)
    K_g[4, 5] = K_g[5, 4] = K_g[4, 11] = K_g[11, 4] = K_g[5, 10] = K_g[10, 5] = K_g[10, 11] = K_g[11, 10] = (Mz1 + Mz2) / (2 * L)
    return K_g