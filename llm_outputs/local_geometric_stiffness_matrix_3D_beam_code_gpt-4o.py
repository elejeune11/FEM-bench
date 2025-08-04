def local_geometric_stiffness_matrix_3D_beam(L: float, A: float, I_rho: float, Fx2: float, Mx2: float, My1: float, Mz1: float, My2: float, Mz2: float) -> np.ndarray:
    K_g = np.zeros((12, 12))
    P = Fx2
    K_g[1, 1] = K_g[7, 7] = 6 * P / (5 * L)
    K_g[1, 7] = K_g[7, 1] = -6 * P / (5 * L)
    K_g[2, 2] = K_g[8, 8] = 6 * P / (5 * L)
    K_g[2, 8] = K_g[8, 2] = -6 * P / (5 * L)
    K_g[4, 4] = K_g[10, 10] = 2 * P / 15
    K_g[5, 5] = K_g[11, 11] = 2 * P / 15
    K_g[4, 10] = K_g[10, 4] = P / 30
    K_g[5, 11] = K_g[11, 5] = P / 30
    K_g[1, 5] = K_g[5, 1] = K_g[7, 11] = K_g[11, 7] = 3 * P / (10 * L)
    K_g[2, 4] = K_g[4, 2] = K_g[8, 10] = K_g[10, 8] = -3 * P / (10 * L)
    K_g[1, 11] = K_g[11, 1] = K_g[7, 5] = K_g[5, 7] = -3 * P / (10 * L)
    K_g[2, 10] = K_g[10, 2] = K_g[8, 4] = K_g[4, 8] = 3 * P / (10 * L)
    T = Mx2
    K_g[3, 3] = K_g[9, 9] = T / (2 * I_rho)
    K_g[3, 9] = K_g[9, 3] = -T / (2 * I_rho)
    My = (My1 + My2) / 2
    Mz = (Mz1 + Mz2) / 2
    K_g[1, 1] += 12 * My / L ** 2
    K_g[2, 2] += 12 * Mz / L ** 2
    K_g[1, 7] -= 12 * My / L ** 2
    K_g[2, 8] -= 12 * Mz / L ** 2
    K_g[4, 4] += 4 * My / L
    K_g[5, 5] += 4 * Mz / L
    K_g[4, 10] -= 2 * My / L
    K_g[5, 11] -= 2 * Mz / L
    K_g[7, 7] += 12 * My / L ** 2
    K_g[8, 8] += 12 * Mz / L ** 2
    K_g[10, 10] += 4 * My / L
    K_g[11, 11] += 4 * Mz / L
    return K_g