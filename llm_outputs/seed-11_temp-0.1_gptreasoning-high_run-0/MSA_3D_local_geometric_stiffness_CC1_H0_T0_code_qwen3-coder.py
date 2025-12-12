import numpy as np
def MSA_3D_local_geometric_stiffness_CC1_H0_T0(L: float, A: float, I_rho: float, Fx2: float, Mx2: float, My1: float, Mz1: float, My2: float, Mz2: float) -> np.ndarray:
    K_g = np.zeros((12, 12))
    Fx = Fx2
    Mt = Mx2
    M1y = My1
    M1z = Mz1
    M2y = My2
    M2z = Mz2
    L2 = L * L
    L3 = L * L2
    if Fx != 0:
        K_g[1, 1] = K_g[7, 7] = Fx / L
        K_g[1, 7] = K_g[7, 1] = -Fx / L
        K_g[2, 2] = K_g[6, 6] = Fx / L
        K_g[2, 6] = K_g[6, 2] = -Fx / L
        K_g[1, 5] = K_g[5, 1] = -Fx / 2
        K_g[2, 4] = K_g[4, 2] = Fx / 2
        K_g[7, 11] = K_g[11, 7] = -Fx / 2
        K_g[6, 10] = K_g[10, 6] = Fx / 2
        K_g[4, 6] = K_g[6, 4] = -Fx / 2
        K_g[5, 7] = K_g[7, 5] = Fx / 2
        K_g[4, 10] = K_g[10, 4] = Fx * L / 6
        K_g[5, 11] = K_g[11, 5] = -Fx * L / 6
        K_g[4, 4] = K_g[10, 10] = Fx * L / 3
        K_g[5, 5] = K_g[11, 11] = Fx * L / 3
        K_g[4, 10] = K_g[10, 4] = Fx * L / 6
        K_g[5, 11] = K_g[11, 5] = -Fx * L / 6
    if Mt != 0:
        K_g[3, 3] = K_g[9, 9] = Mt / L
        K_g[3, 9] = K_g[9, 3] = -Mt / L
    if M1y != 0:
        K_g[2, 5] = K_g[5, 2] = -M1y / 2
        K_g[8, 5] = K_g[5, 8] = M1y / 2
        K_g[2, 11] = K_g[11, 2] = -M1y / 6
        K_g[8, 11] = K_g[11, 8] = M1y / 6
    if M2y != 0:
        K_g[2, 5] = K_g[5, 2] += -M2y / 2
        K_g[8, 5] = K_g[5, 8] += M2y / 2
        K_g[2, 11] = K_g[11, 2] += -M2y / 6
        K_g[8, 11] = K_g[11, 8] += M2y / 6
    if M1z != 0:
        K_g[1, 4] = K_g[4, 1] = M1z / 2
        K_g[7, 4] = K_g[4, 7] = -M1z / 2
        K_g[1, 10] = K_g[10, 1] = M1z / 6
        K_g[7, 10] = K_g[10, 7] = -M1z / 6
    if M2z != 0:
        K_g[1, 4] = K_g[4, 1] += M2z / 2
        K_g[7, 4] = K_g[4, 7] += -M2z / 2
        K_g[1, 10] = K_g[10, 1] += M2z / 6
        K_g[7, 10] = K_g[10, 7] += -M2z / 6
    return K_g + K_g.T - np.diag(np.diag(K_g))