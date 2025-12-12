def MSA_3D_local_element_loads_CC0_H2_T3(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global):
    E = ele_info['E']
    nu = ele_info['nu']
    A = ele_info['A']
    I_y = ele_info['I_y']
    I_z = ele_info['I_z']
    J = ele_info['J']
    local_z_global = ele_info.get('local_z', None)
    Lx = xj - xi
    Ly = yj - yi
    Lz = zj - zi
    L = np.sqrt(Lx ** 2 + Ly ** 2 + Lz ** 2)
    if L == 0:
        raise ValueError('Beam length is zero.')
    ex = np.array([Lx, Ly, Lz]) / L
    if local_z_global is None:
        if abs(Lz) > 0.999 * L:
            ez_global = np.array([0, 1, 0])
        else:
            ez_global = np.array([0, 0, 1])
    else:
        ez_global = np.array(local_z_global, dtype=float)
        ez_global = ez_global / np.linalg.norm(ez_global)
    if abs(np.dot(ex, ez_global)) > 0.999:
        raise ValueError('local_z vector is parallel to the beam axis.')
    ey_global = np.cross(ez_global, ex)
    ey_global = ey_global / np.linalg.norm(ey_global)
    ez_global = np.cross(ex, ey_global)
    ez_global = ez_global / np.linalg.norm(ez_global)
    T = np.array([ex, ey_global, ez_global])
    T12 = np.zeros((12, 12))
    for i in range(3):
        T12[i, i] = T[0, i]
        T12[i, 3 + i] = T[1, i]
        T12[i, 6 + i] = T[2, i]
        T12[3 + i, i] = T[0, i]
        T12[3 + i, 3 + i] = T[1, i]
        T12[3 + i, 6 + i] = T[2, i]
    T12 = np.zeros((12, 12))
    for i in range(3):
        for j in range(3):
            T12[i, j] = T[i, j]
            T12[i + 3, j + 3] = T[i, j]
            T12[i + 6, j + 6] = T[i, j]
            T12[i + 9, j + 9] = T[i, j]
    u_dofs_local = T12 @ u_dofs_global
    (u1, v1, w1, theta_x1, theta_y1, theta_z1) = u_dofs_local[:6]
    (u2, v2, w2, theta_x2, theta_y2, theta_z2) = u_dofs_local[6:]
    k_axial = E * A / L
    k_bend_y = E * I_y / L ** 3
    k_bend_y_11 = 12 * k_bend_y
    k_bend_y_12 = 6 * k_bend_y * L
    k_bend_y_22 = 4 * k_bend_y * L ** 2
    k_bend_y_23 = 2 * k_bend_y * L ** 2
    k_bend_z = E * I_z / L ** 3
    k_bend_z_11 = 12 * k_bend_z
    k_bend_z_12 = 6 * k_bend_z * L
    k_bend_z_22 = 4 * k_bend_z * L ** 2
    k_bend_z_23 = 2 * k_bend_z * L ** 2
    k_torsion = E * J / (2 * (1 + nu)) / L
    k_torsion = E * J / L
    G = E / (2 * (1 + nu))
    k_torsion = G * J / L
    K_local = np.zeros((12, 12))
    K_local[0, 0] = k_axial
    K_local[0, 6] = -k_axial
    K_local[6, 0] = -k_axial
    K_local[6, 6] = k_axial
    K_local[1, 1] = k_bend_y_11
    K_local[1, 5] = k_bend_y_12
    K_local[1, 7] = -k_bend_y_11
    K_local[1, 11] = k_bend_y_12
    K_local[5, 1] = k_bend_y_12
    K_local[5, 5] = k_bend_y_22
    K_local[5, 7] = -k_bend_y_12
    K_local[5, 11] = k_bend_y_23
    K_local[7, 1] = -k_bend_y_11
    K_local[7, 5] = -k_bend_y_12
    K_local[7, 7] = k_bend_y_11
    K_local[7, 11] = -k_bend_y_12
    K_local[11, 1] = k_bend_y_12
    K_local[11, 5] = k_bend_y_23
    K_local[11, 7] = -k_bend_y_12
    K_local[11, 11] = k_bend_y_22
    K_local[2, 2] = k_bend_z_11
    K_local[2, 4] = -k_bend_z_12
    K_local[2, 8] = -k_bend_z_11
    K_local[2, 10] = -k_bend_z_12
    K_local[4, 2] = -k_bend_z_12
    K_local[4, 4] = k_bend_z_22
    K_local[4, 8] = k_bend_z_12
    K_local[4, 10] = k_bend_z_23
    K_local[8, 2] = -k_bend_z_11
    K_local[8, 4] = k_bend_z_12
    K_local[8, 8] = k_bend_z_11
    K_local[8, 10] = -k_bend_z_12
    K_local[10, 2] = -k_bend_z_12
    K_local[10, 4] = k_bend_z_23
    K_local[10, 8] = -k_bend_z_12
    K_local[10, 10] = k_bend_z_22
    K_local[3, 3] = k_torsion
    K_local[3, 9] = -k_torsion
    K_local[9, 3] = -k_torsion
    K_local[9, 9] = k_torsion
    load_dofs_local = K_local @ u_dofs_local
    return load_dofs_local