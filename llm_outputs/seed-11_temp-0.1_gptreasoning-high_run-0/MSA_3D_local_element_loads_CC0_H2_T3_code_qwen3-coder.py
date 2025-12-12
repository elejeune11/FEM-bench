def MSA_3D_local_element_loads_CC0_H2_T3(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global):
    E = ele_info['E']
    nu = ele_info['nu']
    A = ele_info['A']
    Iy = ele_info['I_y']
    Iz = ele_info['I_z']
    J = ele_info['J']
    local_z_input = ele_info.get('local_z', None)
    dx = xj - xi
    dy = yj - yi
    dz = zj - zi
    L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    if L == 0:
        raise ValueError('Beam length is zero.')
    x_axis = np.array([dx, dy, dz]) / L
    if local_z_input is not None:
        local_z = np.array(local_z_input)
        if np.allclose(np.cross(x_axis, local_z), 0):
            raise ValueError('local_z is parallel to the beam axis.')
    else:
        global_z = np.array([0, 0, 1])
        if np.allclose(np.cross(x_axis, global_z), 0):
            local_z = np.array([0, 1, 0])
        else:
            local_z = global_z
    local_z = local_z / np.linalg.norm(local_z)
    y_axis = np.cross(local_z, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    local_z = np.cross(x_axis, y_axis)
    T_local = np.array([x_axis, y_axis, local_z])
    T = np.zeros((12, 12))
    for i in range(4):
        T[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)] = T_local
    u_dofs_local = T @ u_dofs_global
    k_local = np.zeros((12, 12))
    k_local[0, 0] = k_local[6, 6] = E * A / L
    k_local[0, 6] = k_local[6, 0] = -E * A / L
    k_local[3, 3] = k_local[9, 9] = G = E / (2 * (1 + nu))
    k_local[3, 9] = k_local[9, 3] = -G * J / L
    k_local[1, 1] = k_local[7, 7] = 12 * E * Iz / L ** 3
    k_local[1, 5] = k_local[5, 1] = 6 * E * Iz / L ** 2
    k_local[1, 7] = k_local[7, 1] = -12 * E * Iz / L ** 3
    k_local[1, 11] = k_local[11, 1] = 6 * E * Iz / L ** 2
    k_local[5, 5] = 4 * E * Iz / L
    k_local[5, 7] = k_local[7, 5] = -6 * E * Iz / L ** 2
    k_local[5, 11] = k_local[11, 5] = 2 * E * Iz / L
    k_local[7, 11] = k_local[11, 7] = -6 * E * Iz / L ** 2
    k_local[11, 11] = 4 * E * Iz / L
    k_local[2, 2] = k_local[8, 8] = 12 * E * Iy / L ** 3
    k_local[2, 4] = k_local[4, 2] = -6 * E * Iy / L ** 2
    k_local[2, 8] = k_local[8, 2] = -12 * E * Iy / L ** 3
    k_local[2, 10] = k_local[10, 2] = -6 * E * Iy / L ** 2
    k_local[4, 4] = 4 * E * Iy / L
    k_local[4, 8] = k_local[8, 4] = 6 * E * Iy / L ** 2
    k_local[4, 10] = k_local[10, 4] = 2 * E * Iy / L
    k_local[8, 10] = k_local[10, 8] = 6 * E * Iy / L ** 2
    k_local[10, 10] = 4 * E * Iy / L
    load_dofs_local = k_local @ u_dofs_local
    return load_dofs_local