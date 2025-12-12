def MSA_3D_assemble_global_geometric_stiffness_CC1_H4_T3(node_coords: np.ndarray, elements: Sequence[dict], u_global: np.ndarray) -> np.ndarray:
    n_nodes = node_coords.shape[0]
    K_global = np.zeros((6 * n_nodes, 6 * n_nodes))
    for elem in elements:
        i = elem['node_i']
        j = elem['node_j']
        (xi, yi, zi) = node_coords[i]
        (xj, yj, zj) = node_coords[j]
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        if L < 1e-12:
            raise ValueError('Zero-length element detected')
        (dx, dy, dz) = (xj - xi, yj - yi, zj - zi)
        ex = dx / L
        ey = dy / L
        ez = dz / L
        local_z = elem.get('local_z')
        if local_z is None:
            if abs(ez) > 0.99:
                local_z = np.array([0.0, 1.0, 0.0])
            else:
                local_z = np.array([0.0, 0.0, 1.0])
        else:
            local_z = np.array(local_z, dtype=float)
            norm_z = np.linalg.norm(local_z)
            if norm_z < 1e-12:
                raise ValueError('local_z must be a unit vector')
            local_z /= norm_z
        cross = np.cross([ex, ey, ez], local_z)
        if np.linalg.norm(cross) < 1e-12:
            raise ValueError('local_z must not be parallel to beam axis')
        local_y = np.cross(cross, [ex, ey, ez])
        local_y /= np.linalg.norm(local_y)
        local_x = np.array([ex, ey, ez])
        T_local = np.zeros((3, 3))
        T_local[0, :] = local_x
        T_local[1, :] = local_y
        T_local[2, :] = local_z
        Gamma = np.zeros((12, 12))
        Gamma[:3, :3] = T_local
        Gamma[3:6, 3:6] = T_local
        Gamma[6:9, 6:9] = T_local
        Gamma[9:12, 9:12] = T_local
        u_elem = np.zeros(12)
        u_elem[0:3] = u_global[6 * i:6 * i + 3]
        u_elem[3:6] = u_global[6 * i + 3:6 * i + 6]
        u_elem[6:9] = u_global[6 * j:6 * j + 3]
        u_elem[9:12] = u_global[6 * j + 3:6 * j + 6]
        E = elem['E']
        nu = elem['nu']
        A = elem['A']
        I_y = elem['I_y']
        I_z = elem['I_z']
        J = elem['J']
        EA = E * A
        EI_y = E * I_y
        EI_z = E * I_z
        GJ = E / (2 * (1 + nu)) * J
        Fx_i = EA / L * (u_elem[6] - u_elem[0])
        Fx_j = -Fx_i
        Mx_i = GJ / L * (u_elem[9] - u_elem[3])
        Mx_j = -Mx_i
        My_i = 2 * EI_z / L * (2 * u_elem[4] + u_elem[10] - 3 * (u_elem[7] - u_elem[1]) / L)
        My_j = 2 * EI_z / L * (u_elem[4] + 2 * u_elem[10] - 3 * (u_elem[7] - u_elem[1]) / L)
        Mz_i = 2 * EI_y / L * (2 * u_elem[5] + u_elem[11] - 3 * (u_elem[8] - u_elem[2]) / L)
        Mz_j = 2 * EI_y / L * (u_elem[5] + 2 * u_elem[11] - 3 * (u_elem[8] - u_elem[2]) / L)
        k_g_local = np.zeros((12, 12))
        k_g_local[0, 6] = -Fx_i / L
        k_g_local[6, 0] = -Fx_i / L
        k_g_local[0, 0] = Fx_i / L
        k_g_local[6, 6] = Fx_i / L
        k_g_local[1, 9] = Mx_i / L
        k_g_local[9, 1] = -Mx_i / L
        k_g_local[2, 9] = -Mx_i / L
        k_g_local[9, 2] = Mx_i / L
        k_g_local[1, 3] = -Mx_i / L
        k_g_local[3, 1] = Mx_i / L
        k_g_local[2, 3] = Mx_i / L
        k_g_local[3, 2] = -Mx_i / L
        k_g_local[7, 9] = Mx_j / L
        k_g_local[9, 7] = -Mx_j / L
        k_g_local[8, 9] = -Mx_j / L
        k_g_local[9, 8] = Mx_j / L
        k_g_local[7, 3] = -Mx_j / L
        k_g_local[3, 7] = Mx_j / L
        k_g_local[8, 3] = Mx_j / L
        k_g_local[3, 8] = -Mx_j / L
        k_g_local[2, 10] = 6 * My_i / L ** 2
        k_g_local[10, 2] = 6 * My_i / L ** 2
        k_g_local[2, 4] = -6 * My_i / L ** 2
        k_g_local[4, 2] = -6 * My_i / L ** 2
        k_g_local[8, 10] = -6 * My_j / L ** 2
        k_g_local[10, 8] = -6 * My_j / L ** 2
        k_g_local[8, 4] = 6 * My_j / L ** 2
        k_g_local[4, 8] = 6 * My_j / L ** 2
        k_g_local[1, 11] = 6 * Mz_i / L ** 2
        k_g_local[11, 1] = 6 * Mz_i / L ** 2
        k_g_local[1, 5] = -6 * Mz_i / L ** 2
        k_g_local[5, 1] = -6 * Mz_i / L ** 2
        k_g_local[7, 11] = -6 * Mz_j / L ** 2
        k_g_local[11, 7] = -6 * Mz_j / L ** 2
        k_g_local[7, 5] = 6 * Mz_j / L ** 2
        k_g_local[5, 7] = 6 * Mz_j / L ** 2
        k_g_local[4, 10] = 4 * My_i / L
        k_g_local[10, 4] = 4 * My_i / L
        k_g_local[4, 4] = -4 * My_i / L
        k_g_local[10, 10] = -4 * My_i / L
        k_g_local[4, 11] = 2 * My_i / L
        k_g_local[11, 4] = 2 * My_i / L
        k_g_local[10, 11] = 2 * My_i / L
        k_g_local[11, 10] = 2 * My_i / L
        k_g_local[5, 11] = 4 * Mz_i / L
        k_g_local[11, 5] = 4 * Mz_i / L
        k_g_local[5, 5] = -4 * Mz_i / L
        k_g_local[11, 11] = -4 * Mz_i / L
        k_g_local[5, 10] = 2 * Mz_i / L
        k_g_local[10, 5] = 2 * Mz_i / L
        k_g_local[11, 10] = 2 * Mz_i / L
        k_g_local[10, 11] = 2 * Mz_i / L
        k_g_local[4, 11] = 2 * My_i / L
        k_g_local[11, 4] = 2 * My_i / L
        k_g_local[5, 10] = 2 * Mz_i / L
        k_g_local[10, 5] = 2 * Mz_i / L
        k_g_local[10, 4] = 2 * My_j / L
        k_g_local[4, 10] = 2 * My_j / L
        k_g_local[11, 5] = 2 * Mz_j / L
        k_g_local[5, 11] = 2 * Mz_j / L
        K_elem_global = Gamma.T @ k_g_local @ Gamma
        dofs = [6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5, 6 * j, 6 * j + 1, 6 * j + 2, 6 * j + 3, 6 * j + 4, 6 * j + 5]
        for (a, dof_a) in enumerate(dofs):
            for (b, dof_b) in enumerate(dofs):
                K_global[dof_a, dof_b] += K_elem_global[a, b]
    return K_global